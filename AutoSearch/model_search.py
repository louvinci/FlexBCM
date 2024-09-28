import sys
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import init
from torch.autograd import Variable
from genotypes import PRIMITIVES
from operations import OPS
import numpy as np
from einops import rearrange
from einops.layers.torch import Rearrange
from analytical_model.search_hw import search_hw_lat, evaluate_latency

#from analytical_model.search_hw import search_hw_lat, evaluate_latency
def pair(t):
    return t if isinstance(t, tuple) else (t, t)

# https://github.com/YongfeiYan/Gumbel_Softmax_VAE/blob/master/gumbel_softmax_vae.py
def sample_gumbel(shape, eps=1e-20):
    U = torch.rand(shape)
    U = U.cuda()
    return -torch.log(-torch.log(U + eps) + eps)


def gumbel_softmax_sample(logits, temperature=1):
    y = logits + sample_gumbel(logits.size())
    return F.softmax(y / temperature, dim=-1)


def gumbel_softmax(logits, temperature=1):
    """
    ST-gumple-softmax
    input: [*, n_class]
    return: flatten --> [*, n_class] an one-hot vector
    """
    y = gumbel_softmax_sample(F.log_softmax(logits, dim=-1), temperature)

    return y
    
    # if not hard:
    #     return y
    # to one-hot code
    # shape = y.size()
    # _, ind = y.max(dim=-1)
    # y_hard = torch.zeros_like(y).view(-1, shape[-1])
    # y_hard.scatter_(1, ind.view(-1, 1), 1)
    # y_hard = y_hard.view(*shape)
    # # Set gradients w.r.t. y_hard gradients w.r.t. y
    # y_hard = (y_hard - y).detach() + y
    # return y_hard


class MixedOp(nn.Module):

    def __init__(self, C_in, C_out, kernel_size =3, stride=1, pad=1, mode='soft', act_num=1, flops_mode='sum'):
        super(MixedOp, self).__init__()
        self._ops = nn.ModuleList()
        self.mode = mode
        self.act_num = act_num
        self.flops_mode = flops_mode

        for primitive in PRIMITIVES:
            #C_in, C_out, kernel_size, stride, pad:
            op = OPS[primitive](C_in, C_out, kernel_size, stride, pad)
            self._ops.append(op)
        
        self.register_buffer('active_list', torch.tensor(list(range(len(self._ops)))))


    def forward(self, x, alpha, alpha_param=None):
        # int: force #channel; tensor: arch_ratio; float(<=1): force width
        result = 0
        #print('MXOP forward: ', alpha , alpha_param)
        if self.mode == 'soft':
            for i, (w, op) in enumerate(zip(alpha, self._ops)):
                #lwq here use the alpha getting the weighted sum
                result = result + op(x) * w 
                # print(type(op), result.shape)

            self.set_active_list(list(range(len(self._ops))))

        elif self.mode == 'hard':
            rank = alpha.argsort(descending=True)
            self.set_active_list(rank[:self.act_num])

            for i in range(self.act_num):
                result = result + self._ops[rank[i]](x) * ((1-alpha[rank[i]]).detach() + alpha[rank[i]])


        elif self.mode == 'random_hard':
            rank = list(alpha.argsort(descending=True).cpu())
            result = result + self._ops[rank[0]](x) * ((1-alpha[rank[0]]).detach() + alpha[rank[0]])
            
            forward_op_index = rank.pop(0)
            #print(rank)
            sampled_op = np.random.choice(rank, self.act_num-1)

            for i in range(len(sampled_op)):
                result = result + self._ops[sampled_op[i]](x) * ((0-alpha[sampled_op[i]]).detach() + alpha[sampled_op[i]])

            sampled_op = sampled_op.tolist()
            sampled_op.insert(0, forward_op_index)
            #* when update the archietct, the forward function will be used in the another data
            # #!log
            # if self._ops[0].block.stage == "update_arch":
            #     print("forward: ",sampled_op)
            
            self.set_active_list(sampled_op)

        elif self.mode == 'proxy_hard':
            assert alpha_param is not None
            # print('alpha_gumbel:', alpha)
            #* active_list record the first act_num OP index, initial length is the number of ops, then is the act_num
            rank = alpha.argsort(descending=True)
            self.set_active_list(rank[:self.act_num])

            alpha = F.softmax(alpha_param[rank[:self.act_num]], dim=-1)

            #TODO lwq the second act_num operation, ignoring the old grad
            result = result + self._ops[rank[0]](x) * ((1-alpha[0]).detach() + alpha[0])
            for i in range(1,self.act_num):
                result = result + self._ops[rank[i]](x) * ((0-alpha[i]).detach() + alpha[i])
        else:
            print('Wrong search mode:', self.mode)
            sys.exit()

        return result

    # set the active operator list for each block
    def set_active_list(self, active_list):
        if type(active_list) is not torch.Tensor:
            active_list = torch.tensor(active_list).cuda()

        self.active_list.data = active_list.data


    def forward_flops(self, size, alpha):
        # int: force #channel; tensor: arch_ratio; float(<=1): force width
        result = 0
        #print("MixOp: ",size)
        if self.flops_mode == 'sum':
            flops, size_o = self._ops[0].forward_flops(size)
            result = result + flops * alpha[0]
            for i, (w, op) in enumerate(zip(alpha[1:], self._ops[1:])):
                flops, _ = op.forward_flops(size)
                result = result + flops * w
            
        elif self.flops_mode == 'single_path':
            op_id = alpha.argsort(descending=True)[0]
            flops,  size_o = self._ops[op_id].forward_flops(size)
            #result  = flops #! the true flops
            result = alpha[op_id] * flops

        else:
            print('Wrong flops_mode.')
            sys.exit()

        return result,size_o


def conv1x1(in_planes, out_planes, bias=True,stride=1):
    """1x1 convolution"""
    #return Conv(in_planes, out_planes, kernel_size=1, block_size=block_size, stride=stride, padding=0, bias=bias)
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=bias)

class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1, mode='soft', act_num=1, flops_mode='sum'):
        super(BasicBlock, self).__init__()
        #C_in, C_out, layer_id, kernel_size =3, stride=1, pad=1, mode='soft', act_num=1, flops_mode='sum'
        self.conv1 =  MixedOp(in_planes, planes,  kernel_size =3, stride=stride, pad=1, mode=mode, act_num=act_num, flops_mode= flops_mode)
        self.bn1   =  nn.BatchNorm2d(planes)
        self.conv2 =  MixedOp(planes,    planes,  kernel_size =3, stride=1,      pad=1, mode=mode, act_num=act_num, flops_mode= flops_mode)
        self.bn2   =  nn.BatchNorm2d(planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion*planes:
            self.shortcut = nn.Sequential(
                conv1x1(in_planes, self.expansion*planes, stride=stride),
                nn.BatchNorm2d(self.expansion*planes)
            )

    def forward(self, x, alpha_lst, alpha_param):
        out = F.relu(self.bn1(self.conv1(x,alpha_lst[0],alpha_param[0])))
        out = self.bn2(self.conv2(out,alpha_lst[1],alpha_param[1]))
        out += self.shortcut(x)
        out = F.relu(out)
        return out

class Bottleneck(nn.Module):
    expansion = 4
    #
    def __init__(self, in_planes, planes, stride=1, mode='soft', act_num=1, flops_mode='sum'):
        super(Bottleneck, self).__init__()
        self.conv1 = MixedOp(in_planes, planes, kernel_size = 1, stride=1, pad=0,      mode=mode, act_num=act_num, flops_mode= flops_mode)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = MixedOp(planes, planes,    kernel_size = 3, stride=stride, pad=1, mode=mode, act_num=act_num, flops_mode= flops_mode)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = MixedOp(planes, self.expansion*planes, kernel_size = 1, stride=1,pad=0, mode=mode, act_num=act_num, flops_mode= flops_mode)
        self.bn3 = nn.BatchNorm2d(self.expansion*planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion*planes:
            self.shortcut = nn.Sequential(
                conv1x1(in_planes, self.expansion*planes,stride=stride),
                nn.BatchNorm2d(self.expansion*planes)
            )

    def forward(self, x, alpha_lst,alpha_param):
        out = F.relu(self.bn1(self.conv1(x , alpha_lst[0],alpha_param[0])))
        out = F.relu(self.bn2(self.conv2(out, alpha_lst[1],alpha_param[1])))
        out = self.bn3(self.conv3(out, alpha_lst[2],alpha_param[2]))
        out += self.shortcut(x)
        out = F.relu(out)
        return out

class SuperNet(nn.Module):
    def __init__(self, config):
        super(SuperNet, self).__init__()
        self.in_planes = 64
        #config.flops_mode # control the forward flops mode{'sum','single path','multi path'}
        self.mode        =  config.mode
        self.flops_mode  =  config.flops_mode
        self.act_num     =  config.act_num
        self.num_classes =  config.num_classes
        self._criterion = nn.CrossEntropyLoss()
        self.backbone = config.backbone

        self.sample_func = config.sample_func
        self.searched_hw = None
        self.dataset     = config.dataset
        

        if self.backbone == 'RN18':
            block       = BasicBlock
            self.num_blocks  = [2,2,2,2]
            self.blen = 2
        elif self.backbone == 'RN34':
            block       = BasicBlock
            self.num_blocks  = [3,4,6,3]
            self.blen = 2
        elif self.backbone == 'RN50':
            block       = Bottleneck
            self.num_blocks  = [3,4,6,3]
            self.blen = 3
        else:
            raise Exception('Wrong backbone')


        self.b1 = self.num_blocks[0]*self.blen
        self.b2 = sum(self.num_blocks[0:2]) *self.blen
        self.b3 = sum(self.num_blocks[0:3]) *self.blen 
        self.b4 = sum(self.num_blocks[0:4]) *self.blen
        if config.dataset == 'imagenet':
            self.stem = nn.Sequential(
                nn.Conv2d(3, self.in_planes, kernel_size=7, stride=2, padding=3, bias=False),
                nn.BatchNorm2d(self.in_planes),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
            )
        else:
            self.stem = nn.Sequential(
                nn.Conv2d(3, self.in_planes, kernel_size=3, stride=1, padding=1, bias=False),
                nn.BatchNorm2d(self.in_planes),
                nn.ReLU(inplace=True)
            ) 
        
        self.layer1 = self._make_layer(block, 64,  self.num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 128, self.num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 256, self.num_blocks[2], stride=2)
        self.layer4 = self._make_layer(block, 512, self.num_blocks[3], stride=2)
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.linear = nn.Linear(512 * block.expansion, self.num_classes)
        self._arch_params = self._build_arch_parameters()#*lwq init the arichitecture parameters
        self._reset_arch_parameters()
        self._criterion = nn.CrossEntropyLoss()
        self.init_params()

    def init_params(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                init.kaiming_normal_(m.weight, mode='fan_out')
                if m.bias is not None:
                    init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                init.constant_(m.weight, 1)
                init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                init.normal_(m.weight, 0, 0.01)
                if m.bias is not None:
                    init.constant_(m.bias, 0)
        

    def _make_layer(self, block, planes, num_blocks,stride):
        strides = [stride] + [1]*(num_blocks-1)
        layers = []

        for idx, stride in enumerate(strides):
            layers.append(block(self.in_planes, planes, stride, mode=self.mode, act_num=self.act_num, flops_mode=self.flops_mode))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x, temp=1):

        if self.sample_func == 'softmax':
            alpha = F.softmax(getattr(self, "alpha"), dim=-1)
        else:
            alpha = gumbel_softmax(getattr(self, "alpha"), temperature=temp)

        out = self.stem(x)
        #print('net forward: ',alpha,'\n', getattr(self, "alpha"))
        for idx, slayer in enumerate(self.layer1.children() ):
            out = slayer(out, alpha[idx*self.blen: (idx+1)*self.blen], getattr(self, "alpha")[idx*self.blen: (idx+1)*self.blen])
        
        for idx, slayer in enumerate(self.layer2.children() ):
            out = slayer(out, alpha[self.b1+idx*self.blen: self.b1+(idx+1)*self.blen], getattr(self, "alpha")[self.b1+idx*self.blen: self.b1+(idx+1)*self.blen])        

        for idx, slayer in enumerate(self.layer3.children() ):
            out = slayer(out, alpha[self.b2+idx*self.blen: self.b2+(idx+1)*self.blen],getattr(self, "alpha")[self.b2+idx*self.blen: self.b2+(idx+1)*self.blen])

        for idx, slayer in enumerate(self.layer4.children() ):
            out = slayer(out, alpha[self.b3+idx*self.blen: self.b3+(idx+1)*self.blen],getattr(self, "alpha")[self.b3+idx*self.blen: self.b3+(idx+1)*self.blen])        

        out = self.avgpool(out)
        out = out.view(out.size(0), -1)
        out = self.linear(out)
        return out


    def sample_single_path(self):
        op_layerwise_index = []
        layers = [self.layer1, self.layer2, self.layer3, self.layer4]

        for tlayer in layers:
            for block in tlayer:
                if self.backbone == 'RN18' or self.backbone == 'RN34':
                    op_layerwise_index.extend([block.conv1.active_list[0].item(), block.conv2.active_list[0].item()])
                else:
                    op_layerwise_index.extend([block.conv1.active_list[0].item(), block.conv2.active_list[0].item(), block.conv3.active_list[0].item()])
        return op_layerwise_index
 
    def coutlayers(self,container,base,size, alpha):
        flops_total = []
        
        for idx, slayer in enumerate(container):
            t_alpha = alpha[base+idx*self.blen: base+(idx+1)*self.blen]
            flops, size = slayer.conv1.forward_flops(size,t_alpha[0])

            flops_total.append(flops)
            flops, size = slayer.conv2.forward_flops(size,t_alpha[1])
            flops_total.append(flops)
            if self.blen>2:
                flops, size = slayer.conv3.forward_flops(size,t_alpha[2])
                flops_total.append(flops)

        return sum(flops_total), size

    def forward_flops(self,size=(3,224,224),temp=1):
        if self.sample_func == 'softmax':
            alpha = F.softmax(getattr(self, "alpha"), dim=-1)
        else:
            alpha = gumbel_softmax(getattr(self, "alpha"), temperature=temp)
        #don't count the first conv layer
        if self.dataset == 'imagenet' or size == (3,224,224):
            size = (64, 56,56)
        elif self.dataset == 'cifar' or size == (3, 32, 32):
            size = (64, 32,32)
        
        flops1, size = self.coutlayers(self.layer1,0,size,alpha)
        flops2, size = self.coutlayers(self.layer2,self.b1,size,alpha)
        flops3, size = self.coutlayers(self.layer3,self.b2,size,alpha)
        flops4, size = self.coutlayers(self.layer4,self.b3,size,alpha)
        return flops1+flops2+flops3+flops4
        
        

    def _loss(self, input, target, temp=1):

        logit = self(input, temp)
        loss = self._criterion(logit, target)

        return loss

    def _build_arch_parameters(self):
        num_ops = len(PRIMITIVES)
        setattr(self, 'alpha', nn.Parameter(Variable(1e-3*torch.ones(self.blen*sum(self.num_blocks), num_ops), requires_grad=True)))
        return {"alpha": self.alpha}


    def _reset_arch_parameters(self):
        num_ops = len(PRIMITIVES)
        getattr(self, "alpha").data = Variable(1e-3*torch.ones(self.blen*sum(self.num_blocks), num_ops), requires_grad=True)


    def clip(self):
        for line in getattr(self, "alpha"):
            max_index = line.argmax()
            line.data.clamp_(0, 1)
            if line.sum() == 0.0:
                line.data[max_index] = 1.0
            line.data.div_(line.sum())
    

        #return the best latency for the supernet
    def search_for_hw(self, Populations=200,Generations=50,platform="zcu102"):
        alpha_idx = self.sample_single_path()
        #alpha, backbone, platform="zcu102", NIND=200, MAX_GEN=100
        searched_hw, search_fps =search_hw_lat(alpha_idx,self.backbone,platform=platform,NIND=Populations,MAX_GEN=Generations)
        self.searched_hw = searched_hw
        #print(search_fps)
        

    def forward_hw_latency(self,platform="zcu102"):
        assert self.searched_hw is not None

        block_info = self.sample_single_path()
        # # block_latency
        fps,layer_wise_lat  = evaluate_latency(block_info, self.backbone,self.searched_hw,platform)#alpha, backbone, hw_params, platform='zcu102'
        
        
        latency = 0
        for layer_id in range(len(layer_wise_lat)):
            op_id = block_info[layer_id]

            alpha_ste = (1-self.alpha[layer_id][op_id]).detach() + self.alpha[layer_id][op_id]
            #print(alpha_ste, layer_wise_lat[layer_id])
            latency += alpha_ste * layer_wise_lat[layer_id]

        return latency,block_info
    

if __name__ == '__main__':
    from easydict import EasyDict as edict
    config = edict()
    # imageNet
    config.dataset = 'imagenet'
    config.sample_func = 'gumbel_softmax'
    config.mode = 'soft'
    config.flops_mode = 'single_path' # control the forward flops mode{'sum','single_path'}
    config.act_num = 1
    config.num_classes = 1000
    config.backbone = 'RN50'
    model = SuperNet(config).cuda()
    # print(model)
    # exit()
    input = torch.randn(1,3,224,224).cuda()
    re = model(input)
    a=model.sample_single_path()
    print("alpha len: ",len(model.alpha))
    print(len(a))
    model.search_for_hw()
    
    lats,_ = model.forward_hw_latency()
    fps = 1e9 / (lats * (1000 / 200))
    print('final fps, lats:', fps)
    #print(model.forward_flops((3,224,224)))#!note that the flops is already multipy the alpha, so default sum
    #cnt = 0
    # for name, param in model.named_parameters():
    #     print(name)
        