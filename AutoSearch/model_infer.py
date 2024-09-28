import torch
import torch.nn as nn
from torch.nn import init
import torch.nn.functional as F
from operations import OPS
from genotypes import PRIMITIVES
from thop import profile
from analytical_model.search_hw import search_hw_lat, evaluate_latency
#from circulant_2d import BCM_Conv2d_fft as Conv


# def conv1x1(in_planes, out_planes, bias=True, block_size=8, wbit=8, abit=8, stride=1):
#     """1x1 convolution"""
#     return Conv(in_planes, out_planes, kernel_size=1, block_size=block_size, stride=stride, padding=0, bias=bias)

def conv1x1(in_planes, out_planes, bias=True, block_size=8, wbit=8, abit=8, stride=1):
    """1x1 convolution"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, padding=0, bias=bias)

class MixedOp(nn.Module):
    def __init__(self, op_idx, C_in, C_out, kernel_size =3, stride=1, pad=1):
        super(MixedOp, self).__init__()
        self._op = OPS[PRIMITIVES[op_idx]](C_in, C_out, kernel_size, stride, pad)

    def forward(self, x):
        return self._op(x)

    def forward_flops(self, size):
        flops, size_o = self._op.forward_flops(size)
        
        return flops, size_o

class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, op_lst, stride=1):
        super(BasicBlock, self).__init__()

        self.conv1 =  MixedOp(op_lst[0], in_planes, planes,  kernel_size =3, stride=stride, pad=1)
        self.bn1   =  nn.BatchNorm2d(planes)
        self.conv2 =  MixedOp(op_lst[1], planes,    planes,  kernel_size =3, stride=1,      pad=1)
        self.bn2   =  nn.BatchNorm2d(planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion*planes:
            self.shortcut = nn.Sequential(
                #* default bcm4
                conv1x1(in_planes, self.expansion*planes,block_size=4,stride=stride),
                nn.BatchNorm2d(self.expansion*planes)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out

class Bottleneck(nn.Module):
    expansion = 4
    #
    def __init__(self, in_planes, planes, op_lst, stride=1):
        super(Bottleneck, self).__init__()
        self.conv1 = MixedOp(op_lst[0], in_planes, planes, kernel_size=1, stride=1, pad=0)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = MixedOp(op_lst[1], planes, planes,    kernel_size=3, stride=stride, pad=1)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = MixedOp(op_lst[2],planes, self.expansion*planes, kernel_size=1, stride=1,pad=0)
        self.bn3 = nn.BatchNorm2d(self.expansion*planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion*planes:
            self.shortcut = nn.Sequential(
                #* default bcm4
                conv1x1(in_planes, self.expansion*planes,block_size=4,stride=stride),
                nn.BatchNorm2d(self.expansion*planes)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = F.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out
    
class SubNet_Infer(nn.Module):
    def __init__(self, alpha, config):
        super(SubNet_Infer, self).__init__()
        self.in_planes = 64
        self.op_idx_lst = F.softmax(alpha, dim=-1).argmax(-1)
        #print('pick op:\n',self.op_idx_lst)
        
        self.num_classes = config.num_classes
        self.backbone    = config.backbone
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

        self.layer1 = self._make_layer(block, 64,  self.num_blocks[0], self.op_idx_lst[       :self.b1],   stride=1)
        self.layer2 = self._make_layer(block, 128, self.num_blocks[1], self.op_idx_lst[self.b1:self.b2],   stride=2)
        self.layer3 = self._make_layer(block, 256, self.num_blocks[2], self.op_idx_lst[self.b2:self.b3],   stride=2)
        self.layer4 = self._make_layer(block, 512, self.num_blocks[3], self.op_idx_lst[self.b3:self.b4],   stride=2)
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.linear = nn.Linear(512 * block.expansion, self.num_classes)
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


    def _make_layer(self, block, planes, num_blocks, op_lst,stride):
        strides = [stride] + [1]*(num_blocks-1)
        #print(op_lst)
        layers = []

        for idx, stride in enumerate(strides):
            layers.append(block(self.in_planes, planes, op_lst[self.blen*idx:self.blen*(idx+1)], stride))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)
    
    def forward(self,x):

        out = self.stem(x)
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)

        out = self.avgpool(out)
        out = out.view(out.size(0), -1)
        out = self.linear(out)
        return out
    
    def coutlayers(self,container,base,size):
        flops_total = []
        
        for slayer in container:
            flops, size = slayer.conv1.forward_flops(size)

            flops_total.append(flops)
            flops, size = slayer.conv2.forward_flops(size)
            flops_total.append(flops)
            if self.blen>2:
                flops, size = slayer.conv3.forward_flops(size)
                flops_total.append(flops)

        return sum(flops_total), size


    def forward_flops(self, size=(3,224,224)):
        flops_total = []

        if 'imagenet' in self.dataset:
            #don't count the stem layer
            size = (64, 56,56)
        elif 'cifar' in self.dataset:
            size = (64, 32,32)
        
        flops1, size = self.coutlayers(self.layer1,0,size)
        flops2, size = self.coutlayers(self.layer2,self.b1,size)
        flops3, size = self.coutlayers(self.layer3,self.b2,size)
        flops4, size = self.coutlayers(self.layer4,self.b3,size)
        return flops1+flops2+flops3+flops4
    
    def eval_latency(self,backbone,Populations=200,Generations=40,platform="zcu102",hardware=None):
        #[PRIMITIVES[op_id] for op_id in self.op_idx_list]
        block_info = self.op_idx_lst.cpu().numpy()
        #print(len(block_info))
        if hardware is None:
            #print(block_info)#alpha, backbone, platform="zcu102", NIND=200, MAX_GEN=100
            searched_hw,fps = search_hw_lat(block_info,backbone,platform,NIND=Populations, MAX_GEN=Generations)
        else:
            searched_hw = hardware
            #alpha, backbone, hw_params, platform='zcu102'
            #return min(C2D_fps, BCM_fps), max( sum(C2D_layers_lat), sum(BCM_layers_lat) )
            fps, layer_wise_latency = evaluate_latency(block_info,backbone,searched_hw,platform)
        
        return fps, searched_hw

if __name__ == '__main__':
    from easydict import EasyDict as edict
    
    config = edict()
    config.num_classes = 1000
    config.dataset = 'imagenet'
    config.backbone = 'RN34'
    if config.backbone == 'RN18':
        alpha = torch.randn( 18, len(PRIMITIVES))
    elif config.backbone == 'RN34':
        alpha = torch.randn( 34, len(PRIMITIVES)) 
    else:
        alpha = torch.randn( 50, len(PRIMITIVES))

    model = SubNet_Infer(alpha,config).cuda()
    # RN18-all-vanilla 1.67GMacs in ImageNet
    # RN18-all-vanilla 0.208GMACs
    print(model.forward_flops())
    print(model.eval_latency('RN34',Populations=200,Generations=20,platform="zcu102",hardware=None))