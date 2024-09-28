import torch
import torch.nn as nn
import numpy as np
from thop import profile
#from circulant_2d import BCM_Conv2d_fft as BCM_Conv
#from circulant_2d import BCM_Conv2d as BCM_Conv
from circulant_2d import BCMConv2d_FFT_Mconv as BCM_Conv

import sys
import os.path as osp


__all__ = ['TBS_Layer','OPS']

#conv2d bcm op count; m:model, x[0]:input tensor, y[0] output tensor
def count_BCMConv(m, x, y: torch.Tensor):
    #print(x[0].shape,y[0].shape)
    x = x[0]; y = y[0]
    batch_size,N,H,W=x.size()
    M, H_o, W_o  = y.size()
    kernel_size = m.kernel_size

    vanilla_ops = H_o*W_o * kernel_size*kernel_size*N * M
    if m.block_size ==4: # n^2 - > (nlogn+ 3/2n), here nlogn will be reused Tm times. reduce 3/2n, here we round up
        bcm_ops = np.ceil(vanilla_ops *  0.4) #1.5/4
    elif m.block_size == 8:
        bcm_ops = np.ceil(vanilla_ops* 0.2 )# 1.5/8
    elif m.block_size == 16:
        bcm_ops = np.ceil(vanilla_ops* 0.11) # 1.5/16
    else:
        raise Exception('wrong Custom op Count')
    m.total_ops += bcm_ops

custom_ops = {BCM_Conv: count_BCMConv}

# #* keep the caclulated flops and latency data file
flops_lookup_table = {}
flops_file_name = "flops_lookup_table.npy"
if osp.isfile(flops_file_name):
    flops_lookup_table = np.load(flops_file_name, allow_pickle=True).item()




BatchNorm2d = nn.BatchNorm2d
MaxPool = nn.MaxPool2d
#To use the Thop package



class TBS_Conv2d(nn.Module):
    def __init__(self,type_id,in_planes, out_planes, kernel_size=3, block_size=4, stride=1, pad=0, bias=True):
        super(TBS_Conv2d,self).__init__()
        self.in_planes   =  in_planes
        self.out_planes  =  out_planes
        self.kernel_size =  kernel_size
        self.pad         =  pad
        self.bias        =  bias
        self.block_size  =  block_size
        self.stride      =  stride
        self.type_id = type_id

        if type_id == 0:
            self.block = BCM_Conv(in_planes, out_planes, kernel_size=kernel_size, stride=stride, padding=pad, block_size=block_size, bias=bias)
        elif type_id ==1:
            self.block = nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size, stride=stride, padding=pad, bias=bias)
        else:
            raise Exception("Unkown OP")
    
    def forward(self,x):
        x = self.block(x)
        return x
    
    def set_stage(self, stage):
        assert stage == 'update_weight' or stage == 'update_arch'
        self.block.set_stage(stage)

    @staticmethod
    def _flops(type_id, h, w, in_planes, out_planes, kernel_size=3, block_size=4, stride=1, pad=0, bias=True):
        layer = TBS_Conv2d(type_id,in_planes, out_planes, kernel_size=kernel_size, stride=stride, pad=pad, block_size=block_size, bias=bias)
        # here, we need judge blocksize, or pass the block size through the fucntion parameters? such as (1,tokens,token_dim,block_size)?
        flops, params = profile(layer, inputs=(torch.randn(1, in_planes,h,w),), custom_ops=custom_ops)
        return flops

    def forward_flops(self, size):
        c_in, h_in, w_in = size
        
        c_out = self.out_planes

        if self.stride == 1:
            h_out = h_in; w_out = w_in
        else:
            h_out = h_in // 2; w_out = w_in // 2

        name = "Conv_Type{0}_H{1}_W{2}_Cin{3}_Cout{4}_K{5}_blocksize{6}_stride{7}".format( self.type_id,h_in, w_in, c_in, c_out,self.kernel_size,self.block_size, self.stride) 
        if name in flops_lookup_table:
            flops = flops_lookup_table[name]
        else:
            print("not found in flops_lookup_table:", name)
            flops = TBS_Conv2d._flops(self.type_id, h_in, w_in, c_in, c_out,self.kernel_size, self.block_size,self.stride, self.pad, self.bias)
            flops_lookup_table[name] = flops
            np.save(flops_file_name, flops_lookup_table)

        return flops,(c_out, h_out, w_out)

#type_id,in_planes, out_planes, kernel_size=3, block_size=4, stride=1, pad=0,
OPS = {
    'BCM_4'  : lambda C_in, C_out, kernel_size, stride, pad: TBS_Conv2d(0, C_in, C_out, kernel_size,  block_size=4, stride=stride,  pad=pad),
    'BCM_8'  : lambda C_in, C_out, kernel_size, stride, pad: TBS_Conv2d(0, C_in, C_out, kernel_size,  block_size=8, stride=stride,  pad=pad),
    'BCM_16' : lambda C_in, C_out, kernel_size, stride, pad: TBS_Conv2d(0, C_in, C_out, kernel_size,  block_size=16,stride=stride,  pad=pad),
    'vanilla': lambda C_in, C_out, kernel_size, stride, pad: TBS_Conv2d(1, C_in, C_out, kernel_size,  block_size=1, stride=stride,  pad=pad),
}


if __name__ =="__main__":
    c_in,c_out,h,w = 16,32,14,14
    kernel_size, pad , stride = 3,1,1
    input = torch.randn(1,c_in,h,w)

    for k,v in OPS.items():
        block1 = OPS[k](c_in,c_out,kernel_size,stride,pad)
        print(k)
        #print(block1)
        output = block1(input)
        ops, shape = block1.forward_flops( (c_in,h,w) )
        print('ops: {}, shape:{}'.format(ops,shape))
        #print(output.shape)