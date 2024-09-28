import numpy as np
from GA_Problem import GA_Engine_C2D, GA_Engine_BCM, GA_Engine_All
from Seach_Space import C2D_SEARCH, BCM_SEARCH
import time

# zcu102: BRAM18K:1824, DSP:2520, FF:548k,  LUT:274k
# zcu104: BRAM18K:624,  DSP:1728, FF:460k,  LUT:230k, URAM:96
# zc706:  BRAM18K:1090, DSP:900,  FF:437k,  LUT:218k
# U50:    BRAM18K:2688, DSP:5952, FF:1743k, LUT:872k, URAM:640 (2 dies), 47.3Mb BRAM, 180Mb URAM, 318 GB/s HBM2
platform_zc706 = {"name":706,"dsp":900, "bram":1060,"bw":12.8,"Mhz" : 150}  # 54bytes/cycle total
platform_zc102 = {"name":102,"dsp":2700,"bram":1550,"bw":21.3,"Mhz" : 200}  # 85bytes/cycle total, DSP:2520*1.5, BRAM:1824*0.85
platform_zc104 = {"name":104,"dsp":1728,"bram":2111,"bw":21,  "Mhz" : 200}  # 84bytes/cycle total
platform_u50   = {"name":50, "dsp":5952,"bram":3000,"bw":318, "Mhz" : 250}  #
Bw_util = 0.8


class ResNet_GEN():
    def __init__(self, alpha, backbone):
        self.in_planes = 64
        self.op_idx_lst = np.argmax(alpha, axis=-1)
        self.backbone = backbone
        self.H = 56
        self.W = 56
        '''
        input: 
            backbone, RN34 or RN50
            alpha: block size parameters probabilty, for RN34, len(alpha) is 32*4, RN50 len(alpha) is 48*4, (0:vanilla, 1:bcm4, 2:bcm8, 3:bcm16) 
        output:
            layers params( arrary[[cin,cout,hin,K, stride, Pad, bs],...]) 
        '''
        #op_idx_lst = F.softmax(alpha, dim=-1).argmax(-1)
    
        self.net_struct = []
        if backbone == 'RN34':
            num_blocks  = [3,4,6,3]
            blen = 2
            assert len(alpha) >= 32, 'alpha for RN34 is not less than 32'
        elif backbone == 'RN50':
            num_blocks  = [3,4,6,3]
            blen =3
            assert len(alpha) >= 48, 'alpha for RN50 is not less than 48'
        else:
            raise Exception("Unsupport backbone")
        b1 = num_blocks[0]   *  blen
        b2 = sum(num_blocks[0:2]) *  blen
        b3 = sum(num_blocks[0:3]) *  blen
        b4 = sum(num_blocks[0:4]) *  blen

        self.net_struct += self.make_layer( blen, 64,   num_blocks[0], self.op_idx_lst[         : b1 ],   stride=1)
        self.net_struct += self.make_layer( blen, 128,  num_blocks[1], self.op_idx_lst[  b1: b2 ],   stride=2)
        self.net_struct += self.make_layer( blen, 256,  num_blocks[2], self.op_idx_lst[  b2: b3 ],   stride=2)
        self.net_struct += self.make_layer( blen, 512,  num_blocks[3], self.op_idx_lst[  b3: b4 ],   stride=2)

    def make_layer(self, blen, planes, num_blocks, op_lst, stride):
        strides = [stride] + [1]*(num_blocks-1)
        #print(strides)
        layers = []
        for idx, stride in enumerate(strides):
            if blen == 3:
                expansion = 4
                re = self.Bottleneck( self.in_planes, planes,  op_lst[ blen*idx : blen*(idx+1)], stride)
            else:
                re = self.BasicBlock( self.in_planes, planes,  op_lst[ blen*idx : blen*(idx+1)], stride)
                expansion = 1
            #layers.append( re )
            layers+=re
            self.in_planes = planes * expansion
            
        return layers
    
    def Bottleneck(self,in_planes, planes,  op_lst, stride=1,expansion=4):
        blocks = []
        Hin , Win = self.H, self.W
        blocks.append(  [ in_planes, planes,            Hin , Win,                    1,  1,       0,      op_lst[0] ] )
        blocks.append(  [ planes,    planes,            Hin , Win,                    3,  stride,  1,      op_lst[1] ] )
        blocks.append(  [ planes,    expansion*planes,  Hin/stride , Win/stride,      1,  1,       0,      op_lst[2] ]  )
        
        if stride != 1 or in_planes != expansion*planes:
            blocks.append([in_planes, expansion*planes, Hin, Win, 1, stride, 0 , 0])
        
        self.H, self.W = self.H/stride, self.W/stride
        return blocks

    def BasicBlock(self,in_planes, planes, op_lst, stride=1, expansion=1):
        blocks = []
        Hin , Win = self.H, self.W
        blocks.append( [ in_planes, planes,  Hin, Win,                3, stride, 1, op_lst[0]] )
        blocks.append( [ planes,    planes,  Hin/stride, Win/stride,  3, 1,      1, op_lst[1]] )
        if stride != 1 or in_planes != expansion*planes:
            blocks.append( [ in_planes, expansion*planes, Hin, Win, 1,stride, 0, 0])
        
        self.H, self.W = self.H/stride, self.W/stride
        return blocks


def print_params(C2D_params, BCM_params):
    tr_id, tm_id, tn_id, in_bw_id, wt_bw_id, out_bw_id, bn_bw_id = C2D_params[0:7]
    tr, tm, tn = C2D_SEARCH['tr'][tr_id], C2D_SEARCH['tm'][tm_id], C2D_SEARCH['tn'][tn_id]
    in_bw, wt_bw = C2D_SEARCH['in_bw'][in_bw_id], C2D_SEARCH['wt_bw'][wt_bw_id]
    out_bw, bn_bw = C2D_SEARCH['out_bw'][out_bw_id], C2D_SEARCH['bn_bw'][bn_bw_id]
    print("ConvPE params:")
    print("tr: {}, tc: {}, tm: {}, tn: {}".format(tr, tr, tm, tn))
    print("in_bw: {}, wt_bw: {}, out_bw: {}, bn_bw: {}".format(in_bw, wt_bw, out_bw, bn_bw))

    tr_b_id, tm_b_id, tn_b_id, in_bw_b_id, wt_bw_b_id, out_bw_b_id = BCM_params[0:6]
    tr_b, tm_b, tn_b = BCM_SEARCH['tr'][tr_b_id], BCM_SEARCH['tm'][tm_b_id], BCM_SEARCH['tn'][tn_b_id]
    in_bw_b, wt_bw_b, out_bw_b = BCM_SEARCH['in_bw'][in_bw_b_id], BCM_SEARCH['wt_bw'][wt_bw_b_id], BCM_SEARCH['out_bw'][
        out_bw_b_id]
    print("BCMPU params:")
    print("tr: {}, tc: {}, tm: {}, tn: {}".format(tr_b, tr_b, tm_b, tn_b))
    print("in_bw: {}, wt_bw: {}, out_bw: {}".format(in_bw_b, wt_bw_b, out_bw_b))


def Search_HW_Lat(net, platform="zcu102", NIND=200, MAX_GEN=100):
    if '102' in platform:
        tplatform = platform_zc102
    elif '104' in platform:
        tplatform = platform_zc104
    elif '706' in platform:
        tplatform = platform_zc706
    else:
        tplatform = platform_u50

    DSP, BRAM, Bandwidth, Mhz = tplatform['dsp'], tplatform['bram'], tplatform['bw'], tplatform['Mhz']

    BW = np.floor(Bandwidth * Bw_util * (1000 / Mhz))   # bytes / cycle

    BCM_layers = []
    C2D_layers = []

    # [[N, M, H, W, K, S, P, bs], ...]
    for layer in net:
        if layer[7] == 0:
            C2D_layers.append(layer)
        else:
            BCM_layers.append(layer)

    # time(GA_Engine_BCM + GA_Engine_C2D) > time(GA_Engine_All)
    # start = time.time()
    # BCM_params, BCM_lat = GA_Engine_BCM(BCM_layers, DSP, BRAM, BW, NIND=100, MAX_GEN=20)
    # C2D_params, C2D_lat = GA_Engine_C2D(C2D_layers, DSP, BRAM, BW, NIND=100, MAX_GEN=20)
    # end = time.time()
    # print("GA 1 time: %s ms" % ((end - start) * 1000))
    # print_params(C2D_params, BCM_params)

    start = time.time()
    hw_params, latency = GA_Engine_All(C2D_layers, BCM_layers, DSP, BRAM, BW, NIND, MAX_GEN)
    end = time.time()
    print("GA time: %s ms" % ((end - start) * 1000))

    fps = 1e9 / (latency * (1000 / Mhz))
    print(f'==== target platform: {platform} and fps: {fps:.2f} ====')
    print_params(hw_params[0:7], hw_params[7:13])


if __name__ == "__main__":
    alloc_layers = [
        # conv2_x * 3
        [64, 64, 56, 56, 3, 1, 1, 2],
        [64, 64, 56, 56, 3, 1, 1, 2],
        [64, 64, 56, 56, 3, 1, 1, 3],
        [64, 64, 56, 56, 3, 1, 1, 3],
        [64, 64, 56, 56, 3, 1, 1, 2],
        [64, 64, 56, 56, 3, 1, 1, 0],
        # conv3_x * 4
        [64, 128, 56, 56, 3, 2, 1, 4],
        [128, 128, 28, 28, 3, 1, 1, 4],
        [128, 128, 28, 28, 3, 1, 1, 4],
        [128, 128, 28, 28, 3, 1, 1, 3],
        [128, 128, 28, 28, 3, 1, 1, 3],
        [128, 128, 28, 28, 3, 1, 1, 2],
        [128, 128, 28, 28, 3, 1, 1, 1],
        [128, 128, 28, 28, 3, 1, 1, 0],
        # conv4_x * 6
        [128, 256, 28, 28, 3, 2, 1, 4],
        [256, 256, 14, 14, 3, 1, 1, 4],
        [256, 256, 14, 14, 3, 1, 1, 4],
        [256, 256, 14, 14, 3, 1, 1, 4],
        [256, 256, 14, 14, 3, 1, 1, 3],
        [256, 256, 14, 14, 3, 1, 1, 3],
        [256, 256, 14, 14, 3, 1, 1, 3],
        [256, 256, 14, 14, 3, 1, 1, 2],
        [256, 256, 14, 14, 3, 1, 1, 2],
        [256, 256, 14, 14, 3, 1, 1, 2],
        [256, 256, 14, 14, 3, 1, 1, 1],
        [256, 256, 14, 14, 3, 1, 1, 0],
        # conv5_x * 3
        [256, 512, 14, 14, 3, 2, 1, 4],
        [512, 512, 7, 7, 3, 1, 1, 3],
        [512, 512, 7, 7, 3, 1, 1, 2],
        [512, 512, 7, 7, 3, 1, 1, 2],
        [512, 512, 7, 7, 3, 1, 1, 2],
        [512, 512, 7, 7, 3, 1, 1, 0],
    ]
    
    alpha = np.random.rand(32,4) # 32, 48
    # print(np.argmax(alpha, axis=-1))
    backbone = 'RN34' # 'RN50'
    net_layers = ResNet_GEN(alpha,backbone).net_struct

    Search_HW_Lat(alloc_layers,NIND=200, MAX_GEN=60)