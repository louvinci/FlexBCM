import numpy as np
from GA_Problem import GA_Engine_C2D, GA_Engine_BCM, GA_Engine_All
from Seach_Space import C2D_SEARCH, BCM_SEARCH
from ConvPE_Modeling import ConvPE_Modeling
from BCMPU_modeling import BCMPU_modeling
# from analytical_model.GA_Problem import GA_Engine_C2D, GA_Engine_BCM, GA_Engine_All
# from analytical_model.Seach_Space import C2D_SEARCH, BCM_SEARCH
# from analytical_model.ConvPE_Modeling import ConvPE_Modeling
# from analytical_model.BCMPU_modeling import BCMPU_modeling
import time

# zcu102: BRAM18K:1824, DSP:2520, FF:548k,  LUT:274k
# zcu104: BRAM18K:624,  DSP:1728, FF:460k,  LUT:230k, URAM:96
# zc706:  BRAM18K:1090, DSP:900,  FF:437k,  LUT:218k
# U50:    BRAM18K:2688, DSP:5952, FF:1743k, LUT:872k, URAM:640 (2 dies), 47.3Mb BRAM, 180Mb URAM, 318 GB/s HBM2
platform_zc706 = {"name":706,"dsp":900, "bram":1060,"bw":12.8,"Mhz" : 150}  # 54bytes/cycle total
platform_zc102 = {"name":102,"dsp":2520,"bram":1824,"bw":21.3,"Mhz" : 200}  # 85bytes/cycle total, DSP:2520*1.5, BRAM:1824*0.85
platform_zc104 = {"name":104,"dsp":1728,"bram":2111,"bw":21,  "Mhz" : 200}  # 84bytes/cycle total
platform_u50   = {"name":50, "dsp":5952,"bram":3000,"bw":318, "Mhz" : 250}  #
global Bw_util
Bw_util = 0.75
LOG = True

class ResNet_GEN():
    def __init__(self, op_idx_lst, backbone):
        self.in_planes = 64
        self.op_idx_lst = op_idx_lst
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
        if backbone == 'RN18':
            num_blocks  = [2,2,2,2]
            blen = 2
        elif backbone == 'RN34':
            num_blocks  = [3,4,6,3]
            blen = 2
            assert len(op_idx_lst) >= 32, 'alpha for RN34 is not less than 32'
        elif backbone == 'RN50':
            num_blocks  = [3,4,6,3]
            blen =3
            assert len(op_idx_lst) >= 48, 'alpha for RN50 is not less than 48'
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
        blocks.append(  [ in_planes, planes,            Hin , Win,                    1,  1,       0,      op_lst[0], 0 ] ) # the last 0 or 1 means branch or not
        blocks.append(  [ planes,    planes,            Hin , Win,                    3,  stride,  1,      op_lst[1], 0 ] )
        blocks.append(  [ planes,    expansion*planes,  Hin/stride , Win/stride,      1,  1,       0,      op_lst[2], 0 ] )
        
        if stride != 1 or in_planes != expansion*planes:
            blocks.append([in_planes, expansion*planes, Hin, Win, 1, stride, 0 , 0, 1]) #! default vanilla
        
        self.H, self.W = self.H/stride, self.W/stride
        return blocks

    def BasicBlock(self,in_planes, planes, op_lst, stride=1, expansion=1):
        blocks = []
        Hin , Win = self.H, self.W
        blocks.append( [ in_planes, planes,  Hin, Win,                3, stride, 1, op_lst[0], 0 ] )
        blocks.append( [ planes,    planes,  Hin/stride, Win/stride,  3, 1,      1, op_lst[1], 0 ] )
        if stride != 1 or in_planes != expansion*planes:
            blocks.append( [ in_planes, expansion*planes, Hin, Win, 1,stride, 0, 0, 1]) #!
        
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


def evaluate_latency(alpha, backbone, hw_params, platform='zcu102'):
    if '102' in platform:
        tplatform = platform_zc102
    elif '104' in platform:
        tplatform = platform_zc104
    elif '706' in platform:
        tplatform = platform_zc706
    else:
        tplatform = platform_u50

    Mhz = tplatform['Mhz']

    net = ResNet_GEN(alpha,backbone).net_struct
    net = np.array(net).astype(np.int32)

    BCM_layers = []
    C2D_layers = []
    Branch_layers = []

    # [[N, M, H, W, K, S, P, bs, isbranch], ...]
    seq_idx = []
    for layer in net:
        if layer[8] == 1:
            Branch_layers.append(layer)
        else:
            seq_idx.append(layer[7])
            if layer[7] == 0:
                C2D_layers.append(layer)
            else:
                BCM_layers.append(layer)

    C2D_bram, C2D_dsp = 0, 0
    BCM_dsp,  BCM_bram = 0, 0
    C2D_fps,  BCM_fps = 0, 0
    C2D_layers_lat, BCM_layers_lat = [], []
    in_bw, wt_bw, out_bw, bn_bw = 0, 0, 0, 0
    in_bw_b, wt_bw_b, out_bw_b = 0, 0, 0

    C2D_layers +=Branch_layers

    if len(C2D_layers):
        tr_id, tm_id, tn_id, in_bw_id, wt_bw_id, out_bw_id, bn_bw_id = hw_params[0:7]
        tr, tm, tn    = C2D_SEARCH['tr'][tr_id], C2D_SEARCH['tm'][tm_id], C2D_SEARCH['tn'][tn_id]
        in_bw, wt_bw  = C2D_SEARCH['in_bw'][in_bw_id], C2D_SEARCH['wt_bw'][wt_bw_id]
        out_bw, bn_bw = C2D_SEARCH['out_bw'][out_bw_id], C2D_SEARCH['bn_bw'][bn_bw_id]

        C2D_bram, C2D_dsp, _, C2D_layers_lat = ConvPE_Modeling([tr, tr, tm, tn], np.array(C2D_layers),
                                                                    [in_bw, wt_bw, out_bw, bn_bw])

        C2D_fps = 1e9 / (np.sum(C2D_layers_lat) * 1000 / Mhz)

    if len(BCM_layers):
        tr_b_id, tm_b_id, tn_b_id, in_bw_b_id, wt_bw_b_id, out_bw_b_id = hw_params[7:13]
        tr_b, tm_b, tn_b = BCM_SEARCH['tr'][tr_b_id], BCM_SEARCH['tm'][tm_b_id], BCM_SEARCH['tn'][tn_b_id]
        in_bw_b, wt_bw_b = BCM_SEARCH['in_bw'][in_bw_b_id], BCM_SEARCH['wt_bw'][wt_bw_b_id]
        out_bw_b = BCM_SEARCH['out_bw'][out_bw_b_id]

        (BCM_dsp, BCM_bram), BCM_layers_lat = BCMPU_modeling(np.array(BCM_layers),
                                                             [tr_b, tr_b, tm_b, tn_b, in_bw_b, wt_bw_b, out_bw_b])
        
        BCM_fps = 1e9 / (np.sum(BCM_layers_lat) * 1000 / Mhz)

    dsp  = C2D_dsp + BCM_dsp
    bram = C2D_bram + BCM_bram
    total_bw = in_bw + wt_bw + max(out_bw*2, bn_bw) + in_bw_b + wt_bw_b + out_bw_b #TODO branch-add optimizaton

    layer_wise_lat = [] #* recover the alpha_inx layer, only contains the searched layers.
    
    idx, idy = 0, 0
    for i in range(len(seq_idx)):
        if seq_idx[i] == 0:
            layer_wise_lat.append(C2D_layers_lat[idx])
            idx +=1
        else:
            layer_wise_lat.append(BCM_layers_lat[idy])
            idy+=1
    
    if LOG:
        print("Total utilization: \t DSP: {}, \t BRAM: {}, \t BW: {}".format(dsp, bram, total_bw))
        if len(C2D_layers):
            print("ConvPU: tr: {}, tc: {}, tm: {}, tn: {}".format(tr, tr, tm, tn))
        else:
            print("ConvPU: tr: {}, tc: {}, tm: {}, tn: {}".format(0, 0, 0, 0))
        print("in_bw: {}, wt_bw: {}, out_bw: {}, bn_bw: {}".format(in_bw, wt_bw, out_bw, bn_bw))
        print("fps: {:.2f}".format(C2D_fps))
        print("\nBCMPU params:")
        if (len(BCM_layers)):
            print("tr: {}, tc: {}, tm: {}, tn: {}".format(tr_b, tr_b, tm_b, tn_b))
        else:
            print("tr: {}, tc: {}, tm: {}, tn: {}".format(0, 0, 0, 0))
        print("in_bw: {}, wt_bw: {}, out_bw: {}".format(in_bw_b, wt_bw_b, out_bw_b))
        print("fps: {:.2f}".format(BCM_fps))

    return min(C2D_fps, BCM_fps), layer_wise_lat


def search_hw_lat(alpha, backbone, platform="zcu102", NIND=200, MAX_GEN=100):
    #print(type(alpha), backbone)
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
    #print("total BW:", BW)
    net = ResNet_GEN(alpha,backbone).net_struct
    net = np.array(net).astype(np.int32)

    BCM_layers = []
    C2D_layers = []

    # [[N, M, H, W, K, S, P, bs], ...]
    for layer in net:
        if layer[7] == 0:
            C2D_layers.append(layer)
        else:
            BCM_layers.append(layer)

    start = time.time()
    hw_params, latency = GA_Engine_All(C2D_layers, BCM_layers, DSP, BRAM, BW, NIND, MAX_GEN)
    end = time.time()

   

    fps = 1e9 / (latency * (1000 / Mhz))
    if LOG:
        print(f'==== target platform: {platform} and Searched fps: {fps:.2f} ====')
        print("GA time: %s ms" % ((end - start) * 1000))
    return hw_params,fps


if __name__ == "__main__":
    # alloc_layers = [
    #     # conv2_x * 3
    #     [64, 64, 56, 56, 3, 1, 1, 1],
    #     [64, 64, 56, 56, 3, 1, 1, 1],
    #     [64, 64, 56, 56, 3, 1, 1, 3],
    #     [64, 64, 56, 56, 3, 1, 1, 3],
    #     [64, 64, 56, 56, 3, 1, 1, 1],
    #     [64, 64, 56, 56, 3, 1, 1, 1],
    #     # conv3_x * 4
    #     [64, 128, 56, 56, 3, 2, 1, 3],
    #     [128, 128, 28, 28, 3, 1, 1, 3],
    #     [128, 128, 28, 28, 3, 1, 1, 3],
    #     [128, 128, 28, 28, 3, 1, 1, 2],
    #     [128, 128, 28, 28, 3, 1, 1, 2],
    #     [128, 128, 28, 28, 3, 1, 1, 1],
    #     [128, 128, 28, 28, 3, 1, 1, 1],
    #     [128, 128, 28, 28, 3, 1, 1, 1],
    #     # conv4_x * 6
    #     [128, 256, 28, 28, 3, 2, 1, 3],
    #     [256, 256, 14, 14, 3, 1, 1, 3],
    #     [256, 256, 14, 14, 3, 1, 1, 3],
    #     [256, 256, 14, 14, 3, 1, 1, 3],
    #     [256, 256, 14, 14, 3, 1, 1, 2],
    #     [256, 256, 14, 14, 3, 1, 1, 2],
    #     [256, 256, 14, 14, 3, 1, 1, 2],
    #     [256, 256, 14, 14, 3, 1, 1, 1],
    #     [256, 256, 14, 14, 3, 1, 1, 1],
    #     [256, 256, 14, 14, 3, 1, 1, 1],
    #     [256, 256, 14, 14, 3, 1, 1, 1],
    #     [256, 256, 14, 14, 3, 1, 1, 1],
    #     # conv5_x * 3
    #     [256, 512, 14, 14, 3, 2, 1, 3],
    #     [512, 512, 7, 7, 3, 1, 1, 2],
    #     [512, 512, 7, 7, 3, 1, 1, 1],
    #     [512, 512, 7, 7, 3, 1, 1, 1],
    #     [512, 512, 7, 7, 3, 1, 1, 1],
    #     [512, 512, 7, 7, 3, 1, 1, 1],
    # ]
    # #np.random.seed(42)
    # max_fps = 0
    # min_fps = 2000
    # alpha_idx_max, alpha_idx_min = None,None
    # res_lst = []
    # def count_values_in_intervals(lst):
    #     intervals = [10,20,30,40,50,60,70, 80, 100,120]
    #     interval_counts = [(start, end, sum(1 for num in lst if start <= num < end)) for start, end in zip(intervals, intervals[1:])]
    #     return interval_counts
    
    # for _ in range(200):
    #     alpha = np.random.rand(48,4) # 32, 48
    #     #alpha[:,0]+=1
    #     alpha_idx = np.argmax(alpha, axis=-1)
    #     # alpha[:4,0]+=1
    #     # alpha[-10:-1,2]+=1
        
    #     backbone = 'RN50' # 'RN50'
        
    #     # for i in range(len(net_layers)):
    #     #     print(net_layers[i])
    #     # exit()
        
    #     print(alpha_idx)
    #     hw_params,fps = search_hw_lat(alpha_idx,backbone,'zcu102',NIND=200, MAX_GEN=40)
    #     res_lst.append(fps)
    #     evaluate_latency(alpha_idx, backbone, hw_params)
    #     if fps > max_fps:
    #         max_fps        =  fps
    #         alpha_idx_max  =  alpha_idx
    #     if fps < min_fps:
    #         min_fps        =  fps
    #         alpha_idx_min  =  alpha_idx
    # # import os
    # # predir = 'd:/BCM/A100-80 SearchCode/analytical_model/'
    # # logdir = os.path.join(predir,'RN34.log')
    # # with open() as f:
    # print("max fps:",max_fps,"\n idx: ",alpha_idx_max)
    # print("min fps:",min_fps,"\n idx: ",alpha_idx_min)
    # interval = count_values_in_intervals(res_lst)
    # print(interval)
    backbone = 'RN18' # 'RN50'
    #alpha = np.random.rand(16,4) # 32, 48 , 16
    alpha_idx = np.array([0,2,0,2,1,1,2,1,1,2,0,0,0,1,0,0]) 
    hw_params,fps = search_hw_lat(alpha_idx,backbone,'zcu102',NIND=200, MAX_GEN=40)
    evaluate_latency(alpha_idx, backbone, hw_params)