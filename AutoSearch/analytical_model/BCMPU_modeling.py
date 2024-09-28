'''version 8-5
numpy arrary type
layer set :[[[cin,cout,hin,win,K,stride,pad,blk_index]],..,]; blk_index: the blocksize index [0:vanilla, 1:bcm4,2:bcm8,3:bcm16]
hardware parameter, [Tr,Tc,Tm,Tn,bw_in, bw_wt, bw_out]
considering that the seached parameters are small, we directly search hw (including bandwidth) using GA.
================= constraints:=================
            Tm>=16, it seems direct searching the i of 16*i is OK
            Tn>=8,  8*j
            bw_in <= 16, bw_out <=16. bw_wt<=Tn
'''
import numpy as np
global offchip_setup, HBS, AXI16_BRAM, AXI32_BRAM
global WTBurstLength, INBurstLength, WTOutstanding, INOutstanding
offchip_setup=48
HBS=8
AXI16_BRAM = 37
AXI32_BRAM = 67
LOG=False
BLK = np.array([1,4,8,16])
INBurstLength = 32
INOutstanding = 32
OUTBurstLength = 32
OUTOutstanding = 32

def Wt_lat(K,Tn,Tm,blk_size,BW_wt):
    #the actual weight size is Tm/(BS/2)* Tn/(BS/2) K*K * (BS/2)
    #Reshape Tm/(BS/2)* Tn/(BS/2) K*K * (BS/2) -> Tm/(BS/2) *K*K*Tn
    stage_lat = 4 + 24
    loop_cnt = np.ceil(Tn/BW_wt) * (Tm/(blk_size/2)) *K*K
    lat =  offchip_setup + stage_lat + loop_cnt
    return lat

def In_lat(Tr,Tc,Tn,K,S,BW_in=16):
    stage_lat = 30 # FFT setup latency
    #* Tn/HBS is set to do fft operation. below equation makes BW_in <= 16
    in_burst = min(np.ceil( Tn / BW_in), INBurstLength)
    in_bubble = max(offchip_setup - in_burst * (INOutstanding - 1), 0)

    loop_cnt = (S*(Tr-1) + K)*(S*(Tc-1) + K)* np.ceil(Tn/HBS) * np.ceil( np.ceil(2*HBS/BW_in) / in_burst)


    lat = offchip_setup + stage_lat + (in_burst * INOutstanding + in_bubble) * np.ceil(loop_cnt/INOutstanding)  - in_bubble

    return lat

def Out_lat(Tr,Tc, Tm,BW_out=16):
    stage_lat = 31 # IFFT setup latency
    #* Tm/HBS is set to do Ifft operation. below equation makes BW_out <= 16

    out_burst = min(np.ceil(Tm / BW_out), OUTBurstLength)
    loop_cnt = Tr*Tc*np.ceil(Tm/HBS)*np.ceil( np.ceil(2*HBS/BW_out) / out_burst )
    out_bubble = max(offchip_setup - out_burst * (OUTOutstanding - 1), 0)
    lat = offchip_setup+ stage_lat + (out_burst * OUTOutstanding + out_bubble) * np.ceil(loop_cnt / OUTOutstanding) - out_bubble
    return lat

def BCMU_latency(layer_param, hw_param):

    Tr,Tc,Tm,Tn,BW_in,BW_wt,BW_out = hw_param[0:7]
    ch_in,ch_out,Hin,Win        = layer_param[:,0],layer_param[:,1],layer_param[:,2],layer_param[:,3]
    K,Stride,Pad,blk_size_index = layer_param[:,4],layer_param[:,5],layer_param[:,6],layer_param[:,7]
    # here Tm,Tn is the complex compute, so Tm/2,Tn/2
    Hout , Wout = (Hin+2*Pad-K)//Stride + 1, (Win+2*Pad-K)//Stride + 1

    blk_size = BLK[blk_size_index.astype(np.int32)]

    total_lat = 0
    stage_lat = 25
    reuse_cnt = HBS / (blk_size/2)
    loop_cnt = K*K*Tr*Tc*reuse_cnt
    comp_lat = stage_lat + loop_cnt

    in_lat  = In_lat(Tr,Tc,Tn,K,Stride,BW_in)
    wt_lat  = Wt_lat(K,Tn,Tm,blk_size,BW_wt)
    out_lat = Out_lat(Tr,Tc,Tm,BW_out)

    ## compute core
    compute_loop = np.ceil(ch_in/(2*Tn))
    in_wt_lat = np.maximum(in_lat, wt_lat)
    loop_lat  = np.maximum(in_wt_lat, comp_lat)
    comp_core_lat = in_wt_lat + (compute_loop -1) * loop_lat + comp_lat

    ## main function

    total_loop = np.ceil(Hout/Tr) * np.ceil(Wout/Tc) * np.ceil(ch_out/(2*Tm))
    total_lat  = 6 + comp_core_lat + (total_loop-1)*np.maximum(comp_core_lat, out_lat) + out_lat
    if LOG:
        print("in lat:{}, wt lat:{}, comp lat:{}, store lat:{}".format(in_lat,wt_lat,comp_lat,out_lat))
        print("total latency: {}".format(total_lat))
    return total_lat

def BCMU_resource(alloc_layers,hw_params):
    FFT_dsp, Axi_BRAM = 140, AXI16_BRAM #* FFT and IFFT DSP consuming, and axi outstanding BRAM (in:16 outstading, out 16 outstanding)
    Kmax, Smax = max(alloc_layers[:,4]), max(alloc_layers[:,5])
    Tr,Tc,Tm,Tn = hw_params[0:4]
    inlen = (Smax*(Tr-1)+Kmax) * (Smax*(Tc-1)+Kmax)
    outlen = Tr*Tc
    if inlen > 4096:
        in_depth =     np.ceil (  inlen / 4096 )*2
    else:
        in_depth =  np.ceil (  inlen / 2048 )
    inbuf = 4 * Tn * in_depth # don't apply data pack INT8,

    if outlen > 1024:
        out_depth = np.ceil(outlen / 1024) *2
    else:
        out_depth = np.ceil(outlen/512)
    obuf  = 4 * Tm * out_depth  # INT32 complex data,
    D_arrary = 3 * HBS * np.ceil(Tm/(2*HBS)) * np.ceil(Tn/HBS) + 8
    if LOG:
        print("BRAM: in_buf: {}, outbuf:{}".format(inbuf,obuf))
    return FFT_dsp+D_arrary,Axi_BRAM+inbuf+obuf

#* TOP function
def BCMPU_modeling(alloc_layers,hw_params):
    assert len(alloc_layers[0])>=8,'layer param error: Cin,Cout,Hin,Win,K,S,Pad,bs'
    assert len(hw_params)==7,       'hw param error:Tr,Tc,Tm,Tn,BW_in,BW_wt,BW_out'
    layer_wise_lat = []
    dsp, bram = BCMU_resource(alloc_layers,hw_params)

    layer_wise_lat = BCMU_latency(alloc_layers,hw_params)
    # for i in range(len(alloc_layers)):
    #     layer_lat = BCMU_latency(alloc_layers[i],hw_params)
    #     layer_wise_lat.append(layer_lat)
    return (dsp,bram), layer_wise_lat

def test(cfg, log_dir):

    import time
    blk_indx = np.where(BLK == cfg.blk_size)[0][0]
    Hin,Win,Cin,Cout,K,S,Pad = cfg.Hin, cfg.Win, cfg.Cin, cfg.Cout, cfg.K , cfg.Stride, cfg.Pad
    Tr,Tc,Tm,Tn = cfg.Tr,cfg.Tc,cfg.Tm,cfg.Tn
    BW_in,BW_wt,BW_out = cfg.BW_in, cfg.BW_wt, cfg.BW_out  # Byte/cycle

    layers_param = np.array([
                             [Cin,Cout,Hin,Win,K,S,Pad,blk_indx],
                             [64, 128, 28, 28, 1, 1, 0, 1],
                            [128, 128, 7, 7, 3, 1, 1, 1],
                            [128, 256, 7, 7, 1, 2, 0, 1]
                             ])
    hw_param    = np.array([Tr,Tc,Tm,Tn,BW_in,BW_wt,BW_out])
    b_time =  time.time()
    (dsp,bram), layers_lat = BCMPU_modeling(layers_param,hw_param)
    e_time = time.time()
    print("predict  dsp:{} bram:{} latency:{}; comsuming:{:.3f}s".format(dsp,bram, sum(layers_lat),e_time-b_time))
    now = time.time()
    local = time.localtime(now)
    formatted = time.strftime("%Y-%m-%d %H:%M:%S", local)
    # with open(log_dir,'a') as file:
    #     print(formatted,file=file)
    #     print(cfg,file=file)
    #     print("predict latency: {0:.1f}\n\n".format(layers_lat[0]),file=file)

if __name__ == "__main__":
    from easydict import EasyDict as edict
    cfg = edict()
    cfg.blk_size = 8
    cfg.Hin, cfg.Win = 28,28
    cfg.Cin, cfg.Cout = 256,256
    cfg.K = 3
    # cfg.Stride = 1
    # cfg.Pad = 0

    cfg.Stride = 2
    cfg.Pad = 1
    cfg.Tr,cfg.Tc,cfg.Tm,cfg.Tn = 7,7,64,32

    #cfg.Tr,cfg.Tc,cfg.Tm,cfg.Tn = 7,7,32,32
    #cfg.Tr,cfg.Tc,cfg.Tm,cfg.Tn = 14,14,32,16
    #cfg.Tr,cfg.Tc,cfg.Tm,cfg.Tn = 28,28,32,8
    #cfg.Tr,cfg.Tc,cfg.Tm,cfg.Tn = 56,56,16,8
    cfg.BW_in,cfg.BW_wt,cfg.BW_out = 32,1,32

    import os
    predir = 'd:/BCM/Github_BCM_CoDesign/BCM_CNN-Pytorch/Accelerator/simulator/'
    logdir = os.path.join(predir,'simulator_blk.log')
    test(cfg,'./')
    # for tblk in [4,8,16]:
    #     for tK in [3,1]:
    #         for tH in [7,14,28,56]:
    #             for tCin in [64, 128, 256, 512]:
    #                 cfg.Hin, cfg.Win  = tH, tH
    #                 cfg.Cin, cfg.Cout = tCin, tCin
    #                 cfg.blk_size, cfg.K = tblk, tK
    #                 if tK == 3:
    #                     cfg.pad = 1
    #                 else:
    #                     cfg.pad = 0
    #                 logdir = os.path.join(predir,'simulator_blk{0}-3.log'.format(tblk))
    #                 test(cfg,logdir)
    
    