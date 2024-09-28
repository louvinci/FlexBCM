import numpy as np

DDRLatency = 64
BIT = 8
NormWidth = 64
# interface param
WeightBurstLength = 256
WeightOutstanding = 2
InBurstLength = 8
InOutstanding = 64
OutBurstLength = 8
OutOutstanding = 32
NormBurstLength = 64
NormOutstanding = 2

DEBUG = False


# def BRAM_CONSUME(data_bit, data_num):
#     if data_num < 16:
#         return 0
#     if data_bit > 36:
#         bram_perdata = np.ceil(data_bit / 36)
#         num = bram_perdata * np.ceil(data_num / 512)
#     elif data_bit > 18:
#         num = np.ceil(data_num / 512)
#     elif data_bit > 9:
#         num = np.ceil(data_num / 1024)
#     else:
#         num = np.ceil(data_num / 2048)
#     return num

def BRAM_CONSUME(data_bit, data_num):
    if data_num < 16:
        return 0
    if data_bit > 18:
        bram_perdata = np.ceil(data_bit / 36)
        if data_num > 1024:
            num = bram_perdata * np.ceil(data_num / 1024) *2
        else:
            num = bram_perdata * np.ceil(data_num / 512)
    elif data_bit > 9:
        num = np.ceil(data_num / 1024)
    else: # 9*2048, when more that one BRAM18K, the num should be 2
        if data_num >4096:
            num = np.ceil(data_num/4096)*2
        else:
            num = np.ceil(data_num / 2048)
    return num


def ConvPE_Resource(tile_param, K_max, S_max, bw_param):
    tr, tc, tm, tn = tile_param[0:4]
    in_bw, wt_bw, out_bw, bn_bw = bw_param[0:4]
    interface = BRAM_CONSUME(wt_bw * BIT, WeightBurstLength * WeightOutstanding) + \
                BRAM_CONSUME(in_bw * BIT, InBurstLength * InOutstanding) + \
                BRAM_CONSUME(out_bw * BIT, OutBurstLength * OutOutstanding) * 2 + \
                BRAM_CONSUME(bn_bw * BIT, NormBurstLength * NormOutstanding)
    in_buf = BRAM_CONSUME(tn * BIT, (S_max * tr + K_max - S_max) ** 2)
    wt_buf = BRAM_CONSUME(BIT, K_max * K_max) * tn * tm
    out_buf = BRAM_CONSUME(32, tr * tc) * tm
    total_bram = interface + (in_buf + wt_buf + out_buf) * 2
    if DEBUG:
        print("in_buf: {}, wt_buf: {}, out_buf: {}".format(in_buf, wt_buf, out_buf))

    conv_dsp = tn * tm / 2 + tn / 2
    st_dsp = out_bw * 4 + 8         # Norm_Quant need 4
    in_dsp = 14
    wt_dsp = 6

    total_dsp = in_dsp + wt_dsp + conv_dsp + st_dsp + 12
    if DEBUG:
        print("conv_dsp: {}, st_dsp: {}".format(conv_dsp, st_dsp))

    return total_bram, total_dsp


def CE_Latency(tile_param, N, K, S, in_bw, wt_bw):
    tr, tc, tm, tn = tile_param[0:4]
    in_num_cycle = in_bw * 8 / BIT
    wt_num_cycle = wt_bw * 8 / BIT

    in_burst = min(np.ceil(tn / in_bw), InBurstLength)                  # burst length
    in_bubble = max(DDRLatency - in_burst * (InOutstanding - 1), 0)
    in_num_total = (S * tr + K - S) * (S * tc + K - S) * np.ceil(np.ceil(tn / in_bw) / in_burst)    # num of burst req
    depth = 8
    in_lat = depth + DDRLatency + (in_burst * InOutstanding + in_bubble) * np.ceil(in_num_total / InOutstanding) - in_bubble

    # weight can all burst
    depth = 3
    wt_lat = depth + DDRLatency + np.ceil(K * K * tm * tn / wt_num_cycle) + 16

    depth = 12
    conv_lat = depth + K * K * tr * tc + 5

    fill_lat = tr * tc
    in_wt_lat = np.maximum(in_lat, wt_lat)
    loop_lat = np.maximum(in_wt_lat, conv_lat)
    loop_cnt = np.ceil(N / tn)
    total_lat = fill_lat + in_wt_lat + (loop_cnt - 1) * loop_lat + conv_lat + 10
    if DEBUG:
        print("in_lat: {}, wt_lat: {}, conv_lat: {}, ce_lat: {}".format(in_lat, wt_lat, conv_lat, total_lat))
    return total_lat


def St_Latency(tile_param, out_br_bw, bn_bw):
    tr, tc, tm, tn = tile_param[0:4]

    bn_num_cycle = bn_bw * 8 / NormWidth
    depth = 2
    bn_lat = depth + tm / bn_num_cycle

    out_burst = min(np.ceil(tm / out_br_bw), OutBurstLength)
    # out_burst = 1
    out_bubble = max(DDRLatency - out_burst * (OutOutstanding - 1), 0)
    out_num_total = tr * tc * np.ceil(np.ceil(tm / out_br_bw) / out_burst)
    depth = 151
    out_lat = depth + DDRLatency + (out_burst * OutOutstanding + out_bubble) * np.ceil(out_num_total / OutOutstanding) - out_bubble

    total_lat = bn_lat + out_lat + 10
    if DEBUG:
        print("bn_lat: {}, out_lat: {}".format(bn_lat, out_lat))
    return total_lat


def ConvPE_Latency(tile_param, bw_param, layer):
    tr, tc, tm, tn = tile_param[0:4]
    N, M, H, W = layer[:, 0], layer[:, 1], layer[:, 2], layer[:, 3]
    K, S, P = layer[:, 4], layer[:, 5], layer[:, 6]

    Hout = np.floor((H - K + 2 * P) / S) + 1
    Wout = np.floor((W - K + 2 * P) / S) + 1

    in_bw, wt_bw, out_br_bw, bn_bw = bw_param[0:4]

    ce_lat = CE_Latency(tile_param, N, K, S, in_bw, wt_bw)
    st_lat = St_Latency(tile_param, out_br_bw, bn_bw)

    loop_cnt = np.ceil(Hout / tr) * np.ceil(Wout / tc) * np.ceil(M / tm)
    loop_lat = np.maximum(ce_lat, st_lat)
    total_lat = ce_lat + (loop_cnt - 1) * loop_lat + st_lat

    return total_lat


def Near_Pow2(x):
    x = int(x)
    i = -1
    while x:
        x = x >> 1
        i += 1

    return 1 << i


def BandWidth_Alloc(tile_param, layers, assign_bw):
    tr, tc, tm, tn = tile_param[0:4]
    in_wkld, wt_wkld, out_br_wkld, bn_wkld = 0, 0, 0, 0
    for i in range(len(layers)):
        N, M, H, W, K, S, P = layers[i][0:7]
        Hout = (H - K + 2 * P) / S + 1
        Wout = (W - K + 2 * P) / S + 1
        in_wkld += H * W * N * M / tm
        wt_wkld += K * K * M * N * Hout / tr * Wout / tc
        out_br_wkld += Hout * Wout * M * 2  # out + br
        bn_wkld += Hout / tr * Wout / tc * M * NormWidth / 8
    Wkld = in_wkld + wt_wkld + out_br_wkld + bn_wkld

    in_bw = Near_Pow2(np.floor(in_wkld / Wkld * (assign_bw - 5)) + 1)
    wt_bw = Near_Pow2(np.floor(wt_wkld / Wkld * (assign_bw - 5)) + 1)
    out_br_bw = Near_Pow2(np.floor(out_br_wkld / Wkld * (assign_bw - 5)) + 2)
    bn_bw = Near_Pow2(np.floor(bn_wkld / Wkld * (assign_bw - 5)) + 1)

    rest_bw = assign_bw - in_bw - wt_bw - out_br_bw - bn_bw
    in_bw = min(tn, Near_Pow2(in_bw + rest_bw))
    rest_bw = assign_bw - in_bw - wt_bw - out_br_bw - bn_bw
    if wt_wkld > out_br_wkld:
        wt_bw = min(tn, Near_Pow2(wt_bw + rest_bw))
        rest_bw = assign_bw - in_bw - wt_bw - out_br_bw - bn_bw
        out_br_bw = min(tm, Near_Pow2(out_br_wkld + rest_bw))
    else:
        out_br_bw = min(tm, Near_Pow2(out_br_wkld + rest_bw))
        rest_bw = assign_bw - in_bw - wt_bw - out_br_bw - bn_bw
        wt_bw = min(tn, Near_Pow2(wt_bw + rest_bw))
    rest_bw = assign_bw - in_bw - wt_bw - out_br_bw - bn_bw
    bn_bw = min(tm, Near_Pow2(bn_bw + rest_bw))

    if DEBUG:
        print("BW: in: {}, wt: {}, out: {}, bn: {}".format(in_bw, wt_bw, out_br_bw / 2, bn_bw))
    return [in_bw, wt_bw, out_br_bw / 2, bn_bw]


def ConvPE_Modeling(tile_param, alloc_layers, assign_bw=64):
    """
    :param alloc_layers: [[N, M, H, W, K, S, P, bs], ...]
    :param assign_bw:
    :return:
    """
    # layers_lat = []
    # K_max = max(list(map(lambda x: x[4], alloc_layers)))
    # S_max = max(list(map(lambda x: x[5], alloc_layers)))
    K_max = max(alloc_layers[:, 4])
    S_max = max(alloc_layers[:, 5])

    if type(assign_bw) == list:
        band_param = assign_bw
    else:
        band_param = BandWidth_Alloc(tile_param, alloc_layers, assign_bw)

    cpe_bram, cpe_dsp = ConvPE_Resource(tile_param, K_max, S_max, band_param)

    # for i in range(len(alloc_layers)):
    #     lat = ConvPE_Latency(tile_param, band_param, alloc_layers[i])
    #     layers_lat.append(lat)
    layers_lat = ConvPE_Latency(tile_param, band_param, alloc_layers)

    return cpe_bram, cpe_dsp, band_param, layers_lat


if __name__ == "__main__":
    tr = tc = 7
    tm = 64
    tn = 32
    layer = [128, 128, 28, 28, 3, 1, 1, 0]
    alloc_layers = np.array([
        [64, 128, 28, 28, 1, 2, 0, 0],
        [128, 128, 7, 7, 3, 1, 1, 0],
        [128, 256, 7, 7, 1, 2, 0, 0]
        # [256, 256, 7, 7, 3, 1, 1, 0],
        # [128, 128, 14, 14, 3, 1, 1, 0],
        # [256, 256, 14, 14, 3, 1, 1, 0],
        # [64, 64, 28, 28, 3, 1, 1, 0],
        # [128, 128, 28, 28, 3, 1, 1, 0],
        # [64, 64, 56, 56, 3, 2, 1, 0],
    ])
    # assign_bw = 64
    assign_bw = [16, 16, 4, 4]

    in_depth = layer[2]*layer[3]*layer[0]/tn
    wt_depth = layer[4]*layer[4]*layer[0]*layer[1]/tn
    out_h = np.ceil((layer[2]-layer[4]+2*layer[6])/layer[5]) + 1
    out_w = np.ceil((layer[3]-layer[4]+2*layer[6])/layer[5]) + 1
    out_depth = out_h*out_w*layer[1]/8
    print('depth:')
    print('in: {} \t wt: {} \t out: {}'.format(in_depth, wt_depth, out_depth))

    bram, dsp, bw, lat = ConvPE_Modeling([tr, tc, tm, tn], alloc_layers, assign_bw)

    print('bram: {} \t dsp: {} \t latency: {}'.format(bram, dsp, lat))

