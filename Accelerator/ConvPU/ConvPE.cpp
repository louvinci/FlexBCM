#include"ConvPE.h"

// H x W x N/Tn 
void Load_In(   ap_uint<InWidth> *input, 
                ap_int<InWidth> in_buf[S_max*Tr+K_max-S_max][S_max*Tc+K_max-S_max],
                unsigned H, unsigned W, unsigned N,
                unsigned short K, unsigned short S, unsigned short P,
                unsigned row, unsigned col, unsigned nn) {

    unsigned short rloops = S*Tr + K - S;
    unsigned short cloops = S*Tc + K - S;
    int r_base = row*S - P;
    int c_base = col*S - P;
    ap_uint<InWidth> *base_addr;

    In_R:
    for (unsigned short r = 0; r < rloops; r++) {
        In_C:
        for (unsigned short c = 0; c < cloops; c++) {
#pragma HLS PIPELINE
            base_addr = input + (r_base + r)*W*(N >> TnBIT) + (c_base + c)*(N >> TnBIT) + (nn >> TnBIT);
            if (r_base + r < 0 || r_base + r > H - 1 || c_base + c < 0 || c_base + c > W - 1) {
                in_buf[r][c] = 0;
            } else {
                // in_buf[r][c] = 1;
                in_buf[r][c] = *base_addr;
            }
        }
    }

}

// M/Tm x N/Tn x Kx x Ky x Tm
void Load_Wt(   ap_uint<WtWidth> *weight,
                data_t w_buf[Tn][Tm][K_max][K_max],
                unsigned N, unsigned M, unsigned short K, 
                unsigned nn, unsigned mm) {

    unsigned short loop_cnt = K*K*Tm;
    unsigned char kx = 0, ky = 0, m = 0;
    ap_uint<WtWidth> *base_addr = weight + ((mm >> TmBIT)*(N >> TnBIT) + (nn >> TnBIT))*K*K*Tm;
    ap_uint<WtWidth> data_w;

    Lw_l:
    for (unsigned short l = 0; l < loop_cnt; l++) {
#pragma HLS PIPELINE
        data_w = base_addr[l];
        Lw_tn:
        for (unsigned char tn = 0; tn < Tn; tn++) {
#pragma HLS UNROLL
            w_buf[tn][m][kx][ky].range(BIT-1, 0) = data_w.range((tn+1)*BIT-1, tn*BIT);
            // w_buf[tn][m][kx][ky].range(BIT-1, 0) = tn;
        }
        if (m == Tm - 1) {
            m = 0;
            if (ky == K - 1) {
                ky = 0;
                kx++;
            } else {
                ky++;
            }
        } else {
            m++;
        }
    }
}

// return (A*W0, A*W1)
ap_int<32> MUL_INT8(ap_int<8> A, ap_int<8> W0, ap_int<8> W1)
{
    //ap_int<24> W;
    //W = (W0, ap_uint<16>(0)) + ap_int<24>(W1);
    ap_int<25> W= W0;
    W <<= 16;
    W+=W1;
	
    ap_int<16> r0;
    ap_int<16> r1;

    (r0, r1) = A*W;

    r0 = r0+r1[16-1];

    return (r0,r1);
}


void Conv(  ap_int<InWidth> in_buf[S_max*Tr+K_max-S_max][S_max*Tc+K_max-S_max],
            data_t w_buf[Tn][Tm][K_max][K_max],
            ap_int<AccBIT> out_buf[Tm][Tr][Tc],
            unsigned short K, unsigned short S) {
    
    CU_KX:
    for (unsigned char kx = 0; kx < K; kx++) {
        CU_KY:
        for (unsigned char ky = 0; ky < K; ky++) {
            CU_R:
            for (unsigned char r = 0; r < Tr; r++) {
                CU_C:
                for (unsigned char c = 0; c < Tc; c++) {
#pragma HLS PIPELINE
                    CU_M:
                    for (unsigned short m = 0; m < Tm; m+=2) {
#pragma HLS UNROLL
                        CU_N:
                        for (unsigned short n = 0; n < Tn; n++) {
#pragma HLS UNROLL
                            data_t d_in = in_buf[r*S + kx][c*S + ky].range((n+1)*BIT-1, n*BIT);
                            ap_int<AccBIT> d_out = MUL_INT8(d_in, w_buf[n][m][kx][ky], w_buf[n][m+1][kx][ky]);
                            ap_int<16> tmp1 = d_out.range(31, 16);
                            ap_int<16> tmp2 = d_out.range(15, 0);
                            out_buf[m][r][c] += tmp1;
                            out_buf[m+1][r][c] += tmp2;
                        }
                    }
                }
            }
        }
    }
}


void FillZeros(ap_int<AccBIT> out_buf[Tm][Tr][Tc]){

	for(unsigned char tr=0;tr<Tr;tr++){
		for(unsigned char tc=0;tc<Tc;tc++){
#pragma HLS PIPELINE
			for(unsigned short tm=0;tm<Tm;tm++){
#pragma HLS UNROLL
				out_buf[tm][tr][tc] = 0;
			}
		}
	}
}


void Conv_Engine(   ap_uint<InWidth> *input, 
                    ap_uint<WtWidth> *weight,
                    ap_int<AccBIT> out_buf[Tm][Tr][Tc],
                    unsigned H, unsigned W, unsigned N, unsigned M,
                    unsigned short K, unsigned short S, unsigned short P, 
                    unsigned row, unsigned col, unsigned mm) {

    ap_int<InWidth> in_buf_0[S_max*Tr+K_max-S_max][S_max*Tc+K_max-S_max];
#pragma HLS BIND_STORAGE variable=in_buf_0 type=ram_2p impl=bram
    ap_int<InWidth> in_buf_1[S_max*Tr+K_max-S_max][S_max*Tc+K_max-S_max];
#pragma HLS BIND_STORAGE variable=in_buf_0 type=ram_2p impl=bram

    data_t w_buf_0[Tn][Tm][K_max][K_max];
#pragma HLS ARRAY_PARTITION variable=w_buf_0 dim=2 complete
#pragma HLS ARRAY_PARTITION variable=w_buf_0 dim=1 complete
    data_t w_buf_1[Tn][Tm][K_max][K_max];
#pragma HLS ARRAY_PARTITION variable=w_buf_1 dim=2 complete
#pragma HLS ARRAY_PARTITION variable=w_buf_1 dim=1 complete

    bool flag = true;
    unsigned short nn;

    FillZeros(out_buf);

    CE:
    for (nn = 0; nn < N; nn += Tn) {
        if (nn == 0) {
            Load_In(input, in_buf_0, H, W, N, K, S, P, row, col, nn);
            Load_Wt(weight, w_buf_0, N, M, K, nn, mm);
        } else {
            if (flag) {
                Load_In(input, in_buf_1, H, W, N, K, S, P, row, col, nn);
                Load_Wt(weight, w_buf_1, N, M, K, nn, mm);
                Conv(in_buf_0, w_buf_0, out_buf, K, S);
                flag = !flag;
            } else {
                Load_In(input, in_buf_0, H, W, N, K, S, P, row, col, nn);
                Load_Wt(weight, w_buf_0, N, M, K, nn, mm);
                Conv(in_buf_1, w_buf_1, out_buf, K, S);
                flag = !flag;
            }
        }
    }
    if (flag) {
        Conv(in_buf_0, w_buf_0, out_buf, K, S);
    } else {
        Conv(in_buf_1, w_buf_1, out_buf, K, S);
    }

}


// M
void Load_Nm(   ap_uint<NormWidth> *norm,
                ap_uint<NormWidth> norm_buf[Tm],
                unsigned mm) {

    ap_uint<NormWidth> *base_addr = norm + mm;
    
    LN_l:
    for (unsigned short l = 0; l < Tm; l++) {
#pragma HLS PIPELINE
        norm_buf[l] = base_addr[l];
        // norm_buf[l] = l;
    }

}


ap_int<BIT> Norm_Quant(ap_int<AccBIT> in, ap_uint<NormWidth> norm, ap_uint<16> out_shift) {
    const int MAX = (1 << (BIT - 1)) - 1;
    const int MIN = -(1 << (BIT - 1));
    ap_int<AccBIT> wt, bias, t_res;
    wt.range(AccBIT - 1, 0) = norm.range(AccBIT - 1, 0);
    bias.range(AccBIT - 1, 0) = norm.range(2*AccBIT - 1, AccBIT);
    t_res = (in*wt + bias) >> out_shift;

    ap_int<BIT> res;

    if (t_res > MAX) {
        res = MAX;
    } else if (t_res < MIN) {
        res = MIN;
    } else {
        res = t_res;
    }

    return res;
}



void BA_BN_St(  ap_uint<OutWidth> *output,
                ap_uint<OutWidth> *branch,
                ap_uint<NormWidth> *norm,
                ap_int<AccBIT> out_buf[Tm][Tr][Tc],
                unsigned Hout, unsigned Wout, unsigned M, 
                unsigned row, unsigned col, unsigned mm,
                ap_uint<16> out_shift) {

    
    ap_uint<NormWidth> norm_buf[Tm];
#pragma HLS ARRAY_PARTITION variable=norm_buf dim=1 complete

    const unsigned short TmLoops = Tm/OutNum;
    unsigned short MCnt = M/OutNum;
    unsigned short mm_offset = mm/OutNum;
    unsigned RCoffset;
    ap_uint<OutWidth> *out_base;
    ap_uint<OutWidth> *br_base;
    ap_uint<OutWidth> data_s;
    ap_uint<OutWidth> data_b;
    ap_uint<NormWidth> data_n;


    Load_Nm(norm, norm_buf, mm);

//    St_r:
//    for (unsigned short r = 0; r < Tr; r++) {
//        St_c:
//        for (unsigned short c = 0; c < Tc; c++) {
//#pragma HLS PIPELINE II=4
////#pragma HLS DEPENDENCE false inter variable=out_base
//            RCoffset = ((row + r)*Wout + col + c)*MCnt;
//            out_base = output + RCoffset + mm_offset;
//            br_base = branch + RCoffset + mm_offset;
//            St_m:
//            for (unsigned short m = 0; m < TmLoops; m++) {
//#pragma HLS UNROLL
//                data_b = br_base[m];
//                // data_b = m;
//                St_i:
//                for (unsigned char i = 0; i < OutNum; i++) {
//#pragma HLS UNROLL
//                    data_t d_t = data_b.range((i+1)*BIT-1, i*BIT);
//                    out_buf[m*OutNum + i][r][c] += d_t;
//                    data_s.range((i+1)*BIT-1, i*BIT) = Norm_Quant(out_buf[m*OutNum + i][r][c], norm_buf[m*OutNum + i]);
//                }
//                out_base[m] = data_s;
//            }
//        }
//    }

     unsigned short r = 0, c = 0,m=0;
     unsigned loop_cnt = Tr*Tc*TmLoops;
//
//     St_r:
//     for (unsigned short cnt = 0; cnt < loop_cnt; cnt++) {
//#pragma HLS PIPELINE II=1
//    	 RCoffset = ((row + r)*Wout + col + c)*MCnt;
//		 out_base = output + RCoffset + mm_offset;
//		 br_base  = branch + RCoffset + mm_offset;
//		 data_b = br_base[m];
//		 St_i:
//		 for (unsigned char i = 0; i < OutNum; i++) {
//#pragma HLS UNROLL
//			 ap_int<AccBIT> tmp = out_buf[m*OutNum + i][r][c] ;
//			 data_t d_t = data_b.range((i+1)*BIT-1, i*BIT);
//			 tmp+= d_t;
//			 data_t res = Norm_Quant(tmp, norm_buf[m*OutNum + i]);
//			 data_s.range((i+1)*BIT-1, i*BIT) = res.range(BIT-1,0);
//		 }
//		 out_base[m] = data_s;
//
//         if (c == Tc - 1) {
//             c = 0;
//             if(r== Tr-1){
//            	 r=0;
//            	 m+=1;
//             }else{
//            	 r+=1;
//             }
//         } else {
//             c++;
//         }
//
//     }

    //unsigned short r = 0, c = 0;

    St_r:
	for(unsigned short cnt = 0; cnt < Tr*Tc; cnt++){
#pragma HLS PIPELINE II=TmLoops
		RCoffset = ((row + cnt/Tc)*Wout + col + cnt%Tc)*MCnt;
		out_base = output + RCoffset + mm_offset;
		br_base  = branch + RCoffset + mm_offset;
		for (unsigned short m = 0; m < TmLoops; m++) {
			 data_b = br_base[m];
			 St_i:
			 for (unsigned char i = 0; i < OutNum; i++) {
	#pragma HLS UNROLL
				 ap_int<AccBIT> tmp = out_buf[m*OutNum + i][cnt/Tc][cnt%Tc] ;
				 data_t d_t = data_b.range((i+1)*BIT-1, i*BIT);
				 tmp+= d_t;
				 data_t res = Norm_Quant(tmp, norm_buf[m*OutNum + i], out_shift);
				 data_s.range((i+1)*BIT-1, i*BIT) = res.range(BIT-1,0);
			 }
			 out_base[m] = data_s;
		}
	}


}


void ConvPE(ap_uint<InWidth> input[MAXIN/Tn],
            ap_uint<WtWidth> weight[MAXWT/Tn],
            ap_uint<OutWidth> output[MAXOT/OutNum],
            ap_uint<OutWidth> branch[MAXOT/OutNum],
            ap_uint<NormWidth> norm[MAXM],
            unsigned H, unsigned W, unsigned N, unsigned M,
            unsigned short K, unsigned short S, unsigned short P,
            ap_uint<16> out_shift) {
// BestOutstanding = RoundTripCycle(latency) / BrustLenth + 1
#pragma HLS INTERFACE m_axi port=norm  bundle=NORM depth=256 latency=64 max_read_burst_length=64 num_read_outstanding=2 num_write_outstanding=1
#pragma HLS INTERFACE m_axi port=branch  bundle=BR depth=50176 latency=64 max_read_burst_length=8 num_read_outstanding=32 num_write_outstanding=1
#pragma HLS INTERFACE m_axi port=output  bundle=DOUT depth=50176 latency=64 max_write_burst_length=8 num_read_outstanding=1 num_write_outstanding=32
#pragma HLS INTERFACE m_axi port=weight  bundle=WIN depth=73728 latency=64 max_read_burst_length=256 num_read_outstanding=2 num_write_outstanding=1
#pragma HLS INTERFACE m_axi port=input  bundle=DIN depth=50176 latency=64 max_read_burst_length=8 num_read_outstanding=32 num_write_outstanding=1

#pragma HLS INTERFACE s_axilite  bundle=CONFIG port=H
#pragma HLS INTERFACE s_axilite  bundle=CONFIG port=W
#pragma HLS INTERFACE s_axilite  bundle=CONFIG port=N
#pragma HLS INTERFACE s_axilite  bundle=CONFIG port=M
#pragma HLS INTERFACE s_axilite  bundle=CONFIG port=K
#pragma HLS INTERFACE s_axilite  bundle=CONFIG port=S
#pragma HLS INTERFACE s_axilite  bundle=CONFIG port=P
#pragma HLS INTERFACE s_axilite  bundle=CONFIG port=out_shift
#pragma HLS INTERFACE s_axilite  bundle=CONFIG port=return



    ap_int<AccBIT> out_buf_0[Tm][Tr][Tc];
#pragma HLS BIND_STORAGE variable=out_buf_0 type=ram_2p impl=bram
#pragma HLS ARRAY_PARTITION variable=out_buf_0 dim=1 complete
    ap_int<AccBIT> out_buf_1[Tm][Tr][Tc];
#pragma HLS BIND_STORAGE variable=out_buf_1 type=ram_2p impl=bram
#pragma HLS ARRAY_PARTITION variable=out_buf_1 dim=1 complete

    unsigned Hout = (H - K + 2*P)/S + 1;
    unsigned Wout = (W - K + 2*P)/S + 1;
 
    unsigned short loop_cnt = Hout/Tr * Wout/Tc * M/Tm;
    unsigned row = 0, col = 0, mm = 0;
    unsigned pre_row, pre_col, pre_mm;
    bool flag = true;

    PE_l:
    for (unsigned short l = 0; l < loop_cnt; l++) {
        if (l == 0) {
            Conv_Engine(input, weight, out_buf_0, H, W, N, M, K, S, P, row, col, mm);
        } else {
            if (flag) {
                Conv_Engine(input, weight, out_buf_1, H, W, N, M, K, S, P, row, col, mm);
                BA_BN_St(output, branch, norm, out_buf_0, Hout, Wout, M, pre_row, pre_col, pre_mm, out_shift);
                flag = !flag;
            } else {
                Conv_Engine(input, weight, out_buf_0, H, W, N, M, K, S, P, row, col, mm);
                BA_BN_St(output, branch, norm, out_buf_1, Hout, Wout, M, pre_row, pre_col, pre_mm, out_shift);
                flag = !flag;
            } 
        }
        // update loop para
        pre_row = row;
        pre_col = col;
        pre_mm = mm;
        if (mm == M - Tm) {
            mm = 0;
            if (col == Wout - Tc) {
                col = 0;
                row += Tr;
            }
            else {
                col += Tc;
            }
        } else {
            mm += Tm;
        }
        
    }
    if (flag) {
        BA_BN_St(output, branch, norm, out_buf_0, Hout, Wout, M, pre_row, pre_col, pre_mm, out_shift);
    } else {
        BA_BN_St(output, branch, norm, out_buf_1, Hout, Wout, M, pre_row, pre_col, pre_mm, out_shift);
    }
    

}
