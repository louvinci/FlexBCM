#include"compute.h"

//extern "C"{
//void circonv(ap_uint<64>* in1,ap_uint<128>* in2,ap_uint<256>* in3,ap_uint<32>* w,
//		      ap_uint<32>* bias,ap_uint<64>* out1,ap_uint<128>* out2,ap_uint<256>* out3,
//		      int ch_in,int ch_out,int fsize,int blk_size);
//}


extern "C"{
void circonv( ap_uint<Bit*ActWidth> in3[MAXIN/ActWidth], ap_uint<CBit*WtWidth> complex_w[MAXWT/2],
		      ap_uint<AccBit*2> bias[512/2],    ap_uint<Bit*ActWidth> out3[MAXOT/ActWidth],
			  unsigned ch_in,unsigned ch_out,unsigned fsize,unsigned Ksize,unsigned Stride,
			  			  unsigned blk_size);
}
