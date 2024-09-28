#pragma once

#include"vb_fft.h"
#include"config.h"


void load_ifm16(ap_uint<Bit*ActWidth>* in,complexType ifm_buff[Tn][Smax*Tr+K-Smax][Smax*Tc+K-Smax],uint16 r,uint16 c,uint16 n,
		      uint16 fsize, uint16 ch_in, uint8 blk_size, uint2 Stride,uint8 Ksize);

void store_ofm16(ap_uint<Bit*ActWidth>* out,complexType2 ofm_buff[Tm][Tr][Tc],uint16 m,uint16 r,uint16 c,
		       uint16 fsize,uint16 ch_out,uint8 blk_size);

void load_weight(ap_uint<CBit*WtWidth> *w,complexType wt_buff[Tm/2][Tn][K][K],
		unsigned short n,unsigned short m,unsigned short ch_in,uint8 blk_size, uint8 Ksize);

int8 qout(Fix32 in );
