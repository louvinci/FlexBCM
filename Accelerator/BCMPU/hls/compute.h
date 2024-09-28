#pragma once
#include"compute_core.h"
#define B_MIN (4/2)
#define B_MAX (16/2)

void compute(complexType ifm_buff[Tn][Smax*Tr+K-Smax][Smax*Tc+K-Smax],complexType wt_buff[Tm/2][Tn][K][K],
		     complexType2 ofm_buff[Tm][Tr][Tc],uint8 blk_size, uint2 Stride,uint4 Ksize);
