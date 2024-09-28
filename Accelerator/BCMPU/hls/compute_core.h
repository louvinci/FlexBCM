#pragma once
#include"data_transfer.h"

//void compute_core(complexType in_t[Tn],complexType out_t[Tm],complexType wt[Tm/2][Tn],uint16 blk_size,unsigned char bb);

void cmult_opt(complexType a,complexType b,complexType c,complexType2 & ac,complexType2 & bc,bool flag);
complexType2 addtree_4(complexType1 a, complexType1 b, complexType1 c, complexType1 d);

complexType2 addtree_8(complexType1 a,  complexType1 b,  complexType1 c,  complexType1 d,complexType1 a1, complexType1 b1, complexType1 c1, complexType1 d1);

complexType2 addtree_16(complexType1 a,  complexType1 b,  complexType1 c,  complexType1 d,
		               complexType1 a1, complexType1 b1, complexType1 c1, complexType1 d1,
					   complexType1 a2, complexType1 b2, complexType1 c2, complexType1 d2,
					   complexType1 a3, complexType1 b3, complexType1 c3, complexType1 d3);

complexType2 addtree_32(complexType1 a, complexType1 b,  complexType1 c,  complexType1 d,
		               complexType1 a1, complexType1 b1, complexType1 c1, complexType1 d1,
					   complexType1 a2, complexType1 b2, complexType1 c2, complexType1 d2,
					   complexType1 a3, complexType1 b3, complexType1 c3, complexType1 d3,
					   complexType1 a4, complexType1 b4, complexType1 c4, complexType1 d4,
					   complexType1 a5, complexType1 b5, complexType1 c5, complexType1 d5,
					   complexType1 a6, complexType1 b6, complexType1 c6, complexType1 d6,
					   complexType1 a7, complexType1 b7, complexType1 c7, complexType1 d7);

// complex multiplication optimization, and the special value processing
complexType1 cmult(complexType a,complexType b,bool flag);
template<class T>
T Reg(T in);



