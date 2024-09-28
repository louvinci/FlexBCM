#include"compute.h"


void compute(complexType ifm_buff[Tn][Smax*Tr+K-Smax][Smax*Tc+K-Smax],complexType wt_buff[Tm/2][Tn][K][K],
		     complexType2 ofm_buff[Tm][Tr][Tc],uint8 blk_size, uint2 Stride,uint4 Ksize){
#pragma HLS inline off
#pragma HLS ARRAY_PARTITION variable=wt_buff complete dim=1
#pragma HLS ARRAY_PARTITION variable=wt_buff complete dim=2
#pragma HLS ARRAY_PARTITION variable=ofm_buff complete dim=1
#pragma HLS ARRAY_PARTITION variable=ifm_buff complete dim=1

	uint4 bs_divisor = blk_size/2;
	uint4 mm,mb;
	//bool flag[Tm/2][Tn];
	bool flag[Tn];
	complexType1 o[Tm/2][Tn];
#pragma HLS ARRAY_PARTITION variable=o complete dim=0

	complexType1 o_r[Tm][Tn/2];
#pragma HLS ARRAY_PARTITION variable=o_r complete dim=0

	complexType2 out_t[Tm];
#pragma HLS ARRAY_PARTITION variable=out_t complete dim=0

	uint4 cnt = HBS/bs_divisor;
	unsigned char cur_n,cur_m,cur_mm, cur_nn;

	uint2 bb=0;

	Ckr:
	for(uint4 kr=0;kr<Ksize;kr++){
		Ckc:
		for(uint4 kc=0;kc<Ksize;kc++){
			Ctr:
			for(uint8 rr=0;rr<Tr;rr++){
				Ctc:
				for(uint8 cc=0;cc<Tc;cc++){
					Cbb:
					for(uint4 i=0; i < cnt; i++){

#pragma HLS PIPELINE II=1
#pragma HLS DEPENDENCE false inter variable=ofm_buff
#pragma HLS DEPENDENCE false intra variable=ofm_buff
#pragma HLS DEPENDENCE false intra variable=o

						bb = (uint2)i;
						for(uint8 tmm=0; tmm<Tm/HBS; tmm+=2){
#pragma HLS UNROLL
							for(uint8 tnn=0; tnn<Tn/HBS; tnn++){
#pragma HLS UNROLL
								for(uint4 b=0; b<HBS; b++){
#pragma HLS UNROLL
									cur_n = tnn*HBS+b;
									cur_m = bb*(Tm/HBS)+tmm;

									if(blk_size == 4){

										if(cur_n%2==0){
											flag[cur_n]=true;
										}else{
											flag[cur_n]=false;
										}

									}else if (blk_size == 8){

										if(cur_n%4==0){
											flag[cur_n]=true;
										}else{
											flag[cur_n]=false;
										}
									}else{

										if(cur_n%8==0){
											flag[cur_n]=true;
										}else{
											flag[cur_n]=false;
										}
									}
									//Loop 2: Compute core
									complexType x = ifm_buff[cur_n][Stride*rr+kr][Stride*cc+kc];
									complexType w = wt_buff[cur_m][cur_n][kr][kc];
									complexType w2= wt_buff[cur_m+1][cur_n][kr][kc];

									cmult_opt(w,w2,x,o[cur_m][cur_n],o[cur_m+1][cur_n],flag[cur_n]);
								}
							}
						}
						if(bb == cnt-1 ){
							//reshpae
							for(uint8 m=0;m<Tm;m++)
								for(uint8 nn=0;nn<Tn/2;nn++){
									if(blk_size==4){
										  mm=m/2;
										  mb=m%2;
										  if(nn<Tn/2)
											  o_r[mm*2+mb][nn]=o[mm][nn*2+mb];
										  else
											  o_r[mm*2+mb][nn]=(complexType)0;
									}
									else if(blk_size==8){
										mm=m/4;
										mb=m%4;
										if(nn<Tn/4)
											o_r[mm*4+mb][nn]=o[mm][nn*4+mb];
										else
											o_r[mm*4+mb][nn]=(complexType)0;
									}
									else{
										mm=m/8;
										mb=m%8;
										if(nn<Tn/8)
											o_r[mm*8+mb][nn]=o[mm][nn*8+mb];
										else
											o_r[mm*8+mb][nn]=(complexType)0;
									}
								}


							for(unsigned char tmm=0; tmm<Tm;tmm++){
#pragma HLS UNROLL
								complexType2 tmp = ofm_buff[tmm][rr][cc];
#if Tn==8
								out_t[tmm] = addtree_4(o_r[tmm][0],o_r[tmm][1],o_r[tmm][2],o_r[tmm][3]);
#elif Tn==16
								out_t[tmm] = addtree_8(o_r[tmm][0],o_r[tmm][1],o_r[tmm][2],o_r[tmm][3],
													  o_r[tmm][4],o_r[tmm][5],o_r[tmm][6],o_r[tmm][7]);
#elif Tn==32
								out_t[tmm] = addtree_16(o_r[tmm][0], o_r[tmm][1],  o_r[tmm][2],  o_r[tmm][3],
										  	  	  	   o_r[tmm][4],  o_r[tmm][5],  o_r[tmm][6],  o_r[tmm][7],
													   o_r[tmm][8],  o_r[tmm][9],  o_r[tmm][10], o_r[tmm][11],
													   o_r[tmm][12], o_r[tmm][13], o_r[tmm][14], o_r[tmm][15]);
#else
								out_t[tmm] = addtree_32(o_r[tmm][0], o_r[tmm][1],  o_r[tmm][2],   o_r[tmm][3],
													   o_r[tmm][4],  o_r[tmm][5],  o_r[tmm][6],   o_r[tmm][7],
													   o_r[tmm][8],  o_r[tmm][9],  o_r[tmm][10],  o_r[tmm][11],
													   o_r[tmm][12], o_r[tmm][13], o_r[tmm][14],  o_r[tmm][15],
													   o_r[tmm][16], o_r[tmm][17], o_r[tmm][18],  o_r[tmm][19],
													   o_r[tmm][20], o_r[tmm][21], o_r[tmm][22],  o_r[tmm][23],
													   o_r[tmm][24], o_r[tmm][25], o_r[tmm][26],  o_r[tmm][27],
													   o_r[tmm][28], o_r[tmm][29], o_r[tmm][30],  o_r[tmm][31]
														);
#endif

								ofm_buff[tmm][rr][cc] = tmp + out_t[tmm];

							}

						}
						bb+=1;
					}
				}
			}
			
		}
	}
    

}


