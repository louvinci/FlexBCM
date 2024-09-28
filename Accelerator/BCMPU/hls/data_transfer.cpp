#include"data_transfer.h"

int8 qout(Fix32 in ){
	int8 tmp;
	Fix32 in2 = in >> 4;
	if (in2 > 127){
		tmp = 127;
	}else if (in2<-127){
		tmp = -127;
	}else{
		tmp = in2;
	}
	return tmp;
}
void load_weight(ap_uint<CBit*WtWidth> *w,complexType wt_buff[Tm/2][Tn][K][K],
		unsigned short n,unsigned short m,unsigned short ch_in,uint8 blk_size, uint8 Ksize
	){
#pragma HLS inline off

	uint8 Tmloops = Tm *2 / blk_size;

	unsigned offset=Tmloops*Tn*Ksize*Ksize; // loops_cnt
	unsigned addr=0;
	ap_uint<CBit*WtWidth> *waddr = w+(m/Tm*(ch_in/Tn)+n/Tn)*offset;

//	LDW_1:
//	for(unsigned mm=0;mm<Tmloops;mm++)
//		LW_2:for(unsigned i=0;i<Ksize;i++)
//			LW_3:for(unsigned j=0;j<Ksize;j++)
//				for(unsigned nn=0;nn<Tn;nn++){
//#pragma HLS PIPELINE II=1
//					ap_uint<CBit> tmp;
//					int8 real, imag;
//					tmp = waddr[addr];
//					real.range(Bit-1,0)=tmp.range(Bit-1,0);
//					imag.range(Bit-1,0)=tmp.range(2*Bit-1,Bit);
//					wt_buff[mm][nn][i][j]=complexType(real,imag);
//					addr+=1;
//				}

	uint8 nn=0,mm=0,i=0,j=0;
	LDWT:
	for(unsigned cnt=0; cnt < offset;cnt++){
#pragma HLS PIPELINE II=1
		ap_uint<CBit> tmp;
		int8 real, imag;
		tmp = waddr[addr];

		real.range(Bit-1,0)=tmp.range(Bit-1,0);
		imag.range(Bit-1,0)=tmp.range(2*Bit-1,Bit);
		wt_buff[mm][nn][i][j]=complexType(real,imag);
		addr+=1;

		if(nn==Tn-1){
			nn=0;
			if(j==Ksize-1){
				j=0;
				if(i==Ksize-1){
					i = 0;
					if(mm== Tmloops-1){
						mm=0;
					}else{
						mm+=1;
					}
				}else{
					i+=1;
				}
			}else{
				j+=1;
			}
		}else{
			nn+=1;
		}


	}

}




void load_ifm16(ap_uint<Bit*ActWidth>* in,complexType ifm_buff[Tn][Smax*Tr+K-Smax][Smax*Tc+K-Smax],uint16 r,uint16 c,uint16 n,
		      uint16 fsize, uint16 ch_in, uint8 blk_size,uint2 Stride,uint8 Ksize){
	//stride =2, Ksize must be 3
	//Ksize=1, the stride =1,pad=0 when overlooking the downsample layer.
	uint8 rloops,cloops;
	if(Stride == 1){
		rloops = Tr+Ksize-1;
		cloops = Tc+Ksize-1;
	}else{
		rloops = (Tr<<1) + 1;
		cloops = (Tc<<1) + 1;
	}

	uint8 pad = (Ksize == 3) ?1:0;
	short t1 =  Stride*r - pad;
	short t2 =  Stride*c - pad;

	char t3 = r-pad;//this value is not related to pad, just using the pad's value
	char t4 = c-pad;

	int8 tmp[FFT_LEN];
	complexTypeF32 fctmp[FFT_LEN];
	complexType    ctmp[FFT_LEN];

	unsigned offset1 = ch_in>>3;
	unsigned offset2 = fsize*offset1;


	LD16_1:for(uint8 rr=0; rr<rloops; rr++){
		 LD_2:for(uint8 cc=0; cc<cloops; cc++){
			 LD_3:for(uint8 nn=0;nn<Tn/8;nn++){ // +=8 in complex input
#pragma HLS DEPENDENCE variable=ifm_buff array inter false
#pragma HLS PIPELINE II=1
				ap_uint<Bit*ActWidth>* addr = in + (t3+rr)*offset2 + (t4+cc)*offset1 + (n>>3);
				if( rr+t1>=0 && rr+t1<fsize && cc+t2 >= 0 && cc+t2 < fsize)
				{

					ap_uint<Bit*ActWidth> x= addr[nn];
					for(int k=0;k<FFT_LEN;k++){
						tmp[k].range(Bit-1,0)=x.range(k*Bit+Bit-1,k*Bit);
					}


					rfft_opt(tmp,fctmp,blk_size);

					for(unsigned char ii=0;ii<FFT_LEN;ii++){
						int8 qr,qi;
						qr = qout(fctmp[ii].real());
						qi = qout(fctmp[ii].imag());

						ctmp[ii] = complexType(qr,qi);

					}

					if(blk_size == 4){
						ifm_buff[nn<<3][rr][cc].real(ctmp[0].real());
						ifm_buff[(nn<<3)][rr][cc].imag(ctmp[2].real());
						ifm_buff[(nn<<3)+1][rr][cc]=ctmp[1];

						ifm_buff[(nn<<3)+2][rr][cc].real(ctmp[4].real());
						ifm_buff[(nn<<3)+2][rr][cc].imag(ctmp[6].real());
						ifm_buff[(nn<<3)+3][rr][cc]=ctmp[5];

						ifm_buff[(nn<<3)+4][rr][cc].real(ctmp[8].real());
						ifm_buff[(nn<<3)+4][rr][cc].imag(ctmp[10].real());
						ifm_buff[(nn<<3)+5][rr][cc]=ctmp[9];

						ifm_buff[(nn<<3)+6][rr][cc].real(ctmp[12].real());
						ifm_buff[(nn<<3)+6][rr][cc].imag(ctmp[14].real());
						ifm_buff[(nn<<3)+7][rr][cc]=ctmp[13];
					}else if(blk_size == 8){
						ifm_buff[(nn<<3)][rr][cc].real(ctmp[0].real());
						ifm_buff[(nn<<3)][rr][cc].imag(ctmp[4].real());
						ifm_buff[(nn<<3)+1][rr][cc]=ctmp[1];
						ifm_buff[(nn<<3)+2][rr][cc]=ctmp[2];
						ifm_buff[(nn<<3)+3][rr][cc]=ctmp[3];

						ifm_buff[(nn<<3)+4][rr][cc].real(ctmp[8].real());
						ifm_buff[(nn<<3)+4][rr][cc].imag(ctmp[12].real());
						ifm_buff[(nn<<3)+5][rr][cc]=ctmp[9];
						ifm_buff[(nn<<3)+6][rr][cc]=ctmp[10];
						ifm_buff[(nn<<3)+7][rr][cc]=ctmp[11];

					}else{

						ifm_buff[(nn<<3)][rr][cc].real(ctmp[0].real());
						ifm_buff[(nn<<3)][rr][cc].imag(ctmp[8].real());
						ifm_buff[(nn<<3)+1][rr][cc]=ctmp[1];
						ifm_buff[(nn<<3)+2][rr][cc]=ctmp[2];
						ifm_buff[(nn<<3)+3][rr][cc]=ctmp[3];
						ifm_buff[(nn<<3)+4][rr][cc]=ctmp[4];
						ifm_buff[(nn<<3)+5][rr][cc]=ctmp[5];
						ifm_buff[(nn<<3)+6][rr][cc]=ctmp[6];
						ifm_buff[(nn<<3)+7][rr][cc]=ctmp[7];

					}

				}
				else
				{
					for(int k=0;k<FFT_LEN/2;k++)
						ifm_buff[(nn<<3)+k][rr][cc]=(complexType)0;
				}
			}
		}

	}


}



void store_ofm16(ap_uint<Bit*ActWidth>* out,complexType2 ofm_buff[Tm][Tr][Tc],uint16 m,uint16 r,uint16 c,
		       uint16 fsize,uint16 ch_out,uint8 blk_size){
     //store out[m:m+Tm][d:d+Td][r:r+Tr][c:c+Tc]
     //int max_tm=((ch_out-m)<Tm)?ch_out-m:Tm;
	//complexType2   ctmp[FFT_LEN/2];
	complexTypeF32 ctmp2[FFT_LEN/2];
	Fix32 tmp[FFT_LEN];


	unsigned offset1 = ch_out>>3;
	unsigned offset2 = fsize*offset1;

	STFM_16:for(uint8 rr=0;rr<Tr;rr++){
		ST16_2:for(uint8 cc=0;cc<Tc;cc++){
			ST16_3:for(uint8 mm=0;mm<Tm/8;mm++){
#pragma HLS PIPELINE II=1
				ap_uint<Bit*ActWidth>* addr = out + (r+rr)*offset2 + (c+cc)*offset1 + (m>>3);
				for(uint8 b=0;b<FFT_LEN/2;b++){
					// we use complex<fix32> to keep the value due to the fft operation
					// the type will be transformed automatically,
					// but the value is same when not overflow.
					ctmp2[b]=ofm_buff[mm*8+b][rr][cc];
				}

				irfft_opt(ctmp2,tmp,blk_size);
				ap_uint<Bit*ActWidth> data;
				for(uint8 b=0;b<FFT_LEN;b++){
					int8 tmp_out;
					tmp_out = qout(tmp[b]);
					data.range(b*Bit+Bit-1,b*Bit)=tmp_out.range(Bit-1,0);
				}
				
				addr[mm]=data;

			}
		}
		
	}
	 
}





