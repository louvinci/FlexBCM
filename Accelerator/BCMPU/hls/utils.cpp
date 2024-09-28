#include"utils.h"
#include"data_transfer.h"


void random_init(data_t* x,int len){
    for(int i=0;i<len;i++){
          x[i]=(data_t)((rand()%255-127));
    }
}

void crandom_init(complexType* x,int len){
	for(int i=0;i<len;i++){
		x[i].real((rand()%255-127));
		x[i].imag((rand()%255-127));
	}
}
void crandom_init2(complexType2* x,int len){
	for(int i=0;i<len;i++){
		x[i].real((rand()%1024-512));
		x[i].imag((rand()%1024-512));
	}
}
void rcir_conv2d(complexType* in,complexType* weight,complexType2* bias,complexType2* out,int n,int m,int h,int w,int k,int s,int p,int block_size){
     //O[m][d][r][c]
     //In[n][t][h][w],Weight[m/block_size][n/block_size][k][k][k][block_size]

     int h_o=(h+2*p-k)/s+1;
     int w_o=(w+2*p-k)/s+1;
     
     //
     
	for(int i=0;i<h_o;i++)
		for(int j=0;j<w_o;j++)
			for(int ch_o=0;ch_o<m/block_size;ch_o++){

				complexType2 *tmp=new complexType2[block_size];
				for(int b=0;b<block_size;b++){
						tmp[b]=bias[ch_o*block_size+b];
				}

				//for(int dd=0;dd<k;dd++)
					for(int rr=0;rr<k;rr++)
						for(int cc=0;cc<k;cc++)
							for(int ch_i=0;ch_i<n/block_size;ch_i++){
								//in[ch_i*block_size:ch_i*block_size+block_size][d*s+dd-p][i*s+rr-p][j*s+cc-p]
								if((i*s+rr)>=p&&(i*s+rr)<(h+p)&&(j*s+cc)>=p&&(j*s+cc)<(w+p)){
									for(int b=0;b<block_size;b++){
										complexType xs=in[(ch_i*block_size+b)*h*w+(i*s+rr-p)*w+(j*s+cc-p)];
										complexType ws=weight[ch_o*n*k*k+ch_i*k*k*block_size+rr*k*block_size+cc*block_size+b];
										if(b==0){
											complexType2 ctmp;
											ctmp.real((int32)xs.real()*(int32)ws.real());
											ctmp.imag((int32)xs.imag()*(int32)ws.imag());
											tmp[b]+=ctmp;
										}
										else{
											//tmp[b]+=xs*ws;
											complexType2 ctmp;
											ctmp.real((int32)xs.real()*(int32)ws.real()-(int32)xs.imag()*(int32)ws.imag());
											ctmp.imag((int32)xs.real()*(int32)ws.imag()+(int32)xs.imag()*(int32)ws.real());
											tmp[b]+=ctmp;

										}
									}
								}
							}
				//out[ch_o*block_size:ch_o*block_size+block_size][d][i][j]
				for(int b=0;b<block_size;b++)
					out[(ch_o*block_size+b)*h_o*w_o+i*w_o+j]= tmp[b];
     }
}

void interleave(data_t* in,data_t* out,int ch,int h,int w,int b){
	//in[ch][d][h][w]-->out[ch/BL][d][h][w][BL]
	for(int n=0;n<ch/b;n++)
		for(int i=0;i<h*w;i++)
			for(int k=0;k<b;k++){
				out[n*(h*w*b)+i*b+k]=in[(n*b+k)*h*w+i];
			}
}

void TransHWC(data_t* in,data_t* out,int ch,int h,int w){
	//in[ch][h][w]-->out[h][w][ch]

	for(int i=0;i<h*w;i++){
		for(int n=0;n<ch;n++){
			//out[n*(h*w*b)+i*b+k]=in[(n*b+k)*h*w+i];
			out[i*ch+n] = in[n*h*w+i];
		}
	}
}

void deinterleave(data_t* in,data_t* out,int ch,int h,int w,int b){
	//in[ch/BL][h][w][BL]-->out[ch][h][w]
	for(int n=0;n<ch;n++)
		for(int i=0;i<h*w;i++){
			out[n*h*w+i]=in[(n/b)*h*w*b+i*b+n%b];
		}
}

void FFT(complexTypeF32* Xin,complexTypeF32 *Xout,int n){
    if(n<2)
         Xout[0]=Xin[0];
    else
    {
    	complexTypeF32* X1    =new complexTypeF32[n/2];
    	complexTypeF32* X2    =new complexTypeF32[n/2];
    	complexTypeF32* X1_out=new complexTypeF32[n/2];
    	complexTypeF32* X2_out=new complexTypeF32[n/2];
         for(int i=0;i<n;i+=2)
         {
                X1[i/2]=Xin[i];
                X2[i/2]=Xin[i+1];
         }
         FFT(X1,X1_out,n/2);
         FFT(X2,X2_out,n/2);
         complexTypeF32* W=new complexTypeF32[n/2];
         for(int i=0;i<n/2;i++){
            W[i].real(  cos(2*PI*i/n) );
            W[i].imag( -sin(2*PI*i/n) );
         }
         for(int i=0;i<n/2;i++){
                Xout[i]=X1_out[i]+W[i]*X2_out[i];
                Xout[i+n/2]=X1_out[i]-W[i]*X2_out[i];
         }
        delete []X1;
        delete []X2;
        delete []X1_out;
        delete []X2_out;
    }
    return;
}

void RFFT(data_t* in,complexTypeF32* out,int n){
	complexTypeF32 *inc=new complexTypeF32[n];
	for(int k=0;k<n;k++){
		inc[k].real(in[k]);
		inc[k].imag(0);
	}
	FFT(inc,out,n);
}

void IFFT(complexTypeF32* in,Fix32* out,int n){
	complexTypeF32* x=new complexTypeF32[n];
	complexTypeF32* y=new complexTypeF32[n];
	for(int i=0;i<n;i++){
		x[i].real(in[i].real());
		x[i].imag(-in[i].imag());
	}
	FFT(x,y,n);
    for(int i=0;i<n;i++){
    	out[i]=y[i].real()/n;
    }
    delete [] x;
    delete [] y;
}

void IRFFT(complexTypeF32* in, Fix32* out,int n){
	complexTypeF32 *tmp=new complexTypeF32[n];
	for(int i=0;i<=n/2;i++){
		if(i==0){
			tmp[i].real(in[0].real());
			tmp[i].imag(0);
		}
		else if(i==n/2){
			tmp[i].real(in[0].imag());
			tmp[i].imag(0);
		}
		else{
			tmp[i]=in[i];
			tmp[n-i].real(in[i].real());
			tmp[n-i].imag(-in[i].imag());
		}
	}
	IFFT(tmp,out,n);
	delete [] tmp;
}

void feature_trans(data_t *in,complexType *out,int n,int h,int w,int blk){
	//in[n][d][h][w]-->out[n/2][d][h][w]

	complexTypeF32 *ctmp1=new complexTypeF32[blk];
	complexTypeF32 *ctmp2=new complexTypeF32[blk];
	complexType    *ctmp3=new complexType[blk];
	//
	for(int i=0;i<w*h;i++){
		for(int j=0;j<n/blk;j++){

			for(int b=0;b<blk;b++){
				int8 tt=in[(j*blk+b)*h*w+i];
				ctmp1[b].real(tt);
				ctmp1[b].imag(0);
			}
			FFT(ctmp1,ctmp2,blk);
			//here simulate the quant operation
			for(int ii=0;ii<blk;ii++){
				int8 qr,qi;
				qr = qout(ctmp2[ii].real());
				qi = qout(ctmp2[ii].imag());
				ctmp3[ii] = complexType(qr,qi);

//				if(i==0){
//					cout<<"sw: "<<ctmp1[ii]<<" "<<ctmp3[ii]<<endl;
//				}
			}

			for(int b=0;b<blk/2;b++){
				if(b==0){                     //0
					out[(j*blk/2+b)*h*w+i].real(ctmp3[0].real());
					out[(j*blk/2+b)*h*w+i].imag(ctmp3[blk/2].real());
				}
				else{                        //1,2,...,BL/2-1
					out[(j*blk/2+b)*h*w+i].real(ctmp3[b].real());
					out[(j*blk/2+b)*h*w+i].imag(ctmp3[b].imag());
				}
			}
        }
	}
}

void feature_itrans(complexType2* in,data_t* out,int n,int w,int h,int blk){
	//in[n/2][d][h][w]-->out[n][d][h][w]

	complexTypeF32 *ctmp=new complexTypeF32[blk/2];
	Fix32* tmp  =new Fix32[blk];
	//
	for(int i=0;i<h*w;i++)
		for(int j=0;j<n/blk;j++)
		{
			for(int k=0;k<blk/2;k++){
				complexType2 tt = in[(j*blk/2+k)*h*w+i];
				int32 qr,qi;
				qr = tt.real();
				qi = tt.imag();
				ctmp[k]= complexTypeF32((Fix32)qr,(Fix32)qi);
			}
			IRFFT(ctmp,tmp,blk);

			for(int k=0;k<blk;k++){
				out[i*n+j*blk+k]=qout(tmp[k]);
			}
		}
}

//weight_reorg(weightss,tmp,N/2,M/2,Tm,Tn,K,BL/2);// symmetry
void weight_reorg(complexType* in,complexType* out,
		int n,int m,int tm,int tn,int k,int block_size){
	//(m/b,n/b,k,k,,b)-->(m/tm)*(n/tn)*(tm/b,tn/b,k,k,b)
	int blk=tm/block_size*tn/block_size*k*k*block_size;
	int nkkk=n*k*k;
	int kkkb=k*k*block_size;
	int kkb=k*block_size;
	int kb=block_size;
	//
    for(int mm=0;mm<m/tm;mm++)
    for(int nn=0;nn<n/tn;nn++){
    //
    for(int i=0;i<tm/block_size;i++)
   	for(int j=0;j<tn/block_size;j++)
    for(int kr=0;kr<k;kr++)
    for(int kc=0;kc<k;kc++)
    for(int b=0;b<block_size;b++){
        //out(mm,nn,i,j,kd,kr,kc,b)  //in(mm*m/tm+i,nn*n/tn+j,kd,kr,kc,b)
        out[mm*n/tn*blk+nn*blk+i*tn/block_size*kkkb+j*kkkb+kr*kkb+kc*block_size+b]
	    =in[(mm*tm/block_size+i)*nkkk+(nn*tn/block_size+j)*kkkb+kr*kkb+kc*block_size+b];
    }
    }
}

void rego(complexType* in,complexType* out,int n,int m,int tm,int tn,int k,int block_size){
	//(m/tm)脳(n/tn)脳(tm/b,tn/b,kkk,b)-->(m/tm)脳(n/tn)脳(tm/b)脳kkk脳tn
	for(int i=0;i<m/tm*n/tn;i++){
	for(int j=0;j<tm/block_size;j++)
	for(int r=0;r<k*k;r++)
	for(int s=0;s<tn;s++){
       *(out+i*tm*tn*k*k/block_size+j*k*k*tn+r*tn+s)=
       in[i*tm*tn*k*k/block_size+j*tn*k*k+(s/block_size)*k*k*block_size+r*block_size+s%block_size];
	}
	}
}





