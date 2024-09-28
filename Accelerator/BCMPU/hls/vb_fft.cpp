#include "vb_fft.h"
//bit reverse for two 8-point fft
void bit_reverse2_8(complexTypeF32* Xin,complexTypeF32* Xout){
   //the first 8-point fft
   Xout[0]=Xin[0];
   Xout[1]=Xin[4];
   Xout[2]=Xin[2];
   Xout[3]=Xin[6];
   Xout[4]=Xin[1];
   Xout[5]=Xin[5];
   Xout[6]=Xin[3];
   Xout[7]=Xin[7];
   //the second 8-point fft
   Xout[8]=Xin[8];
   Xout[9]=Xin[12];
   Xout[10]=Xin[10];
   Xout[11]=Xin[14];
   Xout[12]=Xin[9];
   Xout[13]=Xin[13];
   Xout[14]=Xin[11];
   Xout[15]=Xin[15];

}
//bit reverse for four 4-point fft
void bit_reverse4_4(complexTypeF32* Xin,complexTypeF32* Xout){
   //the first 4-point fft
   Xout[0]=Xin[0];
   Xout[1]=Xin[2];
   Xout[2]=Xin[1];
   Xout[3]=Xin[3];
   //the second
   Xout[4]=Xin[4];
   Xout[5]=Xin[6];
   Xout[6]=Xin[5];
   Xout[7]=Xin[7];
   //the third
   Xout[8]=Xin[8];
   Xout[9]=Xin[10];
   Xout[10]=Xin[9];
   Xout[11]=Xin[11];
   //the fourth
   Xout[12]=Xin[12];
   Xout[13]=Xin[14];
   Xout[14]=Xin[13];
   Xout[15]=Xin[15];

}
//bit reverse for 16-point fft
void bit_reverse1_16(complexTypeF32* Xin,complexTypeF32* Xout){
   Xout[0]=Xin[0];
   Xout[1]=Xin[8];
   Xout[2]=Xin[4];
   Xout[3]=Xin[12];
   //
   Xout[4]=Xin[2];
   Xout[5]=Xin[10];
   Xout[6]=Xin[6];
   Xout[7]=Xin[14];
   //
   Xout[8]=Xin[1];
   Xout[9]=Xin[9];
   Xout[10]=Xin[5];
   Xout[11]=Xin[13];
   //
   Xout[12]=Xin[3];
   Xout[13]=Xin[11];
   Xout[14]=Xin[7];
   Xout[15]=Xin[15];
}
//16-point fft stage1
void stage1(complexTypeF32 x[FFT_LEN],complexTypeF32 y[FFT_LEN]){
    y[0]=x[0]+x[1];
    y[1]=x[0]-x[1];
    //
    y[2]=x[2]+x[3];
    y[3]=x[2]-x[3];
    //
    y[4]=x[4]+x[5];
    y[5]=x[4]-x[5];
    //
    y[6]=x[6]+x[7];
    y[7]=x[6]-x[7];
    //
    y[8]=x[8]+x[9];
    y[9]=x[8]-x[9];
    //
    y[10]=x[10]+x[11];
    y[11]=x[10]-x[11];
    //
    y[12]=x[12]+x[13];
    y[13]=x[12]-x[13];
    //
    y[14]=x[14]+x[15];
    y[15]=x[14]-x[15];
}
//16-point fft stage2
void stage2(complexTypeF32 x[FFT_LEN],complexTypeF32 y[FFT_LEN]){
//    complexTypeF32 W4;
//    W4.real(0.0);
//    W4.imag(-1.0);
    complexTypeF32 tmp[4];
#pragma HLS ARRAY_PARTITION variable=tmp complete dim=1

    // tmp[0]=W4*x[3];
    // tmp[1]=W4*x[7];
    // tmp[2]=W4*x[11];
    // tmp[3]=W4*x[15];
    tmp[0]=complexTypeF32(x[3].imag(),-x[3].real());
    tmp[1]=complexTypeF32(x[7].imag(),-x[7].real());
    tmp[2]=complexTypeF32(x[11].imag(),-x[11].real());
    tmp[3]=complexTypeF32(x[15].imag(),-x[15].real());
    //
    y[0]=x[0]+x[2];
    y[2]=x[0]-x[2];
    //
    y[1]=x[1]+tmp[0];
    y[3]=x[1]-tmp[0];
    //
    y[4]=x[4]+x[6];
    y[6]=x[4]-x[6];
    //
    y[5]=x[5]+tmp[1];
    y[7]=x[5]-tmp[1];
    //
    y[8]=x[8]+x[10];
    y[10]=x[8]-x[10];
    //
    y[9]=x[9]+tmp[2];
    y[11]=x[9]-tmp[2];
    //
    y[12]=x[12]+x[14];
    y[14]=x[12]-x[14];
    //
    y[13]=x[13]+tmp[3];
    y[15]=x[13]-tmp[3];
}
//16-point fft stage3
void stage3(complexTypeF32 x[FFT_LEN],complexTypeF32 y[FFT_LEN]){
    // complexTypeF32 W2,W4,W6;
    // W2.real(cos(2*PI*2/FFT_LEN));
    // W2.imag(-sin(2*PI*2/FFT_LEN));
    // W4.real(cos(2*PI*4/FFT_LEN));
    // W4.imag(-sin(2*PI*4/FFT_LEN));
    // W6.real(cos(2*PI*6/FFT_LEN));
    // W6.imag(-sin(2*PI*6/FFT_LEN));
    complexTypeF32 tmp[6];
#pragma HLS ARRAY_PARTITION variable=tmp complete dim=1

    // tmp[0]=W2*x[5];
    // tmp[1]=W4*x[6];
    // tmp[2]=W6*x[7];
    // tmp[3]=W2*x[13];
    // tmp[4]=W4*x[14];
    // tmp[5]=W6*x[15];
    tmp[0]=W[2]*x[5];
    tmp[1]=complexTypeF32(x[6].imag(),-x[6].real());
    tmp[2]=W[6]*x[7];
    tmp[3]=W[2]*x[13];
    tmp[4]=complexTypeF32(x[14].imag(),-x[14].real());
    tmp[5]=W[6]*x[15];
    //
    y[0]=x[0]+x[4];
    y[4]=x[0]-x[4];
    //
    y[1]=x[1]+tmp[0];
    y[5]=x[1]-tmp[0];
    //
    y[2]=x[2]+tmp[1];
    y[6]=x[2]-tmp[1];
    //
    y[3]=x[3]+tmp[2];
    y[7]=x[3]-tmp[2];
    //
    y[8]=x[8]+x[12];
    y[12]=x[8]-x[12];
    //
    y[9]=x[9]+tmp[3];
    y[13]=x[9]-tmp[3];
    //
    y[10]=x[10]+tmp[4];
    y[14]=x[10]-tmp[4];
    //
    y[11]=x[11]+tmp[5];
    y[15]=x[11]-tmp[5];
}
//16-point fft stage4
void stage4(complexTypeF32 x[FFT_LEN],complexTypeF32 y[FFT_LEN]){
    // complexTypeF32 W1,W2,W3,W4,W5,W6,W7;
    // W1.real(cos(2*PI*1/FFT_LEN));
    // W1.imag(-sin(2*PI*1/FFT_LEN));
    // W2.real(cos(2*PI*2/FFT_LEN));
    // W2.imag(-sin(2*PI*2/FFT_LEN));
    // W3.real(cos(2*PI*3/FFT_LEN));
    // W3.imag(-sin(2*PI*3/FFT_LEN));
    // W4.real(cos(2*PI*4/FFT_LEN));
    // W4.imag(-sin(2*PI*4/FFT_LEN));
    // W5.real(cos(2*PI*5/FFT_LEN));
    // W5.imag(-sin(2*PI*5/FFT_LEN));
    // W6.real(cos(2*PI*6/FFT_LEN));
    // W6.imag(-sin(2*PI*6/FFT_LEN));
    // W7.real(cos(2*PI*7/FFT_LEN));
    // W7.imag(-sin(2*PI*7/FFT_LEN));
    complexTypeF32 tmp[7];
#pragma HLS ARRAY_PARTITION variable=tmp complete dim=1
    // tmp[0]=W1*x[9];
    // tmp[1]=W2*x[10];
    // tmp[2]=W3*x[11];
    // tmp[3]=W4*x[12];
    // tmp[4]=W5*x[13];
    // tmp[5]=W6*x[14];
    // tmp[6]=W7*x[15];
    tmp[0]=W[1]*x[9];
    tmp[1]=W[2]*x[10];
    tmp[2]=W[3]*x[11];
    tmp[3]=complexTypeF32(x[12].imag(),-x[12].real());
    tmp[4]=W[5]*x[13];
    tmp[5]=W[6]*x[14];
    tmp[6]=W[7]*x[15];
    //
    y[0]=x[0]+x[8];
    y[8]=x[0]-x[8];
    //
    y[1]=x[1]+tmp[0];
    y[9]=x[1]-tmp[0];
    //
    y[2]=x[2]+tmp[1];
    y[10]=x[2]-tmp[1];
    //
    y[3]=x[3]+tmp[2];
    y[11]=x[3]-tmp[2];
    //
    y[4]=x[4]+tmp[3];
    y[12]=x[4]-tmp[3];
    //
    y[5]=x[5]+tmp[4];
    y[13]=x[5]-tmp[4];
    //
    y[6]=x[6]+tmp[5];
    y[14]=x[6]-tmp[5];
    //
    y[7]=x[7]+tmp[6];
    y[15]=x[7]-tmp[6];
}



//fft unit support 4-8-16 points
void fft_opt(complexTypeF32 x[FFT_LEN],complexTypeF32 X[FFT_LEN],unsigned char n){

#pragma HLS ARRAY_PARTITION variable=X complete dim=1
#pragma HLS ARRAY_PARTITION variable=x complete dim=1
    complexTypeF32 Xtmp[STAGE+1][FFT_LEN];
#pragma HLS ARRAY_PARTITION variable=Xtmp complete dim=0

    
    if(n==4){
        bit_reverse4_4(x,Xtmp[0]);
    }else if (n==8){
        bit_reverse2_8(x,Xtmp[0]);
    }else{
        bit_reverse1_16(x,Xtmp[0]);
    }

    stage1(Xtmp[0],Xtmp[1]);
    stage2(Xtmp[1],Xtmp[2]);

    if(n==4){
        for(int i=0;i<FFT_LEN;i++){
            X[i]=Xtmp[2][i];
            //cout<<"i="<<i<<","<<Xtmp[2][i]<<endl;
        }
        return;
    }
    stage3(Xtmp[2],Xtmp[3]);
    if(n==8){
        for(int i=0;i<FFT_LEN;i++){
            X[i]=Xtmp[3][i];
            //cout<<"i="<<i<<","<<Xtmp[3][i]<<endl;
        }
        return;
    }
    stage4(Xtmp[3],Xtmp[4]);
    // if(n==16){
    //     for(int i=0;i<16;i++){
    //         X[i]=Xtmp[4][i];
    //         //cout<<"i="<<i<<","<<Xtmp[4][i]<<endl;
    //     }
    //     return;
    // }
    //
    for(int i=0;i<FFT_LEN;i++){
        X[i]=Xtmp[STAGE][i];
    }
}

void rfft_opt(int8 x[FFT_LEN],complexTypeF32 X[FFT_LEN], unsigned char blk_size){

#pragma HLS ARRAY_PARTITION variable=x complete dim=1
#pragma HLS ARRAY_PARTITION variable=X complete dim=1
#pragma HLS PIPELINE II=1

    // pad the real input to complex data
    complexTypeF32 tmp[FFT_LEN];
#pragma HLS ARRAY_PARTITION variable=tmp complete dim=1
	for(int i=0;i<FFT_LEN;i++){  // set n to FFT_LEN, or 4/8/16, because the loop will be unroll
		// set value not range() copy
		tmp[i]=complexTypeF32(x[i],0);

	}

    fft_opt(tmp, X, blk_size);
}

void irfft_opt(complexTypeF32 Xin[FFT_LEN/2], Fix32 Xout[FFT_LEN], unsigned char blk_size){
#pragma HLS ARRAY_PARTITION variable=Xin complete dim=1
#pragma HLS ARRAY_PARTITION variable=Xout complete dim=1
#pragma HLS PIPELINE II=1

    complexTypeF32 tmp1[FFT_LEN];
    complexTypeF32 tmp2[FFT_LEN];
    #pragma HLS ARRAY_PARTITION variable=tmp1 complete dim=1
    #pragma HLS ARRAY_PARTITION variable=tmp2 complete dim=1

    //rfft-4
    if (blk_size == 4){
    	//fft-4-1
    	tmp1[0]=complexTypeF32(Xin[0].real(), 0);
        tmp1[2]=complexTypeF32(Xin[0].imag(), 0);
        tmp1[1]=complexTypeF32(Xin[1].real(),-Xin[1].imag());
        tmp1[3]=complexTypeF32(Xin[1].real(), Xin[1].imag());
        //fft-4-2
        tmp1[4]=complexTypeF32(Xin[2].real(), 0);
		tmp1[6]=complexTypeF32(Xin[2].imag(), 0);
		tmp1[5]=complexTypeF32(Xin[3].real(),-Xin[3].imag());
		tmp1[7]=complexTypeF32(Xin[3].real(), Xin[3].imag());
		//fft-4-3
		tmp1[8] =complexTypeF32(Xin[4].real(), 0);
		tmp1[10]=complexTypeF32(Xin[4].imag(), 0);
		tmp1[9] =complexTypeF32(Xin[5].real(),-Xin[5].imag());
		tmp1[11]=complexTypeF32(Xin[5].real(), Xin[5].imag());
		//fft-4-4
		tmp1[12]=complexTypeF32(Xin[6].real(), 0);
		tmp1[14]=complexTypeF32(Xin[6].imag(), 0);
		tmp1[13]=complexTypeF32(Xin[7].real(),-Xin[7].imag());
		tmp1[15]=complexTypeF32(Xin[7].real(), Xin[7].imag());

    } else if (blk_size == 8){
    	//fft-8-1
    	tmp1[0]=complexTypeF32(Xin[0].real(), 0);
        tmp1[4]=complexTypeF32(Xin[0].imag(), 0);

        tmp1[1]=complexTypeF32(Xin[1].real(),-Xin[1].imag());
        tmp1[7]=complexTypeF32(Xin[1].real(), Xin[1].imag());

        tmp1[2]=complexTypeF32(Xin[2].real(),-Xin[2].imag());
        tmp1[6]=complexTypeF32(Xin[2].real(), Xin[2].imag());

        tmp1[3]=complexTypeF32(Xin[3].real(),-Xin[3].imag());
        tmp1[5]=complexTypeF32(Xin[3].real(), Xin[3].imag());
        //fft-8-2
        tmp1[8]=complexTypeF32( Xin[4].real(), 0);
		tmp1[12]=complexTypeF32(Xin[4].imag(), 0);

		tmp1[9]=complexTypeF32( Xin[5].real(),-Xin[5].imag());
		tmp1[15]=complexTypeF32(Xin[5].real(), Xin[5].imag());

		tmp1[10]=complexTypeF32(Xin[6].real(),-Xin[6].imag());
		tmp1[14]=complexTypeF32(Xin[6].real(), Xin[6].imag());

		tmp1[11]=complexTypeF32(Xin[7].real(),-Xin[7].imag());
		tmp1[13]=complexTypeF32(Xin[7].real(), Xin[7].imag());

    }else{
    	tmp1[0]=complexTypeF32(Xin[0].real(), 0);
        tmp1[8]=complexTypeF32(Xin[0].imag(), 0);
        for(int i=1;i<8;i++){
            tmp1[i]=complexTypeF32(  Xin[i].real(),-Xin[i].imag());
            tmp1[16-i]=complexTypeF32(Xin[i].real(),Xin[i].imag());
        }
    }
    fft_opt(tmp1,tmp2,blk_size);
    
    for(unsigned i=0;i<FFT_LEN;i++){
    	if (blk_size == 4){
    		Xout[i]=tmp2[i].real()>>2;
    	} else if(blk_size == 8){
    		Xout[i]=tmp2[i].real()>>3;
    	}else{
    		Xout[i]=tmp2[i].real()>>4;
    	}

    }

}


