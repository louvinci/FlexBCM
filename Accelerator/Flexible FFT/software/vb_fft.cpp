#include"rfft.h"



//16-point fft stage1
void stage1(data_t x[N],data_t y[N]){
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
void stage2(data_t x[N],data_t y[N]){
    data_t W4;
    W4.real(0.0);
    W4.imag(-1.0);
    data_t tmp[4];
    tmp[0]=W4*x[3];
    tmp[1]=W4*x[7];
    tmp[2]=W4*x[11];
    tmp[3]=W4*x[15];
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
void stage3(data_t x[N],data_t y[N]){
    data_t W2,W4,W6;
    W2.real(cos(2*PI*2/N));
    W2.imag(-sin(2*PI*2/N));
    W4.real(cos(2*PI*4/N));
    W4.imag(-sin(2*PI*4/N));
    W6.real(cos(2*PI*6/N));
    W6.imag(-sin(2*PI*6/N));
    data_t tmp[6];
    tmp[0]=W2*x[5];
    tmp[1]=W4*x[6];
    tmp[2]=W6*x[7];
    tmp[3]=W2*x[13];
    tmp[4]=W4*x[14];
    tmp[5]=W6*x[15];
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
void stage4(data_t x[N],data_t y[N]){
    data_t W1,W2,W3,W4,W5,W6,W7;
    W1.real(cos(2*PI*1/N));
    W1.imag(-sin(2*PI*1/N));
    W2.real(cos(2*PI*2/N));
    W2.imag(-sin(2*PI*2/N));
    W3.real(cos(2*PI*3/N));
    W3.imag(-sin(2*PI*3/N));
    W4.real(cos(2*PI*4/N));
    W4.imag(-sin(2*PI*4/N));
    W5.real(cos(2*PI*5/N));
    W5.imag(-sin(2*PI*5/N));
    W6.real(cos(2*PI*6/N));
    W6.imag(-sin(2*PI*6/N));
    W7.real(cos(2*PI*7/N));
    W7.imag(-sin(2*PI*7/N));
    data_t tmp[7];
    tmp[0]=W1*x[9];
    tmp[1]=W2*x[10];
    tmp[2]=W3*x[11];
    tmp[3]=W4*x[12];
    tmp[4]=W5*x[13];
    tmp[5]=W6*x[14];
    tmp[6]=W7*x[15];
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
void fft_opt(data_t x[N],data_t X[N],int n){
    
    data_t Xtmp[STAGE+1][N];
    for(int i=0;i<N;i++)
    {
        Xtmp[0][i]=x[i];
    }
    for(int i=0;i<N;i++)
    {
        cout<<Xtmp[0][i]<<" ";
    }
    cout<<endl;
    //
    stage1(Xtmp[0],Xtmp[1]);
    stage2(Xtmp[1],Xtmp[2]);
    if(n==4){
        for(int i=0;i<16;i++){
            X[i]=Xtmp[2][i];
            cout<<"i="<<i<<","<<Xtmp[2][i]<<endl;
        }
        return;
    }
    stage3(Xtmp[2],Xtmp[3]);
    if(n==8){
        for(int i=0;i<16;i++){
            X[i]=Xtmp[3][i];
            //cout<<"i="<<i<<","<<Xtmp[3][i]<<endl;
        }
        return;
    }
    stage4(Xtmp[3],Xtmp[4]);
    if(n==16){
        for(int i=0;i<16;i++){
            X[i]=Xtmp[4][i];
            //cout<<"i="<<i<<","<<Xtmp[4][i]<<endl;
        }
        return;
    }
    //
    for(int i=0;i<N;i++){
        X[i]=Xtmp[STAGE][i];
    }
}



