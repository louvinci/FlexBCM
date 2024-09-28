#include "rfft.h"
double MAX(double a,double b){
   return (a>b)?a:b;
}


//bit reverse for two 8-point fft
void bit_reverse2(data_t* Xin,data_t* Xout){
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
void bit_reverse3(data_t* Xin,data_t* Xout){
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
void bit_reverse1(data_t* Xin,data_t* Xout){
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
//n-point fft(baseline)
void FFTn(data_t* Xin,data_t *Xout,int n){
    if(n<2)
         Xout[0]=Xin[0];
    else
    {
         data_t* X1=new data_t[n/2];
         data_t* X2=new data_t[n/2];
         data_t* X1_out=new data_t[n/2];
         data_t* X2_out=new data_t[n/2];
         for(int i=0;i<n;i+=2)
         {
                X1[i/2]=Xin[i];
                X2[i/2]=Xin[i+1];
         }
         FFTn(X1,X1_out,n/2);
         FFTn(X2,X2_out,n/2);
         data_t* W=new data_t[n/2];
         for(int i=0;i<n/2;i++){
            W[i].real(cos(2*PI*i/n));
            W[i].imag(-sin(2*PI*i/n));
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

void fft8_test(){
    cout<<"fft8 test"<<endl;
    complex<double> X[N];
    complex<double> Y[N];
    complex<double> Z[N];
    complex<double> W[N];
    for(int i=0;i<N;i++){
        X[i].real(N-i);
        X[i].imag(-i);
    }
    //baseline
    FFTn(X,Y,8);                      //the first 8-point fft
    FFTn(&X[8],&Y[8],8);              //the second 8-point fft
    //ours func
    bit_reverse2(X,Z);
    fft_opt(Z,W,8);
    //
    double max_diff=-100.0;
    complex<double> diff;
    for(int i=0;i<N;i++){
      diff=Y[i]-W[i];
      if(MAX(diff.real(),diff.imag()) > max_diff)
        max_diff=MAX(diff.real(),diff.imag());
      //cout<<Y[i]-W[i]<<endl;
    }
    cout<<"max difference is "<<max_diff<<endl;
    cout<<"********************"<<endl;
}

void fft4_test(){
    cout<<"fft4 test"<<endl;
    complex<double> X[N];
    complex<double> Y[N];
    complex<double> Z[N];
    complex<double> W[N];
    for(int i=0;i<N;i++){
        X[i].real(N-i);
        X[i].imag(-i);
    }
    //baseline
    FFTn(X,Y,4);                          //first
    FFTn(&X[4],&Y[4],4);                  //second
    FFTn(&X[8],&Y[8],4);                  //third
    FFTn(&X[12],&Y[12],4);                //fourth
    //our func
    bit_reverse3(X,Z);
    fft_opt(Z,W,4);
    //
    double max_diff=-100.0;
    complex<double> diff;
    for(int i=0;i<N;i++){
      diff=Y[i]-W[i];
      if(MAX(diff.real(),diff.imag())>max_diff)
          max_diff=MAX(diff.real(),diff.imag());
      //cout<<Y[i]-W[i]<<endl;
    }
    cout<<"max difference is "<<max_diff<<endl;
    cout<<"********************"<<endl;
}

void fft16_test(){
    cout<<"fft16 test"<<endl;
    complex<double> X[N];
    complex<double> Y[N];
    complex<double> Z[N];
    complex<double> W[N];
    for(int i=0;i<N;i++){
        X[i].real(N-i);
        X[i].imag(-i);
    }
    //baseline
    FFTn(X,Y,16);
    bit_reverse1(X,Z);
    //our func
    fft_opt(Z,W,16);
    //
    double max_diff;
    complex<double> diff;
    for(int i=0;i<N;i++){
      diff=Y[i]-W[i];
      if(MAX(diff.real(),diff.imag())>max_diff)
           max_diff=MAX(diff.real(),diff.imag());
      //cout<<Y[i]-W[i]<<endl;
    }
    cout<<"max difference is "<<max_diff<<endl;
    cout<<"********************"<<endl;
}

int main(){

    fft4_test();
    fft8_test();
    fft16_test();
    return 0;
}
