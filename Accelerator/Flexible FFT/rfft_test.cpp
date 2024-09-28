#include "vb_fft.h"
using namespace std;

data_t MAX(data_t a,data_t b){
   return (a>b)?a:b;
}

//n-point fft(baseline)
void FFTn(complexType* Xin,complexType *Xout,int n){
    if(n<2)
         Xout[0]=Xin[0];
    else
    {
         complexType* X1=new complexType[n/2];
         complexType* X2=new complexType[n/2];
         complexType* X1_out=new complexType[n/2];
         complexType* X2_out=new complexType[n/2];
         for(int i=0;i<n;i+=2)
         {
                X1[i/2]=Xin[i];
                X2[i/2]=Xin[i+1];
         }
         FFTn(X1,X1_out,n/2);
         FFTn(X2,X2_out,n/2);
         complexType* W=new complexType[n/2];
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
    complex<data_t> X[N];
    complex<data_t> Y[N];
    complex<data_t> Z[N];
    complex<data_t> W[N];
    for(int i=0;i<N;i++){
        X[i].real(N-i);
        X[i].imag(-i);
    }
    //baseline
    FFTn(X,Y,8);                      //the first 8-point fft
    FFTn(&X[8],&Y[8],8);              //the second 8-point fft
    //ours func
    //bit_reverse2(X,Z);
    fft_opt(X,W,8);
    //
    data_t max_diff=-100.0;
    complex<data_t> diff;
    for(int i=0;i<N;i++){
      diff=Y[i]-W[i];
      if(MAX(diff.real(),diff.imag())>max_diff)
        max_diff=MAX(diff.real(),diff.imag());
      //cout<<Y[i]-W[i]<<endl;
    }
    cout<<"max difference is "<<max_diff<<endl;
    cout<<"********************"<<endl;
}

void fft4_test(){
    cout<<"fft4 test"<<endl;
    complex<data_t> X[N];
    complex<data_t> Y[N];
    complex<data_t> Z[N];
    complex<data_t> W[N];
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
    //bit_reverse3(X,Z);
    fft_opt(X,W,4);
    //
    data_t max_diff=-100.0;
    complex<data_t> diff;
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
    complex<data_t> X[N];
    complex<data_t> Y[N];
    complex<data_t> Z[N];
    complex<data_t> W[N];
    for(int i=0;i<N;i++){
        X[i].real(N-i);
        X[i].imag(-i);
    }
    //baseline
    FFTn(X,Y,16);
    //bit_reverse1(X,Z);
    //our func
    fft_opt(X,W,16);
    //
    data_t max_diff;
    complex<data_t> diff;
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
	for(int i=0;i<8;i++){
		cout<<W[i]<<" ";
	}
	cout<<endl;
    fft4_test();
    fft8_test();
    fft16_test();
    return 0;
}
