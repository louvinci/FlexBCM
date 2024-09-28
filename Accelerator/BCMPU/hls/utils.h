
#include<stdlib.h>
#include<stdio.h>

#include<math.h>

#include"config.h"


void TransHWC(data_t* in,data_t* out,int ch,int h,int w);
void random_init(data_t* x,int len);

void crandom_init(complexType* x,int len);
void crandom_init2(complexType2* x,int len);


void rcir_conv2d(complexType* in,complexType* weight,complexType2* bias,complexType2* out,int n,int m,int h,int w,int k,int s,int p,int block_size);
void interleave(data_t* in,data_t* out,int ch,int h,int w,int b);

void deinterleave(data_t* in,data_t* out,int ch,int h,int w,int b);

void FFT(complexTypeF32* Xin,complexTypeF32 *Xout,int n);

void RFFT(data_t* in,complexTypeF32* out,int n);

void IRFFT(complexTypeF32* in, Fix32* out,int n);

void feature_trans(data_t *in,complexType *out,int n,int h,int w,int blk);

void feature_itrans(complexType2* in,data_t* out,int n,int w,int h,int blk);

void weight_reorg(complexType* in,complexType* out,int n,int m,int tm,int tn,int k,int block_size);

void rego(complexType* in,complexType* out,int n,int m,int tm,int tn,int k,int block_size);
