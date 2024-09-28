#include "config.h"
#define  FFT_LEN 16
#define STAGE 4

//#define STAGE ((int)(log(N)/log(2)))
//typedef ap_fixed<16,6,AP_RND,AP_SAT> data_t;
////typedef double data_t;
//typedef complex<data_t> complexTypeF32;


void rfft_opt(int8 x[FFT_LEN],complexTypeF32 X[FFT_LEN], unsigned char blk_size);

void irfft_opt(complexTypeF32 Xin[FFT_LEN/2],Fix32 Xout[FFT_LEN], unsigned char blk_size);


const complexTypeF32 W[16/2]={complexTypeF32(1,0),
                           complexTypeF32(0.92388,-0.382683),
                           complexTypeF32(0.707107,-0.707106),
                           complexTypeF32(0.382684,-0.923879),
		                   complexTypeF32(0,-1),
                           complexTypeF32(-0.382682,-0.92388),
                           complexTypeF32(-0.707105,-0.707108),
                           complexTypeF32(-0.923879,-0.382686)
};


