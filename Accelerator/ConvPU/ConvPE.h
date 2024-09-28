#ifndef _CONVPE_H
#define _CONVPE_H

#include<stdint.h>
#include "ap_int.h"
#include "ap_fixed.h"
#include<iostream>

#define MAXIN 56*56*128
#define MAXWT 3*3*256*256
#define MAXOT 56*56*128
#define MAXM 256


#define BIT 8
#define AccBIT 32

#define TmBIT 5
#define TnBIT 3
#define Tm 32
#define Tn 8
#define Tr 7
#define Tc 7
#define S_max 2
#define K_max 3

#define OutNum 8
#define InWidth (Tn*BIT)
#define WtWidth (Tn*BIT)
#define OutWidth (OutNum*BIT)
#define NormWidth 64  // wt + bias

typedef ap_int<BIT> data_t;

using namespace std;
/***************************************************************************/


extern "C" {
void ConvPE(ap_uint<InWidth> input[MAXIN/Tn],
            ap_uint<WtWidth> weight[MAXWT/Tn],
            ap_uint<OutWidth> output[MAXOT/OutNum],
            ap_uint<OutWidth> branch[MAXOT/OutNum],
            ap_uint<NormWidth> norm[MAXM],
            unsigned H, unsigned W, unsigned N, unsigned M,
            unsigned short K, unsigned short S, unsigned short P,
            ap_uint<16> out_shift
);
}

#endif
