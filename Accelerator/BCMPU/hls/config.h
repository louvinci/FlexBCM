#pragma once
#include<ap_fixed.h>
#include<complex>
#include<iostream>
//#include "/tools/Xilinx/Vitis_HLS/2021.2/include/gmp.h"
#define PI (3.1415927)
#define P 1
#define K 3


#define Tm (64/2)//(128/2)              //
#define Tn (64/2)//(32/2)              //
#define Tr 7
#define Tc 7
#define HBS 8
#define Smax 2

//56*56*64=200704, 3*3*512*512=2359296
#define MAXIN 56*56*128
#define MAXWT 3*3*512*512
#define MAXOT 56*56*128
using namespace std;
//typedef float data_t;
//typedef complex<float> complexType;
#define ActWidth 16
#define WtWidth 1
#define Bit 8
#define AccBit 32
#define CBit 8*2

typedef ap_int<32>  int32;
typedef ap_int<8>   int8;
typedef ap_int<16>  int16;
typedef ap_uint<16> uint16;
typedef ap_uint<8> uint8;
typedef ap_uint<4> uint4;
typedef ap_uint<2> uint2;
//typedef ap_fixed<16,6,AP_RND,AP_SAT> data_t;
// INT8
typedef ap_int<8> data_t;
typedef ap_fixed<32,24,AP_RND,AP_SAT> Fix32;
//typedef float data_t;
//typedef complex<data_t> complexType;
typedef complex<int8>  complexType;
typedef complex<int32> complexType1;
typedef complex<int32> complexType2; // accumulation buffer bit
typedef complex<Fix32> complexTypeF32; // accumulation buffer bit
