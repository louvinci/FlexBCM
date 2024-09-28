#include<iostream>
#include<stdlib.h>
#include<stdio.h>
#include<complex>
#include<ap_fixed.h>
#define PI 3.14159
#define N 16
#define STAGE ((int)(log(N)/log(2)))
using namespace std;

typedef ap_fixed<16,6,AP_RND,AP_SAT> data_t;
//typedef double data_t;
typedef complex<data_t> complexType;

void fft_opt(complexType x[N],complexType X[N],int n);


const complexType W[16/2]={complexType(1,0),
                           complexType(0.92388,-0.382683),
                           complexType(0.707107,-0.707106),
                           complexType(0.382684,-0.923879),
		                   complexType(0,-1),
                           complexType(-0.382682,-0.92388),
                           complexType(-0.707105,-0.707108),
                           complexType(-0.923879,-0.382686)
};

//1,0) (0.923828,-0.382813) (0.707031,-0.707031) (0.382813,-0.923828) (0,-1) (-0.382813,-0.923828) (-0.707031,-0.707031) (-0.923828,-0.382813)

