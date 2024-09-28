#include<iostream>
#include<complex.h>
#include<math.h>
#define PI 3.14159
#define N 16
#define STAGE ((int)(log(N)/log(2)))
using namespace std;
typedef complex<double> data_t;

void fft_opt(data_t x[N],data_t X[N],int n);
