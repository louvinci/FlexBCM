#include"main.h"
#include"utils.h"
//#include"/tools/Xilinx/Vitis_HLS/2021.2/include/gmp.h"
//#define __gmp_const const
#include "config.h"
void test(int BL, int Hin,int Win,int N,int M, int Ksize, int Stride, int Pad){

    int Hout=(Hin+2*Pad-Ksize)/Stride +1;
    int Wout=(Win+2*Pad-Ksize)/Stride +1;

    printf("===========================real Input Init.=========================================\n");
    //input (N* H * W) real number
    data_t* in         =new data_t[MAXIN];
    data_t* in_hw      =new data_t[MAXIN];
    complexType* in_sw =new complexType[MAXIN/2];

    random_init(in,Hin*Win*N);            // generate random real input
    feature_trans(in,in_sw,N,Hin,Win,BL); // generate complex input in software reference
    TransHWC(in,in_hw,N,Hin,Win);         // real number,H*W*N format

    printf("===========================Complex Weight Init.=====================================\n");
    //weight N * M * K * K / (BL/2) complex number, compressed
    complexType* weightss   =new complexType[MAXWT];
    complexType* weight_reg =new complexType[MAXWT];
    complexType* tmp        =new complexType[MAXWT/2];
    //bias
    complexType2 biasss[512];

    crandom_init(weightss,M*N*Ksize*Ksize/(BL*BL)*(BL/2));
	weight_reorg(weightss,tmp,N/2,M/2,Tm,Tn,Ksize,BL/2);// symmetry //M*N*K*K/(BL/2)
	rego(tmp,weight_reg,N/2,M/2,Tm,Tn,Ksize,BL/2);

	crandom_init2(biasss,M/2); //compressed
	/**********************************************************************************************/
    data_t *out_hw       =new data_t[MAXOT]; //real number,H*W*M format

    printf("===========================start compute BCM %d, stride %d====================================\n",BL,Stride);

    circonv((ap_uint<Bit*ActWidth>*)in_hw,(ap_uint<CBit*WtWidth>*)weight_reg,(ap_uint<AccBit*2>*)biasss,
        	(ap_uint<Bit*ActWidth>*)out_hw,N/2,M/2,Hin,Ksize,Stride,BL);

    /**********************************************Golden ref************************************************/
    complexType2 *out_sw  =new complexType2[MAXOT];//M/2*Hout*Wout used

   data_t* outsw_real  =new data_t[MAXOT]; // NHW format
   rcir_conv2d((complexType*)in_sw,(complexType*)weightss,(complexType2*)biasss,(complexType2*)out_sw,
		        N/2,M/2,Hin,Win,Ksize,Stride,Pad,BL/2);
   feature_itrans(out_sw,outsw_real,M,Hout,Wout,BL);


    printf("compare result....\n");
    data_t eps = 5;
    unsigned int errors=0;
    for(int i=0;i<M*Hout*Wout;i++){
    	//cout<<outsw_real[i]<<","<<out_hw[i]<<endl;
    	if (outsw_real[i]!=out_hw[i]){
    		errors+=1;
    		cout<<outsw_real[i]<<","<<out_hw[i]<<endl;
    	}

    }
    cout<<"errors happens: "<<errors<<endl;
}


int main(){

	//test(4,  56,  56,64,64,1,1,0);
	//test(16,  28,  28,64,64,1,1,0);
	//test(16,  28,  28,128,128,1,1,0);
	//test(16,  14,  14,128,128,1,1,0);
	//test(8,  14,  14,256,256,1,1,0);
	//test(8,  7,  7,128,128,1,1,0);
	//test(8,  7,  7,256,256,1,1,0);
	//test(8,  7,  7,512,512,1,1,0);


	//test(8,  7,    7,128,128,3,1,1);
	//test(8,  7,    7,256,256,3,1,1);
	//test(8,  7,    7,512,512,3,1,1);
	//test(8,  14,  14,128,128,3,1,1);
    //test(8,  14,  14,256,256,3,1,1);
	//test(8,  28,  28,64,64,3,1,1);
	//test(8,  28,  28,128,128,3,1,1);
	//test(8,  56,  56,64,64,3,1,1);

	//test(8,  7,    7,128,128,1,1,0);
	//test(8,  7,    7,256,256,1,1,0);
	//test(8,  7,    7,512,512,1,1,0);
	//test(8,  14,  14,128,128,1,1,0);
	//test(8,  14,  14,256,256,1,1,0);
	//test(8,  28,  28,64,64,1,1,0);
	//test(8,  28,  28,128,128,1,1,0);
	//test(8,  56,  56,64,64,1,1,0);


	//test(16,  7,    7,128,128,1,1,0);
	//test(16,  7,    7,256,256,1,1,0);

	//test(16,  14,  14,128,128,1,1,0);
	//test(16,  14,  14,256,256,1,1,0);
	//test(16,  28,  28,64,64,1,1,0);
	//test(16,  28,  28,128,128,1,1,0);
	//test(16,  56,  56,64,64,1,1,0);

	//test(16,  7,    7,128,128,3,1,1);
	//test(16,  7,    7,256,256,3,1,1);

	//test(16,  14,  14,128,128,3,1,1);
	//test(16,  14,  14,256,256,3,1,1);
	//test(16,  28,  28,64,64,3,1,1);
	//test(16,  28,  28,128,128,3,1,1);
	test(16,  56,  56,64,64,3,1,1);
	return 0;
}


/*
 * //	data_t a=7.7, b=6.6;
//	complexType c;
//	c = complexType(a,b);
//	std::cout<<c<<std::endl;
//
//	uint16 a1,b1;
//	data_t a2,b2;
//	a1.range(15,0) = c.real().range(15,0);
//	b1.range(15,0) = c.imag().range(15,0);
//
//	a2.range(15,0) = a1.range(15,0);
//	b2.range(15,0) = b1.range(15,0);
//
//	std::cout<<(data_t)a1<<" "<<(data_t)b1<<std::endl;
//	std::cout<<a2<<" "<<b2<<std::endl;
//
//	test(4,  7,7,64,128);
//	test(8,  7,7,64,128);
//	test(16, 7,7,64,128);

//	ap_uint<32> a;
//	ap_uint<16> b=1,c=4; // must be data_t type
//	data_t b1, c1;
//
//	a= (b(15,0),c(15,0));
//	(b1(15,0),c1(15,0)) = a;
//	std::cout<<b1<<" "<<c1<<std::endl;

//	complex< ap_fixed<16,8>  > a = (complex< ap_fixed<16,8>  >)(1.5,-2);
//	complex< ap_fixed<32,16> > b = (complex< ap_fixed<32,16> >)(2,-1.2);
//	cout<< a*b <<endl; // cannot directly mutiply
 */

