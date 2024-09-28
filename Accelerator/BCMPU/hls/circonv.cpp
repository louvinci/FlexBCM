#include"circonv.h"



void bias_init(complexType2 ofm_buff[Tm][Tr][Tc],complexType2 bias_buff[Tm]){

    for(int rr=0;rr<Tr;rr++)
        for(int cc=0;cc<Tc;cc++){
#pragma HLS PIPELINE
            for(int mm=0;mm<Tm;mm++)
            {
                ofm_buff[mm][rr][cc]=bias_buff[mm];
            }
        }
}
typedef struct
{
    uint16 m;
    uint16 r;
    uint16 c;
}Block;

void compute_output(
		ap_uint<Bit*ActWidth>* in3, ap_uint<CBit*WtWidth>* w,
		complexType2 ofm_buff[Tm][Tr][Tc],
		complexType2 bias_buff[Tm],
		uint16 m,uint16 r,uint16 c,uint16 fsize,uint16 ch_in,uint16 ch_out,uint8 blk_size,
		uint8 Ksize, uint2 Stride
){

	complexType ifm_buff1[Tn][Smax*Tr+K-Smax][Smax*Tc+K-Smax];
	complexType ifm_buff2[Tn][Smax*Tr+K-Smax][Smax*Tc+K-Smax];
#pragma HLS ARRAY_PARTITION variable=ifm_buff1 complete dim=1
#pragma HLS ARRAY_PARTITION variable=ifm_buff2 complete dim=1
#pragma HLS BIND_STORAGE variable=ifm_buff1 type=ram_2p impl=bram
#pragma HLS BIND_STORAGE variable=ifm_buff2 type=ram_2p impl=bram

	complexType wt_buff1[Tm/2][Tn][K][K];
	complexType wt_buff2[Tm/2][Tn][K][K];
#pragma HLS ARRAY_PARTITION variable=wt_buff1 complete dim=2
#pragma HLS ARRAY_PARTITION variable=wt_buff1 complete dim=1
#pragma HLS ARRAY_PARTITION variable=wt_buff2 complete dim=2
#pragma HLS ARRAY_PARTITION variable=wt_buff2 complete dim=1


    uint16 n=0;
    bool pp=true;
    //
    bias_init(ofm_buff,bias_buff);
    load_ifm16(in3,ifm_buff1,r,c,n,fsize,ch_in,blk_size,Stride,Ksize);
    load_weight(w,wt_buff1,n,m,ch_in,blk_size,Ksize);
    for(n=Tn;n<ch_in;n+=Tn){
#pragma HLS LOOP_TRIPCOUNT min=15 max=15 avg=15
        if(pp){
        	load_ifm16(in3,ifm_buff2,r,c,n,fsize,ch_in,blk_size,Stride,Ksize);
            load_weight(w,wt_buff2,n,m,ch_in,blk_size,Ksize);
            compute(ifm_buff1,wt_buff1,ofm_buff,blk_size,Stride,Ksize);
            pp=false;
        }
        else{
        	load_ifm16(in3,ifm_buff1,r,c,n,fsize,ch_in,blk_size,Stride,Ksize);
            load_weight(w,wt_buff1,n,m,ch_in,blk_size,Ksize);
            compute(ifm_buff2,wt_buff2,ofm_buff,blk_size,Stride,Ksize);
            pp=true;
        }
    }
    if(pp){
        compute(ifm_buff1,wt_buff1,ofm_buff,blk_size,Stride,Ksize);
    }
    else{
        compute(ifm_buff2,wt_buff2,ofm_buff,blk_size,Stride,Ksize);
    }
}



void update_block(Block cur_block,Block& next_block,uint16 fsize){
    if(cur_block.c+Tc>=fsize){
        if(cur_block.r+Tr>=fsize){
            next_block.m=cur_block.m+Tm;
            next_block.r=0;
            next_block.c=0;
        }
        else{
            next_block.m=cur_block.m;
            next_block.r=cur_block.r+Tr;
            next_block.c=0;
        } 
    }
    else{
        next_block.m=cur_block.m;
        next_block.r=cur_block.r;
        next_block.c=cur_block.c+Tc;
    }
}

void load_bias(ap_uint<AccBit*2>* bias,complexType2 bias_buff[256/Tm][Tm],uint16 ch_out){
	ap_uint<AccBit*2> tmp;
	int32 real;
	int32 imag;
	BIAS:
	for(int i=0;i<ch_out/Tm;i++)
		for(int j=0;j<Tm;j++){
#pragma HLS PIPELINE II=1
			tmp=*(bias+i*Tm+j);
			real.range(31,0)=tmp.range(31,0);
			imag.range(31,0)=tmp.range(63,32);
			bias_buff[i][j]=complexType2(real,imag);
		}
}

void circonv( ap_uint<Bit*ActWidth> in3[MAXIN/ActWidth], ap_uint<CBit*WtWidth> complex_w[MAXWT/2],
		      ap_uint<AccBit*2> bias[512/2],    ap_uint<Bit*ActWidth> out3[MAXOT/ActWidth],
			  unsigned ch_in,unsigned ch_out,unsigned fsize,unsigned Ksize,unsigned Stride,
			  unsigned blk_size){

#pragma HLS INTERFACE m_axi depth=25088    port=out3 offset=slave bundle=OUT3    latency=64 max_write_burst_length=128 num_read_outstanding=1  num_write_outstanding=32
#pragma HLS INTERFACE m_axi depth=1024     port=bias offset=slave bundle=B       latency=64 max_read_burst_length=128  num_read_outstanding=1  num_write_outstanding=1
#pragma HLS INTERFACE m_axi depth=589824  port=complex_w offset=slave bundle=W1 latency=64 max_read_burst_length=256  num_read_outstanding=8  num_write_outstanding=1
#pragma HLS INTERFACE m_axi depth=25088    port=in3 offset=slave bundle=IN3      latency=64 max_read_burst_length=128  num_read_outstanding=32  num_write_outstanding=1

#pragma HLS INTERFACE s_axilite port=return bundle=CTRL
#pragma HLS INTERFACE s_axilite port=ch_out bundle=CTRL
#pragma HLS INTERFACE s_axilite port=ch_in bundle=CTRL
#pragma HLS INTERFACE s_axilite port=fsize bundle=CTRL
#pragma HLS INTERFACE s_axilite port=Ksize bundle=CTRL
#pragma HLS INTERFACE s_axilite port=blk_size bundle=CTRL
#pragma HLS INTERFACE s_axilite port=Stride bundle=CTRL
//


	complexType2 ofm_buff1[Tm][Tr][Tc];
	complexType2 ofm_buff2[Tm][Tr][Tc];
#pragma HLS ARRAY_PARTITION variable=ofm_buff1 complete dim=1
#pragma HLS ARRAY_PARTITION variable=ofm_buff2 complete dim=1
#pragma HLS BIND_STORAGE variable=ofm_buff1 type=ram_2p impl=bram
#pragma HLS BIND_STORAGE variable=ofm_buff2 type=ram_2p impl=bram

	complexType2 bias_buff[256/Tm][Tm];
#pragma HLS ARRAY_PARTITION variable=bias_buff complete dim=2

    Block cur_block;
    Block next_block;

    unsigned short fsize_o = (Stride == 1)? fsize:(fsize)/2; // in RN34 or RN50,stride =1 or 2.
    bool pp=true;
    cur_block.m=0;

    cur_block.r=0;
    cur_block.c=0;
    next_block.m=0;

    next_block.r=0;
    next_block.c=0;

    load_bias(bias,bias_buff,ch_out);

    //
    compute_output(in3,complex_w,ofm_buff1,bias_buff[cur_block.m/Tm],cur_block.m,cur_block.r,cur_block.c,fsize,ch_in,ch_out,blk_size,Ksize,Stride);

    while(true){
#pragma HLS LOOP_TRIPCOUNT min=7 max=7 avg=7
        update_block(cur_block,next_block,fsize_o);
        if(next_block.m>=ch_out)
            break;
        if(pp){
            compute_output(in3,complex_w,ofm_buff2,bias_buff[next_block.m/Tm],next_block.m,next_block.r,next_block.c,fsize,ch_in,ch_out,blk_size,Ksize,Stride);
            store_ofm16(out3,ofm_buff1,cur_block.m,cur_block.r,cur_block.c,fsize_o,ch_out,blk_size);
            pp=false;
        }
        else{
            compute_output(in3,complex_w,ofm_buff1,bias_buff[next_block.m/Tm],next_block.m,next_block.r,next_block.c,fsize,ch_in,ch_out,blk_size,Ksize,Stride);
            store_ofm16(out3,ofm_buff2,cur_block.m,cur_block.r,cur_block.c,fsize_o,ch_out,blk_size);
            pp=true;
        }
        cur_block.m=next_block.m;
        cur_block.r=next_block.r;
        cur_block.c=next_block.c;
    }
    if(pp){
    	store_ofm16(out3,ofm_buff1,cur_block.m,cur_block.r,cur_block.c,fsize_o,ch_out,blk_size);
    }
    else{
    	store_ofm16(out3,ofm_buff2,cur_block.m,cur_block.r,cur_block.c,fsize_o,ch_out,blk_size);
    }
}



