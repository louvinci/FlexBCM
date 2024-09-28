#include"compute_core.h"


// complex multiplication optimization, and the special value processing
complexType1 cmult(complexType a,complexType b,bool flag){
#pragma HLS INLINE
	int16 tmp1;
	int16 tmp2;
	int8 x1=a.real();
	int8 y1=a.imag();
	int8 x2=b.real();
	int8 y2=b.imag();
	if(flag==false)                  //normal complex mult
	{
		tmp1=x1+y1;
		tmp2=x2+y2;
	}
	else{
		tmp1=x1;
		tmp2=y2;
	}
	int16 k1=x2*tmp1;
	int16 k2=x1*(y2-x2);
	int16 k3=y1*tmp2;
	complexType1 out;
	if(flag==false){
		out.real(k1-k3);
		out.imag(k1+k2);
	}
	else{
		out.real(k1);
		out.imag(k3);
	}
	return out;
}

template<class T>
T Reg(T in){
//#pragma HLS INTERFACE register port=return
#pragma HLS latency min=1 max=1
#pragma HLS INLINE off
#pragma HLS PIPELINE
	return in;
}

ap_int<36> MUL_INT9(ap_int<9> A, ap_int<9> W0, ap_int<9> W1)
{
    ap_int<27> W;
    W = (W0, ap_uint<18>(0)) + ap_int<27>(W1);

    ap_int<18> r0;
    ap_int<18> r1;

    (r0, r1) = A*W;

    r0 = r0+r1[18-1];

    return (r0,r1);
}

void cmult_opt(complexType a,complexType b,complexType c,complexType2 & ac,complexType2 & bc,bool flag){
    ap_int<9> ar=(ap_int<9>)a.real();
    ap_int<9> ai=(ap_int<9>)a.imag();
    ap_int<9> br=(ap_int<9>)b.real();
    ap_int<9> bi=(ap_int<9>)b.imag();
    ap_int<9> cr=(ap_int<9>)c.real();
    ap_int<9> ci=(ap_int<9>)c.imag();
    ap_int<18> r0,r1;
    ap_int<9> arai,brbi,cicr,crci;
    //
    if(flag){
		arai=ar;
		brbi=br;
		cicr=ci;
		crci=ci;
    }
    else{
		arai=ar+ai;               //ar+ai
		brbi=br+bi;
		cicr=ci-cr;               //ci-cr
		crci=cr+ci;               //cr+ci
    }

    //
    (r0,r1)=MUL_INT9(cr,arai,brbi);
    ap_int<32> k11=(ap_int<32>)r0;
    ap_int<32> k21=(ap_int<32>)r1;
    (r0,r1)=MUL_INT9(cicr,ar,br);
    ap_int<32> k12=(ap_int<32>)r0;
    ap_int<32> k22=(ap_int<32>)r1;
    (r0,r1)=MUL_INT9(crci,ai,bi);
    ap_int<32> k13=(ap_int<32>)r0;
    ap_int<32> k23=(ap_int<32>)r1;

    //
    if(flag){
		ac.real(k11);
		ac.imag(k13);
		bc.real(k21);
		bc.imag(k23);
    }
    else{
		ac.real(k11-k13);
		ac.imag(k11+k12);
		bc.real(k21-k23);
		bc.imag(k21+k22);
    }
}

complexType1 cmult2(complexType a,complexType b){

	int16 tmp1;
	int16 tmp2;
	int8 ar=a.real();
	int8 ai=a.imag();
	int8 br=b.real();
	int8 bi=b.imag();

	tmp1=ar+ai;
	tmp2=br+bi;

	int16 k1=br*tmp1;
	int16 k2=ar*(bi-br);
	int16 k3=ai*tmp2;
	complexType1 out;
	out.real(k1-k3);
	out.imag(k1+k2);

	return out;
}

complexType rmult(complexType a,complexType b){
	complexType out;
	out.real(a.real()*b.real());
	out.imag(a.imag()*b.imag());
	return out;
}


complexType2 addtree_4(complexType1 a, complexType1 b, complexType1 c, complexType1 d){
	complexType2 tmp1,tmp2,res;
	tmp1 = Reg(a+b);
	tmp2 = Reg(c+d);
	res  = Reg(tmp1+tmp2);
	return res;
}

complexType2 addtree_8(complexType1 a,  complexType1 b,  complexType1 c,  complexType1 d,
		              complexType1 a1, complexType1 b1, complexType1 c1, complexType1 d1){
	complexType2 tmp1,tmp2,tmp3,tmp4,res1,res2,res;
	tmp1 = Reg(a+b);
	tmp2 = Reg(c+d);
	tmp3 = Reg(a1+b1);
	tmp4 = Reg(c1+d1);
	res1 = Reg(tmp1+tmp2);
	res2 = Reg(tmp3+tmp4);
	res  = Reg(res1+res2);
	return res;
}

complexType2 addtree_16(complexType1 a,  complexType1 b,  complexType1 c,  complexType1 d,
		               complexType1 a1, complexType1 b1, complexType1 c1, complexType1 d1,
					   complexType1 a2, complexType1 b2, complexType1 c2, complexType1 d2,
					   complexType1 a3, complexType1 b3, complexType1 c3, complexType1 d3){
	complexType2 tmp1,tmp2,tmp3,tmp4,tmp5,tmp6,tmp7,tmp8,res1,res2,res3,res4,res5,res6,res;
	tmp1 = Reg(a+b);
	tmp2 = Reg(c+d);
	tmp3 = Reg(a1+b1);
	tmp4 = Reg(c1+d1);
	res1 = Reg(tmp1+tmp2);
	res2 = Reg(tmp3+tmp4);
	res3  = Reg(res1+res2);

	tmp5 = Reg(a2+b2);
	tmp6 = Reg(c2+d2);
	tmp7 = Reg(a3+b3);
	tmp8 = Reg(c3+d3);
	res4 = Reg(tmp5+tmp6);
	res5 = Reg(tmp7+tmp8);
	res6 = Reg(res4+res5);
	res  = Reg(res3 + res6);
	return res;
}


complexType2 addtree_32(complexType1 a,  complexType1 b,  complexType1 c,  complexType1 d,
						complexType1 a1, complexType1 b1, complexType1 c1, complexType1 d1,
						complexType1 a2, complexType1 b2, complexType1 c2, complexType1 d2,
						complexType1 a3, complexType1 b3, complexType1 c3, complexType1 d3,
						complexType1 a4, complexType1 b4, complexType1 c4, complexType1 d4,
						complexType1 a5, complexType1 b5, complexType1 c5, complexType1 d5,
						complexType1 a6, complexType1 b6, complexType1 c6, complexType1 d6,
						complexType1 a7, complexType1 b7, complexType1 c7, complexType1 d7){
	complexType2 tmp1, tmp2, tmp3, tmp4, tmp5, tmp6, tmp7, tmp8,  res1,res2,res3, res4, res5, res6, res7;
	complexType2 tmp9, tmp10,tmp11,tmp12,tmp13,tmp14,tmp15,tmp16, res8,res9,res10,res11,res12,res13,res14, res;
	tmp1 = Reg(a+b);
	tmp2 = Reg(c+d);
	tmp3 = Reg(a1+b1);
	tmp4 = Reg(c1+d1);
	res1 = Reg(tmp1+tmp2);
	res2 = Reg(tmp3+tmp4);
	res3 = Reg(res1+res2);

	tmp5 = Reg(a2+b2);
	tmp6 = Reg(c2+d2);
	tmp7 = Reg(a3+b3);
	tmp8 = Reg(c3+d3);
	res4 = Reg(tmp5+tmp6);
	res5 = Reg(tmp7+tmp8);
	res6 = Reg(res4+res5);

	res7  = Reg(res3 + res6);

	tmp9  = Reg(a4+b4);
	tmp10 = Reg(c4+d4);
	tmp11 = Reg(a5+b5);
	tmp12 = Reg(c5+d5);
	res8  = Reg(tmp9+tmp10);
	res9  = Reg(tmp11+tmp12);
	res10 = Reg(res8+res9);

	tmp13 = Reg(a6+b6);
	tmp14 = Reg(c6+d6);
	tmp15 = Reg(a7+b7);
	tmp16 = Reg(c7+d7);
	res11 = Reg(tmp13+tmp14);
	res12 = Reg(tmp15+tmp16);
	res13 = Reg(res11+res12);

	res14 = Reg(res10 + res13);
	res   = Reg(res7  + res14);
	return res;
}



