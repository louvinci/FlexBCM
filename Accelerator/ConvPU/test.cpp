#include"ConvPE.h"
#include<stdlib.h>
#include<fstream>

using namespace std;



void Generate(data_t *input, data_t *weight, data_t *branch, ap_uint<NormWidth> *Norm,
    unsigned H, unsigned W, unsigned N, unsigned M, unsigned K, unsigned Hout, unsigned Wout) {

    for (int i = 0; i < H * W * N; i++) {
        input[i] = (rand() % 256) - 128;
    }

    for (int i = 0; i < M * N * K * K; i++) {
        weight[i] = (rand() % 256) - 128;
    }

    for (int i = 0; i < Hout * Wout * M; i++) {
        branch[i] = (rand() % 256) - 128;
    }

    for (int i = 0; i < M; i++) {
        ap_int<AccBIT> wt_d, bias_d;
        wt_d = (rand() % 2048) - 1024;
        bias_d = (rand() % 2048) - 1024;
        Norm[i].range(AccBIT - 1, 0) = wt_d.range(AccBIT - 1, 0);
        Norm[i].range(2*AccBIT - 1, AccBIT) = bias_d.range(AccBIT - 1, 0);
    }
    
}



void Load_In(data_t *input, data_t *weight, data_t *output, data_t *branch, ap_uint<NormWidth> *Norm,
            unsigned H, unsigned W, unsigned N, unsigned M, unsigned K, unsigned Hout, unsigned Wout) {

    ifstream f;
    
    f.open("./data/input.txt", ios::in);
    for (int i = 0; i < H * W * N; i++) {
        char d[50];
        f >> d;
        input[i] = atoi(d);
    }
    f.close();

    f.open("./data/weight.txt", ios::out);
    for (int i = 0; i < M * N * K * K; i++) {
        char d[50];
        f >> d;
        weight[i] = atoi(d);
    }
    f.close();

    f.open("./data/branch.txt", ios::out);
    for (int i = 0; i < Hout * Wout * M; i++) {
        char d[50];
        f >> d;
        branch[i] = atoi(d);
    }
    f.close();

    f.open("./data/norm.txt", ios::out);
    for (int i = 0; i < M; i++) {
        char wt[50], bias[50];
        ap_int<AccBIT> wt_d, bias_d;
        f >> wt >> bias;
        wt_d = atoi(wt);
        bias_d = atoi(bias);
        Norm[i].range(AccBIT - 1, 0) = wt_d.range(AccBIT - 1, 0);
        Norm[i].range(2*AccBIT - 1, AccBIT) = bias_d.range(AccBIT - 1, 0);
    }
    f.close();

}


void Save(data_t *output, unsigned size) {
    ofstream f;
    
    f.open("./data/base.txt", ios::out);
    for (int i = 0; i < size; i++) {
        f << output[i] << endl;
    }
    f.close();
}


void Load_O(data_t *output, unsigned size) {
    ifstream f;
    
    f.open("./data/base.txt", ios::in);
    for (int i = 0; i < size; i++) {
        char d[50];
        f >> d;
        output[i] = atoi(d);
    }
    f.close();
}


data_t* pad(data_t* x, int H, int W, int N, int P) {
    int pad_h = H + 2 * P;
    int pad_w = W + 2 * P;
    data_t *x_pad = new data_t[pad_h * pad_w * N];
    memset(x_pad, 0, pad_h * pad_w * N * sizeof(data_t));
    // padding x
    for (int i = 0; i < H; i++) {
        for (int j = 0; j < W; j++) {
            for (int c = 0; c < N; c++) {
                x_pad[((P + i) * pad_w + (P + j)) * N + c] = x[(i * W + j) * N + c];
            }
        }
    }
    return x_pad;
}


ap_int<BIT> Quant(ap_int<AccBIT> in, ap_uint<NormWidth> norm, ap_uint<16> out_shift) {
    const int MAX = (1 << (BIT - 1)) - 1;
    const int MIN = -(1 << (BIT - 1));
    ap_int<AccBIT> wt, bias, t_res;
    wt.range(AccBIT - 1, 0) = norm.range(AccBIT - 1, 0);
    bias.range(AccBIT - 1, 0) = norm.range(2*AccBIT - 1, AccBIT);
    t_res = (in*wt + bias) >> out_shift;

    ap_int<BIT> res;

    if (t_res > MAX) {
        res = MAX;
    } else if (t_res < MIN) {
        res = MIN;
    } else {
        res = t_res;
    }

    return res;
}


void Base(  data_t *input, 
            data_t *weight, 
            data_t *output,
            data_t *branch,
            ap_uint<NormWidth> *Norm,
            unsigned H, unsigned W, unsigned N, unsigned M,
            unsigned K, unsigned S, unsigned P,
            ap_uint<16> out_shift
) {
    unsigned Hout = (H - K + 2*P)/S + 1;
    unsigned Wout = (W - K + 2*P)/S + 1;
    unsigned Pad_H = H + 2*P;
    unsigned Pad_W = W + 2*P;

    data_t *input_pad = pad(input, H, W, N, P);
    ap_int<AccBIT> *d_out = new ap_int<AccBIT>[Hout * Wout * M]();
    memset(d_out, 0, Hout * Wout * M * sizeof(ap_int<AccBIT>)); 

    // conv
    for (int m = 0; m < M; m++) {
        unsigned m_block = m / Tm;
        unsigned mm = m % Tm;
        for (int n = 0; n < N; n++) {
            unsigned n_block = n / Tn;
            unsigned nn = n % Tn;
            for (int r = 0; r < Hout; r++) {
                for (int c = 0; c < Wout; c++) {
                    for (int kx = 0; kx < K; kx++) {
                        for (int ky = 0; ky < K; ky++) {
                            d_out[(r * Wout + c) * M + m] += input_pad[((r * S + kx) * Pad_W + c * S + ky) * N + n]
                                                          * weight[((((m_block * N/Tn + n_block) * K + kx) * K + ky) * Tm + mm) * Tn + nn];
                        }
                    }
                }
            }
        }
    }
    

    // branch-add
    for (int r = 0; r < Hout; r++) {
        for (int c = 0; c < Wout; c++) {
            for (int m = 0; m < M; m++) {
                d_out[(r * Wout + c) * M + m] += branch[(r * Wout + c) * M + m];
            }
        }
    }


    // BN
    for (int r = 0; r < Hout; r++) {
        for (int c = 0; c < Wout; c++) {
            for (int m = 0; m < M; m++) {
                ap_uint<NormWidth> t_norm = Norm[m];
                output[(r * Wout + c) * M + m] = Quant(d_out[(r * Wout + c) * M + m], t_norm, out_shift);
                // !BN
                // output[(r * Wout + c) * M + m] = d_out[(r * Wout + c) * M + m].range(BIT - 1, 0);
            }
        }
    }

    delete [] input_pad;
    delete [] d_out;

}


int main() {
    // 28 * 28 * 32 -> 14 * 14 * 64
    // H = W = 28, N = 32, M = 64
    // K = 3, S = 2, P = 1
    // unsigned H = 28;
    // unsigned W = 28;
    // unsigned N = 32;
    // unsigned M = 64;
    // unsigned K = 3;
    // unsigned S = 2;
    // unsigned P = 1;
    // unsigned Hout = (H - K + 2*P)/S + 1;
    // unsigned Wout = (W - K + 2*P)/S + 1;


    
    // 128 * 128 * 32 -> 64 * 64 * 64
    // unsigned H = 128;
    // unsigned W = 128;
    // unsigned N = 32;
    // unsigned M = 64;
    // unsigned short K = 3;
    // unsigned short S = 2;
    // unsigned short P = 1;
    // unsigned Hout = (H - K + 2*P)/S + 1;
    // unsigned Wout = (W - K + 2*P)/S + 1;


    unsigned H = 7;
    unsigned W = 7;
    unsigned N = 128;
    unsigned M = 128;
    unsigned short K = 3;
    unsigned short S = 1;
    unsigned short P = 1;
    unsigned Hout = (H - K + 2*P)/S + 1;
    unsigned Wout = (W - K + 2*P)/S + 1;


    data_t *input;              // HxWxN
    data_t *weight;             // M/Tm x N/Tn x Kx x Ky x Tm x Tn
    data_t *output;             // HoutxWoutxM
    data_t *branch;             // HoutxWoutxM
    ap_uint<NormWidth> *norm;   // M
    data_t *base_o;             // HoutxWoutxM
    ap_uint<16> out_shift = rand()%8;

    input = new data_t[MAXIN]();
    weight = new data_t[MAXWT]();
    output = new data_t[MAXOT]();
    branch = new data_t[MAXOT]();
    norm = new ap_uint<NormWidth>[MAXM]();

    base_o = new data_t[MAXOT]();

    // Load_In(input, weight, output, branch, norm, H, W, N, M, K, Hout, Wout);
    Generate(input, weight, branch, norm, H, W, N, M, K, Hout, Wout);
    Base(input, weight, base_o, branch, norm, H, W, N, M, K, S, P, out_shift);
    // Save(base_o, Hout * Wout * M);

    ConvPE((ap_uint<InWidth>*)input, (ap_uint<WtWidth>*)weight, (ap_uint<OutWidth>*)output, (ap_uint<OutWidth>*)branch, norm, 
            H, W, N, M, K, S, P, out_shift);


    int errcnt = 0;

     for (int r = 0; r < Hout; r++) {
         for (int c = 0; c < Wout; c++) {
             for (int m = 0; m < M; m++) {
                 data_t out = output[(r * Wout + c) * M + m];
                 data_t ref = base_o[(r * Wout + c) * M + m];
                 if (out != ref) {
                     cout << "[r][c][m]: " << r << " " << c << " " << m << "\t" << "out : " << out << "\t\t" << "ref : " << ref << endl;
                     errcnt++;
                 }
             }
         }
     }

    delete [] input;
    delete [] weight;
    delete [] output;
    delete [] branch;
    delete [] norm;

    if (errcnt == 0) {
        std::cout << "*** TEST PASSED ***" << std::endl;
        return 0;
    } else {
    	std::cout << "diff nums : " << errcnt << std::endl;
        std::cout << "!!! TEST FAILED !!!" << std::endl;
        return 1;
    }


}
