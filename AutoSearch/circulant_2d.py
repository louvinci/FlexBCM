import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.fft
import time
#!Version 6-06, the inference speed can't influcte the training speee. when traing, the vanilla conv is fast.
'''
******************************************************************
'''
def ComplexCon2d(input,weight,bias=None,stride=1,padding=1):
    assert bias == None
    input_r=input.real
    input_i=input.imag
    weight_r=weight.real
    weight_i=weight.imag
    out_rr=F.conv2d(input=input_r,weight=weight_r,bias=None,stride=stride,padding=padding)
    out_ii=F.conv2d(input=input_i,weight=weight_i,bias=None,stride=stride,padding=padding)
    out_ri=F.conv2d(input=input_r,weight=weight_i,bias=None,stride=stride,padding=padding)
    out_ir=F.conv2d(input=input_i,weight=weight_r,bias=None,stride=stride,padding=padding)
    return torch.complex(out_rr-out_ii,out_ri+out_ir)

#implement circonv with B//2+1 complex conv3d
class BCMConv2d_FFT_Mconv(nn.Module):
    def __init__(self,in_channels,out_channels,stride,padding,kernel_size,block_size,bias=True):
        super().__init__()
        self.block_size=block_size
        self.in_channels=in_channels
        self.out_channels=out_channels
        self.stride=stride
        self.padding=padding
        self.kernel_size=kernel_size
        self.add_bias = bias
        assert in_channels % block_size == 0
        assert out_channels % block_size == 0
        self.o_nblocks = out_channels//block_size
        self.i_nblocks = in_channels//block_size
        #
        weight = torch.empty(self.o_nblocks, self.i_nblocks, self.block_size, kernel_size, kernel_size,requires_grad=True)       #默认是(3,3,3)大小的kernel
        inited_weight = torch.nn.init.xavier_uniform_(weight, gain=1)
        self.weight = torch.nn.Parameter(inited_weight)
        if bias:
            uninit_bias = torch.empty(self.out_channels, requires_grad=True)
            inited_bias = torch.nn.init.constant_(uninit_bias, 0.0)
            self.bias = torch.nn.Parameter(inited_bias)
        else:
            self.bias = None
        #

    def forward(self, x):
        batch, ch_in, height, width = x.size()
    
        height_o=(height+2*self.padding-self.kernel_size)//self.stride+1
        width_o =(width+2*self.padding-self.kernel_size)//self.stride+1
        #(m/b,n/b,b,k,k,k)
        W = torch.fft.rfft(self.weight, dim=2)
        #(batch,n,d,r,c)-->(batch,n/b,b,d,r,c)
        X = x.view(batch, ch_in//self.block_size, self.block_size, height, width)
        X = torch.fft.rfft(X, dim=2)
        #
        if self.block_size == 4:
            Yc1=ComplexCon2d(input=X[:, :, 0, :, :],weight=W[:, :, 0, :, :],stride=self.stride,padding=self.padding,bias=None)
            Yc2=ComplexCon2d(input=X[:, :, 1, :, :],weight=W[:, :, 1, :, :],stride=self.stride,padding=self.padding,bias=None)
            Yc3=ComplexCon2d(input=X[:, :, 2, :, :],weight=W[:, :, 2, :, :],stride=self.stride,padding=self.padding,bias=None)
            Yc=torch.stack((Yc1, Yc2, Yc3), dim=2)
        elif self.block_size == 8:
            Yc1=ComplexCon2d(input=X[:, :, 0, :, :],weight=W[:, :, 0, :, :],stride=self.stride,padding=self.padding,bias=None)
            Yc2=ComplexCon2d(input=X[:, :, 1, :, :],weight=W[:, :, 1, :, :],stride=self.stride,padding=self.padding,bias=None)
            Yc3=ComplexCon2d(input=X[:, :, 2, :, :],weight=W[:, :, 2, :, :],stride=self.stride,padding=self.padding,bias=None)
            Yc4=ComplexCon2d(input=X[:, :, 3, :, :],weight=W[:, :, 3, :, :],stride=self.stride,padding=self.padding,bias=None)
            Yc5=ComplexCon2d(input=X[:, :, 4, :, :],weight=W[:, :, 4, :, :],stride=self.stride,padding=self.padding,bias=None)
            Yc=torch.stack((Yc1, Yc2, Yc3, Yc4, Yc5), dim=2)
        else:
            Yc1=ComplexCon2d(input=X[:, :, 0, :, :],weight=W[:, :, 0, :, :],stride=self.stride,padding=self.padding,bias=None)
            Yc2=ComplexCon2d(input=X[:, :, 1, :, :],weight=W[:, :, 1, :, :],stride=self.stride,padding=self.padding,bias=None)
            Yc3=ComplexCon2d(input=X[:, :, 2, :, :],weight=W[:, :, 2, :, :],stride=self.stride,padding=self.padding,bias=None)
            Yc4=ComplexCon2d(input=X[:, :, 3, :, :],weight=W[:, :, 3, :, :],stride=self.stride,padding=self.padding,bias=None)
            Yc5=ComplexCon2d(input=X[:, :, 4, :, :],weight=W[:, :, 4, :, :],stride=self.stride,padding=self.padding,bias=None)
            Yc6=ComplexCon2d(input=X[:, :, 5, :, :],weight=W[:, :, 5, :, :],stride=self.stride,padding=self.padding,bias=None)
            Yc7=ComplexCon2d(input=X[:, :, 6, :, :],weight=W[:, :, 6, :, :],stride=self.stride,padding=self.padding,bias=None)
            Yc8=ComplexCon2d(input=X[:, :, 7, :, :],weight=W[:, :, 7, :, :],stride=self.stride,padding=self.padding,bias=None)
            Yc9=ComplexCon2d(input=X[:, :, 8, :, :],weight=W[:, :, 8, :, :],stride=self.stride,padding=self.padding,bias=None)
            Yc=torch.stack((Yc1, Yc2, Yc3, Yc4, Yc5, Yc6, Yc7, Yc8, Yc9), dim=2)
        #
        Y = torch.fft.irfft(Yc, dim=2)
        #
        return Y.contiguous().view(batch,self.out_channels,height_o,width_o)
    def __repr__(self):
        return 'BCM_Conv2d_MC(in_ch={0}, out_ch={1}, K={2}, block_size={3}, pad={4}, stride={5}, bias={6})'.format(
                                                                     self.in_channels,self.out_channels,self.kernel_size, self.block_size, self.padding, self.stride, self.add_bias)

# w:(block_num_row, block_num_col, block_size); x:(batch_size,in_channels),in_channels = block_num_col * block_size
#!for quantization, add fft operation will increase one quant opeartor for activation.
def bcm_mvm_fft_fast(x,w):  

    num_r, num_c, block_size = w.size()  #
    batch_size,in_channels = x.size()[0], x.size()[1]
    assert num_c * block_size == in_channels, 'BCM_fft_fast num_c:%d, in_channels: %d'%(num_c,in_channels)

    ###############* fast version, here must use the complex64,before using torch.einsum
    #!note that quant step can be executed after the search process,i.e, traning the subnet.
    w_ffted = torch.fft.rfft(input=w.float(), dim=-1)#.type(torch.complex64) 
    # w_ffted fake quant INT8
    x_ffted = torch.fft.rfft(input=x.view(batch_size,num_c, block_size).float(),dim=-1)#.type(torch.complex64)
    #! x_ffted fake quant INT8
    res = torch.einsum('mki,pki->mpi',[x_ffted,w_ffted])
    # INT32
    z_iffted = torch.fft.irfft(res,dim=-1)
    #! fake quant INT8
    return z_iffted.view(batch_size, -1)

# w:(block_num_row, block_num_col, block_size); x:(batch_size,in_channels)
#* this version is slow 
def bcm_mvm_fft(x, w):  
    num_r, num_c, block_size = w.size() 
    batch_size, in_channels = x.size()
    assert num_c * block_size == in_channels, 'mismatch the common dimmension in BCM Linear fft'
    # w的fft变换
    w_ffted = torch.fft.rfft(input=w,dim=-1)    #(num_r,num_c,block_size)
    # x的fft变换
    x_ffted = torch.fft.rfft(input=x.view(batch_size, 1, num_c, block_size),dim=-1) #(batch_size,1,num_c,block_size)
    # z
    z = x_ffted * w_ffted
    # z的ifft变换
    z_iffted = torch.fft.irfft(input=torch.sum(z,dim=2), dim=-1)  #(batch_size,num_r,num_c,block_size)
    # 取实数部分
    return z_iffted.view(batch_size, -1)

# input the circulant matrix, resume to the uncompressed matrix and compute the x*w
# w:(block_num_r block_num_c, block_size); x:(batch_size,in_channels)
def wt_resume_2d(num_r,num_c,block_size,w):
    ori_weight=torch.empty(( num_r * block_size, num_c * block_size),device=w.device)
    for q in range(num_c):
        for j in range(block_size):
            ori_weight[:, q * block_size + j]=torch.roll(w[:,q,:],shifts=j,dims=-1).view(-1)
    
    return ori_weight
    

def img2col(x,K,S=1,P=0,D=1):
    batch_size,channel,h,w=x.size()
    h_o=int((h+2*P-K)/S)+1
    w_o=int((w+2*P-K)/S)+1
    x=F.unfold(x, kernel_size=K, dilation=D,padding=P,stride=S)
    x=x.view(batch_size,channel,K,K,h_o*w_o)
    x=x.permute(0,2,3,1,4)
    x=x.contiguous().view(batch_size,K*K*channel,h_o*w_o)
    return x

class BCM_Conv2d(nn.Module):
    def __init__(self,in_channels,out_channels,stride,padding,kernel_size,block_size,bias=True):
        super().__init__()
        self.block_size=block_size
        self.in_channels=in_channels
        self.out_channels=out_channels
        self.stride=stride
        self.padding=padding
        self.kernel_size=kernel_size
        self.add_bias = bias
        assert in_channels%block_size==0
        assert out_channels%block_size==0
        self.o_nblocks=out_channels//block_size
        self.i_nblocks=in_channels//block_size
        #
        self.w = torch.nn.Parameter(torch.empty(self.o_nblocks,self.i_nblocks,self.block_size,kernel_size,kernel_size,requires_grad=True))
        torch.nn.init.xavier_uniform_(self.w, gain=1)
        if bias:
            self.bias=torch.nn.Parameter(torch.empty(self.out_channels, requires_grad=True))
            torch.nn.init.constant_(self.bias, 0.0)
        else:
            self.bias=None

    def forward(self,x):
        # 根据压缩后权重恢复原始的循环权重
        ori_weight=torch.empty((self.out_channels,self.in_channels,self.kernel_size,self.kernel_size),device=x.device)
        for q in range(self.i_nblocks):
            for j in range(self.block_size):
                #ori_weight[:, q * self.block_size + j, :, :]   = torch.cat([self.w[:,q,-j:,:,:],self.w[:,q,:-j,:,:]],dim=1).view(-1,self.kernel_size,self.kernel_size)
                ori_weight[:, q * self.block_size + j, :, :] = torch.roll(self.w[:,q,:,:,:],shifts=j,dims=-3).view(-1,self.kernel_size,self.kernel_size)
        # 卷积
        return F.conv2d(input=x, weight=ori_weight, bias=self.bias, stride=self.stride, padding=self.padding)
    def __repr__(self):
        return 'BCM_Conv2d_Shift(in_ch={0}, out_ch={1}, K={2}, block_size={3}, pad={4}, stride={5}, bias={6})'.format(
                                                                     self.in_channels,self.out_channels,self.kernel_size, self.block_size, self.padding, self.stride, self.add_bias)
class BCM_Conv2d_fft(nn.Module):
    def __init__(self,input_channel,output_channel,kernel_size,stride,padding,block_size=4,bias=True):
        super(BCM_Conv2d_fft, self).__init__()
        self.input_channel=input_channel
        self.output_channel=output_channel
        self.kernel_size=kernel_size
        self.stride=stride
        self.padding=padding
        self.add_bias = bias
        self.block_size=block_size
        assert input_channel  >= block_size, 'BCM_Conv2d_fft:  input_channel: %d, block_size: %d'%(input_channel,block_size)
        assert output_channel >= block_size, 'BCM_Conv2d_fft: output_channel: %d, block_size: %d'%(output_channel,block_size)
        self.w    = nn.Parameter(torch.zeros(output_channel//block_size, kernel_size*kernel_size*input_channel//block_size, block_size))
        torch.nn.init.xavier_uniform_(self.w, gain=1)
        if bias:
            self.b = nn.Parameter(torch.zeros(output_channel))


    def BlockCirculantMM(self,X, w):  #X(batch_size,K*K*N,h_o*w_o)
        batch_size, row, col = X.size()  # batch_size,K*K*N,h_o*w_o
        X = X.permute(0, 2, 1)  # batch_size,h_o*w_o,K*K*N
        X = X.contiguous().view(-1, row)  # batch_size*h_o*w_o,K*K*N
        #print("LOG2: ", X.size(), " ", w.size())
        O = bcm_mvm_fft_fast(X,w)  # w:(M/block_size,K*K*N/block_size,block_size) X(batch_size*h_o*w_o,K*K*N)
        # (batch_size*h_o*w_o,M)
        O = O.view(batch_size, col, -1)  # batch_size,h_o*w_o,M)
        O = O.permute(0, 2, 1).contiguous()  # batch_size,M,h_o*w_o
        return O

    def forward(self,x):
        batch_size,channel,height,width=x.size()
        
        o_height=(height+2*self.padding-self.kernel_size)//self.stride+1
        o_width=(width+2*self.padding-self.kernel_size)//self.stride+1
        if self.kernel_size ==1:
            if self.stride==2:
                x = x[:,:,::2,::2].contiguous()
            X = x.view(batch_size,channel,-1)
        else:
            X=img2col(x,K=self.kernel_size,S=self.stride,P=self.padding)
        #X=img2col(x,K=self.kernel_size,S=self.stride,P=self.padding)
        #print('LOG: ', X.size())
        out=self.BlockCirculantMM(X,self.w)
        if self.add_bias:
            return out.view(batch_size,self.output_channel,o_height,o_width)+self.b.view(1,-1,1,1)
        else:
            return out.view(batch_size,self.output_channel,o_height,o_width)
    
    def __repr__(self):
        return 'BCM_Conv2d_FFT(in_channel={0}, out_channel={1}, K={2}, block_size={3}, pad={4}, stride={5}, bias={6})'.format(
                                                                     self.input_channel,self.output_channel,self.kernel_size, self.block_size, self.padding, self.stride, self.add_bias)


class BCM_Linear_fft(nn.Module):
    def __init__(self,input_size,output_size,block_size, bias=True):
        super().__init__()
        self.input_size=input_size
        self.output_size=output_size
        self.block_size=block_size
        self.num_r=output_size//block_size
        self.num_c=input_size//block_size
        self.add_bias = bias
        assert input_size%block_size==0
        assert output_size%block_size==0

        self.w=nn.Parameter(torch.zeros(self.num_r,self.num_c,block_size))
        torch.nn.init.xavier_uniform_(self.w, gain=1)
        if bias:
            self.b=nn.Parameter( torch.zeros( 1, self.output_size ))

    def forward(self,x):

        # x:(batch_size,input_size) in ConvNet 
        if self.add_bias:
            return bcm_mvm_fft_fast(x,self.w)+self.b
        else:
            return bcm_mvm_fft_fast(x,self.w)
    def __repr__(self):
        return 'BCM_linearFFT(in_features={0}, out_feartures={1}, block_size={2}, bias={3})'.format(self.input_size,self.output_size,self.block_size, self.add_bias)

class BCM_Linear(nn.Module):
    def __init__(self,in_channels,out_channels,block_size,add_bias=False):
        super().__init__()
        self.block_size=block_size
        self.in_channels=in_channels
        self.out_channels=out_channels
        self.add_bias = add_bias
        assert in_channels%block_size==0
        assert out_channels%block_size==0
        self.o_nblocks=out_channels//block_size
        self.i_nblocks=in_channels//block_size

        self.w=nn.Parameter(torch.empty(self.o_nblocks,self.i_nblocks,self.block_size,requires_grad=True))
        torch.nn.init.xavier_uniform_(self.w, gain=1)
        if add_bias:
            self.bias=nn.Parameter(torch.zeros(self.out_channels, requires_grad=True))
        else:
            self.bias = None

    def forward(self,x):
        # 根据压缩后权重恢复原始的循环权重
        #ori_weight=torch.empty((self.out_channels,self.in_channels),device=x.device)
        
        ori_weight = wt_resume_2d(self.o_nblocks,self.i_nblocks,self.block_size,self.w)
        #print(ori_weight.device)
        # for q in range(self.i_nblocks):
        #     for j in range(self.block_size):
        #         ori_weight[:, q * self.block_size + j]=torch.cat([self.w[:,q,-j:],self.w[:,q,:-j]],dim=1).view(-1)
        #compute
        return F.linear(input=x, weight=ori_weight, bias=self.bias)
    
    def __repr__(self):
        return 'BCM_linear(in_features={0}, out_feartures={1}, block_size={2}, bias={3})'.format(self.in_channels,self.out_channels,self.block_size, self.add_bias)

#! einsum is more fast, 2.5x
def eimsim_cmp(CNT=20):
    batch = 64
    num_r, num_c, block_size = 64,32,4
    x  = torch.randn( batch,num_c*block_size).cuda()
    w  = torch.randn( num_r, num_c, block_size).cuda()
    b_t = time.time()
    for i in range(CNT):
        res1 = bcm_mvm_fft(x,w)
    e_t = time.time()
    print(e_t-b_t)
    for i in range(CNT):
        res2 = bcm_mvm_fft_fast(x,w)
    print(time.time()-e_t)
    #print(res1,'\n',res2)
    
    flag = torch.allclose(res1,res2,atol=1e-5)
    print("einsum fft and fft is same?: {}".format(flag))

#! compare the bcm restore function, the torch.roll is more fast,3x 
def restore_comp(CNT=20):
    r,c,b = 64,64,8
    w = torch.arange(r*c*b).reshape(r,c,b).cuda()
    
    ori_weight=torch.empty((r*b,c*b)).cuda()

    b_t = time.time()
    for i in range(CNT):
        for q in range(c):
            for j in range(b):
                ori_weight[:, q * b + j]=torch.cat([w[:,q,-j:],w[:,q,:-j]],dim=1).view(-1)
    
    e_t = time.time()
    print(e_t-b_t)


    ori_weight2=torch.empty((r*b,c*b)).cuda()
    w2 = torch.arange(r*c*b).reshape(r,c,b).cuda()
    for i in range(CNT):
        ori_weight2 = wt_resume_2d(r,c,b,w2)
        # for q in range(c):
        #     for j in range(b):
        #         ori_weight2[:, q * b + j]=torch.roll(w2[:,q,:],shifts=j,dims=-1).view(-1)
    
    print(time.time()-e_t)

    flag = torch.equal(ori_weight,ori_weight2)
    print("The transformation is same : {}".format(flag))


#! linear: the fft verion is more fast, 35x
def linear_compare(CNT=100):
    # speed and value
    batch = 128
    in_channel,out_channel,blocksize = 64, 64, 8
    x = torch.randn(batch,in_channel).cuda()
    w = torch.randn( out_channel//blocksize,in_channel//blocksize,blocksize).cuda()

    linear_time = BCM_Linear(in_channel,out_channel,blocksize).cuda()
    linear_fft  = BCM_Linear_fft(in_channel,out_channel,blocksize).cuda()
    linear_time.w.data = w
    linear_fft.w.data = w
    
    b_t = time.time()
    for i in range(CNT):
        res_time = linear_time(x)
    e_t  = time.time()
    print(e_t-b_t)
    for i in range(CNT):
        res_fft  = linear_fft(x)
    print(time.time()-e_t)
    #default  1e-05 relative error, atol:absolute error
    flag = torch.allclose(res_time,res_fft,atol=1e-5)
    print("The linear versions are same : {}".format(flag))

#! BCM_Conv2d_fft is  more fast, 13x
def conv2d_compare(CNT=200):
    # X:batch_size,N,H,W; M//block_size,kernel_size*kernel_size*N//block_size,block_size
    # 1x1 conv don't use the im2col, saving time(25%)
    batch,N,H,W = 64,128,56,56
    M,K,block_size,padding = 64,1,16,0
    x = torch.randn(batch,N,H,W).cuda()
    w = torch.randn(M//block_size,N//block_size,block_size,K,K).cuda()
    fft_w = w.permute(0,3,4,1,2).contiguous().view(M//block_size,-1,block_size)
    conv2d_vanilla   =  nn.Conv2d(     N,M,stride=1,padding=padding, kernel_size=K                        ).cuda()
    conv2d_time      =  BCM_Conv2d(    N,M,stride=1,padding=padding, kernel_size=K, block_size=block_size).cuda()
    conv2d_fft       =  BCM_Conv2d_fft(N,M,stride=1,padding=padding, kernel_size=K, block_size=block_size).cuda()
    conv2d_fft_Mconv =  BCMConv2d_FFT_Mconv(N,M,stride=1,padding=padding, kernel_size=K, block_size=block_size).cuda()
    conv2d_time.w.data        = w
    conv2d_fft.w.data         = fft_w
    conv2d_fft_Mconv.weight.data   = w
    print("*"*10+"Begin Test"+"*"*10)
    with torch.no_grad():
        s_t = time.time()
        for i in range(CNT):
            res_vanilla = conv2d_vanilla(x)
        b_t = time.time()
        print('conv2d vanilla,  inference time:', b_t-s_t)
        for i in range(CNT):
            res_time = conv2d_time(x)
        e_t  = time.time()
        print('conv2d shift  ,  inference time:', e_t-b_t)
        for i in range(CNT):
            res_fft  = conv2d_fft(x)
        print('conv2d fft    , inference time: ', time.time()-e_t)

        e2_t = time.time()
        for _ in range(CNT):
            res_fft2  = conv2d_fft_Mconv(x)
        print('conv2d fft Mconv , inference time: ', time.time()-e2_t)

        flag = torch.allclose(res_time,res_fft,atol=1e-4)
        print("The conv versions are same : {}".format(flag))

        flag = torch.allclose(res_time,res_fft2,atol=1e-4)
        print("The conv versions are same : {}".format(flag))


    s_t1 = time.time()
    for i in range(CNT):
        res_vanilla = conv2d_vanilla(x)
        res = torch.sum(res_vanilla)
        res.backward()
    b_t1 = time.time()
    print('conv2d vanilla,  train time:', b_t1-s_t1)
    for i in range(CNT):
        res_time = conv2d_time(x)
        res2 = torch.sum(res_time)
        res2.backward()
    e_t1  = time.time()
    print('conv2d shift  ,  train time:', e_t1-b_t1)
    for i in range(CNT):
        res_fft  = conv2d_fft(x)
        res3 = torch.sum(res_fft)
        res3.backward()
    print('conv2d fft    , train time: ', time.time()-e_t)

    e_t2 = time.time()
    for _ in range(CNT):
        res_fft  = conv2d_fft_Mconv(x)
        res4 = torch.sum(res_fft)
        res4.backward()
    print('conv2d fft Mconv , train time: ', time.time()-e_t2)



if __name__=='__main__':
    
    #restore_comp()
    #linear_compare()
    #eimsim_cmp()
    conv2d_compare()


