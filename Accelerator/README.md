## FlexBCM-Accelearator
Heterogeneous dual core structure containing the ConvPU and BCMPU.

### Description
- ```BCMPU```：
Block-circulant matrix process unit implementation by HLS.
- ```Flexible FFT```: Flexible support for 1xFFT-16, 2xFFT-8, 4xFFT-4 in HLS
- ```ConvPU```：Flexible support for different convolution layers
- ```Simulator```：Accelerator performance simulator
- ```Vitis```: Vitis deployment related files

### Deploy Kernel in HLS
Enter the corresponding file folder and execute the following command

```vitis_hls -f build_hls.tcl```

### Run Simulator
Enter the ```search_engine``` folder, run
```python search_hw```

### Run Vitis
see https://github.com/YunjiQin/vitis_workflow

### Code description

- Tm,Tn >=HBS,here the smallest block size=16 is set due to circulant symmetry, here HBS=8 • 
- Weights are stored directly in the chip in complex form and use circulant compression. Input and output exist in real form outside the chip in NHWC format, channel first. 
- Input and output need FFT/IFFT. After the input is converted into complex numbers through FFT, it is stored on the chip using circulant storage. Therefore, the input and weight slices participating in BCM calculation are all complex numbers well preserved by circulant.

```
After considering conjugate symmetry for complex weights, the size is M/Tm * N/Tn * (Tm/bs) * (Tn/bs) * (bs/2)  
On-chip compute：  
(M/2)/Tm * (N/2 )/Tn * (Tm / (bs/2)) * (Tn/(bs/2)) * (bs/2)  
```


```C++
    HW：
	// Loop 4: reduce results
	for(unsigned char bb=0; bb < HBS/bs_divisor; bb++){
		for(unsigned char m=0;m<Tm;m++){
			for(unsigned char t=0; t<Tn/HBS;t++){
				out_t[m]+=o_r[m][bb*(Tn/HBS)+t];
			}
		}
	}
```
It should be noted that the BS in BCM uses FFT-Hadamard-IFFT. This BS value does not need to be accumulated again. Therefore. When BS=4, in the Tn direction, only Tn/(BS/2)=Tn/2 values need to be accumulated
