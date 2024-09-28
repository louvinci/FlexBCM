## FlexBCM: Hybrid Block-Circulant Neural Network and Accelerator Co-Search on FPGAs

### Environment 
```conda env create -f pytorch2_0.yaml```
### Framework
Given the target model, FPGA specifications, and frames per second (FPS) settings, FlexBCM automatically generates tailored BCM-compressed CNNs and accelerators to balance model accuracy and hardware efficiency. It contains (a) the differentiable compressor and (b) the fast hardware evaluator. These components work together to enable the joint exploration of compression parameters (BS-1/4/8/16) and accelerator structures (tiling and parallelism factors).
### AutoSearch
Using the One-shot method in NAS to search subnets.

Step 1.  Enter the ```AutoSearch``` file folder. Run
```bash SuperNet_train.sh```

Step 2. After get the index, run the following command to train the subnet.
```bash SubNet_train.sh```

### Accelerator
Heterogeneous dual-core structure containing the ConvPU and BCMPU written using HLS.

