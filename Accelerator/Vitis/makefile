
# compiler tools
XILINX_VITIS ?= /tools/Xilinx/Vitis/2022.2
XILINX_VIVADO ?= /tools/Xilinx/Vivado/2022.2
XILINX_HLS ?= $(XILINX_VITIS)/Vivado_HLS

VPP ?= ${XILINX_VITIS}/bin/v++

RM = rm -f
RMDIR = rm -rf

# platform
VITIS_PLATFORM = xilinx_zcu102_base_202220_1
VITIS_PLATFORM_DIR = $(XILINX_VITIS)/base_platforms/xilinx_zcu102_base_202220_1
VITIS_PLATFORM_PATH = $(XILINX_VITIS)/base_platforms/xilinx_zcu102_base_202220_1/xilinx_zcu102_base_202220_1.xpfm

# path
ROOT_DIR=$(realpath $(dir $(lastword $(MAKEFILE_LIST))))
PL_PATH = $(ROOT_DIR)/kernels/Hardware
HW_LINK_PATH = $(ROOT_DIR)/hw_link/Hardware

export

.PHONY: all, clean
all: hw

clean:
	$(MAKE) -C $(HW_LINK_PATH) clean
	$(MAKE) -C $(PL_PATH) clean

hw: krnl
	$(MAKE) -C $(HW_LINK_PATH) all

krnl:
	$(MAKE) -C $(PL_PATH) all

