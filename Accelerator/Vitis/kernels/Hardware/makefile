#
# this file was created by a computer. trust it.
#

TARGET = hw
VPP_OPTS = --target $(TARGET)

# path to src
CIRCONV_SRC = $(ROOT_DIR)/../2DCNN/bn_qout
CONV_SRC = $(ROOT_DIR)/../ConvPE

#
# Hw kernel files
#

BUILD_SUBDIRS += build
HW_KERNEL_OBJS += build/circonv.xo
SRC_KERNEL_OBJS += build/circonv.xo
BUILD_SUBDIRS += build
HW_KERNEL_OBJS += build/ConvPE.xo
SRC_KERNEL_OBJS += build/ConvPE.xo

ALL_MESSAGE_FILES = $(subst .xo,.o,.mdb,$(SRC_KERNEL_OBJS))

#
# primary build targets
#

.PHONY: all clean
all: $(HW_KERNEL_OBJS)


clean:
	-$(RM) $(SRC_KERNEL_OBJS) $(ALL_MESSAGE_FILES) 
	-$(RMDIR) $(BUILD_SUBDIRS)
	-$(RMDIR) .Xil

.PHONY: incremental
incremental: all


nothing:

#
# Hw Kernels compile;
#

build/circonv.xo: $(CIRCONV_SRC)/circonv.cpp $(CIRCONV_SRC)/circonv.h $(CIRCONV_SRC)/compute.cpp $(CIRCONV_SRC)/compute.h $(CIRCONV_SRC)/compute_core.cpp $(CIRCONV_SRC)/compute_core.h $(CIRCONV_SRC)/config.h $(CIRCONV_SRC)/data_transfer.cpp $(CIRCONV_SRC)/data_transfer.h $(CIRCONV_SRC)/vb_fft.cpp $(CIRCONV_SRC)/vb_fft.h $(CIRCONV_SRC)/circonv.h circonv-compile.cfg
	-@mkdir -p $(@D)
	-@$(RM) $@
	$(VPP) $(VPP_OPTS) --compile -I"$(<D)" --kernel_frequency 200 --config circonv-compile.cfg -o"$@" "$<" "$(CIRCONV_SRC)/circonv.h" "$(CIRCONV_SRC)/compute.cpp" "$(CIRCONV_SRC)/compute.h" "$(CIRCONV_SRC)/compute_core.cpp" "$(CIRCONV_SRC)/compute_core.h" "$(CIRCONV_SRC)/config.h" "$(CIRCONV_SRC)/data_transfer.cpp" "$(CIRCONV_SRC)/data_transfer.h" "$(CIRCONV_SRC)/vb_fft.cpp" "$(CIRCONV_SRC)/vb_fft.h"

build/ConvPE.xo: $(CONV_SRC)/ConvPE.cpp $(CONV_SRC)/ConvPE.h ConvPE-compile.cfg
	-@mkdir -p $(@D)
	-@$(RM) $@
	$(VPP) $(VPP_OPTS) --compile -I"$(<D)" --kernel_frequency 200 --config ConvPE-compile.cfg -o"$@" "$<"

