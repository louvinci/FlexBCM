
open_project ConvPU
set_top ConvPE
add_files ConvPE.cpp
add_files ConvPE.h
add_files -tb test.cpp
open_solution "solution1" -flow_target vivado
set_part {xczu9eg-ffvb1156-2-e}
create_clock -period 5 -name default

csim_design -clean
csynth_design
cosim_design
export_design -format ip_catalog
