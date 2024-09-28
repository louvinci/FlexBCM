
open_project BCMPU
set_top circonv
add_files circonv.cpp
add_files circonv.h
add_files compute.cpp
add_files compute.h
add_files compute_core.cpp
add_files compute_core.h
add_files data_transfer.cpp
add_files data_transfer.h
add_files fft.cpp
add_files fft.h
add_files -tb utils.cpp
add_files -tb utils.h
add_files -tb main.cpp
add_files -tb main.h
open_solution "solution1" -flow_target vivado
set_part {xczu9eg-ffvb1156-2-e}
create_clock -period 5 -name default

csim_design -clean
csynth_design
cosim_design
export_design -format ip_catalog
