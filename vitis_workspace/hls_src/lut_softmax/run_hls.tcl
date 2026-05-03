open_project lut_softmax_prj
set_top lut_softmax
add_files lut_softmax.cpp
add_files lut_softmax.h
add_files -tb testbench.cpp
open_solution "solution1" -flow_target vitis
set_part {xczu7ev-ffvc1156-2-e}
create_clock -period 5 -name default
csim_design
csynth_design
exit
