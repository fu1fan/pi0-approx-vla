open_project int8_linear_prj
set_top int8_linear
add_files int8_linear.cpp
add_files int8_linear.h
add_files -tb testbench.cpp
open_solution "solution1" -flow_target vitis
set_part {xczu7ev-ffvc1156-2-e}
create_clock -period 5 -name default
csim_design
csynth_design
exit
