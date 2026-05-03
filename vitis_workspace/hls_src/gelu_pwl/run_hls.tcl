open_project gelu_pwl_prj
set_top gelu_pwl
add_files gelu_pwl.cpp
add_files gelu_pwl.h
add_files -tb testbench.cpp
open_solution "solution1" -flow_target vitis
set_part {xczu7ev-ffvc1156-2-e}
create_clock -period 5 -name default
csim_design
csynth_design
exit
