open_project fixed_projector_prj
set_top fixed_projector
add_files fixed_projector.cpp
add_files fixed_projector.h
add_files -tb testbench.cpp
open_solution "solution1" -flow_target vitis
set_part {xczu7ev-ffvc1156-2-e}
create_clock -period 5 -name default
csim_design
csynth_design
exit
