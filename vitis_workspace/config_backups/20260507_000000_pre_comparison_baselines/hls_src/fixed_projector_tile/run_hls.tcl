open_project fixed_projector_tile_prj -reset
set_top fixed_projector_tile_kernel
add_files kernel.cpp
add_files -tb tb.cpp
open_solution solution1 -reset
set_part {xcvu9p-flga2104-2-i}
create_clock -period 5 -name default
csim_design
csynth_design
exit
