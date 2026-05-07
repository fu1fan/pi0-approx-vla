open_project exact_softmax_hls
set_top exact_softmax_kernel
add_files kernel.cpp
add_files -tb tb.cpp
open_solution "solution1" -flow_target vivado
set_part {xcvu9p-flga2104-2-i}
create_clock -period 7.3 -name default
csim_design
csynth_design
exit
