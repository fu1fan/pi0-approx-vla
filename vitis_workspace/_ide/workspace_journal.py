# 2026-05-06T22:38:57.660756529
import vitis

client = vitis.create_client()
client.set_workspace(path="vitis_workspace")

comp = client.create_hls_component(name = "int8_gemm",cfg_file = ["hls_config.cfg"],template = "empty_hls_component")

comp = client.create_hls_component(name = "lut_softmax",cfg_file = ["hls_config.cfg"],template = "empty_hls_component")

comp = client.create_hls_component(name = "gelu_pwl",cfg_file = ["hls_config.cfg"],template = "empty_hls_component")

comp = client.create_hls_component(name = "rmsnorm_rsqrt",cfg_file = ["hls_config.cfg"],template = "empty_hls_component")

cfg = client.get_config_file(path="/home/fu1fan/Develop/PROJECTS/pi0-approx-vla/vitis_workspace/gelu_pwl/hls_config.cfg")

cfg.set_values(key="syn.file", values=["main.cpp"])

cfg.set_values(key="tb.file", values=["test.cpp"])

cfg = client.get_config_file(path="/home/fu1fan/Develop/PROJECTS/pi0-approx-vla/vitis_workspace/int8_gemm/hls_config.cfg")

cfg.set_values(key="syn.file", values=["main.cpp"])

cfg.set_values(key="tb.file", values=["test.cpp"])

cfg = client.get_config_file(path="/home/fu1fan/Develop/PROJECTS/pi0-approx-vla/vitis_workspace/lut_softmax/hls_config.cfg")

cfg.set_values(key="syn.file", values=["main.cpp"])

cfg.set_values(key="tb.file", values=["test.cpp"])

cfg = client.get_config_file(path="/home/fu1fan/Develop/PROJECTS/pi0-approx-vla/vitis_workspace/rmsnorm_rsqrt/hls_config.cfg")

cfg.set_values(key="syn.file", values=["main.cpp"])

cfg.set_values(key="tb.file", values=["test.cpp"])

