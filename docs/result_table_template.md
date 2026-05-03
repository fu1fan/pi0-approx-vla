# Result Table Templates

## PyTorch Quantization Results

| experiment | shape | variant | device | MSE | MAE | max error | cosine similarity | latency mean ms | latency std ms | estimated weight size MB | remark |
|---|---|---|---|---:|---:|---:|---:|---:|---:|---:|---|
| linear_quant | batch=1,seq=256,in=1024,out=1024 | fp32 | cpu | | | | | | | | baseline |
| projector_quant | batch=1,tokens=256,in=1152,out=2048 | int8_fake_quant | cpu | | | | | | | | |

## Softmax Approximation Results

| shape | variant | device | MSE | MAE | KL divergence | max error | cosine similarity | latency mean ms | latency std ms | remark |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|---|
| heads=8,seq=128x128 | lut_exp_softmax | cpu | | | | | | | | |

## GELU Approximation Results

| shape | variant | device | MSE | MAE | max error | cosine similarity | latency mean ms | latency std ms | remark |
|---|---|---|---:|---:|---:|---:|---:|---:|---|
| 1x256x2048 | pwl_gelu | cpu | | | | | | | |

## HLS Synthesis Results

| kernel | latency cycles | II | estimated clock | LUT | FF | BRAM | DSP | remark |
|---|---:|---:|---:|---:|---:|---:|---:|---|
| int8_linear | | | | | | | | |
| fixed_projector | | | | | | | | |
| lut_softmax | | | | | | | | |
| gelu_pwl | | | | | | | | |
