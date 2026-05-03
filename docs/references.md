# References

No external papers, model weights, or web pages were downloaded during this run.

The current PyTorch experiments rely on module-level approximations implemented locally in `pytorch_exp/`:

- Linear / projector fake quantization: FP32, FP16, INT8 symmetric fake quant, INT4 weight-only fake quant.
- Attention softmax approximation: stabilized exact softmax, LUT exp, PWL exp, Taylor 2/3 exp with clamp.
- GELU / RMSNorm approximation: tanh/PWL/LUT GELU, fake-quantized RMSNorm input, approximate reciprocal square root.

Main local conclusion: INT8 fake quantization is stable for Linear and projector module tests, INT4 weight-only shows larger error, and LUT exp is the most numerically stable softmax approximation in this run.
