import torch
import torch.nn.functional as F


def resolve_device(name):
    if name == "auto":
        if torch.cuda.is_available():
            return "cuda"
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return "mps"
        return "cpu"
    if name == "cuda" and torch.cuda.is_available():
        return "cuda"
    if name == "mps" and hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def fake_quant_symmetric(x, num_bits=8, eps=1e-8):
    qmax = (2 ** (num_bits - 1)) - 1
    qmin = -qmax - 1
    scale = x.detach().abs().max().clamp_min(eps) / qmax
    q = torch.clamp(torch.round(x / scale), qmin, qmax)
    return q * scale, scale


def fake_quant_weight_per_out_channel(w, num_bits=4, eps=1e-8):
    qmax = (2 ** (num_bits - 1)) - 1
    qmin = -qmax - 1
    scales = w.detach().abs().amax(dim=1, keepdim=True).clamp_min(eps) / qmax
    q = torch.clamp(torch.round(w / scales), qmin, qmax)
    return q * scales, scales


def linear_fp16(x, w, b):
    try:
        return F.linear(x.half(), w.half(), b.half()).float()
    except RuntimeError:
        return F.linear(x.float(), w.half().float(), b.half().float())


def linear_int8_fake_quant(x, w, b):
    xq, _ = fake_quant_symmetric(x, 8)
    wq, _ = fake_quant_symmetric(w, 8)
    return F.linear(xq, wq, b)


def linear_int4_weight_only_fake_quant(x, w, b):
    wq, _ = fake_quant_weight_per_out_channel(w, 4)
    return F.linear(x, wq, b)
