import torch


def tensor_metrics(reference, candidate, include_kl=False):
    ref = reference.detach().float().reshape(-1)
    out = candidate.detach().float().reshape(-1)
    diff = out - ref
    metrics = {
        "mse": torch.mean(diff * diff).item(),
        "mae": torch.mean(torch.abs(diff)).item(),
        "max_error": torch.max(torch.abs(diff)).item(),
        "cosine_similarity": torch.nn.functional.cosine_similarity(ref, out, dim=0).item(),
    }
    if include_kl:
        eps = 1e-8
        ref_p = reference.detach().float().clamp_min(eps)
        out_p = candidate.detach().float().clamp_min(eps)
        metrics["kl_divergence"] = torch.sum(ref_p * (torch.log(ref_p) - torch.log(out_p)), dim=-1).mean().item()
    return metrics
