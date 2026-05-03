import argparse
import csv
import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/pi0_approx_vla_matplotlib")
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F

from common.quant_utils import resolve_device


class ToyFlowNet(nn.Module):
    def __init__(self, action_dim=8, cond_dim=16, hidden_dim=128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(action_dim + cond_dim + 8, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, action_dim),
        )

    def forward(self, x_t, t, cond):
        t_features = torch.cat(
            [
                t,
                t * t,
                torch.sin(3.14159265 * t),
                torch.cos(3.14159265 * t),
                torch.sin(2 * 3.14159265 * t),
                torch.cos(2 * 3.14159265 * t),
                torch.sqrt(t.clamp_min(1e-6)),
                torch.log1p(t),
            ],
            dim=-1,
        )
        return self.net(torch.cat([x_t, t_features, cond], dim=-1))


def make_clean_action(cond, action_dim):
    chunks = cond.chunk(2, dim=-1)
    base = torch.tanh(chunks[0][..., :action_dim])
    mod = 0.25 * torch.sin(chunks[1][..., :action_dim])
    return base + mod


def sample_batch(batch_size, action_dim, cond_dim, device):
    cond = torch.randn(batch_size, cond_dim, device=device)
    x0 = torch.randn(batch_size, action_dim, device=device)
    x1 = make_clean_action(cond, action_dim)
    t = torch.rand(batch_size, 1, device=device)
    x_t = (1.0 - t) * x0 + t * x1
    velocity = x1 - x0
    return x_t, t, cond, velocity


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda", "mps"])
    parser.add_argument("--steps", type=int, default=400)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    device = resolve_device(args.device)
    torch.manual_seed(args.seed)
    action_dim = 8
    cond_dim = 16
    model = ToyFlowNet(action_dim=action_dim, cond_dim=cond_dim).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=2e-3, weight_decay=1e-4)

    rows = []
    for step in range(1, args.steps + 1):
        x_t, t, cond, velocity = sample_batch(args.batch_size, action_dim, cond_dim, device)
        pred = model(x_t, t, cond)
        loss = F.mse_loss(pred, velocity)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()

        if step == 1 or step % 10 == 0:
            rows.append({"step": step, "loss": loss.item(), "device": device})

    out_csv = Path("results/csv/toy_flow_matching.csv")
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    with out_csv.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["step", "loss", "device"])
        writer.writeheader()
        writer.writerows(rows)

    out_fig = Path("results/figures/toy_flow_matching_curve.png")
    out_fig.parent.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(7, 4))
    plt.plot([row["step"] for row in rows], [row["loss"] for row in rows], marker="o", markersize=3)
    plt.xlabel("step")
    plt.ylabel("velocity MSE")
    plt.title("Toy Flow Matching Training Curve")
    plt.tight_layout()
    plt.savefig(out_fig, dpi=160)
    plt.close()

    print(f"wrote {out_csv}")
    print(f"wrote {out_fig}")


if __name__ == "__main__":
    main()
