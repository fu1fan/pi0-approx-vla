import argparse
import csv
import os
import time
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/pi0_approx_vla_matplotlib")
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F

from common.pi0_bench import append_research_log, resolve_device, set_seed, tensor_metrics_with_relative_l2
from common.timer import synchronize


class VelocityNet(nn.Module):
    def __init__(self, action_dim_flat, cond_dim, hidden_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(action_dim_flat + cond_dim + 8, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, action_dim_flat),
        )

    def forward(self, x_t, t, cond):
        t_feat = torch.cat(
            [
                t,
                t * t,
                torch.sin(torch.pi * t),
                torch.cos(torch.pi * t),
                torch.sin(2 * torch.pi * t),
                torch.cos(2 * torch.pi * t),
                torch.sqrt(t.clamp_min(1e-6)),
                torch.log1p(t),
            ],
            dim=-1,
        )
        return self.net(torch.cat([x_t, t_feat, cond], dim=-1))


def make_clean_action(cond, action_horizon, action_dim):
    b = cond.shape[0]
    flat_dim = action_horizon * action_dim
    repeats = (flat_dim + cond.shape[1] - 1) // cond.shape[1]
    a = cond.repeat(1, repeats)[:, :flat_dim].reshape(b, action_horizon, action_dim)
    c = cond[:, -action_dim:].unsqueeze(1)
    clean = torch.tanh(0.6 * a) + 0.15 * torch.sin(c)
    return clean.reshape(b, action_horizon * action_dim)


def sample_batch(batch_size, action_horizon, action_dim, cond_dim, device):
    cond = torch.randn(batch_size, cond_dim, device=device)
    x0 = torch.randn(batch_size, action_horizon * action_dim, device=device)
    x1 = make_clean_action(cond, action_horizon, action_dim)
    t = torch.rand(batch_size, 1, device=device)
    x_t = (1.0 - t) * x0 + t * x1
    target_v = x1 - x0
    return x_t, t, cond, target_v, x0, x1


@torch.no_grad()
def integrate(model, x0, cond, steps):
    x = x0.clone()
    dt = 1.0 / steps
    for i in range(steps):
        t = torch.full((x.shape[0], 1), i / steps, device=x.device)
        x = x + dt * model(x, t, cond)
    return x


def timed_integrate(model, x0, cond, steps, repeat, device):
    for _ in range(3):
        integrate(model, x0, cond, steps)
    synchronize(device)
    times = []
    out = None
    for _ in range(repeat):
        start = time.perf_counter()
        out = integrate(model, x0, cond, steps)
        synchronize(device)
        times.append((time.perf_counter() - start) * 1000)
    t = torch.tensor(times)
    return out, t.mean().item(), t.std(unbiased=False).item()


def run(args):
    device = resolve_device(args.device)
    set_seed(args.seed)
    append_research_log(f"[pi0_shape_flow_step_reduction] start device={device}, train_steps={args.train_steps}, hidden_dim={args.hidden_dim}.")
    action_horizon = 50
    action_dim = 32
    cond_dim = 1024
    flat_dim = action_horizon * action_dim
    model = VelocityNet(flat_dim, cond_dim, args.hidden_dim).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)

    losses = []
    model.train()
    for step in range(1, args.train_steps + 1):
        x_t, t, cond, target_v, _, _ = sample_batch(args.batch_size, action_horizon, action_dim, cond_dim, device)
        pred = model(x_t, t, cond)
        loss = F.mse_loss(pred, target_v)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        if step == 1 or step % 25 == 0:
            losses.append((step, loss.item()))

    model.eval()
    _, _, cond, _, x0, clean = sample_batch(args.eval_batch_size, action_horizon, action_dim, cond_dim, device)
    baseline, base_mean, base_std = timed_integrate(model, x0, cond, 10, args.eval_repeat, device)
    rows = []
    for steps in [10, 8, 6, 4, 2]:
        out, mean_ms, std_ms = timed_integrate(model, x0, cond, steps, args.eval_repeat, device)
        ref = baseline if steps != 10 else clean
        metrics = tensor_metrics_with_relative_l2(ref, out)
        rows.append(
            {
                "experiment": "pi0_shape_flow_step_reduction",
                "steps": steps,
                "baseline": "clean_action" if steps == 10 else "10_step_output",
                "device": device,
                "action_horizon": action_horizon,
                "action_dim": action_dim,
                "cond_dim": cond_dim,
                "hidden_dim": args.hidden_dim,
                "train_steps": args.train_steps,
                "latency_mean_ms": mean_ms,
                "latency_std_ms": std_ms,
                "speedup_vs_10_step": base_mean / mean_ms,
                **metrics,
            }
        )

    out_csv = Path("results/csv/pi0_shape_flow_step_reduction.csv")
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    with out_csv.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    print(f"wrote {out_csv}")

    fig_error = Path("results/figures/pi0_shape_flow_step_reduction_error.png")
    fig_error.parent.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(6, 4))
    plt.plot([r["steps"] for r in rows], [r["mse"] for r in rows], marker="o")
    plt.gca().invert_xaxis()
    plt.xlabel("Euler steps")
    plt.ylabel("MSE vs baseline")
    plt.title("pi0-shape toy flow step reduction error")
    plt.tight_layout()
    plt.savefig(fig_error, dpi=160)
    plt.close()
    print(f"wrote {fig_error}")

    fig_latency = Path("results/figures/pi0_shape_flow_step_reduction_latency.png")
    plt.figure(figsize=(6, 4))
    plt.plot([r["steps"] for r in rows], [r["latency_mean_ms"] for r in rows], marker="o")
    plt.xlabel("Euler steps")
    plt.ylabel("latency_mean_ms")
    plt.title("pi0-shape toy flow step reduction latency")
    plt.tight_layout()
    plt.savefig(fig_latency, dpi=160)
    plt.close()
    print(f"wrote {fig_latency}")

    summary = Path("results/pi0_shape_flow_step_reduction_summary.md")
    summary.write_text(
        "\n".join(
            [
                "# pi0-shape Toy Flow Step Reduction Summary",
                "",
                "This is a toy flow matching benchmark with action chunk shape aligned to pi0 (`action_horizon=50`, `action_dim=32`). It does not run real pi0 and does not represent real action quality.",
                "",
                f"- device: `{device}`",
                f"- train steps: {args.train_steps}",
                f"- initial/final sampled training loss: {losses[0][1]:.6f} -> {losses[-1][1]:.6f}",
                f"- 2-step speedup vs 10-step: {next(r['speedup_vs_10_step'] for r in rows if r['steps'] == 2):.3f}x",
                f"- 2-step MSE vs 10-step output: {next(r['mse'] for r in rows if r['steps'] == 2):.6e}",
                "",
                "Interpretation: this only illustrates the latency-error tradeoff of reducing flow integration steps under pi0-like action shape. It cannot predict real robot task success or real pi0 action quality.",
                "",
                "Outputs:",
                "- `results/csv/pi0_shape_flow_step_reduction.csv`",
                "- `results/figures/pi0_shape_flow_step_reduction_error.png`",
                "- `results/figures/pi0_shape_flow_step_reduction_latency.png`",
            ]
        )
    )
    print(f"wrote {summary}")
    append_research_log(f"[pi0_shape_flow_step_reduction] completed rows={len(rows)}, csv={out_csv}.")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda", "mps"])
    parser.add_argument("--train-steps", type=int, default=500)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--eval-batch-size", type=int, default=64)
    parser.add_argument("--eval-repeat", type=int, default=20)
    parser.add_argument("--hidden-dim", type=int, default=512)
    parser.add_argument("--lr", type=float, default=2e-3)
    parser.add_argument("--seed", type=int, default=345)
    args = parser.parse_args()
    run(args)


if __name__ == "__main__":
    main()
