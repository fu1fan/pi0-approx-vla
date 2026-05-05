# pi0-shape Toy Flow Step Reduction Summary

This is a toy flow matching benchmark with action chunk shape aligned to pi0 (`action_horizon=50`, `action_dim=32`). It does not run real pi0 and does not represent real action quality.

- device: `cuda`
- train steps: 1500
- initial/final sampled training loss: 1.227344 -> 1.008992
- 2-step speedup vs 10-step: 4.460x
- 2-step MSE vs 10-step output: 4.319841e-03

Interpretation: this only illustrates the latency-error tradeoff of reducing flow integration steps under pi0-like action shape. It cannot predict real robot task success or real pi0 action quality.

Outputs:
- `results/csv/pi0_shape_flow_step_reduction.csv`
- `results/figures/pi0_shape_flow_step_reduction_error.png`
- `results/figures/pi0_shape_flow_step_reduction_latency.png`