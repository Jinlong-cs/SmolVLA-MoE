# Architecture Notes

## Claim

SmolVLA-MoE is designed as a general compact VLA framework, not a LIBERO-only policy. LIBERO is the first benchmark used to validate the architecture.

## Module Responsibilities

```text
Dense VLM backbone:
  Turns images and language into context tokens.

Flow matching:
  Defines the continuous action chunk generation target.

MoE action decoder:
  Allocates action-decoder capacity across shared and routed experts.
```

The flow objective and MoE structure are different layers of the design. Flow matching decides what the action decoder learns to predict. MoE decides how the decoder spends parameters and compute.

## Default MoE

The initial action FFN design is:

```text
Shared SwiGLU expert, always active
+ 4 routed SwiGLU experts
+ top-1 chunk-level router
+ load-balancing loss
+ router z-loss
```

Chunk-level routing is the conservative default because it keeps all tokens in one action chunk on the same routed expert, which should reduce temporal inconsistency compared with per-action-token routing.

## Benchmark-Agnostic Constraints

Do not hard-code:

- LIBERO suite id or task id
- fixed camera count
- fixed action dimension
- fixed action horizon
- benchmark-specific expert names

Allowed config-level variables:

- action dimension
- state dimension
- camera keys
- action horizon
- action normalization/statistics
- optional embodiment id in future adapters

## Required Ablations

Before claiming the MoE design is better, compare:

- Dense SmolVLA-style action expert.
- Same-active SmolVLA-MoE.
- Same-total dense action expert.
- MoE with and without shared expert.
- Top-1 and top-2 routing.
- Different flow sampling step counts.
