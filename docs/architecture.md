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
+ top-k chunk-level router
+ load-balancing loss
+ router z-loss
```

Chunk-level routing is the conservative default because it keeps all tokens in one action chunk on the same routed expert, which should reduce temporal inconsistency compared with per-action-token routing.

## Official SmolVLA Residual-MoE Variant

The official-compatible variant starts from `HuggingFaceVLA/smolvla_libero` and keeps the official SmolVLA policy,
flow objective, state/action projections, KV-cache inference path, and action chunk semantics intact.

The only structural change is inside the action expert MLPs:

```text
original action-expert MLP
  -> original action-expert MLP + residual top-2 MoE adapter
```

Default training freezes the official dense path and trains only:

- top-2 MoE routers
- low-rank SwiGLU residual experts
- residual scale parameters

This is intentionally more conservative than replacing the whole action expert with a sparse decoder. It lets the
first official-based experiment answer whether sparse residual capacity can improve LIBERO while preserving the
released dense SmolVLA behavior at initialization.

## Top-2 Official-Scheduler Experiment

The current official-compatible branch default tests whether less brittle collaborative routing improves the
`libero_spatial` and `libero_10` regressions seen in the earlier top-1 experiment. It keeps the same 4 low-rank residual experts but
activates the two highest-probability experts per action chunk.

The training optimizer and scheduler are aligned to the official SmolVLA preset:

```text
AdamW lr=1e-4, betas=(0.9, 0.95), eps=1e-8, weight_decay=1e-10
grad clip norm=10
1000-step warmup, cosine decay over 30000 steps to 2.5e-6
```

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
- embodiment id in future adapters
