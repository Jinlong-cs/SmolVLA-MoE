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

## Official SmolVLA Residual-MoE Variant

The official-compatible variant starts from `HuggingFaceVLA/smolvla_libero` and keeps the official SmolVLA policy,
flow objective, state/action projections, KV-cache inference path, and action chunk semantics intact.

The only structural change is inside the action expert MLPs:

```text
original action-expert MLP
  -> original action-expert MLP + residual top-k MoE adapter
```

Default training freezes the official dense path and trains only:

- MoE routers
- low-rank SwiGLU residual experts
- residual scale parameters

This is intentionally more conservative than replacing the whole action expert with a sparse decoder. It lets the
first official-based experiment answer whether sparse residual capacity can improve LIBERO while preserving the
released dense SmolVLA behavior at initialization.

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
