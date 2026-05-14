# VastAI Runbook Draft

This runbook is intentionally conservative. Fill actual machine paths, W&B URLs, and checkpoint paths into the `vla-train` workspace before and after launching jobs.

## Setup

```bash
cd /workspace
git clone <repo-url> SmolVLA-MoE
cd /workspace/SmolVLA-MoE
python -m venv .venv
source .venv/bin/activate
pip install -U pip
pip install -e ".[hf,train]"
python -m compileall src scripts
```

## LIBERO Training Skeleton

```bash
export HF_HOME=/workspace/.hf_home
export HF_HUB_ENABLE_HF_TRANSFER=1
export WANDB_API_KEY=...
export WANDB_MODE=online

torchrun --standalone --nproc_per_node=8 scripts/train.py \
  --config configs/train/libero_8gpu.yaml
```

The default W&B run is configured as:

```text
project: smolvla-moe-libero
name: smolvla_moe_full_finetune_libero
group: libero_8gpu_full_finetune
mode: online
```

The script writes the W&B id to:

```text
outputs/libero/smolvla_moe_full_finetune/wandb_id.txt
```

If the same output directory is reused, `resume: allow` will reconnect to the existing W&B run. Use a new `output_dir` or delete that id file when starting a truly new run.

## Required Records

Record these in `workspaces/smolvla-moe-libero/`:

- repo commit and dirty state
- exact train command
- dataset path or HF cache path
- GPU type/count and per-GPU/global batch
- W&B URL
- `metrics.jsonl` path
- checkpoint path
- train loss and router expert usage
- LIBERO closed-loop suite success after eval is wired
