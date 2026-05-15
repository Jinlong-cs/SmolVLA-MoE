# VastAI Runbook Draft

This runbook is intentionally conservative. Fill actual machine paths, W&B URLs, and checkpoint paths into the `vla-train` workspace before and after launching jobs.

## Setup

```bash
cd /workspace
git clone <repo-url> SmolVLA-MoE
cd /workspace/SmolVLA-MoE
uv venv --python 3.12 .venv
source .venv/bin/activate
uv pip install -e .
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

## Official SmolVLA Residual-MoE Training

```bash
export HF_HOME=/workspace/.hf_home
export HF_HUB_ENABLE_HF_TRANSFER=1
export WANDB_API_KEY=...
export WANDB_MODE=online

torchrun --standalone --nproc_per_node=8 scripts/train_official_smolvla_moe.py \
  --config configs/train/official_smolvla_moe_libero_8gpu.yaml
```

This run loads `HuggingFaceVLA/smolvla_libero` first, wraps official action-expert MLPs with residual top-1 MoE
adapters, and freezes the official dense checkpoint by default.

Eval this branch with the official server script:

```bash
OPENPI_ROOT=/workspace/openpi \
HF_HOME=/workspace/.hf_home \
MUJOCO_GL=egl \
PYOPENGL_PLATFORM=egl \
PYTHONPATH=/workspace/SmolVLA-MoE/src:/workspace/openpi/src:/workspace/openpi/packages/openpi-client/src:/workspace/openpi/third_party/libero \
/workspace/openpi/.venv/bin/python scripts/eval_libero.py \
  --checkpoint outputs/libero/official_smolvla_moe_residual/checkpoints/final.pt \
  --config configs/train/official_smolvla_moe_libero_8gpu.yaml \
  --server-script scripts/serve_official_smolvla_moe_policy.py \
  --suite all \
  --num-trials 50 \
  --replan-steps 1 \
  --output-dir outputs/libero/eval/official_smolvla_moe_residual_final \
  --server-python /workspace/openpi/.venv/bin/python \
  --libero-python /workspace/openpi/examples/libero/.venv/bin/python
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

## LIBERO Eval With Videos

```bash
OPENPI_ROOT=/workspace/openpi \
HF_HOME=/workspace/.hf_home \
MUJOCO_GL=egl \
PYOPENGL_PLATFORM=egl \
PYTHONPATH=/workspace/SmolVLA-MoE/src:/workspace/openpi/src:/workspace/openpi/packages/openpi-client/src:/workspace/openpi/third_party/libero \
/workspace/openpi/.venv/bin/python scripts/eval_libero.py \
  --checkpoint outputs/libero/smolvla_moe_full_finetune/checkpoints/final.pt \
  --config configs/train/libero_8gpu.yaml \
  --suite all \
  --num-trials 50 \
  --output-dir outputs/libero/eval/final \
  --server-python /workspace/openpi/.venv/bin/python \
  --libero-python /workspace/openpi/examples/libero/.venv/bin/python
```

The eval wrapper writes `summary.json`, `video_manifest.json`, `videos.html`, and per-suite MP4 files under `outputs/libero/eval/final/`. Use `--no-save-videos` for metric-only runs.

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
