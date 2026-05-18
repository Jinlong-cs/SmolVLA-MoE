from __future__ import annotations

import os
from pathlib import Path
import time
from typing import Any

import torch
from torch.nn.parallel import DistributedDataParallel

from smolvla_moe.official.data import build_official_train_data
from smolvla_moe.official.policy import OfficialSmolVLAMoEPolicy
from smolvla_moe.official.policy import build_official_smolvla_moe_policy
from smolvla_moe.official.policy import count_parameters
from smolvla_moe.training.observability import JsonlLogger
from smolvla_moe.training.observability import WandbLogger
from smolvla_moe.training.observability import collect_resource_metrics
from smolvla_moe.training.schedulers import build_scheduler
from smolvla_moe.utils.checkpoint import load_checkpoint
from smolvla_moe.utils.checkpoint import save_checkpoint
from smolvla_moe.utils.seed import set_seed


def train_official_smolvla_moe(
    config: dict[str, Any],
    max_steps_override: int | None = None,
    resume_from: str | Path | None = None,
) -> None:
    ddp = _distributed_env()
    rank = ddp["rank"]
    world_size = ddp["world_size"]
    device = _resolve_device(config, ddp["local_rank"])
    set_seed(int(config.get("seed", 7)))
    _maybe_init_distributed(world_size, device)

    model = build_official_smolvla_moe_policy(config).to(device)
    total_params = count_parameters(model)
    trainable_params = count_parameters(model, trainable_only=True)
    if world_size > 1:
        model = DistributedDataParallel(
            model,
            device_ids=[ddp["local_rank"]] if device.type == "cuda" else None,
            find_unused_parameters=bool(config.get("train", {}).get("ddp_find_unused_parameters", False)),
        )

    data = build_official_train_data(config, rank=rank, world_size=world_size)
    iterator = iter(data)

    train_config = config.get("train", {})
    max_steps = int(max_steps_override or train_config.get("max_steps", 1000))
    optimizer = torch.optim.AdamW(
        (p for p in model.parameters() if p.requires_grad),
        lr=float(train_config.get("learning_rate", 1e-4)),
        betas=tuple(train_config.get("betas", (0.9, 0.999))),
        eps=float(train_config.get("eps", 1e-8)),
        weight_decay=float(train_config.get("weight_decay", 0.01)),
    )
    scheduler = build_scheduler(optimizer, train_config, max_steps)
    resume_step = 0
    if resume_from is not None:
        resume_step = load_checkpoint(resume_from, _unwrap(model), optimizer, scheduler, map_location=device)
        _set_sampler_epoch(data, resume_step)
    amp_dtype = _amp_dtype(train_config.get("amp_dtype"))
    use_amp = device.type == "cuda" and amp_dtype is not None
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp and amp_dtype == torch.float16)
    output_dir = Path(str(config.get("output_dir", "outputs/run")))
    if rank == 0:
        output_dir.mkdir(parents=True, exist_ok=True)
    save_every = int(train_config.get("save_every", 0))
    final_checkpoint = bool(train_config.get("save_final_checkpoint", True))
    log_every = int(train_config.get("log_every", 20))
    grad_clip_norm = float(train_config.get("grad_clip_norm", 0.0))
    jsonl_logger = JsonlLogger(output_dir) if rank == 0 else None
    wandb_logger = WandbLogger(config, output_dir, rank)
    last_log_time = time.time()

    if rank == 0:
        run_url = f" wandb_url={wandb_logger.url}" if wandb_logger.url else ""
        print(
            "device=%s world_size=%d total_params=%s trainable_params=%s patched_layers=%d resume_step=%d%s"
            % (
                device,
                world_size,
                f"{total_params:,}",
                f"{trainable_params:,}",
                len(_unwrap(model).patched_layers),
                resume_step,
                run_url,
            )
        )

    for step in range(resume_step + 1, max_steps + 1):
        try:
            batch = next(iterator)
        except StopIteration:
            _set_sampler_epoch(data, step)
            iterator = iter(data)
            batch = next(iterator)
        batch = _move_batch(batch, device)

        optimizer.zero_grad(set_to_none=True)
        with torch.autocast(device_type=device.type, dtype=amp_dtype, enabled=use_amp):
            metrics = model(batch)
            loss = metrics["loss"]
        scaler.scale(loss).backward()
        if grad_clip_norm > 0:
            scaler.unscale_(optimizer)
            grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip_norm)
        else:
            grad_norm = _grad_norm(model.parameters(), device)
        scaler.step(optimizer)
        scaler.update()
        if scheduler is not None:
            scheduler.step()

        should_log = step == 1 or (log_every > 0 and step % log_every == 0)
        if should_log:
            reduced_metrics = _mean_metrics(metrics, world_size=world_size)
            reduced_grad_norm = _mean_tensor(_as_tensor(grad_norm, device), world_size=world_size)

        if rank == 0 and should_log:
            now = time.time()
            elapsed = max(now - last_log_time, 1e-6)
            last_log_time = now
            logged = _detach_metrics(reduced_metrics)
            logged["train/grad_norm"] = float(reduced_grad_norm.detach().cpu())
            logged["train/lr"] = float(optimizer.param_groups[0]["lr"])
            logged["performance/steps_per_sec"] = float(log_every / elapsed if step > 1 else 1.0 / elapsed)
            logged["performance/samples_per_sec"] = (
                logged["performance/steps_per_sec"] * int(train_config.get("batch_size", 1)) * world_size
            )
            logged["params/total"] = float(total_params)
            logged["params/trainable"] = float(trainable_params)
            logged.update(collect_resource_metrics())
            print(_format_metrics(step, max_steps, logged))
            if jsonl_logger is not None:
                jsonl_logger.log("train", step, logged)
            wandb_logger.log(logged, step=step)

        if rank == 0 and save_every > 0 and step % save_every == 0:
            checkpoint_path = output_dir / "checkpoints" / f"step_{step:06d}.pt"
            save_checkpoint(checkpoint_path, _unwrap(model), optimizer, step, config, scheduler)
            wandb_logger.log_checkpoint(checkpoint_path, step, aliases=[f"step-{step}"])

    if rank == 0 and final_checkpoint:
        final_path = output_dir / "checkpoints" / "final.pt"
        save_checkpoint(final_path, _unwrap(model), optimizer, max_steps, config, scheduler)
        wandb_logger.log_checkpoint(final_path, max_steps, aliases=["final", f"step-{max_steps}"])
    if rank == 0:
        wandb_logger.finish()
    _cleanup_distributed(world_size)


def _unwrap(model: torch.nn.Module) -> OfficialSmolVLAMoEPolicy:
    return model.module if isinstance(model, DistributedDataParallel) else model


def _move_batch(batch: dict[str, Any], device: torch.device) -> dict[str, Any]:
    moved: dict[str, Any] = {}
    for key, value in batch.items():
        moved[key] = value.to(device) if torch.is_tensor(value) else value
    return moved


def _set_sampler_epoch(data: Any, step: int) -> None:
    sampler = getattr(data, "sampler_for_epoch", None)
    if sampler is not None and hasattr(sampler, "set_epoch"):
        sampler.set_epoch(step)


def _distributed_env() -> dict[str, int]:
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    rank = int(os.environ.get("RANK", "0"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    return {"world_size": world_size, "rank": rank, "local_rank": local_rank}


def _maybe_init_distributed(world_size: int, device: torch.device) -> None:
    if world_size > 1 and not torch.distributed.is_initialized():
        backend = "nccl" if device.type == "cuda" else "gloo"
        torch.distributed.init_process_group(backend=backend)


def _cleanup_distributed(world_size: int) -> None:
    if world_size > 1 and torch.distributed.is_initialized():
        torch.distributed.destroy_process_group()


def _resolve_device(config: dict[str, Any], local_rank: int) -> torch.device:
    requested = str(config.get("device", "auto"))
    if requested == "auto":
        requested = "cuda" if torch.cuda.is_available() else "cpu"
    if requested == "cuda":
        torch.cuda.set_device(local_rank)
        return torch.device("cuda", local_rank)
    return torch.device(requested)


def _amp_dtype(name: Any) -> torch.dtype | None:
    if name is None:
        return None
    if str(name) == "bfloat16":
        return torch.bfloat16
    if str(name) == "float16":
        return torch.float16
    raise ValueError(f"Unsupported amp dtype: {name}")


def _mean_metrics(metrics: dict[str, torch.Tensor], world_size: int) -> dict[str, torch.Tensor]:
    if world_size <= 1 or not torch.distributed.is_initialized():
        return metrics
    reduced: dict[str, torch.Tensor] = {}
    for key, value in metrics.items():
        if not torch.is_tensor(value):
            continue
        tensor = value.detach().float().clone()
        torch.distributed.all_reduce(tensor, op=torch.distributed.ReduceOp.SUM)
        reduced[key] = tensor / float(world_size)
    return reduced


def _mean_tensor(tensor: torch.Tensor, world_size: int) -> torch.Tensor:
    if world_size <= 1 or not torch.distributed.is_initialized():
        return tensor
    reduced = tensor.detach().float().clone()
    torch.distributed.all_reduce(reduced, op=torch.distributed.ReduceOp.SUM)
    return reduced / float(world_size)


def _as_tensor(value: torch.Tensor | float, device: torch.device) -> torch.Tensor:
    if torch.is_tensor(value):
        return value.detach().to(device=device, dtype=torch.float32)
    return torch.tensor(float(value), device=device, dtype=torch.float32)


def _detach_metrics(metrics: dict[str, torch.Tensor]) -> dict[str, float]:
    result: dict[str, float] = {}
    for key, value in metrics.items():
        if key == "expert_usage":
            for idx, usage in enumerate(value.detach().float().cpu().tolist()):
                result[f"moe/expert_usage_{idx}"] = float(usage)
        elif torch.is_tensor(value) and value.ndim == 0:
            result[f"train/{key}"] = float(value.detach().cpu())
    return result


def _format_metrics(step: int, max_steps: int, metrics: dict[str, float]) -> str:
    keys = ["train/loss", "train/flow_loss", "train/load_balance_loss", "train/router_z_loss", "train/grad_norm"]
    summary = " ".join(f"{key}={metrics[key]:.5f}" for key in keys if key in metrics)
    return f"step={step}/{max_steps} {summary}"


def _grad_norm(parameters: Any, device: torch.device) -> torch.Tensor:
    norms = []
    for param in parameters:
        if param.grad is not None:
            norms.append(param.grad.detach().norm(2))
    if not norms:
        return torch.zeros((), device=device)
    return torch.stack(norms).norm(2)
