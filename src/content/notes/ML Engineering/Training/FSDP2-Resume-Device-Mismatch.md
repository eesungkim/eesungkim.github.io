---
title: FSDP2 Checkpoint Resume Crashes on the First Training Step (cuda:0 vs cuda:1)
description: Why resuming from an FSDP2/DCP sharded checkpoint passes validation but crashes the first optimizer step with "Expected all tensors to be on the same device", how to diagnose it DTensor-safely, and when to use weights-only loading instead of a full resume
tags: [pytorch, fsdp2, dtensor, lightning, nemo, checkpoint, debugging]
date: 2026-07-19
---

**Environment:** NeMo speechlm2 (SALM) · PyTorch FSDP2 / DTensor · Lightning `ModelParallelStrategy` · `torchrun` multi-GPU

**TL;DR** — Pointing `resume_if_exists` at an FSDP2 (DCP) checkpoint restores the **full optimizer state**. If the mesh or world size differs from when the checkpoint was saved, resharding can materialize some of that state on `cuda:0` while rank *N*'s parameters live on `cuda:N`. Validation is forward-only and never touches optimizer state, so it passes; the first `optimizer.step()` with `foreach` AdamW calls `torch._foreach_*`, which requires all tensors on the same device, and crashes. If the goal is "fine-tune from pretrained weights", use a **weights-only load** (`init_from_checkpoint`) instead of a full resume.

## Symptom

Training starts from an FSDP2 sharded (DCP) checkpoint via `exp_manager.resume_if_exists: true`. Startup is normal, the sanity validation step passes, then the first real training step raises:

```
RuntimeError: Expected all tensors to be on the same device,
but found at least two devices, cuda:0 and cuda:1!
```

The traceback is misleading: its top frame points at an innocent line in the training script (`datamodule = DataModule(...)`) — merely where the object handed to `trainer.fit()` was created. The exception is raised much deeper, inside the optimizer step.

## Why "validation passes" narrows it down

With `num_sanity_val_steps: 1`, Lightning's execution order is:

1. Sanity validation step — **forward only**. No backward, no optimizer.
2. First training step — the first time `backward()` + `optimizer.step()` run.

"Validation works but the next step crashes" therefore means *"forward works but the first optimizer step crashes."* If parameters were on the wrong device, forward would have died first. The suspects are **tensors forward never touches** — AdamW's moment states `exp_avg` / `exp_avg_sq`.

The error's phrasing is itself a clue: `Expected all tensors to be on the same device` is thrown by the `torch._foreach_*` kernel family, and the optimizer was on that path:

```yaml
optimizer:
  _target_: torch.optim.AdamW
  lr: 5e-4
  betas: [0.9, 0.98]
  weight_decay: 1e-3
  foreach: true   # <- batches params and states into fused-list kernels
```

With `foreach: true`, AdamW hands parameters and states as lists to batched kernels like `torch._foreach_add_`, which require **every tensor in the list on the same device**.

## Root cause

`resume_if_exists` is designed to **continue an interrupted run**: it restores not just weights but the entire training state — optimizer moments, LR scheduler, step counter.

An FSDP2 checkpoint is in DCP (distributed checkpoint) format: tensors are sharded according to the device mesh at save time. Relaunching with the **same world size and mesh** works — each rank pulls its own shard onto its own GPU. But pointing it at a checkpoint from a *different* run (different GPU count or strategy) forces DCP to **reshard**, and redistributed optimizer state can be materialized on `cuda:0` instead of each rank's own GPU:

```
rank 0 (GPU cuda:0)                 rank 1 (GPU cuda:1)
─────────────────────               ─────────────────────
param local shard  cuda:0  OK       param local shard  cuda:1  OK
exp_avg (restored) cuda:0  OK       exp_avg (restored) cuda:0  MISMATCH ✗

rank 1: optimizer.step()
  → torch._foreach_add_([param@cuda:1, …], [exp_avg@cuda:0, …])
  → RuntimeError
```

Rank 0 survives by accident — the misplaced device happens to be its own. This bug only fires on ranks ≥ 1.

Causal chain: **resume restores optimizer state** → **resharding lands some state on `cuda:0`** → **forward never reads it, so validation passes** → **the first `foreach` step explodes**.

## Diagnostic (DTensor-safe)

Compare param and state devices right before the first training step. FSDP2 trap: params and states are **DTensors**, and `.device` can report the *logical* device. The crash happens at the **local shard** level, so compare `.to_local().device`:

```python
# temporarily inserted at the top of training_step
if batch_idx == 0:
    from collections import Counter
    from torch.distributed.tensor import DTensor

    def _dev(t):
        # For DTensor, look at the physical device of the local shard
        return t.to_local().device if isinstance(t, DTensor) else t.device

    opt = self.trainer.optimizers[0]
    pdev, sdev, mism = Counter(), Counter(), 0
    for g in opt.param_groups:          # all groups, not just group 0
        for p in g["params"]:
            st = opt.state.get(p, {})
            pd = _dev(p)
            pdev[str(pd)] += 1
            if "exp_avg" in st:
                ed = _dev(st["exp_avg"])
                sdev[str(ed)] += 1
                mism += int(ed != pd)
    print(f"[rank{self.global_rank}] params={dict(pdev)} "
          f"exp_avg={dict(sdev)} mismatches={mism}", flush=True)
```

Execution order is `training_step` (this print) → `backward` → `optimizer.step()` (crash), so the log lands right before the death. Confirmed output looks like:

```
[rank1] params={'cuda:1': 87} exp_avg={'cuda:0': 20, 'cuda:1': 67} mismatches=20
```

`exp_avg={}` with `mismatches=0` means no optimizer state exists (fresh optimizer) — different bug.

## Fix

"Resuming a run" and "starting from weights" are different features:

| Goal | Mechanism | Restores optimizer? | Requirements |
|---|---|---|---|
| Continue the **same interrupted run** | `resume_if_exists` | Yes | Same world size · same mesh · same trainable set |
| Start a **new run** from pretrained weights | `model.init_from_checkpoint` | No (fresh) | None — DCP dir, HF dir, or `.ckpt` all supported |

```yaml
model:
  init_from_checkpoint: /path/to/fsdp2_ckpt_dir   # DCP directory (has .metadata)

exp_manager:
  resume_if_exists: false   # never create the misplaced optimizer state at all
```

This path (NeMo speechlm2's `init_from_training_checkpoint`) loads only the model `state_dict` via DCP, **in-place into parameters already on each rank's GPU** — resharding has no opportunity to misplace anything, and the optimizer is built fresh on the correct device.

**Non-fixes:** switching to `apex FusedAdam` or setting `foreach: false`. Fused kernels and the single-tensor path both require the same device alignment, so they crash in the same place — or worse, apex doesn't understand DTensor and can silently misbehave. Swapping the kernel relocates the symptom; fixing the state's device is the cure. (For fused, the DTensor-aware option is `torch.optim.AdamW(fused=True)`.)

## LR settings for the fresh-optimizer restart

**Anchor the LR to where the checkpoint stopped.** The model is already on a convergence trajectory; re-heating to a from-scratch peak LR causes early loss spikes. Set the new schedule's peak near the LR the old schedule had at that checkpoint — extract it directly:

```python
from torch.distributed.checkpoint.format_utils import dcp_to_torch_save
import torch

dcp_to_torch_save("path/to/fsdp2_ckpt_dir", "flat.pt")
ck = torch.load("flat.pt", map_location="cpu", weights_only=False)
print(ck["optimizer_states"][0]["param_groups"][0]["lr"])
```

**Always warm up, even briefly.** A fresh Adam starts with zero second moments; with `β₂ = 0.98` they normalize in ~50 steps, so a 500–1000 step warmup absorbs the instability completely. Transplanting Adam moments from the checkpoint means re-implementing DTensor resharding alignment by hand — that is, re-implementing this bug. Don't.

## Takeaways

1. **"Validation passes, first training step crashes" points at the optimizer.** Suspect tensors forward never touches — moment states, gradient buffers — first.
2. **The top frame of a stack trace can be where execution passed through, not where the exception lives.** The error message's phrasing (`_foreach_*`'s same-device wording) was a more precise clue than the frame.
3. **A full-state resume is only safe under its contract: same run, same mesh.** Outside that contract, load weights only.
4. **When debugging FSDP2, read `.to_local().device`, not `.device`.** A DTensor's logical device can hide the physical location of the local shard.
