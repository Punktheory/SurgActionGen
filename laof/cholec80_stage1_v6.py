"""
LAOF Stage 1 v6 — Flow AutoEncoder on Cholec80

Why v6 instead of IDM-based approaches (v1-v5):
  All IDM variants collapsed because surgical video adjacent frames are
  near-identical. IDM has no direct pixel-level supervision signal, so it
  always finds the shortcut: output a constant z.

  FlowAutoEncoder (FAE) uses optical flow obs_of[t] as both input and target.
  Since every frame has a different flow map (already computed by RAFT and
  stored in NPZ chunks), the encoder is forced to produce different z for
  different frames. Collapse is structurally impossible.

Architecture:
  FAE encoder:  obs_of[t] (3,128,128) → 128-dim z_t
  FAE decoder:  z_t → reconstructed flow (3,64,64) [internal resolution]
  WM:           (zeros, z_t) → frame-diff prediction
  FM:           z_t → flow reconstruction (same as FAE but separate weights)

Training:
  loss = fae_recon + wm_flowdiff + fm_flow
  FAE is trained purely on flow reconstruction — cannot collapse.
  WM/FM use z.detach() so they don't interfere with FAE.

Checkpoint keys: "fae", "wm", "fm", "step"

Monitor for collapse: z_std should converge to ~0.088.
  If z_std → 0:   F.normalize causing constant output (collapse).
  If z_std >> 0.088: code bug.

Launch (8 GPUs, from project root):
  CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 torchrun \\
      --nproc_per_node=8 --master_port=29505 \\
      laof/cholec80_stage1_v6.py \\
      --steps 50000 --bs 64 --lr 3e-4 \\
      --out_dir exp_results/fae_v6

Single GPU:
  python laof/cholec80_stage1_v6.py --steps 50000 --bs 128 --gpu 0 \\
      --out_dir exp_results/fae_v6
"""

import os
import sys
import math
import argparse

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from omegaconf import OmegaConf
from pathlib import Path

from utils import utils
from utils import cholec80_data_loader
from utils.cholec80_data_loader import Cholec80ChunkDataset, _DATA_ROOT


# ─── DDP helpers ──────────────────────────────────────────────────────────────
def is_ddp():         return dist.is_available() and dist.is_initialized()
def get_rank():       return dist.get_rank()       if is_ddp() else 0
def get_world_size(): return dist.get_world_size() if is_ddp() else 1
def is_main():        return get_rank() == 0

def setup_ddp():
    if "RANK" in os.environ and "WORLD_SIZE" in os.environ:
        dist.init_process_group(backend="nccl")
        torch.cuda.set_device(dist.get_rank())
        return True
    return False

def cleanup_ddp():
    if is_ddp():
        dist.destroy_process_group()


# ─── piecewise-linear LR schedule (replaces doy.PiecewiseLinearSchedule) ─────
def piecewise_linear_lr(step, milestones, lrs):
    """Return LR at `step` by linearly interpolating between (milestones, lrs)."""
    for i in range(len(milestones) - 1):
        s0, s1 = milestones[i], milestones[i + 1]
        l0, l1 = lrs[i],       lrs[i + 1]
        if s0 <= step <= s1:
            t = (step - s0) / max(s1 - s0, 1)
            return l0 + t * (l1 - l0)
    return lrs[-1]


# ─── config ───────────────────────────────────────────────────────────────────
def build_cfg(args):
    conf_path = Path(__file__).parent / "conf" / "defualt.yaml"
    cfg = OmegaConf.load(conf_path)
    OmegaConf.update(cfg, "env_name",     "cholec80",             merge=True)
    OmegaConf.update(cfg, "exp_name",     "cholec80/stage1_laof", merge=True)
    OmegaConf.update(cfg, "stage1.steps", args.steps,             merge=True)
    OmegaConf.update(cfg, "stage1.bs",    args.bs,                merge=True)
    OmegaConf.update(cfg, "stage1.lr",    args.lr,                merge=True)
    cfg.model.la_dim = 128
    cfg.model.ta_dim = 7
    cfg.stage3.batch_size       = cfg.stage3.num_envs * cfg.stage3.num_steps
    cfg.stage3.minibatch_size   = cfg.stage3.batch_size // cfg.stage3.num_minibatches
    return cfg


# ─── main ─────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(description="LAOF Stage 1 v6 — Flow AutoEncoder")
    parser.add_argument("--steps",   type=int,   default=50_000)
    parser.add_argument("--bs",      type=int,   default=128)
    parser.add_argument("--lr",      type=float, default=3e-4)
    parser.add_argument("--seed",    type=int,   default=88)
    parser.add_argument("--gpu",     type=int,   default=0,
                        help="GPU index for single-GPU mode (ignored under torchrun)")
    parser.add_argument("--out_dir", type=str,   default="exp_results/fae_v6",
                        help="Directory to save checkpoints")
    args = parser.parse_args()

    ddp_active = setup_ddp()
    rank       = get_rank()
    world_size = get_world_size()

    if not ddp_active:
        torch.cuda.set_device(args.gpu)
        device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(f"cuda:{rank}")

    cfg = build_cfg(args)
    exp_dir = Path(args.out_dir)
    if is_main():
        exp_dir.mkdir(parents=True, exist_ok=True)

    if is_main():
        eff_bs = args.bs * world_size
        print(f"\n{'='*60}")
        print(f"  LAOF Stage 1 v6 — FlowAutoEncoder")
        print(f"  world_size={world_size}  device={device}")
        print(f"  steps={args.steps}  per-GPU bs={args.bs}  eff bs={eff_bs}")
        print(f"  lr={args.lr}  scaled_lr={args.lr * world_size:.2e}")
        print(f"  out_dir={exp_dir}")
        print(f"{'='*60}\n")

    torch.manual_seed(args.seed + rank)

    # ── models ──
    fae, wm, fm = utils.create_dynamics_models_flow_fae(cfg.model)
    fae.to(device); wm.to(device); fm.to(device)

    if ddp_active:
        fae = DDP(fae, device_ids=[rank], find_unused_parameters=False)
        wm  = DDP(wm,  device_ids=[rank], find_unused_parameters=False)
        fm  = DDP(fm,  device_ids=[rank], find_unused_parameters=False)

    fae_mod = fae.module if ddp_active else fae
    wm_mod  = wm.module  if ddp_active else wm
    fm_mod  = fm.module  if ddp_active else fm

    if is_main():
        total = (sum(p.numel() for p in fae_mod.parameters())
               + sum(p.numel() for p in wm_mod.parameters())
               + sum(p.numel() for p in fm_mod.parameters()))
        print(f"[models] total params: {total/1e6:.1f}M  on {device}")

    # ── data ──
    all_train_chunks = sorted((_DATA_ROOT / "train").glob("*.npz"))
    n_total   = len(all_train_chunks)
    per_rank  = (n_total + world_size - 1) // world_size
    my_chunks = all_train_chunks[rank * per_rank : (rank + 1) * per_rank]

    if is_main():
        print(f"[data] train chunks total={n_total}, per rank≈{per_rank}")

    from utils.cholec80_data_loader import Cholec80DataLoader
    train_loader = Cholec80DataLoader.__new__(Cholec80DataLoader)
    train_loader.datasets = [Cholec80ChunkDataset(p, stride=1) for p in my_chunks]
    train_loader._total   = sum(len(d) for d in train_loader.datasets)
    print(f"[rank {rank}] {len(my_chunks)} chunks, {train_loader._total:,} pairs")

    _, test_loader, _ = cholec80_data_loader.load("cholec80")

    train_iter = train_loader.get_iter(batch_size=args.bs, infinite=True, device=str(device))
    test_iter  = test_loader.get_iter(batch_size=128,      infinite=True, device=str(device))

    if is_main():
        print("[data] iterators ready, starting training…\n")

    # ── optimiser + piecewise-linear LR (replaces doy.LRScheduler) ──
    scaled_lr = args.lr * world_size
    all_params = [*fae.parameters(), *wm.parameters(), *fm.parameters()]
    opt = torch.optim.Adam(all_params, lr=scaled_lr)

    # warmup 0→50 steps, then cosine decay to 1% of peak
    lr_milestones = [0,           50,          args.steps + 1]
    lr_values     = [0.1*scaled_lr, scaled_lr, 0.01*scaled_lr]

    def update_lr(step):
        lr = piecewise_linear_lr(step, lr_milestones, lr_values)
        for pg in opt.param_groups:
            pg["lr"] = lr
        return lr

    # ── checkpoint resume ──
    start_step = 0
    ckpt_latest = exp_dir / "idm_fdm_v6.pt"
    if ckpt_latest.exists() and is_main():
        ckpt = torch.load(ckpt_latest, map_location="cpu")
        fae_mod.load_state_dict(ckpt["fae"])
        wm_mod.load_state_dict(ckpt["wm"])
        fm_mod.load_state_dict(ckpt["fm"])
        start_step = ckpt.get("step", 0)
        print(f"[resume] loaded checkpoint at step={start_step}")

    # ── train / test / save helpers ──
    def train_step(step):
        fae.train(); wm.train(); fm.train()
        lr_now = update_lr(step)
        batch  = next(train_iter)

        fae_recon, fae_contrast = fae_mod.label_flow(batch)
        fae_loss = fae_recon + fae_mod._CONTRAST_W * fae_contrast

        wm_loss = wm_mod.label_flowdiff(batch)
        fm_loss = fm_mod.label_flow(batch)

        loss = fae_loss + wm_loss + fm_loss

        opt.zero_grad()
        loss.backward()
        grad_norm = torch.nn.utils.clip_grad_norm_(all_params, 2.0)
        opt.step()

        if is_main() and step % 100 == 0:
            with torch.no_grad():
                z_std = batch["la"].std(dim=0).mean().item()
            print(f"  step {step:6d}/{args.steps}"
                  f"  loss={loss.item():.4f}"
                  f"  (recon={fae_recon.item():.4f}"
                  f"  nce={fae_contrast.item():.4f}"
                  f"  wm={wm_loss.item():.4f}"
                  f"  fm={fm_loss.item():.4f})"
                  f"  z_std={z_std:.4f}"
                  f"  grad={grad_norm.item():.3f}"
                  f"  lr={lr_now:.2e}", flush=True)

    def test_step(step):
        fae.eval(); wm.eval(); fm.eval()
        with torch.no_grad():
            batch = next(test_iter)
            fae_recon, fae_contrast = fae_mod.label_flow(batch)
            wm_loss  = wm_mod.label_flowdiff(batch)
            fm_loss  = fm_mod.label_flow(batch)
        if is_main():
            print(f"  [TEST] step {step:6d}"
                  f"  recon={fae_recon.item():.4f}"
                  f"  nce={fae_contrast.item():.4f}"
                  f"  wm={wm_loss.item():.4f}"
                  f"  fm={fm_loss.item():.4f}", flush=True)

    def save_ckpt(step):
        if not is_main(): return
        payload   = {"fae": fae_mod.state_dict(),
                     "wm":  wm_mod.state_dict(),
                     "fm":  fm_mod.state_dict(),
                     "step": step}
        step_path = exp_dir / f"idm_fdm_v6_ckpt{step:06d}.pt"
        torch.save(payload, step_path)
        latest = exp_dir / "idm_fdm_v6.pt"
        if latest.exists() or latest.is_symlink():
            latest.unlink()
        latest.symlink_to(step_path.name)
        print(f"  [CKPT] saved → {step_path}  (step={step})", flush=True)

    # ── main loop ──
    for step in range(start_step, args.steps):
        train_step(step)
        if step % 500  == 0 and step > start_step: test_step(step)
        if step % 5000 == 0 and step > start_step: save_ckpt(step)

    save_ckpt(args.steps)

    if is_main():
        print(f"\n{'='*60}")
        print(f"  Stage 1 v6 COMPLETE  →  {exp_dir / 'idm_fdm_v6.pt'}")
        print(f"{'='*60}\n")

    cleanup_ddp()


if __name__ == "__main__":
    main()
