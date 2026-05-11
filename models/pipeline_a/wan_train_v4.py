"""
Pipeline A — WanActionAdapter v4 Training

Changes from v3:
  A. fb_norm_var_loss (NEW): penalises variance of per-frame fb magnitudes.
     This directly targets flow_var — if some frames receive a stronger push than
     others, generated motion intensity varies between frames → high flow_var.
     loss_fb_norm_var = var_across_T_lat(frame_bias.norm(dim=-1))
     Default lambda: fb_norm_var_lambda=5.0

  B. gc_upper_loss (NEW): prevents gc from dominating time_emb signal.
     gc is injected into time_emb (diffusion timestep encoding). Large gc norm
     effectively tells the DiT it's at a different noise level, disrupting its
     learned temporal dynamics. Upper-bound constraint: gc_norm < gc_max_norm.
     loss_gc_upper = relu(gc.norm(dim=-1) - gc_max_norm).mean()
     Default: gc_upper_lambda=2.0, gc_max_norm=0.5

  C. mag_loss changed from L2-to-target to lower-bound only.
     v3 used (norm - 1.0)^2, which conflicts with gc_upper_loss (pulls to 1.0
     while upper bound pushes to <0.5). v4 uses relu(0.1 - norm) — only
     penalises if norm < 0.1, no upper pull.

  D. fb_norm_var_lambda increased fb_smooth_lambda: 10.0 → 30.0 (default).
     v3 showed smooth_loss stabilised but fb_frame_std still 0.006.
     3× lambda should compress temporal variation further.

  E. fb_sens_loss (NEW): sens_loss extended to fb mean across T_lat.
     Ensures fb also varies with different z, not just gc.
     fb_mean = frame_bias.mean(dim=1)
     fb_sens = relu(cos_sim(fb_mean, fb_neg) - margin).mean()
     Weight: 0.5 × sens_loss magnitude.

  F. total_steps default: 20000 → 30000.
     v3 at step 20000 showed sens still oscillating; 30k gives more stable
     convergence of the three-objective loss (smoothness + sensitivity + norm).

Training loss (v4 total):
  loss = sens_loss                          # gc z-sensitivity
       + 0.5 * fb_sens_loss                 # fb z-sensitivity (NEW E)
       + mag_loss                           # gc norm lower bound (modified C)
       + gc_upper_lambda * gc_upper_loss    # gc norm upper bound (NEW B)
       + smooth_lambda * smooth_loss        # fb direction smoothness (lambda 30→D)
       + fb_norm_var_lambda * fb_norm_var_loss  # fb norm uniformity (NEW A)

Usage (4-GPU DDP, from project root):
  torchrun --nproc_per_node=4 --master_port=29513 \\
      models/pipeline_a/wan_train_v4.py \\
      --wan_dir   /path/to/Wan2.1 \\
      --ckpt_dir  models/wan2_1_1_3b \\
      --lora_path wan_lora_cholec80.safetensors \\
      --data_dir  data/processed/pipeline_a_v2/train \\
      --cache_dir data/processed/pipeline_a_v2/wan_orig_cache_256 \\
      --out_dir   exp_results/pipeline_a/wan_lora_256_v4 \\
      --height 256 --width 448 \\
      --batch_size 1 --total_steps 30000 --save_every 3000 \\
      --lr 1e-4 --skip_precompute \\
      --sens_margin 0.0 --smooth_lambda 30.0 \\
      --fb_norm_var_lambda 5.0 --gc_upper_lambda 2.0 --gc_max_norm 0.5
"""

import os, sys, argparse
from pathlib import Path

# wan_dir is passed via --wan_dir; added to sys.path at runtime in main()
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)) + "/../..")

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

from wan_action_adapter import (
    WanActionAdapter,
    set_wan_global_cond, set_wan_frame_bias,
    register_wan_orig_time_hook, register_wan_orig_patch_hook,
)

# ── LoRA merge ────────────────────────────────────────────────────────────────

_LORA_TO_MODULE = {
    "self_attn_q":  "self_attn.q", "self_attn_k":  "self_attn.k",
    "self_attn_v":  "self_attn.v", "self_attn_o":  "self_attn.o",
    "cross_attn_q": "cross_attn.q", "cross_attn_k": "cross_attn.k",
    "cross_attn_v": "cross_attn.v", "cross_attn_o": "cross_attn.o",
    "ffn_0": "ffn.0", "ffn_2": "ffn.2",
}

def _lora_key_to_module_path(prefix):
    if not prefix.startswith("lora_unet_blocks_"):
        return None
    rest = prefix[len("lora_unet_blocks_"):]
    parts = rest.split("_", 1)
    if len(parts) != 2 or not parts[0].isdigit():
        return None
    suf = _LORA_TO_MODULE.get(parts[1])
    return f"blocks.{parts[0]}.{suf}" if suf else None

def merge_lora_into_model(model, lora_path, lora_scale=1.0, rank=0):
    import safetensors.torch as st
    lora_sd   = st.load_file(lora_path)
    prefixes  = {k[:-len(".lora_down.weight")] for k in lora_sd if k.endswith(".lora_down.weight")}
    mod_dict  = dict(model.named_modules())
    merged = skipped = 0
    for prefix in sorted(prefixes):
        down = lora_sd.get(prefix + ".lora_down.weight")
        up   = lora_sd.get(prefix + ".lora_up.weight")
        if down is None or up is None:
            skipped += 1; continue
        alpha_v = lora_sd.get(prefix + ".alpha")
        r     = down.shape[0]
        alpha = alpha_v.float().item() if alpha_v is not None else float(r)
        mp    = _lora_key_to_module_path(prefix)
        if mp is None:
            skipped += 1; continue
        mod = mod_dict.get(mp)
        if mod is None or not hasattr(mod, "weight"):
            skipped += 1; continue
        delta = (up.float() @ down.float()) * (alpha / r * lora_scale)
        with torch.no_grad():
            mod.weight.data += delta.to(mod.weight.dtype).to(mod.weight.device)
        merged += 1
    if rank == 0:
        print(f"[LoRA merge] merged={merged}  skipped={skipped}", flush=True)


# ── DDP helpers ───────────────────────────────────────────────────────────────

def setup_ddp():
    rank       = int(os.environ.get("RANK", 0))
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    world_size = int(os.environ.get("WORLD_SIZE", 1))
    if world_size > 1:
        torch.distributed.init_process_group("nccl")
    torch.cuda.set_device(local_rank)
    return rank, local_rank, world_size

def log(rank, *args):
    if rank == 0:
        print(*args, flush=True)


# ── Wan loaders ───────────────────────────────────────────────────────────────

def load_wan_model(wan_dir, ckpt_dir, device, dtype):
    from wan.modules.model import WanModel
    from wan.configs import WAN_CONFIGS
    from easydict import EasyDict
    cfg = EasyDict(WAN_CONFIGS["t2v-1.3B"])
    model = WanModel(
        dim=cfg.dim, ffn_dim=cfg.ffn_dim, freq_dim=cfg.freq_dim,
        num_heads=cfg.num_heads, num_layers=cfg.num_layers,
        window_size=getattr(cfg, "window_size", (-1, -1)),
        qk_norm=cfg.qk_norm, cross_attn_norm=cfg.cross_attn_norm, eps=cfg.eps,
    )
    import safetensors.torch as st
    sd = st.load_file(str(Path(ckpt_dir) / "diffusion_pytorch_model.safetensors"))
    model.load_state_dict(sd, strict=True)
    model.requires_grad_(False).eval()
    return model.to(device, dtype=dtype)

def load_wan_vae(ckpt_dir, device, dtype):
    from wan.modules.vae import WanVAE
    vae = WanVAE(vae_pth=str(Path(ckpt_dir) / "Wan2.1_VAE.pth"))
    vae.model = vae.model.to(device, dtype=dtype)
    vae.model.requires_grad_(False).eval()
    return vae

def load_wan_t5(wan_dir, ckpt_dir, device, dtype):
    from wan.modules.t5 import T5EncoderModel
    return T5EncoderModel(
        text_len=512, dtype=dtype, device=torch.device("cpu"),
        checkpoint_path=str(Path(ckpt_dir) / "models_t5_umt5-xxl-enc-bf16.pth"),
        tokenizer_path=str(Path(ckpt_dir) / "google" / "umt5-xxl"),
    )


# ── Dataset ───────────────────────────────────────────────────────────────────

class WanOrigDataset(Dataset):
    def __init__(self, data_dir, cache_dir, target_h=480, target_w=832):
        self.cache_root = Path(cache_dir)
        self.target_h   = target_h
        self.target_w   = target_w
        lat_dir = self.cache_root / "video_latent"
        txt_dir = self.cache_root / "text_emb"
        all_npz = sorted(Path(data_dir).glob("*.npz"))
        valid = []
        for p in all_npz:
            stem = p.stem.rsplit("_seg", 1)
            clip_name, seg_idx = stem[0], int(stem[1])
            base_key = f"{clip_name}_seg{seg_idx:03d}"
            lat_key  = f"{base_key}_{target_h}x{target_w}"
            if not (lat_dir / f"{lat_key}.pt").exists():
                continue
            txt_key = lat_key
            if not (txt_dir / f"{txt_key}.pt").exists():
                matches = list(txt_dir.glob(f"{base_key}_*.pt"))
                if not matches:
                    continue
                txt_key = matches[0].stem
            valid.append((p, lat_key, txt_key))
        self.samples = valid
        print(f"[WanOrigDataset] {len(valid)}/{len(all_npz)} cached", flush=True)

    def __len__(self): return len(self.samples)

    def __getitem__(self, idx):
        npz_path, lat_key, txt_key = self.samples[idx]
        data     = np.load(npz_path, allow_pickle=True)
        z_seq    = torch.from_numpy(data["z_seq"].astype(np.float32))
        lat_dir  = self.cache_root / "video_latent"
        txt_dir  = self.cache_root / "text_emb"
        latent   = torch.load(lat_dir / f"{lat_key}.pt", map_location="cpu")
        text_emb = torch.load(txt_dir / f"{txt_key}.pt", map_location="cpu")
        return {"z_seq": z_seq, "latent": latent,
                "text_emb": text_emb.squeeze(0) if text_emb.dim() == 3 else text_emb}

def collate_fn(batch):
    return {
        "z_seq":    torch.stack([b["z_seq"]    for b in batch]),
        "latent":   torch.stack([b["latent"]   for b in batch]),
        "text_emb": torch.stack([b["text_emb"] for b in batch]),
    }


# ── Precompute ────────────────────────────────────────────────────────────────

@torch.no_grad()
def precompute(vae, t5, data_dir, cache_dir, target_h, target_w, device, dtype, rank):
    lat_dir = Path(cache_dir) / "video_latent"
    txt_dir = Path(cache_dir) / "text_emb"
    lat_dir.mkdir(parents=True, exist_ok=True)
    txt_dir.mkdir(parents=True, exist_ok=True)
    npz_list = sorted(Path(data_dir).glob("*.npz"))
    log(rank, f"[precompute] {len(npz_list)} segments")
    for i, npz_path in enumerate(npz_list):
        stem      = npz_path.stem.rsplit("_seg", 1)
        clip_name, seg_idx = stem[0], int(stem[1])
        key       = f"{clip_name}_seg{seg_idx:03d}_{target_h}x{target_w}"
        lat_path  = lat_dir / f"{key}.pt"
        txt_path  = txt_dir / f"{key}.pt"
        if lat_path.exists() and txt_path.exists():
            continue
        data       = np.load(npz_path, allow_pickle=True)
        frames_np  = data["frames_256"]
        caption    = str(data["caption"])[:512]
        frames_f32 = torch.from_numpy(frames_np.astype(np.float32)) / 255.0 * 2.0 - 1.0
        if frames_f32.shape[2] != target_h or frames_f32.shape[3] != target_w:
            frames_f32 = F.interpolate(frames_f32, size=(target_h, target_w),
                                       mode="bilinear", align_corners=False)
        vid    = frames_f32.permute(1, 0, 2, 3).unsqueeze(0).to(device, dtype)
        latent = vae.encode([vid[0]])[0]
        torch.save(latent.cpu(), lat_path)
        text_emb = t5([caption], device=torch.device("cpu"))
        if isinstance(text_emb, list):
            text_emb = text_emb[0]
        torch.save(text_emb.cpu().unsqueeze(0), txt_path)
        if (i + 1) % 50 == 0:
            log(rank, f"  [{i+1}/{len(npz_list)}] {key}")
    log(rank, "[precompute] done")


# ── Training loop ─────────────────────────────────────────────────────────────

def train(args, rank, local_rank, world_size):
    device = torch.device(f"cuda:{local_rank}")
    dtype  = torch.bfloat16

    log(rank, f"[init] rank={rank}/{world_size}  device={device}")

    if not args.skip_precompute and rank == 0:
        log(rank, "[precompute] loading VAE and T5...")
        vae = load_wan_vae(args.ckpt_dir, device, dtype)
        t5  = load_wan_t5(args.wan_dir, args.ckpt_dir, device, dtype)
        precompute(vae, t5, args.data_dir, args.cache_dir,
                   args.height, args.width, device, dtype, rank)
        del vae, t5
        torch.cuda.empty_cache()
    if world_size > 1:
        torch.distributed.barrier()

    log(rank, "[model] loading WanModel + LoRA...")
    wan_model = load_wan_model(args.wan_dir, args.ckpt_dir, device, dtype)
    if args.lora_path:
        merge_lora_into_model(wan_model, args.lora_path, args.lora_scale, rank)
    wan_model.requires_grad_(False).eval()

    adapter = WanActionAdapter(
        la_dim=128, hidden_dim=1536,
        temporal_compression=4, mlp_hidden=512,
    ).to(device, dtype=dtype)
    n_params = sum(p.numel() for p in adapter.parameters())
    log(rank, f"  WanActionAdapter: {n_params/1e6:.2f}M params")

    if world_size > 1:
        adapter = torch.nn.parallel.DistributedDataParallel(
            adapter, device_ids=[local_rank], output_device=local_rank,
            find_unused_parameters=True)
    adapter_mod = adapter.module if world_size > 1 else adapter

    _time_hook  = register_wan_orig_time_hook(wan_model)
    _patch_hook = register_wan_orig_patch_hook(wan_model)

    dataset = WanOrigDataset(args.data_dir, args.cache_dir, args.height, args.width)
    if len(dataset) == 0:
        log(rank, "[ERROR] no cached segments"); return

    sampler = (torch.utils.data.distributed.DistributedSampler(dataset, shuffle=True)
               if world_size > 1 else None)
    loader  = DataLoader(dataset, batch_size=args.batch_size,
                         sampler=sampler, shuffle=(sampler is None),
                         num_workers=2, pin_memory=True,
                         collate_fn=collate_fn, drop_last=True)

    opt   = torch.optim.AdamW(adapter.parameters(), lr=args.lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.total_steps, eta_min=args.lr * 0.01)

    num_train_steps = 1000
    betas = torch.linspace(0.00085**0.5, 0.012**0.5, num_train_steps, dtype=torch.float32)**2
    alphas_cumprod = torch.cumprod(1.0 - betas, dim=0).to(device)

    def add_noise(x0, noise, t):
        a = alphas_cumprod[t][:, None, None, None, None].sqrt()
        s = (1 - alphas_cumprod[t])[:, None, None, None, None].sqrt()
        return a * x0 + s * noise

    out_dir   = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    ckpt_path = out_dir / "wan_action_adapter.pt"
    global_step = 0

    if ckpt_path.exists() and rank == 0:
        ckpt = torch.load(ckpt_path, map_location="cpu")
        adapter_mod.load_state_dict(ckpt["adapter"])
        global_step = ckpt.get("step", 0)
        for _ in range(global_step):
            sched.step()
        log(rank, f"[resume] step={global_step}  lr={sched.get_last_lr()[0]:.2e}")

    log(rank, f"[train] total_steps={args.total_steps}  batch={args.batch_size}  lr={args.lr}")
    log(rank, f"  smooth_λ={args.smooth_lambda}  fb_norm_var_λ={args.fb_norm_var_lambda}"
              f"  gc_upper_λ={args.gc_upper_lambda}  gc_max_norm={args.gc_max_norm}"
              f"  sens_margin={args.sens_margin}")

    adapter.train()
    loss_ema  = None
    prev_gc   = None
    prev_fb_mean = None

    while global_step < args.total_steps:
        if sampler: sampler.set_epoch(global_step // max(1, len(loader)))
        for batch in loader:
            if global_step >= args.total_steps: break

            z_seq    = batch["z_seq"].to(device, dtype=dtype)
            latent   = batch["latent"].to(device, dtype=dtype)
            text_emb = batch["text_emb"].to(device, dtype=dtype)
            B        = z_seq.shape[0]

            timesteps = torch.randint(0, num_train_steps, (B,), device=device)
            noise     = torch.randn_like(latent)
            noisy_lat = add_noise(latent, noise, timesteps).to(dtype)

            global_cond, frame_bias = adapter(z_seq)
            set_wan_global_cond(global_cond)
            set_wan_frame_bias(frame_bias)

            _, _, T_lat, H_lat, W_lat = noisy_lat.shape
            seq_len = T_lat * (H_lat // 2) * (W_lat // 2)
            with torch.no_grad(), torch.amp.autocast("cuda", dtype=dtype):
                pred = wan_model(
                    x=[noisy_lat[i] for i in range(B)],
                    t=timesteps.float(),
                    context=[text_emb[i] for i in range(B)],
                    seq_len=seq_len,
                )
            set_wan_global_cond(None)
            set_wan_frame_bias(None)

            pred     = torch.stack(pred, dim=0)
            mse_loss = F.mse_loss(pred, noise.to(pred.dtype))  # monitoring only

            # ── [A] fb norm variance loss (NEW v4) ───────────────────────────
            # Each T_lat frame should push the DiT with equal magnitude.
            # Variance of per-frame norms → high = some frames pushed harder → flow_var.
            fb_norms = frame_bias.norm(dim=-1)                  # (B, T_lat)
            if fb_norms.shape[1] > 1:
                fb_norm_var_loss = fb_norms.var(dim=-1).mean()
            else:
                fb_norm_var_loss = torch.zeros(1, device=device)

            # ── [B] fb direction smoothness (v3, stronger lambda) ────────────
            if frame_bias.shape[1] > 1:
                fb_diff     = frame_bias[:, 1:] - frame_bias[:, :-1]
                smooth_loss = fb_diff.pow(2).mean()
            else:
                smooth_loss = torch.zeros(1, device=device)

            # ── [C] gc upper-bound loss (NEW v4) ─────────────────────────────
            # Prevents gc from overwhelming time_emb and disrupting DiT dynamics.
            gc_upper_loss = F.relu(global_cond.norm(dim=-1) - args.gc_max_norm).mean()

            # ── [D] gc z-sensitivity + lower-bound (modified from v3) ────────
            # mag_loss: lower-bound only (v3 used L2-to-1.0 which conflicts with upper bound)
            if B > 1:
                gc_neg = torch.roll(global_cond, shifts=1, dims=0)
            elif prev_gc is not None:
                gc_neg = prev_gc.expand_as(global_cond)
            else:
                gc_neg = None

            if gc_neg is not None:
                gc_cos    = F.cosine_similarity(global_cond, gc_neg, dim=-1)
                sens_loss = F.relu(gc_cos - args.sens_margin).mean()
                mag_loss  = F.relu(0.1 - global_cond.norm(dim=-1)).mean()  # lower bound only
            else:
                sens_loss = torch.zeros(1, device=device)
                mag_loss  = torch.zeros(1, device=device)

            # ── [E] fb z-sensitivity (NEW v4) ────────────────────────────────
            # Extend sens_loss to fb mean — ensures fb also responds to z content.
            fb_mean = frame_bias.mean(dim=1)  # (B, hidden_dim)
            if B > 1:
                fb_neg_mean = torch.roll(fb_mean, shifts=1, dims=0)
            elif prev_fb_mean is not None:
                fb_neg_mean = prev_fb_mean.expand_as(fb_mean)
            else:
                fb_neg_mean = None

            if fb_neg_mean is not None:
                fb_cos      = F.cosine_similarity(fb_mean, fb_neg_mean, dim=-1)
                fb_sens_loss = F.relu(fb_cos - args.sens_margin).mean()
            else:
                fb_sens_loss = torch.zeros(1, device=device)

            # ── Total loss ────────────────────────────────────────────────────
            loss = (sens_loss                                   # gc z-sensitivity
                  + 0.5 * fb_sens_loss                         # fb z-sensitivity (NEW E)
                  + mag_loss                                    # gc lower bound (modified C)
                  + args.gc_upper_lambda * gc_upper_loss        # gc upper bound (NEW B)
                  + args.smooth_lambda * smooth_loss            # fb smoothness (lambda 30 D)
                  + args.fb_norm_var_lambda * fb_norm_var_loss) # fb norm variance (NEW A)

            loss.backward()
            torch.nn.utils.clip_grad_norm_(adapter.parameters(), 1.0)
            opt.step()
            opt.zero_grad()
            sched.step()

            prev_gc      = global_cond.detach()
            prev_fb_mean = fb_mean.detach()

            global_step += 1
            step_loss = loss.item()
            loss_ema  = step_loss if loss_ema is None else 0.98*loss_ema + 0.02*step_loss

            if global_step % args.log_every == 0 and rank == 0:
                with torch.no_grad():
                    gc_std      = global_cond.std().item()
                    gc_diff     = (global_cond - (gc_neg if gc_neg is not None
                                  else global_cond)).norm(dim=-1).mean().item()
                    gc_norm_    = global_cond.norm(dim=-1).mean().item()
                    fb_frame_std = frame_bias.std(dim=1).mean().item()
                    fb_norm_std  = fb_norms.std(dim=-1).mean().item()   # per-clip fb-norm spread
                log(rank,
                    f"  step={global_step:6d}  loss={loss_ema:.4f}  "
                    f"sens={sens_loss.item():.4f}  fb_sens={fb_sens_loss.item():.4f}  "
                    f"mag={mag_loss.item():.4f}  gc_upper={gc_upper_loss.item():.4f}  "
                    f"smooth={smooth_loss.item():.5f}  fb_nv={fb_norm_var_loss.item():.5f}  "
                    f"gc_norm={gc_norm_:.3f}  gc_diff={gc_diff:.4f}  "
                    f"fb_frame_std={fb_frame_std:.4f}  fb_norm_std={fb_norm_std:.4f}  "
                    f"mse={mse_loss.item():.4f}  lr={sched.get_last_lr()[0]:.2e}")

            if global_step % args.save_every == 0 and rank == 0:
                payload   = {"step": global_step, "adapter": adapter_mod.state_dict()}
                torch.save(payload, ckpt_path)
                step_path = out_dir / f"wan_action_adapter_step{global_step:06d}.pt"
                torch.save(payload, step_path)
                log(rank, f"  [CKPT] {step_path.name}")

    if rank == 0:
        payload = {"step": global_step, "adapter": adapter_mod.state_dict(),
                   "optim": opt.state_dict()}
        torch.save(payload, ckpt_path)
        log(rank, f"[done] step={global_step}")

    if world_size > 1:
        torch.distributed.destroy_process_group()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--wan_dir",     required=True,
                        help="Path to Wan2.1 source directory (contains wan/ package)")
    parser.add_argument("--ckpt_dir",    default="models/wan2_1_1_3b")
    parser.add_argument("--lora_path",   default=None)
    parser.add_argument("--lora_scale",  type=float, default=1.0)
    parser.add_argument("--data_dir",    default="data/processed/pipeline_a_v2/train")
    parser.add_argument("--cache_dir",   default="data/processed/pipeline_a_v2/wan_orig_cache")
    parser.add_argument("--out_dir",     default="exp_results/pipeline_a/wan_lora_256_v4")
    parser.add_argument("--height",      type=int, default=256)
    parser.add_argument("--width",       type=int, default=448)
    parser.add_argument("--batch_size",  type=int, default=1)
    parser.add_argument("--total_steps", type=int, default=30000)
    parser.add_argument("--save_every",  type=int, default=3000)
    parser.add_argument("--lr",          type=float, default=1e-4)
    parser.add_argument("--gpu",         type=int, default=0)
    parser.add_argument("--skip_precompute", action="store_true")
    # v3 args kept
    parser.add_argument("--sens_margin",   type=float, default=0.0)
    parser.add_argument("--smooth_lambda", type=float, default=30.0,
                        help="fb direction smoothness weight (v3=10 → v4=30)")
    parser.add_argument("--log_every",     type=int, default=100)
    # v4 new args
    parser.add_argument("--fb_norm_var_lambda", type=float, default=5.0,
                        help="[NEW v4] weight for fb per-frame norm variance loss")
    parser.add_argument("--gc_upper_lambda",    type=float, default=2.0,
                        help="[NEW v4] weight for gc norm upper-bound loss")
    parser.add_argument("--gc_max_norm",        type=float, default=0.5,
                        help="[NEW v4] gc norm upper bound (default 0.5)")
    args = parser.parse_args()

    # Make wan library importable
    sys.path.insert(0, args.wan_dir)

    if args.lora_path and not Path(args.lora_path).is_absolute():
        proj_root = Path(__file__).resolve().parents[2]
        args.lora_path = str(proj_root / args.lora_path)

    rank, local_rank, world_size = setup_ddp()
    if world_size == 1:
        torch.cuda.set_device(args.gpu)
    train(args, rank, local_rank, world_size)


if __name__ == "__main__":
    main()
