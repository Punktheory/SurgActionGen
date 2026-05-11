"""
Full 200-prompt inference: baseline + lora + adapter at 5 scales.

For each prompt, generates:
  baseline  — Wan2.1 only (no LoRA)
  lora      — Wan2.1 + LoRA
  scale_01/03/05/07/10  — Wan2.1 + LoRA + WanActionAdapter v4

Output structure:
  final_result/
    prompt_000_baseline.mp4
    prompt_000_lora.mp4
    scale_01/prompt_000_adapter.mp4
    scale_03/prompt_000_adapter.mp4
    scale_05/prompt_000_adapter.mp4
    scale_07/prompt_000_adapter.mp4
    scale_10/prompt_000_adapter.mp4
    ... (up to prompt_199)

Resume-safe: skips files that already exist.

Usage (8 GPUs × 25 prompts each, from project root):
  for i in $(seq 0 7); do
      nohup python -u inference/run_inference.py \\
          --wan_dir /path/to/Wan2.1 \\
          --gpu $i --start_idx $((i*25)) --end_idx $(((i+1)*25)) \\
          > final_result/log_gpu${i}.txt 2>&1 &
  done
  wait

Or run a single GPU smoke test:
  python -u inference/run_inference.py --wan_dir /path/to/Wan2.1 --gpu 0 --start_idx 0 --end_idx 2
"""

import os, sys, gc, argparse, json
from pathlib import Path
import torch
import numpy as np

def _strip_conda_libs():
    ld = os.environ.get("LD_LIBRARY_PATH", "")
    if not ld:
        return
    bad = ("/miniconda3/lib", "/anaconda3/lib", "/Micromamba/")
    parts = [p for p in ld.split(":") if p.strip() and not any(b in p for b in bad)]
    os.environ["LD_LIBRARY_PATH"] = ":".join(parts)

_strip_conda_libs()

proj_root = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(proj_root / "models" / "pipeline_a"))
sys.path.insert(0, str(proj_root))
# wan_dir (Wan2.1 source) is inserted into sys.path in main() after argparse

import safetensors.torch as st

# ── constants ─────────────────────────────────────────────────────────────────

ADAPTER_SCALES = [0.1, 0.3, 0.5, 0.7, 1.0]
SCALE_DIRS     = {"0.1": "scale_01", "0.3": "scale_03", "0.5": "scale_05",
                  "0.7": "scale_07", "1.0": "scale_10"}

# ── LoRA merge ────────────────────────────────────────────────────────────────

_LORA_TO_MODULE = {
    "self_attn_q":  "self_attn.q",  "self_attn_k":  "self_attn.k",
    "self_attn_v":  "self_attn.v",  "self_attn_o":  "self_attn.o",
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

def merge_lora(model, lora_path):
    lora_sd = st.load_file(str(lora_path))
    prefixes = {k[:-len(".lora_down.weight")] for k in lora_sd if k.endswith(".lora_down.weight")}
    merged = 0
    for prefix in prefixes:
        down = lora_sd.get(prefix + ".lora_down.weight")
        up   = lora_sd.get(prefix + ".lora_up.weight")
        if down is None or up is None:
            continue
        r = down.shape[0]
        alpha = lora_sd[prefix + ".alpha"].float().item() if prefix + ".alpha" in lora_sd else float(r)
        mp = _lora_key_to_module_path(prefix)
        if mp is None:
            continue
        try:
            mod = model.model
            for attr in mp.split("."):
                mod = getattr(mod, attr)
            delta = (up.float() @ down.float()) * (alpha / r)
            with torch.no_grad():
                mod.weight.data += delta.to(mod.weight.dtype).to(mod.weight.device)
            merged += 1
        except AttributeError:
            pass
    print(f"[LoRA] merged={merged}", flush=True)


# ── video utils ───────────────────────────────────────────────────────────────

def tensor_to_uint8(video):
    v = video.float().cpu().clamp(-1, 1)
    v = ((v + 1) / 2 * 255).byte()
    return v.permute(1, 2, 3, 0).numpy()

def save_mp4(frames, path, fps=8):
    import imageio
    imageio.mimwrite(str(path), frames, fps=fps, codec="libx264",
                     output_params=["-crf", "23", "-pix_fmt", "yuv420p"])


# ── Wan model ─────────────────────────────────────────────────────────────────

def build_wan(ckpt_dir, device_id, dtype):
    from wan.text2video import WanT2V
    from wan.configs import WAN_CONFIGS
    from easydict import EasyDict
    cfg = EasyDict(WAN_CONFIGS["t2v-1.3B"])
    cfg.param_dtype = dtype
    return WanT2V(
        config=cfg, checkpoint_dir=str(ckpt_dir), device_id=device_id,
        rank=0, t5_fsdp=False, dit_fsdp=False, use_usp=False, t5_cpu=True,
    )

def generate_video(t2v, prompt, height, width, frames, steps, seed):
    video = t2v.generate(
        input_prompt=prompt, size=(width, height), frame_num=frames,
        shift=5.0, sample_solver="unipc", sampling_steps=steps,
        guide_scale=5.0, seed=seed, offload_model=True,
    )
    return tensor_to_uint8(video)


# ── z predictor ───────────────────────────────────────────────────────────────

def load_z_predictor(ckpt_path, device):
    from models.pipeline_b.train_z_predictor import FrozenCLIPText, ZPredictor
    clip  = FrozenCLIPText().to(device)
    model = ZPredictor(text_dim=768, d_model=256, n_heads=4, n_layers=4,
                       T=32, z_dim=128).to(device)
    ckpt  = torch.load(str(ckpt_path), map_location="cpu")
    model.load_state_dict(ckpt["model"])
    model.eval()
    print(f"[z_predictor] step={ckpt.get('step')}", flush=True)
    return clip, model

@torch.no_grad()
def predict_z_seq(clip, model, prompt, device, dtype, T=33):
    text_emb = clip([prompt], device)
    z = model(text_emb)          # (1, 32, 128)
    if T > 32:
        z = torch.cat([z, z[:, -1:].expand(1, T - 32, 128)], dim=1)
    return z[:, :T].to(dtype)


# ── main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--wan_dir",         required=True,
                        help="Path to Wan2.1 source directory (contains wan/ package)")
    parser.add_argument("--ckpt_dir",        default="models/wan2_1_1_3b")
    parser.add_argument("--lora_path",       default=None)
    parser.add_argument("--adapter_ckpt",    default="exp_results/pipeline_a/wan_lora_256_v4/wan_action_adapter_step030000.pt")
    parser.add_argument("--z_predictor_ckpt",default="exp_results/pipeline_b/z_predictor_step010000.pt")
    parser.add_argument("--prompt_json",     default="prompt_set.json")
    parser.add_argument("--out_dir",         default="final_result")
    parser.add_argument("--height",          type=int, default=256)
    parser.add_argument("--width",           type=int, default=448)
    parser.add_argument("--frames",          type=int, default=33)
    parser.add_argument("--steps",           type=int, default=20)
    parser.add_argument("--seed",            type=int, default=42)
    parser.add_argument("--gpu",             type=int, default=0)
    parser.add_argument("--start_idx",       type=int, default=0,   help="first prompt index (inclusive)")
    parser.add_argument("--end_idx",         type=int, default=25,  help="last prompt index (exclusive)")
    parser.add_argument("--scales",          type=float, nargs="+", default=ADAPTER_SCALES)
    args = parser.parse_args()

    sys.path.insert(0, args.wan_dir)

    device = torch.device(f"cuda:{args.gpu}")
    dtype  = torch.bfloat16

    # Resolve paths relative to project root
    def rp(p):
        q = Path(p)
        return q if q.is_absolute() else proj_root / q

    out_dir = rp(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Scale subdirs
    scale_dirs = {}
    for s in args.scales:
        key = f"{s:.1f}"
        dname = SCALE_DIRS.get(key, f"scale_{int(s*10):02d}")
        d = out_dir / dname
        d.mkdir(parents=True, exist_ok=True)
        scale_dirs[s] = d

    # Load prompts
    with open(rp(args.prompt_json)) as f:
        all_prompts = json.load(f)
    if isinstance(all_prompts, dict):
        all_prompts = list(all_prompts.values())

    end_idx = min(args.end_idx, len(all_prompts))
    prompts = all_prompts[args.start_idx:end_idx]
    indices = list(range(args.start_idx, end_idx))

    print(f"\n{'='*70}", flush=True)
    print(f"  GPU={args.gpu}  prompts [{args.start_idx}, {end_idx})  "
          f"{args.width}×{args.height}  frames={args.frames}  steps={args.steps}", flush=True)
    print(f"  scales: {args.scales}", flush=True)
    print(f"  out: {out_dir}", flush=True)
    print(f"{'='*70}\n", flush=True)

    # ── Phase 1: Baseline (Wan only, no LoRA) ────────────────────────────────
    need_baseline = [i for i, idx in enumerate(indices)
                     if not (out_dir / f"prompt_{idx:03d}_baseline.mp4").exists()]

    if need_baseline:
        print(f"[Phase 1] Loading Wan2.1 (no LoRA) — {len(need_baseline)} baseline videos", flush=True)
        t2v = build_wan(rp(args.ckpt_dir), args.gpu, dtype)

        for local_i in need_baseline:
            idx = indices[local_i]
            prompt = prompts[local_i]
            out_path = out_dir / f"prompt_{idx:03d}_baseline.mp4"
            print(f"  [baseline] {idx:03d}/{end_idx-1}: {prompt[:70]}...", flush=True)
            frames = generate_video(t2v, prompt, args.height, args.width,
                                    args.frames, args.steps, args.seed)
            save_mp4(frames, out_path)
            print(f"  → {out_path.name}", flush=True)

        del t2v
        torch.cuda.empty_cache(); gc.collect()
        print("[Phase 1] done\n", flush=True)
    else:
        print("[Phase 1] all baseline files exist — skipping model load\n", flush=True)

    # ── Phase 2: LoRA + Adapter ───────────────────────────────────────────────
    # Determine which prompts still need lora or any adapter scale
    need_lora = [i for i, idx in enumerate(indices)
                 if not (out_dir / f"prompt_{idx:03d}_lora.mp4").exists()]
    need_adapter = {s: [i for i, idx in enumerate(indices)
                         if not (scale_dirs[s] / f"prompt_{idx:03d}_adapter.mp4").exists()]
                    for s in args.scales}
    any_adapter_needed = any(len(v) > 0 for v in need_adapter.values())

    if not need_lora and not any_adapter_needed:
        print("[Phase 2] all lora + adapter files exist — skipping\n", flush=True)
    else:
        print(f"[Phase 2] Loading Wan2.1 + LoRA + Adapter", flush=True)
        t2v = build_wan(rp(args.ckpt_dir), args.gpu, dtype)
        merge_lora(t2v, rp(args.lora_path))

        # Load adapter
        from wan_action_adapter import (
            WanActionAdapter, set_wan_global_cond, set_wan_frame_bias,
            register_wan_orig_time_hook, register_wan_orig_patch_hook,
        )
        adapter_ckpt = torch.load(str(rp(args.adapter_ckpt)), map_location="cpu")
        hidden_dim = t2v.model.blocks[0].self_attn.q.weight.shape[0]
        adapter = WanActionAdapter(
            la_dim=128, hidden_dim=hidden_dim,
            temporal_compression=4, mlp_hidden=512,
            zero_init_injectors=False,
        )
        adapter.load_state_dict(adapter_ckpt["adapter"], strict=False)
        adapter.eval().to(device, dtype=dtype)
        print(f"[adapter] step={adapter_ckpt.get('step')}", flush=True)

        h1 = register_wan_orig_time_hook(t2v.model)
        h2 = register_wan_orig_patch_hook(t2v.model)

        # Load z predictor
        z_clip, z_model = load_z_predictor(rp(args.z_predictor_ckpt), device)

        # All unique local indices we still need to touch
        needed_local = sorted(set(need_lora) | set().union(*need_adapter.values()))

        for local_i in needed_local:
            idx = indices[local_i]
            prompt = prompts[local_i]
            print(f"\n  [prompt {idx:03d}] {prompt[:70]}...", flush=True)

            # Predict z (used by all adapter scales)
            z_seq = predict_z_seq(z_clip, z_model, prompt, device, dtype, T=args.frames)
            with torch.no_grad():
                gc_raw, fb_raw = adapter(z_seq)

            # Lora video (no adapter injection)
            lora_path_out = out_dir / f"prompt_{idx:03d}_lora.mp4"
            if local_i in need_lora:
                set_wan_global_cond(None)
                set_wan_frame_bias(None)
                frames = generate_video(t2v, prompt, args.height, args.width,
                                        args.frames, args.steps, args.seed)
                save_mp4(frames, lora_path_out)
                print(f"    → lora saved", flush=True)

            # Adapter videos at each scale
            for s in args.scales:
                if local_i not in need_adapter[s]:
                    continue
                adp_path = scale_dirs[s] / f"prompt_{idx:03d}_adapter.mp4"
                set_wan_global_cond(gc_raw * s)
                set_wan_frame_bias(fb_raw * s)
                frames = generate_video(t2v, prompt, args.height, args.width,
                                        args.frames, args.steps, args.seed)
                save_mp4(frames, adp_path)
                print(f"    → scale={s} saved", flush=True)

        set_wan_global_cond(None)
        set_wan_frame_bias(None)
        h1.remove(); h2.remove()
        del t2v, adapter, z_clip, z_model
        torch.cuda.empty_cache(); gc.collect()
        print("\n[Phase 2] done", flush=True)

    print(f"\n[DONE] GPU={args.gpu}  prompts [{args.start_idx}, {end_idx})", flush=True)


if __name__ == "__main__":
    main()
