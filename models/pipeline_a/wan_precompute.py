"""
Two-phase Wan2.1 precompute for Pipeline A:

Phase 1: VAE latent encoding (GPU, fast)
Phase 2: T5 text encoding (CPU, batched by unique captions - 4106 unique vs 52632 total)

Usage:
  conda activate cogvideox
  python -u models/pipeline_a/wan_precompute.py \
      --wan_dir  /path/to/Wan2.1 \
      --ckpt_dir models/wan2_1_1_3b \
      --data_dir data/processed/pipeline_a_v2/train \
      --cache_dir data/processed/pipeline_a_v2/wan_orig_cache_256 \
      --height 480 --width 832 --gpu 1

Phase 1 alone (skip T5 if you already have text embeddings):
  python -u models/pipeline_a/wan_precompute.py ... --skip_t5

Phase 2 alone (skip VAE if you already have video latents):
  python -u models/pipeline_a/wan_precompute.py ... --skip_vae
"""

import sys, os, argparse
from pathlib import Path
# wan_dir is passed via --wan_dir; added to sys.path at runtime in main()

import numpy as np
import torch
import torch.nn.functional as F


def phase1_vae(args, device, dtype):
    from wan.modules.vae import WanVAE

    print(f"[Phase1-VAE] loading VAE...", flush=True)
    vae = WanVAE(vae_pth=str(Path(args.ckpt_dir) / "Wan2.1_VAE.pth"))
    vae.model = vae.model.to(device, dtype=dtype)
    vae.model.requires_grad_(False).eval()
    # move scale tensors to the target device (they are created on cuda:0 by default)
    vae.mean = vae.mean.to(device)
    vae.std  = vae.std.to(device)
    vae.scale = [vae.mean, 1.0 / vae.std]

    lat_dir = Path(args.cache_dir) / "video_latent"
    lat_dir.mkdir(parents=True, exist_ok=True)
    H, W = args.height, args.width

    npz_list = sorted(Path(args.data_dir).glob("*.npz"))
    # shard the work across multiple workers
    if args.shard_world > 1:
        npz_list = npz_list[args.shard_rank::args.shard_world]
    print(f"[Phase1-VAE] rank={args.shard_rank}/{args.shard_world} "
          f"handling {len(npz_list)} segments", flush=True)

    done = 0
    for i, npz_path in enumerate(npz_list):
        stem = npz_path.stem.rsplit("_seg", 1)
        if len(stem) < 2:
            continue
        clip_name, seg_idx = stem[0], int(stem[1])
        key = f"{clip_name}_seg{seg_idx:03d}_{H}x{W}"
        lat_path = lat_dir / f"{key}.pt"
        if lat_path.exists():
            continue

        try:
            data      = np.load(npz_path, allow_pickle=True)
            frames_np = data["frames_256"]
        except Exception as e:
            print(f"  [SKIP] corrupted NPZ {npz_path.name}: {e}", flush=True)
            continue

        frames_f32 = torch.from_numpy(frames_np.astype(np.float32)) / 255.0 * 2.0 - 1.0
        if frames_f32.shape[2] != H or frames_f32.shape[3] != W:
            frames_f32 = F.interpolate(frames_f32, size=(H, W),
                                       mode="bilinear", align_corners=False)
        vid = frames_f32.permute(1, 0, 2, 3).to(device, dtype=dtype)   # (C, T, H, W)

        with torch.no_grad():
            lat_list = vae.encode([vid])
        latent = lat_list[0]   # (C_lat, T_lat, H_lat, W_lat)
        torch.save(latent.cpu(), lat_path)
        done += 1

        if (i + 1) % 200 == 0:
            print(f"  [Phase1] [{i+1}/{len(npz_list)}] done={done}", flush=True)

    print(f"[Phase1-VAE] done. Encoded {done} segments.", flush=True)
    del vae
    torch.cuda.empty_cache()


def phase2_t5(args, device, dtype):
    from wan.modules.t5 import T5EncoderModel

    # Collect all unique captions and their segment keys
    print("[Phase2-T5] scanning captions...", flush=True)
    H, W = args.height, args.width

    txt_dir = Path(args.cache_dir) / "text_emb"
    txt_dir.mkdir(parents=True, exist_ok=True)

    caption_to_keys = {}   # caption → list of (npz_path, key)
    npz_list = sorted(Path(args.data_dir).glob("*.npz"))

    for npz_path in npz_list:
        stem = npz_path.stem.rsplit("_seg", 1)
        if len(stem) < 2:
            continue
        clip_name, seg_idx = stem[0], int(stem[1])
        key = f"{clip_name}_seg{seg_idx:03d}_{H}x{W}"
        txt_path = txt_dir / f"{key}.pt"
        if txt_path.exists():
            continue

        try:
            data    = np.load(npz_path, allow_pickle=True)
            caption = str(data["caption"])[:512]
        except Exception as e:
            print(f"  [SKIP] corrupted NPZ {npz_path.name}: {e}", flush=True)
            continue
        if caption not in caption_to_keys:
            caption_to_keys[caption] = []
        caption_to_keys[caption].append((npz_path, key))

    unique_captions = list(caption_to_keys.keys())
    total_segs = sum(len(v) for v in caption_to_keys.values())
    print(f"  Unique captions to encode: {len(unique_captions)}", flush=True)
    print(f"  Total segments to cache:   {total_segs}", flush=True)

    if not unique_captions:
        print("[Phase2-T5] all text embeddings already cached.", flush=True)
        return

    # Load T5 on GPU (runs separately from VAE workers, plenty of VRAM)
    print(f"[Phase2-T5] loading T5 on {device}...", flush=True)
    t5_device = device
    t5 = T5EncoderModel(
        text_len        = 512,
        dtype           = dtype,   # bfloat16 on GPU (~5GB)
        device          = t5_device,
        checkpoint_path = str(Path(args.ckpt_dir) / "models_t5_umt5-xxl-enc-bf16.pth"),
        tokenizer_path  = str(Path(args.ckpt_dir) / "google" / "umt5-xxl"),
    )
    print("[Phase2-T5] T5 loaded. Encoding...", flush=True)

    # Encode in batches of unique captions
    batch_size = 8
    caption_emb_cache = {}   # caption → tensor

    for start in range(0, len(unique_captions), batch_size):
        batch_caps = unique_captions[start:start + batch_size]
        with torch.no_grad():
            embs = t5(batch_caps, device=t5_device)   # list of (L, C) tensors

        for cap, emb in zip(batch_caps, embs):
            caption_emb_cache[cap] = emb.cpu().unsqueeze(0)   # (1, L, C)

            # save for all segments with this caption
            for _, key in caption_to_keys[cap]:
                txt_path = txt_dir / f"{key}.pt"
                torch.save(caption_emb_cache[cap], txt_path)

        if (start // batch_size + 1) % 10 == 0:
            print(f"  [Phase2] [{start+len(batch_caps)}/{len(unique_captions)}] captions encoded",
                  flush=True)

    print(f"[Phase2-T5] done. {len(unique_captions)} unique captions → {total_segs} files cached.",
          flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--wan_dir",    required=True,
                        help="Path to Wan2.1 source directory (contains wan/ package)")
    parser.add_argument("--ckpt_dir",   default="models/wan2_1_1_3b")
    parser.add_argument("--data_dir",   default="data/processed/pipeline_a_v2/train")
    parser.add_argument("--cache_dir",  default="data/processed/pipeline_a_v2/wan_orig_cache_256")
    parser.add_argument("--height",     type=int, default=256)
    parser.add_argument("--width",      type=int, default=448)
    parser.add_argument("--gpu",        type=int, default=0)
    parser.add_argument("--skip_vae",   action="store_true")
    parser.add_argument("--skip_t5",    action="store_true")
    # data-parallel sharding: this worker handles shard rank/world_size
    parser.add_argument("--shard_rank", type=int, default=0)
    parser.add_argument("--shard_world",type=int, default=1)
    args = parser.parse_args()

    sys.path.insert(0, args.wan_dir)

    device = torch.device(f"cuda:{args.gpu}")
    dtype  = torch.bfloat16

    if not args.skip_vae:
        phase1_vae(args, device, dtype)

    # only rank 0 handles T5 (shared captions)
    if not args.skip_t5 and args.shard_rank == 0:
        phase2_t5(args, device, dtype)

    print("\n[done] precompute complete!", flush=True)

    # report counts
    lat_dir = Path(args.cache_dir) / "video_latent"
    txt_dir = Path(args.cache_dir) / "text_emb"
    n_lat = len(list(lat_dir.glob("*.pt"))) if lat_dir.exists() else 0
    n_txt = len(list(txt_dir.glob("*.pt"))) if txt_dir.exists() else 0
    print(f"  video_latent: {n_lat} files", flush=True)
    print(f"  text_emb:     {n_txt} files", flush=True)


if __name__ == "__main__":
    main()
