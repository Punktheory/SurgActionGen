"""
Pipeline A — Step A1 v3: Extract (frames_256, z_seq) from existing cholec80_laof_256 chunks.

v2 was wrong: it re-ran RAFT from raw MP4s, duplicating work already done in
prepare_cholec80_laof.py. This v3 reads directly from the already-preprocessed
cholec80_laof_256 NPZ chunks (obs + obs_of already computed), runs only FAE
encode to get z_seq, and writes pipeline_a_v2 NPZs.

Source: data/processed/cholec80_laof_256/{train,test}/*.npz
  obs    : (N, 256, 256, 3) uint8  — RGB frames, contiguous across clips
  obs_of : (N, 256, 256, 3) uint8  — optical flow HSV-RGB
  done   : (N,) bool               — True at the LAST frame of each clip

Caption matching:
  Clips were stored in alphabetical MP4 order (video01_clip01, video01_clip02, …).
  done=True marks the last frame of each clip. We reconstruct clip boundaries
  from done, then zip with the sorted MP4 list to get clip names and captions.
  Clips shorter than T frames (same ones skipped in prepare_cholec80_laof.py)
  are already absent from the chunks — no need to filter again.

Output: data/processed/pipeline_a_v2/{train|test}/{clip_name}_seg{i:03d}.npz
  frames_256 : (T, 3, 256, 256) uint8
  obs_of_256 : (T, 3, 256, 256) uint8
  z_seq      : (T, 128)         float32, L2-normalized (norm=1.0)
  caption    : str
  triplets   : list
  clip_name  : str
  seg_idx    : int

Run (single GPU, from project root):
  python -u laof/pipeline_a_extract_v3.py --gpu 0

Run (multi-GPU — split by chunk files):
  for i in 0 1 2 3 4 5 6 7; do
    python -u laof/pipeline_a_extract_v3.py \
        --gpu $i --shard $i --num_shards 8 &
  done; wait
"""

import os, sys, json, argparse
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

def _strip_conda_libs():
    ld = os.environ.get("LD_LIBRARY_PATH", "")
    bad = ("/miniconda3/lib", "/anaconda3/lib", "/Micromamba/")
    parts = [p for p in ld.split(":") if p.strip() and not any(b in p for b in bad)]
    os.environ["LD_LIBRARY_PATH"] = ":".join(parts)
_strip_conda_libs()

import numpy as np
import torch
import torch.nn.functional as F
from pathlib import Path
from omegaconf import OmegaConf


# ── FAE loader ─────────────────────────────────────────────────────────────────

def load_fae(fae_ckpt: str, device):
    from utils import utils
    conf_path = Path(__file__).parent / "conf" / "defualt.yaml"
    cfg = OmegaConf.load(conf_path)
    cfg.model.la_dim = 128
    cfg.model.ta_dim = 7
    cfg.stage3.batch_size = cfg.stage3.num_envs * cfg.stage3.num_steps
    cfg.stage3.minibatch_size = cfg.stage3.batch_size // cfg.stage3.num_minibatches
    sd = torch.load(fae_ckpt, map_location="cpu")
    fae, _, _ = utils.create_dynamics_models_flow_fae(cfg.model)
    fae.load_state_dict(sd["fae"])
    fae.eval().to(device)
    print(f"[FAE] loaded from {fae_ckpt}  step={sd.get('step','?')}", flush=True)
    return fae


@torch.no_grad()
def encode_z_batched(fae, obs_of_hwc: np.ndarray, device, batch_size=512) -> np.ndarray:
    """
    obs_of_hwc : (N, H, W, 3) uint8 — optical flow HSV-RGB at any resolution
    Returns    : (N, 128) float32, L2-normalized
    """
    # uint8 → float32 [-0.5, 0.5], (N, 3, H, W)
    x = torch.from_numpy(obs_of_hwc).float().permute(0, 3, 1, 2) / 255.0 - 0.5
    # resize to 128×128 (FAE model input size)
    if x.shape[-1] != 128 or x.shape[-2] != 128:
        x = F.interpolate(x, size=(128, 128), mode="bilinear", align_corners=False)
    z_list = []
    for start in range(0, len(x), batch_size):
        z = fae.encode(x[start:start + batch_size].to(device))
        z_list.append(z.cpu())
    return torch.cat(z_list).numpy()   # (N, 128), norm=1.0


# ── clip boundary reconstruction ───────────────────────────────────────────────

def split_chunk_into_clips(obs, obs_of, done):
    """
    Given one NPZ chunk's arrays, split into per-clip lists using done flags.
    done[i]=True means frame i is the LAST frame of a clip.

    Returns list of (obs_clip, obs_of_clip) tuples, each (L, H, W, 3) uint8.
    """
    clips = []
    start = 0
    for i, is_done in enumerate(done):
        if is_done:
            clips.append((obs[start:i+1], obs_of[start:i+1]))
            start = i + 1
    # trailing frames without a done marker (shouldn't happen but handle anyway)
    if start < len(done):
        clips.append((obs[start:], obs_of[start:]))
    return clips


# ── main ───────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--laof_dir",     default="data/processed/cholec80_laof_256")
    parser.add_argument("--clips_dir",    default="data/raw/cholec80_action/clips")
    parser.add_argument("--captions_json",default="data/raw/cholec80_action/clips_captions.json")
    parser.add_argument("--fae_ckpt",     default="LAOF/laof/exp_results/cholec80/idm_fdm_v6.pt")
    parser.add_argument("--out_dir",      default="data/processed/pipeline_a_v2")
    parser.add_argument("--T",            type=int, default=32)
    parser.add_argument("--fae_batch",    type=int, default=512)
    parser.add_argument("--gpu",          type=int, default=0)
    parser.add_argument("--shard",        type=int, default=0)
    parser.add_argument("--num_shards",   type=int, default=1)
    parser.add_argument("--no_skip",      action="store_true")
    args = parser.parse_args()

    proj_root     = Path(__file__).resolve().parents[2]
    laof_dir      = proj_root / args.laof_dir
    clips_dir     = proj_root / args.clips_dir
    captions_json = proj_root / args.captions_json
    fae_ckpt      = proj_root / args.fae_ckpt
    out_dir       = proj_root / args.out_dir

    device = torch.device(f"cuda:{args.gpu}")
    print(f"[init] device={device}  shard={args.shard}/{args.num_shards}", flush=True)

    fae = load_fae(str(fae_ckpt), device)

    # load captions
    with open(captions_json) as f:
        caps_list = json.load(f)
    cap_map = {e["video"]: e for e in caps_list}

    # build ordered clip name lists matching exactly what prepare_cholec80_laof.py stored.
    # prepare skipped clips with < T frames — we must skip the same ones here so that
    # the clip index into clip_names_ordered matches the clip index in the chunks.
    all_clips = sorted(clips_dir.glob("*.mp4"))
    def vid_num(p): return int(p.stem.split("_")[0].replace("video", ""))

    import cv2
    def frame_count(mp4_path):
        cap = cv2.VideoCapture(str(mp4_path))
        n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        cap.release()
        return n

    train_clip_names = [c.name for c in all_clips
                        if vid_num(c) < 73 and frame_count(c) >= args.T]
    test_clip_names  = [c.name for c in all_clips
                        if vid_num(c) >= 73 and frame_count(c) >= args.T]
    print(f"[data] {len(train_clip_names)} train clips (>={args.T}f), "
          f"{len(test_clip_names)} test clips (>={args.T}f)", flush=True)

    (out_dir / "train").mkdir(parents=True, exist_ok=True)
    (out_dir / "test").mkdir(parents=True, exist_ok=True)

    def process_split(split: str, chunk_paths, clip_names_ordered):
        # shard by chunk index
        my_chunks = chunk_paths[args.shard::args.num_shards]
        print(f"\n[{split}] {len(my_chunks)} chunks (shard {args.shard})", flush=True)

        # we need to know how many clips precede each chunk to index clip_names_ordered
        # build a cumulative clip count across ALL chunks (not just my shard)
        clip_offsets = []
        cumulative = 0
        for p in chunk_paths:
            clip_offsets.append(cumulative)
            done = np.load(p, mmap_mode='r')['done']
            cumulative += int(done.sum())
        total_clips_in_chunks = cumulative
        print(f"[{split}] total clips in chunks: {total_clips_in_chunks}  "
              f"ordered clip names: {len(clip_names_ordered)}", flush=True)

        total_segs = skipped = 0
        for chunk_path in my_chunks:
            chunk_idx = chunk_paths.index(chunk_path)
            clip_offset = clip_offsets[chunk_idx]

            print(f"  [chunk {chunk_idx:03d}] loading...", flush=True)
            data   = np.load(chunk_path)
            obs    = data["obs"]      # (N, 256, 256, 3) uint8
            obs_of = data["obs_of"]   # (N, 256, 256, 3) uint8
            done   = data["done"]     # (N,) bool

            clips = split_chunk_into_clips(obs, obs_of, done)
            print(f"  [chunk {chunk_idx:03d}] {len(clips)} clips, encoding z...", flush=True)

            for local_i, (frames, flows) in enumerate(clips):
                global_clip_idx = clip_offset + local_i
                if global_clip_idx >= len(clip_names_ordered):
                    print(f"  [WARN] clip index {global_clip_idx} out of range, skip", flush=True)
                    continue

                clip_filename = clip_names_ordered[global_clip_idx]
                clip_name     = clip_filename.replace(".mp4", "")
                meta          = cap_map.get(clip_filename, {})
                caption       = meta.get("caption", "")
                triplets      = meta.get("triplets", [])

                N = len(frames)
                if N < args.T:
                    skipped += 1
                    continue

                seg0_path = out_dir / split / f"{clip_name}_seg000.npz"
                if not args.no_skip and seg0_path.exists():
                    skipped += 1
                    continue

                # encode all frames in this clip with FAE
                z_all = encode_z_batched(fae, flows, device, batch_size=args.fae_batch)

                # segment into T-frame windows
                frames_chw = frames.transpose(0, 3, 1, 2)  # (N, 3, H, W)
                flows_chw  = flows.transpose(0, 3, 1, 2)

                num_segs = N // args.T
                for seg_i in range(num_segs):
                    s, e = seg_i * args.T, seg_i * args.T + args.T
                    np.savez_compressed(
                        out_dir / split / f"{clip_name}_seg{seg_i:03d}.npz",
                        frames_256 = frames_chw[s:e].astype(np.uint8),
                        obs_of_256 = flows_chw[s:e].astype(np.uint8),
                        z_seq      = z_all[s:e].astype(np.float32),
                        caption    = np.array(caption,   dtype=object),
                        triplets   = np.array(triplets,  dtype=object),
                        clip_name  = np.array(clip_name, dtype=object),
                        seg_idx    = np.array(seg_i,     dtype=np.int32),
                    )
                    total_segs += 1

            print(f"  [chunk {chunk_idx:03d}] done  total_segs={total_segs}", flush=True)

        print(f"[{split}] DONE  total_segs={total_segs}  skipped={skipped}", flush=True)

    train_chunks = sorted((laof_dir / "train").glob("*.npz"))
    test_chunks  = sorted((laof_dir / "test").glob("*.npz"))

    process_split("train", train_chunks, train_clip_names)
    process_split("test",  test_chunks,  test_clip_names)

    print("\n[ALL DONE]", flush=True)


if __name__ == "__main__":
    main()
