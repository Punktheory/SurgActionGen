"""
Cholec80 → LAOF data preprocessing pipeline.

Reads cholec80 MP4 clips, extracts frames at full resolution, computes RAFT
optical flow at 256×256, and saves everything at 256×256 as .npz chunks.

Fixes vs. previous version:
  1. Frames are resized from original resolution directly to raft_size (256)
     before RAFT — no more 64→256 upscale of already-degraded frames.
  2. flow_to_rgb now uses COLOR_HSV2RGB (not BGR) so channels are consistent
     with how PyTorch dataloaders read the arrays.
  3. Flow magnitude is clipped to a global MAX_FLOW constant instead of
     per-image normalisation — static frames are dark, fast frames are bright,
     absolute speed information is preserved.
  4. Multi-GPU: clips are sharded across GPUs via torch.multiprocessing.
     Each GPU worker processes its own shard and writes its own chunks.
     A final merge step renumbers chunks into a single contiguous sequence.

Output structure:
  data/processed/cholec80_laof_256/
    train/  chunks of TRAIN_CHUNK_SIZE frames, 256×256
    test/   chunks of TEST_CHUNK_SIZE  frames, 256×256

Each chunk.npz contains:
  obs      : (N, 256, 256, 3) uint8  — RGB frames
  obs_of   : (N, 256, 256, 3) uint8  — optical flow as HSV-RGB (RGB channel order)
  done     : (N,)         bool       — True at clip boundaries

Usage (8 GPUs):
  conda activate raft
  python tests/prepare_cholec80_laof.py --gpus 0,1,2,3,4,5,6,7
"""

import argparse
import os
import shutil
from pathlib import Path
import multiprocessing as mp

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from torchvision.models.optical_flow import Raft_Large_Weights, raft_large
from tqdm import tqdm

# ─── paths (set via --clip_dir / --out_dir argparse args) ────────────────────
CLIP_DIR = None  # overridden in parse_args()
OUT_DIR  = None  # overridden in parse_args()

TRAIN_CHUNK_SIZE = 32_768
TEST_CHUNK_SIZE  = 4_096
TRAIN_RATIO      = 0.8

# Maximum optical flow magnitude (pixels at raft_size resolution) mapped to
# pixel value 255. Flows larger than this are clipped. 20px covers the vast
# majority of inter-frame motion in 25fps surgical video.
MAX_FLOW = 20.0


# ─── argument parsing ─────────────────────────────────────────────────────────
def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--clip_dir",     type=str,   required=True,
                   help="Path to Cholec80 MP4 clips directory")
    p.add_argument("--out_dir",      type=str,   default="data/processed/cholec80_laof_256",
                   help="Output directory for NPZ chunks")
    p.add_argument("--img_size",     type=int,   default=256,
                   help="storage resolution for both frames and flow (default 256)")
    p.add_argument("--frame_stride", type=int,   default=1)
    p.add_argument("--batch_size",   type=int,   default=16,
                   help="RAFT inference batch size per GPU")
    p.add_argument("--gpus",         type=str,   default="0",
                   help="comma-separated GPU ids, e.g. 0,1,2,3,4,5,6,7")
    args = p.parse_args()
    global CLIP_DIR, OUT_DIR
    CLIP_DIR = Path(args.clip_dir)
    OUT_DIR  = Path(args.out_dir)
    return args


# ─── RAFT ─────────────────────────────────────────────────────────────────────
def load_raft(device: str):
    weights = Raft_Large_Weights.DEFAULT
    model = raft_large(weights=weights).to(device).eval()
    transforms = weights.transforms()
    return model, transforms


@torch.no_grad()
def compute_flow_batch(model, transforms, frames_t, frames_t1, device, img_size):
    """
    Args:
        frames_t, frames_t1 : (B, img_size, img_size, 3) uint8 RGB
    Returns:
        flow_rgb : (B, img_size, img_size, 3) uint8 HSV-RGB (RGB channel order)
    """
    def to_tensor(frames):
        return torch.from_numpy(frames).permute(0, 3, 1, 2).to(device)

    img1 = to_tensor(frames_t)
    img2 = to_tensor(frames_t1)

    # Apply official RAFT transforms: uint8 [0,255] → normalized float, contrast enhanced
    img1_n, img2_n = transforms(img1, img2)

    flow_preds = model(img1_n, img2_n)
    flow = flow_preds[-1].cpu().numpy()  # (B, 2, H, W)

    return flow_to_rgb_batch(flow)


def flow_to_rgb_batch(flow: np.ndarray) -> np.ndarray:
    """
    Convert optical flow (B, 2, H, W) → HSV-RGB image (B, H, W, 3) uint8.

    Encoding: Hue=direction, Saturation=magnitude (clipped to MAX_FLOW), Value=255.
    Static regions → S=0, V=255 → white/light lavender.
    Moving regions → S high → saturated color.
    """
    B, _, H, W = flow.shape
    out = np.zeros((B, H, W, 3), dtype=np.uint8)
    for i in range(B):
        u = flow[i, 0]
        v = flow[i, 1]
        mag, ang = cv2.cartToPolar(u, v)
        hsv = np.zeros((H, W, 3), dtype=np.uint8)
        hsv[..., 0] = (ang * 180 / np.pi / 2).astype(np.uint8)          # Hue [0,180]
        hsv[..., 1] = np.clip(mag / MAX_FLOW * 255, 0, 255).astype(np.uint8)  # Saturation = magnitude
        hsv[..., 2] = 255                                                  # Value always bright
        out[i] = cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB)
    return out


# ─── frame extraction ─────────────────────────────────────────────────────────
def extract_frames(clip_path: Path, img_size: int, stride: int) -> np.ndarray:
    """
    Extract frames from MP4, resize directly from original resolution to
    img_size (no intermediate downscale). Returns (N, img_size, img_size, 3)
    uint8 RGB.
    """
    cap = cv2.VideoCapture(str(clip_path))
    frames = []
    idx = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        if idx % stride == 0:
            # cv2 reads BGR; convert to RGB before resizing
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame_rgb = cv2.resize(frame_rgb, (img_size, img_size),
                                   interpolation=cv2.INTER_AREA)
            frames.append(frame_rgb)
        idx += 1
    cap.release()
    if frames:
        return np.stack(frames, axis=0)
    return np.zeros((0, img_size, img_size, 3), dtype=np.uint8)


# ─── chunk writer ─────────────────────────────────────────────────────────────
class ChunkWriter:
    def __init__(self, out_dir: Path, chunk_size: int, split: str, gpu_id: int):
        self.out_dir    = out_dir
        self.chunk_size = chunk_size
        self.split      = split
        self.gpu_id     = gpu_id
        self.chunk_idx  = 0
        self._obs_buf   = []
        self._of_buf    = []
        self._done_buf  = []
        self._buf_len   = 0
        out_dir.mkdir(parents=True, exist_ok=True)

    def add_clip(self, obs, of_, done):
        self._obs_buf.append(obs)
        self._of_buf.append(of_)
        self._done_buf.append(done)
        self._buf_len += len(obs)
        while self._buf_len >= self.chunk_size:
            self._flush_one_chunk()

    def _flush_one_chunk(self):
        obs  = np.concatenate(self._obs_buf,  axis=0)
        of_  = np.concatenate(self._of_buf,   axis=0)
        done = np.concatenate(self._done_buf, axis=0)

        chunk_obs  = obs[:self.chunk_size]
        chunk_of   = of_[:self.chunk_size]
        chunk_done = done[:self.chunk_size]

        self._obs_buf  = [obs[self.chunk_size:]]  if len(obs)  > self.chunk_size else []
        self._of_buf   = [of_[self.chunk_size:]]  if len(of_)  > self.chunk_size else []
        self._done_buf = [done[self.chunk_size:]] if len(done) > self.chunk_size else []
        self._buf_len  = len(self._obs_buf[0]) if self._obs_buf else 0

        # prefix with gpu_id so shards don't collide on disk
        fname = f"gpu{self.gpu_id:02d}_{self.chunk_idx:04d}.npz"
        out_path = self.out_dir / fname
        np.savez_compressed(out_path,
                            obs=chunk_obs, obs_of=chunk_of, done=chunk_done)
        print(f"  [gpu{self.gpu_id} {self.split}] chunk {self.chunk_idx} → {fname} "
              f"({len(chunk_obs)} frames, {chunk_done.sum()} boundaries)", flush=True)
        self.chunk_idx += 1

    def flush_remainder(self):
        if not self._obs_buf:
            return
        obs  = np.concatenate(self._obs_buf,  axis=0)
        of_  = np.concatenate(self._of_buf,   axis=0)
        done = np.concatenate(self._done_buf, axis=0)
        if len(obs) == 0:
            return
        fname = f"gpu{self.gpu_id:02d}_{self.chunk_idx:04d}.npz"
        out_path = self.out_dir / fname
        np.savez_compressed(out_path, obs=obs, obs_of=of_, done=done)
        print(f"  [gpu{self.gpu_id} {self.split}] remainder → {fname} ({len(obs)} frames)", flush=True)
        self.chunk_idx += 1


# ─── per-GPU worker ───────────────────────────────────────────────────────────
def worker(gpu_id: int, clip_shard: list, split: str, args):
    """Process one shard of clips on a single GPU."""
    device = f"cuda:{gpu_id}"
    print(f"[gpu{gpu_id}] starting — {len(clip_shard)} clips, split={split}", flush=True)

    raft, raft_transforms = load_raft(device)
    chunk_size = TRAIN_CHUNK_SIZE if split == "train" else TEST_CHUNK_SIZE
    writer = ChunkWriter(OUT_DIR / split, chunk_size, split, gpu_id)

    for clip_path in tqdm(clip_shard, desc=f"gpu{gpu_id}", position=gpu_id, leave=True):
        frames = extract_frames(clip_path, args.img_size, args.frame_stride)
        N = len(frames)
        if N < 2:
            continue

        obs_of = np.zeros_like(frames)
        done   = np.zeros(N, dtype=bool)
        done[-1] = True

        n_pairs = N - 1
        for start in range(0, n_pairs, args.batch_size):
            end = min(start + args.batch_size, n_pairs)
            f_t  = frames[start:end]
            f_t1 = frames[start + 1:end + 1]
            obs_of[start:end] = compute_flow_batch(
                raft, raft_transforms, f_t, f_t1, device, args.img_size)

        writer.add_clip(frames, obs_of, done)

    writer.flush_remainder()
    print(f"[gpu{gpu_id}] done — {writer.chunk_idx} chunks written", flush=True)


# ─── merge: renumber gpu shards into contiguous 0000.npz, 0001.npz … ─────────
def merge_shards(split: str):
    split_dir = OUT_DIR / split
    shards = sorted(split_dir.glob("gpu*.npz"))
    print(f"\n[merge {split}] renaming {len(shards)} shard files …")
    for new_idx, shard in enumerate(shards):
        new_name = split_dir / f"{new_idx:04d}.npz"
        shutil.move(str(shard), str(new_name))
    print(f"[merge {split}] done — {len(shards)} chunks in {split_dir}")


# ─── main ─────────────────────────────────────────────────────────────────────
def main():
    args = parse_args()
    gpu_ids = [int(g) for g in args.gpus.split(",")]
    n_gpus  = len(gpu_ids)

    print(f"GPUs        : {gpu_ids}")
    print(f"img_size    : {args.img_size}×{args.img_size}")
    print(f"batch_size  : {args.batch_size} pairs/GPU")
    print(f"MAX_FLOW    : {MAX_FLOW} px  (global magnitude normalisation)")
    print(f"Output dir  : {OUT_DIR}")

    all_clips = sorted(CLIP_DIR.glob("*.mp4"))
    if not all_clips:
        raise RuntimeError(f"No .mp4 files found in {CLIP_DIR}")
    print(f"Total clips : {len(all_clips)}")

    n_train = int(len(all_clips) * TRAIN_RATIO)
    train_clips = all_clips[:n_train]
    test_clips  = all_clips[n_train:]
    print(f"Train clips : {len(train_clips)}")
    print(f"Test  clips : {len(test_clips)}\n")

    # ── train: shard across all GPUs ─────────────────────────────────────────
    print("=== TRAIN ===")
    train_shards = [train_clips[i::n_gpus] for i in range(n_gpus)]
    procs = []
    mp.set_start_method("spawn", force=True)
    for i, gid in enumerate(gpu_ids):
        p = mp.Process(target=worker,
                       args=(gid, train_shards[i], "train", args))
        p.start()
        procs.append(p)
    for p in procs:
        p.join()
    merge_shards("train")

    # ── test: one GPU is enough (small set) ──────────────────────────────────
    print("\n=== TEST ===")
    # still shard across all GPUs for speed
    test_shards = [test_clips[i::n_gpus] for i in range(n_gpus)]
    procs = []
    for i, gid in enumerate(gpu_ids):
        shard = test_shards[i]
        if not shard:
            continue
        p = mp.Process(target=worker,
                       args=(gid, shard, "test", args))
        p.start()
        procs.append(p)
    for p in procs:
        p.join()
    merge_shards("test")

    print("\n✓ Preprocessing complete.")
    print(f"  Train → {OUT_DIR / 'train'}")
    print(f"  Test  → {OUT_DIR / 'test'}")


if __name__ == "__main__":
    main()
