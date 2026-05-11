"""
Cholec80 data loader for LAOF Stage 1 training.

Reads .npz chunks produced by tests/prepare_cholec80_laof.py and yields
TensorDict batches compatible with LAOF's IDM/WM/FlowDecoder training.

Chunk npz format  (N frames per chunk):
  obs    : (N, H, W, 3) uint8  — RGB frames
  obs_of : (N, H, W, 3) uint8  — optical flow as HSV-RGB
  done   : (N,)         bool   — True at clip boundaries

Batch format returned by get_iter():
  obs    : (B, 2, 3, H, W)  float32  [-0.5, 0.5]   — (frame_t, frame_t+1)
  obs_of : (B, 2, 3, H, W)  float32  [-0.5, 0.5]   — (flow_t→t+1, zero)
  ta     : (B, 2)            long     0              — dummy action label
  done   : (B, 2)            bool                   — episode boundary flags

RAM strategy: one chunk at a time per rank.
  - At 128×128, one resized chunk = ~1.6 GB.
  - 8 ranks × 1 chunk = ~13 GB peak — well within 233 GB.
  - ConcatDataset across all chunks is avoided: that caused workers to load
    all 7 chunks simultaneously (7 × 4.2 GB = 29 GB per worker → OOM).
"""

from __future__ import annotations

import random
from pathlib import Path
from typing import Generator

import numpy as np
import torch
import torch.nn.functional as F
from tensordict import TensorDict
from torch.utils.data import DataLoader, Dataset

# data root: points to 256×256 preprocessed data; loader resizes to MODEL_IMG_SIZE
_DATA_ROOT     = Path(__file__).resolve().parents[3] / "data" / "processed" / "cholec80_laof_256"
MODEL_IMG_SIZE = 128  # model input resolution (IDM/WM/FlowDecoder all expect this)


# ─── normalisation ────────────────────────────────────────────────────────────
def _normalize(x: torch.Tensor) -> torch.Tensor:
    """uint8 [0,255] → float32 [-0.5, 0.5]"""
    return x.float() / 255.0 - 0.5


# ─── per-chunk dataset ────────────────────────────────────────────────────────
class Cholec80ChunkDataset(Dataset):
    """
    Loads ONE .npz chunk into RAM (resized to MODEL_IMG_SIZE) and serves
    sliding-window frame pairs from it.

    Only one chunk is loaded at a time per rank — avoids the OOM that
    occurs when ConcatDataset triggers loading of all chunks simultaneously.
    """

    def __init__(self, path: Path, stride: int = 1, min_diff: float = 0.0):
        """
        stride   : sample (obs_t, obs_{t+stride}) instead of adjacent frames.
                   Increases motion signal at the cost of fewer valid pairs.
        min_diff : skip pairs where mean |obs_{t+stride} - obs_t| < min_diff
                   (in uint8 space, 0–255). Applied lazily in __getitem__.
        """
        self.path     = path
        self.stride   = stride
        self.min_diff = min_diff
        self._obs     = None
        self._obs_of  = None

        # Read done array only (tiny) to build valid index without loading frames.
        data = np.load(path, mmap_mode='r')
        done = np.array(data["done"])
        self.done = torch.from_numpy(done)
        # valid_idx: i is valid iff done[i..i+stride-1] are all False
        # (no clip boundary crossed within the stride window)
        N = len(done)
        if stride == 1:
            self.valid_idx = torch.where(~self.done[:-1])[0]
        else:
            # rolling OR: any done in [i, i+stride) disqualifies i
            done_t = self.done.float()
            # cs has length N+1; sum(done[i:i+stride]) = cs[i+stride] - cs[i]
            # for i in [0, N-stride): length = N-stride, indices i+stride in [stride, N)
            cs = torch.cat([torch.zeros(1), done_t.cumsum(0)])  # (N+1,)
            window_sum = cs[stride:N] - cs[:N - stride]          # (N-stride,)
            self.valid_idx = torch.where(window_sum == 0)[0]

    def _ensure_loaded(self):
        if self._obs is not None:
            return
        data = np.load(self.path)
        obs_np    = data["obs"]    # (N, H, W, 3) uint8
        obs_of_np = data["obs_of"]
        N = obs_np.shape[0]
        # Resize in small batches to avoid a 8 GB float32 spike (256-chunk × 256px).
        # Process 512 frames at a time: peak per batch ≈ 512×3×256×256×4B ≈ 400 MB.
        BATCH = 512
        if MODEL_IMG_SIZE != 256:
            obs_out    = torch.empty(N, 3, MODEL_IMG_SIZE, MODEL_IMG_SIZE, dtype=torch.uint8)
            obs_of_out = torch.empty(N, 3, MODEL_IMG_SIZE, MODEL_IMG_SIZE, dtype=torch.uint8)
            for start in range(0, N, BATCH):
                end = min(start + BATCH, N)
                b_obs = torch.from_numpy(obs_np[start:end]).permute(0, 3, 1, 2).float()
                obs_out[start:end] = F.interpolate(b_obs, size=MODEL_IMG_SIZE,
                                                   mode="bilinear", align_corners=False).byte()
                b_of = torch.from_numpy(obs_of_np[start:end]).permute(0, 3, 1, 2).float()
                obs_of_out[start:end] = F.interpolate(b_of, size=MODEL_IMG_SIZE,
                                                      mode="bilinear", align_corners=False).byte()
            self._obs    = obs_out
            self._obs_of = obs_of_out
        else:
            self._obs    = torch.from_numpy(obs_np).permute(0, 3, 1, 2).contiguous()
            self._obs_of = torch.from_numpy(obs_of_np).permute(0, 3, 1, 2).contiguous()

    def __len__(self) -> int:
        return len(self.valid_idx)

    def __getitem__(self, idx: int) -> dict:
        self._ensure_loaded()
        i  = int(self.valid_idx[idx])
        j  = i + self.stride
        obs_pair    = torch.stack([self._obs[i],    self._obs[j]],    dim=0)
        # obs_of[i] is the flow for the t→t+1 step; with stride>1 there is no
        # stored flow for t→t+stride, so we reuse obs_of[i] as a proxy signal.
        obs_of_pair = torch.stack([self._obs_of[i], self._obs_of[j]], dim=0)
        done_pair   = torch.stack([self.done[i],    self.done[j]],    dim=0)
        ta_pair     = torch.zeros(2, dtype=torch.long)
        return {
            "obs":     obs_pair,
            "obs_of":  obs_of_pair,
            "obs_sam": obs_of_pair,
            "done":    done_pair,
            "ta":      ta_pair,
        }


# ─── multi-chunk loader ───────────────────────────────────────────────────────
class Cholec80DataLoader:
    """
    Cycles through .npz chunk files one at a time, keeping only one chunk
    in RAM per rank at any moment.

    get_iter() yields an infinite stream of normalised TensorDict batches.
    Chunks are shuffled each epoch; within each chunk samples are shuffled.
    """

    def __init__(self, split: str, max_chunks: int | None = None,
                 stride: int = 1, min_diff: float = 0.0):
        split_dir = _DATA_ROOT / split
        if not split_dir.exists():
            raise FileNotFoundError(
                f"Cholec80 LAOF data not found at {split_dir}.\n"
                f"Run tests/prepare_cholec80_laof.py first."
            )
        chunk_files = sorted(split_dir.glob("*.npz"))
        if max_chunks is not None:
            chunk_files = chunk_files[:max_chunks]
        if not chunk_files:
            raise FileNotFoundError(f"No .npz chunks found in {split_dir}")

        self.chunk_files = chunk_files
        # Build valid_idx for each chunk (reads only done arrays — fast)
        self.datasets = [Cholec80ChunkDataset(p, stride=stride, min_diff=min_diff)
                         for p in chunk_files]
        self._total   = sum(len(d) for d in self.datasets)
        stride_str = f"  stride={stride}" if stride > 1 else ""
        print(f"[Cholec80DataLoader:{split}] {len(self.datasets)} chunks, "
              f"{self._total:,} valid frame-pairs{stride_str}")

    def __len__(self) -> int:
        return self._total

    def get_iter(
        self,
        batch_size: int,
        infinite: bool = True,
        shuffle_buffer: int = 0,
        device: str = "cpu",
    ) -> Generator[TensorDict, None, None]:
        """
        Yield normalised TensorDict batches.
        One chunk is loaded at a time; chunks are shuffled each epoch.
        """
        order = list(range(len(self.datasets)))
        while True:
            random.shuffle(order)
            for chunk_idx in order:
                ds = self.datasets[chunk_idx]
                # Load this chunk (only if not already loaded from a prior pass)
                loader = DataLoader(
                    ds,
                    batch_size=batch_size,
                    shuffle=True,
                    drop_last=True,
                    num_workers=0,   # main process: chunk already in RAM, no need for workers
                    collate_fn=_collate,
                )
                for batch in loader:
                    batch["obs"]     = _normalize(batch["obs"]).to(device)
                    batch["obs_of"]  = _normalize(batch["obs_of"]).to(device)
                    batch["obs_sam"] = _normalize(batch["obs_sam"]).to(device)
                    batch["done"]    = batch["done"].to(device)
                    batch["ta"]      = batch["ta"].to(device)
                    yield batch
            if not infinite:
                break


def _collate(samples: list[dict]) -> TensorDict:
    keys = samples[0].keys()
    stacked = {k: torch.stack([s[k] for s in samples], dim=0) for k in keys}
    return TensorDict(stacked, batch_size=len(samples))


# ─── convenience factory ──────────────────────────────────────────────────────
def load(env_name: str = "cholec80",
         max_test_chunks: int = 5,
         stride: int = 1,
         min_diff: float = 0.0) -> tuple[Cholec80DataLoader, Cholec80DataLoader, Cholec80DataLoader]:
    assert env_name == "cholec80", f"cholec80_data_loader only supports env_name='cholec80', got '{env_name}'"
    train = Cholec80DataLoader("train", stride=stride, min_diff=min_diff)
    test  = Cholec80DataLoader("test", max_chunks=max_test_chunks)  # test always stride=1
    return train, test, test
