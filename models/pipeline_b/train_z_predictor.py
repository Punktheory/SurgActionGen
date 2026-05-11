"""
Pipeline B — Non-autoregressive Transformer: caption → z_seq (32×128)

Architecture:
  CLIPTextModel (frozen) → pooled text embedding (768,)
       ↓  Linear(768 → d_model)
  32 learned position queries (32, d_model)
       ↓  Transformer Decoder (n_layers, d_model, n_heads)
          cross-attention on text context
       ↓  Linear(d_model → 128) + F.normalize
  z_pred (32, 128)

Loss: mean over frames of (1 - cosine_similarity(z_pred, z_true, dim=-1))

Training data: data/processed/pipeline_a_v2/train/*.npz
  each NPZ: z_seq (32, 128) float32 L2-norm=1, caption str

Run (single GPU, from project root):
  conda activate cogvideox
  python -u models/pipeline_b/train_z_predictor.py --gpu 0

Run (multi-GPU DDP, 8 GPUs):
  torchrun \\
      --nproc_per_node=8 --master_port=29520 \\
      models/pipeline_b/train_z_predictor.py \\
      --total_steps 10000 --batch_size 64 --lr 3e-4 --gpu 0

Checkpoint: exp_results/pipeline_b/z_predictor.pt
  keys: step, model, optim
"""

import os, sys, argparse
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.utils.tensorboard import SummaryWriter


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


# ── Dataset ───────────────────────────────────────────────────────────────────

class ZSeqDataset(Dataset):
    def __init__(self, data_dir: str, T: int = 32):
        self.T = T
        # pre-filter: skip corrupted / empty NPZ files
        all_paths = sorted(Path(data_dir).glob("*.npz"))
        self.samples = []
        skipped = 0
        for p in all_paths:
            try:
                d = np.load(p, allow_pickle=True)
                _ = d["z_seq"]
                _ = str(d["caption"])
                self.samples.append(p)
            except Exception:
                skipped += 1
        if skipped:
            print(f"[ZSeqDataset] skipped {skipped} corrupted NPZ files", flush=True)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        d = np.load(self.samples[idx], allow_pickle=True)
        z = torch.from_numpy(d["z_seq"].astype(np.float32))   # (N, 128)
        N = z.shape[0]
        if N < self.T:
            z = torch.cat([z, z[-1:].expand(self.T - N, -1)], dim=0)
        z = z[:self.T]                                          # (T, 128)
        caption = str(d["caption"])
        return caption, z


def collate_fn(batch):
    captions = [b[0] for b in batch]
    z = torch.stack([b[1] for b in batch])   # (B, T, 128)
    return captions, z


# ── CLIP text encoder (frozen) ────────────────────────────────────────────────

class FrozenCLIPText(nn.Module):
    def __init__(self, model_id: str = "openai/clip-vit-large-patch14"):
        super().__init__()
        from transformers import CLIPTextModel, CLIPTokenizer
        self.tokenizer = CLIPTokenizer.from_pretrained(model_id)
        self.model     = CLIPTextModel.from_pretrained(model_id)
        self.model.eval().requires_grad_(False)

    @torch.no_grad()
    def forward(self, captions: list, device) -> torch.Tensor:
        inp = self.tokenizer(
            captions, return_tensors="pt",
            padding=True, truncation=True, max_length=77
        )
        inp = {k: v.to(device) for k, v in inp.items()}
        out = self.model(**inp)
        return out.pooler_output   # (B, 768)


# ── Z Predictor ───────────────────────────────────────────────────────────────

class ZPredictor(nn.Module):
    """Non-autoregressive Transformer: text embedding → z_seq (T, 128)"""

    def __init__(
        self,
        text_dim:  int = 768,
        d_model:   int = 256,
        n_heads:   int = 4,
        n_layers:  int = 4,
        T:         int = 32,
        z_dim:     int = 128,
    ):
        super().__init__()
        self.T = T

        # project text embedding to d_model
        self.text_proj = nn.Linear(text_dim, d_model)

        # learned position queries: (T, d_model)
        self.queries = nn.Parameter(torch.randn(T, d_model) * 0.02)

        # Transformer decoder
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=d_model, nhead=n_heads,
            dim_feedforward=d_model * 4,
            dropout=0.1, batch_first=True,
        )
        self.decoder = nn.TransformerDecoder(decoder_layer, num_layers=n_layers)

        # output head
        self.out_proj = nn.Linear(d_model, z_dim)

    def forward(self, text_emb: torch.Tensor) -> torch.Tensor:
        """
        text_emb : (B, text_dim)
        Returns  : (B, T, z_dim) L2-normalized
        """
        B = text_emb.shape[0]
        ctx = self.text_proj(text_emb).unsqueeze(1)       # (B, 1, d_model)
        q   = self.queries.unsqueeze(0).expand(B, -1, -1) # (B, T, d_model)
        h   = self.decoder(q, ctx)                        # (B, T, d_model)
        z   = self.out_proj(h)                             # (B, T, z_dim)
        return F.normalize(z, dim=-1)


# ── training loop ─────────────────────────────────────────────────────────────

def train(args, rank, local_rank, world_size):
    device = torch.device(f"cuda:{local_rank}")

    log(rank, f"[init] rank={rank}/{world_size}  device={device}")

    # CLIP text encoder (frozen, stays on CPU for memory efficiency then move emb to GPU)
    log(rank, "[model] loading CLIP text encoder...")
    clip = FrozenCLIPText().to(device)

    # z predictor
    model = ZPredictor(
        text_dim=768, d_model=args.d_model, n_heads=args.n_heads,
        n_layers=args.n_layers, T=args.T, z_dim=128,
    ).to(device)

    n_params = sum(p.numel() for p in model.parameters())
    log(rank, f"  ZPredictor: {n_params/1e6:.3f}M params")

    if world_size > 1:
        model = nn.parallel.DistributedDataParallel(
            model, device_ids=[local_rank], output_device=local_rank)
    model_module = model.module if world_size > 1 else model

    # dataset
    dataset = ZSeqDataset(args.data_dir, T=args.T)
    log(rank, f"[data] {len(dataset)} NPZ samples")

    sampler = (torch.utils.data.distributed.DistributedSampler(dataset, shuffle=True)
               if world_size > 1 else None)
    loader  = DataLoader(
        dataset, batch_size=args.batch_size,
        sampler=sampler, shuffle=(sampler is None),
        num_workers=2, pin_memory=True,
        collate_fn=collate_fn, drop_last=True,
    )

    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)

    # warmup 500 steps then cosine decay — rebuilt from global_step on resume
    # so lr is always correct regardless of checkpoint state
    warmup_steps = 500
    def lr_lambda(step):
        if step < warmup_steps:
            return step / max(1, warmup_steps)
        progress = (step - warmup_steps) / max(1, args.total_steps - warmup_steps)
        return max(1e-2, 0.5 * (1.0 + torch.cos(torch.tensor(progress * 3.14159)).item()))
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lr_lambda)

    out_dir   = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    ckpt_path = out_dir / "z_predictor.pt"

    global_step = 0
    if ckpt_path.exists() and not args.fresh:
        ckpt = torch.load(ckpt_path, map_location="cpu")
        model_module.load_state_dict(ckpt["model"])
        global_step = ckpt.get("step", 0)
        # rewind scheduler to correct step (do NOT restore optim lr state)
        for _ in range(global_step):
            sched.step()
        log(rank, f"[resume] step={global_step}  lr={sched.get_last_lr()[0]:.2e}")

    log(rank, f"[train] total_steps={args.total_steps}  batch={args.batch_size}  lr={args.lr}")
    writer = SummaryWriter(log_dir=str(out_dir / "tb_logs")) if rank == 0 else None
    model.train()
    loss_ema = None

    while global_step < args.total_steps:
        if sampler:
            sampler.set_epoch(global_step // max(1, len(loader)))
        for captions, z_true in loader:
            if global_step >= args.total_steps:
                break

            z_true = z_true.to(device)                       # (B, T, 128)
            text_emb = clip(captions, device)                 # (B, 768)

            z_pred = model(text_emb)                          # (B, T, 128)

            # cosine loss: mean over frames and batch
            cos = F.cosine_similarity(z_pred, z_true, dim=-1) # (B, T)
            loss = (1 - cos).mean()

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            sched.step()
            opt.zero_grad()

            global_step += 1
            loss_ema = loss.item() if loss_ema is None else 0.98 * loss_ema + 0.02 * loss.item()

            if global_step % 100 == 0 and rank == 0:
                with torch.no_grad():
                    mean_cos = F.cosine_similarity(z_pred.detach(), z_true, dim=-1).mean().item()
                lr_now = sched.get_last_lr()[0]
                log(rank, f"  step={global_step:6d}  loss={loss_ema:.4f}  "
                          f"cos={mean_cos:.4f}  lr={lr_now:.2e}")
                if writer is not None:
                    writer.add_scalar("train/loss",     loss_ema,  global_step)
                    writer.add_scalar("train/cos_sim",  mean_cos,  global_step)
                    writer.add_scalar("train/lr",       lr_now,    global_step)

            if global_step % args.save_every == 0 and rank == 0:
                payload = {"step": global_step, "model": model_module.state_dict()}
                torch.save(payload, ckpt_path)
                step_path = out_dir / f"z_predictor_step{global_step:06d}.pt"
                torch.save(payload, step_path)
                log(rank, f"  [CKPT] {step_path.name}")

    if rank == 0:
        payload = {"step": global_step, "model": model_module.state_dict()}
        torch.save(payload, ckpt_path)
        log(rank, f"[done] step={global_step}")
        if writer is not None:
            writer.close()

    if world_size > 1:
        torch.distributed.destroy_process_group()


# ── main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir",    default="data/processed/pipeline_a_v2/train")
    parser.add_argument("--out_dir",     default="exp_results/pipeline_b")
    parser.add_argument("--T",           type=int,   default=32)
    parser.add_argument("--d_model",     type=int,   default=256)
    parser.add_argument("--n_heads",     type=int,   default=4)
    parser.add_argument("--n_layers",    type=int,   default=4)
    parser.add_argument("--batch_size",  type=int,   default=64)
    parser.add_argument("--total_steps", type=int,   default=10000)
    parser.add_argument("--save_every",  type=int,   default=1000)
    parser.add_argument("--lr",          type=float, default=3e-4)
    parser.add_argument("--gpu",         type=int,   default=0)
    parser.add_argument("--fresh",       action="store_true",
                        help="ignore existing checkpoint, start from scratch")
    args = parser.parse_args()

    rank, local_rank, world_size = setup_ddp()
    if world_size == 1:
        torch.cuda.set_device(args.gpu)

    train(args, rank, local_rank, world_size)


if __name__ == "__main__":
    main()
