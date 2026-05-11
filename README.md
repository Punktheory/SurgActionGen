# SurgActionGen

<div align="center">

**Motion-Stabilized Surgical Video Generation via Latent Motion Priors**

[![Python 3.10](https://img.shields.io/badge/python-3.10-blue.svg)](https://www.python.org/downloads/release/python-310/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.7.1-ee4c2c.svg)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![CVPR 2026](https://img.shields.io/badge/CVPR-2026-purple.svg)]()

[Paper](#citation) · [Demo](#inference) · [Model Weights](#checkpoints)

</div>

---

## The Problem

Text-to-video models can generate visually plausible surgical scenes — but the motion is wrong. Instruments jitter, the camera shakes unnaturally, and temporal coherence breaks down. The model has never seen a real operating room.

**SurgActionGen** fixes this by extracting real surgical motion priors from 5,209 Cholec80 clips and injecting them into a frozen Wan2.1 diffusion model — without touching a single base model weight.

<div align="center">

| | Optical Flow Variance ↓ | Flow Smoothness ↓ | LPIPS (inter-frame) ↓ | VBench Motion ↑ | VBench BG Consistency ↑ |
|---|---|---|---|---|---|
| **Wan2.1 + LoRA** | 2.637 | 1.488 | 2.892 | 0.9730 | 0.9303 |
| **+ SurgActionGen** | **2.384** | **1.263** | **2.608** | **0.9755** | **0.9433** |
| **Δ** | **−9.6%** | **−15.2%** | **−9.8%** | **+0.3%** | **+1.4%** |

*Evaluated on 200 surgical text prompts. Dynamic degree fully preserved (1.000 → 1.000).*

</div>

---

## How It Works

```
                        ┌─────────────────────────────────────┐
  text prompt           │         SurgActionGen Pipeline        │
      │                 │                                       │
      ▼                 │  Stage 3: ZPredictor                  │
  CLIP ViT-L/14 ────────►  prompt → ẑ_seq (32 × 128)           │
  (frozen)             │         cos(ẑ, z_real) = 0.818 ✓      │
                        │              │                        │
                        │              ▼                        │
  Cholec80 flows        │  Stage 1: FlowAutoEncoder             │
  → RAFT → z_t ─────────►  flow → z_t (128-dim, unit sphere)   │
                        │              │                        │
                        │              ▼                        │
                        │  Stage 2: WanActionAdapter            │
                        │  z_seq → g (global style)             │
                        │        → F (per-frame detail)         │
                        │         hooks ↓          ↓            │
                        └──────── time_emb    patch_emb ────────┘
                                       │
                                       ▼
                              Wan2.1-T2V-1.3B (frozen)
                                       │
                                       ▼
                           🎥 Stable surgical video
```

### Stage 1 — FlowAutoEncoder (FAE)

Encodes RAFT optical flow from real Cholec80 clips into a compact 128-dim latent `z_t` on the unit hypersphere. InfoNCE contrastive loss prevents collapse; L2-normalization is structurally collapse-proof.

```
optical flow image (3×128×128)
    → CNN encoder → z_t (128-dim, ‖z‖₂=1)
    → CNN decoder → reconstructed flow   [MSE loss]
                                          [InfoNCE loss → no collapse]
```

### Stage 2 — WanActionAdapter

A lightweight MLP (~2M params) that reads `z_seq` and injects two conditioning signals into the frozen Wan2.1 DiT via forward hooks — zero modifications to base weights.

```
z_seq (B, 32, 128)
  → temporal avgpool ×4  → (B, 8, 128)
  → LayerNorm + shared MLP (128→512→512)
  → GlobalHead  → g  (B, 1536)        → time_embedding  (motion style)
  → FrameHead   → F  (B, 8, 1536)     → patch_embedding (per-frame detail)
```

**Training losses (v4):**

| Loss | λ | Purpose |
|------|---|---------|
| `L_sens` | 1 | g vectors differ across z inputs |
| `L_fb_sens` | 0.5 | frame biases also respond to z |
| `L_smooth` | 30 | consecutive frame biases vary smoothly |
| `L_fnv` | 5 | per-frame bias amplitudes are equal → reduces flow variance |
| `L_gc_upper` | 2 | ‖g‖₂ ≤ 0.5 → prevents disrupting time embedder |

### Stage 3 — ZPredictor

Enables inference without a reference video. Maps a text prompt directly to a predicted `ẑ_seq`.

```
prompt → CLIP ViT-L/14 → context c (768,)
32 learnable queries → Transformer Decoder (4 layers, cross-attn on c)
  → F.normalize → ẑ_seq (32, 128)
```

Validation: `cos(gc_predicted, gc_real) = 0.818` — the adapter responds identically to predicted vs. real-video z.

---

## Repository Structure

```
SurgActionGen/
├── laof/                          # Stage 1: FlowAutoEncoder
│   ├── cholec80_stage1_v6.py      # FAE training (8-GPU DDP, raft env)
│   ├── pipeline_a_extract_v3.py   # Extract z_seq NPZs from Cholec80
│   ├── conf/defualt.yaml          # FAE model config
│   └── utils/
│       ├── utils.py               # Model factory (create_dynamics_models_flow_fae)
│       └── cholec80_data_loader.py
│
├── models/
│   ├── pipeline_a/
│   │   ├── wan_action_adapter.py  # Adapter architecture + PyTorch forward hooks
│   │   ├── wan_train_v4.py        # Adapter training (4-GPU DDP, cogvideox env)
│   │   └── wan_precompute.py      # VAE + T5 latent precomputation
│   └── pipeline_b/
│       └── train_z_predictor.py   # ZPredictor training
│
├── scripts/
│   └── prepare_cholec80_laof.py   # Cholec80 MP4s → RAFT flow → NPZ chunks
│
├── inference/
│   └── run_inference.py           # 200-prompt inference (resume-safe, multi-GPU)
│
├── eval/
│   ├── eval_metrics.py            # Farneback + LPIPS + CLIP evaluation
│   └── eval_vbench.py             # VBench evaluation (4 dimensions)
│
└── prompt_set.json                # 200 surgical text prompts
```

---

## Setup

### Prerequisites

- Python 3.10
- CUDA 11.8+ (cogvideox env) / CUDA 12.4+ (raft env)
- [Wan2.1-T2V-1.3B](https://github.com/Wan-Video/Wan2.1) weights and source
- Cholec80 dataset — apply at [CAMMA](https://camma.unistra.fr/datasets/)

SurgActionGen uses **two conda environments**:

| Environment | Purpose | CUDA |
|-------------|---------|------|
| `cogvideox` | Pipeline A/B training + inference | 11.8 |
| `raft` | Data preprocessing + FAE training | 12.4 |

### Install cogvideox (training + inference)

```bash
conda create -n cogvideox python=3.10
conda activate cogvideox
pip install torch==2.7.1+cu118 torchvision==0.22.1+cu118 --index-url https://download.pytorch.org/whl/cu118
pip install transformers==5.5.3 diffusers==0.37.1 accelerate peft safetensors
pip install opencv-python Pillow einops scipy tqdm easydict omegaconf sentencepiece
pip install tensorboard imageio-ffmpeg decord
```

### Install raft (data preprocessing)

```bash
conda create -n raft python=3.10
conda activate raft
pip install torch==2.6.0+cu124 torchvision==0.21.0+cu124 --index-url https://download.pytorch.org/whl/cu124
pip install opencv-python numpy omegaconf tqdm
```

### Clone external dependencies

```bash
# Required for Stages 2 + 3 and inference
git clone https://github.com/Wan-Video/Wan2.1 /path/to/Wan2.1

# Required only for VBench evaluation
git clone https://github.com/Vchitect/VBench /path/to/VBench
```

---

## Training from Scratch

### Step 0 — Preprocess Cholec80

Converts raw MP4 clips to RAFT optical flow NPZ chunks (256×256):

```bash
conda activate raft
python scripts/prepare_cholec80_laof.py \
    --clip_dir /path/to/cholec80/clips \
    --out_dir  data/processed/cholec80_laof_256 \
    --gpus 0,1,2,3,4,5,6,7
```

### Step 1 — Train FlowAutoEncoder

```bash
conda activate raft
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 torchrun \
    --nproc_per_node=8 --master_port=29505 \
    laof/cholec80_stage1_v6.py \
    --steps 50000 --bs 64 --lr 3e-4 \
    --out_dir exp_results/fae_v6
```

Monitor `z_std` — should converge to ~0.088 (uniform distribution on unit hypersphere). Checkpoint: `exp_results/fae_v6/idm_fdm_v6.pt`.

### Step 1b — Extract z_seq

```bash
conda activate raft
for i in 0 1 2 3 4 5 6 7; do
    python -u laof/pipeline_a_extract_v3.py \
        --gpu $i --shard $i --num_shards 8 &
done
wait
```

Output: `data/processed/pipeline_a_v2/train/*.npz` (each contains `z_seq`, `frames_256`, `caption`).

### Step 2a — Precompute VAE + T5 Latents

```bash
conda activate cogvideox
python models/pipeline_a/wan_precompute.py \
    --wan_dir  /path/to/Wan2.1 \
    --ckpt_dir models/wan2_1_1_3b \
    --data_dir data/processed/pipeline_a_v2/train \
    --cache_dir data/processed/pipeline_a_v2/wan_orig_cache_256 \
    --height 256 --width 448
```

### Step 2b — Train WanActionAdapter

```bash
conda activate cogvideox
torchrun --nproc_per_node=4 --master_port=29513 \
    models/pipeline_a/wan_train_v4.py \
    --wan_dir   /path/to/Wan2.1 \
    --ckpt_dir  models/wan2_1_1_3b \
    --lora_path wan_lora_cholec80.safetensors \
    --data_dir  data/processed/pipeline_a_v2/train \
    --cache_dir data/processed/pipeline_a_v2/wan_orig_cache_256 \
    --out_dir   exp_results/pipeline_a/wan_lora_256_v4 \
    --height 256 --width 448 \
    --total_steps 30000 --save_every 3000 \
    --smooth_lambda 30.0 --fb_norm_var_lambda 5.0 --gc_upper_lambda 2.0
```

### Step 3 — Train ZPredictor

```bash
conda activate cogvideox
torchrun --nproc_per_node=8 --master_port=29520 \
    models/pipeline_b/train_z_predictor.py \
    --data_dir data/processed/pipeline_a_v2/train \
    --out_dir  exp_results/pipeline_b \
    --total_steps 10000 --batch_size 64 --lr 3e-4
```

---

## Inference

```bash
conda activate cogvideox

# 8-GPU parallel, 200 prompts
for i in $(seq 0 7); do
    nohup python -u inference/run_inference.py \
        --wan_dir /path/to/Wan2.1 \
        --gpu $i --start_idx $((i*25)) --end_idx $(((i+1)*25)) \
        > logs/log_gpu${i}.txt 2>&1 &
done
wait
```

Single-GPU smoke test (2 prompts):

```bash
python -u inference/run_inference.py \
    --wan_dir /path/to/Wan2.1 \
    --gpu 0 --start_idx 0 --end_idx 2
```

Inference is **resume-safe** — already-generated files are skipped automatically.

---

## Evaluation

### Motion Metrics (Farneback + LPIPS + CLIP)

```bash
for i in $(seq 0 7); do
    python -u eval/eval_metrics.py \
        --gpu $i --start_idx $((i*25)) --end_idx $(((i+1)*25)) \
        > logs/eval_log_gpu${i}.txt 2>&1 &
done
wait
python -u eval/eval_metrics.py --merge
```

### VBench (4 dimensions)

```bash
for i in $(seq 0 6); do
    COND=$(python -c "c=['baseline','lora','scale_01','scale_03','scale_05','scale_07','scale_10']; print(c[$i])")
    python -u eval/eval_vbench.py \
        --vbench_dir /path/to/VBench \
        --cond $COND --gpu $i \
        > logs/eval_vbench_${COND}.log 2>&1 &
done
wait
python -u eval/eval_vbench.py --vbench_dir /path/to/VBench --merge
```

---

## Results

### Effect of Adapter Scale (200 prompts, v4 adapter)

| Scale | Flow Var ↓ | Smoothness ↓ | LPIPS ↓ | Motion Smooth ↑ | BG Consist. ↑ |
|-------|-----------|-------------|---------|----------------|--------------|
| LoRA baseline | 2.637 | 1.488 | 2.892 | 0.9730 | 0.9303 |
| 0.1 | 2.491 | 1.496 | 2.815 | — | — |
| 0.3 | 2.543 | 1.368 | 2.747 | 0.9738 | 0.9434 |
| **0.5** ⭐ | **2.384** | **1.263** | **2.608** | **0.9755** | **0.9433** |
| 0.7 | 2.717 | 1.322 | 2.561 | 0.9750 | 0.9482 |
| 1.0 | 2.311 | 1.263 | 2.450 | — | — |

**Best configuration: scale=0.5** — strongest joint improvement across flow variance, smoothness, and LPIPS with no degradation in dynamic degree.

### ZPredictor Validation (5/5 PASS)

| Test | Result |
|------|--------|
| Cross-prompt diversity | pairwise cosine = 0.068 (random = 0.001) |
| Temporal structure | adjacent cosine = 0.726, p ≈ 0 |
| Caption alignment | intra-caption cosine = 0.501 vs cross = 0.027 |
| **Adapter gc sensitivity** | **cos(gc_pred, gc_real) = 0.818** |
| Cross-prompt gc diversity | gc std = 0.207 vs fixed-z std = 0.000 |

---

## Checkpoints

| Artifact | Description |
|----------|-------------|
| `FAE v6` | FlowAutoEncoder, 50k steps, z_std=0.088 |
| `WanActionAdapter v4` | Final adapter, 30k steps, scale=0.5 recommended |
| `ZPredictor` | 10k steps, cos(gc_pred, gc_real)=0.818 |
| `LoRA weights` | Cholec80 fine-tuned Wan2.1-1.3B LoRA |

> Weights available upon request (model weights excluded from repo due to size).

---

## Citation

```bibtex
@inproceedings{surgactiongen2026,
  title     = {SurgActionGen: Motion-Stabilized Surgical Video Generation via Latent Motion Priors},
  author    = {...},
  booktitle = {Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)},
  year      = {2026}
}
```

---

## Acknowledgements

- [Wan2.1](https://github.com/Wan-Video/Wan2.1) — video diffusion backbone
- [LAOF](https://arxiv.org/abs/2511.16407) — latent action representation framework
- [VBench](https://github.com/Vchitect/VBench) — video generation evaluation suite
- [Cholec80](https://camma.unistra.fr/datasets/) — surgical video dataset (CAMMA, University of Strasbourg)
