<div align="center">

<h1>SurgActionGen</h1>

<p><em>Surgical video generation that actually moves like surgery.</em></p>

<p>
  <a href="#"><img src="https://img.shields.io/badge/CVPR-2026-7B2D8B?style=flat-square"/></a>
  <a href="#"><img src="https://img.shields.io/badge/arXiv-2026-red?style=flat-square"/></a>
  <a href="#"><img src="https://img.shields.io/badge/Python-3.10-3776AB?style=flat-square&logo=python&logoColor=white"/></a>
  <a href="#"><img src="https://img.shields.io/badge/PyTorch-2.7-EE4C2C?style=flat-square&logo=pytorch&logoColor=white"/></a>
  <a href="LICENSE"><img src="https://img.shields.io/badge/License-MIT-green?style=flat-square"/></a>
</p>

<br>

<img src="assets/teaser.png" width="860" alt="Comparison: Un-fine-tuned vs LoRA vs LoRA + SurgActionGen"/>

<p><sub>
  <b>Top:</b> Wan2.1 (no fine-tuning) &nbsp;·&nbsp;
  <b>Middle:</b> Wan2.1 + LoRA &nbsp;·&nbsp;
  <b>Bottom:</b> Wan2.1 + LoRA + <b>SurgActionGen</b> — stable, coherent surgical motion
</sub></p>

</div>

---

## The Problem

Standard text-to-video models have never seen an operating room. They generate surgical scenes that *look* right but *move* wrong — instruments jitter, the camera shakes, temporal coherence collapses.

The fix isn't more LoRA fine-tuning. It's **motion priors from real surgery**.

---

## What We Do

We extract the motion signature of 5,209 real Cholec80 surgical clips, compress it into a 128-dim latent code `z`, and inject it into a frozen Wan2.1 DiT through two lightweight adapters. The base model never changes. The motion does.

<div align="center">

```
  "laparoscopic cholecystectomy,          →    Wan2.1-T2V-1.3B
   clipping the cystic duct"                   (frozen weights)
           │                                         ↑
           │  CLIP ViT-L/14                          │  forward hooks
           ▼                                         │
      ZPredictor          z_seq (32×128) ──── WanActionAdapter
   (text → motion)           ↑                  (2M params)
                             │
                     FlowAutoEncoder
                  (real Cholec80 motion)
```

</div>

Three stages. One forward pass at inference. No reference video needed.

---

## Results

<div align="center">

| | Flow Variance | Smoothness | LPIPS | Motion (VBench) | BG Consistency |
|:---:|:---:|:---:|:---:|:---:|:---:|
| LoRA | 2.637 | 1.488 | 2.892 | 0.9730 | 0.9303 |
| **Ours** | **2.384** | **1.263** | **2.608** | **0.9755** | **0.9433** |
| | −9.6% | −15.2% | −9.8% | +0.3% | +1.4% |

*200 prompts · 8 GPUs · dynamic degree fully preserved*

</div>

The adapter adds **motion stability** without touching appearance, content, or dynamic degree.

---

## Quickstart

**Prereqs:** Wan2.1-T2V-1.3B weights · Cholec80 (licensed) · CUDA 11.8+

```bash
# 1. Clone + install
git clone https://github.com/Punktheory/SurgActionGen
cd SurgActionGen

conda create -n cogvideox python=3.10 && conda activate cogvideox
pip install torch==2.7.1+cu118 torchvision --index-url https://download.pytorch.org/whl/cu118
pip install -r requirements.txt

git clone https://github.com/Wan-Video/Wan2.1 /path/to/Wan2.1

# 2. Run inference (single GPU, 2 prompts to verify)
python inference/run_inference.py \
    --wan_dir /path/to/Wan2.1 \
    --gpu 0 --start_idx 0 --end_idx 2
```

Output appears in `final_result/`. Each prompt produces 7 videos: baseline, LoRA, and 5 adapter scales.

---

## Training

<details>
<summary><b>Step 0 — Preprocess Cholec80</b> (raft env)</summary>

```bash
conda create -n raft python=3.10 && conda activate raft
pip install torch==2.6.0+cu124 torchvision --index-url https://download.pytorch.org/whl/cu124
pip install opencv-python numpy omegaconf tqdm

python scripts/prepare_cholec80_laof.py \
    --clip_dir /path/to/cholec80/clips \
    --out_dir  data/processed/cholec80_laof_256 \
    --gpus 0,1,2,3,4,5,6,7
```
</details>

<details>
<summary><b>Step 1 — Train FlowAutoEncoder</b> (raft env)</summary>

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 torchrun \
    --nproc_per_node=8 --master_port=29505 \
    laof/cholec80_stage1_v6.py \
    --steps 50000 --bs 64 --lr 3e-4 \
    --out_dir exp_results/fae_v6
```

Watch `z_std` → should converge to ~0.088.
</details>

<details>
<summary><b>Step 1b — Extract z_seq</b> (raft env)</summary>

```bash
for i in 0 1 2 3 4 5 6 7; do
    python -u laof/pipeline_a_extract_v3.py --gpu $i --shard $i --num_shards 8 &
done
wait
```
</details>

<details>
<summary><b>Step 2 — Train WanActionAdapter</b> (cogvideox env)</summary>

```bash
# Precompute VAE + T5 latents
python models/pipeline_a/wan_precompute.py \
    --wan_dir /path/to/Wan2.1 --ckpt_dir models/wan2_1_1_3b \
    --data_dir data/processed/pipeline_a_v2/train \
    --cache_dir data/processed/pipeline_a_v2/wan_orig_cache_256 \
    --height 256 --width 448

# Train adapter
torchrun --nproc_per_node=4 --master_port=29513 \
    models/pipeline_a/wan_train_v4.py \
    --wan_dir /path/to/Wan2.1 --ckpt_dir models/wan2_1_1_3b \
    --lora_path wan_lora_cholec80.safetensors \
    --data_dir data/processed/pipeline_a_v2/train \
    --cache_dir data/processed/pipeline_a_v2/wan_orig_cache_256 \
    --out_dir exp_results/pipeline_a/wan_lora_256_v4 \
    --height 256 --width 448 --total_steps 30000 \
    --smooth_lambda 30.0 --fb_norm_var_lambda 5.0 --gc_upper_lambda 2.0
```
</details>

<details>
<summary><b>Step 3 — Train ZPredictor</b> (cogvideox env)</summary>

```bash
torchrun --nproc_per_node=8 --master_port=29520 \
    models/pipeline_b/train_z_predictor.py \
    --data_dir data/processed/pipeline_a_v2/train \
    --out_dir  exp_results/pipeline_b \
    --total_steps 10000 --batch_size 64 --lr 3e-4
```

Converges to `cos(ẑ_pred, z_real) = 0.818` — predicted z drives the adapter identically to ground-truth z from real video.
</details>

---

## Architecture in 30 Seconds

**FlowAutoEncoder** — CNN encoder maps RAFT optical flow to a 128-dim code `z` on the unit hypersphere. InfoNCE loss forces different frames to different z. Structurally collapse-proof.

**WanActionAdapter** — 2M-param MLP reads `z_seq` and writes two signals into the frozen DiT: `g` into the time embedder (global motion style), `F` into the patch embedder (per-frame detail). Five proxy losses train it without access to diffusion gradients.

**ZPredictor** — 4-layer Transformer Decoder maps a CLIP text embedding to `ẑ_seq`. The predicted z is functionally equivalent to z extracted from a real surgical video (`cos = 0.818`).

---

## Citation

```bibtex
@inproceedings{surgactiongen2026,
  title     = {SurgActionGen: Motion-Stabilized Surgical Video Generation via Latent Motion Priors},
  author    = {...},
  booktitle = {CVPR},
  year      = {2026}
}
```

---

<div align="center">
<sub>Built on <a href="https://github.com/Wan-Video/Wan2.1">Wan2.1</a> · evaluated with <a href="https://github.com/Vchitect/VBench">VBench</a> · trained on <a href="https://camma.unistra.fr/datasets/">Cholec80</a></sub>
</div>
