"""
Full 200-prompt, 7-condition evaluation for final_result/.

7 conditions:
  baseline                      final_result/prompt_{i:03d}_baseline.mp4
  lora                          final_result/prompt_{i:03d}_lora.mp4
  scale_01                      final_result/scale_01/prompt_{i:03d}_adapter.mp4
  scale_03                      final_result/scale_03/prompt_{i:03d}_adapter.mp4
  scale_05                      final_result/scale_05/prompt_{i:03d}_adapter.mp4
  scale_07                      final_result/scale_07/prompt_{i:03d}_adapter.mp4
  scale_10                      final_result/scale_10/prompt_{i:03d}_adapter.mp4

Phase 1 — run 8 parallel workers (each covers 25 prompts):
  for i in $(seq 0 7); do
    nohup python -u eval/eval_metrics.py \
        --gpu $i --start_idx $((i*25)) --end_idx $(((i+1)*25)) \
        > final_result/eval_log_gpu${i}.txt 2>&1 &
  done

Phase 2 — merge partial results into one JSON + print table:
  python -u eval/eval_metrics.py --merge
"""

import os, sys, json, argparse
import numpy as np
from pathlib import Path


def _strip_conda_libs():
    ld = os.environ.get("LD_LIBRARY_PATH", "")
    bad = ("/miniconda3/lib", "/anaconda3/lib", "/Micromamba/")
    parts = [p for p in ld.split(":") if p.strip() and not any(b in p for b in bad)]
    os.environ["LD_LIBRARY_PATH"] = ":".join(parts)
_strip_conda_libs()

import cv2
import torch
import torch.nn.functional as F
import torchvision.models as tvm
from torchvision.transforms import Normalize


PROJ_ROOT  = Path(__file__).resolve().parents[1]
FINAL_DIR  = PROJ_ROOT / "final_result"
PROMPT_JSON = PROJ_ROOT / "prompt_set.json"

CONDS = ["baseline", "lora", "scale_01", "scale_03", "scale_05", "scale_07", "scale_10"]
METRICS = ["tofv_mean", "tofv_var", "flow_smoothness", "lpips_mean", "lpips_var", "clip_score"]

def video_path(cond: str, idx: int) -> Path:
    if cond == "baseline":
        return FINAL_DIR / f"prompt_{idx:03d}_baseline.mp4"
    if cond == "lora":
        return FINAL_DIR / f"prompt_{idx:03d}_lora.mp4"
    scale = cond  # e.g. "scale_05"
    return FINAL_DIR / scale / f"prompt_{idx:03d}_adapter.mp4"


# ── video I/O ──────────────────────────────────────────────────────────────────

def read_video_frames(mp4_path) -> np.ndarray:
    cap = cv2.VideoCapture(str(mp4_path))
    frames = []
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    cap.release()
    return np.stack(frames) if frames else None


# ── metrics ────────────────────────────────────────────────────────────────────

def compute_flow_metrics(frames: np.ndarray) -> dict:
    gray = [cv2.cvtColor(f, cv2.COLOR_RGB2GRAY) for f in frames]
    mags = []
    for t in range(len(gray) - 1):
        flow = cv2.calcOpticalFlowFarneback(
            gray[t], gray[t+1], None, 0.5, 3, 15, 3, 5, 1.2, 0)
        mags.append(float(np.sqrt(flow[...,0]**2 + flow[...,1]**2).mean()))
    mags = np.array(mags)
    return {
        "tofv_mean":       float(mags.mean()),
        "tofv_var":        float(mags.var()),
        "flow_smoothness": float(np.abs(np.diff(mags)).mean()) if len(mags) > 1 else 0.0,
    }


class VGGPerceptual(torch.nn.Module):
    def __init__(self):
        super().__init__()
        vgg = tvm.vgg16(weights=tvm.VGG16_Weights.IMAGENET1K_V1)
        self.features = torch.nn.Sequential(*list(vgg.features.children())[:10])
        for p in self.parameters(): p.requires_grad_(False)
        self.norm = Normalize(mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225])
    def forward(self, x):
        return self.features(self.norm(x))


@torch.no_grad()
def compute_lpips_metrics(frames: np.ndarray, vgg, device) -> dict:
    t = torch.from_numpy(frames).float().permute(0,3,1,2) / 255.0
    dists = [F.mse_loss(vgg(t[i:i+1].to(device)), vgg(t[i+1:i+2].to(device))).item()
             for i in range(len(frames)-1)]
    dists = np.array(dists)
    return {"lpips_mean": float(dists.mean()), "lpips_var": float(dists.var())}


def load_clip(device):
    from transformers import CLIPModel, CLIPProcessor
    mid = "openai/clip-vit-large-patch14"
    print(f"  [CLIP] loading {mid}...", flush=True)
    model = CLIPModel.from_pretrained(mid).to(device).eval()
    proc  = CLIPProcessor.from_pretrained(mid)
    return model, proc


@torch.no_grad()
def compute_clip_score(frames, prompt, clip_model, proc, device) -> float:
    from PIL import Image
    idxs = np.linspace(0, len(frames)-1, min(len(frames), 8), dtype=int)
    pil  = [Image.fromarray(frames[i]) for i in idxs]
    inp  = proc(text=[prompt], images=pil, return_tensors="pt", padding=True)
    inp  = {k: v.to(device) for k,v in inp.items()}
    out  = clip_model(**inp)
    t_emb = F.normalize(out.text_embeds, dim=-1)
    i_emb = F.normalize(out.image_embeds, dim=-1)
    return float((i_emb @ t_emb.T).squeeze(-1).cpu().numpy().mean())


# ── worker ─────────────────────────────────────────────────────────────────────

def run_worker(args):
    device = torch.device(f"cuda:{args.gpu}")
    print(f"[worker] gpu={args.gpu} prompts={args.start_idx}..{args.end_idx-1}", flush=True)

    with open(PROMPT_JSON) as f:
        all_prompts = json.load(f)

    print("[model] loading VGG16...", flush=True)
    vgg = VGGPerceptual().to(device)
    clip_model, clip_proc = load_clip(device)

    partial = {}  # {str(i): {cond: metrics_dict}}

    for i in range(args.start_idx, args.end_idx):
        if i >= len(all_prompts):
            break
        prompt = all_prompts[i]
        print(f"\n[prompt {i:03d}] {prompt[:70]}...", flush=True)
        partial[str(i)] = {}

        for cond in CONDS:
            mp4 = video_path(cond, i)
            if not mp4.exists():
                print(f"  [{cond}] MISSING", flush=True)
                partial[str(i)][cond] = None
                continue

            frames = read_video_frames(mp4)
            if frames is None:
                print(f"  [{cond}] EMPTY", flush=True)
                partial[str(i)][cond] = None
                continue

            m = {
                **compute_flow_metrics(frames),
                **compute_lpips_metrics(frames, vgg, device),
                "clip_score": compute_clip_score(frames, prompt, clip_model, clip_proc, device),
            }
            partial[str(i)][cond] = m
            print(f"  [{cond}] tofv_var={m['tofv_var']:.4f}  "
                  f"smooth={m['flow_smoothness']:.4f}  "
                  f"lpips={m['lpips_mean']:.4f}  "
                  f"clip={m['clip_score']:.4f}", flush=True)

    out_path = FINAL_DIR / f"eval_partial_gpu{args.gpu}.json"
    with open(out_path, "w") as f:
        json.dump(partial, f, indent=2)
    print(f"\n[saved] {out_path}", flush=True)


# ── merge ──────────────────────────────────────────────────────────────────────

def run_merge(n_total=200):
    print("[merge] collecting partial results...", flush=True)
    partials = sorted(FINAL_DIR.glob("eval_partial_gpu*.json"))
    if not partials:
        print("No partial files found in", FINAL_DIR); return

    # Warn if any GPU partial is missing
    found_gpus = sorted(int(p.stem.replace("eval_partial_gpu", "")) for p in partials)
    expected_gpus = list(range(8))
    missing_gpus = [g for g in expected_gpus if g not in found_gpus]
    if missing_gpus:
        print(f"  WARNING: missing partial files for GPU(s): {missing_gpus}", flush=True)
        print(f"  Those prompts will be absent from the summary!", flush=True)

    all_data = {}
    for p in partials:
        with open(p) as f:
            all_data.update(json.load(f))

    # Verify prompt coverage
    found_idxs = sorted(int(k) for k in all_data.keys())
    expected_idxs = list(range(n_total))
    missing_idxs = [i for i in expected_idxs if i not in found_idxs]
    if missing_idxs:
        print(f"  WARNING: {len(missing_idxs)} prompt(s) missing from data: {missing_idxs[:10]}{'...' if len(missing_idxs)>10 else ''}", flush=True)
    else:
        print(f"  OK: all {n_total} prompts present", flush=True)

    print(f"  loaded {len(all_data)} prompt entries from {len(partials)} files", flush=True)

    # Aggregate per condition
    per_cond = {c: {k: [] for k in METRICS} for c in CONDS}
    for i_str, cond_dict in all_data.items():
        for cond in CONDS:
            m = cond_dict.get(cond)
            if m is None: continue
            for k in METRICS:
                v = m.get(k)
                if v is not None:
                    per_cond[cond][k].append(v)

    summary = {}
    for c in CONDS:
        summary[c] = {}
        for k in METRICS:
            vals = per_cond[c][k]
            summary[c][k] = {"mean": float(np.mean(vals)), "std": float(np.std(vals)), "n": len(vals)} if vals else {}

    result = {"per_prompt": all_data, "summary": summary}
    out_json = FINAL_DIR / "eval_metrics.json"
    with open(out_json, "w") as f:
        json.dump(result, f, indent=2)
    print(f"[saved] {out_json}", flush=True)

    # Print table
    col_w = 12
    header = f"{'Metric':<22}" + "".join(f"{c:>{col_w}}" for c in CONDS)
    print("\n" + "=" * (22 + col_w * len(CONDS)))
    print(f"  7-Condition Evaluation  ({len(all_data)} prompts)")
    print("=" * (22 + col_w * len(CONDS)))
    print(header)
    print("-" * (22 + col_w * len(CONDS)))
    for k in METRICS:
        row = f"{k:<22}"
        for c in CONDS:
            v = summary[c][k].get("mean")
            row += f"  {v:>10.4f}" if v is not None else f"  {'—':>10}"
        print(row)
    print("=" * (22 + col_w * len(CONDS)))

    # Delta vs lora
    print(f"\n  Delta vs lora (adapter scales):")
    print(f"{'Metric':<22}" + "".join(f"{'s'+c.split('_')[1]:>{col_w}}" for c in CONDS if c.startswith("scale")))
    print("-" * (22 + col_w * 5))
    lora_s = summary["lora"]
    for k in METRICS:
        lv = lora_s[k].get("mean")
        if lv is None: continue
        row = f"{k:<22}"
        for c in [c for c in CONDS if c.startswith("scale")]:
            av = summary[c][k].get("mean")
            if av is None:
                row += f"  {'—':>10}"
            else:
                pct = (av - lv) / abs(lv) * 100
                better = pct < 0 if k != "clip_score" else pct > 0
                row += f"  {pct:>+9.1f}%{'✓' if better else '✗'}"
        print(row)
    print()


# ── main ───────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--merge",     action="store_true", help="merge partial results and print table")
    parser.add_argument("--gpu",       type=int, default=0)
    parser.add_argument("--start_idx", type=int, default=0)
    parser.add_argument("--end_idx",   type=int, default=25)
    args = parser.parse_args()

    if args.merge:
        run_merge()
    else:
        run_worker(args)


if __name__ == "__main__":
    main()
