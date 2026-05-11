"""
VBench evaluation for 200-prompt, 7-condition final_result/.

7 conditions:
  baseline    final_result/prompt_{i:03d}_baseline.mp4
  lora        final_result/prompt_{i:03d}_lora.mp4
  scale_01    final_result/scale_01/prompt_{i:03d}_adapter.mp4
  scale_03    final_result/scale_03/prompt_{i:03d}_adapter.mp4
  scale_05    final_result/scale_05/prompt_{i:03d}_adapter.mp4
  scale_07    final_result/scale_07/prompt_{i:03d}_adapter.mp4
  scale_10    final_result/scale_10/prompt_{i:03d}_adapter.mp4

4 VBench dimensions:
  motion_smoothness     AMT-S frame interpolation quality  (higher = smoother)
  dynamic_degree        RAFT optical flow activity         (higher = more motion)
  background_consistency CLIP ViT-B/32 bg stability       (higher = more stable)
  subject_consistency   DINO ViT-B/16 subject stability   (higher = more stable)

Phase 1 — run 7 conditions in parallel (one per GPU):
  for i in $(seq 0 6); do
    COND=$(python -c "c=['baseline','lora','scale_01','scale_03','scale_05','scale_07','scale_10']; print(c[$i])")
    nohup python -u eval/eval_vbench.py \
        --vbench_dir /path/to/VBench \
        --cond $COND --gpu $i \
        > final_result/eval_vbench_${COND}.log 2>&1 &
  done

Phase 2 — merge results and print table:
  python -u eval/eval_vbench.py --vbench_dir /path/to/VBench --merge

Outputs: final_result/vbench/<cond>/<cond>_eval_results.json
         final_result/vbench/vbench_summary.json
"""

import os, sys, json, shutil, argparse
from pathlib import Path

# VBENCH_DIR is passed via --vbench_dir; inserted into sys.path in main()
import torch

PROJ_ROOT = Path(__file__).resolve().parents[1]
FINAL_DIR = PROJ_ROOT / "final_result"

CONDS = ["baseline", "lora", "scale_01", "scale_03", "scale_05", "scale_07", "scale_10"]

DEFAULT_DIMS = [
    "motion_smoothness",
    "dynamic_degree",
    "background_consistency",
    "subject_consistency",
]


def video_path(cond: str, idx: int) -> Path:
    if cond == "baseline":
        return FINAL_DIR / f"prompt_{idx:03d}_baseline.mp4"
    if cond == "lora":
        return FINAL_DIR / f"prompt_{idx:03d}_lora.mp4"
    return FINAL_DIR / cond / f"prompt_{idx:03d}_adapter.mp4"


def collect_videos(cond: str, n: int) -> list:
    paths = []
    missing = []
    for i in range(n):
        p = video_path(cond, i)
        if p.exists():
            paths.append(str(p))
        else:
            missing.append(i)
    if missing:
        print(f"  [warn] {cond}: {len(missing)} missing — {missing[:5]}{'...' if len(missing)>5 else ''}", flush=True)
    return paths


def run_vbench_on_videos(video_paths: list, name: str, dimensions: list,
                          out_dir: Path, device: torch.device, vbench_dir: str):
    from vbench import VBench

    tmp_dir = out_dir / f"_tmp_{name}"
    tmp_dir.mkdir(parents=True, exist_ok=True)
    for vp in video_paths:
        dst = tmp_dir / Path(vp).name
        if not dst.exists():
            os.symlink(os.path.abspath(vp), dst)

    vb = VBench(
        device=str(device),
        full_info_dir=f"{vbench_dir}/vbench/VBench_full_info.json",
        output_path=str(out_dir / name),
    )
    vb.evaluate(
        videos_path=str(tmp_dir),
        name=name,
        dimension_list=dimensions,
        mode="custom_input",
    )
    shutil.rmtree(tmp_dir, ignore_errors=True)


def load_scores(result_json: Path, dimensions: list) -> dict:
    if not result_json.exists():
        return {}
    with open(result_json) as f:
        data = json.load(f)
    scores = {}
    for dim in dimensions:
        if dim not in data:
            continue
        val = data[dim]
        if isinstance(val, (list, tuple)) and len(val) >= 1:
            scores[dim] = float(val[0])
        elif isinstance(val, (int, float)):
            scores[dim] = float(val)
    return scores


def run_single_cond(args):
    """Evaluate one condition on one GPU."""
    out_dir = Path(args.out_dir) if args.out_dir else FINAL_DIR / "vbench"
    out_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device(f"cuda:{args.gpu}")
    cond = args.cond
    print(f"[init] cond={cond}  device={device}", flush=True)
    print(f"[init] dimensions={args.dimensions}", flush=True)

    vids = collect_videos(cond, args.n_prompts)
    print(f"[data] {cond}: {len(vids)} videos", flush=True)
    if not vids:
        print(f"[error] no videos found for {cond}"); return

    result_json = out_dir / cond / f"{cond}_eval_results.json"
    if result_json.exists():
        print(f"[vbench] '{cond}' already done — skipping "
              f"(delete {result_json} to re-run)", flush=True)
        return

    print(f"[vbench] evaluating '{cond}' ({len(vids)} videos)...", flush=True)
    try:
        run_vbench_on_videos(vids, cond, args.dimensions, out_dir, device, args.vbench_dir)
        print(f"[vbench] '{cond}' DONE", flush=True)
    except Exception as e:
        print(f"[error] '{cond}': {e}", flush=True)


def run_merge(args):
    """Collect all per-condition results and print summary table."""
    out_dir = Path(args.out_dir) if args.out_dir else FINAL_DIR / "vbench"
    dims = args.dimensions

    print("[merge] loading per-condition results...", flush=True)
    summary = {}
    for cond in CONDS:
        result_json = out_dir / cond / f"{cond}_eval_results.json"
        scores = load_scores(result_json, dims)
        summary[cond] = scores
        status = "OK" if scores else "MISSING"
        print(f"  {cond}: {status}  {scores}", flush=True)

    # Check completeness
    missing = [c for c in CONDS if not summary[c]]
    if missing:
        print(f"\n  WARNING: results missing for: {missing}", flush=True)
    else:
        print(f"\n  OK: all 7 conditions present", flush=True)

    out_json = out_dir / "vbench_summary.json"
    with open(out_json, "w") as f:
        json.dump({"n_prompts": args.n_prompts, "summary": summary}, f, indent=2)
    print(f"[saved] {out_json}", flush=True)

    # Print table
    col_w = 12
    sep = "=" * (26 + col_w * len(CONDS))
    print("\n" + sep)
    print(f"  VBench  7-Condition  ({args.n_prompts} prompts)")
    print(sep)
    print(f"{'Dimension':<26}" + "".join(f"{c:>{col_w}}" for c in CONDS))
    print("-" * (26 + col_w * len(CONDS)))
    for dim in dims:
        row = f"{dim:<26}"
        for c in CONDS:
            v = summary[c].get(dim)
            row += f"  {v:>10.4f}" if v is not None else f"  {'—':>10}"
        print(row)
    print(sep)

    # Delta vs lora
    print(f"\n  Delta vs lora (adapter scales):")
    scale_conds = [c for c in CONDS if c.startswith("scale")]
    print(f"{'Dimension':<26}" + "".join(f"{'s'+c.split('_')[1]:>{col_w}}" for c in scale_conds))
    print("-" * (26 + col_w * 5))
    lora_s = summary.get("lora", {})
    for dim in dims:
        lv = lora_s.get(dim)
        if lv is None:
            continue
        row = f"{dim:<26}"
        for c in scale_conds:
            av = summary[c].get(dim)
            if av is None:
                row += f"  {'—':>10}"
            else:
                pct = (av - lv) / abs(lv) * 100
                better = pct > 0  # all 4 VBench dims: higher is better
                row += f"  {pct:>+9.1f}%{'✓' if better else '✗'}"
        print(row)
    print()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--vbench_dir", required=True,
                        help="Path to VBench source directory (contains vbench/ package)")
    parser.add_argument("--cond",       default=None,
                        choices=CONDS,
                        help="single condition to evaluate (required unless --merge)")
    parser.add_argument("--merge",      action="store_true",
                        help="collect all per-condition results and print summary table")
    parser.add_argument("--gpu",        type=int, default=0)
    parser.add_argument("--n_prompts",  type=int, default=200)
    parser.add_argument("--dimensions", nargs="+", default=DEFAULT_DIMS)
    parser.add_argument("--out_dir",    default=None,
                        help="defaults to final_result/vbench/")
    args = parser.parse_args()

    sys.path.insert(0, args.vbench_dir)

    if args.merge:
        run_merge(args)
    elif args.cond:
        run_single_cond(args)
    else:
        parser.error("provide --cond <condition> or --merge")


if __name__ == "__main__":
    main()
