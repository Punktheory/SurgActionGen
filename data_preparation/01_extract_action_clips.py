"""
Step 1 — Extract action clips from Cholec80 + CholecT50.

CholecT50 provides instrument-verb-target triplet annotations at 1 FPS for
50 of the 80 Cholec80 videos.  This script:

  1. Reads the per-video CholecT50 triplet files (one annotation per second).
  2. Groups contiguous seconds that share the same triplet set into segments.
  3. Rejects segments shorter than --min_sec seconds.
  4. Maps each 1-FPS annotation segment to the corresponding 25-FPS Cholec80
     frame range and extracts it as a short MP4 clip.
  5. Writes clips_metadata.json listing each clip's source video, 1-FPS frame
     range, and triplet labels.

Expected input layout
---------------------
  cholec80_dir/
    videos/          # full-length MP4s: VID01.mp4 … VID80.mp4   (or video01.mp4)
    annotations/     # CholecT50 triplet files (see --anno_format below)

CholecT50 annotation formats supported (--anno_format)
-------------------------------------------------------
  "triplet_csv"  (default)
    Tab- or comma-separated file per video, named VID01.txt or video01.txt.
    Columns (order matters):
      frame_id  instrument  verb  target
    frame_id is the 1-FPS annotation index (starting from 0 or 1).
    Multiple rows with the same frame_id = multiple triplets that second.

  "cholect50_official"
    Multi-hot CSV as distributed at https://github.com/CAMMA-public/cholect50
    Each row: Frame, <6 instrument flags>, <9 verb flags>, <14 target flags>
    Triplets are inferred from co-occurring non-null instrument+verb+target
    columns.

Output
------
  out_dir/clips/              — short MP4s named videoXX_clipYY.mp4
  out_dir/clips_metadata.json — [{video, frame_range, triplets}, ...]

Usage
-----
  python 01_extract_action_clips.py \\
      --cholec80_dir /path/to/cholec80 \\
      --anno_dir     /path/to/cholect50/annotations \\
      --out_dir      data/cholec80_action \\
      [--min_sec 3] [--fps 25] [--anno_format triplet_csv]
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import re
import subprocess
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple

# ── vocabulary ────────────────────────────────────────────────────────────────
INSTRUMENTS = ['grasper', 'bipolar', 'hook', 'scissors', 'clipper', 'irrigator']
VERBS       = ['grasp', 'retract', 'dissect', 'coagulate', 'clip', 'cut',
               'aspirate', 'irrigate', 'pack']
TARGETS     = ['gallbladder', 'cystic_plate', 'cystic_duct', 'cystic_pedicle',
               'cystic_artery', 'gallbladder_peritoneum', 'gallbladder_wall',
               'liver', 'adhesion', 'omentum', 'peritoneum', 'gut',
               'specimen_bag', 'fluid']

# column names used in the official multi-hot CSVs
_INST_COLS   = ['Grasper','Bipolar','Hook','Scissors','Clipper','Irrigator']
_VERB_COLS   = ['Grasp','Retract','Dissect','Coagulate','Clip','Cut',
                'Aspirate','Irrigate','Pack']
_TARGET_COLS = ['Gallbladder','Cystic_plate','Cystic_duct','Cystic_pedicle',
                'Cystic_artery','Gallbladder_peritoneum','Gallbladder_wall',
                'Liver','Adhesion','Omentum','Peritoneum','Gut',
                'Specimen_bag','Fluid']

# CholecT50 covers these 45 of the 80 Cholec80 videos (1-indexed)
CHOLECT50_VIDEOS = {
    1,2,4,5,6,8,10,12,13,14,15,18,22,23,25,26,27,29,31,32,
    35,36,40,42,43,47,48,49,50,51,52,56,57,60,62,65,66,68,
    70,73,74,75,78,79,80
}


# ── annotation readers ────────────────────────────────────────────────────────

def read_triplet_csv(path: Path) -> Dict[int, List[Tuple[str, str, str]]]:
    """Read a simple per-row triplet CSV/TSV.

    Expected columns (any delimiter): frame_id  instrument  verb  target
    Returns {frame_id: [(inst, verb, tgt), ...]}
    """
    frame_triplets: Dict[int, List[Tuple[str, str, str]]] = defaultdict(list)
    with open(path, newline='') as fh:
        # auto-detect delimiter
        sample = fh.read(2048); fh.seek(0)
        dialect = csv.Sniffer().sniff(sample, delimiters=',\t ')
        reader  = csv.reader(fh, dialect)
        for row in reader:
            row = [c.strip() for c in row]
            if not row or row[0].startswith('#') or not row[0].isdigit():
                continue
            if len(row) < 4:
                continue
            frame_id = int(row[0])
            inst, verb, tgt = row[1].lower(), row[2].lower(), row[3].lower()
            frame_triplets[frame_id].append((inst, verb, tgt))
    return dict(frame_triplets)


def read_multihot_csv(path: Path) -> Dict[int, List[Tuple[str, str, str]]]:
    """Read official CholecT50 multi-hot CSV and recover triplets.

    Triplets are inferred as the Cartesian product of all active instruments,
    verbs, and targets.  This is an approximation — the official per-triplet
    labels are preferred if available.
    """
    frame_triplets: Dict[int, List[Tuple[str, str, str]]] = defaultdict(list)
    with open(path, newline='') as fh:
        reader = csv.DictReader(fh)
        for row in reader:
            try:
                frame_id = int(row['Frame'])
            except (KeyError, ValueError):
                continue
            insts  = [n.lower() for n in _INST_COLS   if row.get(n,'0').strip() == '1']
            verbs  = [n.lower() for n in _VERB_COLS    if row.get(n,'0').strip() == '1']
            tgts   = [n.lower() for n in _TARGET_COLS  if row.get(n,'0').strip() == '1']
            # produce one triplet per (instrument, verb, target) combination
            for inst in (insts or ['null']):
                for verb in (verbs or ['null']):
                    for tgt in (tgts or ['null']):
                        frame_triplets[frame_id].append((inst, verb, tgt))
    return dict(frame_triplets)


def find_anno_file(anno_dir: Path, video_id: int,
                   fmt: str) -> Optional[Path]:
    """Locate annotation file for video_id under anno_dir."""
    candidates = [
        anno_dir / f"VID{video_id:02d}.txt",
        anno_dir / f"video{video_id:02d}.txt",
        anno_dir / f"VID{video_id:02d}.csv",
        anno_dir / f"video{video_id:02d}.csv",
    ]
    for p in candidates:
        if p.exists():
            return p
    return None


def find_video_file(video_dir: Path, video_id: int) -> Optional[Path]:
    """Locate the Cholec80 full-length MP4 for video_id."""
    candidates = [
        video_dir / f"video{video_id:02d}.mp4",
        video_dir / f"VID{video_id:02d}.mp4",
        video_dir / f"video{video_id:02d}.MP4",
    ]
    for p in candidates:
        if p.exists():
            return p
    return None


# ── segmentation ─────────────────────────────────────────────────────────────

def triplets_key(triplets: List[Tuple[str,str,str]]) -> str:
    """Canonical string key for a set of triplets (order-independent)."""
    return '|'.join(sorted(f"{i}/{v}/{t}" for i, v, t in triplets))


def segment_annotations(
    frame_triplets: Dict[int, List[Tuple[str,str,str]]],
    min_sec: int,
) -> List[Tuple[int, int, List[Tuple[str,str,str]]]]:
    """Group contiguous annotation frames with the same triplet set.

    Returns list of (start_anno_frame, end_anno_frame, triplets).
    Both ends are inclusive.  Segments shorter than min_sec are dropped.
    """
    if not frame_triplets:
        return []
    all_frames = sorted(frame_triplets.keys())
    segments = []
    seg_start = all_frames[0]
    seg_key   = triplets_key(frame_triplets[all_frames[0]])
    seg_trips = frame_triplets[all_frames[0]]

    for prev, curr in zip(all_frames, all_frames[1:]):
        curr_key = triplets_key(frame_triplets[curr])
        if curr != prev + 1 or curr_key != seg_key:
            # flush segment
            length = prev - seg_start + 1
            if length >= min_sec:
                segments.append((seg_start, prev, seg_trips))
            seg_start = curr
            seg_key   = curr_key
            seg_trips = frame_triplets[curr]

    # flush last
    last = all_frames[-1]
    length = last - seg_start + 1
    if length >= min_sec:
        segments.append((seg_start, last, seg_trips))

    return segments


# ── clip extraction ───────────────────────────────────────────────────────────

def extract_clip_ffmpeg(
    video_path: Path,
    out_path: Path,
    start_frame: int,
    end_frame: int,
    fps: int,
) -> bool:
    """Extract [start_frame, end_frame] (inclusive, 25 FPS) from video_path."""
    start_sec = start_frame / fps
    duration  = (end_frame - start_frame + 1) / fps
    cmd = [
        "ffmpeg", "-y", "-loglevel", "error",
        "-ss", f"{start_sec:.4f}",
        "-i", str(video_path),
        "-t", f"{duration:.4f}",
        "-c:v", "libx264", "-crf", "18",
        "-an",    # no audio
        str(out_path),
    ]
    result = subprocess.run(cmd, capture_output=True)
    return result.returncode == 0


# ── main ──────────────────────────────────────────────────────────────────────

def build_dataset(args) -> None:
    cholec80_dir = Path(args.cholec80_dir)
    anno_dir     = Path(args.anno_dir)
    out_dir      = Path(args.out_dir)
    clips_dir    = out_dir / "clips"
    clips_dir.mkdir(parents=True, exist_ok=True)

    metadata = []
    total_clips = 0

    for vid_id in sorted(CHOLECT50_VIDEOS):
        anno_path  = find_anno_file(anno_dir, vid_id, args.anno_format)
        video_path = find_video_file(cholec80_dir / "videos", vid_id)

        if anno_path is None:
            print(f"[skip] video{vid_id:02d}: annotation not found in {anno_dir}")
            continue
        if video_path is None:
            print(f"[skip] video{vid_id:02d}: MP4 not found in {cholec80_dir/'videos'}")
            continue

        print(f"\nvideo{vid_id:02d}: {anno_path.name}")

        if args.anno_format == "cholect50_official":
            frame_triplets = read_multihot_csv(anno_path)
        else:
            frame_triplets = read_triplet_csv(anno_path)

        segments = segment_annotations(frame_triplets, min_sec=args.min_sec)
        print(f"  {len(frame_triplets)} annotated seconds → {len(segments)} segments")

        clip_idx = 1
        for anno_start, anno_end, trips in segments:
            # Map 1-FPS annotation indices to 25-FPS frame indices
            frame_start = anno_start * args.fps
            frame_end   = (anno_end + 1) * args.fps - 1   # inclusive

            clip_name = f"video{vid_id:02d}_clip{clip_idx:02d}.mp4"
            out_path  = clips_dir / clip_name

            if not out_path.exists() or args.overwrite:
                ok = extract_clip_ffmpeg(
                    video_path, out_path,
                    frame_start, frame_end, args.fps,
                )
                status = "ok" if ok else "FAIL"
            else:
                status = "skip(exists)"

            if status == "ok" or status == "skip(exists)":
                metadata.append({
                    "video":       clip_name,
                    "source_video": video_path.name,
                    "frame_range": [anno_start, anno_end],   # 1-FPS annotation range
                    "fps_range":   [frame_start, frame_end], # 25-FPS Cholec80 range
                    "triplets":    [list(t) for t in trips],
                })
                total_clips += 1
            clip_idx += 1

        print(f"  {clip_idx - 1} clips extracted")

    meta_path = out_dir / "clips_metadata.json"
    with open(meta_path, "w") as f:
        json.dump(metadata, f, indent=2)

    print(f"\n{'='*60}")
    print(f"Total clips : {total_clips}")
    print(f"Clips dir   : {clips_dir}")
    print(f"Metadata    : {meta_path}")
    print("Next step   : python 02_generate_captions.py")


def parse_args():
    p = argparse.ArgumentParser(
        description="Extract action clips from Cholec80 using CholecT50 triplet annotations."
    )
    p.add_argument("--cholec80_dir", required=True,
                   help="Root directory of Cholec80 (contains videos/ subdirectory).")
    p.add_argument("--anno_dir", required=True,
                   help="Directory containing CholecT50 per-video annotation files.")
    p.add_argument("--out_dir", default="data/cholec80_action",
                   help="Output directory for clips + metadata (default: data/cholec80_action).")
    p.add_argument("--anno_format", default="triplet_csv",
                   choices=["triplet_csv", "cholect50_official"],
                   help="Annotation file format (default: triplet_csv).")
    p.add_argument("--min_sec", type=int, default=3,
                   help="Minimum segment duration in seconds (default: 3).")
    p.add_argument("--fps", type=int, default=25,
                   help="Cholec80 frame rate (default: 25).")
    p.add_argument("--overwrite", action="store_true",
                   help="Re-extract clips that already exist on disk.")
    return p.parse_args()


if __name__ == "__main__":
    build_dataset(parse_args())
