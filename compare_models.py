#!/usr/bin/env python3
"""
ResQ-AI — Detector comparison: Faster R-CNN vs YOLOv8
======================================================
Runs BOTH detectors on the same frames and reports, for each:
    - people count per frame
    - inference speed (ms/frame, FPS)
    - accuracy against your known ground-truth counts (MAE, RMSE, MAPE,
      % of frames within ±1 person, % within ±10%)

WHY COUNT-BASED ACCURACY (not mAP)
-----------------------------------
mAP needs ground-truth bounding boxes per person. You said you only have
known TOTAL counts for some frames, not boxes — so this script scores
both models the way that's actually measurable from what you have: how
close each model's predicted count is to the true count. This is exactly
the number that matters downstream anyway, since density/risk are both
derived from the count.

GROUND TRUTH FILE FORMAT (--gt-csv)
------------------------------------
A plain CSV with two columns:

    frame,true_count
    0000.jpg,42
    0007.jpg,51
    frame_120,38

- If --input is a FOLDER of images, "frame" must match the image filename.
- If --input is a VIDEO, "frame" must match the frame index (0-based,
  after applying --frame-step) as a plain integer string, e.g. "120".

You only need rows for the frames you actually know the true count for —
everything else is still counted and shown, just excluded from accuracy.

USAGE
-----
  # A folder of still images with a few known counts
  python compare_models.py --input ./known_frames --gt-csv ground_truth.csv

  # A video, sampling every 30th frame, with counts known for a few of them
  python compare_models.py --input crowd.mp4 --frame-step 30 --gt-csv ground_truth.csv

  # Restrict counting to your calibrated ROI (fairer, matches app.py behaviour)
  python compare_models.py --input crowd.mp4 --calib-file calibration.json --gt-csv ground_truth.csv

Outputs (in --output-dir):
  per_frame_comparison.csv   - every frame, both models' counts + timings
  accuracy_summary.csv       - MAE/RMSE/MAPE/etc for each model
  comparison_chart.png       - true vs predicted counts on ground-truth frames
"""

import argparse
import os
import time

import cv2
import numpy as np
import pandas as pd

from resq_ai_analyzer import DEVICE, FrameSource, GroundPlane, PersonDetector, to_bgr3
from yolo_detector import YOLOPersonDetector

IMAGE_EXTS = (".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff")


# ─────────────────────────── Frame gathering ─────────────────────────

def gather_frames_from_dir(path):
    files = sorted(f for f in os.listdir(path) if f.lower().endswith(IMAGE_EXTS))
    for fname in files:
        img = cv2.imread(os.path.join(path, fname), cv2.IMREAD_UNCHANGED)
        if img is None:
            print(f"[WARN] Could not read {fname}, skipping")
            continue
        yield fname, to_bgr3(img)


def gather_frames_from_source(path, frame_step):
    source = FrameSource(path)
    idx = 0
    while True:
        frame = source.read()
        if frame is None:
            break
        if idx % max(frame_step, 1) == 0:
            yield str(idx), frame
        idx += 1
    source.release()


# ─────────────────────────── Counting helper ─────────────────────────

def count_people(detector, frame, ground=None):
    t0 = time.time()
    boxes, _ = detector.detect(frame)
    elapsed = time.time() - t0

    if ground is not None:
        feet = detector.foot_points(boxes)
        count = int(ground.inside_roi(feet).sum()) if len(feet) else 0
    else:
        count = len(boxes)

    return count, elapsed


# ─────────────────────────── Metrics ──────────────────────────────────

def compute_metrics(df, pred_col, true_col):
    err = df[pred_col] - df[true_col]
    abs_err = err.abs()
    safe_true = df[true_col].replace(0, np.nan)

    return {
        "MAE": round(float(abs_err.mean()), 3),
        "RMSE": round(float(np.sqrt((err ** 2).mean())), 3),
        "MAPE_pct": round(float((abs_err / safe_true).mean() * 100), 2),
        "Within_1_person_pct": round(float((abs_err <= 1).mean() * 100), 1),
        "Within_10pct_pct": round(float((abs_err <= 0.1 * df[true_col]).mean() * 100), 1),
        "Bias_mean_signed_error": round(float(err.mean()), 3),
        "N_frames": int(len(df)),
    }


# ─────────────────────────────── Main ─────────────────────────────────

def main():
    ap = argparse.ArgumentParser(description="Compare Faster R-CNN vs YOLOv8 for crowd counting")
    ap.add_argument("--input", required=True, help="video file, image file, or folder of images")
    ap.add_argument("--gt-csv", default=None, help="CSV with columns: frame,true_count")
    ap.add_argument("--calib-file", default=None, help="calibration.json to restrict counts to ROI")
    ap.add_argument("--yolo-weights", default="yolov8n.pt")
    ap.add_argument("--conf", type=float, default=0.5)
    ap.add_argument("--frame-step", type=int, default=30, help="sample every Nth frame (video only)")
    ap.add_argument("--output-dir", default=".")
    args = ap.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    # ── Load both detectors ────────────────────────────────────────
    frcnn = PersonDetector(DEVICE, conf=args.conf)
    yolo = YOLOPersonDetector(args.yolo_weights, device=DEVICE, conf=args.conf)

    # ── Optional shared ROI, so both models are judged on the same area ──
    ground = None
    if args.calib_file and os.path.exists(args.calib_file):
        import json
        with open(args.calib_file, "r", encoding="utf-8") as f:
            calib = json.load(f)
        ground = GroundPlane.from_calibration(calib)
        print(f"[INFO] Counting restricted to calibrated ROI ({ground.area_m2:.1f} m²)")
    else:
        print("[INFO] No calibration file given — counting all detections in the full frame")

    # ── Gather frames ──────────────────────────────────────────────
    if os.path.isdir(args.input):
        frame_iter = gather_frames_from_dir(args.input)
    else:
        frame_iter = gather_frames_from_source(args.input, args.frame_step)

    rows = []
    for key, frame in frame_iter:
        frcnn_count, frcnn_t = count_people(frcnn, frame, ground)
        yolo_count, yolo_t = count_people(yolo, frame, ground)

        rows.append({
            "frame": key,
            "frcnn_count": frcnn_count,
            "yolo_count": yolo_count,
            "frcnn_ms": round(frcnn_t * 1000, 1),
            "yolo_ms": round(yolo_t * 1000, 1),
        })

        print(f"  {key:>12s}  FRCNN={frcnn_count:4d} ({frcnn_t*1000:6.1f} ms)   "
              f"YOLO={yolo_count:4d} ({yolo_t*1000:6.1f} ms)")

    df = pd.DataFrame(rows)
    if df.empty:
        print("[ERROR] No frames were processed — check --input.")
        return

    per_frame_path = os.path.join(args.output_dir, "per_frame_comparison.csv")
    df.to_csv(per_frame_path, index=False)

    # ── Speed summary (always available, no GT needed) ─────────────
    print("\n=== SPEED ===")
    speed_rows = []
    for name, col in [("Faster R-CNN", "frcnn_ms"), ("YOLOv8", "yolo_ms")]:
        mean_ms = df[col].mean()
        fps = 1000.0 / mean_ms if mean_ms > 0 else float("inf")
        speed_rows.append({"model": name, "avg_ms_per_frame": round(mean_ms, 1), "fps": round(fps, 2)})
        print(f"  {name:<14s}  {mean_ms:7.1f} ms/frame   ~{fps:5.2f} FPS")

    # ── Agreement between the two models (works even with no GT) ───
    diff = (df["frcnn_count"] - df["yolo_count"]).abs()
    print(f"\n=== MODEL AGREEMENT (no ground truth needed) ===")
    print(f"  Mean |FRCNN - YOLO| count difference: {diff.mean():.2f} people/frame")
    print(f"  Frames where they match exactly:      {(diff == 0).mean() * 100:.1f}%")

    # ── Accuracy against ground truth, if provided ──────────────────
    summary_rows = []
    if args.gt_csv and os.path.exists(args.gt_csv):
        gt = pd.read_csv(args.gt_csv, dtype={"frame": str})
        merged = df.merge(gt, on="frame", how="inner")

        if merged.empty:
            print(f"\n[WARN] None of the 'frame' values in {args.gt_csv} matched processed "
                  f"frames. Check that they use the same filenames / frame indices.")
        else:
            print(f"\n=== ACCURACY vs GROUND TRUTH ({len(merged)} labeled frames) ===")
            for name, col in [("Faster R-CNN", "frcnn_count"), ("YOLOv8", "yolo_count")]:
                m = compute_metrics(merged, col, "true_count")
                m["model"] = name
                summary_rows.append(m)
                print(f"\n  {name}")
                for k, v in m.items():
                    if k != "model":
                        print(f"    {k:<22s}: {v}")

            # ── Chart: true vs predicted on the labeled frames ───────
            try:
                import matplotlib.pyplot as plt

                plotdf = merged.sort_values("frame")
                x = range(len(plotdf))
                plt.figure(figsize=(10, 5))
                plt.plot(x, plotdf["true_count"], "o-", label="Ground truth", color="black")
                plt.plot(x, plotdf["frcnn_count"], "s--", label="Faster R-CNN", color="tab:blue")
                plt.plot(x, plotdf["yolo_count"], "^--", label="YOLOv8", color="tab:orange")
                plt.xticks(list(x), plotdf["frame"], rotation=45, ha="right")
                plt.ylabel("People count")
                plt.title("Predicted vs. true crowd count")
                plt.legend()
                plt.tight_layout()
                chart_path = os.path.join(args.output_dir, "comparison_chart.png")
                plt.savefig(chart_path, dpi=150)
                print(f"\n[INFO] Chart saved -> {chart_path}")
            except ImportError:
                print("[WARN] matplotlib not installed, skipping chart (pip install matplotlib)")
    else:
        print("\n[INFO] No --gt-csv provided (or file not found) — skipping accuracy metrics. "
              "Only speed and model-agreement stats are shown above.")

    if summary_rows:
        summary_path = os.path.join(args.output_dir, "accuracy_summary.csv")
        acc_df = pd.DataFrame(summary_rows)
        speed_df = pd.DataFrame(speed_rows)
        acc_df.merge(speed_df, on="model", how="left").to_csv(summary_path, index=False)
        print(f"[INFO] Accuracy + speed summary saved -> {summary_path}")

    print(f"[INFO] Per-frame comparison saved -> {per_frame_path}")


if __name__ == "__main__":
    main()