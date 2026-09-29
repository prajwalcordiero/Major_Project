#!/usr/bin/env python3
"""
ResQ-AI — Human-readable report generator
==========================================
Turns the CSV output of compare_models.py into a single, readable HTML
report you can open in any browser (or print to PDF from the browser's
Print dialog if you need a PDF/paper copy).

USAGE
-----
    python generate_report.py --dir .

Reads, from --dir (default: current folder):
    per_frame_comparison.csv    (required)
    accuracy_summary.csv        (optional — skipped if not found)
    comparison_chart.png        (optional — embedded if found)

Writes:
    report.html
"""

import argparse
import base64
import os
from datetime import datetime

import pandas as pd


def embed_image(path):
    if not os.path.exists(path):
        return None
    with open(path, "rb") as f:
        data = base64.b64encode(f.read()).decode("ascii")
    return f"data:image/png;base64,{data}"


def build_verdict(summary_df):
    if summary_df is None or summary_df.empty:
        return ("<p>No ground-truth accuracy data was available for this run — "
                "only speed and per-frame counts are shown below.</p>")

    by_model = {r["model"]: r for _, r in summary_df.iterrows()}
    if "Faster R-CNN" not in by_model or "YOLOv8" not in by_model:
        return ""

    frcnn, yolo = by_model["Faster R-CNN"], by_model["YOLOv8"]

    more_accurate = "Faster R-CNN" if frcnn["MAE"] < yolo["MAE"] else "YOLOv8"
    less_accurate = "YOLOv8" if more_accurate == "Faster R-CNN" else "Faster R-CNN"
    mae_diff = abs(frcnn["MAE"] - yolo["MAE"])
    better_mae = frcnn["MAE"] if more_accurate == "Faster R-CNN" else yolo["MAE"]
    worse_mae = yolo["MAE"] if more_accurate == "Faster R-CNN" else frcnn["MAE"]

    frcnn_fps = frcnn.get("fps", 0) or 0
    yolo_fps = yolo.get("fps", 0) or 0
    faster_model = "YOLOv8" if yolo_fps > frcnn_fps else "Faster R-CNN"
    slower_model = "Faster R-CNN" if faster_model == "YOLOv8" else "YOLOv8"
    ratio = (yolo_fps / frcnn_fps) if faster_model == "YOLOv8" and frcnn_fps else \
            (frcnn_fps / yolo_fps if yolo_fps else 1.0)

    return f"""
    <div class="verdict">
      <h2>Verdict</h2>
      <ul>
        <li><b>{more_accurate}</b> was more accurate on your labeled frames — mean absolute
            error of {better_mae:.2f} people/frame vs {worse_mae:.2f} for {less_accurate}
            (a difference of {mae_diff:.2f}).</li>
        <li><b>{faster_model}</b> ran about {ratio:.1f}&times; faster than {slower_model}
            ({frcnn_fps:.2f} vs {yolo_fps:.2f} FPS).</li>
        <li>If accuracy and speed point to different models, it's a real trade-off: pick
            {more_accurate} for offline/careful analysis where every count matters, or
            {faster_model} if you need to process a live feed in real time and can tolerate
            a bit more counting error.</li>
      </ul>
    </div>
    """


def main():
    ap = argparse.ArgumentParser(description="Generate a readable HTML report from compare_models.py output")
    ap.add_argument("--dir", default=".", help="folder containing the CSV/PNG output files")
    ap.add_argument("--output", default=None, help="output HTML path (default: <dir>/report.html)")
    args = ap.parse_args()

    per_frame_path = os.path.join(args.dir, "per_frame_comparison.csv")
    summary_path = os.path.join(args.dir, "accuracy_summary.csv")
    chart_path = os.path.join(args.dir, "comparison_chart.png")
    out_path = args.output or os.path.join(args.dir, "report.html")

    if not os.path.exists(per_frame_path):
        print(f"[ERROR] {per_frame_path} not found — run compare_models.py first.")
        return

    per_frame = pd.read_csv(per_frame_path)
    summary = pd.read_csv(summary_path) if os.path.exists(summary_path) else None
    chart_uri = embed_image(chart_path)

    verdict_html = build_verdict(summary)
    summary_html = (
        summary.to_html(index=False, classes="table", border=0)
        if summary is not None
        else "<p><i>No accuracy_summary.csv found (no ground truth was supplied to compare_models.py).</i></p>"
    )
    per_frame_html = per_frame.to_html(index=False, classes="table", border=0)
    chart_html = (
        f'<img src="{chart_uri}" alt="Comparison chart" '
        f'style="max-width:100%;border:1px solid #333;border-radius:8px;margin:16px 0;">'
        if chart_uri else "<p><i>No comparison_chart.png found (only created when --gt-csv is supplied).</i></p>"
    )

    html = f"""<!DOCTYPE html>
<html>
<head>
<meta charset="utf-8">
<title>ResQ-AI Detector Comparison Report</title>
<style>
  body {{ font-family: -apple-system, "Segoe UI", Roboto, Arial, sans-serif; background:#0f1115; color:#e6e6e6; margin:0; padding:32px; }}
  .wrap {{ max-width: 960px; margin: 0 auto; }}
  h1 {{ font-size: 26px; margin-bottom:4px; }}
  h2 {{ font-size:18px; margin-top:32px; border-bottom:1px solid #333; padding-bottom:6px; }}
  .meta {{ color:#9aa0a6; font-size:13px; margin-bottom:24px; }}
  table.table {{ border-collapse: collapse; width:100%; margin: 12px 0 24px; font-size:13px; }}
  table.table th {{ background:#1c1f26; text-align:left; padding:8px 10px; border-bottom:2px solid #333; }}
  table.table td {{ padding:6px 10px; border-bottom:1px solid #23262d; }}
  table.table tr:nth-child(even) td {{ background:#15171c; }}
  .verdict {{ background:#141a12; border-left:4px solid #22c55e; padding:14px 18px; border-radius:6px; }}
  .verdict li {{ margin-bottom:8px; line-height:1.5; }}
  .scroll {{ max-height: 420px; overflow:auto; border:1px solid #23262d; border-radius:8px; }}
  @media print {{ body {{ background:#fff; color:#000; }} .scroll {{ max-height:none; overflow:visible; }} }}
</style>
</head>
<body>
<div class="wrap">
  <h1>🚨 ResQ-AI — Detector Comparison Report</h1>
  <div class="meta">Faster R-CNN vs YOLOv8 &nbsp;·&nbsp; Generated {datetime.now().strftime("%Y-%m-%d %H:%M")}</div>

  {verdict_html}

  <h2>Accuracy &amp; speed summary</h2>
  {summary_html}

  <h2>True vs. predicted counts (chart)</h2>
  {chart_html}

  <h2>Per-frame results</h2>
  <div class="scroll">{per_frame_html}</div>
</div>
</body>
</html>
"""

    with open(out_path, "w", encoding="utf-8") as f:
        f.write(html)

    print(f"[SUCCESS] Report written -> {out_path}")
    print("Open it by double-clicking the file, or dragging it into any browser.")


if __name__ == "__main__":
    main()