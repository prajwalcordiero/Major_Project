#!/usr/bin/env python3
"""
ResQ-AI — Live Crowd Risk Dashboard
====================================
A CCTV-style monitoring app. Point it at a video file, a webcam or an RTSP
stream and it analyses frame by frame as the footage plays: live people count,
live density heatmap, live crowd-risk verdict, and a rolling risk timeline.

Run locally:
    python -m streamlit run app.py

Keep app.py and resq_ai_analyzer.py in the same folder.
"""

import html
import os
import tempfile
import time
from datetime import datetime
from collections import deque

import cv2
import pandas as pd
import streamlit as st

from resq_ai_analyzer import (
    D_COMFORTABLE,
    D_CRITICAL,
    D_DANGEROUS,
    D_RESTRICTED,
    DEVICE,
    RISK_CRITICAL,
    RISK_DANGER,
    RISK_WARN,
    CrowdAnalyzer,
    FrameSource,
    PersonDetector,
)


# ─────────────────────────── Streamlit config ──────────────────────

st.set_page_config(
    page_title="ResQ-AI | Crowd Risk Monitor",
    page_icon="🚨",
    layout="wide",
)


# ─────────────────────────── Styling ───────────────────────────────

st.markdown(
    """
    <style>
      .block-container {
          padding-top: 2rem;
          padding-bottom: 1rem;
      }

      .banner {
          padding: 14px 20px;
          border-radius: 10px;
          font-size: 22px;
          font-weight: 700;
          letter-spacing: .5px;
          margin-bottom: 12px;
      }

      .b-ok {
          background: #0d3b1e;
          color: #4ade80;
          border-left: 8px solid #22c55e;
      }

      .b-warn {
          background: #3b2f0d;
          color: #fbbf24;
          border-left: 8px solid #f59e0b;
      }

      .b-dang {
          background: #3b1d0d;
          color: #fb923c;
          border-left: 8px solid #ea580c;
      }

      .b-crit {
          background: #3b0d0d;
          color: #f87171;
          border-left: 8px solid #ef4444;
          animation: pulse 1s infinite;
      }

      @keyframes pulse {
          0% {
              opacity: 1;
          }

          50% {
              opacity: .55;
          }

          100% {
              opacity: 1;
          }
      }
    </style>
    """,
    unsafe_allow_html=True,
)


# ───────────────────────── Model caching ───────────────────────────

@st.cache_resource(show_spinner="Loading Faster R-CNN ResNet-50 detector…")
def get_detector(conf: float):
    return PersonDetector(DEVICE, conf=conf)


def banner_class(score):
    if score >= RISK_CRITICAL:
        return "b-crit"

    if score >= RISK_DANGER:
        return "b-dang"

    if score >= RISK_WARN:
        return "b-warn"

    return "b-ok"


def _safe_max(series, default=0.0):
    """Return a numeric maximum without failing on an empty/invalid series."""
    if series is None or len(series) == 0:
        return default
    values = pd.to_numeric(series, errors="coerce").dropna()
    return float(values.max()) if not values.empty else default


def _safe_mean(series, default=0.0):
    """Return a numeric mean without failing on an empty/invalid series."""
    if series is None or len(series) == 0:
        return default
    values = pd.to_numeric(series, errors="coerce").dropna()
    return float(values.mean()) if not values.empty else default


def build_session_summary(df):
    """Build a compact summary from one analysis run."""
    if df is None or df.empty:
        return None

    time_values = pd.to_numeric(df.get("time_s"), errors="coerce").dropna()
    duration = float(time_values.max()) if not time_values.empty else 0.0

    risk = pd.to_numeric(df.get("risk"), errors="coerce").fillna(0.0)
    count = pd.to_numeric(df.get("count"), errors="coerce").fillna(0.0)
    mean_density = pd.to_numeric(
        df.get("mean_density"), errors="coerce"
    ).fillna(0.0)
    peak_density = pd.to_numeric(
        df.get("peak_density"), errors="coerce"
    ).fillna(0.0)

    # Estimate time spent in each risk band from consecutive samples.
    time_in_bands = {
        "Normal": 0.0,
        "Warning": 0.0,
        "Danger": 0.0,
        "Critical": 0.0,
    }

    if len(df) > 1 and "time_s" in df.columns:
        t = pd.to_numeric(df["time_s"], errors="coerce").to_numpy()
        r = risk.to_numpy()
        for i in range(len(df) - 1):
            if pd.isna(t[i]) or pd.isna(t[i + 1]):
                continue
            dt = max(0.0, float(t[i + 1] - t[i]))
            score = float(r[i])

            if score >= RISK_CRITICAL:
                time_in_bands["Critical"] += dt
            elif score >= RISK_DANGER:
                time_in_bands["Danger"] += dt
            elif score >= RISK_WARN:
                time_in_bands["Warning"] += dt
            else:
                time_in_bands["Normal"] += dt

    alarm_count = 0
    if "alarm" in df.columns:
        alarm_count = int(
            pd.Series(df["alarm"]).fillna(False).astype(bool).sum()
        )

    return {
        "generated_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "samples": int(len(df)),
        "duration_s": duration,
        "max_people": int(round(_safe_max(count))),
        "average_people": _safe_mean(count),
        "max_density": _safe_max(peak_density),
        "average_density": _safe_mean(mean_density),
        "max_risk": _safe_max(risk),
        "average_risk": _safe_mean(risk),
        "alarm_samples": alarm_count,
        "time_in_bands": time_in_bands,
    }


def build_html_report(summary, df):
    """Create a standalone HTML report that can be saved or shared."""
    if not summary:
        return ""

    band_rows = "".join(
        f"<tr><td>{html.escape(label)}</td><td>{seconds:.1f} s</td></tr>"
        for label, seconds in summary["time_in_bands"].items()
    )

    table_rows = ""
    if df is not None and not df.empty:
        columns = [
            c
            for c in [
                "time_s",
                "count",
                "mean_density",
                "peak_density",
                "risk",
                "status",
                "alarm",
            ]
            if c in df.columns
        ]
        preview = df[columns].tail(100).copy()
        table_rows = preview.to_html(
            index=False,
            classes="metrics",
            border=0,
        )

    return f"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>ResQ-AI Crowd Risk Report</title>
<style>
body {{ font-family: Arial, sans-serif; margin: 32px; background:#0f172a; color:#e2e8f0; }}
.card {{ background:#1e293b; border-radius:12px; padding:20px; margin:16px 0; }}
h1 {{ margin-bottom:4px; }}
.grid {{ display:grid; grid-template-columns:repeat(auto-fit,minmax(180px,1fr)); gap:12px; }}
.metric {{ background:#334155; padding:16px; border-radius:10px; }}
.value {{ font-size:26px; font-weight:700; margin-top:6px; }}
table {{ width:100%; border-collapse:collapse; }}
th,td {{ padding:8px; border-bottom:1px solid #475569; text-align:left; }}
.small {{ color:#94a3b8; }}
</style>
</head>
<body>
<h1>🚨 ResQ-AI Crowd Risk Report</h1>
<p class="small">Generated {html.escape(summary["generated_at"])}</p>

<div class="card">
<div class="grid">
<div class="metric"><div>Maximum people</div><div class="value">{summary["max_people"]}</div></div>
<div class="metric"><div>Average people</div><div class="value">{summary["average_people"]:.1f}</div></div>
<div class="metric"><div>Maximum density</div><div class="value">{summary["max_density"]:.2f} p/m²</div></div>
<div class="metric"><div>Average density</div><div class="value">{summary["average_density"]:.2f} p/m²</div></div>
<div class="metric"><div>Maximum risk</div><div class="value">{summary["max_risk"] * 100:.1f}%</div></div>
<div class="metric"><div>Samples</div><div class="value">{summary["samples"]}</div></div>
</div>
</div>

<div class="card">
<h2>Estimated time by risk band</h2>
<table><thead><tr><th>Risk band</th><th>Time</th></tr></thead>
<tbody>{band_rows}</tbody></table>
<p class="small">Time is estimated from consecutive analysis samples and depends on playback/frame-skip settings.</p>
</div>

<div class="card">
<h2>Metrics (last 100 samples)</h2>
{table_rows}
</div>
</body>
</html>"""


# ───────────────────────── Sidebar ─────────────────────────────────

st.sidebar.title("ResQ-AI Control")

st.sidebar.caption(
    f"Inference device: **{str(DEVICE).upper()}**"
)


# ───────────────────────── Video source ────────────────────────────

src_kind = st.sidebar.radio(
    "Video source",
    ["Upload file", "Local path", "Webcam / RTSP"],
)

source_path = None


if src_kind == "Upload file":

    up = st.sidebar.file_uploader(
        "Video or image",
        type=[
            "mp4",
            "avi",
            "mov",
            "mkv",
            "png",
            "jpg",
            "jpeg",
        ],
    )

    if up is not None:
        suffix = os.path.splitext(up.name)[1]

        tmp = tempfile.NamedTemporaryFile(
            delete=False,
            suffix=suffix,
        )

        tmp.write(up.read())
        tmp.close()

        source_path = tmp.name


elif src_kind == "Local path":

    source_path = (
        st.sidebar.text_input(
            "Path",
            "crowd.mp4",
        )
        or None
    )


else:

    source_path = (
        st.sidebar.text_input(
            "Webcam index or RTSP URL",
            "0",
        )
        or None
    )


# ───────────────────────── Calibration ─────────────────────────────

st.sidebar.divider()

st.sidebar.subheader("Scale calibration")

calib_mode = st.sidebar.radio(
    "Mode",
    [
        "Auto (person height)",
        "calibration.json",
        "Enter 4 points",
    ],
    help=(
        "Auto ignores perspective — fine for a quick demo, "
        "not for real metres."
    ),
)

calib = None


if calib_mode == "calibration.json":

    cpath = st.sidebar.text_input(
        "File",
        "calibration.json",
    )

    if os.path.exists(cpath):

        import json

        with open(cpath, "r", encoding="utf-8") as f:
            calib = json.load(f)

        st.sidebar.success("Calibration loaded")

    else:

        st.sidebar.warning(
            "File not found — falling back to auto"
        )


elif calib_mode == "Enter 4 points":

    st.sidebar.caption(
        "Pixel coords, in order: "
        "near-left, near-right, far-right, far-left"
    )

    raw = st.sidebar.text_input(
        "Points",
        "180,610 980,610 840,330 320,330",
    )

    cw = st.sidebar.number_input(
        "Real width (m)",
        min_value=1.0,
        max_value=500.0,
        value=12.0,
    )

    cl = st.sidebar.number_input(
        "Real depth (m)",
        min_value=1.0,
        max_value=500.0,
        value=8.0,
    )

    try:

        pts = [
            [float(v) for v in p.split(",")]
            for p in raw.split()
        ]

        if len(pts) == 4:

            calib = {
                "image_points": pts,
                "world_points": [
                    [0, 0],
                    [cw, 0],
                    [cw, cl],
                    [0, cl],
                ],
            }

            st.sidebar.success(
                f"Reference area {cw * cl:.1f} m²"
            )

    except ValueError:

        st.sidebar.error(
            "Could not parse points"
        )


# ───────────────────────── Performance ─────────────────────────────

st.sidebar.divider()

st.sidebar.subheader("Performance")

conf = st.sidebar.slider(
    "Detection confidence",
    min_value=0.1,
    max_value=0.9,
    value=0.5,
    step=0.05,
)

detect_every = st.sidebar.slider(
    "Detect every N frames",
    min_value=1,
    max_value=8,
    value=1,
    help=(
        "Raise this on CPU. "
        "Boxes persist between detections."
    ),
)

frame_skip = st.sidebar.slider(
    "Playback stride",
    min_value=1,
    max_value=10,
    value=1,
    help=(
        "Skip frames to keep up with a live stream."
    ),
)

show_boxes = st.sidebar.checkbox(
    "Show detection boxes",
    True,
)

realtime = st.sidebar.checkbox(
    "Throttle to source FPS",
    False,
)


# ───────────────────────── Start / Stop ────────────────────────────

c1, c2 = st.sidebar.columns(2)

if c1.button(
    "▶ Start",
    width="stretch",
    type="primary",
):
    st.session_state.running = True


if c2.button(
    "■ Stop",
    width="stretch",
):
    st.session_state.running = False


if st.sidebar.button("🗑 Clear previous results", width="stretch"):
    st.session_state.history = None
    st.session_state.summary = None
    st.session_state.running = False
    st.rerun()


# Initialize session state

st.session_state.setdefault(
    "running",
    False,
)

st.session_state.setdefault(
    "history",
    None,
)

st.session_state.setdefault(
    "summary",
    None,
)


# ───────────────────────── Main layout ────────────────────────────

st.title(
    "ResQ-AI — Real-Time Crowd Risk Monitor"
)
st.caption(
    "Live people detection, crowd density, risk scoring and post-run analytics."
)


banner_ph = st.empty()


kpi_ph = st.container()

k1, k2, k3, k4, k5 = kpi_ph.columns(5)

m_count = k1.empty()
m_area = k2.empty()
m_avg = k3.empty()
m_peak = k4.empty()
m_risk = k5.empty()


left, right = st.columns([3, 1])


video_ph = left.empty()


right.markdown("**Bird's-eye density**")

map_ph = right.empty()


right.markdown("**Density scale**")

right.caption(
    f"🟢 < {D_COMFORTABLE} p/m² free flow  \n"
    f"🟡 {D_COMFORTABLE}–{D_RESTRICTED} busy  \n"
    f"🟠 {D_RESTRICTED}–{D_DANGEROUS} restricted  \n"
    f"🔴 {D_DANGEROUS}–{D_CRITICAL} dangerous  \n"
    f"⛔ > {D_CRITICAL} crush conditions"
)


log_ph = right.empty()


st.markdown("### Risk timeline")

chart_ph = st.empty()

dl_ph = st.empty()


# ───────────────────────── Analytics helpers ────────────────────────


def render_session_report(df):
    """Render post-run analytics and export controls."""
    if df is None or df.empty:
        return

    summary = build_session_summary(df)
    st.session_state.summary = summary

    st.markdown("## 📊 Session analytics")
    st.caption(
        f"Processed {summary['samples']:,} analysis samples over "
        f"{summary['duration_s']:.1f} seconds."
    )

    a1, a2, a3, a4, a5 = st.columns(5)
    a1.metric("Peak people", summary["max_people"])
    a2.metric("Average people", f"{summary['average_people']:.1f}")
    a3.metric("Peak density", f"{summary['max_density']:.2f} p/m²")
    a4.metric("Peak risk", f"{summary['max_risk'] * 100:.1f}%")
    a5.metric("Alarm samples", summary["alarm_samples"])

    st.markdown("### Risk & density history")
    chart_df = df.copy()
    chart_cols = [
        c for c in ["risk", "peak_density"]
        if c in chart_df.columns
    ]
    if "time_s" in chart_df.columns and chart_cols:
        st.line_chart(
            chart_df.set_index("time_s")[chart_cols]
        )

    b1, b2, b3 = st.columns(3)
    b1.download_button(
        "⬇ Download metrics CSV",
        df.to_csv(index=False),
        "resq_ai_metrics.csv",
        "text/csv",
        width="stretch",
    )
    b2.download_button(
        "⬇ Download HTML report",
        build_html_report(summary, df),
        "resq_ai_report.html",
        "text/html",
        width="stretch",
    )
    if b3.button("🗑 Clear session report", width="stretch"):
        st.session_state.history = None
        st.session_state.summary = None
        st.rerun()

    with st.expander("Risk-band timing"):
        timing = pd.DataFrame(
            [
                {
                    "Risk band": label,
                    "Estimated time (s)": round(seconds, 1),
                }
                for label, seconds in summary["time_in_bands"].items()
            ]
        )
        st.dataframe(
            timing,
            hide_index=True,
            width="stretch",
        )


def render_idle():
    banner_ph.markdown(
        '<div class="banner b-ok">'
        'STANDBY — no active stream'
        '</div>',
        unsafe_allow_html=True,
    )

    for ph, lbl in [
        (m_count, "People"),
        (m_area, "Area"),
        (m_avg, "Avg density"),
        (m_peak, "Peak density"),
        (m_risk, "Risk"),
    ]:
        ph.metric(lbl, "—")

    if st.session_state.history is not None:
        df = st.session_state.history

        st.markdown("### Last completed run")

        chart_cols = [
            c for c in ["risk", "peak_density"]
            if c in df.columns
        ]
        if "time_s" in df.columns and chart_cols:
            chart_ph.line_chart(
                df.set_index("time_s")[chart_cols]
            )

        render_session_report(df)
    else:
        st.info(
            "Pick a source in the sidebar and press **Start**. "
            "When the run finishes, session analytics and an HTML "
            "report will appear here."
        )


# ───────────────────────── Main processing loop ────────────────────

if not st.session_state.running or not source_path:

    render_idle()

    if st.session_state.history is not None:

        df = st.session_state.history

        chart_ph.line_chart(
            df.set_index("time_s")[
                ["risk", "peak_density"]
            ]
        )

        dl_ph.download_button(
            "⬇ Download metrics CSV",
            df.to_csv(index=False),
            "resq_ai_metrics.csv",
            "text/csv",
        )

    else:

        st.info(
            "Pick a source in the sidebar and press **Start**."
        )


else:

    # ───────────────────── Open source ─────────────────────────────

    try:

        source = FrameSource(source_path)

        first = source.first_frame()

    except IOError as e:

        st.error(str(e))

        st.session_state.running = False

        st.stop()


    # ───────────────────── Load detector ───────────────────────────

    detector = get_detector(conf)


    # ───────────────────── Initialize analyzer ─────────────────────

    try:

        engine = CrowdAnalyzer(
            first,
            source.fps,
            calib=calib,
            detector=detector,
        )

    except ValueError as e:

        st.error(
            f"{e} Detected nobody in the first frame — "
            "lower the confidence threshold or supply a calibration."
        )

        st.session_state.running = False

        st.stop()


    # ───────────────────── Calibration warning ─────────────────────

    if not engine.calibrated:

        st.warning(
            "Running uncalibrated: scale is estimated from "
            "average person height and perspective is not corrected. "
            "Area and density are approximate.",
            icon="⚠️",
        )


    # ───────────────────── Runtime variables ───────────────────────

    rows = []
    st.session_state.summary = None

    risk_hist = deque(
        maxlen=600
    )

    alerts = deque(
        maxlen=6
    )

    delay = (
        1.0 / source.fps
        if realtime
        else 0.0
    )

    idx = 0


    # ───────────────────── Frame loop ─────────────────────────────

    while st.session_state.running:

        frame = source.read()

        if frame is None:
            break

        idx += 1


        # Frame skipping

        if frame_skip > 1 and idx % frame_skip:

            continue


        t0 = time.time()


        # Process frame

        vis, m = engine.process(
            frame,
            detect_every=detect_every,
            show_boxes=show_boxes,
        )


        # Store metrics

        rows.append(
            {
                k: v
                for k, v in m.items()
                if k != "hotspot"
            }
        )


        risk_hist.append(
            (
                m["time_s"],
                m["risk"],
                m["peak_density"],
            )
        )


        # ───────────────── Banner + KPIs ──────────────────────────

        banner_ph.markdown(
            f'<div class="banner {banner_class(m["risk"])}">'
            f'{m["status"]}'
            f' &nbsp;·&nbsp; '
            f'{m["risk"] * 100:.0f}% risk'
            f'</div>',
            unsafe_allow_html=True,
        )


        m_count.metric(
            "People",
            m["count"],
        )


        m_area.metric(
            "Area",
            f"{m['area_m2']:.0f} m²",
        )


        m_avg.metric(
            "Avg density",
            f"{m['mean_density']:.2f} p/m²",
        )


        m_peak.metric(
            "Peak density",
            f"{m['peak_density']:.2f} p/m²",
            m["density_label"],
        )


        m_risk.metric(
            "Risk",
            f"{m['risk'] * 100:.1f}%",
        )


        # ───────────────── Video output ───────────────────────────

        video_rgb = cv2.cvtColor(
            vis,
            cv2.COLOR_BGR2RGB,
        )

        video_ph.image(
            video_rgb,
            width="stretch",
        )


        # ───────────────── Density minimap ────────────────────────

        mini = engine.grid.minimap(
            engine.density,
            size=260,
        )

        mini_rgb = cv2.cvtColor(
            mini,
            cv2.COLOR_BGR2RGB,
        )

        map_ph.image(
            mini_rgb,
            width="stretch",
        )


        # ───────────────── Alerts ─────────────────────────────────

        if m["alarm"]:

            stamp = (
                f"t={m['time_s']:.1f}s · "
                f"{m['count']} people · "
                f"{m['peak_density']:.1f} p/m²"
            )

            if not alerts or alerts[-1] != stamp:

                alerts.append(stamp)


            log_ph.error(
                "**ALERTS**\n\n"
                + "\n\n".join(
                    f"🚨 {a}"
                    for a in alerts
                )
            )


        # ───────────────── Risk chart ─────────────────────────────

        if len(risk_hist) % 5 == 0:

            hd = pd.DataFrame(
                risk_hist,
                columns=[
                    "time_s",
                    "risk",
                    "peak_density",
                ],
            )

            chart_ph.line_chart(
                hd.set_index("time_s")
            )


        # ───────────────── Real-time throttling ───────────────────

        if delay:

            time.sleep(
                max(
                    0.0,
                    delay - (time.time() - t0),
                )
            )


        # Stop after image

        if source.is_image:

            break


    # ───────────────────── Cleanup ────────────────────────────────

    source.release()

    st.session_state.running = False


    # ───────────────────── Save history ───────────────────────────

    if rows:
        st.session_state.history = pd.DataFrame(rows)
        st.session_state.summary = build_session_summary(
            st.session_state.history
        )

    st.success(
        "Stream finished. Scroll down to view the session analytics "
        "and export the report."
    )
