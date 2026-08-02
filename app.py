#!/usr/bin/env python3
"""
ResQ-AI — Live Crowd Risk Dashboard
===================================
A CCTV-style monitoring app. Point it at a video file, a webcam or an RTSP
stream and it analyses frame by frame as the footage plays: live people count,
live density heatmap, live stampede verdict, and a rolling risk timeline.

    pip install streamlit pandas
    streamlit run app.py

Keep app.py and resq_ai_analyzer.py in the same folder.
"""

import os
import tempfile
import time
from collections import deque

import cv2
import numpy as np
import pandas as pd
import streamlit as st

from resq_ai_analyzer import (
    D_COMFORTABLE, D_CRITICAL, D_DANGEROUS, D_RESTRICTED,
    DEVICE, RISK_CRITICAL, RISK_DANGER, RISK_WARN,
    CrowdAnalyzer, FrameSource, PersonDetector, to_bgr3,
)

st.set_page_config(page_title="ResQ-AI | Crowd Risk Monitor", page_icon="🚨", layout="wide")

# ─────────────────────────── styling ────────────────────────────────
st.markdown("""
<style>
  .block-container {padding-top: 2rem; padding-bottom: 1rem;}
  .banner {padding: 14px 20px; border-radius: 10px; font-size: 22px;
           font-weight: 700; letter-spacing: .5px; margin-bottom: 12px;}
  .b-ok   {background:#0d3b1e; color:#4ade80; border-left:8px solid #22c55e;}
  .b-warn {background:#3b2f0d; color:#fbbf24; border-left:8px solid #f59e0b;}
  .b-dang {background:#3b1d0d; color:#fb923c; border-left:8px solid #ea580c;}
  .b-crit {background:#3b0d0d; color:#f87171; border-left:8px solid #ef4444;
           animation: pulse 1s infinite;}
  @keyframes pulse {0%{opacity:1;} 50%{opacity:.55;} 100%{opacity:1;}}
</style>
""", unsafe_allow_html=True)


# ───────────────────────── model caching ────────────────────────────
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


# ──────────────────────────── sidebar ───────────────────────────────
st.sidebar.title("ResQ-AI Control")
st.sidebar.caption(f"Inference device: **{str(DEVICE).upper()}**")

src_kind = st.sidebar.radio("Video source", ["Upload file", "Local path", "Webcam / RTSP"])
source_path = None

if src_kind == "Upload file":
    up = st.sidebar.file_uploader("Video or image", type=["mp4", "avi", "mov", "mkv",
                                                          "png", "jpg", "jpeg"])
    if up is not None:
        suffix = os.path.splitext(up.name)[1]
        tmp = tempfile.NamedTemporaryFile(delete=False, suffix=suffix)
        tmp.write(up.read())
        tmp.close()
        source_path = tmp.name
elif src_kind == "Local path":
    source_path = st.sidebar.text_input("Path", "crowd.mp4") or None
else:
    source_path = st.sidebar.text_input("Webcam index or RTSP URL", "0") or None

st.sidebar.divider()
st.sidebar.subheader("Scale calibration")

calib_mode = st.sidebar.radio(
    "Mode",
    ["Auto (person height)", "calibration.json", "Enter 4 points"],
    help="Auto ignores perspective — fine for a quick demo, not for real metres.",
)

calib = None
if calib_mode == "calibration.json":
    cpath = st.sidebar.text_input("File", "calibration.json")
    if os.path.exists(cpath):
        import json
        calib = json.load(open(cpath))
        st.sidebar.success("Calibration loaded")
    else:
        st.sidebar.warning("File not found — falling back to auto")
elif calib_mode == "Enter 4 points":
    st.sidebar.caption("Pixel coords, in order: near-left, near-right, far-right, far-left")
    raw = st.sidebar.text_input("Points", "180,610 980,610 840,330 320,330")
    cw = st.sidebar.number_input("Real width (m)", 1.0, 500.0, 12.0)
    cl = st.sidebar.number_input("Real depth (m)", 1.0, 500.0, 8.0)
    try:
        pts = [[float(v) for v in p.split(",")] for p in raw.split()]
        if len(pts) == 4:
            calib = {"image_points": pts,
                     "world_points": [[0, 0], [cw, 0], [cw, cl], [0, cl]]}
            st.sidebar.success(f"Reference area {cw * cl:.1f} m²")
    except ValueError:
        st.sidebar.error("Could not parse points")

st.sidebar.divider()
st.sidebar.subheader("Performance")
conf = st.sidebar.slider("Detection confidence", 0.1, 0.9, 0.5, 0.05)
detect_every = st.sidebar.slider("Detect every N frames", 1, 8, 1,
                                 help="Raise this on CPU. Boxes persist between detections.")
frame_skip = st.sidebar.slider("Playback stride", 1, 10, 1,
                               help="Skip frames to keep up with a live stream.")
show_boxes = st.sidebar.checkbox("Show detection boxes", True)
realtime = st.sidebar.checkbox("Throttle to source FPS", False)

c1, c2 = st.sidebar.columns(2)
if c1.button("▶ Start", use_container_width=True, type="primary"):
    st.session_state.running = True
if c2.button("■ Stop", use_container_width=True):
    st.session_state.running = False

st.session_state.setdefault("running", False)
st.session_state.setdefault("history", None)


# ──────────────────────────── layout ────────────────────────────────
st.title("🚨 ResQ-AI — Real-Time Crowd Risk Monitor")

banner_ph = st.empty()
kpi_ph = st.container()
k1, k2, k3, k4, k5 = kpi_ph.columns(5)
m_count, m_area, m_avg, m_peak, m_risk = k1.empty(), k2.empty(), k3.empty(), k4.empty(), k5.empty()

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


def render_idle():
    banner_ph.markdown('<div class="banner b-ok">STANDBY — no stream</div>',
                       unsafe_allow_html=True)
    for ph, lbl in [(m_count, "People"), (m_area, "Area"), (m_avg, "Avg density"),
                    (m_peak, "Peak density"), (m_risk, "Risk")]:
        ph.metric(lbl, "—")


# ──────────────────────────── main loop ─────────────────────────────
if not st.session_state.running or not source_path:
    render_idle()
    if st.session_state.history is not None:
        df = st.session_state.history
        chart_ph.line_chart(df.set_index("time_s")[["risk", "peak_density"]])
        dl_ph.download_button("⬇ Download metrics CSV", df.to_csv(index=False),
                              "resq_ai_metrics.csv", "text/csv")
    else:
        st.info("Pick a source in the sidebar and press **Start**.")
else:
    try:
        source = FrameSource(source_path)
        first = source.first_frame()
    except IOError as e:
        st.error(str(e))
        st.session_state.running = False
        st.stop()

    detector = get_detector(conf)

    try:
        engine = CrowdAnalyzer(first, source.fps, calib=calib, detector=detector)
    except ValueError as e:
        st.error(f"{e}  Detected nobody in the first frame — lower the confidence "
                 f"threshold or supply a calibration.")
        st.session_state.running = False
        st.stop()

    if not engine.calibrated:
        st.warning("Running uncalibrated: scale is estimated from average person height "
                   "and perspective is not corrected. Area and density are approximate.",
                   icon="⚠️")

    rows, risk_hist = [], deque(maxlen=600)
    alerts = deque(maxlen=6)
    delay = 1.0 / source.fps if realtime else 0.0
    idx = 0

    while st.session_state.running:
        frame = source.read()
        if frame is None:
            break
        idx += 1
        if frame_skip > 1 and idx % frame_skip:
            continue

        t0 = time.time()
        vis, m = engine.process(frame, detect_every=detect_every, show_boxes=show_boxes)

        rows.append({k: v for k, v in m.items() if k != "hotspot"})
        risk_hist.append((m["time_s"], m["risk"], m["peak_density"]))

        # banner + KPIs
        banner_ph.markdown(
            f'<div class="banner {banner_class(m["risk"])}">{m["status"]} '
            f'&nbsp;·&nbsp; {m["risk"] * 100:.0f}% risk</div>',
            unsafe_allow_html=True,
        )
        m_count.metric("People", m["count"])
        m_area.metric("Area", f"{m['area_m2']:.0f} m²")
        m_avg.metric("Avg density", f"{m['mean_density']:.2f} p/m²")
        m_peak.metric("Peak density", f"{m['peak_density']:.2f} p/m²", m["density_label"])
        m_risk.metric("Risk", f"{m['risk'] * 100:.1f}%")

        video_ph.image(cv2.cvtColor(vis, cv2.COLOR_BGR2RGB), use_container_width=True)
        mini = engine.grid.minimap(engine.density, size=260)
        map_ph.image(cv2.cvtColor(mini, cv2.COLOR_BGR2RGB), use_container_width=True)

        if m["alarm"]:
            stamp = f"t={m['time_s']:.1f}s · {m['count']} people · {m['peak_density']:.1f} p/m²"
            if not alerts or alerts[-1] != stamp:
                alerts.append(stamp)
            log_ph.error("**ALERTS**\n\n" + "\n\n".join(f"🚨 {a}" for a in alerts))

        if len(risk_hist) % 5 == 0:
            hd = pd.DataFrame(risk_hist, columns=["time_s", "risk", "peak_density"])
            chart_ph.line_chart(hd.set_index("time_s"))

        if delay:
            time.sleep(max(0.0, delay - (time.time() - t0)))

        if source.is_image:
            break

    source.release()
    st.session_state.running = False
    if rows:
        st.session_state.history = pd.DataFrame(rows)
        dl_ph.download_button("⬇ Download metrics CSV",
                              st.session_state.history.to_csv(index=False),
                              "resq_ai_metrics.csv", "text/csv")
    st.success("Stream finished.")