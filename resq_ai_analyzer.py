#!/usr/bin/env python3
"""
ResQ-AI — Unified Crowd Risk Analyzer (core engine + CLI)
=========================================================
Per frame it produces:
    1. Person detection + count            (Faster R-CNN ResNet-50 FPN)
    2. Ground-plane area in square metres  (homography calibration)
    3. A metric density heatmap            (people / m^2, bird's-eye grid)
    4. Motion turbulence                   (Farneback optical flow)
    5. A stampede risk verdict             (density + turbulence fusion)

This module is importable — app.py uses CrowdAnalyzer directly.

USAGE
-----
  # calibrate once per camera (opens a click window; falls back to matplotlib)
  python resq_ai_analyzer.py --input crowd.mp4 --calibrate

  # headless calibration (no GUI at all)
  python resq_ai_analyzer.py --input crowd.mp4 --calibrate \
      --calib-points "180,610 980,610 840,330 320,330" --calib-size 12x8

  # analyse
  python resq_ai_analyzer.py --input crowd.mp4 --output result.mp4
  python resq_ai_analyzer.py --input image.png  --output result.png
"""

import argparse
import csv
import json
import math
import os
import sys
from collections import deque

import cv2
import numpy as np
import torch
from torchvision.models.detection import (
    FasterRCNN_ResNet50_FPN_Weights,
    fasterrcnn_resnet50_fpn,
)

# ══════════════════════════════ CONFIG ══════════════════════════════
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

CONF_THRESHOLD      = 0.50
USE_GRAYSCALE_INPUT = False
PERSON_HEIGHT_M     = 1.65

CELL_SIZE_M         = 0.20
SIGMA_M             = 0.60
DISPLAY_MAX_DENSITY = 6.0

D_COMFORTABLE       = 2.0
D_RESTRICTED        = 3.5
D_DANGEROUS         = 4.5
D_CRITICAL          = 5.5

RISK_WARN           = 0.35
RISK_DANGER         = 0.60
RISK_CRITICAL       = 0.80
ALARM_SUSTAIN_SEC   = 2.0

IMAGE_EXTS = (".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff")
# ════════════════════════════════════════════════════════════════════


# ───────────────────────── 0. FRAME HYGIENE ─────────────────────────
def to_bgr3(img):
    """
    Force any frame into 3-channel BGR uint8.

    PNGs frequently carry an alpha channel (H,W,4) and screenshots are
    sometimes grayscale (H,W). Everything downstream — blending, warping,
    the detector — assumes 3 channels, so we normalise once here.
    """
    if img is None:
        return None
    if img.dtype != np.uint8:
        img = np.clip(img, 0, 255).astype(np.uint8)
    if img.ndim == 2:
        return cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    if img.shape[2] == 4:
        return cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)
    if img.shape[2] == 1:
        return cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    return img


def gui_available():
    """True only if this OpenCV build has highgui (opencv-python, not headless)."""
    try:
        cv2.namedWindow("__resq_probe__", cv2.WINDOW_NORMAL)
        cv2.destroyWindow("__resq_probe__")
        return True
    except Exception:
        return False


class FrameSource:
    """Uniform reader for images, video files, webcams and RTSP streams."""

    def __init__(self, path):
        self.path = path
        self.is_image = isinstance(path, str) and path.lower().endswith(IMAGE_EXTS)
        self.cap = None

        if self.is_image:
            img = cv2.imread(path, cv2.IMREAD_UNCHANGED)   # never VideoCapture for stills
            if img is None:
                raise IOError(f"Cannot read image: {path}")
            self.image = to_bgr3(img)
            self.fps = 1.0
            self._served = False
        else:
            src = int(path) if isinstance(path, str) and path.isdigit() else path
            self.cap = cv2.VideoCapture(src)
            if not self.cap.isOpened():
                raise IOError(f"Cannot open video source: {path}")
            fps = self.cap.get(cv2.CAP_PROP_FPS)
            self.fps = fps if fps and fps > 1 else 25.0

    def first_frame(self):
        if self.is_image:
            return self.image.copy()
        ok, f = self.cap.read()
        if not ok:
            raise IOError("Empty video source.")
        self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
        return to_bgr3(f)

    def read(self):
        if self.is_image:
            if self._served:
                return None
            self._served = True
            return self.image.copy()
        ok, f = self.cap.read()
        return to_bgr3(f) if ok else None

    def release(self):
        if self.cap is not None:
            self.cap.release()


# ─────────────────────────── 1. DETECTION ───────────────────────────
class PersonDetector:
    """Faster R-CNN ResNet-50 FPN restricted to the COCO 'person' class (id 1)."""

    def __init__(self, device=DEVICE, conf=CONF_THRESHOLD, grayscale=USE_GRAYSCALE_INPUT):
        print(f"[INFO] Loading Faster R-CNN ResNet-50 FPN on {device} ...")
        weights = FasterRCNN_ResNet50_FPN_Weights.DEFAULT
        self.model = fasterrcnn_resnet50_fpn(weights=weights, box_score_thresh=conf)
        self.model.to(device).eval()
        self.device = device
        self.conf = conf
        self.grayscale = grayscale

    def detect(self, frame_bgr):
        img = to_bgr3(frame_bgr)
        if self.grayscale:
            g = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            img = cv2.merge([g, g, g])

        rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        tensor = torch.from_numpy(rgb).permute(2, 0, 1).float().div(255.0).to(self.device)

        with torch.no_grad():
            pred = self.model([tensor])[0]

        keep = (pred["labels"] == 1) & (pred["scores"] >= self.conf)
        return pred["boxes"][keep].cpu().numpy(), pred["scores"][keep].cpu().numpy()

    @staticmethod
    def foot_points(boxes):
        if len(boxes) == 0:
            return np.zeros((0, 2), np.float32)
        x = (boxes[:, 0] + boxes[:, 2]) * 0.5
        y = boxes[:, 3]
        return np.stack([x, y], axis=1).astype(np.float32)


# ────────────────────── 2. GROUND PLANE / AREA ──────────────────────
def polygon_area(pts):
    x, y = pts[:, 0], pts[:, 1]
    return 0.5 * abs(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1)))


class GroundPlane:
    def __init__(self, H_img2world, roi_img_pts):
        self.H = np.asarray(H_img2world, np.float64)
        self.Hinv = np.linalg.inv(self.H)
        self.roi_img = np.asarray(roi_img_pts, np.float32)
        self.roi_world = self.to_world(self.roi_img)
        self.area_m2 = float(polygon_area(self.roi_world))

    @classmethod
    def from_calibration(cls, calib):
        img_pts = np.asarray(calib["image_points"], np.float32)
        world_pts = np.asarray(calib["world_points"], np.float32)
        return cls(cv2.getPerspectiveTransform(img_pts, world_pts), img_pts)

    @classmethod
    def auto_from_height(cls, px_per_m, frame_shape, margin=0.02):
        h, w = frame_shape[:2]
        s = 1.0 / float(px_per_m)
        H = np.array([[s, 0, 0], [0, s, 0], [0, 0, 1]], np.float64)
        mx, my = w * margin, h * margin
        roi = np.array([[mx, my], [w - mx, my], [w - mx, h - my], [mx, h - my]], np.float32)
        return cls(H, roi)

    def to_world(self, pts_img):
        pts = np.asarray(pts_img, np.float32).reshape(-1, 1, 2)
        if len(pts) == 0:
            return np.zeros((0, 2), np.float32)
        return cv2.perspectiveTransform(pts, self.H).reshape(-1, 2)

    def to_image(self, pts_world):
        pts = np.asarray(pts_world, np.float32).reshape(-1, 1, 2)
        if len(pts) == 0:
            return np.zeros((0, 2), np.float32)
        return cv2.perspectiveTransform(pts, self.Hinv).reshape(-1, 2)

    def inside_roi(self, pts_img):
        poly = self.roi_img.astype(np.int32)
        return np.array(
            [cv2.pointPolygonTest(poly, (float(p[0]), float(p[1])), False) >= 0 for p in pts_img],
            dtype=bool,
        )

    def px_per_m_at(self, img_pt):
        w0 = self.to_world([img_pt])[0]
        probes = self.to_image([w0 + [1.0, 0.0], w0 + [0.0, 1.0]])
        d1 = np.linalg.norm(probes[0] - img_pt)
        d2 = np.linalg.norm(probes[1] - img_pt)
        return max((d1 + d2) * 0.5, 1e-6)


# ───────────────────── 3. METRIC DENSITY HEATMAP ────────────────────
class DensityGrid:
    def __init__(self, ground, cell=CELL_SIZE_M, sigma_m=SIGMA_M, max_cells=1200):
        wp = ground.roi_world
        self.xmin, self.ymin = wp.min(axis=0)
        self.xmax, self.ymax = wp.max(axis=0)

        span_x = max(self.xmax - self.xmin, 1e-3)
        span_y = max(self.ymax - self.ymin, 1e-3)

        self.cell = max(cell, span_x / max_cells, span_y / max_cells)
        self.W = max(int(math.ceil(span_x / self.cell)), 2)
        self.H = max(int(math.ceil(span_y / self.cell)), 2)

        self.sigma_cells = max(sigma_m / self.cell, 0.8)
        k = int(self.sigma_cells * 6) | 1
        self.ksize = (k, k)

        M_grid2world = np.array(
            [[self.cell, 0, self.xmin], [0, self.cell, self.ymin], [0, 0, 1]], np.float64
        )
        self.M_grid2img = ground.Hinv @ M_grid2world

        roi_grid = ((wp - [self.xmin, self.ymin]) / self.cell).astype(np.int32)
        self.mask = np.zeros((self.H, self.W), np.uint8)
        cv2.fillPoly(self.mask, [roi_grid], 255)
        self.maskf = (self.mask > 0).astype(np.float32)

        print(f"[INFO] Density grid {self.W}x{self.H} cells @ {self.cell:.2f} m/cell")

    def compute(self, world_pts):
        acc = np.zeros((self.H, self.W), np.float32)
        for wx, wy in world_pts:
            gx = int((wx - self.xmin) / self.cell)
            gy = int((wy - self.ymin) / self.cell)
            if 0 <= gx < self.W and 0 <= gy < self.H:
                acc[gy, gx] += 1.0
        blurred = cv2.GaussianBlur(acc, self.ksize, self.sigma_cells)
        return (blurred / (self.cell ** 2)) * self.maskf

    def stats(self, density):
        vals = density[self.mask > 0]
        if vals.size == 0:
            return 0.0, 0.0
        return float(vals.max()), float(np.percentile(vals, 99))

    def hotspot_image_point(self, density):
        idx = int(np.argmax(density))
        gy, gx = divmod(idx, self.W)
        pt = np.array([[gx * self.cell + self.xmin, gy * self.cell + self.ymin]], np.float32)
        p = self._world_to_img(pt)[0]
        return int(round(p[0])), int(round(p[1]))

    def _world_to_img(self, pts_world):
        g = (np.asarray(pts_world, np.float32) - [self.xmin, self.ymin]) / self.cell
        g = np.hstack([g, np.ones((len(g), 1), np.float32)])
        p = (self.M_grid2img @ g.T).T
        return p[:, :2] / p[:, 2:3]

    def overlay(self, frame, density, alpha_max=0.70):
        frame = to_bgr3(frame).astype(np.float32)
        h, w = frame.shape[:2]

        norm = np.clip(density / DISPLAY_MAX_DENSITY, 0, 1)
        norm8 = (norm * 255).astype(np.uint8)

        colour = cv2.applyColorMap(norm8, cv2.COLORMAP_JET).astype(np.float32)
        alpha = (norm * alpha_max).astype(np.float32)

        colour_img = cv2.warpPerspective(colour, self.M_grid2img, (w, h))
        alpha_img = cv2.warpPerspective(alpha, self.M_grid2img, (w, h))[..., None]

        return np.clip(frame * (1 - alpha_img) + colour_img * alpha_img, 0, 255).astype(np.uint8)

    def minimap(self, density, size=200):
        norm = np.clip(density / DISPLAY_MAX_DENSITY, 0, 1)
        img = cv2.applyColorMap((norm * 255).astype(np.uint8), cv2.COLORMAP_JET)
        img[self.mask == 0] = (25, 25, 25)
        scale = size / max(self.W, self.H)
        return cv2.resize(img, (max(int(self.W * scale), 1), max(int(self.H * scale), 1)))


# ──────────────────────── 4. MOTION / TURBULENCE ────────────────────
class MotionAnalyzer:
    def __init__(self, ground, frame_shape, scale=0.5):
        self.scale = scale
        self.prev = None
        h, w = frame_shape[:2]
        poly = (ground.roi_img * scale).astype(np.int32)
        self.mask = np.zeros((int(h * scale), int(w * scale)), np.uint8)
        cv2.fillPoly(self.mask, [poly], 255)
        self.px_per_m = ground.px_per_m_at(ground.roi_img.mean(axis=0)) * scale

    def update(self, frame_bgr, fps):
        small = cv2.resize(to_bgr3(frame_bgr), None, fx=self.scale, fy=self.scale)
        gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)

        if self.prev is None or self.prev.shape != gray.shape:
            self.prev = gray
            return 0.0, 0.0

        flow = cv2.calcOpticalFlowFarneback(self.prev, gray, None, 0.5, 3, 15, 3, 5, 1.2, 0)
        self.prev = gray

        fx, fy = flow[..., 0], flow[..., 1]
        mag = np.sqrt(fx * fx + fy * fy)
        sel = (self.mask > 0) & (mag > 0.25)
        if sel.sum() < 50:
            return 0.0, 0.0

        m = mag[sel]
        speed_ms = float(m.mean()) / self.px_per_m * fps
        ux, uy = fx[sel] / m, fy[sel] / m
        wts = m / m.sum()
        R = math.hypot(float((ux * wts).sum()), float((uy * wts).sum()))
        return speed_ms, float(np.clip(1.0 - R, 0.0, 1.0))


# ────────────────────────── 5. RISK ENGINE ──────────────────────────
class RiskEngine:
    LEVELS = [
        (RISK_CRITICAL, "CRITICAL: STAMPEDE RISK",  (0, 0, 255)),
        (RISK_DANGER,   "DANGER: CRUSH CONDITIONS", (0, 80, 255)),
        (RISK_WARN,     "WARNING: HIGH DENSITY",    (0, 165, 255)),
        (-1.0,          "NORMAL FLOW",              (0, 200, 0)),
    ]

    def __init__(self, fps, ema=0.85):
        self.ema = ema
        self.score = 0.0
        self.fps = max(fps, 1.0)
        self.critical_frames = 0
        self.count_hist = deque(maxlen=int(self.fps * 3) + 1)

    def update(self, peak_density, mean_density, speed_ms, disorder, count):
        d_term = float(np.clip((peak_density - D_COMFORTABLE) / (D_CRITICAL - D_COMFORTABLE), 0, 1))
        gate = float(np.clip(peak_density / D_RESTRICTED, 0, 1))
        m_term = float(np.clip(0.6 * disorder + 0.4 * min(speed_ms / 1.8, 1.0), 0, 1)) * gate

        self.count_hist.append(count)
        surge = 0.0
        if len(self.count_hist) == self.count_hist.maxlen:
            old = max(self.count_hist[0], 1)
            surge = float(np.clip((count - old) / old, 0, 1)) * gate

        raw = float(np.clip(0.62 * d_term + 0.26 * m_term + 0.12 * surge, 0, 1))
        self.score = self.ema * self.score + (1 - self.ema) * raw

        self.critical_frames = self.critical_frames + 1 if self.score >= RISK_CRITICAL else 0
        alarm = self.critical_frames >= int(ALARM_SUSTAIN_SEC * self.fps)
        return self.score, alarm

    @staticmethod
    def label(score):
        for thr, text, colour in RiskEngine.LEVELS:
            if score > thr:
                return text, colour
        return "NORMAL FLOW", (0, 200, 0)

    @staticmethod
    def density_label(d):
        if d >= D_CRITICAL:
            return "CRITICAL"
        if d >= D_DANGEROUS:
            return "DANGEROUS"
        if d >= D_RESTRICTED:
            return "RESTRICTED"
        if d >= D_COMFORTABLE:
            return "BUSY"
        return "FREE FLOW"


def on_alert(metrics):
    """Hook for the app layer: push notification, SMS, siren, dashboard event."""
    print(f"[ALERT] Sustained stampede risk  |  {json.dumps(metrics)}")


# ────────────────────────────── 6. UI ───────────────────────────────
FONT = cv2.FONT_HERSHEY_SIMPLEX


def draw_hud(frame, ground, boxes, metrics, minimap=None, show_boxes=True):
    frame = to_bgr3(frame)
    h, w = frame.shape[:2]
    score = metrics["risk"]
    status, colour = RiskEngine.label(score)

    if show_boxes:
        for x1, y1, x2, y2 in np.asarray(boxes).astype(int):
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 1)

    cv2.polylines(frame, [ground.roi_img.astype(np.int32)], True, (255, 255, 0), 2)

    if metrics.get("hotspot") and score > RISK_WARN:
        hx, hy = metrics["hotspot"]
        cv2.circle(frame, (hx, hy), 26, colour, 3)
        cv2.putText(frame, "HOTSPOT", (hx - 34, hy - 34), FONT, 0.5, colour, 1, cv2.LINE_AA)

    panel = frame.copy()
    cv2.rectangle(panel, (0, 0), (w, 160), (0, 0, 0), -1)
    cv2.addWeighted(panel, 0.72, frame, 0.28, 0, frame)

    cv2.putText(frame, "ResQ-AI  |  REAL-TIME CROWD RISK ANALYTICS",
                (18, 28), FONT, 0.6, (255, 255, 255), 1, cv2.LINE_AA)
    cv2.putText(frame, status, (18, 70), cv2.FONT_HERSHEY_DUPLEX, 1.0, colour, 2, cv2.LINE_AA)

    cv2.putText(frame, (f"PEOPLE {metrics['count']:>3d}   AREA {metrics['area_m2']:.1f} m2   "
                        f"AVG {metrics['mean_density']:.2f} p/m2   "
                        f"PEAK {metrics['peak_density']:.2f} p/m2 [{metrics['density_label']}]"),
                (18, 96), FONT, 0.52, (230, 230, 230), 1, cv2.LINE_AA)
    cv2.putText(frame, f"FLOW {metrics['speed_ms']:.2f} m/s   DISORDER {metrics['disorder']:.2f}",
                (18, 118), FONT, 0.52, (180, 180, 180), 1, cv2.LINE_AA)

    bx, by, bw, bh = 18, 130, 380, 16
    cv2.rectangle(frame, (bx, by), (bx + bw, by + bh), (55, 55, 55), -1)
    cv2.rectangle(frame, (bx, by), (bx + int(bw * score), by + bh), colour, -1)
    cv2.putText(frame, f"RISK {score * 100:5.1f}%", (bx + bw + 14, by + 14),
                FONT, 0.6, colour, 2, cv2.LINE_AA)

    if minimap is not None:
        mh, mw = minimap.shape[:2]
        x0, y0 = w - mw - 14, h - mh - 14
        if x0 > 0 and y0 > 20:
            cv2.rectangle(frame, (x0 - 4, y0 - 22), (x0 + mw + 4, y0 + mh + 4), (0, 0, 0), -1)
            frame[y0:y0 + mh, x0:x0 + mw] = minimap
            cv2.putText(frame, "BIRD'S-EYE DENSITY", (x0 - 2, y0 - 7),
                        FONT, 0.42, (255, 255, 255), 1, cv2.LINE_AA)

    if metrics.get("alarm"):
        cv2.rectangle(frame, (0, 0), (w - 1, h - 1), (0, 0, 255), 8)

    return frame


# ──────────────────── 7. ANALYZER (used by CLI + app) ───────────────
class CrowdAnalyzer:
    """Stateful pipeline. Feed it frames, get back an annotated frame + metrics."""

    def __init__(self, first_frame, fps=25.0, calib=None, px_per_m=None,
                 detector=None, conf=CONF_THRESHOLD):
        first_frame = to_bgr3(first_frame)
        self.detector = detector or PersonDetector(DEVICE, conf=conf)
        self.fps = max(fps, 1.0)

        if calib is not None:
            self.ground = GroundPlane.from_calibration(calib)
            self.calibrated = True
            print(f"[INFO] Calibrated ROI area: {self.ground.area_m2:.1f} m2")
        else:
            ppm = px_per_m or estimate_px_per_m(self.detector.detect(first_frame)[0])
            if ppm is None:
                raise ValueError("No people detected for auto-scale — calibrate instead.")
            self.ground = GroundPlane.auto_from_height(ppm, first_frame.shape)
            self.calibrated = False
            print(f"[WARN] Uncalibrated: {ppm:.1f} px/m from person height; "
                  f"perspective NOT corrected, area {self.ground.area_m2:.1f} m2 approximate.")

        self.grid = DensityGrid(self.ground)
        self.motion = MotionAnalyzer(self.ground, first_frame.shape)
        self.risk = RiskEngine(self.fps)
        self.frame_i = 0
        self.boxes = np.zeros((0, 4), np.float32)
        self.density = np.zeros((self.grid.H, self.grid.W), np.float32)

    def process(self, frame, detect_every=1, annotate=True, show_boxes=True):
        frame = to_bgr3(frame)

        if self.frame_i % max(detect_every, 1) == 0:
            self.boxes, _ = self.detector.detect(frame)

        feet = PersonDetector.foot_points(self.boxes)
        inside = self.ground.inside_roi(feet) if len(feet) else np.zeros(0, bool)
        count = int(inside.sum())

        self.density = self.grid.compute(self.ground.to_world(feet[inside]))
        peak, _ = self.grid.stats(self.density)
        mean_density = count / self.ground.area_m2 if self.ground.area_m2 > 0 else 0.0

        speed_ms, disorder = self.motion.update(frame, self.fps)
        score, alarm = self.risk.update(peak, mean_density, speed_ms, disorder, count)

        metrics = {
            "frame": self.frame_i,
            "time_s": round(self.frame_i / self.fps, 2),
            "count": count,
            "area_m2": round(self.ground.area_m2, 2),
            "mean_density": round(mean_density, 3),
            "peak_density": round(peak, 3),
            "density_label": RiskEngine.density_label(peak),
            "speed_ms": round(speed_ms, 3),
            "disorder": round(disorder, 3),
            "risk": round(float(score), 4),
            "status": RiskEngine.label(score)[0],
            "alarm": bool(alarm),
            "hotspot": self.grid.hotspot_image_point(self.density) if count else None,
        }

        vis = frame
        if annotate:
            vis = self.grid.overlay(frame, self.density)
            vis = draw_hud(vis, self.ground, self.boxes, metrics,
                           self.grid.minimap(self.density), show_boxes=show_boxes)

        self.frame_i += 1
        return vis, metrics


# ────────────────────────── 8. CALIBRATION ──────────────────────────
def _calib_click_cv2(frame):
    pts, win = [], "ResQ-AI Calibration"
    order = ["NEAR-LEFT", "NEAR-RIGHT", "FAR-RIGHT", "FAR-LEFT"]

    def on_mouse(event, x, y, flags, _):
        if event == cv2.EVENT_LBUTTONDOWN and len(pts) < 4:
            pts.append([float(x), float(y)])

    cv2.namedWindow(win, cv2.WINDOW_NORMAL)
    cv2.setMouseCallback(win, on_mouse)
    while True:
        vis = frame.copy()
        for i, p in enumerate(pts):
            cv2.circle(vis, (int(p[0]), int(p[1])), 6, (0, 255, 255), -1)
            cv2.putText(vis, order[i], (int(p[0]) + 8, int(p[1]) - 8), FONT, 0.5, (0, 255, 255), 1)
        if len(pts) == 4:
            cv2.polylines(vis, [np.array(pts, np.int32)], True, (0, 255, 255), 2)
        nxt = order[len(pts)] if len(pts) < 4 else "press ENTER"
        cv2.putText(vis, f"Click: {nxt}", (14, 28), FONT, 0.7, (0, 255, 255), 2)
        cv2.imshow(win, vis)
        k = cv2.waitKey(20) & 0xFF
        if k == 27:
            cv2.destroyAllWindows()
            return None
        if k == ord("r"):
            pts.clear()
        if k in (13, 10) and len(pts) == 4:
            break
    cv2.destroyAllWindows()
    return pts


def _calib_click_matplotlib(frame):
    """Fallback when OpenCV was built headless (opencv-python-headless)."""
    import matplotlib.pyplot as plt
    print("[CALIB] OpenCV GUI unavailable — using matplotlib. Click 4 points.")
    fig = plt.figure(figsize=(13, 7))
    plt.imshow(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    plt.title("Click: near-left -> near-right -> far-right -> far-left")
    plt.axis("off")
    pts = plt.ginput(4, timeout=0)
    plt.close(fig)
    return [[float(x), float(y)] for x, y in pts] if len(pts) == 4 else None


def run_calibration(frame, out_path, points=None, size=None):
    if points is None:
        points = _calib_click_cv2(frame) if gui_available() else _calib_click_matplotlib(frame)
    if not points or len(points) != 4:
        print("[CALIB] Aborted.")
        return None

    if size is None:
        width = float(input("Real width  (near edge, metres): "))
        length = float(input("Real length (depth,     metres): "))
    else:
        width, length = size

    calib = {
        "image_points": points,
        "world_points": [[0.0, 0.0], [width, 0.0], [width, length], [0.0, length]],
        "note": "world_points are metres on the ground plane",
    }
    with open(out_path, "w") as f:
        json.dump(calib, f, indent=2)
    print(f"[CALIB] Saved -> {out_path}  ({width * length:.1f} m2 reference)")
    return calib


def estimate_px_per_m(boxes):
    if len(boxes) == 0:
        return None
    heights = boxes[:, 3] - boxes[:, 1]
    heights = heights[heights > 5]
    return float(np.median(heights)) / PERSON_HEIGHT_M if heights.size else None


# ──────────────────────────── 9. CLI ────────────────────────────────
def analyze(args):
    source = FrameSource(args.input)
    first = source.first_frame()

    if args.calibrate:
        pts = None
        if args.calib_points:
            pts = [[float(v) for v in p.split(",")] for p in args.calib_points.split()]
        size = None
        if args.calib_size:
            w_, l_ = args.calib_size.lower().split("x")
            size = (float(w_), float(l_))
        run_calibration(first, args.calib_file, pts, size)
        source.release()
        return

    calib = None
    if not args.no_calib and os.path.exists(args.calib_file):
        with open(args.calib_file) as f:
            calib = json.load(f)

    engine = CrowdAnalyzer(first, source.fps, calib=calib)

    h, w = first.shape[:2]
    writer = None
    if not source.is_image:
        writer = cv2.VideoWriter(args.output, cv2.VideoWriter_fourcc(*"mp4v"), source.fps, (w, h))

    show = args.show and gui_available()
    if args.show and not show:
        print("[WARN] --show ignored: this OpenCV build has no GUI support.")

    log = open(args.csv, "w", newline="")
    logger = csv.writer(log)
    logger.writerow(["frame", "time_s", "count", "area_m2", "mean_density",
                     "peak_density", "speed_ms", "disorder", "risk", "alarm"])

    print("[INFO] Analyzing ...")
    while True:
        frame = source.read()
        if frame is None:
            break

        vis, m = engine.process(frame, show_boxes=not args.no_boxes)

        logger.writerow([m["frame"], m["time_s"], m["count"], m["area_m2"], m["mean_density"],
                         m["peak_density"], m["speed_ms"], m["disorder"], m["risk"], int(m["alarm"])])
        if m["alarm"] and m["frame"] % int(max(source.fps, 1)) == 0:
            on_alert({k: v for k, v in m.items() if k != "hotspot"})
        if args.stream_json:
            print(json.dumps({k: v for k, v in m.items() if k != "hotspot"}), flush=True)

        if writer is not None:
            writer.write(vis)
        if show:
            cv2.imshow("ResQ-AI", vis)
            if cv2.waitKey(1) & 0xFF == 27:
                break

        if source.is_image:
            out = args.output if args.output.lower().endswith(IMAGE_EXTS) else "resq_ai_output.png"
            cv2.imwrite(out, vis)
            print(f"\n--- RESULTS ---\nPeople : {m['count']}\nArea   : {m['area_m2']:.1f} m2\n"
                  f"Avg    : {m['mean_density']:.2f} p/m2\n"
                  f"Peak   : {m['peak_density']:.2f} p/m2 ({m['density_label']})\n"
                  f"Verdict: {m['status']}\nImage  -> {out}")
            break
        if m["frame"] % 30 == 0:
            print(f"  frame {m['frame']}  count={m['count']:3d}  "
                  f"peak={m['peak_density']:.2f} p/m2  risk={m['risk']:.2f}")

    source.release()
    if writer is not None:
        writer.release()
    log.close()
    if show:
        cv2.destroyAllWindows()
    print(f"[SUCCESS] Metrics -> {args.csv}")


def main():
    p = argparse.ArgumentParser(description="ResQ-AI unified crowd risk analyzer")
    p.add_argument("--input", required=True, help="image, video, webcam index or RTSP url")
    p.add_argument("--output", default="resq_ai_output.mp4")
    p.add_argument("--csv", default="resq_ai_metrics.csv")
    p.add_argument("--calib-file", dest="calib_file", default="calibration.json")
    p.add_argument("--calibrate", action="store_true")
    p.add_argument("--calib-points", default=None,
                   help='headless: "x1,y1 x2,y2 x3,y3 x4,y4"')
    p.add_argument("--calib-size", default=None, help='headless: "12x8" (width x length in m)')
    p.add_argument("--no-calib", action="store_true")
    p.add_argument("--no-boxes", action="store_true")
    p.add_argument("--show", action="store_true")
    p.add_argument("--stream-json", action="store_true")
    analyze(p.parse_args())


if __name__ == "__main__":
    main()