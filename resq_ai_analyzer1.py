#!/usr/bin/env python3
"""
ResQ-AI — Unified Crowd Risk Analyzer
=====================================
One script that produces, per frame:

    1. Person detection + count            (Faster R-CNN ResNet-50 FPN)
    2. Ground-plane area in square metres  (homography calibration)
    3. A metric density heatmap            (people / m^2, bird's-eye grid)
    4. Motion turbulence                   (Farneback optical flow)
    5. A stampede risk verdict             (density + turbulence fusion)

Why a bird's-eye grid instead of blurring boxes in the image?
------------------------------------------------------------
A heatmap drawn directly on the camera image is perspectively wrong: people
far from the camera occupy fewer pixels, so an image-space Gaussian makes the
back of the crowd look sparse even when it is packed. We instead project each
person's foot point onto the ground plane (metres), accumulate density on a
metric grid, and warp that grid back onto the video for display. The number
you read off the bar is then a real people/m^2 figure that can be compared to
published crowd-safety thresholds.

USAGE
-----
  # Step 1 (once per camera): click the 4 corners of a known ground rectangle
  python resq_ai_analyzer.py --input crowd.mp4 --calibrate

  # Step 2: run the analysis
  python resq_ai_analyzer.py --input crowd.mp4 --output result.mp4

  # Works on a single image too
  python resq_ai_analyzer.py --input image1.png --output result.png

  # No calibration available? Falls back to auto-scale from person height
  python resq_ai_analyzer.py --input crowd.mp4 --no-calib
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

CONF_THRESHOLD      = 0.50   # detection confidence
USE_GRAYSCALE_INPUT = False  # your original preprocessing; see note in README section
PERSON_HEIGHT_M     = 1.65   # used only by the auto-calibration fallback

CELL_SIZE_M         = 0.20   # bird's-eye grid resolution (metres per cell)
SIGMA_M             = 0.60   # density kernel radius (metres) - ~ personal space
DISPLAY_MAX_DENSITY = 6.0    # p/m^2 that maps to full red on the heatmap

# Crowd-safety thresholds (people per square metre), after Fruin / Still
D_COMFORTABLE       = 2.0
D_RESTRICTED        = 3.5
D_DANGEROUS         = 4.5
D_CRITICAL          = 5.5

RISK_WARN           = 0.35
RISK_DANGER         = 0.60
RISK_CRITICAL       = 0.80
ALARM_SUSTAIN_SEC   = 2.0    # risk must stay critical this long before ALERT

DETECT_EVERY_N      = 1      # raise to 2-3 on CPU for speed
# ════════════════════════════════════════════════════════════════════


# ─────────────────────────── 1. DETECTION ───────────────────────────
class PersonDetector:
    """Faster R-CNN ResNet-50 FPN restricted to the COCO 'person' class (id 1)."""

    def __init__(self, device, conf=CONF_THRESHOLD, grayscale=USE_GRAYSCALE_INPUT):
        print(f"[INFO] Loading Faster R-CNN ResNet-50 FPN on {device} ...")
        weights = FasterRCNN_ResNet50_FPN_Weights.DEFAULT
        self.model = fasterrcnn_resnet50_fpn(weights=weights, box_score_thresh=conf)
        self.model.to(device).eval()
        self.device = device
        self.conf = conf
        self.grayscale = grayscale

    def detect(self, frame_bgr):
        img = frame_bgr
        if self.grayscale:
            g = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            img = cv2.merge([g, g, g])

        rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        tensor = torch.from_numpy(rgb).permute(2, 0, 1).float().div(255.0).to(self.device)

        with torch.no_grad():
            pred = self.model([tensor])[0]

        keep = (pred["labels"] == 1) & (pred["scores"] >= self.conf)
        boxes = pred["boxes"][keep].cpu().numpy()
        scores = pred["scores"][keep].cpu().numpy()
        return boxes, scores

    @staticmethod
    def foot_points(boxes):
        """Ground contact point of each person: bottom-centre of the box."""
        if len(boxes) == 0:
            return np.zeros((0, 2), np.float32)
        x = (boxes[:, 0] + boxes[:, 2]) * 0.5
        y = boxes[:, 3]
        return np.stack([x, y], axis=1).astype(np.float32)


# ────────────────────── 2. GROUND PLANE / AREA ──────────────────────
def polygon_area(pts):
    """Shoelace formula. pts: (N,2) in metres -> area in m^2."""
    x, y = pts[:, 0], pts[:, 1]
    return 0.5 * abs(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1)))


class GroundPlane:
    """
    Holds the image -> world (metres) homography and the region of interest.

    Two ways to build it:
      * from_calibration : a real 4-point homography (accurate, handles perspective)
      * auto_from_height : uniform scale estimated from average person height
                           (no perspective correction - use only as a fallback)
    """

    def __init__(self, H_img2world, roi_img_pts):
        self.H = np.asarray(H_img2world, np.float64)
        self.Hinv = np.linalg.inv(self.H)
        self.roi_img = np.asarray(roi_img_pts, np.float32)
        self.roi_world = self.to_world(self.roi_img)
        self.area_m2 = float(polygon_area(self.roi_world))

    # -- constructors ------------------------------------------------
    @classmethod
    def from_calibration(cls, calib):
        img_pts = np.asarray(calib["image_points"], np.float32)
        world_pts = np.asarray(calib["world_points"], np.float32)
        H = cv2.getPerspectiveTransform(img_pts, world_pts)
        return cls(H, img_pts)

    @classmethod
    def auto_from_height(cls, px_per_m, frame_shape, margin=0.02):
        h, w = frame_shape[:2]
        s = 1.0 / float(px_per_m)
        H = np.array([[s, 0, 0], [0, s, 0], [0, 0, 1]], np.float64)
        mx, my = w * margin, h * margin
        roi = np.array([[mx, my], [w - mx, my], [w - mx, h - my], [mx, h - my]], np.float32)
        return cls(H, roi)

    # -- transforms --------------------------------------------------
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
        """Boolean mask: which image points fall inside the measured region."""
        poly = self.roi_img.astype(np.int32)
        return np.array(
            [cv2.pointPolygonTest(poly, (float(p[0]), float(p[1])), False) >= 0 for p in pts_img],
            dtype=bool,
        )

    def px_per_m_at(self, img_pt):
        """Local scale, needed to turn optical-flow pixels into metres."""
        w0 = self.to_world([img_pt])[0]
        probes = self.to_image([w0 + [1.0, 0.0], w0 + [0.0, 1.0]])
        d1 = np.linalg.norm(probes[0] - img_pt)
        d2 = np.linalg.norm(probes[1] - img_pt)
        return max((d1 + d2) * 0.5, 1e-6)


# ───────────────────── 3. METRIC DENSITY HEATMAP ────────────────────
class DensityGrid:
    """Bird's-eye people/m^2 map, plus warping helpers for display."""

    def __init__(self, ground, cell=CELL_SIZE_M, sigma_m=SIGMA_M, max_cells=1200):
        wp = ground.roi_world
        self.xmin, self.ymin = wp.min(axis=0)
        self.xmax, self.ymax = wp.max(axis=0)

        span_x = max(self.xmax - self.xmin, 1e-3)
        span_y = max(self.ymax - self.ymin, 1e-3)

        # keep the grid a sane size no matter how big the scene is
        self.cell = max(cell, span_x / max_cells, span_y / max_cells)
        self.W = max(int(math.ceil(span_x / self.cell)), 2)
        self.H = max(int(math.ceil(span_y / self.cell)), 2)

        self.sigma_cells = max(sigma_m / self.cell, 0.8)
        k = int(self.sigma_cells * 6) | 1
        self.ksize = (k, k)

        # grid pixel -> world metres -> image pixels
        M_grid2world = np.array(
            [[self.cell, 0, self.xmin], [0, self.cell, self.ymin], [0, 0, 1]], np.float64
        )
        self.M_grid2img = ground.Hinv @ M_grid2world

        # ROI mask in grid space
        roi_grid = ((wp - [self.xmin, self.ymin]) / self.cell).astype(np.int32)
        self.mask = np.zeros((self.H, self.W), np.uint8)
        cv2.fillPoly(self.mask, [roi_grid], 255)
        self.maskf = (self.mask > 0).astype(np.float32)

        print(f"[INFO] Density grid {self.W}x{self.H} cells @ {self.cell:.2f} m/cell")

    def compute(self, world_pts):
        """Return a (H,W) float map in people per square metre."""
        acc = np.zeros((self.H, self.W), np.float32)
        for wx, wy in world_pts:
            gx = int((wx - self.xmin) / self.cell)
            gy = int((wy - self.ymin) / self.cell)
            if 0 <= gx < self.W and 0 <= gy < self.H:
                acc[gy, gx] += 1.0

        # Gaussian blur preserves total mass, so dividing by the cell area
        # converts "people per cell" into "people per m^2".
        blurred = cv2.GaussianBlur(acc, self.ksize, self.sigma_cells)
        density = blurred / (self.cell ** 2)
        return density * self.maskf

    def stats(self, density):
        vals = density[self.mask > 0]
        if vals.size == 0:
            return 0.0, 0.0
        return float(vals.max()), float(np.percentile(vals, 99))

    def hotspot_image_point(self, density):
        idx = int(np.argmax(density))
        gy, gx = divmod(idx, self.W)
        pt = np.array([[gx * self.cell + self.xmin, gy * self.cell + self.ymin]], np.float32)
        return tuple(np.round(self._world_to_img(pt)[0]).astype(int))

    def _world_to_img(self, pts_world):
        M = self.M_grid2img
        g = (np.asarray(pts_world, np.float32) - [self.xmin, self.ymin]) / self.cell
        g = np.hstack([g, np.ones((len(g), 1), np.float32)])
        p = (M @ g.T).T
        return p[:, :2] / p[:, 2:3]

    def overlay(self, frame, density, alpha_max=0.70):
        """Warp the metric heatmap back onto the camera image."""
        h, w = frame.shape[:2]
        norm = np.clip(density / DISPLAY_MAX_DENSITY, 0, 1)
        norm8 = (norm * 255).astype(np.uint8)

        colour = cv2.applyColorMap(norm8, cv2.COLORMAP_JET)
        alpha = (norm * alpha_max).astype(np.float32)

        colour_img = cv2.warpPerspective(colour, self.M_grid2img, (w, h))
        alpha_img = cv2.warpPerspective(alpha, self.M_grid2img, (w, h))[..., None]

        return (frame * (1 - alpha_img) + colour_img * alpha_img).astype(np.uint8)

    def minimap(self, density, size=200):
        norm = np.clip(density / DISPLAY_MAX_DENSITY, 0, 1)
        img = cv2.applyColorMap((norm * 255).astype(np.uint8), cv2.COLORMAP_JET)
        img[self.mask == 0] = (25, 25, 25)
        scale = size / max(self.W, self.H)
        return cv2.resize(img, (max(int(self.W * scale), 1), max(int(self.H * scale), 1)))


# ──────────────────────── 4. MOTION / TURBULENCE ────────────────────
class MotionAnalyzer:
    """
    Crowd crushes are not just dense - they are dense AND incoherent.
    We measure mean speed and directional disorder with dense optical flow.
    """

    def __init__(self, ground, frame_shape, scale=0.5):
        self.scale = scale
        self.prev = None
        h, w = frame_shape[:2]
        poly = (ground.roi_img * scale).astype(np.int32)
        self.mask = np.zeros((int(h * scale), int(w * scale)), np.uint8)
        cv2.fillPoly(self.mask, [poly], 255)
        centroid = ground.roi_img.mean(axis=0)
        self.px_per_m = ground.px_per_m_at(centroid) * scale

    def update(self, frame_bgr, fps):
        small = cv2.resize(frame_bgr, None, fx=self.scale, fy=self.scale)
        gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)

        if self.prev is None:
            self.prev = gray
            return 0.0, 0.0

        flow = cv2.calcOpticalFlowFarneback(
            self.prev, gray, None, 0.5, 3, 15, 3, 5, 1.2, 0
        )
        self.prev = gray

        fx, fy = flow[..., 0], flow[..., 1]
        mag = np.sqrt(fx * fx + fy * fy)
        sel = (self.mask > 0) & (mag > 0.25)          # ignore static background
        if sel.sum() < 50:
            return 0.0, 0.0

        m = mag[sel]
        speed_ms = float(m.mean()) / self.px_per_m * fps

        # circular coherence of motion direction, magnitude-weighted
        ux, uy = fx[sel] / m, fy[sel] / m
        wts = m / m.sum()
        R = math.hypot(float((ux * wts).sum()), float((uy * wts).sum()))
        disorder = float(np.clip(1.0 - R, 0.0, 1.0))

        return speed_ms, disorder


# ────────────────────────── 5. RISK ENGINE ──────────────────────────
class RiskEngine:
    """Fuses density and motion into a single 0-1 stampede risk score."""

    LEVELS = [
        (RISK_CRITICAL, "CRITICAL: STAMPEDE RISK", (0, 0, 255)),
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
        # Density is the dominant term: below ~2 p/m^2 a crush is not physically
        # possible, above ~5.5 p/m^2 people lose the ability to control movement.
        d_term = np.clip((peak_density - D_COMFORTABLE) / (D_CRITICAL - D_COMFORTABLE), 0, 1)

        # Motion only matters once the crowd is packed, hence the gate.
        gate = float(np.clip(peak_density / D_RESTRICTED, 0, 1))
        m_term = float(np.clip(0.6 * disorder + 0.4 * min(speed_ms / 1.8, 1.0), 0, 1)) * gate

        # Sudden influx of people is an independent early warning.
        self.count_hist.append(count)
        surge = 0.0
        if len(self.count_hist) == self.count_hist.maxlen:
            old = max(self.count_hist[0], 1)
            surge = float(np.clip((count - old) / old, 0, 1)) * gate

        raw = float(np.clip(0.62 * d_term + 0.26 * m_term + 0.12 * surge, 0, 1))
        self.score = self.ema * self.score + (1 - self.ema) * raw

        if self.score >= RISK_CRITICAL:
            self.critical_frames += 1
        else:
            self.critical_frames = 0

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


def draw_hud(frame, ground, boxes, metrics, minimap, show_boxes=True):
    h, w = frame.shape[:2]
    score = metrics["risk"]
    status, colour = RiskEngine.label(score)

    if show_boxes:
        for x1, y1, x2, y2 in boxes.astype(int):
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 1)

    cv2.polylines(frame, [ground.roi_img.astype(np.int32)], True, (255, 255, 0), 2)

    if metrics["hotspot"] is not None and score > RISK_WARN:
        cv2.circle(frame, metrics["hotspot"], 26, colour, 3)
        cv2.putText(frame, "HOTSPOT", (metrics["hotspot"][0] - 34, metrics["hotspot"][1] - 34),
                    FONT, 0.5, colour, 1, cv2.LINE_AA)

    panel = frame.copy()
    cv2.rectangle(panel, (0, 0), (w, 160), (0, 0, 0), -1)
    cv2.addWeighted(panel, 0.72, frame, 0.28, 0, frame)

    cv2.putText(frame, "ResQ-AI  |  REAL-TIME CROWD RISK ANALYTICS",
                (18, 28), FONT, 0.6, (255, 255, 255), 1, cv2.LINE_AA)
    cv2.putText(frame, status, (18, 70), cv2.FONT_HERSHEY_DUPLEX, 1.0, colour, 2, cv2.LINE_AA)

    line = (f"PEOPLE {metrics['count']:>3d}   "
            f"AREA {metrics['area_m2']:.1f} m2   "
            f"AVG {metrics['mean_density']:.2f} p/m2   "
            f"PEAK {metrics['peak_density']:.2f} p/m2 [{metrics['density_label']}]")
    cv2.putText(frame, line, (18, 96), FONT, 0.52, (230, 230, 230), 1, cv2.LINE_AA)

    line2 = f"FLOW {metrics['speed_ms']:.2f} m/s   DISORDER {metrics['disorder']:.2f}"
    cv2.putText(frame, line2, (18, 118), FONT, 0.52, (180, 180, 180), 1, cv2.LINE_AA)

    bx, by, bw, bh = 18, 130, 380, 16
    cv2.rectangle(frame, (bx, by), (bx + bw, by + bh), (55, 55, 55), -1)
    cv2.rectangle(frame, (bx, by), (bx + int(bw * score), by + bh), colour, -1)
    cv2.putText(frame, f"RISK {score * 100:5.1f}%", (bx + bw + 14, by + 14),
                FONT, 0.6, colour, 2, cv2.LINE_AA)

    if minimap is not None:
        mh, mw = minimap.shape[:2]
        x0, y0 = w - mw - 14, h - mh - 14
        cv2.rectangle(frame, (x0 - 4, y0 - 22), (x0 + mw + 4, y0 + mh + 4), (0, 0, 0), -1)
        frame[y0:y0 + mh, x0:x0 + mw] = minimap
        cv2.putText(frame, "BIRD'S-EYE DENSITY", (x0 - 2, y0 - 7),
                    FONT, 0.42, (255, 255, 255), 1, cv2.LINE_AA)

    if metrics["alarm"]:
        cv2.rectangle(frame, (0, 0), (w - 1, h - 1), (0, 0, 255), 8)

    return frame


# ────────────────────────── 7. CALIBRATION ──────────────────────────
def run_calibration(frame, out_path):
    """Click 4 corners of a known ground rectangle, then type its dimensions."""
    pts = []
    win = "ResQ-AI Calibration"

    def on_mouse(event, x, y, flags, _):
        if event == cv2.EVENT_LBUTTONDOWN and len(pts) < 4:
            pts.append([float(x), float(y)])

    cv2.namedWindow(win, cv2.WINDOW_NORMAL)
    cv2.setMouseCallback(win, on_mouse)

    order = ["NEAR-LEFT", "NEAR-RIGHT", "FAR-RIGHT", "FAR-LEFT"]
    print("\n[CALIB] Click the 4 corners of a ground rectangle you can measure")
    print("[CALIB] Order: near-left -> near-right -> far-right -> far-left")
    print("[CALIB] ENTER = confirm, r = reset, ESC = abort\n")

    while True:
        vis = frame.copy()
        for i, p in enumerate(pts):
            cv2.circle(vis, (int(p[0]), int(p[1])), 6, (0, 255, 255), -1)
            cv2.putText(vis, order[i], (int(p[0]) + 8, int(p[1]) - 8),
                        FONT, 0.5, (0, 255, 255), 1)
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
    width = float(input("Real width  (near edge, metres): "))
    length = float(input("Real length (depth,     metres): "))

    calib = {
        "image_points": pts,
        "world_points": [[0.0, 0.0], [width, 0.0], [width, length], [0.0, length]],
        "note": "world_points are metres on the ground plane",
    }
    with open(out_path, "w") as f:
        json.dump(calib, f, indent=2)
    print(f"[CALIB] Saved -> {out_path}  ({width * length:.1f} m2 reference)")
    return calib


def estimate_px_per_m(boxes):
    """Fallback scale: median person bounding-box height maps to PERSON_HEIGHT_M."""
    if len(boxes) == 0:
        return None
    heights = boxes[:, 3] - boxes[:, 1]
    heights = heights[heights > 5]
    if heights.size == 0:
        return None
    return float(np.median(heights)) / PERSON_HEIGHT_M


# ──────────────────────────── 8. PIPELINE ───────────────────────────
def analyze(args):
    is_image = os.path.splitext(args.input)[1].lower() in (".jpg", ".jpeg", ".png", ".bmp")

    cap = cv2.VideoCapture(args.input)
    if not cap.isOpened():
        sys.exit(f"[ERROR] Cannot open input: {args.input}")

    ok, first = cap.read()
    if not ok:
        sys.exit("[ERROR] Empty input.")
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    if is_image:
        fps = 1.0
    h, w = first.shape[:2]

    # -- calibration mode --------------------------------------------
    if args.calibrate:
        run_calibration(first, args.calib_file)
        return

    detector = PersonDetector(DEVICE)

    # -- build the ground plane --------------------------------------
    ground = None
    if not args.no_calib and os.path.exists(args.calib_file):
        with open(args.calib_file) as f:
            ground = GroundPlane.from_calibration(json.load(f))
        print(f"[INFO] Calibrated ROI area: {ground.area_m2:.1f} m2")
    else:
        boxes0, _ = detector.detect(first)
        ppm = estimate_px_per_m(boxes0)
        if ppm is None:
            sys.exit("[ERROR] No people found for auto-scale. Run with --calibrate.")
        ground = GroundPlane.auto_from_height(ppm, first.shape)
        print(f"[WARN] Uncalibrated mode: {ppm:.1f} px/m from person height.")
        print(f"[WARN] Perspective is NOT corrected - area {ground.area_m2:.1f} m2 is approximate.")

    grid = DensityGrid(ground)
    motion = MotionAnalyzer(ground, first.shape)
    risk = RiskEngine(fps)

    writer = None
    if not is_image:
        writer = cv2.VideoWriter(args.output, cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))

    log = open(args.csv, "w", newline="")
    logger = csv.writer(log)
    logger.writerow(["frame", "time_s", "count", "area_m2", "mean_density",
                     "peak_density", "speed_ms", "disorder", "risk", "alarm"])

    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
    frame_i = 0
    boxes = np.zeros((0, 4), np.float32)
    print("[INFO] Analyzing ...")

    while True:
        ok, frame = cap.read()
        if not ok:
            break

        if frame_i % DETECT_EVERY_N == 0:
            boxes, _ = detector.detect(frame)

        feet = PersonDetector.foot_points(boxes)
        inside = ground.inside_roi(feet) if len(feet) else np.zeros(0, bool)
        feet_roi = feet[inside]
        count = int(inside.sum())

        world_pts = ground.to_world(feet_roi)
        density = grid.compute(world_pts)
        peak, p99 = grid.stats(density)
        mean_density = count / ground.area_m2 if ground.area_m2 > 0 else 0.0

        speed_ms, disorder = motion.update(frame, fps)
        score, alarm = risk.update(peak, mean_density, speed_ms, disorder, count)

        metrics = {
            "frame": frame_i,
            "time_s": round(frame_i / fps, 2),
            "count": count,
            "area_m2": round(ground.area_m2, 2),
            "mean_density": round(mean_density, 3),
            "peak_density": round(peak, 3),
            "density_label": RiskEngine.density_label(peak),
            "speed_ms": round(speed_ms, 3),
            "disorder": round(disorder, 3),
            "risk": round(score, 4),
            "alarm": bool(alarm),
            "hotspot": grid.hotspot_image_point(density) if count else None,
        }

        if alarm and frame_i % int(max(fps, 1)) == 0:
            on_alert({k: v for k, v in metrics.items() if k != "hotspot"})
        if args.stream_json:
            print(json.dumps({k: v for k, v in metrics.items() if k != "hotspot"}), flush=True)

        logger.writerow([metrics["frame"], metrics["time_s"], count, metrics["area_m2"],
                         metrics["mean_density"], metrics["peak_density"],
                         metrics["speed_ms"], metrics["disorder"], metrics["risk"],
                         int(alarm)])

        vis = grid.overlay(frame, density)
        vis = draw_hud(vis, ground, boxes, metrics, grid.minimap(density),
                       show_boxes=not args.no_boxes)

        if writer is not None:
            writer.write(vis)
        if args.show:
            cv2.imshow("ResQ-AI", vis)
            if cv2.waitKey(1) & 0xFF == 27:
                break

        frame_i += 1
        if is_image:
            cv2.imwrite(args.output, vis)
            print(f"\n--- RESULTS ---\nPeople : {count}\n"
                  f"Area   : {ground.area_m2:.1f} m2\n"
                  f"Avg    : {mean_density:.2f} p/m2\n"
                  f"Peak   : {peak:.2f} p/m2 ({metrics['density_label']})\n"
                  f"Verdict: {RiskEngine.label(score)[0]}")
            break
        if frame_i % 30 == 0:
            print(f"  frame {frame_i}  count={count:3d}  peak={peak:.2f} p/m2  risk={score:.2f}")

    cap.release()
    if writer is not None:
        writer.release()
    log.close()
    cv2.destroyAllWindows()
    print(f"\n[SUCCESS] Video -> {args.output}\n[SUCCESS] Metrics -> {args.csv}")


def main():
    p = argparse.ArgumentParser(description="ResQ-AI unified crowd risk analyzer")
    p.add_argument("--input", required=True, help="video or image path")
    p.add_argument("--output", default="resq_ai_output.mp4")
    p.add_argument("--csv", default="resq_ai_metrics.csv")
    p.add_argument("--calib-file", dest="calib_file", default="calibration.json")
    p.add_argument("--calibrate", action="store_true", help="run the 4-point calibration tool")
    p.add_argument("--no-calib", action="store_true", help="force auto-scale fallback")
    p.add_argument("--no-boxes", action="store_true", help="hide detection boxes")
    p.add_argument("--show", action="store_true", help="live preview window")
    p.add_argument("--stream-json", action="store_true", help="print metrics as JSON lines")
    analyze(p.parse_args())


if __name__ == "__main__":
    main()