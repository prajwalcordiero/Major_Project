#!/usr/bin/env python3
"""
YOLOv8 person detector — drop-in replacement for the Faster R-CNN
`PersonDetector` in resq_ai_analyzer.py.

Same public interface, so it can be swapped into CrowdAnalyzer or the
comparison script with zero other code changes:

    detector.detect(frame_bgr) -> (boxes_xyxy: np.ndarray[N,4], scores: np.ndarray[N])
    YOLOPersonDetector.foot_points(boxes) -> np.ndarray[N,2]

Install:
    pip install ultralytics --break-system-packages

Weights: "yolov8n.pt" (nano, fastest, least accurate) is downloaded
automatically on first run. Swap in "yolov8s.pt" / "yolov8m.pt" /
"yolov8l.pt" / "yolov8x.pt" for more accuracy at the cost of speed —
that trade-off is itself worth including in your writeup.
"""

import numpy as np
from ultralytics import YOLO

# COCO class id for 'person'. NOTE: this is 0 in the YOLO/Ultralytics label
# map, but 1 in the torchvision/Faster R-CNN label map used elsewhere in
# this project — that's not a bug, the two model zoos just index COCO
# classes differently (torchvision reserves 0 for "background").
COCO_PERSON_CLASS_ID = 0


class YOLOPersonDetector:
    """Same call signature as resq_ai_analyzer.PersonDetector."""

    def __init__(self, weights="yolov8n.pt", device="cpu", conf=0.5):
        print(f"[INFO] Loading YOLOv8 ({weights}) on {device} ...")
        self.model = YOLO(weights)
        self.device = self._normalize_device(device)
        self.conf = conf

    @staticmethod
    def _normalize_device(device):
        # Ultralytics wants "cpu" or a CUDA index (0, 1, ...), not a
        # torch.device object.
        s = str(device)
        if "cuda" in s:
            return 0
        return "cpu"

    def detect(self, frame_bgr):
        results = self.model.predict(
            source=frame_bgr,
            conf=self.conf,
            classes=[COCO_PERSON_CLASS_ID],
            device=self.device,
            verbose=False,
        )[0]

        if results.boxes is None or len(results.boxes) == 0:
            return np.zeros((0, 4), np.float32), np.zeros((0,), np.float32)

        boxes = results.boxes.xyxy.cpu().numpy().astype(np.float32)
        scores = results.boxes.conf.cpu().numpy().astype(np.float32)
        return boxes, scores

    @staticmethod
    def foot_points(boxes):
        """Identical to PersonDetector.foot_points: bottom-center of each box."""
        if len(boxes) == 0:
            return np.zeros((0, 2), np.float32)
        x = (boxes[:, 0] + boxes[:, 2]) * 0.5
        y = boxes[:, 3]
        return np.stack([x, y], axis=1).astype(np.float32)