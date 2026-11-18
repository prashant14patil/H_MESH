"""Person detection and single-subject tracking for the 360-degree pipeline.

Detector: torchvision Keypoint R-CNN (COCO person boxes + 17 keypoints).
Tracker: motion (IoU / centre distance / size) + appearance (HSV histograms of
upper and lower body) association for ONE operator-confirmed subject.

The tracker never silently switches to another person: a frame is accepted
only if the matched detection is geometrically and visually consistent with
the subject, otherwise the frame is marked ``lost`` / ``ambiguous`` and an
event is recorded in the report.
"""
from __future__ import annotations

import math
import threading
from dataclasses import dataclass, field

import cv2
import numpy as np

COCO_NAMES = ["nose", "left_eye", "right_eye", "left_ear", "right_ear",
              "left_shoulder", "right_shoulder", "left_elbow", "right_elbow",
              "left_wrist", "right_wrist", "left_hip", "right_hip",
              "left_knee", "right_knee", "left_ankle", "right_ankle"]


@dataclass
class Detection:
    box: np.ndarray            # xyxy pixels
    score: float
    keypoints: np.ndarray      # [17, 2] pixels
    keypoint_scores: np.ndarray  # [17] raw keypoint logits (higher = more confident)
    descriptor: np.ndarray | None = None

    @property
    def height(self) -> float:
        return float(self.box[3] - self.box[1])

    @property
    def center(self) -> np.ndarray:
        return np.array([(self.box[0] + self.box[2]) / 2, (self.box[1] + self.box[3]) / 2])


# ---------------------------------------------------------------- detector
class PersonDetector:
    """Lazy, thread-safe Keypoint R-CNN wrapper."""

    def __init__(self, device: str | None = None, min_score: float = 0.6) -> None:
        self._device = device
        self._model = None
        self._lock = threading.Lock()
        self.min_score = min_score

    def _load(self):
        import torch
        import torchvision
        if self._model is None:
            with self._lock:
                if self._model is None:
                    device = self._device or ("cuda" if torch.cuda.is_available() else "cpu")
                    model = torchvision.models.detection.keypointrcnn_resnet50_fpn(weights="DEFAULT")
                    self._model = model.to(device).eval()
                    self._device = device
        return self._model

    def detect(self, bgr: np.ndarray) -> list[Detection]:
        import torch
        model = self._load()
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        tensor = torch.from_numpy(rgb).permute(2, 0, 1).float().div(255).to(self._device)
        with torch.no_grad():
            out = model([tensor])[0]
        result = []
        for box, label, score, kps, kp_scores in zip(out["boxes"], out["labels"], out["scores"],
                                                      out["keypoints"], out["keypoints_scores"]):
            if int(label) != 1 or float(score) < self.min_score:
                continue
            det = Detection(box=box.cpu().numpy().astype(np.float64), score=float(score),
                            keypoints=kps[:, :2].cpu().numpy().astype(np.float64),
                            keypoint_scores=kp_scores.cpu().numpy().astype(np.float64))
            det.descriptor = appearance_descriptor(bgr, det.box)
            result.append(det)
        return result


_SHARED_DETECTOR: PersonDetector | None = None
_SHARED_LOCK = threading.Lock()


def get_shared_detector() -> PersonDetector:
    """One process-wide detector so quality analysis and reconstruction share GPU weights."""
    global _SHARED_DETECTOR
    with _SHARED_LOCK:
        if _SHARED_DETECTOR is None:
            _SHARED_DETECTOR = PersonDetector()
        return _SHARED_DETECTOR


def keypoint_confidence01(scores) -> np.ndarray:
    """Map Keypoint R-CNN heat-map logits to [0, 1] weights (0.5 -> 0, >= 4.5 -> 1)."""
    return np.clip((np.asarray(scores, dtype=np.float64) - 0.5) / 4.0, 0.0, 1.0)


# --------------------------------------------------------------- appearance
def appearance_descriptor(bgr: np.ndarray, box) -> np.ndarray:
    """Concatenated HSV histograms of the upper and lower body halves."""
    h, w = bgr.shape[:2]
    x0, y0, x1, y1 = [int(round(v)) for v in box]
    x0, x1 = max(0, x0), min(w, x1)
    y0, y1 = max(0, y0), min(h, y1)
    if x1 - x0 < 8 or y1 - y0 < 8:
        return np.zeros(2 * 12 * 4)
    # Use the central 60% width so background at the box edge matters less.
    margin = int((x1 - x0) * 0.2)
    crop = bgr[y0:y1, x0 + margin:x1 - margin]
    if crop.size == 0:
        return np.zeros(2 * 12 * 4)
    hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
    half = hsv.shape[0] // 2
    parts = []
    for part in (hsv[:half], hsv[half:]):
        hist = cv2.calcHist([part], [0, 1], None, [12, 4], [0, 180, 0, 256]).flatten()
        total = hist.sum()
        parts.append(hist / total if total > 0 else hist)
    return np.concatenate(parts) / 2.0


def appearance_distance(a: np.ndarray | None, b: np.ndarray | None) -> float:
    """Bhattacharyya distance in [0, 1] between two descriptors."""
    if a is None or b is None or a.sum() <= 0 or b.sum() <= 0:
        return 1.0
    a = a / a.sum()
    b = b / b.sum()
    bc = float(np.sum(np.sqrt(a * b)))
    return float(math.sqrt(max(0.0, 1.0 - bc)))


def iou_xyxy(a, b) -> float:
    ix0, iy0 = max(a[0], b[0]), max(a[1], b[1])
    ix1, iy1 = min(a[2], b[2]), min(a[3], b[3])
    inter = max(0.0, ix1 - ix0) * max(0.0, iy1 - iy0)
    union = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return float(inter / union) if union > 0 else 0.0


# ------------------------------------------------------------------ tracker
@dataclass
class TrackerConfig:
    max_cost: float = 0.85
    min_iou_or_close: float = 0.15      # need IoU above this OR centre within max_center_dist
    max_center_dist: float = 0.6        # in units of subject box height
    max_height_ratio: float = 1.6
    appearance_reject: float = 0.55     # absolute appearance rejection
    ambiguity_margin: float = 0.08      # best vs runner-up cost
    reacquire_appearance: float = 0.30  # strict appearance for re-identification
    appearance_alpha: float = 0.15      # reference update rate
    w_iou: float = 0.45
    w_center: float = 0.25
    w_app: float = 0.25
    w_size: float = 0.05


@dataclass
class TrackRecord:
    index: int                      # position in the sampled sequence
    status: str                     # tracked | lost | ambiguous | no_detection
    box: list | None = None
    score: float | None = None
    iou_prev: float | None = None
    appearance: float | None = None
    cost: float | None = None
    n_detections: int = 0
    detection: Detection | None = field(default=None, repr=False)
    note: str | None = None


class SubjectTracker:
    """Track one subject through an ordered sequence of detection lists."""

    def __init__(self, seed: Detection, config: TrackerConfig | None = None) -> None:
        self.cfg = config or TrackerConfig()
        self.reference = seed.descriptor.copy()
        self.last = seed
        self.last_index = None
        self.events: list[dict] = []
        self.lost_run = 0

    def _cost(self, det: Detection):
        cfg = self.cfg
        last = self.last
        iou = iou_xyxy(det.box, last.box)
        cdist = float(np.linalg.norm(det.center - last.center) / max(last.height, 1.0))
        ratio = max(det.height, 1.0) / max(last.height, 1.0)
        size = abs(math.log(ratio))
        app = appearance_distance(det.descriptor, self.reference)
        cost = (cfg.w_iou * (1 - iou) + cfg.w_center * min(cdist, 2.0) / 2.0
                + cfg.w_app * app + cfg.w_size * min(size, 1.0))
        return cost, iou, cdist, ratio, app

    def step(self, index: int, detections: list[Detection]) -> TrackRecord:
        cfg = self.cfg
        if not detections:
            self.lost_run += 1
            return TrackRecord(index, "no_detection", n_detections=0)
        scored = []
        for det in detections:
            cost, iou, cdist, ratio, app = self._cost(det)
            scored.append((cost, det, iou, cdist, ratio, app))
        scored.sort(key=lambda s: s[0])
        cost, det, iou, cdist, ratio, app = scored[0]
        n = len(detections)
        geometric_ok = ((iou >= cfg.min_iou_or_close or cdist <= cfg.max_center_dist)
                        and 1 / cfg.max_height_ratio <= ratio <= cfg.max_height_ratio)
        if self.lost_run > 0:
            # After losing the subject only strict appearance + size may re-identify.
            geometric_ok = (app <= cfg.reacquire_appearance
                            and 1 / cfg.max_height_ratio <= ratio <= cfg.max_height_ratio)
        record = TrackRecord(index, "tracked", box=det.box.tolist(), score=det.score,
                             iou_prev=iou, appearance=app, cost=cost, n_detections=n, detection=det)
        if not geometric_ok or cost > cfg.max_cost or app > cfg.appearance_reject:
            record.status = "lost"
            record.detection = None
            record.note = (f"best candidate failed consistency (iou={iou:.2f}, centre_dist={cdist:.2f}, "
                           f"height_ratio={ratio:.2f}, appearance={app:.2f})")
            if app > cfg.appearance_reject and (iou >= cfg.min_iou_or_close or cdist <= cfg.max_center_dist):
                self.events.append({"type": "possible_identity_switch", "index": index,
                                    "detail": "geometrically plausible detection rejected for different appearance",
                                    "appearance": app, "iou": iou})
                record.note += "; flagged possible identity switch"
            self.lost_run += 1
            return record
        if len(scored) > 1 and scored[1][0] - cost < cfg.ambiguity_margin and scored[1][2] > 0.05:
            record.status = "ambiguous"
            record.detection = None
            record.note = "two overlapping detections score similarly; frame rejected"
            self.events.append({"type": "ambiguous_overlap", "index": index,
                                "cost_margin": float(scored[1][0] - cost)})
            self.lost_run += 1
            return record
        if self.lost_run > 0:
            self.events.append({"type": "reacquired", "index": index, "frames_lost": self.lost_run,
                                "appearance": app})
        self.lost_run = 0
        self.last = det
        a = cfg.appearance_alpha
        self.reference = (1 - a) * self.reference + a * det.descriptor
        return record


def track_sequence(detections_per_frame: list[list[Detection]], seed_index: int, seed: Detection,
                   config: TrackerConfig | None = None):
    """Track forward and backward from the operator-confirmed seed detection.

    Returns (records, events) where records has one entry per frame, ordered.
    """
    n = len(detections_per_frame)
    records: list[TrackRecord | None] = [None] * n
    events: list[dict] = []
    seed_record = TrackRecord(seed_index, "tracked", box=seed.box.tolist(), score=seed.score,
                              iou_prev=1.0, appearance=0.0, cost=0.0,
                              n_detections=len(detections_per_frame[seed_index]), detection=seed,
                              note="operator-confirmed seed")
    records[seed_index] = seed_record
    for direction in (1, -1):
        tracker = SubjectTracker(seed, config)
        i = seed_index + direction
        while 0 <= i < n:
            rec = tracker.step(i, detections_per_frame[i])
            records[i] = rec
            i += direction
        for event in tracker.events:
            event = dict(event)
            event["direction"] = "forward" if direction == 1 else "backward"
            events.append(event)
    events.sort(key=lambda e: e["index"])
    return records, events


# ---------------------------------------------------------------- selection
def candidate_summary(det: Detection, image_size: tuple[int, int]) -> dict:
    w, h = image_size
    x0, y0, x1, y1 = det.box
    return {"box_xyxy": [float(x0), float(y0), float(x1), float(y1)],
            "box_norm": [float(x0 / w), float(y0 / h), float(x1 / w), float(y1 / h)],
            "score": det.score, "area_fraction": float((x1 - x0) * (y1 - y0) / (w * h))}


def pick_selection_frame(detections_per_frame: list[list[Detection]], min_area: float = 0.02):
    """Choose the frame on which the operator confirms the subject.

    Returns (frame_position, candidates) where candidates are detections large
    enough to be a subject. The first frame containing any candidate is used;
    if several candidate people exist anywhere in the first frames of the
    sequence the operator must pick.
    """
    for i, dets in enumerate(detections_per_frame):
        big = [d for d in dets if d.height > 0]
        if big:
            return i, big
    return None, []


def count_multi_person_frames(detections_per_frame: list[list[Detection]], min_height_frac: float,
                              image_height: int) -> int:
    count = 0
    for dets in detections_per_frame:
        if sum(1 for d in dets if d.height / image_height >= min_height_frac) >= 2:
            count += 1
    return count
