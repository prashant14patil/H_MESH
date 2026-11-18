"""Subject silhouettes for multi-view shape fitting.

Three independent mask sources are combined per frame:

* Mask R-CNN v2 (COCO, GPU): instance mask of the *tracked* subject, matched to the
  tracker's box. This is what ties the silhouette to one person: pixels claimed by
  any other detected person instance are removed.
* GrabCut (CPU), seeded by the Mask R-CNN mask: follows image edges more tightly than
  Mask R-CNN's 28x28 mask head.
* MediaPipe selfie segmenter (CPU TFLite) on a padded subject crop: an
  independently trained person segmenter.

The final mask is a per-pixel majority vote (2 of 3), restricted to the padded subject
box and to the component that contains the subject's keypoints. The mean pairwise IoU of
the three sources is reported as ``agreement``: a direct, measured reliability signal used
to exclude silhouettes that the sources disagree on.

Clothing, hair and shoes are part of the silhouette; an SMPL body fitted to it is a
*clothed-silhouette* body shape, not a measurement of the body beneath clothing.
"""
from __future__ import annotations

import logging
import os
import threading
from dataclasses import dataclass, field
from pathlib import Path

import cv2
import numpy as np

logger = logging.getLogger(__name__)

MIN_AGREEMENT = 0.85          # mean pairwise IoU of the three mask sources
MIN_BOX_FILL = 0.20           # mask area / box area (slim profile views fill ~0.28 of a wide box)
MAX_BOX_FILL = 0.95
_MODEL_DIR = Path(os.getenv("MEDIAPIPE_MODEL_DIR", "models/mediapipe"))


def mask_iou(a: np.ndarray, b: np.ndarray) -> float:
    a, b = a.astype(bool), b.astype(bool)
    union = np.logical_or(a, b).sum()
    return float(np.logical_and(a, b).sum() / union) if union else 0.0


def _box_iou(a, b) -> float:
    x0, y0 = max(a[0], b[0]), max(a[1], b[1])
    x1, y1 = min(a[2], b[2]), min(a[3], b[3])
    inter = max(0.0, x1 - x0) * max(0.0, y1 - y0)
    union = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return float(inter / union) if union > 0 else 0.0


def padded_box(box, image_wh, pad_frac=0.08):
    w, h = image_wh
    x0, y0, x1, y1 = [float(v) for v in box]
    pad = pad_frac * (y1 - y0)
    return (int(max(0, x0 - pad)), int(max(0, y0 - pad)), int(min(w, x1 + pad)), int(min(h, y1 + pad)))


def majority_vote(masks: list[np.ndarray]) -> np.ndarray:
    votes = np.sum([m.astype(np.uint8) for m in masks], axis=0)
    return (votes * 2 > len(masks)).astype(np.uint8)


def clean_mask(mask: np.ndarray, keypoints: np.ndarray | None, box_height: float) -> tuple[np.ndarray, int]:
    """Remove thin spurs and stray blobs; keep the component holding the subject.

    Returns (mask, number of significant components before selection).
    """
    k = max(3, int(round(box_height * 0.006)) | 1)
    m = cv2.morphologyEx(mask.astype(np.uint8), cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k)))
    n, labels, stats, _ = cv2.connectedComponentsWithStats(m, connectivity=8)
    if n <= 1:
        return m, 0
    areas = stats[1:, cv2.CC_STAT_AREA]
    significant = int((areas > 0.02 * areas.max()).sum())
    keep = 1 + int(np.argmax(areas))
    if keypoints is not None and len(keypoints):
        votes = np.zeros(n, int)
        h, w = m.shape
        for x, y in keypoints:
            xi, yi = int(round(x)), int(round(y))
            if 0 <= xi < w and 0 <= yi < h and labels[yi, xi] > 0:
                votes[labels[yi, xi]] += 1
        if votes.max() > 0:
            keep = int(np.argmax(votes))
    out = (labels == keep).astype(np.uint8)
    # Fill small enclosed holes (sensor noise, prints on clothing) but keep large gaps such as arm-torso.
    inv = (1 - out).astype(np.uint8)
    n2, lab2, st2, _ = cv2.connectedComponentsWithStats(inv, connectivity=4)
    small = 0.0015 * box_height ** 2
    for i in range(1, n2):
        x, y, bw, bh, area = st2[i]
        touches_border = x == 0 or y == 0 or x + bw >= out.shape[1] or y + bh >= out.shape[0]
        if not touches_border and area < small:
            out[lab2 == i] = 1
    return out, significant


@dataclass
class SubjectMask:
    mask: np.ndarray                   # full-image uint8 {0,1}
    agreement: float                   # mean pairwise IoU of sources
    source_iou: dict = field(default_factory=dict)
    box_fill: float = 0.0
    components: int = 0
    touches_border: bool = False
    other_people_removed_px: int = 0
    reasons: list = field(default_factory=list)

    @property
    def usable(self) -> bool:
        return not self.reasons


class SubjectSegmenter:
    """Lazy, thread-safe holder for the GPU Mask R-CNN and CPU MediaPipe segmenter."""

    def __init__(self, device: str | None = None):
        self._device = device
        self._mrcnn = None
        self._selfie = None
        self._lock = threading.Lock()

    @property
    def device(self) -> str:
        self._load()
        return self._device

    def _load(self):
        import torch
        import torchvision
        with self._lock:
            if self._mrcnn is None:
                device = self._device or ("cuda" if torch.cuda.is_available() else "cpu")
                weights = torchvision.models.detection.MaskRCNN_ResNet50_FPN_V2_Weights.COCO_V1
                self._mrcnn = torchvision.models.detection.maskrcnn_resnet50_fpn_v2(weights=weights).to(device).eval()
                self._device = device
            if self._selfie is None:
                try:
                    from mediapipe.tasks import python as mpp
                    from mediapipe.tasks.python import vision as mpv
                    opts = mpv.ImageSegmenterOptions(
                        base_options=mpp.BaseOptions(model_asset_path=str(_MODEL_DIR / "selfie_segmenter.tflite")),
                        output_confidence_masks=True)
                    self._selfie = mpv.ImageSegmenter.create_from_options(opts)
                except Exception as exc:  # noqa: BLE001
                    logger.warning("MediaPipe selfie segmenter unavailable: %s", exc)
                    self._selfie = False

    def _instances(self, bgr):
        import torch
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        tensor = torch.from_numpy(rgb).permute(2, 0, 1).float().div(255).to(self._device)
        with torch.no_grad():
            out = self._mrcnn([tensor])[0]
        keep = [(out["boxes"][i].cpu().numpy(), (out["masks"][i, 0] > 0.5).cpu().numpy().astype(np.uint8))
                for i in range(len(out["labels"])) if int(out["labels"][i]) == 1 and float(out["scores"][i]) >= 0.5]
        return keep

    def _selfie_mask(self, bgr, crop):
        if not self._selfie:
            return None
        import mediapipe as mp
        x0, y0, x1, y1 = crop
        rgb = np.ascontiguousarray(cv2.cvtColor(bgr[y0:y1, x0:x1], cv2.COLOR_BGR2RGB))
        res = self._selfie.segment(mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb))
        conf = res.confidence_masks[0].numpy_view()
        out = np.zeros(bgr.shape[:2], np.uint8)
        out[y0:y1, x0:x1] = (cv2.resize(conf, (x1 - x0, y1 - y0)) > 0.5).astype(np.uint8)
        return out

    @staticmethod
    def _grabcut(bgr, seed, crop, box_height):
        x0, y0, x1, y1 = crop
        img = bgr[y0:y1, x0:x1]
        s = seed[y0:y1, x0:x1]
        scale = min(1.0, 700.0 / max(img.shape[0], 1))
        small = cv2.resize(img, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
        ss = cv2.resize(s, (small.shape[1], small.shape[0]), interpolation=cv2.INTER_NEAREST)
        band = max(3, int(round(box_height * scale * 0.02)))
        gc = np.full(ss.shape, cv2.GC_BGD, np.uint8)
        gc[cv2.dilate(ss, np.ones((2 * band + 1, 2 * band + 1), np.uint8)) > 0] = cv2.GC_PR_BGD
        gc[ss > 0] = cv2.GC_PR_FGD
        gc[cv2.erode(ss, np.ones((band + 1, band + 1), np.uint8)) > 0] = cv2.GC_FGD
        bg, fg = np.zeros((1, 65)), np.zeros((1, 65))
        try:
            cv2.grabCut(small, gc, None, bg, fg, 4, cv2.GC_INIT_WITH_MASK)
        except cv2.error:
            return None
        res = ((gc == cv2.GC_FGD) | (gc == cv2.GC_PR_FGD)).astype(np.uint8)
        out = np.zeros(seed.shape, np.uint8)
        out[y0:y1, x0:x1] = cv2.resize(res, (x1 - x0, y1 - y0), interpolation=cv2.INTER_LINEAR) > 0
        return out

    def segment(self, bgr: np.ndarray, box, keypoints=None) -> SubjectMask:
        """Silhouette of the subject identified by ``box`` (the tracker's box)."""
        self._load()
        h, w = bgr.shape[:2]
        box_height = float(box[3] - box[1])
        crop = padded_box(box, (w, h))
        instances = self._instances(bgr)
        reasons = []
        if not instances:
            empty = np.zeros((h, w), np.uint8)
            return SubjectMask(mask=empty, agreement=0.0, reasons=["no_person_instance_mask"])
        ious = [_box_iou(b, box) for b, _ in instances]
        best = int(np.argmax(ious))
        if ious[best] < 0.5:
            reasons.append("instance_mask_does_not_match_tracked_subject")
        mrcnn = instances[best][1]
        others = np.zeros((h, w), np.uint8)
        for i, (_, m) in enumerate(instances):
            if i != best:
                others |= m
        sources = {"mask_rcnn": mrcnn}
        grab = self._grabcut(bgr, mrcnn, crop, box_height)
        if grab is not None:
            sources["grabcut"] = grab
        selfie = self._selfie_mask(bgr, crop)
        if selfie is not None:
            sources["selfie"] = selfie
        region = np.zeros((h, w), np.uint8)
        region[crop[1]:crop[3], crop[0]:crop[2]] = 1
        for key in sources:
            sources[key] = sources[key] & region
        names = list(sources)
        pair = {f"{a}~{b}": mask_iou(sources[a], sources[b]) for i, a in enumerate(names) for b in names[i + 1:]}
        agreement = float(np.mean(list(pair.values()))) if pair else 0.0
        voted = majority_vote(list(sources.values())) if len(sources) >= 3 else mrcnn & region
        removed = int((voted & others).sum())
        voted = voted & (1 - others)
        mask, components = clean_mask(voted, keypoints, box_height)
        area = float(mask.sum())
        bx = max(1.0, (box[2] - box[0]) * (box[3] - box[1]))
        fill = area / bx
        touches = bool(mask[0].any() or mask[-1].any() or mask[:, 0].any() or mask[:, -1].any())
        if len(sources) < 3:
            reasons.append("fewer_than_three_mask_sources")
        if agreement < MIN_AGREEMENT:
            reasons.append(f"mask_sources_disagree(agreement={agreement:.2f})")
        if not MIN_BOX_FILL <= fill <= MAX_BOX_FILL:
            reasons.append(f"mask_area_implausible(fill={fill:.2f})")
        if touches:
            reasons.append("silhouette_touches_image_border")
        if removed > 0.02 * area:
            reasons.append("overlaps_another_person")
        return SubjectMask(mask=mask, agreement=agreement, source_iou=pair, box_fill=fill, components=components,
                           touches_border=touches, other_people_removed_px=removed, reasons=reasons)

    def release(self):
        """Free GPU memory held by Mask R-CNN (it is reloaded lazily)."""
        import torch
        with self._lock:
            self._mrcnn = None
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


_SHARED: SubjectSegmenter | None = None
_SHARED_LOCK = threading.Lock()


def get_shared_segmenter() -> SubjectSegmenter:
    global _SHARED
    with _SHARED_LOCK:
        if _SHARED is None:
            _SHARED = SubjectSegmenter()
        return _SHARED


# ------------------------------------------------------------- fit targets
def silhouette_target(mask: np.ndarray, box, keypoints=None, n_contour: int = 400, map_height: int = 256):
    """Precompute what the silhouette loss needs, in full-image pixel coordinates.

    Returns (contour_points [N,2], contour_weights [N], (outside_distance_map, (ox, oy, s))):
    the distance map holds, for each low-resolution cell, the distance (in map cells) to the
    nearest mask pixel; ``pixel = origin + cell * s``.
    """
    h, w = mask.shape
    x0, y0, x1, y1 = padded_box(box, (w, h), pad_frac=0.15)
    crop = mask[y0:y1, x0:x1]
    s = (y1 - y0) / float(map_height)
    small = cv2.resize(crop, (max(2, int(round((x1 - x0) / s))), map_height), interpolation=cv2.INTER_AREA) > 0.5
    outside = cv2.distanceTransform((~small).astype(np.uint8), cv2.DIST_L2, 5).astype(np.float32)
    contours, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
    pts = np.concatenate([c[:, 0, :] for c in contours], 0).astype(np.float64) if contours else np.zeros((0, 2))
    if len(pts) > n_contour:
        pts = pts[np.linspace(0, len(pts) - 1, n_contour).astype(int)]
    weights = np.ones(len(pts))
    if keypoints is not None and len(pts):
        # Hair and shoes are not part of SMPL's body surface: down-weight contour above the
        # shoulders and below the ankles (COCO 5/6 shoulders, 15/16 ankles).
        shoulder_y = float(np.min(keypoints[[5, 6], 1]))
        ankle_y = float(np.max(keypoints[[15, 16], 1]))
        head = pts[:, 1] < shoulder_y - 0.02 * (ankle_y - shoulder_y)
        feet = pts[:, 1] > ankle_y
        weights[head | feet] = 0.3
    return pts, weights, (outside, (float(x0), float(y0), s))
