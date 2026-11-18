"""SHAPY body-shape estimate for a completed reconstruction run (alternative measurement model).

SHAPY (Choutas et al., CVPR 2022) regresses SMPL-X body shape from a single image and was trained with
anthropometric supervision. Only its released weights (models/shapy, git-ignored) and its HRNet module
from the public code (models/shapy/code) are used; the SMPL-X body is the one stored in the checkpoint.

For a completed run it re-reads the run's selected frames from the original video (rotated to display
orientation and checked against the run's mask size), crops the subject (1.2 x box, 256 px), predicts
SMPL-X shape per frame, takes the robust median, builds the body in the pipeline's measurement A-pose and
measures it with the pipeline's own circumference code. Results are stored in relative units next to the
run, so centimetres always follow the run's current height.
"""
from __future__ import annotations

import importlib.util
import json
import logging
import os
import sys
import threading
import time
import types
from pathlib import Path

import cv2
import numpy as np
import torch

from backend.services import body_measurements as bm
from backend.services import smpl_multiview as smv

logger = logging.getLogger(__name__)

ROOT = Path(__file__).resolve().parents[2]
SHAPY_DIR = Path(os.getenv("SHAPY_DIR", str(ROOT / "models" / "shapy")))
CKPT = SHAPY_DIR / "trained_models" / "shapy" / "SHAPY_A" / "checkpoints" / "best_checkpoint"
HRNET_PY = SHAPY_DIR / "code" / "regressor" / "human_shape" / "models" / "backbone" / "hrnet.py"
RESULT_FILE = "shapy_measurements.json"
# 2: horizontal-flip test-time augmentation (each crop and its mirror image). On two recordings of the same
#    person it cut the between-recording waist/hip difference from ~0.6 cm to ~0.2 cm.
RESULT_VERSION = 2
MESH_FILE = "shapy_body_mesh.obj"
CROP = 256
SCALE = 1.2                       # SHAPY test-time bbox enlargement
MEAN = np.array([0.485, 0.456, 0.406], np.float32)
STD = np.array([0.229, 0.224, 0.225], np.float32)
HRNET_CFG = {
    "use_old_impl": False, "pretrained_layers": ["*"], "pretrained_path": "",
    "stage1": {"num_modules": 1, "num_branches": 1, "num_blocks": [4], "num_channels": [64],
               "block": "BOTTLENECK", "fuse_method": "SUM"},
    "stage2": {"num_modules": 1, "num_branches": 2, "num_blocks": [4, 4], "num_channels": [48, 96],
               "block": "BASIC", "fuse_method": "SUM"},
    "stage3": {"num_modules": 4, "num_branches": 3, "num_blocks": [4, 4, 4], "num_channels": [48, 96, 192],
               "block": "BASIC", "fuse_method": "SUM"},
    "stage4": {"num_modules": 3, "num_branches": 4, "num_blocks": [4, 4, 4, 4],
               "num_channels": [48, 96, 192, 384], "block": "BASIC", "fuse_method": "SUM"},
}
_LOCK = threading.Lock()
_CACHE: dict = {}


class ShapyUnavailable(RuntimeError):
    """SHAPY weights or code are not installed."""


def available() -> bool:
    return CKPT.is_file() and HRNET_PY.is_file()


def _load_hrnet_module():
    if "loguru" not in sys.modules:  # hrnet.py only uses loguru for log lines
        stub = types.ModuleType("loguru")
        stub.logger = logging.getLogger("shapy.hrnet")
        sys.modules["loguru"] = stub
    spec = importlib.util.spec_from_file_location("shapy_hrnet", HRNET_PY)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


class ShapyShape(torch.nn.Module):
    """HRNet-W48 features -> 3-stage iterative MLP -> SMPL-X parameters (only betas are used)."""

    def __init__(self, state: dict):
        super().__init__()
        from omegaconf import OmegaConf
        hr = _load_hrnet_module()
        self.backbone = hr.HighResolutionNet(OmegaConf.create(HRNET_CFG))
        self.backbone.load_state_dict({k[len("backbone."):]: v for k, v in state.items()
                                       if k.startswith("backbone.")}, strict=True)
        w = {k[len("regressor.module."):]: v for k, v in state.items() if k.startswith("regressor.module.")}
        self.l0, self.l1, self.out = torch.nn.Linear(2193, 1024), torch.nn.Linear(1024, 1024), torch.nn.Linear(1024, 145)
        self.l0.load_state_dict({"weight": w["layer_000.0.weight"], "bias": w["layer_000.0.bias"]})
        self.l1.load_state_dict({"weight": w["layer_001.0.weight"], "bias": w["layer_001.0.bias"]})
        self.out.load_state_dict({"weight": w["output_layer.weight"], "bias": w["output_layer.bias"]})
        self.register_buffer("mean_param", state["regressor.mean_param"].float())
        self.register_buffer("betas_idxs", state["betas_idxs"].long())

    @torch.no_grad()
    def forward(self, images):  # (B, 3, 256, 256) normalised RGB
        feat = self.backbone(images)["concat"]
        params = self.mean_param.expand(len(images), -1)
        for _ in range(3):  # IterativeRegression: 3 stages; MLP blocks are Linear+Dropout (no activation)
            params = params + self.out(self.l1(self.l0(torch.cat([feat, params], dim=1))))
        return params[:, self.betas_idxs]


class SmplxFromCheckpoint:
    """SMPL-X neutral (10 betas) from the buffers stored in the SHAPY checkpoint."""

    def __init__(self, state: dict, device):
        g = lambda k: state[f"model.{k}"].to(device)  # noqa: E731
        self.v_template, self.shapedirs = g("v_template").float(), g("shapedirs").float()
        self.posedirs, self.J_regressor = g("posedirs").float(), g("J_regressor").float()
        self.parents, self.lbs_weights = g("parents").long(), g("lbs_weights").float()
        self.faces = state["model.faces_tensor"].long().cpu().numpy()
        self.device = device

    @torch.no_grad()
    def posed(self, betas: np.ndarray, body_pose_rotmats21: np.ndarray):
        from smplx.lbs import lbs
        rot = torch.eye(3, device=self.device).repeat(1, 55, 1, 1)
        rot[0, 1:22] = torch.as_tensor(body_pose_rotmats21, dtype=torch.float32, device=self.device)
        b = torch.as_tensor(betas, dtype=torch.float32, device=self.device)[None]
        verts, joints = lbs(b, rot, self.v_template, self.shapedirs, self.posedirs, self.J_regressor,
                            self.parents, self.lbs_weights, pose2rot=False)
        v, j = verts[0].cpu().numpy().astype(np.float64), joints[0].cpu().numpy().astype(np.float64)
        floor = v[:, 1].min()
        v[:, 1] -= floor
        j[:, 1] -= floor
        return v, j


def _models():
    if not available():
        raise ShapyUnavailable("SHAPY is not installed: expected models/shapy/trained_models/shapy/SHAPY_A and "
                               "models/shapy/code (see scripts/shapy_compare.py)")
    if "net" not in _CACHE:
        device = "cuda" if torch.cuda.is_available() else "cpu"
        state = torch.load(CKPT, map_location="cpu", weights_only=True)["model"]
        _CACHE.update(device=device, net=ShapyShape(state).to(device).eval(),
                      body=SmplxFromCheckpoint(state, device))
        logger.info("SHAPY loaded on %s", device)
    return _CACHE["net"], _CACHE["body"], _CACHE["device"]


# ------------------------------------------------------------- image prep
def read_display_frames(video: Path, indices, display_hw) -> dict[int, np.ndarray]:
    """Frames in display orientation; refuses frames whose size does not match the run's masks."""
    cap = cv2.VideoCapture(str(video))
    cap.set(cv2.CAP_PROP_ORIENTATION_AUTO, 0)  # rotate explicitly, exactly once
    meta = int(round(cap.get(cv2.CAP_PROP_ORIENTATION_META) or 0)) % 360
    rotate = {90: cv2.ROTATE_90_CLOCKWISE, 180: cv2.ROTATE_180, 270: cv2.ROTATE_90_COUNTERCLOCKWISE}.get(meta)
    out = {}
    try:
        for idx in sorted(set(int(i) for i in indices)):
            cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
            ok, frame = cap.read()
            if not ok:
                continue
            if rotate is not None:
                frame = cv2.rotate(frame, rotate)
            if frame.shape[:2] != tuple(display_hw):
                raise ValueError(f"frame {idx}: decoded size {frame.shape[:2]} (rotation {meta}) does not match "
                                 f"the run's display size {tuple(display_hw)}")
            out[idx] = frame
    finally:
        cap.release()
    return out


def mask_box(mask: np.ndarray, frame_hw) -> tuple | None:
    ys, xs = np.nonzero(mask)
    if len(xs) < 50:
        return None
    sy, sx = frame_hw[0] / mask.shape[0], frame_hw[1] / mask.shape[1]
    return (xs.min() * sx, ys.min() * sy, (xs.max() + 1) * sx, (ys.max() + 1) * sy)


def crop_for_shapy(frame_bgr: np.ndarray, box) -> np.ndarray:
    """Square crop of SCALE x the longer box side, centred on the box, resized to 256, ImageNet-normalised RGB."""
    x0, y0, x1, y1 = box
    size = SCALE * max(x1 - x0, y1 - y0)
    s = CROP / size
    cx, cy = (x0 + x1) / 2, (y0 + y1) / 2
    affine = np.array([[s, 0, CROP / 2 - s * cx], [0, s, CROP / 2 - s * cy]], np.float32)
    patch = cv2.warpAffine(frame_bgr, affine, (CROP, CROP), flags=cv2.INTER_LINEAR,
                           borderMode=cv2.BORDER_CONSTANT, borderValue=0)
    rgb = cv2.cvtColor(patch, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    return (rgb - MEAN) / STD


# -------------------------------------------------------------- run level
def compute_for_run(run_dir: Path, video: Path) -> dict:
    """Run SHAPY on a completed run's selected frames and store relative measurements + mesh."""
    selected = json.loads((run_dir / "selected_frames.json").read_text(encoding="utf-8"))["selected"]
    fit = np.load(run_dir / "diagnostics" / "fit_inputs.npz", allow_pickle=False)
    mh, mw = (int(x) for x in fit["mask_shape"])
    frames = read_display_frames(video, [s["source_frame_index"] for s in selected], (mh, mw))
    crops, used = [], []
    for i, s in enumerate(selected):
        frame = frames.get(int(s["source_frame_index"]))
        if frame is None or i >= len(fit["mask_usable"]) or not bool(fit["mask_usable"][i]):
            continue
        mask = np.unpackbits(fit["masks_packed"][i])[: mh * mw].reshape(mh, mw)  # pipeline packing
        box = mask_box(mask, frame.shape[:2])
        if box is None:
            continue
        crops.append(crop_for_shapy(frame, box))
        used.append(int(s["source_frame_index"]))
    if len(crops) < 3:
        raise ValueError(f"Only {len(crops)} usable frames for SHAPY (need at least 3)")

    started = time.time()
    with _LOCK:
        net, body, device = _models()
        stack = np.stack(crops)
        # Flip TTA: the mirror image of a centred crop is the crop of the mirrored frame; body shape is
        # left-right symmetric, so both views estimate the same betas with independent errors.
        stack = np.concatenate([stack, stack[:, :, ::-1]])
        batch = torch.as_tensor(np.ascontiguousarray(stack)).permute(0, 3, 1, 2).to(device)
        betas_each = np.concatenate([net(batch[k:k + 8]).cpu().numpy() for k in range(0, len(batch), 8)])
        betas, robust_std, outliers = smv.robust_beta_aggregate(betas_each)
        v, j = body.posed(betas, smv.measurement_pose()[:21])
        faces = body.faces
    meas = bm.measure_body(v, faces, j[:24])
    smv.write_obj(run_dir / MESH_FILE, v, faces,
                  header=("SHAPY SMPL-X neutral shape, measurement A-pose (relative units; y up, +z front, feet on y=0)\n"
                          "External geometry only - not a Tanita BIA/BCM measurement."))
    record = bm.strip_rings(meas)
    record.update({
        "model": "SHAPY (SHAPY_A checkpoint, SMPL-X neutral, 10 betas)", "version": RESULT_VERSION,
        "test_time_augmentation": "horizontal flip (each frame and its mirror image)",
        "frames_used": len(crops), "frame_indices": used, "betas_smplx": betas.tolist(),
        "betas_each": betas_each.tolist(), "beta_robust_std": robust_std.tolist(),
        "outlier_frames": [int(o) for o in outliers], "mesh_file": MESH_FILE,
        "computed_at": time.time(), "seconds": round(time.time() - started, 2),
        "note": "Same A-pose and circumference definitions as the pipeline; SHAPY's shape only.",
    })
    (run_dir / RESULT_FILE).write_text(json.dumps(record, indent=2, allow_nan=False), encoding="utf-8")
    return record


def summarize(record: dict, height_cm: float | None) -> dict:
    """Relative and (if a height is set) cm circumferences for API responses."""
    rel = {k: m.get("tape_circumference_over_height") if m.get("valid") else None
           for k, m in record["measurements"].items()}
    out = {"model": record.get("model"), "frames_used": record.get("frames_used"),
           "valid_count": record.get("valid_count"), "total": record.get("total"),
           "measurements": rel, "mesh_obj": MESH_FILE, "computed_at": record.get("computed_at"),
           "beta_robust_std": record.get("beta_robust_std")}
    if height_cm is not None:
        scaled = bm.scale_to_height(record, height_cm)
        out["height_cm"] = height_cm
        out["measurements_cm"] = {k: m.get("tape_circumference_cm") for k, m in scaled["measurements"].items()}
    return out
