"""Run SHAPY (shape-focused body regressor) on a reconstruction's selected frames and compare girths.

SHAPY (Choutas et al., CVPR 2022) predicts SMPL-X body shape from one image and was trained with
anthropometric supervision. This runner uses only its released weights (models/shapy, from the
SHAPY download) and its HRNet module from the public code (models/shapy/code); it does not need the
full SHAPY install. The SMPL-X body model used is the one stored inside the SHAPY checkpoint.

For one completed reconstruction run it:
  1. re-reads the selected frames from the original video and crops the subject (1.2 x box, 256 px,
     the SHAPY test-time crop) using the run's saved subject silhouette;
  2. predicts SMPL-X shape (10 betas) per frame and takes the robust median across frames;
  3. builds the SMPL-X body in the same measurement A-pose, scales it to the given height and measures
     it with the pipeline's own circumference code, next to the run's current values.

Usage:
  .venv/Scripts/python.exe scripts/shapy_compare.py RUN_ID --height-cm 176 [--out outputs/shapy/<run>.json]
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import sys
import types
from pathlib import Path

import cv2
import numpy as np
import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from backend.services import body_measurements as bm  # noqa: E402
from backend.services import smpl_multiview as smv  # noqa: E402

SHAPY = ROOT / "models" / "shapy"
CKPT = SHAPY / "trained_models" / "shapy" / "SHAPY_A" / "checkpoints" / "best_checkpoint"
HRNET_PY = SHAPY / "code" / "regressor" / "human_shape" / "models" / "backbone" / "hrnet.py"
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


def _load_hrnet_module():
    if "loguru" not in sys.modules:  # hrnet.py only uses loguru for log lines
        import logging
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
        self.l0 = torch.nn.Linear(2193, 1024)
        self.l1 = torch.nn.Linear(1024, 1024)
        self.out = torch.nn.Linear(1024, 145)
        self.l0.load_state_dict({"weight": w["layer_000.0.weight"], "bias": w["layer_000.0.bias"]})
        self.l1.load_state_dict({"weight": w["layer_001.0.weight"], "bias": w["layer_001.0.bias"]})
        self.out.load_state_dict({"weight": w["output_layer.weight"], "bias": w["output_layer.bias"]})
        self.register_buffer("mean_param", state["regressor.mean_param"].float())
        self.register_buffer("betas_idxs", state["betas_idxs"].long())

    @torch.no_grad()
    def forward(self, images):  # images: (B, 3, 256, 256) normalised RGB
        feat = self.backbone(images)["concat"]
        params = self.mean_param.expand(len(images), -1)
        for _ in range(3):  # IterativeRegression, 3 stages, MLP blocks are Linear+Dropout (no activation)
            params = params + self.out(self.l1(self.l0(torch.cat([feat, params], dim=1))))
        return params[:, self.betas_idxs]


class SmplxFromCheckpoint:
    """SMPL-X neutral (10 betas) built from the buffers stored in the SHAPY checkpoint."""

    def __init__(self, state: dict, device):
        g = lambda k: state[f"model.{k}"].to(device)  # noqa: E731
        self.v_template, self.shapedirs = g("v_template").float(), g("shapedirs").float()
        self.posedirs, self.J_regressor = g("posedirs").float(), g("J_regressor").float()
        self.parents, self.lbs_weights = g("parents").long(), g("lbs_weights").float()
        self.faces = state["model.faces_tensor"].long().numpy()
        self.device = device

    @torch.no_grad()
    def posed(self, betas: np.ndarray, body_pose_rotmats21: np.ndarray):
        from smplx.lbs import lbs
        rot = torch.eye(3, device=self.device).repeat(1, 55, 1, 1)
        rot[0, 1:22] = torch.as_tensor(body_pose_rotmats21, dtype=torch.float32, device=self.device)
        b = torch.as_tensor(betas, dtype=torch.float32, device=self.device)[None]
        verts, joints = lbs(b, rot, self.v_template, self.shapedirs, self.posedirs, self.J_regressor,
                            self.parents, self.lbs_weights, pose2rot=False)
        v, j = verts[0].cpu().numpy(), joints[0].cpu().numpy()
        floor = v[:, 1].min()
        v[:, 1] -= floor
        j[:, 1] -= floor
        return v, j


def _read_frames(video: Path, indices: list[int], display_hw: tuple[int, int]) -> dict[int, np.ndarray]:
    """Frames in display orientation; verified against the run's (display-oriented) mask size."""
    cap = cv2.VideoCapture(str(video))
    cap.set(cv2.CAP_PROP_ORIENTATION_AUTO, 0)  # rotate explicitly below, exactly once
    meta = int(round(cap.get(cv2.CAP_PROP_ORIENTATION_META) or 0)) % 360
    rotate = {90: cv2.ROTATE_90_CLOCKWISE, 180: cv2.ROTATE_180, 270: cv2.ROTATE_90_COUNTERCLOCKWISE}.get(meta)
    out = {}
    for idx in sorted(set(indices)):
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ok, frame = cap.read()
        if not ok:
            continue
        if rotate is not None:
            frame = cv2.rotate(frame, rotate)
        if frame.shape[:2] != tuple(display_hw):
            cap.release()
            raise RuntimeError(f"frame {idx}: decoded size {frame.shape[:2]} (rotation meta {meta}) does not match "
                               f"the run's display size {tuple(display_hw)}; refusing to crop a mis-oriented frame")
        out[idx] = frame
    cap.release()
    return out


def _crop(frame_bgr: np.ndarray, box) -> np.ndarray:
    x0, y0, x1, y1 = box
    size = SCALE * max(x1 - x0, y1 - y0)
    cx, cy = (x0 + x1) / 2, (y0 + y1) / 2
    s = CROP / size
    affine = np.array([[s, 0, CROP / 2 - s * cx], [0, s, CROP / 2 - s * cy]], np.float32)
    patch = cv2.warpAffine(frame_bgr, affine, (CROP, CROP), flags=cv2.INTER_LINEAR,
                           borderMode=cv2.BORDER_CONSTANT, borderValue=0)
    rgb = cv2.cvtColor(patch, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    return (rgb - MEAN) / STD


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("run_id")
    ap.add_argument("--height-cm", type=float, required=True)
    ap.add_argument("--out")
    args = ap.parse_args()
    height_cm = bm.validate_height_cm(args.height_cm)
    run_dir = ROOT / "outputs" / "reconstructions" / args.run_id
    status = json.loads((run_dir / "status.json").read_text(encoding="utf-8"))
    selected = json.loads((run_dir / "selected_frames.json").read_text(encoding="utf-8"))["selected"]
    fit = np.load(run_dir / "diagnostics" / "fit_inputs.npz", allow_pickle=False)
    video = next((ROOT / "data" / "captures" / status["video_id"]).glob("original.*"))

    mh, mw = (int(x) for x in fit["mask_shape"])
    frames = _read_frames(video, [s["source_frame_index"] for s in selected], (mh, mw))
    boxes, crops, used = [], [], []
    for i, s in enumerate(selected):
        frame = frames.get(s["source_frame_index"])
        if frame is None or not bool(fit["mask_usable"][i]):
            continue
        mask = np.unpackbits(fit["masks_packed"][i])[: mh * mw].reshape(mh, mw)  # same packing as the pipeline
        ys, xs = np.nonzero(mask)
        if len(xs) < 50:
            continue
        H, W = frame.shape[:2]
        sx, sy = W / mw, H / mh
        box = (xs.min() * sx, ys.min() * sy, (xs.max() + 1) * sx, (ys.max() + 1) * sy)
        boxes.append(box)
        crops.append(_crop(frame, box))
        used.append(s)
    if len(crops) < 3:
        print("Too few usable frames", file=sys.stderr)
        return 1

    device = "cuda" if torch.cuda.is_available() else "cpu"
    state = torch.load(CKPT, map_location="cpu", weights_only=True)["model"]
    net = ShapyShape(state).to(device).eval()
    batch = torch.as_tensor(np.stack(crops)).permute(0, 3, 1, 2).to(device)
    betas_each = np.concatenate([net(batch[k:k + 8]).cpu().numpy() for k in range(0, len(batch), 8)])
    betas, robust_std, outliers = smv.robust_beta_aggregate(betas_each)

    body = SmplxFromCheckpoint(state, device)
    v, j = body.posed(betas, smv.measurement_pose()[:21])
    meas = bm.measure_body(v, body.faces, j[:24])
    scaled = bm.scale_to_height(bm.strip_rings(meas), height_cm)
    current = json.loads((run_dir / "measurements_cm.json").read_text(encoding="utf-8")) \
        if (run_dir / "measurements_cm.json").is_file() else None
    if current and abs(current["height_cm"] - height_cm) > 1e-6:
        current = bm.scale_to_height(json.loads((run_dir / "measurements.json").read_text(encoding="utf-8")), height_cm)

    print(f"SHAPY on {len(crops)} frames of run {args.run_id[:8]} (outliers excluded: {list(outliers)})")
    print(f"per-frame beta spread (robust SD): {np.round(robust_std, 2).tolist()}")
    print(f"{'site':16s} {'current':>8s} {'SHAPY':>8s} {'diff':>7s}")
    rows = {}
    for site, m in scaled["measurements"].items():
        c = current["measurements"].get(site, {}).get("tape_circumference_cm") if current else None
        s_ = m.get("tape_circumference_cm")
        rows[site] = {"current_cm": c, "shapy_cm": s_}
        diff = f"{s_ - c:+7.1f}" if (c is not None and s_ is not None) else "      -"
        print(f"{site:16s} {c if c is not None else float('nan'):8.1f} {s_ if s_ is not None else float('nan'):8.1f} {diff}")
    out = {"run_id": args.run_id, "height_cm": height_cm, "frames_used": len(crops),
           "frame_indices": [s["source_frame_index"] for s in used], "betas_smplx": betas.tolist(),
           "betas_each": betas_each.tolist(), "beta_robust_std": robust_std.tolist(),
           "outlier_frames": [int(o) for o in outliers], "measurements": rows,
           "note": "SHAPY SMPL-X neutral shape in the pipeline's A-pose, scaled to the given height; "
                   "circumferences use the pipeline's own section definitions."}
    out_path = Path(args.out) if args.out else ROOT / "outputs" / "shapy" / f"{args.run_id}.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(out, indent=2), encoding="utf-8")
    # Save one crop sheet so the input to SHAPY can be checked by eye.
    sheet = np.concatenate([np.clip((c * STD + MEAN) * 255, 0, 255).astype(np.uint8) for c in crops[:8]], axis=1)
    cv2.imwrite(str(out_path.with_suffix(".crops.jpg")), cv2.cvtColor(sheet, cv2.COLOR_RGB2BGR))
    from backend.services.smpl_multiview import write_obj
    write_obj(out_path.with_suffix(".obj"), v * (height_cm / 100.0) / (v[:, 1].max() - v[:, 1].min()), body.faces,
              header=f"SHAPY SMPL-X neutral shape, measurement A-pose, scaled to {height_cm} cm (metres)")
    print(f"wrote {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
