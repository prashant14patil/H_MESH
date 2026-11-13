"""Opt-in HMR2 diagnostic output; no production inference changes."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import cv2
import numpy as np


def export_frame(engine, image_path, bbox_xyxy, output_dir, frame_id):
    """Run an authorized explicit-box diagnostic on an already-loaded engine.

    This is separate from the public inference workflow. Caller must use an
    exclusive output directory and ensure subject association and authorization.
    """
    import torch
    from PIL import Image
    from hmr2.datasets.vitdet_dataset import ViTDetDataset
    from hmr2.datasets.utils import gen_trans_from_patch_cv
    from hmr2.utils import recursive_to

    if not engine.is_ready:
        raise RuntimeError("HMR2 engine not ready")
    path = Path(image_path)
    data = path.read_bytes()
    import io
    image = np.asarray(Image.open(io.BytesIO(data)).convert("RGB"))
    height, width = image.shape[:2]
    box = np.asarray(bbox_xyxy, dtype=np.float32)
    if (box.shape != (4,) or not np.isfinite(box).all()
            or not (0 <= box[0] < box[2] <= width and 0 <= box[1] < box[3] <= height)):
        raise ValueError("Explicit subject box must lie within the source image")
    cv_image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    dataset = ViTDetDataset(engine.model_cfg, cv_image, box[None, :])
    batch = next(iter(torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=False, num_workers=0)))
    center = batch["box_center"][0].numpy()
    size = float(batch["box_size"][0])
    image_size = int(engine.model_cfg.MODEL.IMAGE_SIZE)
    affine = gen_trans_from_patch_cv(float(center[0]), float(center[1]), size, size,
                                     image_size, image_size, 1.0, 0.0)
    inverse = cv2.invertAffineTransform(affine)
    batch_device = recursive_to(batch, engine.device)
    with torch.no_grad():
        out = engine.model(batch_device)
    params = out["pred_smpl_params"]
    arrays = {
        "vertices": out["pred_vertices"][0].detach().cpu().numpy(),
        "faces": np.asarray(engine.smpl_faces),
        "betas": params["betas"][0].detach().cpu().numpy(),
        "body_pose_rotmat": params["body_pose"][0].detach().cpu().numpy(),
        "global_orient_rotmat": params["global_orient"][0].detach().cpu().numpy(),
        "pred_cam": out["pred_cam"][0].detach().cpu().numpy(),
        "pred_cam_t": out["pred_cam_t"][0].detach().cpu().numpy(),
        "focal_length_crop": out["focal_length"][0].detach().cpu().numpy(),
        "image_to_crop_affine": affine,
        "crop_to_image_affine": inverse,
    }
    for key, value in arrays.items():
        if not np.issubdtype(value.dtype, np.number) or not np.isfinite(value).all():
            raise ValueError(f"Invalid diagnostic output: {key}")
    if arrays["betas"].size != 10 or arrays["body_pose_rotmat"].size != 23 * 9 or arrays["global_orient_rotmat"].size != 9:
        raise ValueError("Unexpected SMPL parameter dimensions")
    if arrays["pred_cam"].size != 3 or arrays["pred_cam_t"].size != 3 or arrays["focal_length_crop"].size != 2:
        raise ValueError("Unexpected HMR2 camera dimensions")
    for key in ("body_pose_rotmat", "global_orient_rotmat"):
        rotations = arrays[key].reshape(-1, 3, 3)
        if not np.allclose(rotations @ rotations.transpose(0, 2, 1), np.eye(3), atol=1e-3) or not np.allclose(np.linalg.det(rotations), 1, atol=1e-3):
            raise ValueError(f"Invalid rotation matrices: {key}")
    if not np.allclose(cv2.invertAffineTransform(affine), inverse) or abs(np.linalg.det(affine[:, :2])) < 1e-12:
        raise ValueError("Invalid crop affine transform")
    if arrays["vertices"].ndim != 2 or arrays["vertices"].shape[1] != 3:
        raise ValueError("Invalid vertex shape")
    if arrays["faces"].ndim != 2 or arrays["faces"].shape[1] != 3:
        raise ValueError("Invalid face shape")
    if arrays["faces"].min() < 0 or arrays["faces"].max() >= len(arrays["vertices"]):
        raise ValueError("Invalid mesh face indices")
    directory = Path(output_dir)
    directory.mkdir(parents=True, exist_ok=False)
    # Reconstruct the normalized model input crop; it includes dataset antialiasing.
    crop = batch["img"][0].numpy().transpose(1, 2, 0)
    mean = np.asarray(engine.model_cfg.MODEL.IMAGE_MEAN)
    std = np.asarray(engine.model_cfg.MODEL.IMAGE_STD)
    crop = np.clip((crop * std + mean) * 255, 0, 255).astype(np.uint8)
    if not cv2.imwrite(str(directory / "model_crop.png"), cv2.cvtColor(crop, cv2.COLOR_RGB2BGR)):
        raise OSError("Unable to save model input crop")
    np.savez_compressed(directory / "diagnostics.npz", **arrays)
    metadata = {
        "schema_version": 1, "frame_id": frame_id, "source_filename": path.name,
        "source_sha256": hashlib.sha256(data).hexdigest(), "image_size_wh": [width, height],
        "bbox_xyxy_pixels": box.tolist(), "bbox_source": "explicit_user_selection",
        "crop_center_xy": center.tolist(), "crop_size_pixels": size,
        "model_input_size": image_size, "device": engine.device,
        "arrays": {key: {"shape": list(value.shape), "dtype": str(value.dtype)} for key, value in arrays.items()},
        "conventions": {
            "crop_affine": "original-image pixel coordinates to square model crop pixels",
            "camera": "HMR2 crop camera predictions, not measured calibration",
            "pose": "HMR2 predicted rotation matrices; model convention",
            "vertices": "HMR2 model output coordinates, not aligned or metrically validated",
            "faces": "zero-based indices",
        },
    }
    (directory / "metadata.json").write_text(json.dumps(metadata, indent=2, allow_nan=False), encoding="utf-8")
    return metadata
