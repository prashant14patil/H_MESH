"""
backend/services/inference_engine.py
-------------------------------------
Local GPU inference engine for HMR2 / 4D-Humans body mesh reconstruction.

This module REPLACES the Google Colab + Flask + ngrok pipeline with
direct, in-process model execution on the local machine's GPU (CUDA)
or CPU if no GPU is available.

Responsibilities (merged from the old colab/cells/*.py notebook cells):
    - Load the HMR2 checkpoint once at process startup (singleton).
    - Optionally load a ViTDet / detectron2 person detector; if that
      is unavailable (e.g. on Windows where detectron2 is hard to
      build), silently fall back to treating the whole image as the
      person bounding box.
    - Run inference on an uploaded image and export an OBJ mesh.
    - Track job status in an in-memory dict, exactly like the Colab
      Flask server did, but without any HTTP hop — it all happens in
      this Python process.

Environment variables (see .env.example):
    HMR2_REPO_PATH     – Path to the cloned 4D-Humans repo.
    HMR2_CHECKPOINT    – Path to the HMR2 .ckpt file.
    DEVICE             – "cuda", "cpu", or "auto" (default: auto).
    MESH_OUTPUT_DIR    – Where generated .obj files are written.
    USE_DETECTRON2     – "1"/"true" to attempt loading the detector,
                         "0"/"false" to always use the full-image
                         fallback (default: true — but falls back
                         automatically on any failure).

Public surface:
    InferenceEngine          – Singleton-style class, use get_engine().
    get_engine()              – Returns the process-wide InferenceEngine.
    EngineNotReadyError       – Raised when the model failed to load.
"""

from __future__ import annotations

import io
import logging
import os
import sys
import threading
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
_HMR2_REPO_PATH = os.getenv("HMR2_REPO_PATH", "models/4D-Humans")
_CHECKPOINT_PATH = os.getenv("HMR2_CHECKPOINT", "")
_DEVICE_PREF = os.getenv("DEVICE", "auto").lower()
_OUTPUT_DIR = Path(os.getenv("MESH_OUTPUT_DIR", "outputs/meshes"))
_USE_DETECTRON2 = os.getenv("USE_DETECTRON2", "true").lower() in ("1", "true", "yes")

_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


class EngineNotReadyError(RuntimeError):
    """Raised when the inference engine could not be initialised."""


# ---------------------------------------------------------------------------
# Engine
# ---------------------------------------------------------------------------
class InferenceEngine:
    """
    Loads the HMR2 model once (in a background thread) and runs inference
    for each submitted job.

    This mirrors the logic previously spread across the Colab notebook
    cells (cell1_deps, cell2_paths, cell3_model, cell4_detector,
    cell5_inference) but executes locally, with no Flask/ngrok hop.
    """

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._ready = False
        self._loading = False
        self._load_error: str | None = None

        self.device: str = "cpu"
        self.model = None
        self.model_cfg = None
        self.smpl_faces = None
        self.detector = None  # None => full-image fallback

        # job_id -> {"status": ..., "mesh_path": ..., "metadata": ..., "detail": ...}
        self.jobs: dict[str, dict[str, Any]] = {}

        # Construction returns immediately; the (potentially slow) model
        # load happens in a background thread so that creating/fetching
        # the singleton (e.g. from a health-check request) never blocks.
        self.load_async()

    def load_async(self) -> None:
        """Kick off model loading in a background thread (idempotent)."""
        with self._lock:
            if self._ready or self._loading:
                return
            self._loading = True
        threading.Thread(target=self._load, daemon=True).start()

    # ------------------------------------------------------------------
    # Loading
    # ------------------------------------------------------------------
    def _load(self) -> None:
        try:
            # hmr2.utils.renderer forces PYOPENGL_PLATFORM="egl" at import
            # time unless this var is already set. EGL is not available on
            # a headless Windows machine, which would crash the import even
            # though this engine never actually uses hmr2's renderer (we
            # export OBJ files manually in _export_obj). Pre-set it to the
            # native Win32 OpenGL platform (confirmed available on this
            # machine) to avoid that forced EGL import failure.
            os.environ.setdefault("PYOPENGL_PLATFORM", "win32")

            # hmr2.configs uses os.environ["HOME"] to build its cache dir
            # path, which does not exist on Windows (Windows provides
            # USERPROFILE instead). Populate it from USERPROFILE/HOMEDRIVE+
            # HOMEPATH so the import doesn't crash with a TypeError.
            if not os.environ.get("HOME"):
                os.environ["HOME"] = os.environ.get("USERPROFILE") or (
                    os.environ.get("HOMEDRIVE", "") + os.environ.get("HOMEPATH", "")
                ) or str(Path.home())

            import torch

            if _DEVICE_PREF == "cuda" or (_DEVICE_PREF == "auto" and torch.cuda.is_available()):
                self.device = "cuda" if torch.cuda.is_available() else "cpu"
            else:
                self.device = "cpu"

            if self.device == "cpu":
                logger.warning(
                    "No CUDA GPU detected (or DEVICE forced to cpu). "
                    "Inference will run on CPU and will be much slower."
                )
            else:
                logger.info("GPU detected: %s", torch.cuda.get_device_name(0))

            repo = str(Path(_HMR2_REPO_PATH).resolve())
            if repo not in sys.path:
                sys.path.insert(0, repo)

            if not _CHECKPOINT_PATH or not os.path.isfile(_CHECKPOINT_PATH):
                raise EngineNotReadyError(
                    f"HMR2 checkpoint not found at '{_CHECKPOINT_PATH}'. "
                    "Set HMR2_CHECKPOINT in .env, or run "
                    "scripts/download_models.py to fetch it automatically."
                )

            # Patch torch.load: the official HMR2 checkpoint contains
            # Python/Lightning objects, not pure tensors, so PyTorch>=2.0's
            # strict weights_only=True default rejects it. This is safe
            # because we load a trusted, official checkpoint file.
            _orig_torch_load = torch.load

            def _patched_torch_load(f, *args, **kwargs):
                kwargs.setdefault("weights_only", False)
                return _orig_torch_load(f, *args, **kwargs)

            torch.load = _patched_torch_load

            # hmr2.models.check_smpl_exists() -> convert_pkl() writes the
            # converted SMPL file to {CACHE_DIR_4DHUMANS}/data/smpl/SMPL_NEUTRAL.pkl
            # but never creates that parent directory first, so open(new_pkl, "wb")
            # raises FileNotFoundError on a fresh machine (verified by traceback
            # during Phase A testing on this machine). This directory lives
            # entirely under the user's HOME cache folder, outside the repo and
            # outside vendor code, so creating it here (not editing vendor code)
            # is the minimal, targeted fix.
            from hmr2.configs import CACHE_DIR_4DHUMANS

            os.makedirs(os.path.join(CACHE_DIR_4DHUMANS, "data", "smpl"), exist_ok=True)

            # hmr2.configs.get_config(update_cachedir=True) rewrites
            # cfg.SMPL.MEAN_PARAMS / JOINT_REGRESSOR_EXTRA (relative paths
            # from model_config.yaml, e.g. "data/smpl_mean_params.npz") to be
            # relative to CACHE_DIR_4DHUMANS, under the assumption that the
            # official hmr2_data.tar.gz bundle was extracted there. In this
            # project's setup, those auxiliary files instead live in the
            # cloned repo at HMR2_REPO_PATH/data/. Mirror them into the cache
            # dir (copy-if-missing, never overwrite) so the vendor code finds
            # them where it expects, without having to edit vendor code or
            # the model_config.yaml it reads. Verified missing via traceback
            # during Phase A testing on this machine.
            import shutil

            _repo_data_dir = Path(_HMR2_REPO_PATH) / "data"
            _cache_data_dir = Path(CACHE_DIR_4DHUMANS) / "data"
            _cache_data_dir.mkdir(parents=True, exist_ok=True)
            for _fname in ("smpl_mean_params.npz", "SMPL_to_J19.pkl"):
                _src = _repo_data_dir / _fname
                _dst = _cache_data_dir / _fname
                if _src.is_file() and not _dst.exists():
                    shutil.copy2(_src, _dst)
                    logger.info("Copied %s into HMR2 cache dir (%s)", _fname, _dst)

            # HMR2.__init__ (models/hmr2.py) unconditionally constructs a
            # pyrender.OffscreenRenderer-backed MeshRenderer/SkeletonRenderer
            # (init_renderer=True by default) even though this engine never
            # calls into them — we export OBJ files manually in _export_obj.
            # On this headless Windows machine, pyrender has no working
            # offscreen backend (EGL unavailable, win32 unsupported by
            # pyrender's offscreen.py), so model construction crashes with
            # "Unsupported PyOpenGL platform: win32" before inference can
            # ever run. Verified by traceback during Phase A testing.
            #
            # Fix: flip only the *default value* of the init_renderer
            # parameter (True -> False), leaving the __init__ signature
            # byte-identical. A signature-changing wrapper (e.g. adding
            # **kwargs) was tried first and broke pytorch_lightning's
            # HyperparametersMixin.save_hyperparameters(), which uses frame
            # introspection (collect_init_args) that expects the exact
            # original parameter names in the call frame — it raised
            # KeyError('kwargs'). Mutating __defaults__ in place avoids
            # that entirely and has zero effect on inference output since
            # the renderer is never used on the inference path
            # (forward_step() does not reference self.renderer /
            # self.mesh_renderer).
            import hmr2.models.hmr2 as _hmr2_module

            _hmr2_module.HMR2.__init__.__defaults__ = (False,)

            from hmr2.models import load_hmr2

            logger.info("Loading HMR2 checkpoint: %s", _CHECKPOINT_PATH)
            self.model, self.model_cfg = load_hmr2(_CHECKPOINT_PATH)
            self.model = self.model.to(self.device).eval()
            logger.info("HMR2 model loaded on %s", self.device)

            self.smpl_faces = self._locate_smpl_faces(self.model)
            logger.info("SMPL faces ready: %d triangles", self.smpl_faces.shape[0])

            self.detector = self._load_detector() if _USE_DETECTRON2 else None

            self._ready = True
            self._load_error = None

        except Exception as exc:  # noqa: BLE001 - we want to capture any load failure
            logger.exception("Failed to initialise InferenceEngine")
            self._ready = False
            self._load_error = str(exc)
        finally:
            self._loading = False

    @staticmethod
    def _locate_smpl_faces(model) -> Any:
        """SMPL faces attribute path varies across 4D-Humans versions."""
        for attr in ("smpl", "smpl_head", "body_model", "smpl_model"):
            obj = getattr(model, attr, None)
            if obj is None:
                continue
            faces = getattr(obj, "faces", None)
            if faces is not None:
                return faces
        faces = getattr(model, "faces", None)
        if faces is not None:
            return faces
        avail = [a for a in dir(model) if not a.startswith("_")]
        raise AttributeError(f"Cannot find SMPL faces. Model attrs: {avail}")

    def _load_detector(self):
        """Attempt to load the ViTDet/detectron2 person detector.
        Returns None (full-image fallback) on any failure."""
        try:
            import hmr2 as _h
            from hmr2.utils.utils_detectron2 import DefaultPredictor_Lazy
            from detectron2.config import LazyConfig

            cfg_f = os.path.join(
                os.path.dirname(_h.__file__),
                "configs",
                "cascade_mask_rcnn_vitdet_h_75ep.py",
            )
            if not os.path.isfile(cfg_f):
                logger.warning(
                    "Detectron2 config not found at %s — using full-image fallback.",
                    cfg_f,
                )
                return None

            detector = DefaultPredictor_Lazy(LazyConfig.load(cfg_f))
            logger.info("ViTDet person detector loaded.")
            return detector
        except Exception as exc:  # noqa: BLE001
            logger.warning(
                "detectron2 unavailable (%s) — using full-image fallback "
                "(whole image treated as the person bounding box).",
                exc,
            )
            return None

    # ------------------------------------------------------------------
    # Public status
    # ------------------------------------------------------------------
    @property
    def is_ready(self) -> bool:
        return self._ready

    @property
    def load_error(self) -> str | None:
        return self._load_error

    # ------------------------------------------------------------------
    # Detection helper
    # ------------------------------------------------------------------
    def _detect_person_box(self, img_rgb):
        import numpy as np
        import cv2

        if self.detector is not None:
            try:
                out = self.detector(cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR))
                inst = out["instances"]
                mask = inst.pred_classes == 0  # class 0 = person
                if mask.sum() > 0:
                    boxes = inst.pred_boxes.tensor[mask].cpu().numpy()
                    areas = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
                    return boxes[np.argmax(areas)][None]
            except Exception as exc:  # noqa: BLE001
                logger.warning("Detection error (%s) — falling back to full image.", exc)

        h, w = img_rgb.shape[:2]
        return np.array([[0, 0, w, h]], dtype=np.float32)

    # ------------------------------------------------------------------
    # Core inference
    # ------------------------------------------------------------------
    def _run_inference(self, image_bytes: bytes, job_id: str) -> dict[str, Any]:
        import numpy as np
        import cv2
        import torch
        from PIL import Image as PILImage
        from hmr2.utils import recursive_to
        from hmr2.datasets.vitdet_dataset import ViTDetDataset

        img_np = np.array(PILImage.open(io.BytesIO(image_bytes)).convert("RGB"))
        img_cv2 = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)

        box = self._detect_person_box(img_np)

        ds = ViTDetDataset(self.model_cfg, img_cv2, box)
        dl = torch.utils.data.DataLoader(ds, batch_size=1, shuffle=False, num_workers=0)

        verts = faces = None
        for batch in dl:
            batch = recursive_to(batch, self.device)
            with torch.no_grad():
                out = self.model(batch)
            verts = out["pred_vertices"][0].cpu().numpy()  # (6890, 3)
            faces = self.smpl_faces.astype(np.int64)        # (13776, 3)
            break  # single-person inference

        if verts is None:
            raise ValueError("No person detected in image.")

        mesh_path = self._export_obj(verts, faces, job_id)

        return {
            "status": "completed",
            "mesh_path": str(mesh_path),
            "metadata": {
                "vertices": int(verts.shape[0]),
                "faces": int(faces.shape[0]),
                "body_model": "SMPL",
            },
        }

    def _export_obj(self, verts, faces, job_id: str) -> Path:
        path = _OUTPUT_DIR / f"{job_id}.obj"
        with open(path, "w") as fh:
            fh.write(f"# HMR2 mesh -- job {job_id} (local GPU inference)\n")
            for v in verts:
                fh.write(f"v {v[0]:.6f} {v[1]:.6f} {v[2]:.6f}\n")
            for face in faces:
                fh.write(f"f {int(face[0]) + 1} {int(face[1]) + 1} {int(face[2]) + 1}\n")
        return path

    # ------------------------------------------------------------------
    # Job management (mirrors the old Colab Flask /infer, /status, /mesh)
    # ------------------------------------------------------------------
    def submit_job(self, job_id: str, image_bytes: bytes) -> None:
        """Queue a job and run it in a background thread."""
        if self._loading:
            # Model still loading in background — wait is handled by caller
            # via EngineNotReadyError; the frontend will poll health/status.
            raise EngineNotReadyError(
                "Inference engine is still loading the model. Please wait "
                "and retry in a few seconds."
            )
        if not self._ready:
            raise EngineNotReadyError(
                self._load_error or "Inference engine is not ready."
            )

        self.jobs[job_id] = {"status": "queued"}

        def _worker():
            self.jobs[job_id] = {"status": "processing"}
            try:
                result = self._run_inference(image_bytes, job_id)
                self.jobs[job_id] = result
            except Exception as exc:  # noqa: BLE001
                logger.exception("Job %s failed", job_id)
                self.jobs[job_id] = {"status": "failed", "detail": str(exc)}

        threading.Thread(target=_worker, daemon=True).start()

    def get_job(self, job_id: str) -> dict[str, Any] | None:
        return self.jobs.get(job_id)


# ---------------------------------------------------------------------------
# Process-wide singleton
# ---------------------------------------------------------------------------
_engine: InferenceEngine | None = None
_engine_lock = threading.Lock()


def get_engine() -> InferenceEngine:
    """Return the process-wide InferenceEngine, creating it on first call.

    Construction is non-blocking: the model loads in a background thread,
    so calling this from a health-check or startup hook never hangs the
    FastAPI event loop.
    """
    global _engine
    if _engine is None:
        with _engine_lock:
            if _engine is None:
                _engine = InferenceEngine()
    return _engine
