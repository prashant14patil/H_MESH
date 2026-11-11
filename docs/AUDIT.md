# Tanita Human Digital Twin POC — Repository Audit (Phase 1)

**Audit date:** 2026-11-11
**Scope:** Full repository inspection — backend, frontend, scripts, dependencies, environment, and local hardware compatibility.

---

## (a) Repository Audit Summary

### Backend (`backend/`)
| File | Status | Notes |
|---|---|---|
| `main.py` | ✅ Implemented | FastAPI app, CORS, `/health`, background model-load thread. Clean. |
| `routes/inference.py` | ✅ Implemented | `POST /submit`, `GET /status/{id}`, `GET /mesh/{id}`. Correct contract, no path leakage, proper HTTP codes. |
| `services/inference_engine.py` | ⚠️ Implemented, **never successfully run** | Singleton engine, async model load, CPU/CUDA device selection, optional detectron2 fallback, in-memory job dict, OBJ export. Logic is sound on paper but has never executed end-to-end in this environment (see Critical Bugs). |
| `utils/validators.py` | ✅ Implemented | Extension + magic-byte + size validation. Does **not** validate EXIF orientation, resolution, or aspect ratio (needed for Phase 2). |

### Frontend (`frontend/`)
| File | Status | Notes |
|---|---|---|
| `src/App.jsx` | ✅ Implemented | Full state machine (idle→uploading→queued→processing→completed/failed), capped polling, cleanup on unmount. |
| `src/components/ImageUploader.jsx` | ✅ Implemented | Drag/drop, client-side type/size validation, preview. No body/orientation pre-check. |
| `src/components/MeshViewer.jsx` | ✅ Implemented | R3F + drei OrbitControls, OBJ loading, lighting, reset-camera. **Missing:** wireframe toggle, floor/grid, GLB/glTF export, mesh-quality indicator. |
| `src/components/ProcessingStatus.jsx` | ✅ Implemented | Visual pipeline indicator, progress bar, error display. |
| `src/components/ResultPanel.jsx` | ✅ Implemented | Metadata + OBJ download. Correct disclaimer already present. |
| `src/services/api.js` | ✅ Implemented | Clean fetch wrapper, timeouts, error propagation. |
| `frontend/app.py` (Streamlit) | ⚠️ Legacy/parallel UI | Duplicate frontend (Streamlit + Plotly). Not referenced by README quick-start. Not wired into current workflow — flagged legacy. |

### Scripts (`scripts/`)
| File | Status | Notes |
|---|---|---|
| `download_models.py` | ⚠️ Implemented, untested here | Clones 4D-Humans, downloads `hmr2_data.tar.gz` from one hardcoded URL, no checksum, no resume/retry. Single point of failure. |
| `test_integration.py` | ✅ Implemented | Good smoke-test coverage, but requires a live backend + loaded model to run. |

### Configuration
| File | Status | Notes |
|---|---|---|
| `.env.example` | ✅ Present, correct | All required variables documented. |
| `.env` | ❌ **MISSING** | `HMR2_CHECKPOINT` unset — engine cannot load. |
| `requirements.txt` | ⚠️ Partially outdated | Pins versions not necessarily installable on Python 3.12 (see compatibility section). |
| `.gitignore` | ✅ Correct | `.env`, `models/`, `outputs/`, `node_modules/` excluded. |

---

## (b) Current Architecture Diagram

```
┌─────────────────────────┐        HTTP (fetch)         ┌──────────────────────────────┐
│  React 18 + Vite         │ ───────────────────────────▶│  FastAPI (uvicorn)            │
│  (localhost:5173)        │                              │  (localhost:8000)             │
│                          │◀──────────────────────────── │                                │
│  ImageUploader            │   JSON job_id/status/mesh    │  routes/inference.py          │
│  MeshViewer (R3F)         │                              │  services/inference_engine.py │
│  ResultPanel              │                              │                                │
└─────────────────────────┘                              │  ┌──────────────────────────┐ │
                                                            │  │ InferenceEngine singleton │ │
  (Legacy, parallel, unused)                                │  │ - loads HMR2 checkpoint   │ │
┌─────────────────────────┐        HTTP (requests)         │  │ - optional detectron2     │ │
│  Streamlit app.py         │ ───────────────────────────▶ │  │ - runs CUDA/CPU           │ │
│  (localhost:8501)         │                              │  │ - writes .obj to disk     │ │
└─────────────────────────┘                              │  └──────────────────────────┘ │
                                                            └──────────────────────────────┘
                                                                           │
                                                                           ▼
                                                              outputs/meshes/{job_id}.obj
                                                              models/4D-Humans/ (not present)

NO persistence layer — jobs live only in an in-memory Python dict, lost on every
server restart. NO Person/Session/Measurement/BCM data model exists yet.
```

---

## (c) Implemented vs Missing Feature Matrix

| Phase | Feature | Status |
|---|---|---|
| 1 | Codebase audit | ✅ This document |
| 2 | HMR2 reconstruction baseline | ⚠️ Code exists, never successfully executed — missing deps + missing model weights |
| 2 | CPU/CUDA graceful fallback | ✅ Implemented correctly |
| 2 | No-person / multi-person / partial-body handling | ❌ Missing — only full-image fallback bbox |
| 2 | EXIF orientation handling | ❌ Missing |
| 3 | Interactive 3D viewer (rotate/zoom/pan/reset) | ✅ Implemented |
| 3 | Wireframe toggle, floor/grid, GLB/glTF export | ❌ Missing |
| 4 | Body measurement engine | ❌ Not started |
| 5 | Tanita BCM/BIA ingestion | ❌ Not started |
| 6 | Persistent data model | ❌ Not started — no DB at all |
| 7 | Dashboard | ❌ Not started |
| 8 | Segmental visualization | ❌ Not started |
| 9 | Scoring/AI research module | ❌ Not started |
| 10 | Automated tests | ⚠️ One integration script exists; no unit tests; no CI |

---

## (d) Critical Bugs & Technical Risks

1. **BLOCKING — `.env` missing.** `HMR2_CHECKPOINT`/`HMR2_REPO_PATH` unset → `InferenceEngine._load()` fails immediately with `EngineNotReadyError` on first submit. Verified by inspection of `inference_engine.py` lines 131-139.
2. **BLOCKING — Required Python packages not installed.** Verified via `pip list`: `smplx`, `hmr2`, `yacs`, `pytorch-lightning`, `chumpy`, `webdataset` are **all absent** from the current environment. Only `einops`, `scikit-image`, `torch`(+cu121), `opencv`, `numpy`, `Pillow`, `fastapi`, `uvicorn` are present. The inference engine cannot import `hmr2.models` at all right now.
3. **BLOCKING — `models/4D-Humans` repo not cloned, no checkpoint downloaded.** `scripts/download_models.py` has never been run in this environment; `models/` and `outputs/` directories do not exist.
4. **requirements.txt version risk on Python 3.12.** The repo recommends Python 3.10/3.11, but the active environment is **Python 3.12.10**. Packages pinned in requirements.txt (`pytorch-lightning==2.2.5`, `smplx==0.1.28`, `chumpy==0.70`, `webdataset==0.2.100`) were released before/around Python 3.12's stabilization and may have C-extension or `distutils`-removal incompatibilities (distutils was removed in 3.12). `chumpy` in particular is known to break on newer NumPy (`numpy==1.26.4` pin vs. installed `numpy 2.2.6` — a **major version mismatch**, since NumPy 2.x removed APIs chumpy depends on).
5. **In-memory job store.** `InferenceEngine.jobs` is a plain dict — all job history is lost on every backend restart. No persistence, no multi-process safety (won't work if uvicorn is run with `--workers > 1`).
6. **No malware/size-bomb protection beyond size check.** `validate_image` checks magic bytes + size, but does not guard against decompression bombs (e.g., a small PNG that decodes to a huge bitmap) before passing to PIL/`cv2`.
7. **Single-person assumption with silent fallback.** If detectron2 isn't available (likely on this Windows machine), the *entire image* is treated as the person bounding box. A photo with no person, multiple people, or a cropped body will silently produce a garbage/meaningless mesh rather than raising a clear validation error — directly contradicts Phase 2 requirements.
8. **CORS allowlist hardcoded** to only localhost ports — fine for POC, flagged for production.
9. **Streamlit `frontend/app.py`** duplicates functionality and could drift out of sync with the backend contract (e.g., it already doesn't handle the `failed`/`error` distinction identically to the React app). Recommend marking explicitly "legacy/experimental, not part of primary workflow" in README.
10. **No logging persistence / structured error envelope.** Errors are logged via Python `logging` to stdout only; no log file, no correlation ID beyond `job_id`.

---

## (e) Hardware / Dependency Compatibility Assessment

**Actual machine used for this audit** (not the one described in the task's hardware constraints — see note below):
- OS: Windows 11
- Python: 3.12.10
- GPU: **NVIDIA GeForce RTX 4070 Laptop GPU** — CUDA available, `torch==2.5.1+cu121` already installed and functional (`torch.cuda.is_available() == True`).

> **Note:** The task description mentions a target deployment machine with an Intel Core 5 220U, 32 GB RAM, Intel integrated graphics (no NVIDIA GPU), and Python 3.12 pending approval. The *current dev/audit machine* has a discrete NVIDIA GPU and Python 3.12 already active. This plan accounts for **both**: CPU-only compatibility is preserved in code (already implemented in `inference_engine.py`'s device-selection logic), and the integration work in this session uses the available CUDA GPU to validate the pipeline faster. Documentation will cover both paths.

| Dependency | Pinned version | Installed version | Python 3.12 risk |
|---|---|---|---|
| torch / torchvision | unpinned (install via CUDA selector) | 2.5.1+cu121 / 0.20.1+cu121 | ✅ Compatible |
| numpy | ==1.26.4 | 2.2.6 (mismatch) | ⚠️ NumPy 2.x may break `chumpy`, older `smplx` releases |
| pytorch-lightning | ==2.2.5 | **not installed** | Needs install; 2.2.5 supports Python 3.12 |
| smplx | ==0.1.28 | **not installed** | Pure Python, should install fine |
| yacs | ==0.1.8 | **not installed** | Pure Python, fine |
| webdataset | ==0.2.100 | **not installed** | Pure Python, fine |
| einops | ==0.8.0 | 0.8.1 installed (close enough) | ✅ Fine |
| chumpy | ==0.70 | **not installed** | ⚠️ **High risk** — chumpy hasn't been updated since ~2020, uses deprecated `numpy` APIs (`np.bool`, `np.int`, `np.float` aliases removed in NumPy ≥1.24, and `distutils` removed in Python 3.12 which some chumpy install paths may touch). This is the single highest-risk dependency for the whole pipeline. |
| scikit-image | ==0.24.0 | 0.25.2 installed | ✅ Close enough |
| detectron2 | optional, not pinned | **not installed** | Expected — Windows build is notoriously hard; code already has automatic fallback, which is the correct approach. |
| hmr2 (4D-Humans repo) | N/A — not a pip package | **repo not cloned** | Must run `scripts/download_models.py` |

**Conclusion:** The CPU/CUDA device-selection architecture in `inference_engine.py` is already correct and future-proof for the target low-spec Intel-integrated-graphics machine — no code changes needed there. The blocking issue is **dependency installation**, not device support. `chumpy==0.70` pinned against `numpy==1.26.4` is the main compatibility risk; this audit recommends pinning numpy to `<2.0` for this virtualenv specifically (isolated from any other NumPy 2.x projects on the machine) and installing packages in the documented order (torch first, then requirements.txt).

---

## (f) Prioritized Implementation Plan

| Priority | Item | Phase |
|---|---|---|
| P0 | Create `.env` from `.env.example`, fill in paths | 2 |
| P0 | Install missing Python deps (`pytorch-lightning`, `smplx`, `yacs`, `chumpy`, `webdataset`) with NumPy pinned `<2.0` in this project's environment | 2 |
| P0 | Run `scripts/download_models.py` to clone 4D-Humans + fetch checkpoint | 2 |
| P0 | Validate end-to-end: submit a real photo → confirm OBJ mesh generated and rendered in the viewer | 2, 3 |
| P1 | Add image sanity checks: min resolution, aspect ratio, EXIF orientation auto-correct, decompression-bomb guard | 2 |
| P1 | Add explicit no-person / multi-person / partial-body error responses (not silent full-image fallback when detection is available) | 2 |
| P1 | Add wireframe toggle, floor/grid, GLB/glTF export to MeshViewer | 3 |
| P2 | Introduce SQLite persistence layer + data model (Person, CaptureSession, Image, Reconstruction, AvatarVersion) | 6 |
| P2 | Body measurement module (height calibration, landmark-based circumference estimates) with confidence/limitations labelling | 4 |
| P2 | BCM/BIA manual entry + CSV/JSON import module, linked to Person + timestamp | 5 |
| P3 | Dashboard UI (profile, measurement panel, BCM panel, timeline) | 7 |
| P3 | Segmental visualization mapping (labelled as estimated/mapped, not derived) | 8 |
| P4 | Scoring/AI research module (experimental, clearly labelled) | 9 |
| Ongoing | Unit/integration tests, structured logging, README updates | 10 |

---

## (g) Files Requiring Modification or Creation (this session)

**To be fixed now (P0, Phase 2 reliability):**
- `.env` — create from `.env.example` (not committed, local only)
- `requirements.txt` — pin `numpy<2.0` explicitly, document install order
- `backend/utils/validators.py` — add resolution/aspect-ratio/decompression-bomb checks
- `backend/services/inference_engine.py` — improve error reporting for no-person detection; add EXIF orientation correction
- `scripts/download_models.py` — add checksum/retry notes (deferred if time-limited)

**To be created (Phase 6+, subsequent sessions):**
- `backend/db/` — SQLite models + migrations (Person, CaptureSession, Reconstruction, BodyMeasurement, BCMRecord, TimelineEvent)
- `backend/routes/measurements.py`, `backend/routes/bcm.py`, `backend/routes/persons.py`
- `backend/services/measurement_engine.py`
- `frontend/src/components/Dashboard/*`, `BCMPanel.jsx`, `TimelineComparison.jsx`

This audit will be updated as each phase is completed.

---

## (h) Environment Setup Session Log — 2026-11-11 (Phase A execution)

This section documents the exact sequence of fixes applied to get the
local HMR2 pipeline past dependency/import errors, for reproducibility.
See `README.md` "Quick start" for the consolidated step-by-step version.

### Actions taken, in order

1. **`.env` fixed:** `HMR2_CHECKPOINT` was empty; set to the checkpoint
   found on disk at
   `models/4D-Humans/logs/train/multiruns/hmr2/0/checkpoints/epoch=35-step=1000000.ckpt`.

2. **Global environment install attempted, then reverted.** Installing
   `pytorch-lightning`, `smplx`, `chumpy`, `yacs`, `webdataset` directly
   into the global Python 3.12 site-packages downgraded `numpy`
   2.2.6→1.26.4 and reinstalled `opencv-python`, which broke version
   constraints for unrelated global tools (`inference-gpu`, `ultralytics`,
   `trackers`, `xformers`, `langchain`). Additionally, importing `hmr2`
   pulled in `pytorch_lightning` → `torchmetrics` → `transformers`, which
   requires `tokenizers>=0.21,<0.22`, conflicting with the global
   `tokenizers==0.19.1` used by other tools.
   **Resolution:** uninstalled all project packages from the global
   environment and restored original versions
   (`numpy==2.2.6`, `einops==0.8.1`, `scikit-image==0.25.2`,
   `opencv-contrib-python==4.10.0.84`, `opencv-contrib-python-headless==4.12.0.88`,
   `opencv-python-headless==4.11.0.86`).

3. **Created isolated `.venv`** for this project
   (`python -m venv .venv`). All subsequent installs happened here only.

4. **Installed PyTorch + CUDA** (already present in `.venv` from a prior
   session): `torch==2.5.1+cu121`, `torchvision==0.20.1+cu121`,
   confirmed `torch.cuda.is_available() == True` on the RTX 4070 Laptop GPU.

5. **Installed `requirements.txt`** — all pinned versions resolved
   cleanly in the clean venv (fastapi, uvicorn, numpy<2.0, opencv-python
   4.10.0.84, pytorch-lightning 2.2.5, smplx 0.1.28, yacs 0.1.8,
   webdataset 0.2.100, einops 0.8.0, chumpy 0.70, scikit-image 0.24.0).

6. **Installed the local `hmr2` package**
   (`pip install -e models/4D-Humans --no-deps`) — must use `--no-deps`,
   otherwise pip resolves `hmr2`'s `setup.py` install_requires list
   itself and silently upgrades numpy past the `<2.0` ceiling.

7. **Installed `hmr2`'s remaining transitive import-time dependencies**
   not covered by `requirements.txt`: `pyrender`, `gdown`, `dill`,
   `pandas`, `timm`. This step upgraded `numpy` to 2.5.4 and `scipy` to
   1.18.1 as a side effect (via `pyrender`'s own dependency resolution).
   **Re-pinned immediately:** `numpy==1.26.4`, `scipy==1.12.0`.

8. **Bug found — EGL import crash (Windows-specific):**
   `hmr2/utils/renderer.py` line 1-3 unconditionally sets
   `os.environ['PYOPENGL_PLATFORM'] = 'egl'` if not already set, before
   importing `pyrender`. EGL is unavailable on this headless Windows
   machine → `ImportError: Unable to load EGL library`.
   **Fix applied in `backend/services/inference_engine.py`** (not in
   vendor code): `os.environ.setdefault("PYOPENGL_PLATFORM", "win32")`
   before any `hmr2` import. Confirmed `OpenGL.platform.win32.Win32Platform`
   loads successfully on this machine. This engine never calls into
   `hmr2`'s renderer (OBJ export is manual), so this has zero functional
   effect on inference output.

9. **Bug found — `HOME` env var crash (Windows-specific):**
   `hmr2/configs/__init__.py` does
   `os.path.join(os.environ.get("HOME"), ".cache")`. Windows does not
   set `HOME` by default (uses `USERPROFILE` instead) → `TypeError:
   expected str, bytes or os.PathLike object, not NoneType`.
   **Fix applied in `backend/services/inference_engine.py`**: populate
   `os.environ["HOME"]` from `USERPROFILE` (falling back to
   `HOMEDRIVE+HOMEPATH`, then `Path.home()`) before any `hmr2` import.

10. **Verified, after both fixes:** `from hmr2.models import load_hmr2`
    imports successfully in `.venv`. Running the engine's `_load()` via
    `get_engine()` correctly detects `device: cuda` (GPU auto-selection
    logic needed no changes — it was already correct) and proceeds to
    load the checkpoint.

11. **Current blocker — SMPL body model missing (expected, licensed
    asset, not a bug):** `hmr2.models.load_hmr2()` calls
    `check_smpl_exists()`, which raises `FileNotFoundError` because
    `models/4D-Humans/data/smpl/` is empty. This file
    (`SMPL_NEUTRAL.pkl`) has a separate, restrictive license from the
    SMPL authors and is deliberately **not** bundled in the 4D-Humans
    repo clone or the HMR2 checkpoint tarball.
    `scripts/download_models.py` already correctly documents this exact
    manual step (registration at https://smplify.is.tue.mpg.de/)
    — confirmed no code/script changes needed here.
    **Status: awaiting user to obtain and place this file**, after
    which the full end-to-end test
    (`scripts/test_integration.py`) will be re-run to validate
    model-load → image submit → status poll → mesh fetch → OBJ
    vertex/face sanity check, before proceeding to Phase B.

### Files actually modified this session
- `.env` — set `HMR2_CHECKPOINT`.
- `backend/services/inference_engine.py` — added two defensive
  `os.environ.setdefault(...)` calls at the top of `_load()` (Windows
  EGL + HOME fixes described above). No other logic changed.
- `README.md` — documented the full reproducible setup sequence,
  including the venv requirement and the two Windows fixes.
- `docs/AUDIT.md` — this session log.

No changes were made to `backend/main.py`, `backend/routes/inference.py`,
`backend/utils/validators.py`, any frontend file, or any file under
`models/4D-Humans/` (vendor code).
