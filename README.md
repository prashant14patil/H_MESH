# Tanitas Human Digital Twin — Local GPU POC

End-to-end proof-of-concept for 3D human body mesh reconstruction from a
single RGB photograph — **running entirely on your local machine's GPU**.

**No Colab, no Flask tunnel, no ngrok** — HMR2 runs directly inside this
FastAPI backend process.

```
React frontend  ──►  FastAPI backend  ──►  HMR2 model (local GPU/CPU)  ──►  OBJ mesh
     ▲                                                                         │
     └─────────────────────── 3D viewer (React Three Fiber) ◄─────────────────┘
```

---

## Architecture

| Layer | Technology | Runs on |
|-------|-----------|---------|
| **Frontend** | React 18 + Vite + Tailwind + React Three Fiber | `localhost:5173` |
| **Backend** | FastAPI + uvicorn | `localhost:8000` |
| **Inference** | HMR2 / 4D-Humans (PyTorch) | **In-process, local GPU/CPU** |

**Key design decisions:**

- Inference happens in the same Python process as the FastAPI backend —
  no network hop, no external service, no secrets to manage.
- A background thread loads the HMR2 checkpoint at startup so the server
  can answer `/health` immediately while the (multi-second) model load
  happens in parallel.
- detectron2 (the optional ViTDet person detector) is OFF by default
  (`USE_DETECTRON2=false`) because it is notoriously difficult to build
  on Windows. Without it, the engine automatically treats the whole
  image as the person's bounding box — this works fine for typical
  full-body front-facing photos.

---

## Prerequisites

| Tool | Version |
|------|---------|
| Python | 3.10, 3.11, or 3.12 (3.12 confirmed working, see version pinning notes below) |
| Node.js | 18 or later |
| npm | 9 or later |
| NVIDIA GPU + driver | Recommended (CUDA 11.8 / 12.x) — CPU also works, just slower |
| Git | For cloning the 4D-Humans repo |

> You do **not** need a Google account or any Colab notebook for this version.

---

## Quick start

> **Use an isolated virtual environment.** Installing directly into a
> global/shared Python installation risks numpy/opencv version conflicts
> with other projects/tools on the same machine — this is not optional,
> it was observed to actually break other installed tools during setup
> (see "Environment setup session log" at the bottom of `docs/AUDIT.md`).

### 1 — Clone and set up environment variables

```bash
copy .env.example .env        # Windows
# cp .env.example .env        # Linux / macOS
```

### 2 — Create an isolated virtual environment

```bash
python -m venv .venv
.venv\Scripts\activate        # Windows
# source .venv/bin/activate   # Linux / macOS
```

Run **all** subsequent commands with this environment activated.

### 3 — Install PyTorch with CUDA support (do this FIRST, separately)

Pick the command matching your GPU/CUDA driver from
https://pytorch.org/get-started/locally/. Example for CUDA 12.1:

```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
```

If you don't have an NVIDIA GPU, install the CPU build instead:

```bash
pip install torch torchvision
```

Verify CUDA is visible to PyTorch:

```bash
python -c "import torch; print(torch.cuda.is_available(), torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU only')"
```

### 4 — Install the remaining Python dependencies

```bash
# From the project root
pip install -r requirements.txt
```

**Version pinning is load-bearing here** — `numpy` MUST stay `<2.0`
because `chumpy` (an SMPL/HMR2 dependency) uses numpy aliases removed in
2.x. Later steps below (installing the local `hmr2` package and its
transitive deps) can silently upgrade numpy past this ceiling — if that
happens, re-pin immediately:
```bash
pip install "numpy<2.0,>=1.24" "scipy<1.13"
```

### 5 — Download the HMR2 model (one-time setup)

This replaces the manual Colab Cell 1/Cell 2 steps with a single script
that clones the 4D-Humans repo and downloads the checkpoint + SMPL data:

```bash
python scripts/download_models.py
```

This prints `HMR2_REPO_PATH` and `HMR2_CHECKPOINT` values at the end —
copy them into your `.env` file (they usually match the defaults already
in `.env.example`, but double-check the checkpoint path it finds).

> **Note:** this downloads several GB of data. If the automatic download
> fails (the official host occasionally changes), follow the manual
> instructions printed by the script, or see the official
> [4D-Humans README](https://github.com/shubham-goel/4D-Humans) for the
> current checkpoint download link.

### 6 — Install the local `hmr2` package

The cloned 4D-Humans repo is itself a pip package and must be installed
for `import hmr2` to work. Install it in editable mode with `--no-deps`
(its dependencies are already covered by `requirements.txt` + the extra
packages below; letting pip resolve them itself can upgrade `numpy` past
the `<2.0` ceiling required above):

```bash
pip install -e models/4D-Humans --no-deps
```

It also needs a few runtime packages **not** listed in
`requirements.txt` (used only by `hmr2`'s own renderer/download
utilities — not by our engine's inference path — but required for the
package to **import successfully**):

```bash
pip install pyrender gdown dill pandas timm
```

⚠️ This step tends to upgrade `numpy`/`scipy` past the required ceiling —
**immediately re-pin afterwards**:
```bash
pip install "numpy<2.0,>=1.24" "scipy<1.13"
```

### 7 — Obtain the SMPL body model (manual, licensed — required)

The SMPL neutral body model file is **not bundled** with the 4D-Humans
repo or the HMR2 checkpoint bundle, because it has its own, separate,
more restrictive license. HMR2 inference **will fail at model-load
time** without it (`FileNotFoundError: SMPL model not found...`).

1. Register for a free account at https://smplify.is.tue.mpg.de/
2. Download the SMPL model files for Python (version 1.0.0).
3. Place the **neutral** model file at:
   ```
   models/4D-Humans/data/smpl/SMPL_NEUTRAL.pkl
   ```
   (rename the downloaded file if needed — the loader checks for this
   exact filename, or alternatively
   `data/basicModel_neutral_lbs_10_207_0_v1.0.0.pkl` relative to the
   project root as a second candidate path.)

### 8 — Windows-specific import fixes (already applied in code)

`backend/services/inference_engine.py` defensively sets two environment
variables immediately before importing `hmr2`, to work around two
import-time crashes that are specific to headless Windows:

- **`PYOPENGL_PLATFORM=win32`** — `hmr2.utils.renderer` forces
  `PYOPENGL_PLATFORM=egl` at import time, which fails on headless Windows
  (no EGL device). This engine never actually uses `hmr2`'s renderer —
  OBJ export is done manually in `_export_obj()` — so forcing the native
  `win32` OpenGL backend avoids the crash with zero functional impact.
- **`HOME`** — `hmr2.configs` builds its cache directory from
  `os.environ["HOME"]`, which does not exist on Windows (Windows uses
  `USERPROFILE`). The engine populates `HOME` from `USERPROFILE` (or
  `HOMEDRIVE`+`HOMEPATH`, or `Path.home()` as a last resort) before the
  `hmr2` import happens.

These are additive, defensive `os.environ.setdefault(...)` calls only —
no vendor code in `models/4D-Humans/` is modified, and behavior on
Linux/macOS (where `HOME`/EGL already work normally) is unaffected.

### 9 — Start the FastAPI backend

```bash
# From the project root
uvicorn backend.main:app --reload --port 8000
```

The model begins loading in the background. Check progress with:

```bash
curl http://localhost:8000/health
# {"status":"ok","service":"tanitas-backend","model_ready":false,"device":"cuda","model_load_error":null}
# ... wait a few seconds for the checkpoint to load, then model_ready becomes true
```

### 10 — Start the React frontend

```bash
cd frontend
npm install
npm run dev
```

Open **http://localhost:5173** in your browser.

### 11 — Run the automated integration test (optional but recommended)

Once `/health` reports `model_ready: true`:

```bash
python scripts/test_integration.py --image path/to/photo.jpg
```

This exercises the full contract: `GET /health` → `POST
/api/inference/submit` → poll `GET /api/inference/status/{id}` →
`GET /api/inference/mesh/{id}`, and validates the downloaded OBJ has a
plausible vertex/face count.

---

## End-to-end workflow

1. Upload a front-facing full-body JPEG or PNG photograph.
2. Click **Generate 3D Mesh**.
3. The React app submits the image to `POST /api/inference/submit`.
4. The backend validates the image and runs HMR2 inference **directly
   in-process** on your GPU — no outbound network call is made.
5. The React app polls `GET /api/inference/status/{job_id}` every 3 seconds.
6. When inference finishes (typically a few seconds on a modern GPU,
   longer on CPU), the status becomes `completed`.
7. The React app loads the OBJ mesh from `GET /api/inference/mesh/{job_id}`.
8. The 3D mesh renders in the interactive viewer (rotate, zoom, pan).
9. Click **Download OBJ mesh** to save the file locally.

---

## Project structure

```
├── backend/
│   ├── main.py                     FastAPI app entry point
│   ├── routes/
│   │   └── inference.py            POST /submit, GET /status, GET /mesh
│   ├── services/
│   │   └── inference_engine.py     Local HMR2 model load + inference (NO network calls)
│   └── utils/
│       └── validators.py           Image type + size validation
├── scripts/
│   ├── download_models.py          One-time: clone 4D-Humans + download checkpoint
│   └── test_integration.py         End-to-end smoke test (health/submit/poll/mesh)
├── frontend/                       Same React app as the Colab POC (unmodified UI)
│   ├── package.json
│   ├── vite.config.js
│   └── src/
│       ├── App.jsx
│       ├── components/
│       └── services/api.js
├── models/                         4D-Humans repo + checkpoint (created by download_models.py)
│   └── 4D-Humans/data/smpl/SMPL_NEUTRAL.pkl   Manual step — see "Quick start" step 7
├── outputs/
│   └── meshes/                     Generated OBJ files (auto-created)
├── .env.example
└── requirements.txt
```

---

## Environment variables reference

| Variable | Default | Description |
|----------|---------|--------------|
| `HMR2_REPO_PATH` | `models/4D-Humans` | Path to the cloned 4D-Humans repo |
| `HMR2_CHECKPOINT` | *(required)* | Path to the HMR2 `.ckpt` file |
| `DEVICE` | `auto` | `cuda`, `cpu`, or `auto` |
| `USE_DETECTRON2` | `false` | Attempt to load the ViTDet person detector |
| `BACKEND_PORT` | `8000` | FastAPI server port |
| `MAX_UPLOAD_SIZE_MB` | `10` | Maximum image upload size |
| `MESH_OUTPUT_DIR` | `outputs/meshes` | Local OBJ output directory |
| `VITE_API_BASE_URL` | *(blank)* | Backend URL for React; leave blank when using Vite proxy |

---

## Enabling detectron2 (optional, advanced)

By default the engine uses a full-image fallback instead of a dedicated
person detector, which works well for typical photos where the subject
fills most of the frame. If you want tighter person cropping:

1. Set `USE_DETECTRON2=true` in `.env`.
2. Install detectron2 (Linux/WSL strongly recommended — building from
   source on native Windows is difficult):
   ```bash
   pip install git+https://github.com/facebookresearch/detectron2.git
   ```
3. Restart the backend. If the import fails for any reason, the engine
   automatically falls back to the full-image mode — the app keeps working.

---

## Limitations

- HMR2 estimates external body shape from a single image — it does not
  measure body fat, muscle mass, or internal composition.
- CPU inference works but is significantly slower than GPU (expect tens
  of seconds to minutes per image vs. a few seconds on a modern GPU).
- This is a proof-of-concept, not a production system.
- Measurements derived from the mesh (height, circumferences, etc.) are
  **image-based estimates only** and are not a substitute for, nor
  equivalent to, actual Tanita body composition monitor (BCM) readings.
  These two data sources are kept clearly separate throughout the
  application and are never blended into a single number.
