# Production integration and HMR2 regression report

## Scope and environment

Windows 11, Python 3.12.10, NVIDIA GeForce RTX 4070 Laptop GPU. Tests used the actual `backend.main:app` and actual Vite frontend on local test ports, with separate capture and mesh output directories.

The combined test process used `.venv-pose` with `.venv/Lib/site-packages` appended to `sys.path`. NumPy 1.26.4, SciPy 1.12.0, MediaPipe 0.10.14, JAX/JAXlib 0.4.30 and OpenCV-contrib 4.10.0.84 came from the isolated environment; HMR2 runtime packages including Torch 2.5.1+cu121 came from the existing `.venv`. The isolated environment passed `pip check`, but the combined overlay is **not a clean, independently resolved production environment** and `pip check` in the isolated environment does not validate packages imported from the appended path.

The existing `.venv` `pip check` reported missing `ml-dtypes`, `matplotlib`, `opencv-contrib-python` and `sounddevice` for the partially installed MediaPipe/JAX packages. Pandas 3.1.1 also requires NumPy >=2.0.2, conflicting with the installed 1.26.4. NumPy remained 1.26.4 throughout this milestone. No packages were changed in the HMR2 environment. Playwright 1.49.1 was installed only in `.venv-pose` for browser testing, using an already installed Chrome.

## Main frontend and backend video workflow

A real 8.687-second 1280×720 rotation WebM was selected in the **actual frontend** using Chrome automation. The UI sent:

- Upload: HTTP 201.
- Extraction start: HTTP 202; status polling and frame metadata: HTTP 200.
- Automatic pose-processing start: HTTP 202; pose status polling: HTTP 200.
- Pose results and viewpoint coverage: HTTP 200.
- Frame and pose visualization assets: HTTP 200.

Backend logs reported 30 extracted frames, zero rejected and 30/30 person detections, with five frames requiring review. The frontend displayed 30 thumbnails, the insufficient-viewpoint-coverage warning, and a decoded pose visualization and landmark table. No browser page errors were reported. This was one frontend upload; the concurrent tests below used the API directly.

An invalid WebM upload containing non-video bytes returned HTTP 400 with an unrecognized-container error.

## Multiple jobs

Two concurrent API uploads of the same real video received distinct IDs (`ecff78fe-773e-4eb8-a785-c356e3be4d19` and `cdd88fcb-9763-4e94-8b71-b489218f1f61`). Each extracted 30 frames, completed pose processing with 30 person detections and five frames requiring review, and returned pose results and viewpoint coverage HTTP 200. Each has a separate output directory and processing summary; both stored source SHA-256 hashes match the source video. This test does not prove isolation under arbitrary load, failure conditions or all MediaPipe concurrency patterns.

## HMR2 single-photo regression

A fresh main-backend process in the existing `.venv` loaded the configured HMR2 checkpoint on CUDA. A real full-body frame was submitted via the inference API, reached `completed`, and returned an OBJ via HTTP 200: 461,988 bytes, 6,890 finite vertices and 13,776 faces, with face indices in range. This meets the existing integration test's minimum geometry checks and matches the backend's 13,776 SMPL face count. It does not establish anatomical accuracy.

The combined overlay process also loaded HMR2 on CUDA and completed a fresh inference with the same vertex and face counts. The actual frontend subsequently uploaded a photo, displayed its mesh results section and fetched the mesh with HTTP 200. The test did not independently assess the rendered mesh's anatomical quality.

## Viewpoint investigation

The contact sheet shows opposing profiles around 2–3 and 5–6 seconds. In these frames, shoulder visibilities are nearly identical (about 0.999 on both sides), so the estimator's 0.05 visibility-gap threshold produces `intermediate` with `side_view_but_left_right_ambiguous`, rather than a left/right label.

The relative depth signs of shoulders and hips reverse between the profile intervals: for example, at 2.319 seconds left/right shoulder z is 0.0559/-0.3308 and hip z is 0.1178/-0.1175; at 5.518 seconds shoulder z is -0.3255/0.0323 and hip z is -0.1202/0.1203. Nose position also changes relative to the shoulders. These observations suggest a possible depth-based signal, but do not validate camera-relative left/right semantics, mirror handling, or reliability across people, lighting and poses. No heuristic change was made on the basis of one recording; ambiguity and the coverage warning remain.

## Final checks and limitations

- Frontend production build passed; Vite warned that a JavaScript chunk exceeds 500 kB.
- Three sequential-sampling regression tests passed.
- Main backend `/health` returned HTTP 200, `model_ready: true`, CUDA, after the video and photo workflows.
- Browser video and photo workflows were tested with one recording and one photo; the browser test confirms mesh fetch, not a clinical or geometric accuracy assessment.
- Full production dependency resolution remains a blocker: the existing environment has conflicts and the successful combined test used a test-only import overlay. Reproduce in a clean, fully resolved environment before deployment.
- Side classification and confirmed 360° coverage remain unvalidated. More labeled videos and mirrored/camera-relative tests are needed.
- Variable-frame-rate and timestamp-discontinuity coverage remains limited; the tested browser WebM and three sampling unit tests are not an exhaustive codec suite.
- Capture and mesh test outputs contain personal data under `data/production_validation` and `outputs/production_validation`; handle according to the applicable privacy policy.
- HMR2 architecture and inference logic were not modified. Multi-view reconstruction was not started.

## Files added during this milestone

- `scripts/validate_hmr2_regression.py`
- `scripts/validate_production_jobs.py`
- `scripts/validate_production_browser.py`
- `scripts/inspect_viewpoint_signals.py`
- `docs/PRODUCTION_VALIDATION_REPORT.md`

Earlier uncommitted video-pipeline and validation changes remain in the working tree; they were not introduced by this milestone.
