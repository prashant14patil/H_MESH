# Community Labeling

Graphify is running in assistant/skill mode (no API key). You are the host
assistant (Claude Code / Codex / Gemini CLI). Read the community listing below
and write 2-5 word plain-language names for each.

## Language

Write every name in English (en). Do not switch languages.

## Communities

Community 0: render_inference_section(, render_result_section(, app.py, build_mesh_figure(, check_backend_health(, fetch_mesh_bytes(, main(, parse_obj(, poll_status(, frontend/app.py ---------------- Streamlit web application, Parse a Wavefront OBJ file and return (vertices, faces).  , Build a Plotly 3D Mesh figure from SMPL vertices and faces.
Community 1: InferenceEngine, inference_engine.py, get_engine(, ._detect_person_box(, ._export_obj(, .get_job(, .__init__(, .is_ready(, ._load(, .load_async(, ._load_detector(, .load_error(
Community 2: EngineNotReadyError, ValidationError, inference.py, get_inference_status(, get_mesh(, backend/routes/inference.py ---------------------------- A, Returns the current status of a job tracked by the local eng, Returns the reconstructed OBJ mesh file for a completed job., Accepts an uploaded image, validates it, and submits it to t, submit_inference(, RuntimeError, .submit_job(
Community 3: _check_extension(, _check_magic_bytes(, validate_image(, validators.py, _check_size(, _get_extension(, backend/utils/validators.py ---------------------------- I, Detect the actual image type from magic bytes and verify it , Reject files that exceed the configured size limit., Return the lowercase file extension including the leading do, Validate an uploaded image by extension, magic bytes, and si, Reject files whose extension is not in the allowlist.
Community 4: _ok(, test_integration.py, _fail(, main(, scripts/test_integration.py ---------------------------- E, _step(, test_backend_health(, test_download_mesh(, test_poll(, test_submit(
Community 5: MeshErrorBoundary, MeshViewer.jsx, LoadingMesh(, .componentDidCatch(, .constructor(, .getDerivedStateFromError(, .render(, MeshViewer(, OBJMesh(
Community 6: api.js, BASE_URL, checkHealth(, getJobStatus(, getMeshUrl(, submitImage(, App.jsx, App(
Community 7: download_models.py, clone_repo(, download_checkpoint(, main(, print_summary(, scripts/download_models.py --------------------------- One, _run(
Community 8: main.py, health_check(, on_shutdown(, on_startup(, backend/main.py --------------- FastAPI application entry , Liveness probe that also reports whether the HMR2 model has 
Community 9: ProcessingStatus.jsx, PHASE_CONFIG, ProcessingStatus(, Spinner(, StatusIcon(
Community 10: ImageUploader.jsx, ALLOWED_TYPES, ImageUploader(
Community 11: __init__.py, Tanitas backend package — local GPU inference (no Google Col
Community 12: Header.jsx, Header(
Community 13: ResultPanel.jsx, ResultPanel(
Community 14: __init__.py, API route modules for the tanitas local-GPU backend.
Community 15: __init__.py, Local inference services for the tanitas backend (no externa
Community 16: __init__.py, Utility helpers for the tanitas local-GPU backend.
Community 17: __init__.py
Community 18: postcss.config.js
Community 19: tailwind.config.js
Community 20: vite.config.js
Community 21: main.jsx

## Instructions

Write a single JSON object mapping each community id (as a string) to its
2-5 word name to: c:\Users\icroc\Desktop\texti\POC-MESH\.graphify\label-instructions\communities.json

Example:
```json
{
  "0": "Authentication Flow",
  "1": "Authentication Flow",
  "2": "Authentication Flow"
}
```

Then re-run `graphify update` (or `graphify label`) to ingest the names.
