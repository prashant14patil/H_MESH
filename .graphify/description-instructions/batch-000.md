# Node Description Batch 1 of 4

Graphify is running in assistant/skill mode (no API key). You are the host
assistant (Claude Code / Codex / Gemini CLI). Read the prompt below and write
your JSON answer to the answer file.

## Prompt

You are documenting nodes in a knowledge graph.
For each entry below, write ONE concise factual plain-language sentence
describing what it is or does. Use only the provided context.
For a code symbol (kind=code-symbol — a function, class, or constant),
describe what the function/symbol does based on its name, source location
and neighbors — e.g. "Resolves the configured ontology profile from graphify.yaml.".
Write every description in English (en). Do not switch languages.
No marketing language.
Respond ONLY with a JSON object mapping each node id (as a string) to its
one-sentence description — no prose, no markdown fences.

- "services_inference_engine_inferenceengine": "InferenceEngine" | kind=code-symbol | source=tanitas/backend/services/inference_engine.py:L68 | neighbors=[inference_engine.py, get_engine(), ._detect_person_box(), ._export_obj(), .get_job(), .__init__()]
- "frontend_app": "app.py" | kind=code-symbol | source=tanitas/frontend/app.py:L1 | neighbors=[build_mesh_figure(), check_backend_health(), fetch_mesh_bytes(), main(), parse_obj(), poll_status()]
- "utils_validators_validationerror": "ValidationError" | kind=code-symbol | source=tanitas/backend/utils/validators.py:L57 | neighbors=[backend/routes/inference.py ----------…, Returns the current status of a job tra…, Returns the reconstructed OBJ mesh file…, Accepts an uploaded image, validates it…, validators.py, _check_extension()]
- "scripts_test_integration": "test_integration.py" | kind=code-symbol | source=tanitas/scripts/test_integration.py:L1 | neighbors=[_fail(), main(), _ok(), _step(), test_backend_health(), test_download_mesh()]
- "services_inference_engine_enginenotreadyerror": "EngineNotReadyError" | kind=code-symbol | source=tanitas/backend/services/inference_engine.py:L61 | neighbors=[backend/routes/inference.py ----------…, Returns the current status of a job tra…, Returns the reconstructed OBJ mesh file…, Accepts an uploaded image, validates it…, inference_engine.py, RuntimeError]
- "utils_validators": "validators.py" | kind=code-symbol | source=tanitas/backend/utils/validators.py:L1 | neighbors=[_check_extension(), _check_magic_bytes(), _check_size(), _get_extension(), validate_image(), ValidationError]
- "frontend_app_render_inference_section": "render_inference_section()" | kind=code-symbol | source=tanitas/frontend/app.py:L267 | neighbors=[app.py, main(), Handle submission and polling for a giv…, poll_status(), render_result_section(), submit_image()]
- "frontend_app_render_result_section": "render_result_section()" | kind=code-symbol | source=tanitas/frontend/app.py:L329 | neighbors=[app.py, Display the 3D mesh result once inferen…, render_inference_section(), build_mesh_figure(), fetch_mesh_bytes(), parse_obj()]
- "scripts_download_models": "download_models.py" | kind=code-symbol | source=tanitas/scripts/download_models.py:L1 | neighbors=[clone_repo(), download_checkpoint(), main(), print_summary(), _run(), scripts/download_models.py -----------…]
- "components_meshviewer_mesherrorboundary": "MeshErrorBoundary" | kind=code-symbol | source=tanitas/frontend/src/components/MeshViewer.jsx:L67 | neighbors=[MeshViewer.jsx, .componentDidCatch(), .constructor(), .getDerivedStateFromError(), .render()]
- "scripts_test_integration_fail": "_fail()" | kind=code-symbol | source=tanitas/scripts/test_integration.py:L48 | neighbors=[test_integration.py, test_backend_health(), test_download_mesh(), test_poll(), test_submit()]
- "scripts_test_integration_main": "main()" | kind=code-symbol | source=tanitas/scripts/test_integration.py:L149 | neighbors=[test_integration.py, test_backend_health(), test_download_mesh(), test_poll(), test_submit()]
- "scripts_test_integration_ok": "_ok()" | kind=code-symbol | source=tanitas/scripts/test_integration.py:L45 | neighbors=[test_integration.py, test_backend_health(), test_download_mesh(), test_poll(), test_submit()]
- "scripts_test_integration_step": "_step()" | kind=code-symbol | source=tanitas/scripts/test_integration.py:L52 | neighbors=[test_integration.py, test_backend_health(), test_download_mesh(), test_poll(), test_submit()]
- "scripts_test_integration_test_backend_health": "test_backend_health()" | kind=code-symbol | source=tanitas/scripts/test_integration.py:L58 | neighbors=[test_integration.py, main(), _fail(), _ok(), _step()]
- "scripts_test_integration_test_download_mesh": "test_download_mesh()" | kind=code-symbol | source=tanitas/scripts/test_integration.py:L118 | neighbors=[test_integration.py, main(), _fail(), _ok(), _step()]
- "scripts_test_integration_test_poll": "test_poll()" | kind=code-symbol | source=tanitas/scripts/test_integration.py:L97 | neighbors=[test_integration.py, main(), _fail(), _ok(), _step()]
- "scripts_test_integration_test_submit": "test_submit()" | kind=code-symbol | source=tanitas/scripts/test_integration.py:L74 | neighbors=[test_integration.py, main(), _fail(), _ok(), _step()]
- "services_api": "api.js" | kind=code-symbol | source=tanitas/frontend/src/services/api.js:L1 | neighbors=[BASE_URL, checkHealth(), getJobStatus(), getMeshUrl(), submitImage()]
- "src_app": "App.jsx" | kind=code-symbol | source=tanitas/frontend/src/App.jsx:L1 | neighbors=[checkHealth(), getJobStatus(), getMeshUrl(), submitImage(), App()]
- "utils_validators_check_extension": "_check_extension()" | kind=code-symbol | source=tanitas/backend/utils/validators.py:L95 | neighbors=[validators.py, _get_extension(), ValidationError, Reject files whose extension is not in …, validate_image()]
- "utils_validators_check_magic_bytes": "_check_magic_bytes()" | kind=code-symbol | source=tanitas/backend/utils/validators.py:L109 | neighbors=[validators.py, _get_extension(), ValidationError, Detect the actual image type from magic…, validate_image()]
- "utils_validators_validate_image": "validate_image()" | kind=code-symbol | source=tanitas/backend/utils/validators.py:L64 | neighbors=[validators.py, Validate an uploaded image by extension…, _check_extension(), _check_magic_bytes(), _check_size()]
- "backend_main": "main.py" | kind=code-symbol | source=tanitas/backend/main.py:L1 | neighbors=[health_check(), on_shutdown(), on_startup(), backend/main.py --------------- FastA…]
- "components_meshviewer": "MeshViewer.jsx" | kind=code-symbol | source=tanitas/frontend/src/components/MeshViewer.jsx:L1 | neighbors=[LoadingMesh(), MeshErrorBoundary, MeshViewer(), OBJMesh()]
- "components_processingstatus": "ProcessingStatus.jsx" | kind=code-symbol | source=tanitas/frontend/src/components/ProcessingStatus.jsx:L1 | neighbors=[PHASE_CONFIG, ProcessingStatus(), Spinner(), StatusIcon()]
- "frontend_app_main": "main()" | kind=code-symbol | source=tanitas/frontend/app.py:L393 | neighbors=[app.py, render_inference_section(), render_sidebar(), render_upload_section()]
- "routes_inference": "inference.py" | kind=code-symbol | source=tanitas/backend/routes/inference.py:L1 | neighbors=[get_inference_status(), get_mesh(), submit_inference(), backend/routes/inference.py ----------…]
- "scripts_download_models_main": "main()" | kind=code-symbol | source=tanitas/scripts/download_models.py:L126 | neighbors=[download_models.py, clone_repo(), download_checkpoint(), print_summary()]
- "services_inference_engine": "inference_engine.py" | kind=code-symbol | source=tanitas/backend/services/inference_engine.py:L1 | neighbors=[EngineNotReadyError, get_engine(), InferenceEngine, backend/services/inference_engine.py -…]
- "services_inference_engine_inferenceengine_load": "._load()" | kind=code-symbol | source=tanitas/backend/services/inference_engine.py:L109 | neighbors=[InferenceEngine, EngineNotReadyError, ._load_detector(), ._locate_smpl_faces()]
- "utils_validators_check_size": "_check_size()" | kind=code-symbol | source=tanitas/backend/utils/validators.py:L143 | neighbors=[validators.py, ValidationError, Reject files that exceed the configured…, validate_image()]
- "utils_validators_get_extension": "_get_extension()" | kind=code-symbol | source=tanitas/backend/utils/validators.py:L157 | neighbors=[validators.py, _check_extension(), _check_magic_bytes(), Return the lowercase file extension inc…]
- "frontend_app_build_mesh_figure": "build_mesh_figure()" | kind=code-symbol | source=tanitas/frontend/app.py:L144 | neighbors=[app.py, Build a Plotly 3D Mesh figure from SMPL…, render_result_section()]
- "frontend_app_check_backend_health": "check_backend_health()" | kind=code-symbol | source=tanitas/frontend/app.py:L62 | neighbors=[app.py, Return True if the backend health endpo…, render_sidebar()]
- "frontend_app_fetch_mesh_bytes": "fetch_mesh_bytes()" | kind=code-symbol | source=tanitas/frontend/app.py:L92 | neighbors=[app.py, Download the OBJ file from the backend …, render_result_section()]
- "frontend_app_parse_obj": "parse_obj()" | kind=code-symbol | source=tanitas/frontend/app.py:L107 | neighbors=[app.py, Parse a Wavefront OBJ file and return (…, render_result_section()]
- "frontend_app_poll_status": "poll_status()" | kind=code-symbol | source=tanitas/frontend/app.py:L82 | neighbors=[app.py, GET the current status for a job from t…, render_inference_section()]
- "frontend_app_render_sidebar": "render_sidebar()" | kind=code-symbol | source=tanitas/frontend/app.py:L197 | neighbors=[app.py, main(), check_backend_health()]
- "frontend_app_render_upload_section": "render_upload_section()" | kind=code-symbol | source=tanitas/frontend/app.py:L233 | neighbors=[app.py, main(), Render the upload widget and return (by…]

## Instructions

Write a single JSON object mapping each node id to a one-sentence description
to: c:\Users\icroc\Desktop\texti\POC-MESH\.graphify\description-instructions\batch-000.json

Keep each description factual and concise (one sentence). No markdown, no prose
outside the JSON object. It is acceptable to omit a node if context is
insufficient — but include every node you can ground confidently.

Example answer format:
```json
{
  "node_id_1": "Resolves the configured ontology profile from graphify.yaml.",
  "node_id_2": "Colonel James Barclay, an antagonist in The Crooked Man."
}
```
