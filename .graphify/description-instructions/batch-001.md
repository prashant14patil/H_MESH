# Node Description Batch 2 of 4

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
For an entity node (any other kind — e.g. a person, place, event, object),
describe what the entity is and its role, grounded in its type, its
relations (neighbors) and the provided citations/evidence — e.g.
"Lady Carfax, a wealthy heiress who disappears en route to Lausanne.".
Ground entity descriptions in the citations/evidence when present; do not
speculate beyond the context, so a node with no supporting context may be
left out of the reply.
Write every description in English (en). Do not switch languages.
No marketing language.
Respond ONLY with a JSON object mapping each node id (as a string) to its
one-sentence description — no prose, no markdown fences.

- "frontend_app_submit_image": "submit_image()" | kind=code-symbol | source=tanitas/frontend/app.py:L71 | neighbors=[app.py, POST the image to the backend and retur…, render_inference_section()]
- "routes_inference_rationale_1": "backend/routes/inference.py\r ----------------------------\r API routes for HMR2 i" | kind=entity | source=tanitas/backend/routes/inference.py:L1 | neighbors=[inference.py, EngineNotReadyError, ValidationError]
- "routes_inference_rationale_103": "Returns the current status of a job tracked by the local engine.\r \r     Returns" | kind=entity | source=tanitas/backend/routes/inference.py:L103 | neighbors=[get_inference_status(), EngineNotReadyError, ValidationError]
- "routes_inference_rationale_153": "Returns the reconstructed OBJ mesh file for a completed job.\r \r     The mesh was" | kind=entity | source=tanitas/backend/routes/inference.py:L153 | neighbors=[get_mesh(), EngineNotReadyError, ValidationError]
- "routes_inference_rationale_54": "Accepts an uploaded image, validates it, and submits it to the local\r     HMR2 i" | kind=entity | source=tanitas/backend/routes/inference.py:L54 | neighbors=[submit_inference(), EngineNotReadyError, ValidationError]
- "scripts_download_models_clone_repo": "clone_repo()" | kind=code-symbol | source=tanitas/scripts/download_models.py:L61 | neighbors=[download_models.py, _run(), main()]
- "services_inference_engine_get_engine": "get_engine()" | kind=code-symbol | source=tanitas/backend/services/inference_engine.py:L345 | neighbors=[inference_engine.py, InferenceEngine, Return the process-wide InferenceEngine…]
- "services_inference_engine_inferenceengine_load_async": ".load_async()" | kind=code-symbol | source=tanitas/backend/services/inference_engine.py:L98 | neighbors=[InferenceEngine, .__init__(), Kick off model loading in a background …]
- "services_inference_engine_inferenceengine_load_detector": "._load_detector()" | kind=code-symbol | source=tanitas/backend/services/inference_engine.py:L187 | neighbors=[InferenceEngine, ._load(), Attempt to load the ViTDet/detectron2 p…]
- "services_inference_engine_inferenceengine_locate_smpl_faces": "._locate_smpl_faces()" | kind=code-symbol | source=tanitas/backend/services/inference_engine.py:L172 | neighbors=[InferenceEngine, ._load(), SMPL faces attribute path varies across…]
- "services_inference_engine_inferenceengine_run_inference": "._run_inference()" | kind=code-symbol | source=tanitas/backend/services/inference_engine.py:L254 | neighbors=[InferenceEngine, ._detect_person_box(), ._export_obj()]
- "services_inference_engine_inferenceengine_submit_job": ".submit_job()" | kind=code-symbol | source=tanitas/backend/services/inference_engine.py:L307 | neighbors=[InferenceEngine, EngineNotReadyError, Queue a job and run it in a background …]
- "backend_main_health_check": "health_check()" | kind=code-symbol | source=tanitas/backend/main.py:L88 | neighbors=[main.py, Liveness probe that also reports whethe…]
- "components_imageuploader": "ImageUploader.jsx" | kind=code-symbol | source=tanitas/frontend/src/components/ImageUploader.jsx:L1 | neighbors=[ALLOWED_TYPES, ImageUploader()]
- "routes_inference_get_inference_status": "get_inference_status()" | kind=code-symbol | source=tanitas/backend/routes/inference.py:L102 | neighbors=[inference.py, Returns the current status of a job tra…]
- "routes_inference_get_mesh": "get_mesh()" | kind=code-symbol | source=tanitas/backend/routes/inference.py:L152 | neighbors=[inference.py, Returns the reconstructed OBJ mesh file…]
- "routes_inference_submit_inference": "submit_inference()" | kind=code-symbol | source=tanitas/backend/routes/inference.py:L51 | neighbors=[inference.py, Accepts an uploaded image, validates it…]
- "scripts_download_models_download_checkpoint": "download_checkpoint()" | kind=code-symbol | source=tanitas/scripts/download_models.py:L70 | neighbors=[download_models.py, main()]
- "scripts_download_models_print_summary": "print_summary()" | kind=code-symbol | source=tanitas/scripts/download_models.py:L105 | neighbors=[download_models.py, main()]
- "scripts_download_models_run": "_run()" | kind=code-symbol | source=tanitas/scripts/download_models.py:L56 | neighbors=[download_models.py, clone_repo()]
- "services_api_checkhealth": "checkHealth()" | kind=code-symbol | source=tanitas/frontend/src/services/api.js:L20 | neighbors=[api.js, App.jsx]
- "services_api_getjobstatus": "getJobStatus()" | kind=code-symbol | source=tanitas/frontend/src/services/api.js:L72 | neighbors=[api.js, App.jsx]
- "services_api_getmeshurl": "getMeshUrl()" | kind=code-symbol | source=tanitas/frontend/src/services/api.js:L98 | neighbors=[api.js, App.jsx]
- "services_api_submitimage": "submitImage()" | kind=code-symbol | source=tanitas/frontend/src/services/api.js:L38 | neighbors=[api.js, App.jsx]
- "services_inference_engine_inferenceengine_detect_person_box": "._detect_person_box()" | kind=code-symbol | source=tanitas/backend/services/inference_engine.py:L232 | neighbors=[InferenceEngine, ._run_inference()]
- "services_inference_engine_inferenceengine_export_obj": "._export_obj()" | kind=code-symbol | source=tanitas/backend/services/inference_engine.py:L294 | neighbors=[InferenceEngine, ._run_inference()]
- "services_inference_engine_inferenceengine_init": ".__init__()" | kind=code-symbol | source=tanitas/backend/services/inference_engine.py:L78 | neighbors=[InferenceEngine, .load_async()]
- "backend_init": "__init__.py" | kind=code-symbol | source=tanitas/backend/__init__.py:L1 | neighbors=[Tanitas backend package — local GPU inf…]
- "backend_init_rationale_1": "Tanitas backend package — local GPU inference (no Google Colab)." | kind=entity | source=tanitas/backend/__init__.py:L1 | neighbors=[__init__.py]
- "backend_main_on_shutdown": "on_shutdown()" | kind=code-symbol | source=tanitas/backend/main.py:L127 | neighbors=[main.py]
- "backend_main_on_startup": "on_startup()" | kind=code-symbol | source=tanitas/backend/main.py:L107 | neighbors=[main.py]
- "backend_main_rationale_1": "backend/main.py\r ---------------\r FastAPI application entry point for the Tanita" | kind=entity | source=tanitas/backend/main.py:L1 | neighbors=[main.py]
- "backend_main_rationale_89": "Liveness probe that also reports whether the HMR2 model has finished\r     loadin" | kind=entity | source=tanitas/backend/main.py:L89 | neighbors=[health_check()]
- "components_header": "Header.jsx" | kind=code-symbol | source=tanitas/frontend/src/components/Header.jsx:L1 | neighbors=[Header()]
- "components_header_header": "Header()" | kind=code-symbol | source=tanitas/frontend/src/components/Header.jsx:L7 | neighbors=[Header.jsx]
- "components_imageuploader_allowed_types": "ALLOWED_TYPES" | kind=code-symbol | source=tanitas/frontend/src/components/ImageUploader.jsx:L15 | neighbors=[ImageUploader.jsx]
- "components_imageuploader_imageuploader": "ImageUploader()" | kind=code-symbol | source=tanitas/frontend/src/components/ImageUploader.jsx:L18 | neighbors=[ImageUploader.jsx]
- "components_meshviewer_loadingmesh": "LoadingMesh()" | kind=code-symbol | source=tanitas/frontend/src/components/MeshViewer.jsx:L87 | neighbors=[MeshViewer.jsx]
- "components_meshviewer_mesherrorboundary_componentdidcatch": ".componentDidCatch()" | kind=code-symbol | source=tanitas/frontend/src/components/MeshViewer.jsx:L75 | neighbors=[MeshErrorBoundary]
- "components_meshviewer_mesherrorboundary_constructor": ".constructor()" | kind=code-symbol | source=tanitas/frontend/src/components/MeshViewer.jsx:L68 | neighbors=[MeshErrorBoundary]

## Instructions

Write a single JSON object mapping each node id to a one-sentence description
to: c:\Users\icroc\Desktop\texti\POC-MESH\.graphify\description-instructions\batch-001.json

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
