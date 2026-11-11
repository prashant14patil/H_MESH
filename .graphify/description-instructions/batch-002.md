# Node Description Batch 3 of 4

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

- "components_meshviewer_mesherrorboundary_getderivedstatefromerror": ".getDerivedStateFromError()" | kind=code-symbol | source=tanitas/frontend/src/components/MeshViewer.jsx:L72 | neighbors=[MeshErrorBoundary]
- "components_meshviewer_mesherrorboundary_render": ".render()" | kind=code-symbol | source=tanitas/frontend/src/components/MeshViewer.jsx:L78 | neighbors=[MeshErrorBoundary]
- "components_meshviewer_meshviewer": "MeshViewer()" | kind=code-symbol | source=tanitas/frontend/src/components/MeshViewer.jsx:L99 | neighbors=[MeshViewer.jsx]
- "components_meshviewer_objmesh": "OBJMesh()" | kind=code-symbol | source=tanitas/frontend/src/components/MeshViewer.jsx:L22 | neighbors=[MeshViewer.jsx]
- "components_processingstatus_phase_config": "PHASE_CONFIG" | kind=code-symbol | source=tanitas/frontend/src/components/ProcessingStatus.jsx:L15 | neighbors=[ProcessingStatus.jsx]
- "components_processingstatus_processingstatus": "ProcessingStatus()" | kind=code-symbol | source=tanitas/frontend/src/components/ProcessingStatus.jsx:L90 | neighbors=[ProcessingStatus.jsx]
- "components_processingstatus_spinner": "Spinner()" | kind=code-symbol | source=tanitas/frontend/src/components/ProcessingStatus.jsx:L48 | neighbors=[ProcessingStatus.jsx]
- "components_processingstatus_statusicon": "StatusIcon()" | kind=code-symbol | source=tanitas/frontend/src/components/ProcessingStatus.jsx:L61 | neighbors=[ProcessingStatus.jsx]
- "components_resultpanel": "ResultPanel.jsx" | kind=code-symbol | source=tanitas/frontend/src/components/ResultPanel.jsx:L1 | neighbors=[ResultPanel()]
- "components_resultpanel_resultpanel": "ResultPanel()" | kind=code-symbol | source=tanitas/frontend/src/components/ResultPanel.jsx:L14 | neighbors=[ResultPanel.jsx]
- "frontend_app_rationale_1": "frontend/app.py\r ----------------\r Streamlit web application for the Tanita Huma" | kind=entity | source=tanitas/frontend/app.py:L1 | neighbors=[app.py]
- "frontend_app_rationale_108": "Parse a Wavefront OBJ file and return (vertices, faces).\r \r     Parameters" | kind=entity | source=tanitas/frontend/app.py:L108 | neighbors=[parse_obj()]
- "frontend_app_rationale_145": "Build a Plotly 3D Mesh figure from SMPL vertices and faces.\r \r     The mesh is c" | kind=entity | source=tanitas/frontend/app.py:L145 | neighbors=[build_mesh_figure()]
- "frontend_app_rationale_234": "Render the upload widget and return (bytes, filename) or (None, None)." | kind=entity | source=tanitas/frontend/app.py:L234 | neighbors=[render_upload_section()]
- "frontend_app_rationale_268": "Handle submission and polling for a given image." | kind=entity | source=tanitas/frontend/app.py:L268 | neighbors=[render_inference_section()]
- "frontend_app_rationale_330": "Display the 3D mesh result once inference is complete." | kind=entity | source=tanitas/frontend/app.py:L330 | neighbors=[render_result_section()]
- "frontend_app_rationale_63": "Return True if the backend health endpoint responds OK." | kind=entity | source=tanitas/frontend/app.py:L63 | neighbors=[check_backend_health()]
- "frontend_app_rationale_72": "POST the image to the backend and return the parsed JSON response." | kind=entity | source=tanitas/frontend/app.py:L72 | neighbors=[submit_image()]
- "frontend_app_rationale_83": "GET the current status for a job from the backend." | kind=entity | source=tanitas/frontend/app.py:L83 | neighbors=[poll_status()]
- "frontend_app_rationale_93": "Download the OBJ file from the backend mesh endpoint." | kind=entity | source=tanitas/frontend/app.py:L93 | neighbors=[fetch_mesh_bytes()]
- "routes_init": "__init__.py" | kind=code-symbol | source=tanitas/backend/routes/__init__.py:L1 | neighbors=[API route modules for the tanitas local…]
- "routes_init_rationale_1": "API route modules for the tanitas local-GPU backend." | kind=entity | source=tanitas/backend/routes/__init__.py:L1 | neighbors=[__init__.py]
- "runtimeerror": "RuntimeError" | kind=code-symbol | neighbors=[EngineNotReadyError]
- "scripts_download_models_rationale_1": "scripts/download_models.py\r ---------------------------\r One-time setup script t" | kind=entity | source=tanitas/scripts/download_models.py:L1 | neighbors=[download_models.py]
- "scripts_test_integration_rationale_1": "scripts/test_integration.py\r ----------------------------\r End-to-end integratio" | kind=entity | source=tanitas/scripts/test_integration.py:L1 | neighbors=[test_integration.py]
- "services_api_base_url": "BASE_URL" | kind=code-symbol | source=tanitas/frontend/src/services/api.js:L14 | neighbors=[api.js]
- "services_inference_engine_inferenceengine_get_job": ".get_job()" | kind=code-symbol | source=tanitas/backend/services/inference_engine.py:L334 | neighbors=[InferenceEngine]
- "services_inference_engine_inferenceengine_is_ready": ".is_ready()" | kind=code-symbol | source=tanitas/backend/services/inference_engine.py:L222 | neighbors=[InferenceEngine]
- "services_inference_engine_inferenceengine_load_error": ".load_error()" | kind=code-symbol | source=tanitas/backend/services/inference_engine.py:L226 | neighbors=[InferenceEngine]
- "services_inference_engine_rationale_1": "backend/services/inference_engine.py\r -------------------------------------\r Loc" | kind=entity | source=tanitas/backend/services/inference_engine.py:L1 | neighbors=[inference_engine.py]
- "services_inference_engine_rationale_173": "SMPL faces attribute path varies across 4D-Humans versions." | kind=entity | source=tanitas/backend/services/inference_engine.py:L173 | neighbors=[._locate_smpl_faces()]
- "services_inference_engine_rationale_188": "Attempt to load the ViTDet/detectron2 person detector.\r         Returns None (fu" | kind=entity | source=tanitas/backend/services/inference_engine.py:L188 | neighbors=[._load_detector()]
- "services_inference_engine_rationale_308": "Queue a job and run it in a background thread." | kind=entity | source=tanitas/backend/services/inference_engine.py:L308 | neighbors=[.submit_job()]
- "services_inference_engine_rationale_346": "Return the process-wide InferenceEngine, creating it on first call.\r \r     Const" | kind=entity | source=tanitas/backend/services/inference_engine.py:L346 | neighbors=[get_engine()]
- "services_inference_engine_rationale_62": "Raised when the inference engine could not be initialised." | kind=entity | source=tanitas/backend/services/inference_engine.py:L62 | neighbors=[EngineNotReadyError]
- "services_inference_engine_rationale_69": "Loads the HMR2 model once (in a background thread) and runs inference\r     for e" | kind=entity | source=tanitas/backend/services/inference_engine.py:L69 | neighbors=[InferenceEngine]
- "services_inference_engine_rationale_99": "Kick off model loading in a background thread (idempotent)." | kind=entity | source=tanitas/backend/services/inference_engine.py:L99 | neighbors=[.load_async()]
- "services_init": "__init__.py" | kind=code-symbol | source=tanitas/backend/services/__init__.py:L1 | neighbors=[Local inference services for the tanita…]
- "services_init_rationale_1": "Local inference services for the tanitas backend (no external network calls)." | kind=entity | source=tanitas/backend/services/__init__.py:L1 | neighbors=[__init__.py]
- "src_app_app": "App()" | kind=code-symbol | source=tanitas/frontend/src/App.jsx:L35 | neighbors=[App.jsx]

## Instructions

Write a single JSON object mapping each node id to a one-sentence description
to: c:\Users\icroc\Desktop\texti\POC-MESH\.graphify\description-instructions\batch-002.json

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
