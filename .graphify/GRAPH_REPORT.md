# Graph Report - .  (2026-11-11)

## Corpus Check
- Corpus is ~12,580 words - fits in a single context window. You may not need a graph.

## Summary
- 154 nodes · 217 edges · 11 communities detected
- Extraction: 96% EXTRACTED · 4% INFERRED · 0% AMBIGUOUS · INFERRED: 8 edges (avg confidence: 0.5)
- Token cost: 0 input · 0 output
- Edge kinds: contains: 57 · rationale_for: 49 · calls: 46 · MODIFIES: 26 · method: 16 · uses: 8 · ON_BRANCH: 5 · imports: 4 · PARENT_OF: 4 · inherits: 2


## Input Scope
- Requested: auto
- Resolved: committed (source: default-auto)
- Included files: 24 · Candidates: 100
- Excluded: 5 untracked · 48579 ignored · 0 sensitive · 0 missing committed
- Recommendation: Use --scope all or graphify.yaml inputs.corpus for a knowledge-base folder.

## Graph Freshness
- Built from Git commit: `a26dcfb`
- Compare this hash to `git rev-parse HEAD` before trusting freshness-sensitive graph output.
## God Nodes (most connected - your core abstractions)
1. `InferenceEngine` - 15 edges
2. `ValidationError` - 10 edges
3. `EngineNotReadyError` - 9 edges
4. `render_inference_section()` - 6 edges
5. `render_result_section()` - 6 edges
6. `get_engine()` - 5 edges
7. `validate_image()` - 5 edges
8. `_check_extension()` - 5 edges
9. `_check_magic_bytes()` - 5 edges
10. `MeshErrorBoundary` - 5 edges

## Surprising Connections (you probably didn't know these)
- `backend/routes/inference.py ---------------------------- API routes for HMR2 i` --uses--> `EngineNotReadyError`  [INFERRED]
  backend/routes/inference.py → backend/services/inference_engine.py
- `backend/routes/inference.py ---------------------------- API routes for HMR2 i` --uses--> `ValidationError`  [INFERRED]
  backend/routes/inference.py → backend/utils/validators.py
- `Returns the current status of a job tracked by the local engine.      Returns` --uses--> `EngineNotReadyError`  [INFERRED]
  backend/routes/inference.py → backend/services/inference_engine.py
- `Returns the current status of a job tracked by the local engine.      Returns` --uses--> `ValidationError`  [INFERRED]
  backend/routes/inference.py → backend/utils/validators.py
- `Returns the reconstructed OBJ mesh file for a completed job.      The mesh was` --uses--> `EngineNotReadyError`  [INFERRED]
  backend/routes/inference.py → backend/services/inference_engine.py

## Communities

### Community 0 - "Community 0"
Cohesion: 0.08
Nodes (12): Tanitas backend package — local GPU inference (no Google Colab)., main, 1a9dc19 first commit, 8f1a4d9 first commit, a26dcfb update backed added BMR, b06db69 first commit, e1ee660 prototype phase-1, ALLOWED_TYPES (+4 more)

### Community 1 - "Community 1"
Cohesion: 0.12
Nodes (24): get_inference_status(), get_mesh(), backend/routes/inference.py ---------------------------- API routes for HMR2 i, Returns the current status of a job tracked by the local engine.      Returns, Returns the reconstructed OBJ mesh file for a completed job.      The mesh was, Accepts an uploaded image, validates it, and submits it to the local     HMR2 i, submit_inference(), RuntimeError (+16 more)

### Community 2 - "Community 2"
Cohesion: 0.10
Nodes (12): InferenceEngine, SMPL faces attribute path varies across 4D-Humans versions., Attempt to load the ViTDet/detectron2 person detector.         Returns None (fu, SMPL faces attribute path varies across 4D-Humans versions., Attempt to load the ViTDet/detectron2 person detector.         Returns None (fu, SMPL faces attribute path varies across 4D-Humans versions., Attempt to load the ViTDet/detectron2 person detector.         Returns None (fu, Queue a job and run it in a background thread. (+4 more)

### Community 3 - "Community 3"
Cohesion: 0.13
Nodes (21): build_mesh_figure(), check_backend_health(), fetch_mesh_bytes(), main(), parse_obj(), poll_status(), frontend/app.py ---------------- Streamlit web application for the Tanita Huma, Parse a Wavefront OBJ file and return (vertices, faces).      Parameters (+13 more)

### Community 4 - "Community 4"
Cohesion: 0.56
Nodes (9): _fail(), main(), _ok(), scripts/test_integration.py ---------------------------- End-to-end integratio, _step(), test_backend_health(), test_download_mesh(), test_poll() (+1 more)

### Community 5 - "Community 5"
Cohesion: 0.22
Nodes (1): MeshErrorBoundary

### Community 6 - "Community 6"
Cohesion: 0.25
Nodes (5): health_check(), backend/main.py --------------- FastAPI application entry point for the Tanita, # NOTE: load_dotenv() MUST run before any backend module is imported., Liveness probe that also reports whether the HMR2 model has finished     loadin, Liveness probe that also reports whether the HMR2 model has finished     loadin

### Community 7 - "Community 7"
Cohesion: 0.39
Nodes (6): clone_repo(), download_checkpoint(), main(), print_summary(), scripts/download_models.py --------------------------- One-time setup script t, _run()

### Community 8 - "Community 8"
Cohesion: 0.36
Nodes (5): BASE_URL, checkHealth(), getJobStatus(), getMeshUrl(), submitImage()

### Community 9 - "Community 9"
Cohesion: 0.33
Nodes (5): get_engine(), backend/services/inference_engine.py ------------------------------------- Loc, Return the process-wide InferenceEngine, creating it on first call.      Const, Return the process-wide InferenceEngine, creating it on first call.      Const, Return the process-wide InferenceEngine, creating it on first call.      Const

### Community 10 - "Community 10"
Cohesion: 0.40
Nodes (1): PHASE_CONFIG

## Knowledge Gaps
- **47 isolated node(s):** `Tanitas backend package — local GPU inference (no Google Colab).`, `backend/main.py --------------- FastAPI application entry point for the Tanita`, `Liveness probe that also reports whether the HMR2 model has finished     loadin`, `# NOTE: load_dotenv() MUST run before any backend module is imported.`, `API route modules for the tanitas local-GPU backend.` (+42 more)
  These have ≤1 connection - possible missing edges or undocumented components.
- **Thin community `Community 5`** (1 nodes): `MeshErrorBoundary`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 10`** (1 nodes): `PHASE_CONFIG`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.

## Suggested Questions
_Questions this graph is uniquely positioned to answer:_

- **Why does `InferenceEngine` connect `Community 2` to `Community 9`?**
  _High betweenness centrality (0.229) - this node is a cross-community bridge._
- **Why does `EngineNotReadyError` connect `Community 1` to `Community 9`, `Community 2`?**
  _High betweenness centrality (0.098) - this node is a cross-community bridge._
- **Are the 4 inferred relationships involving `ValidationError` (e.g. with `backend/routes/inference.py ---------------------------- API routes for HMR2 i` and `Returns the current status of a job tracked by the local engine.      Returns`) actually correct?**
  _`ValidationError` has 4 INFERRED edges - model-reasoned connections that need verification._
- **Are the 4 inferred relationships involving `EngineNotReadyError` (e.g. with `backend/routes/inference.py ---------------------------- API routes for HMR2 i` and `Returns the current status of a job tracked by the local engine.      Returns`) actually correct?**
  _`EngineNotReadyError` has 4 INFERRED edges - model-reasoned connections that need verification._
- **What connects `Tanitas backend package — local GPU inference (no Google Colab).`, `backend/main.py --------------- FastAPI application entry point for the Tanita`, `Liveness probe that also reports whether the HMR2 model has finished     loadin` to the rest of the system?**
  _47 weakly-connected nodes found - possible documentation gaps or missing edges._
- **Should `Community 0` be split into smaller, more focused modules?**
  _Cohesion score 0.082010582010582 - nodes in this community are weakly interconnected._
- **Should `Community 1` be split into smaller, more focused modules?**
  _Cohesion score 0.12 - nodes in this community are weakly interconnected._