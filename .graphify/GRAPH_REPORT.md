# Graph Report - .  (2026-11-11)

## Corpus Check
- Corpus is ~9,553 words - fits in a single context window. You may not need a graph.

## Summary
- 135 nodes · 170 edges · 15 communities detected
- Extraction: 95% EXTRACTED · 5% INFERRED · 0% AMBIGUOUS · INFERRED: 8 edges (avg confidence: 0.5)
- Token cost: 0 input · 0 output
- Edge kinds: contains: 57 · calls: 46 · rationale_for: 37 · method: 16 · uses: 8 · imports: 4 · inherits: 2


## Input Scope
- Requested: all
- Resolved: all (source: configured-default)
- Included files: 23 · Candidates: recursive
- Excluded: 0 untracked · 0 ignored · 0 sensitive · 0 missing committed

## Graph Freshness
- Built from Git commit: `1a9dc19`
- Compare this hash to `git rev-parse HEAD` before trusting freshness-sensitive graph output.
## God Nodes (most connected - your core abstractions)
1. `InferenceEngine` - 15 edges
2. `ValidationError` - 10 edges
3. `EngineNotReadyError` - 9 edges
4. `render_inference_section()` - 6 edges
5. `render_result_section()` - 6 edges
6. `validate_image()` - 5 edges
7. `_check_extension()` - 5 edges
8. `_check_magic_bytes()` - 5 edges
9. `MeshErrorBoundary` - 5 edges
10. `_ok()` - 5 edges

## Surprising Connections (you probably didn't know these)
- `backend/routes/inference.py ---------------------------- API routes for HMR2 i` --uses--> `EngineNotReadyError`  [INFERRED]
  tanitas/backend/routes/inference.py → tanitas/backend/services/inference_engine.py
- `backend/routes/inference.py ---------------------------- API routes for HMR2 i` --uses--> `ValidationError`  [INFERRED]
  tanitas/backend/routes/inference.py → tanitas/backend/utils/validators.py
- `Returns the current status of a job tracked by the local engine.      Returns` --uses--> `EngineNotReadyError`  [INFERRED]
  tanitas/backend/routes/inference.py → tanitas/backend/services/inference_engine.py
- `Returns the current status of a job tracked by the local engine.      Returns` --uses--> `ValidationError`  [INFERRED]
  tanitas/backend/routes/inference.py → tanitas/backend/utils/validators.py
- `Returns the reconstructed OBJ mesh file for a completed job.      The mesh was` --uses--> `EngineNotReadyError`  [INFERRED]
  tanitas/backend/routes/inference.py → tanitas/backend/services/inference_engine.py

## Communities

### Community 0 - "Community 0"
Cohesion: 0.13
Nodes (21): build_mesh_figure(), check_backend_health(), fetch_mesh_bytes(), main(), parse_obj(), poll_status(), frontend/app.py ---------------- Streamlit web application for the Tanita Huma, Parse a Wavefront OBJ file and return (vertices, faces).      Parameters (+13 more)

### Community 1 - "Community 1"
Cohesion: 0.13
Nodes (8): get_engine(), InferenceEngine, backend/services/inference_engine.py ------------------------------------- Loc, SMPL faces attribute path varies across 4D-Humans versions., Attempt to load the ViTDet/detectron2 person detector.         Returns None (fu, Return the process-wide InferenceEngine, creating it on first call.      Const, Loads the HMR2 model once (in a background thread) and runs inference     for e, Kick off model loading in a background thread (idempotent).

### Community 2 - "Community 2"
Cohesion: 0.17
Nodes (14): get_inference_status(), get_mesh(), backend/routes/inference.py ---------------------------- API routes for HMR2 i, Returns the current status of a job tracked by the local engine.      Returns, Returns the reconstructed OBJ mesh file for a completed job.      The mesh was, Accepts an uploaded image, validates it, and submits it to the local     HMR2 i, submit_inference(), RuntimeError (+6 more)

### Community 3 - "Community 3"
Cohesion: 0.24
Nodes (11): _check_extension(), _check_magic_bytes(), _check_size(), _get_extension(), backend/utils/validators.py ---------------------------- Input validation help, Detect the actual image type from magic bytes and verify it matches     the dec, Reject files that exceed the configured size limit., Return the lowercase file extension including the leading dot. (+3 more)

### Community 4 - "Community 4"
Cohesion: 0.56
Nodes (9): _fail(), main(), _ok(), scripts/test_integration.py ---------------------------- End-to-end integratio, _step(), test_backend_health(), test_download_mesh(), test_poll() (+1 more)

### Community 5 - "Community 5"
Cohesion: 0.22
Nodes (1): MeshErrorBoundary

### Community 6 - "Community 6"
Cohesion: 0.36
Nodes (5): BASE_URL, checkHealth(), getJobStatus(), getMeshUrl(), submitImage()

### Community 7 - "Community 7"
Cohesion: 0.48
Nodes (6): clone_repo(), download_checkpoint(), main(), print_summary(), scripts/download_models.py --------------------------- One-time setup script t, _run()

### Community 8 - "Community 8"
Cohesion: 0.33
Nodes (3): health_check(), backend/main.py --------------- FastAPI application entry point for the Tanita, Liveness probe that also reports whether the HMR2 model has finished     loadin

### Community 9 - "Community 9"
Cohesion: 0.40
Nodes (1): PHASE_CONFIG

### Community 10 - "Community 10"
Cohesion: 0.67
Nodes (1): ALLOWED_TYPES

### Community 11 - "Community 11"
Cohesion: 1.00
Nodes (1): Tanitas backend package — local GPU inference (no Google Colab).

### Community 14 - "Community 14"
Cohesion: 1.00
Nodes (1): API route modules for the tanitas local-GPU backend.

### Community 15 - "Community 15"
Cohesion: 1.00
Nodes (1): Local inference services for the tanitas backend (no external network calls).

### Community 16 - "Community 16"
Cohesion: 1.00
Nodes (1): Utility helpers for the tanitas local-GPU backend.

## Knowledge Gaps
- **36 isolated node(s):** `Tanitas backend package — local GPU inference (no Google Colab).`, `backend/main.py --------------- FastAPI application entry point for the Tanita`, `Liveness probe that also reports whether the HMR2 model has finished     loadin`, `API route modules for the tanitas local-GPU backend.`, `Local inference services for the tanitas backend (no external network calls).` (+31 more)
  These have ≤1 connection - possible missing edges or undocumented components.
- **Thin community `Community 5`** (1 nodes): `MeshErrorBoundary`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 9`** (1 nodes): `PHASE_CONFIG`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 10`** (1 nodes): `ALLOWED_TYPES`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 11`** (1 nodes): `Tanitas backend package — local GPU inference (no Google Colab).`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 14`** (1 nodes): `API route modules for the tanitas local-GPU backend.`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 15`** (1 nodes): `Local inference services for the tanitas backend (no external network calls).`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 16`** (1 nodes): `Utility helpers for the tanitas local-GPU backend.`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.

## Suggested Questions
_Questions this graph is uniquely positioned to answer:_

- **Why does `EngineNotReadyError` connect `Community 2` to `Community 1`?**
  _High betweenness centrality (0.068) - this node is a cross-community bridge._
- **Why does `ValidationError` connect `Community 2` to `Community 3`?**
  _High betweenness centrality (0.056) - this node is a cross-community bridge._
- **Why does `InferenceEngine` connect `Community 1` to `Community 2`?**
  _High betweenness centrality (0.050) - this node is a cross-community bridge._
- **Are the 4 inferred relationships involving `ValidationError` (e.g. with `backend/routes/inference.py ---------------------------- API routes for HMR2 i` and `Returns the current status of a job tracked by the local engine.      Returns`) actually correct?**
  _`ValidationError` has 4 INFERRED edges - model-reasoned connections that need verification._
- **Are the 4 inferred relationships involving `EngineNotReadyError` (e.g. with `backend/routes/inference.py ---------------------------- API routes for HMR2 i` and `Returns the current status of a job tracked by the local engine.      Returns`) actually correct?**
  _`EngineNotReadyError` has 4 INFERRED edges - model-reasoned connections that need verification._
- **What connects `Tanitas backend package — local GPU inference (no Google Colab).`, `backend/main.py --------------- FastAPI application entry point for the Tanita`, `Liveness probe that also reports whether the HMR2 model has finished     loadin` to the rest of the system?**
  _36 weakly-connected nodes found - possible documentation gaps or missing edges._
- **Should `Community 0` be split into smaller, more focused modules?**
  _Cohesion score 0.1341991341991342 - nodes in this community are weakly interconnected._