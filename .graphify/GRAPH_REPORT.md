# Graph Report - .  (2026-11-12)

## Corpus Check
- Corpus is ~22,895 words - fits in a single context window. You may not need a graph.

## Summary
- 292 nodes · 433 edges · 13 communities detected
- Extraction: 93% EXTRACTED · 7% INFERRED · 0% AMBIGUOUS · INFERRED: 30 edges (avg confidence: 0.5)
- Token cost: 0 input · 0 output
- Edge kinds: contains: 109 · rationale_for: 98 · calls: 80 · MODIFIES: 43 · method: 40 · uses: 30 · imports: 15 · ON_BRANCH: 7 · PARENT_OF: 6 · inherits: 5


## Input Scope
- Requested: auto
- Resolved: committed (source: default-auto)
- Included files: 35 · Candidates: 119
- Excluded: 64 untracked · 65148 ignored · 0 sensitive · 0 missing committed
- Recommendation: Use --scope all or graphify.yaml inputs.corpus for a knowledge-base folder.

## Graph Freshness
- Built from Git commit: `06aee9c`
- Compare this hash to `git rev-parse HEAD` before trusting freshness-sensitive graph output.
## God Nodes (most connected - your core abstractions)
1. `PoseProcessingError` - 16 edges
2. `VideoCaptureEngine` - 16 edges
3. `InferenceEngine` - 15 edges
4. `PoseProcessingEngine` - 15 edges
5. `VideoProcessingError` - 13 edges
6. `VideoValidationError` - 12 edges
7. `ValidationError` - 10 edges
8. `EngineNotReadyError` - 9 edges
9. `render_inference_section()` - 6 edges
10. `render_result_section()` - 6 edges

## Surprising Connections (you probably didn't know these)
- `backend/routes/pose.py ------------------------- API routes for Module 3: Huma` --uses--> `PoseProcessingError`  [INFERRED]
  backend/routes/pose.py → backend/services/pose_processor.py
- `Returns the full processing_summary.json: per-frame person     detection, pose` --uses--> `PoseProcessingError`  [INFERRED]
  backend/routes/pose.py → backend/services/pose_processor.py
- `Serves one of the per-frame outputs written under     ``processed_frames/{frame` --uses--> `PoseProcessingError`  [INFERRED]
  backend/routes/pose.py → backend/services/pose_processor.py
- `Derives a 360-degree viewpoint coverage estimate from the pose     landmarks al` --uses--> `PoseProcessingError`  [INFERRED]
  backend/routes/pose.py → backend/services/pose_processor.py
- `Kicks off background processing of every frame previously extracted     for ``v` --uses--> `PoseProcessingError`  [INFERRED]
  backend/routes/pose.py → backend/services/pose_processor.py

## Communities

### Community 0 - "Community 0"
Cohesion: 0.05
Nodes (16): Tanitas backend package — local GPU inference (no Google Colab)., main, 1a9dc19 first commit, 8f1a4d9 first commit, a26dcfb update backed added BMR, b06db69 first commit, e1ee660 prototype phase-1, f30bc43 working SMPL model (+8 more)

### Community 1 - "Community 1"
Cohesion: 0.07
Nodes (16): 06aee9c phase cheakpoint 1, VIEWPOINT_COLORS, VIEWPOINT_LABELS, ALLOWED_TYPES, BASE_URL, getCaptureStatus(), getFrameImageUrl(), getFrameMetadata() (+8 more)

### Community 2 - "Community 2"
Cohesion: 0.10
Nodes (29): get_capture_status(), get_frame_image(), get_frames(), process_video(), backend/routes/capture.py --------------------------- API routes for 360 video, Returns the current status of a capture job: "uploaded",     "processing", "com, Returns the full metadata.json written by frame extraction:     video info (fps, Serves a single extracted frame (e.g. ``frame_001.jpg``) as a JPEG     image, u (+21 more)

### Community 3 - "Community 3"
Cohesion: 0.09
Nodes (16): get_capture_engine(), backend/services/video_processor.py ------------------------------------- Loca, Core CV pipeline: load video, sample evenly spaced candidate         frames, sc, Walk every frame via grab()+POS_MSEC to get a real duration., Return trustworthy fps/duration/total_frames/resolution., Decode forward to each target instead of seeking unreliable WebM timestamps., Time-based sampling pipeline: resolve trustworthy timing info,         sample c, Return the process-wide VideoCaptureEngine, creating it on first call. (+8 more)

### Community 4 - "Community 4"
Cohesion: 0.11
Nodes (25): get_inference_status(), get_mesh(), backend/routes/inference.py ---------------------------- API routes for HMR2 i, Returns the current status of a job tracked by the local engine.      Returns, Returns the reconstructed OBJ mesh file for a completed job.      The mesh was, Accepts an uploaded image, validates it, and submits it to the local     HMR2 i, submit_inference(), EngineNotReadyError (+17 more)

### Community 5 - "Community 5"
Cohesion: 0.10
Nodes (13): get_engine(), InferenceEngine, SMPL faces attribute path varies across 4D-Humans versions., Attempt to load the ViTDet/detectron2 person detector.         Returns None (fu, SMPL faces attribute path varies across 4D-Humans versions., Attempt to load the ViTDet/detectron2 person detector.         Returns None (fu, SMPL faces attribute path varies across 4D-Humans versions., Attempt to load the ViTDet/detectron2 person detector.         Returns None (fu (+5 more)

### Community 6 - "Community 6"
Cohesion: 0.13
Nodes (21): build_mesh_figure(), check_backend_health(), fetch_mesh_bytes(), main(), parse_obj(), poll_status(), frontend/app.py ---------------- Streamlit web application for the Tanita Huma, Parse a Wavefront OBJ file and return (vertices, faces).      Parameters (+13 more)

### Community 7 - "Community 7"
Cohesion: 0.14
Nodes (9): get_pose_engine(), PoseProcessingEngine, backend/services/pose_processor.py ------------------------------------- Modul, Load processing_summary.json for a processed video, if present., Resolve a processed-frame asset (segmentation_mask.png,         pose_visualizat, Kick off background pose/segmentation processing for a video., Best-effort front/side/back heuristic based on the horizontal         separatio, Return the process-wide PoseProcessingEngine, creating it on first call. (+1 more)

### Community 8 - "Community 8"
Cohesion: 0.17
Nodes (17): get_pose_frame_asset(), get_pose_results(), get_pose_status(), get_viewpoint_coverage(), process_pose(), backend/routes/pose.py ------------------------- API routes for Module 3: Huma, Returns the full processing_summary.json: per-frame person     detection, pose, Returns the full processing_summary.json: per-frame person     detection, pose (+9 more)

### Community 9 - "Community 9"
Cohesion: 0.20
Nodes (7): health_check(), backend/main.py --------------- FastAPI application entry point for the Tanita, Liveness probe that also reports whether the HMR2 model has finished     loadin, Liveness probe that also reports whether the HMR2 model has finished     loadin, # NOTE: load_dotenv() MUST run before any backend module is imported., Liveness probe that also reports whether the HMR2 model has finished     loadin, Liveness probe that also reports whether the HMR2 model has finished     loadin

### Community 10 - "Community 10"
Cohesion: 0.56
Nodes (9): _fail(), main(), _ok(), scripts/test_integration.py ---------------------------- End-to-end integratio, _step(), test_backend_health(), test_download_mesh(), test_poll() (+1 more)

### Community 11 - "Community 11"
Cohesion: 0.39
Nodes (6): clone_repo(), download_checkpoint(), main(), print_summary(), scripts/download_models.py --------------------------- One-time setup script t, _run()

### Community 12 - "Community 12"
Cohesion: 0.36
Nodes (5): BASE_URL, checkHealth(), getJobStatus(), getMeshUrl(), submitImage()

## Knowledge Gaps
- **85 isolated node(s):** `Tanitas backend package — local GPU inference (no Google Colab).`, `backend/main.py --------------- FastAPI application entry point for the Tanita`, `Liveness probe that also reports whether the HMR2 model has finished     loadin`, `# NOTE: load_dotenv() MUST run before any backend module is imported.`, `API route modules for the tanitas local-GPU backend.` (+80 more)
  These have ≤1 connection - possible missing edges or undocumented components.

## Suggested Questions
_Questions this graph is uniquely positioned to answer:_

- **Why does `PoseProcessingError` connect `Community 8` to `Community 7`, `Community 2`?**
  _High betweenness centrality (0.112) - this node is a cross-community bridge._
- **Why does `EngineNotReadyError` connect `Community 4` to `Community 0`, `Community 2`, `Community 5`?**
  _High betweenness centrality (0.111) - this node is a cross-community bridge._
- **Why does `InferenceEngine` connect `Community 5` to `Community 0`, `Community 4`?**
  _High betweenness centrality (0.110) - this node is a cross-community bridge._
- **Are the 10 inferred relationships involving `PoseProcessingError` (e.g. with `backend/routes/pose.py ------------------------- API routes for Module 3: Huma` and `Returns the full processing_summary.json: per-frame person     detection, pose`) actually correct?**
  _`PoseProcessingError` has 10 INFERRED edges - model-reasoned connections that need verification._
- **What connects `Tanitas backend package — local GPU inference (no Google Colab).`, `backend/main.py --------------- FastAPI application entry point for the Tanita`, `Liveness probe that also reports whether the HMR2 model has finished     loadin` to the rest of the system?**
  _85 weakly-connected nodes found - possible documentation gaps or missing edges._
- **Should `Community 0` be split into smaller, more focused modules?**
  _Cohesion score 0.05353535353535353 - nodes in this community are weakly interconnected._
- **Should `Community 1` be split into smaller, more focused modules?**
  _Cohesion score 0.07057057057057058 - nodes in this community are weakly interconnected._