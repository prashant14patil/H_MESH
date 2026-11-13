# Graph Report - .  (2026-11-13)

## Corpus Check
- Corpus is ~42,061 words - fits in a single context window. You may not need a graph.

## Summary
- 601 nodes · 919 edges · 40 communities detected
- Extraction: 92% EXTRACTED · 8% INFERRED · 0% AMBIGUOUS · INFERRED: 74 edges (avg confidence: 0.5)
- Token cost: 0 input · 0 output
- Edge kinds: contains: 233 · calls: 195 · rationale_for: 177 · method: 105 · MODIFIES: 89 · uses: 74 · imports: 19 · inherits: 10 · ON_BRANCH: 8 · PARENT_OF: 7 · imports_from: 2


## Input Scope
- Requested: auto
- Resolved: committed (source: default-auto)
- Included files: 74 · Candidates: 192
- Excluded: 15 untracked · 67546 ignored · 0 sensitive · 0 missing committed
- Recommendation: Use --scope all or graphify.yaml inputs.corpus for a knowledge-base folder.

## Graph Freshness
- Built from Git commit: `a9e9cee`
- Compare this hash to `git rev-parse HEAD` before trusting freshness-sensitive graph output.
## God Nodes (most connected - your core abstractions)
1. `VideoCaptureEngine` - 19 edges
2. `PoseProcessingError` - 17 edges
3. `Camera` - 16 edges
4. `CalibrationFailure` - 16 edges
5. `PoseProcessingEngine` - 16 edges
6. `InferenceEngine` - 15 edges
7. `CalibrationError` - 14 edges
8. `VideoProcessingError` - 13 edges
9. `VideoValidationError` - 12 edges
10. `FitError` - 11 edges

## Surprising Connections (you probably didn't know these)
- `Synthetic camera geometry experiment; no human model or video is used.  Run: p` --uses--> `Camera`  [INFERRED]
  scripts/run_synthetic_multiview.py → backend/services/camera_geometry.py
- `Regression tests for forward-only temporal frame sampling.  Run from the proje` --uses--> `VideoCaptureEngine`  [INFERRED]
  scripts/test_sequential_sampling.py → backend/services/video_processor.py
- `FakeCapture` --uses--> `VideoCaptureEngine`  [INFERRED]
  scripts/test_sequential_sampling.py → backend/services/video_processor.py
- `SequentialSamplingTests` --uses--> `VideoCaptureEngine`  [INFERRED]
  scripts/test_sequential_sampling.py → backend/services/video_processor.py
- `Folder-based checkerboard calibration reports; no production routes or models.` --uses--> `CalibrationFailure`  [INFERRED]
  backend/services/real_camera_calibration.py → backend/services/checkerboard_calibration.py

## Communities

### Community 0 - "Community 0"
Cohesion: 0.07
Nodes (31): board_points(), calibrate_images(), calibrate_observations(), CalibrationFailure, detect_corners(), load_calibration(), Isolated OpenCV intrinsic calibration; no models or capture routes are used., Detect and refine checkerboard inner corners in one grayscale image. (+23 more)

### Community 1 - "Community 1"
Cohesion: 0.07
Nodes (18): FakeCapture, Regression tests for forward-only temporal frame sampling.  Run from the proje, SequentialSamplingTests, get_capture_engine(), backend/services/video_processor.py ------------------------------------- Loca, Core CV pipeline: load video, sample evenly spaced candidate         frames, sc, Walk every frame via grab()+POS_MSEC to get a real duration., Return trustworthy fps/duration/total_frames/resolution. (+10 more)

### Community 2 - "Community 2"
Cohesion: 0.10
Nodes (23): CalibrationError, Camera, _points(), Model-independent pinhole cameras.  Convention: x_cam = R_world_to_camera @ x_, Transform 3D points from source camera coordinates to destination camera coordin, Euclidean pixel errors; callers must mask missing observations first., Invalid camera parameters or geometry., Return world-space camera origin and unit directions for pixels. (+15 more)

### Community 3 - "Community 3"
Cohesion: 0.10
Nodes (28): get_capture_status(), get_frame_image(), get_frames(), process_video(), backend/routes/capture.py --------------------------- API routes for 360 video, Returns the current status of a capture job: "uploaded",     "processing", "com, Returns the full metadata.json written by frame extraction:     video info (fps, Serves a single extracted frame (e.g. ``frame_001.jpg``) as a JPEG     image, u (+20 more)

### Community 4 - "Community 4"
Cohesion: 0.12
Nodes (19): fit_heldout_camera(), fit_shared_shape(), FitError, _objective(), Experimental, model-agnostic shared-shape fitting primitives.  No model assets, Report pixel and image-diagonal-normalized errors, excluding missing points., Fit one beta across at least two views; fixed poses, regularized cameras., Invalid data or unsuccessful optimization. (+11 more)

### Community 5 - "Community 5"
Cohesion: 0.11
Nodes (27): get_pose_frame_asset(), get_pose_results(), get_pose_status(), get_viewpoint_coverage(), process_pose(), backend/routes/pose.py ------------------------- API routes for Module 3: Huma, Returns the full processing_summary.json: per-frame person     detection, pose, Returns the full processing_summary.json: per-frame person     detection, pose (+19 more)

### Community 6 - "Community 6"
Cohesion: 0.11
Nodes (25): get_inference_status(), get_mesh(), backend/routes/inference.py ---------------------------- API routes for HMR2 i, Returns the current status of a job tracked by the local engine.      Returns, Returns the reconstructed OBJ mesh file for a completed job.      The mesh was, Accepts an uploaded image, validates it, and submits it to the local     HMR2 i, submit_inference(), EngineNotReadyError (+17 more)

### Community 7 - "Community 7"
Cohesion: 0.10
Nodes (15): get_pose_engine(), PoseProcessingEngine, backend/services/pose_processor.py ------------------------------------- Modul, Load processing_summary.json for a processed video, if present., Load processing_summary.json for a processed video, if present., Resolve a processed-frame asset (segmentation_mask.png,         pose_visualizat, Resolve a processed-frame asset (segmentation_mask.png,         pose_visualizat, Kick off background pose/segmentation processing for a video. (+7 more)

### Community 8 - "Community 8"
Cohesion: 0.10
Nodes (13): get_engine(), InferenceEngine, SMPL faces attribute path varies across 4D-Humans versions., Attempt to load the ViTDet/detectron2 person detector.         Returns None (fu, SMPL faces attribute path varies across 4D-Humans versions., Attempt to load the ViTDet/detectron2 person detector.         Returns None (fu, SMPL faces attribute path varies across 4D-Humans versions., Attempt to load the ViTDet/detectron2 person detector.         Returns None (fu (+5 more)

### Community 9 - "Community 9"
Cohesion: 0.13
Nodes (21): build_mesh_figure(), check_backend_health(), fetch_mesh_bytes(), main(), parse_obj(), poll_status(), frontend/app.py ---------------- Streamlit web application for the Tanita Huma, Parse a Wavefront OBJ file and return (vertices, faces).      Parameters (+13 more)

### Community 10 - "Community 10"
Cohesion: 0.12
Nodes (11): main(), Generate independent meshes using the existing single-image inference API., A simple orthographic vertex preview, not a surface-quality assessment., run_batch(), save_preview(), validate_obj(), BatchTests, fake_obj() (+3 more)

### Community 11 - "Community 11"
Cohesion: 0.16
Nodes (9): HTTP tests against an ephemeral server with a fake inference engine., VideoMeshApiTests, BatchTests, fake_decision(), fake_report(), fake_session(), FakeCapture, FakeEngine (+1 more)

### Community 12 - "Community 12"
Cohesion: 0.10
Nodes (6): Tanitas backend package — local GPU inference (no Google Colab)., e1ee660 prototype phase-1, ALLOWED_TYPES, API route modules for the tanitas local-GPU backend., Local inference services for the tanitas backend (no external network calls)., Utility helpers for the tanitas local-GPU backend.

### Community 13 - "Community 13"
Cohesion: 0.11
Nodes (7): a9e9cee point 1 wokring as proto for 360, Inspect sequential decoding and write a diagnostic contact sheet.  Usage: pyth, Print pose landmarks for selected frames without exporting images., Validate a fresh HMR2 inference through the real backend API.  Run: python scr, Live browser-to-backend video mesh smoke test. No mocked requests., Browser smoke test with mocked APIs and an existing OBJ fixture., Synthetic OBJ validation tests. No licensed model assets are required.

### Community 14 - "Community 14"
Cohesion: 0.18
Nodes (12): 06aee9c phase cheakpoint 1, BASE_URL, getCaptureStatus(), getFrameMetadata(), processVideo(), uploadVideo(), BASE_URL, getPoseFrameAssetUrl() (+4 more)

### Community 15 - "Community 15"
Cohesion: 0.17
Nodes (8): look_at(), main(), metrics(), plot_result(), Synthetic camera geometry experiment; no human model or video is used.  Run: p, scene(), MultiViewTests, Synthetic multi-view geometry and frame-selection tests.

### Community 16 - "Community 16"
Cohesion: 0.18
Nodes (10): bbox_normalized(), centered(), fit_corresponding(), icp(), Model-independent mesh diagnostics and point-cloud alignment., Translation plus uniform max-extent scale; does not alter proportions., Symmetric nearest-vertex distances (not surface distances)., Least-squares proper rigid/similarity fit for explicit point correspondences. (+2 more)

### Community 17 - "Community 17"
Cohesion: 0.18
Nodes (8): health_check(), backend/main.py --------------- FastAPI application entry point for the Tanita, Liveness probe that also reports whether the HMR2 model has finished     loadin, Liveness probe that also reports whether the HMR2 model has finished     loadin, Liveness probe that also reports whether the HMR2 model has finished     loadin, # NOTE: load_dotenv() MUST run before any backend module is imported., Liveness probe that also reports whether the HMR2 model has finished     loadin, Liveness probe that also reports whether the HMR2 model has finished     loadin

### Community 18 - "Community 18"
Cohesion: 0.25
Nodes (8): main, 1a9dc19 first commit, 8f1a4d9 first commit, a26dcfb update backed added BMR, b06db69 first commit, f30bc43 working SMPL model, scripts/_generate_phase_b_plan.py ---------------------------------- One-shot, backend/services/inference_engine.py ------------------------------------- Loc

### Community 19 - "Community 19"
Cohesion: 0.20
Nodes (2): VIEWPOINT_COLORS, VIEWPOINT_LABELS

### Community 20 - "Community 20"
Cohesion: 0.31
Nodes (8): stem(), VideoMeshPanel(), getFrameImageUrl(), BASE_URL, getMeshBatch(), meshAssetUrl(), request(), startMeshBatch()

### Community 21 - "Community 21"
Cohesion: 0.56
Nodes (9): _fail(), main(), _ok(), scripts/test_integration.py ---------------------------- End-to-end integratio, _step(), test_backend_health(), test_download_mesh(), test_poll() (+1 more)

### Community 22 - "Community 22"
Cohesion: 0.44
Nodes (9): get_asset(), get_run(), _persist(), Background video-frame inference batches, using the existing inference engine., _root(), _serialized_worker(), start_batch(), valid_uuid() (+1 more)

### Community 23 - "Community 23"
Cohesion: 0.22
Nodes (1): MeshErrorBoundary

### Community 24 - "Community 24"
Cohesion: 0.33
Nodes (8): analyze_proxy_sequence(), assess_sequence(), _point(), Experimental fixed-camera sequence assessment; no physical yaw or reconstruction, Assess existing extracted frames; real capture suitability remains unconfirmed., Signed x/depth torso-axis angle, NOT validated body yaw.      Angles are relat, Analyze adjacent proxy angles; missing values or jumps split trajectories., torso_proxy()

### Community 25 - "Community 25"
Cohesion: 0.39
Nodes (6): clone_repo(), download_checkpoint(), main(), print_summary(), scripts/download_models.py --------------------------- One-time setup script t, _run()

### Community 26 - "Community 26"
Cohesion: 0.36
Nodes (5): BASE_URL, checkHealth(), getJobStatus(), getMeshUrl(), submitImage()

### Community 27 - "Community 27"
Cohesion: 0.29
Nodes (3): BaseModel, BatchRequest, Endpoints for batches of independent single-frame meshes.

### Community 29 - "Community 29"
Cohesion: 0.40
Nodes (1): PHASE_CONFIG

### Community 30 - "Community 30"
Cohesion: 0.60
Nodes (4): main(), Experimental fixed-camera calibration and capture validation CLI., synthetic_demo(), visualize()

### Community 31 - "Community 31"
Cohesion: 0.70
Nodes (4): main(), Exercise the capture and pose HTTP APIs with a supplied video.  Example:, request(), wait_for_job()

### Community 32 - "Community 32"
Cohesion: 0.50
Nodes (3): poll(), process(), Test independent capture and pose jobs through an existing main backend.

### Community 33 - "Community 33"
Cohesion: 0.50
Nodes (4): Experimental fixed-camera capture metadata validation.  No camera pose, metric, Combine extracted-frame and pose metadata without modifying production outputs., validate_capture(), _xy()

### Community 34 - "Community 34"
Cohesion: 0.50
Nodes (4): export_obj(), Minimal geometry validation and exclusive OBJ export for experiments.  Dimensi, Validate then create a new OBJ; never overwrite an existing file., validate_mesh()

### Community 35 - "Community 35"
Cohesion: 0.67
Nodes (3): lap(), main(), Read-only analysis of existing extraction and source video; no inference.

### Community 36 - "Community 36"
Cohesion: 0.50
Nodes (1): MeshTests

### Community 37 - "Community 37"
Cohesion: 0.67
Nodes (1): ALLOWED_TYPES

### Community 38 - "Community 38"
Cohesion: 0.67
Nodes (1): Diagnose pose/viewpoint output on sequentially decoded video frames.  This byp

### Community 39 - "Community 39"
Cohesion: 0.67
Nodes (1): Exercise the actual frontend with local video and photo fixtures.  Requires Pl

### Community 40 - "Community 40"
Cohesion: 0.67
Nodes (1): Verify the video mesh API using explicitly authorized existing video frames.

## Knowledge Gaps
- **149 isolated node(s):** `Tanitas backend package — local GPU inference (no Google Colab).`, `backend/main.py --------------- FastAPI application entry point for the Tanita`, `Liveness probe that also reports whether the HMR2 model has finished     loadin`, `# NOTE: load_dotenv() MUST run before any backend module is imported.`, `API route modules for the tanitas local-GPU backend.` (+144 more)
  These have ≤1 connection - possible missing edges or undocumented components.
- **Thin community `Community 19`** (2 nodes): `VIEWPOINT_COLORS`, `VIEWPOINT_LABELS`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 23`** (1 nodes): `MeshErrorBoundary`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 29`** (1 nodes): `PHASE_CONFIG`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 36`** (1 nodes): `MeshTests`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 37`** (1 nodes): `ALLOWED_TYPES`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 38`** (1 nodes): `Diagnose pose/viewpoint output on sequentially decoded video frames.  This byp`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 39`** (1 nodes): `Exercise the actual frontend with local video and photo fixtures.  Requires Pl`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 40`** (1 nodes): `Verify the video mesh API using explicitly authorized existing video frames.`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.

## Suggested Questions
_Questions this graph is uniquely positioned to answer:_

- **Why does `VideoCaptureEngine` connect `Community 1` to `Community 3`?**
  _High betweenness centrality (0.064) - this node is a cross-community bridge._
- **Why does `InferenceEngine` connect `Community 8` to `Community 18`, `Community 6`?**
  _High betweenness centrality (0.054) - this node is a cross-community bridge._
- **Are the 3 inferred relationships involving `VideoCaptureEngine` (e.g. with `FakeCapture` and `Regression tests for forward-only temporal frame sampling.  Run from the proje`) actually correct?**
  _`VideoCaptureEngine` has 3 INFERRED edges - model-reasoned connections that need verification._
- **Are the 10 inferred relationships involving `PoseProcessingError` (e.g. with `backend/routes/pose.py ------------------------- API routes for Module 3: Huma` and `Returns the full processing_summary.json: per-frame person     detection, pose`) actually correct?**
  _`PoseProcessingError` has 10 INFERRED edges - model-reasoned connections that need verification._
- **Are the 8 inferred relationships involving `Camera` (e.g. with `Synthetic camera geometry experiment; no human model or video is used.  Run: p` and `Synthetic/general multi-view geometry. No model or camera poses are inferred.`) actually correct?**
  _`Camera` has 8 INFERRED edges - model-reasoned connections that need verification._
- **What connects `Tanitas backend package — local GPU inference (no Google Colab).`, `backend/main.py --------------- FastAPI application entry point for the Tanita`, `Liveness probe that also reports whether the HMR2 model has finished     loadin` to the rest of the system?**
  _149 weakly-connected nodes found - possible documentation gaps or missing edges._
- **Should `Community 0` be split into smaller, more focused modules?**
  _Cohesion score 0.06859903381642513 - nodes in this community are weakly interconnected._