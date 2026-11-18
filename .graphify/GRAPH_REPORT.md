# Graph Report - .  (2026-11-13)

## Corpus Check
- 94 files · ~56,521 words
- Verdict: corpus is large enough that graph structure adds value.

## Summary
- 775 nodes · 1238 edges · 55 communities detected
- Extraction: 92% EXTRACTED · 8% INFERRED · 0% AMBIGUOUS · INFERRED: 104 edges (avg confidence: 0.5)
- Token cost: 0 input · 0 output
- Edge kinds: contains: 320 · calls: 264 · rationale_for: 213 · method: 135 · MODIFIES: 121 · uses: 104 · imports: 46 · inherits: 16 · ON_BRANCH: 9 · PARENT_OF: 8 · imports_from: 2


## Input Scope
- Requested: auto
- Resolved: committed (source: default-auto)
- Included files: 94 · Candidates: 255
- Excluded: 5 untracked · 68992 ignored · 0 sensitive · 0 missing committed
- Recommendation: Use --scope all or graphify.yaml inputs.corpus for a knowledge-base folder.

## Graph Freshness
- Built from Git commit: `9b9252a`
- Compare this hash to `git rev-parse HEAD` before trusting freshness-sensitive graph output.
## God Nodes (most connected - your core abstractions)
1. `PoseProcessingError` - 22 edges
2. `VideoProcessingError` - 19 edges
3. `VideoCaptureEngine` - 19 edges
4. `VideoValidationError` - 18 edges
5. `Camera` - 16 edges
6. `CalibrationFailure` - 16 edges
7. `PoseProcessingEngine` - 16 edges
8. `InferenceEngine` - 15 edges
9. `ViewpointEstimationError` - 15 edges
10. `CalibrationError` - 14 edges

## Surprising Connections (you probably didn't know these)
- `Synthetic camera geometry experiment; no human model or video is used.  Run: p` --uses--> `Camera`  [INFERRED]
  scripts/run_synthetic_multiview.py → backend/services/camera_geometry.py
- `Regression tests for forward-only temporal frame sampling.  Run from the proje` --uses--> `VideoCaptureEngine`  [INFERRED]
  scripts/test_sequential_sampling.py → backend/services/video_processor.py
- `FakeCapture` --uses--> `VideoCaptureEngine`  [INFERRED]
  scripts/test_sequential_sampling.py → backend/services/video_processor.py
- `SequentialSamplingTests` --uses--> `VideoCaptureEngine`  [INFERRED]
  scripts/test_sequential_sampling.py → backend/services/video_processor.py
- `AuthorizationRequest` --uses--> `VideoProcessingError`  [INFERRED]
  backend/routes/capture.py → backend/services/video_processor.py

## Communities

### Community 0 - "Community 0"
Cohesion: 0.07
Nodes (31): board_points(), calibrate_images(), calibrate_observations(), CalibrationFailure, detect_corners(), load_calibration(), Isolated OpenCV intrinsic calibration; no models or capture routes are used., Detect and refine checkerboard inner corners in one grayscale image. (+23 more)

### Community 1 - "Community 1"
Cohesion: 0.07
Nodes (18): FakeCapture, Regression tests for forward-only temporal frame sampling.  Run from the proje, SequentialSamplingTests, get_capture_engine(), Core CV pipeline: load video, sample evenly spaced candidate         frames, sc, Walk every frame via grab()+POS_MSEC to get a real duration., Return trustworthy fps/duration/total_frames/resolution., Decode forward to each target instead of seeking unreliable WebM timestamps. (+10 more)

### Community 2 - "Community 2"
Cohesion: 0.09
Nodes (33): get_capture_status(), get_frame_image(), get_frames(), process_video(), backend/routes/capture.py --------------------------- API routes for 360 video, Kicks off background frame extraction (FPS/resolution/duration     parsing, eve, Returns the current status of a capture job: "uploaded",     "processing", "com, Returns the current status of a capture job: "uploaded",     "processing", "com (+25 more)

### Community 3 - "Community 3"
Cohesion: 0.10
Nodes (23): CalibrationError, Camera, _points(), Model-independent pinhole cameras.  Convention: x_cam = R_world_to_camera @ x_, Transform 3D points from source camera coordinates to destination camera coordin, Euclidean pixel errors; callers must mask missing observations first., Invalid camera parameters or geometry., Return world-space camera origin and unit directions for pixels. (+15 more)

### Community 4 - "Community 4"
Cohesion: 0.08
Nodes (11): AuthorizationTests, Recording authorization tests using temporary synthetic data only., Subject-review contract tests with synthetic images and detections., SubjectReviewTests, VideoMeshApiTests, BatchTests, fake_decision(), fake_report() (+3 more)

### Community 5 - "Community 5"
Cohesion: 0.07
Nodes (14): VIEWPOINT_COLORS, VIEWPOINT_LABELS, BASE_URL, getCaptureStatus(), getFrameImageUrl(), getFrameMetadata(), processVideo(), uploadVideo() (+6 more)

### Community 6 - "Community 6"
Cohesion: 0.10
Nodes (32): get_pose_frame_asset(), get_pose_results(), get_pose_status(), get_viewpoint_coverage(), process_pose(), Returns the full processing_summary.json: per-frame person     detection, pose, Returns the full processing_summary.json: per-frame person     detection, pose, Returns the full processing_summary.json: per-frame person     detection, pose (+24 more)

### Community 7 - "Community 7"
Cohesion: 0.12
Nodes (19): fit_heldout_camera(), fit_shared_shape(), FitError, _objective(), Experimental, model-agnostic shared-shape fitting primitives.  No model assets, Report pixel and image-diagonal-normalized errors, excluding missing points., Fit one beta across at least two views; fixed poses, regularized cameras., Invalid data or unsuccessful optimization. (+11 more)

### Community 8 - "Community 8"
Cohesion: 0.11
Nodes (25): get_inference_status(), get_mesh(), backend/routes/inference.py ---------------------------- API routes for HMR2 i, Returns the current status of a job tracked by the local engine.      Returns, Returns the reconstructed OBJ mesh file for a completed job.      The mesh was, Accepts an uploaded image, validates it, and submits it to the local     HMR2 i, submit_inference(), EngineNotReadyError (+17 more)

### Community 9 - "Community 9"
Cohesion: 0.08
Nodes (11): BaseModel, AuthorizationRequest, QualityRequest, Opt-in, unvalidated capture quality report endpoints., CreateSession, Manual subject review endpoints., ReviewDecision, BatchRequest (+3 more)

### Community 10 - "Community 10"
Cohesion: 0.11
Nodes (14): get_pose_engine(), PoseProcessingEngine, Load processing_summary.json for a processed video, if present., Load processing_summary.json for a processed video, if present., Resolve a processed-frame asset (segmentation_mask.png,         pose_visualizat, Resolve a processed-frame asset (segmentation_mask.png,         pose_visualizat, Kick off background pose/segmentation processing for a video., Kick off background pose/segmentation processing for a video. (+6 more)

### Community 11 - "Community 11"
Cohesion: 0.10
Nodes (13): get_engine(), InferenceEngine, SMPL faces attribute path varies across 4D-Humans versions., Attempt to load the ViTDet/detectron2 person detector.         Returns None (fu, SMPL faces attribute path varies across 4D-Humans versions., Attempt to load the ViTDet/detectron2 person detector.         Returns None (fu, SMPL faces attribute path varies across 4D-Humans versions., Attempt to load the ViTDet/detectron2 person detector.         Returns None (fu (+5 more)

### Community 12 - "Community 12"
Cohesion: 0.13
Nodes (21): build_mesh_figure(), check_backend_health(), fetch_mesh_bytes(), main(), parse_obj(), poll_status(), frontend/app.py ---------------- Streamlit web application for the Tanita Huma, Parse a Wavefront OBJ file and return (vertices, faces).      Parameters (+13 more)

### Community 13 - "Community 13"
Cohesion: 0.12
Nodes (11): main(), Generate independent meshes using the existing single-image inference API., A simple orthographic vertex preview, not a surface-quality assessment., run_batch(), save_preview(), validate_obj(), BatchTests, fake_obj() (+3 more)

### Community 14 - "Community 14"
Cohesion: 0.10
Nodes (6): Tanitas backend package — local GPU inference (no Google Colab)., e1ee660 prototype phase-1, ALLOWED_TYPES, API route modules for the tanitas local-GPU backend., Local inference services for the tanitas backend (no external network calls)., Utility helpers for the tanitas local-GPU backend.

### Community 15 - "Community 15"
Cohesion: 0.14
Nodes (10): 06aee9c phase cheakpoint 1, 9b9252a working frames with view point, a9e9cee point 1 wokring as proto for 360, ALLOWED_TYPES, backend/routes/pose.py ------------------------- API routes for Module 3: Huma, Export authorized HMR2 diagnostics for one explicitly boxed frame., backend/services/pose_processor.py ------------------------------------- Modul, backend/services/video_processor.py ------------------------------------- Loca (+2 more)

### Community 16 - "Community 16"
Cohesion: 0.28
Nodes (17): create_session(), _frame(), frame_suggestion(), get_decision(), get_session(), _iou(), Explicit human subject review; no biometric identification., Restore only decisions whose persisted inputs still match the report and frames. (+9 more)

### Community 17 - "Community 17"
Cohesion: 0.17
Nodes (8): look_at(), main(), metrics(), plot_result(), Synthetic camera geometry experiment; no human model or video is used.  Run: p, scene(), MultiViewTests, Synthetic multi-view geometry and frame-selection tests.

### Community 18 - "Community 18"
Cohesion: 0.19
Nodes (8): base, boxStyle(), colors, DetectionBoxes(), DetectionSummary(), QualityChecks(), QualityReasons(), base

### Community 19 - "Community 19"
Cohesion: 0.30
Nodes (14): check_readiness(), get_asset(), get_run(), _persist(), _prepare_batch(), Background video-frame inference batches, using the existing inference engine., Collect independent blockers; submission's full gate remains authoritative., Apply the same image and reviewed-crop validation for preflight and submission. (+6 more)

### Community 20 - "Community 20"
Cohesion: 0.20
Nodes (4): Capture, JobTests, Pose, Background quality job tests using synthetic fixtures.

### Community 21 - "Community 21"
Cohesion: 0.25
Nodes (12): association(), check(), labels, mandatoryChecks, qualityChecks(), qualityGroup(), reasonText(), selectionBlockReason() (+4 more)

### Community 22 - "Community 22"
Cohesion: 0.18
Nodes (10): bbox_normalized(), centered(), fit_corresponding(), icp(), Model-independent mesh diagnostics and point-cloud alignment., Translation plus uniform max-extent scale; does not alter proportions., Symmetric nearest-vertex distances (not surface distances)., Least-squares proper rigid/similarity fit for explicit point correspondences. (+2 more)

### Community 23 - "Community 23"
Cohesion: 0.18
Nodes (8): health_check(), backend/main.py --------------- FastAPI application entry point for the Tanita, Liveness probe that also reports whether the HMR2 model has finished     loadin, Liveness probe that also reports whether the HMR2 model has finished     loadin, Liveness probe that also reports whether the HMR2 model has finished     loadin, # NOTE: load_dotenv() MUST run before any backend module is imported., Liveness probe that also reports whether the HMR2 model has finished     loadin, Liveness probe that also reports whether the HMR2 model has finished     loadin

### Community 24 - "Community 24"
Cohesion: 0.25
Nodes (8): main, 1a9dc19 first commit, 8f1a4d9 first commit, a26dcfb update backed added BMR, b06db69 first commit, f30bc43 working SMPL model, scripts/_generate_phase_b_plan.py ---------------------------------- One-shot, backend/services/inference_engine.py ------------------------------------- Loc

### Community 25 - "Community 25"
Cohesion: 0.35
Nodes (9): stem(), VideoMeshPanel(), BASE_URL, batchPayload(), checkMeshReadiness(), getMeshBatch(), meshAssetUrl(), request() (+1 more)

### Community 26 - "Community 26"
Cohesion: 0.56
Nodes (9): _fail(), main(), _ok(), scripts/test_integration.py ---------------------------- End-to-end integratio, _step(), test_backend_health(), test_download_mesh(), test_poll() (+1 more)

### Community 27 - "Community 27"
Cohesion: 0.27
Nodes (9): get_job(), get_report(), _hog_observations(), _nms_diagnostics(), Background, opt-in capture quality reports using existing extracted frames., HOG is a weak full-person detector, not reliable identity verification., Conservative IoU NMS for diagnostics only; input remains gate evidence.      I, start_quality() (+1 more)

### Community 28 - "Community 28"
Cohesion: 0.22
Nodes (1): MeshErrorBoundary

### Community 29 - "Community 29"
Cohesion: 0.33
Nodes (7): analyze_frames(), _features(), _iou(), _landmark_box(), QualityConfig, Additive capture quality diagnostics; does not change extraction or inference., Analyze all extracted frames; optional detections keyed by filename.      Dete

### Community 30 - "Community 30"
Cohesion: 0.33
Nodes (8): analyze_proxy_sequence(), assess_sequence(), _point(), Experimental fixed-camera sequence assessment; no physical yaw or reconstruction, Assess existing extracted frames; real capture suitability remains unconfirmed., Signed x/depth torso-axis angle, NOT validated body yaw.      Angles are relat, Analyze adjacent proxy angles; missing values or jumps split trajectories., torso_proxy()

### Community 31 - "Community 31"
Cohesion: 0.31
Nodes (3): fixture_pose(), QualityTests, Synthetic quality-policy tests: no real detector or model assets.

### Community 32 - "Community 32"
Cohesion: 0.39
Nodes (6): clone_repo(), download_checkpoint(), main(), print_summary(), scripts/download_models.py --------------------------- One-time setup script t, _run()

### Community 33 - "Community 33"
Cohesion: 0.36
Nodes (5): BASE_URL, checkHealth(), getJobStatus(), getMeshUrl(), submitImage()

### Community 34 - "Community 34"
Cohesion: 0.67
Nodes (5): attest(), _directory(), Recording-bound operator attestation; not authenticated proof of consent., require_authorization(), _source_hash()

### Community 35 - "Community 35"
Cohesion: 0.33
Nodes (2): MeshTests, Synthetic OBJ validation tests. No licensed model assets are required.

### Community 36 - "Community 36"
Cohesion: 0.40
Nodes (1): PHASE_CONFIG

### Community 37 - "Community 37"
Cohesion: 0.60
Nodes (4): main(), Experimental fixed-camera calibration and capture validation CLI., synthetic_demo(), visualize()

### Community 38 - "Community 38"
Cohesion: 0.70
Nodes (4): main(), Exercise the capture and pose HTTP APIs with a supplied video.  Example:, request(), wait_for_job()

### Community 39 - "Community 39"
Cohesion: 0.50
Nodes (3): poll(), process(), Test independent capture and pose jobs through an existing main backend.

### Community 40 - "Community 40"
Cohesion: 0.50
Nodes (4): Experimental fixed-camera capture metadata validation.  No camera pose, metric, Combine extracted-frame and pose metadata without modifying production outputs., validate_capture(), _xy()

### Community 41 - "Community 41"
Cohesion: 0.50
Nodes (4): export_obj(), Minimal geometry validation and exclusive OBJ export for experiments.  Dimensi, Validate then create a new OBJ; never overwrite an existing file., validate_mesh()

### Community 42 - "Community 42"
Cohesion: 0.67
Nodes (3): lap(), main(), Read-only analysis of existing extraction and source video; no inference.

### Community 43 - "Community 43"
Cohesion: 0.50
Nodes (3): export_frame(), Opt-in HMR2 diagnostic output; no production inference changes., Run an authorized explicit-box diagnostic on an already-loaded engine.      Th

### Community 44 - "Community 44"
Cohesion: 0.67
Nodes (1): Inspect sequential decoding and write a diagnostic contact sheet.  Usage: pyth

### Community 45 - "Community 45"
Cohesion: 0.67
Nodes (1): Print pose landmarks for selected frames without exporting images.

### Community 46 - "Community 46"
Cohesion: 0.67
Nodes (1): Validate a fresh HMR2 inference through the real backend API.  Run: python scr

### Community 47 - "Community 47"
Cohesion: 0.67
Nodes (1): Live browser-to-backend video mesh smoke test. No mocked requests.

### Community 48 - "Community 48"
Cohesion: 0.67
Nodes (1): Diagnose pose/viewpoint output on sequentially decoded video frames.  This byp

### Community 49 - "Community 49"
Cohesion: 0.67
Nodes (1): Exercise the actual frontend with local video and photo fixtures.  Requires Pl

### Community 50 - "Community 50"
Cohesion: 0.67
Nodes (1): Real browser capture/quality gate smoke test; no API mocking.

### Community 51 - "Community 51"
Cohesion: 0.67
Nodes (1): Live Stage 1.1 blocked-quality path; no mocked APIs.

### Community 52 - "Community 52"
Cohesion: 0.67
Nodes (1): Isolated Stage 1.2 UI validation. All APIs and images are synthetic fixtures.

### Community 53 - "Community 53"
Cohesion: 0.67
Nodes (1): Verify the video mesh API using explicitly authorized existing video frames.

### Community 54 - "Community 54"
Cohesion: 0.67
Nodes (1): Browser smoke test with mocked APIs and an existing OBJ fixture.

## Knowledge Gaps
- **181 isolated node(s):** `Tanitas backend package — local GPU inference (no Google Colab).`, `backend/main.py --------------- FastAPI application entry point for the Tanita`, `Liveness probe that also reports whether the HMR2 model has finished     loadin`, `# NOTE: load_dotenv() MUST run before any backend module is imported.`, `API route modules for the tanitas local-GPU backend.` (+176 more)
  These have ≤1 connection - possible missing edges or undocumented components.
- **Thin community `Community 28`** (1 nodes): `MeshErrorBoundary`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 35`** (2 nodes): `MeshTests`, `Synthetic OBJ validation tests. No licensed model assets are required.`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 36`** (1 nodes): `PHASE_CONFIG`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 44`** (1 nodes): `Inspect sequential decoding and write a diagnostic contact sheet.  Usage: pyth`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 45`** (1 nodes): `Print pose landmarks for selected frames without exporting images.`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 46`** (1 nodes): `Validate a fresh HMR2 inference through the real backend API.  Run: python scr`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 47`** (1 nodes): `Live browser-to-backend video mesh smoke test. No mocked requests.`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 48`** (1 nodes): `Diagnose pose/viewpoint output on sequentially decoded video frames.  This byp`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 49`** (1 nodes): `Exercise the actual frontend with local video and photo fixtures.  Requires Pl`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 50`** (1 nodes): `Real browser capture/quality gate smoke test; no API mocking.`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 51`** (1 nodes): `Live Stage 1.1 blocked-quality path; no mocked APIs.`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 52`** (1 nodes): `Isolated Stage 1.2 UI validation. All APIs and images are synthetic fixtures.`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 53`** (1 nodes): `Verify the video mesh API using explicitly authorized existing video frames.`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 54`** (1 nodes): `Browser smoke test with mocked APIs and an existing OBJ fixture.`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.

## Suggested Questions
_Questions this graph is uniquely positioned to answer:_

- **Why does `VideoCaptureEngine` connect `Community 1` to `Community 15`?**
  _High betweenness centrality (0.048) - this node is a cross-community bridge._
- **Why does `VideoProcessingError` connect `Community 2` to `Community 9`, `Community 15`, `Community 1`?**
  _High betweenness centrality (0.044) - this node is a cross-community bridge._
- **Why does `InferenceEngine` connect `Community 11` to `Community 24`, `Community 8`?**
  _High betweenness centrality (0.044) - this node is a cross-community bridge._
- **Are the 15 inferred relationships involving `PoseProcessingError` (e.g. with `backend/routes/pose.py ------------------------- API routes for Module 3: Huma` and `Returns the full processing_summary.json: per-frame person     detection, pose`) actually correct?**
  _`PoseProcessingError` has 15 INFERRED edges - model-reasoned connections that need verification._
- **Are the 12 inferred relationships involving `VideoProcessingError` (e.g. with `AuthorizationRequest` and `backend/routes/capture.py --------------------------- API routes for 360 video`) actually correct?**
  _`VideoProcessingError` has 12 INFERRED edges - model-reasoned connections that need verification._
- **What connects `Tanitas backend package — local GPU inference (no Google Colab).`, `backend/main.py --------------- FastAPI application entry point for the Tanita`, `Liveness probe that also reports whether the HMR2 model has finished     loadin` to the rest of the system?**
  _181 weakly-connected nodes found - possible documentation gaps or missing edges._
- **Should `Community 0` be split into smaller, more focused modules?**
  _Cohesion score 0.06859903381642513 - nodes in this community are weakly interconnected._