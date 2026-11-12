# Video-to-mesh UI integration report

## Implementation

The capture workspace now displays **Step 3 — 3D Mesh Reconstruction** after extraction and pose processing completes. Users can inspect frame thumbnails, select up to 12 frames, and start an independent mesh batch. Initial selection prefers frames that are not flagged blurry and have all tracked pose landmarks visible. When all or most selected frames are blurry or have unknown blur status, the panel displays a warning. If no frames satisfy the default criteria, users can select frames manually.

The results panel polls the batch status, displays each frame's queued, processing, completed or failed state, and retains successful results when another frame fails. It reuses the existing interactive OBJ viewer with rotate, zoom and pan; the selector switches between completed meshes. The source image is a snapshot stored with the run. Viewpoint labels and scores come from the existing unvalidated heuristic, not a reliable orientation classifier.

The UI states: **Individual 3D Meshes — Multi-view reconstruction pending.** The predictions are independent; no mesh fusion, measurement validation or body-composition estimation is performed.

## Backend

`POST /api/video-mesh/runs` accepts an extracted `video_id` and 1–12 frame filenames. The service checks UUIDs and filenames against extraction metadata, snapshots images into a uniquely identified run, and starts a background worker. The worker submits frames sequentially through the existing `InferenceEngine.submit_job`, polls each job, and copies successful OBJ files to the run. A failed frame does not prevent later frames from being attempted.

`GET /api/video-mesh/runs/{run_id}` returns per-frame status and metadata. Run-scoped frame snapshots and completed OBJ files are served through `/api/video-mesh/runs/{run_id}/frame/{filename}` and `/api/video-mesh/runs/{run_id}/mesh/{filename}`. Completed run metadata is persisted under `VIDEO_MESH_OUTPUT_DIR`, defaulting to `outputs/video_mesh_ui/`; unique run IDs avoid overwriting previous runs. Interrupted in-memory jobs are reported as interrupted after a server restart.

The existing single-photo route and HMR2 inference logic were not modified. No new model or rendering dependency was installed.

## Tests and observations

- **Backend regressions:** 35 tests passed, including the new batch worker and HTTP tests, existing video batch client, camera/pose-related synthetic tests, shared-shape tests and sequential extraction tests. Fake-engine tests covered partial failures, validation, snapshot assets and preservation of earlier runs.
- **Frontend production build:** `npm --prefix frontend run build` passed. Vite reported a chunk-size warning for the approximately 1 MB minified JavaScript bundle.
- **Browser smoke test:** `scripts/validate_video_mesh_ui.py` passed with mocked capture, pose and batch API responses and a previously generated OBJ fixture. It exercised frame selection, an all-blurry warning, a failed frame, mesh selection and two OBJ fetches, with no reported page errors. It did not verify a live upload-to-pose-to-inference workflow or anatomical quality. A canvas and OBJ fetches were observed; independent visual validation of the rendered surface was not performed.
- **Real-engine batch API:** run `537529b6-76f0-44a9-9f3c-2a46a2e78144` processed existing frames `frame_003.jpg` and `frame_015.jpg` on CUDA. Both completed and their downloaded OBJ files passed basic structural checks, each with 6,890 vertices and 13,776 faces. This test used isolated output directories and the full-image detector fallback; the source frames were flagged blurry. It does not verify correct subject association or reconstruction accuracy.
- **Single-photo regression:** the existing inference API completed a fresh image job on CUDA and returned a structurally valid OBJ with 6,890 vertices and 13,776 faces.

The user confirmed authorization for independent HMR2/SMPL inference on these video frames. This does not imply authorization for multi-view fitting.

## Files changed

- `backend/services/video_mesh_batch.py` — asynchronous run management and run-scoped assets.
- `backend/routes/video_mesh.py` and `backend/main.py` — new batch routes and registration.
- `frontend/src/services/videoMeshApi.js` — batch API client.
- `frontend/src/components/VideoMeshPanel.jsx` and `frontend/src/components/CaptureWorkspace.jsx` — selection, progress and viewer integration.
- `backend/tests/test_video_mesh_batch.py`, `backend/tests/test_video_mesh_api.py` — worker and HTTP tests.
- `scripts/validate_video_mesh_ui.py`, `scripts/validate_video_mesh_batch_real.py` — browser and real-engine verification helpers.
- `docs/VIDEO_TO_MESH_UI_INTEGRATION_REPORT.md` — this report.

## Limitations and operational notes

- A full end-to-end browser test with real video upload, MediaPipe processing, the new batch endpoint and live HMR2 was not performed in this milestone. The browser and real-engine checks were separate.
- Run status is persisted and retrievable by run ID, but the UI does not yet list or restore previous runs after navigation or reload. Background jobs are in-process and are not resumed after a server restart.
- The batch lock serializes only the new video batches within one process. Single-photo inference and multiple backend processes are not coordinated by that lock.
- The new endpoints follow the existing local POC's security model; authentication and authorization controls were not added. Run artifacts and source images contain sensitive personal data.
- Pose-based full-body and viewpoint flags are heuristics. Partial bystanders, blur, occlusion and detector fallback can compromise outputs. Review images and meshes before further use.
- A structurally valid OBJ is not proof of anatomical correctness, calibrated measurements or 360° coverage.
