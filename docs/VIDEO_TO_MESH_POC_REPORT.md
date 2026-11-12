# Video-to-mesh POC report

## Result

The user confirmed authorization to use the installed HMR2 checkpoint and SMPL model for independent single-image inference on video frames. This confirmation does not extend to multi-view fitting.

The capture at `data/production_validation/ecff78fe-773e-4eb8-a785-c356e3be4d19` contains **30 extracted frames** from an 8.687-second, 1280×720 recording. Ten frames were manually selected from a contact sheet. The existing single-image inference API generated **10 independent OBJ files from 10 selections**, with no reported batch failures. Batch elapsed time was **10.797 seconds**, excluding model startup and the separate single-frame regression.

Each saved OBJ was re-read and checked for finite vertex coordinates, at least 100 vertices and faces, and face indices within range. All ten have **6,890 vertices and 13,776 faces**. These structural checks do not establish anatomical accuracy, correct subject association or suitability for reconstruction.

## Selection and quality

Selection details, timestamps, subjective viewpoint confidence, blur scores, missing landmarks and reasons are in:

- `outputs/video_to_mesh/review_ecff78/selection.json`
- `outputs/video_to_mesh/ecff78_batch_001/results.json`

Selected extracted frames: 003, 006, 008, 010, 012, 015, 017, 020, 024 and 027. Numbers are extracted-frame indices, not original video frame numbers. Labels are tentative visual judgments; `profile_a` and `profile_b` distinguish opposing profiles without asserting anatomical left or right.

**All selected frames were marked blurry by the existing threshold.** Several profile frames had missing or low-visibility landmarks and were included for view diversity despite those warnings. Partial bystanders are visible at the image edges. The existing inference engine may use a full-image fallback if its person detector is unavailable; the API output does not identify the detector path or establish which person was reconstructed. Review the outputs before using them for any further work.

## Output files and preview

Run directory: `outputs/video_to_mesh/ecff78_batch_001/`

- `selected_frames/` — copies of the selected source JPEGs.
- `meshes/` — ten separate OBJ files.
- `previews/` — static source-image/OBJ-vertex-cloud comparisons.
- `results.json` — per-frame status, job ID, time, selection rationale and output paths.
- `report.md` — batch summary.

Open the JPG files in `previews/` to inspect source/vertex projections. These are **not surface renders** or an anatomical quality assessment. Open individual OBJ files in an OBJ-compatible 3D viewer to inspect the surface. The first preview was visually inspected and shows a human-shaped vertex cloud; the other nine previews exist but were not independently reviewed for subject association or anatomical quality. Existing preview captions may have low contrast; the helper was improved for future runs.

The source contact sheet is `outputs/video_to_mesh/review_ecff78/contact.jpg`. These images and derived meshes are sensitive personal data; handle them according to the applicable privacy policy.

## API and tests

The existing frontend and batch client use `POST /api/inference/submit` with a JPEG or PNG multipart field named `image`, `GET /api/inference/status/{job_id}` for polling, and `GET /api/inference/mesh/{job_id}` to retrieve the OBJ. No inference engine, HMR2 architecture or frontend code was modified.

A fresh single-image regression on `frame_003.jpg` completed through the existing API on CUDA and returned an OBJ with 6,890 vertices and 13,776 faces. The ten-frame batch then completed through the same API. The model startup used a dedicated local backend at `127.0.0.1:18901` and separate engine output under `outputs/video_to_mesh/backend_ecff78/`.

The mocked batch tests verify that one failed job does not stop a subsequent successful job, invalid OBJ data is rejected, path traversal is rejected and an existing run directory is not overwritten. The real batch itself had no observed failures.

```text
python -m unittest backend.tests.test_video_to_mesh backend.tests.test_real_camera_calibration backend.tests.test_fixed_camera_validation backend.tests.test_camera_geometry backend.tests.test_multiview_geometry backend.tests.test_shared_shape_fitter backend.tests.test_spike_mesh scripts.test_sequential_sampling -v
Ran 32 tests: OK
```

The batch-client tests were rerun after the preview-caption adjustment: 3 passed.

## Limitations and next steps

- These are **independent single-image HMR2 predictions**, not one fused or unified 360° mesh. No vertices were combined and no multi-view fitting was performed.
- Do not interpret the results as validated metric body dimensions, anatomical accuracy or RGB-derived internal body composition.
- Viewpoint labels and confidence are subjective contact-sheet observations, not validated orientation classifications. Blur, occlusion and possible bystander interference may degrade results.
- Review every mesh and its source frame, verify subject association, and capture a sharper recording with a single person and complete body framing. A unified mesh requires a separately authorized and validated reconstruction method.

## Files added

- `scripts/video_to_mesh.py` — isolated sequential API batch client.
- `backend/tests/test_video_to_mesh.py` — mocked batch and OBJ validation tests.
- `docs/VIDEO_TO_MESH_POC_REPORT.md` — this report.

The separately requested controlled-rotation validator was interrupted by this task; `backend/services/controlled_rotation.py` was created before the task switch but was not used or tested for this mesh batch.
