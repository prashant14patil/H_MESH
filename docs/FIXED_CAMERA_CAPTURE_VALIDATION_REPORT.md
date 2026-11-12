# Fixed-camera capture calibration and validation

## Scope and status

This is an isolated capture-validation foundation, not a validated human reconstruction. No SMPL or HMR2 assets were loaded, executed, copied, or downloaded. Existing production capture, pose, viewpoint, inference, geometry, and mesh-export implementations were not changed.

**Real camera calibration is pending.** No clearly named checkerboard image set was found under `data` or `docs`. Calibration numbers produced by the demonstration are synthetic and must not be used for a real camera.

## Existing architecture and compatibility

- `camera_geometry.py` uses a distortion-free pinhole camera with world-to-camera convention `Xc = R Xw + t`, +X right, +Y down, +Z forward.
- `multiview_geometry.py` triangulates stationary world points seen by multiple cameras. Its moving-camera assumptions do not directly apply to a rotating person recorded by one fixed camera; the new validation workflow does not use it to triangulate capture frames.
- `video_processor.py` records frame timestamps, blur scores and flags, and comparison scores; it already removes some duplicate candidates. `select_frames` preserves metadata and requires an explicit blur override.
- `pose_processor.py` provides detection, 13 named landmarks, visibility and segmentation metadata. A visibility flag alone cannot establish that the person remains completely inside the image.
- `viewpoint_estimator.py` provides unvalidated heuristic labels and scores. In particular, an ambiguous side can be labelled `intermediate`; the experimental validator maps that case to `unknown`.

## Implemented experimental components

### Checkerboard calibration

`backend/services/checkerboard_calibration.py` detects inner corners in explicitly supplied images with OpenCV, refines them, and estimates camera intrinsics and distortion using `calibrateCamera`. The serialized JSON includes schema version, image dimensions, camera matrix, distortion coefficients, provenance, board dimensions and square size, view count, per-view reprojection errors and overall RMS. It does not save checkerboard poses as camera motion.

The utility checks input dimensions, resolution consistency, minimum detections, finite parameters, image bounds, RMS threshold, and simple differences/estimated tilt between board views. These checks are screening heuristics, **not proof that intrinsics or distortion are accurate**. A low training reprojection error does not replace independent validation with diverse held-out images. Save uses exclusive creation; load checks an optional exact image resolution.

`undistort_pixels` explicitly converts distorted pixel coordinates into undistorted pixels using the same intrinsic matrix. The existing pinhole camera functions do not model distortion, so callers must use this step before applying them to distorted observations.

### Fixed-camera metadata validation

`backend/services/fixed_camera_validation.py` combines manually selected frame metadata with existing pose summaries without changing either. Camera motion is recorded as `assumed_fixed_not_verified`; capture extrinsics are **not estimated**. Person movement is represented separately as normalized 2D pelvis-center displacement from the first chronologically selected frame with visible hips. It is not metric displacement, 3D translation, depth, or a full rigid person transform.

Per-frame output includes timestamp, original quality metadata, person-detection and visibility flags, boundary-margin check, duplicate-comparison status, displacement proxy, warnings, and the raw viewpoint heuristic and score. Blurry frames are excluded unless explicitly overridden, and their warning is retained. Extractor-rejected duplicate records are preserved separately; accepted frames with no comparison score are marked unknown.

View labels are unvalidated hints and their scores are **not calibrated probabilities**. Ambiguous side and front/back outputs remain unknown. Coverage is always reported as `unconfirmed`; it cannot establish 360° completion. The existing estimator does not reliably identify intermediate orientations or continuous yaw. No person orientation angle or 3D rotation is estimated.

### CLI and visualization

`scripts/fixed_camera_experiment.py` has `synthetic-demo`, `calibrate`, and `validate` modes. Each requires a new output directory; Matplotlib is needed for plotting. The existing `.venv-pose` provides Matplotlib without installing or upgrading packages.

Synthetic demonstration:

```powershell
.\.venv-pose\Scripts\python.exe scripts/fixed_camera_experiment.py synthetic-demo --output outputs/fixed_camera/new_synthetic_run
```

For real calibration, specify a new output directory, the number of inner corners, square size in a consistent unit, and explicit image paths:

```powershell
.\.venv-pose\Scripts\python.exe scripts/fixed_camera_experiment.py calibrate --output outputs/fixed_camera/new_calibration --columns 9 --rows 6 --square-size 0.025 --images path\board1.jpg path\board2.jpg path\board3.jpg path\board4.jpg path\board5.jpg path\board6.jpg
```

Use more than the minimum number of views with different positions, tilts and image coverage. Use images from the same camera, resolution and camera settings as the capture. Calibration output is not proof of camera stability, focus consistency or acceptable real-world accuracy.

For existing metadata and pose results, explicitly select frame filenames:

```powershell
.\.venv-pose\Scripts\python.exe scripts/fixed_camera_experiment.py validate --output outputs/fixed_camera/new_capture_review --metadata path\metadata.json --pose-summary path\processing_summary.json --frame frame_001.jpg --frame frame_009.jpg
```

Use `--allow-blurry` only for an intentional review override; blur flags remain. `--calibration path\calibration.json` checks resolution and reports provenance. `--frame-dir path\extracted_frames` optionally displays selected frame images in a contact sheet. Those images and output files may contain personal data and must be handled according to the applicable privacy policy.

Outputs include `capture_validation.json`, `capture_validation.png`, `frame_sequence.png` when frames are selected, and `calibration.png`. A supplied calibration is copied as a validated JSON record into the new run directory. Calibration mode also produces detected-corner overlays when detection succeeds. Without frame images, the sequence visualization uses labelled placeholders.

## Synthetic results

`outputs/fixed_camera/synthetic_002` was generated with ten noiseless synthetic checkerboard observations, 9×6 inner corners and 640×480 pixels. The known synthetic focal lengths were 800 and 810 px; estimated values were approximately 800.00044 and 810.00048 px. Overall synthetic RMS was approximately **0.000011 px**, with maximum estimated board-normal difference 26.13°. These are idealized synthetic observations and do not demonstrate real camera accuracy.

Four synthetic pose records were reviewed. The output retained one blurry-frame override, labelled two ambiguous side examples `unknown`, and reported coverage as `unconfirmed`. Synthetic calibration is explicitly marked `synthetic_demo_only; real calibration pending`.

## Tests

Executed in the existing Python environment with OpenCV 4.10.0 and NumPy 1.26.4:

```text
python -m unittest backend.tests.test_fixed_camera_validation backend.tests.test_camera_geometry backend.tests.test_multiview_geometry backend.tests.test_shared_shape_fitter backend.tests.test_spike_mesh scripts.test_sequential_sampling -v
Ran 25 tests in 4.279s: OK
```

The five new tests cover synthetic corner calibration, distorted-pixel undistortion, save/load and resolution checks, generated checkerboard-image detection, missing and degenerate inputs, blur override, ambiguous orientation, missing landmarks, boundary clipping, displacement warnings and duplicate metadata. Existing geometry, shared-shape, mesh and sequential-sampling tests also passed.

Production video capture, MediaPipe processing, HMR2 inference and frontend workflows were not rerun in this milestone. No production behavior was intentionally modified.

## Classification

### Implemented and tested

- Isolated checkerboard calibration and synthetic image/corner tests.
- Explicit distortion, image resolution, provenance and reprojection reporting, save/load and undistortion.
- Metadata-based fixed-camera assumptions, image-space movement proxy, visibility/boundary checks, blur override, duplicate evidence and unconfirmed viewpoint coverage.
- Synthetic demo and separate visualizations.
- Existing geometry and shared-shape regression suite.

### Implemented but not fully tested

- Checkerboard detection/calibration has not been validated against real calibration images, held-out real images, lens/focus changes or broad board-pose diversity.
- The orientation heuristic and boundary/movement thresholds have not been validated on a labelled population. An image-space pelvis shift does not distinguish body rotation, camera movement and actual translation.
- The optional real-frame contact sheet and checkerboard overlay have not been exercised on real imagery in this milestone.

### Blocked

- Real camera calibration and real-camera accuracy evaluation require suitable checkerboard images from the actual camera and capture settings.
- SMPL/HMR2 use remains blocked by missing authorization.

### Not implemented

- Estimation or verification of physical camera extrinsics or camera stability.
- Reliable continuous person yaw, validated front/back/left/right/intermediate classification, metric 3D person translation or body reconstruction.
- A production capture API or frontend integration.

## Files created

- `backend/services/checkerboard_calibration.py`
- `backend/services/fixed_camera_validation.py`
- `backend/tests/test_fixed_camera_validation.py`
- `scripts/fixed_camera_experiment.py`
- `docs/FIXED_CAMERA_CAPTURE_VALIDATION_REPORT.md`

## Recommended next steps

Capture diverse checkerboard images and independent held-out images at the actual recording resolution and camera settings; assess calibration error and stability before use. Validate orientation and quality thresholds against labelled recordings, including mirrored views and off-axis motion. Treat unknown orientation and inadequate coverage as unresolved rather than forcing a classification.
