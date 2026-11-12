# Real-camera calibration workflow: implementation report

## Status

The folder-based workflow is implemented and tested using generated checkerboard images. **No actual checkerboard photographs were processed for this milestone; real-camera calibration and independent validation remain pending.** The synthetic tests do not establish camera accuracy. No SMPL or HMR2 assets were accessed, and production capture, inference and frontend behavior were not changed.

## Implementation

`backend/tools/calibrate_camera.py` provides a CLI backed by `backend/services/real_camera_calibration.py`. The workflow reuses corner detection, `calibrate_observations`, camera configuration validation and serialization from `backend/services/checkerboard_calibration.py`. The latter now exposes a shared detection helper and opt-in projected-corner diagnostics; its existing calibration defaults remain unchanged.

The input folder is scanned non-recursively for `.jpg`, `.jpeg` and `.png` files, case-insensitively. Each image is recorded as detected or rejected with a reason, including unreadable images and missing checkerboards. A mixed-resolution set fails rather than combining incompatible observations. At least six successful detections are required by default. Existing diversity checks screen near-identical views and estimated board tilt.

A successful run creates `outputs/fixed_camera/real_<run_id>/` by default. Run IDs are generated randomly unless supplied; an existing directory is never overwritten. The run includes:

- `calibration.json`: intrinsics, distortion, resolution, board configuration, per-view and overall RMS, provenance, accepted/rejected images and screening thresholds.
- `image_results.json`: detection status, rejection reason and per-image reprojection RMS; detected images flagged as high-error remain in the fit.
- `calibration_report.md`: status, thresholds, image results, limitations and recommendations.
- `undistorted_samples/`: up to three paired before/after images by default.
- `reprojection_visualization.png`: contact sheet of detected corners (green) and fitted projections (red crosses).

A failed run caused by insufficient detections, mixed resolutions or calibration failure retains `image_results.json` and `calibration_report.md` in its run directory. An invalid or empty input folder fails before creating a run.

Per-image error is the root mean squared Euclidean pixel distance between detected and reprojected corners. The overall OpenCV RMS uses the calibration observations. Default screening labels are: `acceptable_screening` at overall RMS ≤0.5 px with no per-image RMS above 1.0 px; `review` above 0.5 px or with any per-image RMS above 1.0 px; and `poor` above 2.0 px overall. All thresholds are configurable and are **heuristics**, not proof of accuracy. Detected images are never automatically removed for high residuals. Existing degeneracy checks can reject the entire fit.

## CLI

From the project root:

```powershell
python -m backend.tools.calibrate_camera --input C:\path\to\checkerboards --columns 9 --rows 6 --square-size 0.025
```

`--columns` and `--rows` are the number of **inner corner intersections**, not the number of squares. The values shown are examples only: count the actual board. `--square-size` is one measured square's side length in a consistent physical unit; for example, `0.025` means 25 mm when using metres. The unit is not inferred or recorded separately, so document it with the calibration. Square size determines the board-coordinate scale; it does not measure the person.

Use `--help` for options, including `--min-views`, `--warning-rms-px`, `--poor-rms-px`, `--high-image-error-px`, `--samples`, `--run-id` and `--output-root`. A poor screening result is still written as a diagnostic calibration, not silently presented as validated. Check the report and independently validate before use.

## Requirements for the first real-world test

1. Use the actual fixed RGB camera and the same recording resolution, orientation, focus, zoom, stabilization and other settings intended for capture. Prevent camera/settings changes between calibration and recording.
2. Use a flat, rigid checkerboard with accurately known square size; count its inner corners in both directions and supply those values explicitly.
3. Capture at least six sharp, well-lit, unobstructed JPG/JPEG/PNG photographs at one consistent resolution. Prefer substantially more, with the complete board visible at varied positions, scales and tilts across the image, including near edges. Avoid repeated near-identical or all-frontoparallel views.
4. Reserve additional independent checkerboard photographs for a separate held-out assessment. This CLI reports training residuals only; held-out assessment is not implemented automatically.
5. Inspect the rejected-image list, corner overlay, residuals and before/after samples. Treat low RMS as insufficient evidence by itself; confirm stability and independent reprojection/undistortion behavior before using the configuration.

Images and output artifacts may contain sensitive information; store and dispose of them according to the applicable privacy policy.

## Tests and results

```text
python -m unittest backend.tests.test_real_camera_calibration backend.tests.test_fixed_camera_validation backend.tests.test_camera_geometry backend.tests.test_multiview_geometry backend.tests.test_shared_shape_fitter backend.tests.test_spike_mesh scripts.test_sequential_sampling -v
Ran 29 tests in 4.924s: OK
```

The new tests use generated perspective-warped checkerboards in temporary directories. They cover supported extensions, unreadable and non-checkerboard files, inconsistent resolution, insufficient detections, output structure, projected-corner error calculation, high-error retention, CLI help and missing-folder errors, and exclusive run creation. The existing 25 synthetic/regression tests also passed. No real-camera end-to-end validation, production video capture, MediaPipe or HMR2 inference regression was run in this milestone.

## Classification and limitations

- **Implemented and synthetically tested:** folder input, OpenCV checkerboard calibration, residual screening, diagnostic artifacts, failure reports and no-overwrite runs.
- **Implemented but not fully tested:** real-image behavior across lighting, blur, lens types, distortion, image sizes and checkerboard layouts; threshold suitability and camera stability. The visualization may be large for folders with many images.
- **Blocked/pending:** calibration and independent evaluation of the actual RGB camera, until suitable photographs are supplied.
- **Not implemented:** automatic held-out evaluation, distortion-model selection, physical camera-stability verification or 3D human reconstruction.

## Files

Created:
- `backend/services/real_camera_calibration.py`
- `backend/tools/calibrate_camera.py`
- `backend/tests/test_real_camera_calibration.py`
- `docs/REAL_CAMERA_CALIBRATION_IMPLEMENTATION_REPORT.md`

Modified:
- `backend/services/checkerboard_calibration.py` — shared corner detector and opt-in projected-corner diagnostics.
