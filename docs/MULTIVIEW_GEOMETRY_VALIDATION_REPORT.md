# Multi-view geometry validation report

## Scope

This is a synthetic geometry feasibility experiment, **not real human reconstruction or a body-measurement validation**. No SMPL or HMR2 model asset or checkpoint was loaded, executed, copied, downloaded or accessed for this work. No production route, capture, pose, inference, mesh-export or frontend behavior was changed.

## Existing architecture inspected

The existing shared-shape fitter accepts pixel observations, confidence weights, camera-oriented body rotations and per-view camera translations. Its projection is perspective and differentiable in PyTorch; it does not accept an arbitrary world-to-camera rotation directly. The existing `spike_mesh.py` validates and exports OBJ geometry; this experiment reconstructs landmarks rather than a mesh and does not duplicate that functionality.

Frame extraction metadata already records timestamps, `is_blurry`, `blur_score` and duplicate scores. Pose summaries contain named normalized landmarks and visibility. These metadata do not provide calibrated camera poses. The prior spike report documents approximate camera assumptions and the unresolved SMPL/HMR2 authorization blocker.

The current Python environment has NumPy 1.26.4 and Torch installed. Matplotlib is not installed there but is available in the existing isolated `.venv-pose`; that environment was used for plots. No packages were installed or upgraded.

## Implemented modules

- `backend/services/camera_geometry.py`: explicit pinhole intrinsics, extrinsics and provenance (`synthetic`, `approximate`, or `calibrated`); world/camera transforms, projection, world-space rays and pixel reprojection error. Missing, non-finite, invalid rotation and behind-camera inputs are rejected.
- `backend/services/multiview_geometry.py`: multi-camera projection; confidence-weighted ray-intersection triangulation; missing/low-confidence handling; degeneracy checks; optional pair-hypothesis reprojection outlier rejection; and explicit metadata-only frame selection.
- `scripts/run_synthetic_multiview.py`: reproducible synthetic experiment, quantitative JSON and visualization.
- `backend/tests/test_camera_geometry.py` and `backend/tests/test_multiview_geometry.py`: synthetic regression tests.

No adapter to `shared_shape_fitter.py` was added. A safe future adapter must transform the model's body orientation into each camera's coordinate system, preserve crop/image conventions, and establish compatible joint mappings and differentiability. Simply passing world-space landmarks into the existing fitter would be incorrect.

## Camera model and triangulation

World-to-camera convention is `Xc = R Xw + t`; `R` is a proper world-to-camera rotation, camera +X points right, +Y down and +Z forward. Pixel origin is top-left. Intrinsics are `(fx, fy, cx, cy)` in pixels with explicit image dimensions. There is no lens distortion. World units are arbitrary synthetic units.

For each landmark, the triangulator minimizes confidence-weighted squared perpendicular distance to the selected camera rays. It requires at least two sufficiently non-parallel views, checks conditioning and positive depth, and returns an unavailable point with a reason when geometry is insufficient. The optional outlier mechanism tests camera-pair hypotheses and chooses a consensus by reprojection threshold. Two-view cases cannot identify an outlier reliably; small consensus sets and correlated errors can still produce incorrect results.

## Experiment configuration

Run:

```powershell
.\.venv-pose\Scripts\python.exe scripts/run_synthetic_multiview.py --output outputs/synthetic_geometry/geometry_poc_003 --seed 42 --noise 1.0 --heldout 3
```

The output directory must not already exist. It contains `metrics.json` and `noisy_views.png`. The image shows four virtual camera positions, known and reconstructed 3D landmarks, four 2D views and per-view reprojection errors. Camera labels front/left/back/right refer to the synthetic world, not anatomical or real camera labels.

The scene contains 14 fixed, articulated human-like landmarks. Four virtual cameras are at `[0,0,-4]`, `[-4,0,0]`, `[0,0,4]`, and `[4,0,0]`, looking at the origin. True intrinsics are `fx=fy=800` pixels, principal point `(320,240)`, image size 640×480. Cameras 0–2 reconstruct points; camera 3 is held out entirely from triangulation. The noisy case adds independent Gaussian image noise with standard deviation 1 pixel. The missing/low-confidence case removes three observations and assigns confidence 0.02 to three others; each point retains at least two usable training observations.

The approximate-calibration case uses focal length 840 instead of 800 and shifts each assumed camera center by `[0.08, 0, 0.04]` world units, with orientations recomputed toward the origin. Its held-out error is computed using the **approximate** held-out camera and noisy observations, not the true camera.

## Quantitative results

One deterministic seed, 14 reconstructed landmarks per scenario:

| Scenario | Mean 3D error (world units) | Max 3D error | Training mean reprojection (px) | Held-out mean reprojection (px) |
|---|---:|---:|---:|---:|
| Exact calibration, no noise | 6.31e-16 | 1.89e-15 | 7.44e-14 | 1.05e-13 |
| 1 px noise, three fit views | 0.004801 | 0.011093 | 0.6488 | 1.1117 |
| Noise + missing/low-confidence | 0.006799 | 0.014436 | 0.5271 | 1.4295 |
| Noise + approximate calibration | 0.028664 | 0.048704 | 1.2178 | 2.8671 |
| Noise, all four views used | 0.004408 | 0.007649 | 0.7216 | N/A — no held-out view |

The fourth view is excluded from the first four reconstructions. The all-four-view run is a separate comparison and its training error is not a held-out metric. Missing and low-confidence training observations are excluded from the corresponding error average. These values describe a single synthetic seed and camera arrangement; they are not confidence intervals or real-world accuracy estimates.

## Frame-quality handling

`select_frames` requires explicit filenames and preserves original metadata. Blurry selected frames are excluded by default; `allow_blurry=True` accepts them only with their original `is_blurry` field and an additional `quality_warning`. The result reports accepted, excluded, manually selected and blurry-accepted counts.

A metadata-only CLI check selected two previously extracted blurry frames with `--allow-blurry`: 2 accepted, 28 excluded, 2 manually selected and 2 blurry accepted. Both blur warnings remained. These real frame records were **not** used as synthetic reconstruction inputs. The blur threshold was not changed.

## Regression results

```text
python -m unittest backend.tests.test_camera_geometry backend.tests.test_multiview_geometry backend.tests.test_shared_shape_fitter backend.tests.test_spike_mesh scripts.test_sequential_sampling -v
Ran 20 tests in 4.267s: OK
```

This includes eight new camera/multiview/selection tests, the existing nine shared-shape/mesh tests and three sequential-sampling tests. Tests cover transforms, projection/rays, exact triangulation, a held-out view, noise, missing and low-confidence observations, an obvious outlier, degenerate cameras and invalid calibration. Production video capture, MediaPipe, HMR2 inference and frontend workflows were **not rerun** in this milestone; their behavior is unchanged but not newly verified. Frontend files were not touched.

## Classification and limitations

### Implemented and tested
- Explicit synthetic/approximate/calibrated camera provenance, projection, inverse rays and coordinate transforms.
- Synthetic multi-view triangulation, confidence filtering, missing-data handling, optional outlier rejection and held-out reprojection.
- Reproducible experiment, separate visualization and frame-quality selection with explicit blurry-frame override.
- Existing synthetic fitter, mesh and sequential-sampling regression tests.

### Implemented but not fully tested
- Geometry is validated on one ideal pinhole scene and a small deterministic noise test. Lens distortion, rolling shutter, occlusion, calibration uncertainty distributions, numerical stability over a broad baseline range and large datasets are not validated.
- Pair-consensus outlier rejection is not a guarantee, especially with two observations or multiple outliers.
- The frame selector operates on extraction metadata only; it does not assess pose visibility or prove view coverage.

### Blocked
- Real SMPL/HMR2 fitting remains blocked by missing authorization. Real camera poses, calibration and metric scale are unavailable for the existing recording.

### Not implemented
- Camera calibration estimation from real images, real-video triangulation, a shared-shape fitter adapter, real 3D body mesh reconstruction and body measurement validation.

## Files created

- `backend/services/camera_geometry.py`
- `backend/services/multiview_geometry.py`
- `backend/tests/test_camera_geometry.py`
- `backend/tests/test_multiview_geometry.py`
- `scripts/run_synthetic_multiview.py`
- `docs/MULTIVIEW_GEOMETRY_VALIDATION_REPORT.md`

Synthetic run artifacts are under ignored `outputs/synthetic_geometry/`. No dependency manifest or existing application file was modified.

## Recommended next step

Expand synthetic tests to noisy calibration, low-baseline and multi-outlier cases; then design a separate camera-to-fitter adapter with explicit coordinate and joint contracts. Real capture integration requires independently validated calibration/motion handling and any necessary model authorization.
