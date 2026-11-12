# Shared-shape fitting feasibility spike

## Status

**The requested real SMPL/HMR2 spike is blocked and its success criterion is not met.** The project owner stated that the locally installed SMPL model and HMR2 checkpoint are **not authorized** for this experimental fitting and real-video evaluation, and instructed synthetic testing only. No licensed model assets or checkpoint were loaded for this spike, downloaded or redistributed.

## Existing architecture inspected

- `inference_engine.py` runs the existing HMR2 model and exports `pred_vertices` with SMPL faces to OBJ. Its public output does not persist shape, pose or camera parameters.
- The vendored HMR2 forward pass exposes `pred_smpl_params` (`betas`, `body_pose`, `global_orient`), `pred_cam`, `pred_cam_t`, focal length and predicted joints. These fields were inspected in source only, not exercised by this spike.
- `ViTDetDataset` returns box center, box size and image dimensions. Its crop transform and HMR2's crop-to-full camera convention need explicit verification before comparing projections to full-image landmarks.
- HMR2's SMPL wrapper remaps its joints to an OpenPose-style order. MediaPipe outputs 13 named normalized landmarks and visibility values. The correspondence must be explicitly mapped and validated.
- Existing frame metadata contains timestamps and blur flags; pose outputs contain per-frame landmarks, review flags and segmentation masks. The prior validated recording marks all 30 frames blurry under its configured threshold; viewpoint side labels are ambiguous.
- The existing viewer loads OBJ and applies a fixed orientation transform. Coordinate compatibility for a new canonical mesh is not established.
- The existing environment uses Python 3.12.10, NumPy 1.26.4, Torch 2.5.1+cu121 and smplx 0.1.28. Prior reports document dependency conflicts and a test-only overlay. No dependencies or NumPy versions were changed.

## Implemented architecture

`backend/services/shared_shape_fitter.py` provides model-agnostic experimental primitives. It accepts a caller-supplied differentiable body-model callable, pixel-space observations and explicit per-view parameters; it does not load or initialize HMR2 or SMPL.

One shared beta vector is optimized with Adam against at least two views with at least four valid landmarks each. Body pose and global orientation are fixed inputs. Each view has an independently optimized, regularized camera translation. The loss uses confidence-weighted robust landmark reprojection, an L2 zero-centered shape prior and a camera-deviation prior. It rejects non-finite or nonpositive camera-space depth and requires an improvement in both regularized objective and landmark data loss plus a nontrivial beta change. This is not a convergence guarantee and does not prove that shape, rather than camera movement, caused the data improvement.

`fit_heldout_camera` freezes shape and pose and fits held-out camera translation. `reprojection_metrics` reports per-view pixel and normalized mean errors. Held-out camera parameters are estimated using held-out landmarks, so this is conditional evaluation, not fully independent prediction.

`backend/services/spike_mesh.py` validates numeric finite vertices, triangle indices and degenerate faces before exclusive-create OBJ export. Dimensions are explicitly uncalibrated model units. It has only been tested on synthetic geometry.

## Camera assumptions and limitations

The generic fitter uses perspective projection with supplied focal length and image center in pixels. Per-view camera translations are independent approximations; this is **not** a calibrated fixed-camera/rotating-person solution. It does not currently enforce a consistent physical camera across frames, optimize pose, implement silhouette loss, select viewpoints or validate camera/crop conversion. Monocular scale is ambiguous; no metric measurements are claimed.

## Synthetic test results

Executed:

```text
python -m unittest backend.tests.test_shared_shape_fitter backend.tests.test_spike_mesh -v
Ran 9 tests in 5.887s: OK
```

Tests covered three synthetic training views contributing nonzero gradients to the same beta, a held-out view, missing/zero-weight observations, insufficient observations, invalid initialization, no-op optimization, invalid camera depth, two synthetic view selections, and synthetic OBJ validation/export.

In one noise-free toy experiment, training angles were 0.0, 0.8 and 1.5 radians and held-out angle was 2.3 radians. The known synthetic target beta was `[0.3, -0.15]`; the fitted beta was approximately `[0.299709, -0.149983]`. The training landmark loss changed from `0.000161259` to `4.48e-11`. Held-out mean reprojection error changed from **13.7182 px** for the zero-beta baseline to **0.0070 px** for the fitted beta. Held-out camera translation was separately optimized for both. These values describe only the toy model with noiseless observations and known rotations and do not demonstrate SMPL or human accuracy.

## Mesh and measurements

No SMPL mesh, selected real-frame list, HMR2 initialization, visual body comparisons or real held-out metrics were produced. The OBJ tests use a three-vertex synthetic triangle; they test serialization and invalid-geometry rejection, not human mesh quality or watertightness. No independent measurements were supplied or evaluated. Bounding dimensions from the helper are model-space extents, not body measurements.

## Classification

### Implemented and synthetic-tested
- Model-agnostic shared-beta optimization and conditional held-out landmark evaluation.
- Missing/invalid observation and optimization failure checks, and synthetic repeatability.
- Synthetic mesh validation and no-overwrite OBJ export.

### Implemented but not fully tested
- Fitting primitives have not been tested with actual SMPL, HMR2 outputs, real-camera data, GPU fitting or real video. They are not integrated into an application route or visualization workflow.
- A changed beta and lower landmark data loss are necessary checks, not evidence of a converged or anatomically accurate fit.

### Blocked
- Any real HMR2 initialization, use of the licensed SMPL assets, real-video fitting, real unified OBJ and real held-out evaluation, pending verified authorization.
- Metric measurement validation, pending independent measurements and scale calibration.

### Not implemented
- HMR2 parameter adapter, frame-selection pipeline, camera/crop round-trip validation, SMPL joint mapping, actual SMPL body-model integration, silhouette fitting, per-view pose optimization, production route, CLI for real fitting, real-data visual comparisons and frontend integration.

## Changed files

- `backend/services/shared_shape_fitter.py`
- `backend/services/spike_mesh.py`
- `backend/tests/test_shared_shape_fitter.py`
- `backend/tests/test_spike_mesh.py`
- `docs/SHARED_SHAPE_SMPL_SPIKE_REPORT.md`

No existing inference, capture, pose, viewpoint or frontend implementation was modified for this spike.

## Recommended next step

Obtain and document the necessary asset and checkpoint authorization before using them. Once authorized, first implement and test an isolated HMR2 adapter, verify joint and crop-to-image projection conventions and differentiability, then fit a small diverse training set and evaluate a frozen shared shape on excluded real views. Preserve the single-photo workflow and report actual convergence, ablations, held-out errors and missing measurement references without claiming anatomical accuracy.
