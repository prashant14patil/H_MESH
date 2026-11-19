# Phase 4: End-to-end 360° body scan

Date: 19 Nov 2026 · Status: **one command turns a 360° recording into one unified mesh, measurements and quality reports. It has been run end to end on a real recording. Measurement accuracy is still NOT validated against tape.**

## 1. Audit: what existed, what was missing

Most of the requested pipeline already existed in `backend/services/body_reconstruction.py` (see `RECONSTRUCTION_ACCURACY_INVESTIGATION.md`). Phase 4 reuses all of it and adds only what was missing.

| Requirement | Before Phase 4 | Phase 4 |
|---|---|---|
| Person detection, every frame | Keypoint R-CNN (`person_tracking.py`) | reused |
| Consistent identity | tracker with identity-switch and loss detection; operator picks the subject when several people are present | reused |
| Orientation / rotation angle | per-frame heading from HMR2, cross-checked with 2D shoulder order | **added:** rotation trajectory (direction, speed, s per turn, reversals); front/back flips detected against temporal neighbours |
| Duplicate / redundant frames | previous-frame difference; one frame per 15° heading bin | reused |
| Blurry / unsuitable frames | relative Laplacian sharpness; border, occlusion and missing-joint checks | reused |
| Missing viewing angles | 4-label counts plus a "total rotation < 270°" warning | **added:** 24-bin coverage, largest circular gap, named missing arcs |
| Informative frame selection | sharpness + keypoint confidence + verified heading per bin | reused; **plus** frames rejected when both cues say the heading is wrong |
| Multi-view fusion | ONE shared SMPL β, per-view pose and camera, keypoints + silhouettes, 4-fold held-out validation | reused (not averaging of per-frame meshes) |
| Mesh validation | OBJ re-read; topology, watertight, upright, facing | reused |
| Measurements | plane sections at SMPL joints, 12 sites | reused; height is shown as *supplied*, never as measured |
| Real-world units | only from operator-supplied barefoot height | reused (no invented calibration) |
| HMR2 vs SHAPY comparison | API + UI | reused; also available from the CLI (`--shapy`) |
| Capture-quality assessment | separate MediaPipe job (`capture_quality_job.py`), not connected to reconstruction | **added inside the pipeline:** 7 checks, each with an actionable instruction |
| Reconstruction-quality verdict | metrics scattered through the report | **added:** `reconstruction_quality_report.json` with a verdict, gated by view coverage |
| One-command execution | `run_reconstruction.py` needed a recording already uploaded through the web API | **added:** `scripts/scan_360.py VIDEO --consent-attested [--height-cm H] [--shapy]` |
| Results visualisation | React panel only | **added:** `scan_report.html` per run; capture verdict and feedback in the React panel |
| Resource usage | stage timings, peak GPU allocation | **added:** reserved GPU memory, process peak RSS, capture-stats timing |

## 2. Architecture

```
scripts/scan_360.py  (or POST /api/reconstruction via the UI)
 ├─ ingest        validate_video → save → extract preview frames → consent attestation   (video_processor, capture_authorization)
 └─ body_reconstruction._pipeline
     1 decode       oriented frames, rotation applied exactly once (verified)
     2 detect       Keypoint R-CNN                                                            (person_tracking)
     3 confirm      operator selects the subject if several people are present
     4 track        single-subject tracking, identity switches rejected
       capture stats  subject lighting, background motion, hip/ankle anchors               (scan_quality)  NEW
     5 hmr2         real HMR2 per usable frame → β, pose, orientation, camera
     6 views        heading + 2D shoulder check + rotation trajectory                         (scan_quality)  NEW
     7 select       one sharp frame per 15° bin; flips rejected → capture_quality_report.json (scan_quality)  NEW
     8 masks        Mask R-CNN + GrabCut + MediaPipe vote                                     (subject_masks)
     9 fit          shared-shape multi-view SMPL fit; 4-fold held-out validation              (smpl_multiview)
    10 export       A-pose mesh OBJ/PLY → validation → circumferences → reports
                    → reconstruction_quality_report.json                                      (scan_quality)  NEW
 ├─ scale         cm only from --height-cm                                                    (body_measurements)
 ├─ shapy         optional comparison on the same selected frames                             (shapy_shape)
 └─ report        scan_report.html                                                            (scan_report)   NEW
```

New modules: `backend/services/scan_quality.py` (pure NumPy/OpenCV, unit-tested), `backend/services/scan_report.py`, `scripts/scan_360.py`. Tests: `backend/tests/test_scan_quality.py` (10 tests).

### 2.1 Heading flips (a change in frame selection)
HMR2 sometimes returns the body facing backwards. A frame is **rejected** only when two independent cues agree:
- its heading is more than 60° from the circular mean of its 3+3 temporal neighbours, **and**
- the 2D shoulder order contradicts it.

Either cue alone is reported but changes nothing. Total rotation is measured on the cleaned trajectory, so a flip can neither fake nor hide a full turn (see the unit tests).

### 2.2 Capture checks (`capture_quality_report.json`)

| Check | Measured from | Warn / fail |
|---|---|---|
| angular_coverage | cleaned headings of usable frames | largest gap > 45° / > 90°; rotation < 330° / < 270° |
| rotation_steadiness | trajectory | turn < 8 s or > 90 s, reversals, > 15% flips (warn only) |
| resolution | median subject box height | < 600 px / < 350 px |
| lighting | luminance of the subject box | mean outside 55–205, clipped > 5% / 15%, contrast, variation across the turn |
| body_visibility | usable ÷ tracked; other people | < 60% / < 35% |
| camera_consistency | background phase correlation (subject masked out); subject-height CV; ankle-line and hip drift relative to subject height | background shift p90 > 1% / 3% of width; … |
| useful_frames | selected views, reliable silhouettes, blur rejects | < 12 / < 6 views; < 8 / < 4 silhouettes |

The verdict is `good`, `acceptable_with_warnings` or `insufficient`. These are engineering defaults, not validated cut-offs. An insufficient capture still produces a mesh, but the run warns and the reconstruction verdict is downgraded.

Drift deliberately uses **hip and ankle midpoints**, not box edges. On the first real run, box edges reported 17% sideways drift that did not exist: the arms move the box centre by up to about 180 px, and the lowest stepping foot moves its bottom edge.

## 3. Results on a real recording

Recording: original of `777541cd` (1080×1920, 60 fps, 25.8 s, one subject, about 1¼ turns), re-ingested as `7206df13`.

```
.venv\Scripts\python.exe scripts/scan_360.py --video-id 7206df13-… --height-cm 179 --shapy --baseline-run 9e126c35-…
```
Run `88ae7ffc`: outputs in `outputs/reconstructions/88ae7ffc-2c8e-47e4-828b-aadcc54b0ba8/`. The mesh is `final_body_mesh.obj` (relative units) and `final_body_mesh_metric.obj` (metres, scaled from height). The page is `scan_report.html`.

### 3.1 Capture-quality report: **good**

| Check | Status | Key values |
|---|---|---|
| angular_coverage | pass | 360° rotation, largest gap 16°, 24/24 bins; selected views F5 / R8 / B4 / L7 |
| rotation_steadiness | pass | 26 s per turn, clockwise, 0 reversals, 0 heading outliers |
| resolution | pass | subject 1120 px (58% of frame height) |
| lighting | pass | luma 151, contrast 70, 1.9% clipped, 5% variation |
| body_visibility | pass | 129/129 tracked frames usable, no other people |
| camera_consistency | pass | background shift p90 0.12% of width (camera fixed); height CV 2.4%; ankle drift 4.2%, hip drift 8.2% of body height |
| useful_frames | pass | 24 views, 24 reliable silhouettes, 0 blurry |

### 3.2 Reconstruction-quality report: **consistent** (internal consistency only)

| Check | Value |
|---|---|
| mesh geometry | valid, watertight, upright, facing +z, 1 component, 6890 vertices / 13776 faces |
| keypoint reprojection | 0.68% of body height (median) |
| held-out silhouette IoU | 0.815 (keypoint-only 0.792, HMR2 aggregate 0.779, per-frame HMR2 0.781) |
| protrusion outside clothed outline | 1.65% |
| proportions (model/observed torso) | all ≤ 1.05 |
| shape plausibility | max \|β\| 0.9 SD |
| measurement sections | 12/12 valid |
| view coverage | pass |

### 3.3 Measurements (cm come only from the supplied 179 cm height; **not tape-validated**)

| Site | Pipeline cm | ×height | SHAPY cm | Keypoint-only ×height |
|---|---|---|---|---|
| height | 179.0 (supplied) | 1 | 179.0 (supplied) | 1 |
| chest | 114.3 | 0.638 | 106.1 | 0.580 |
| waist | 100.3 | 0.560 | 88.5 | 0.488 |
| abdomen | 104.0 | 0.581 | 90.8 | 0.511 |
| hip | 117.1 | 0.654 | 105.0 | 0.601 |
| upper arm L / R | 35.8 / 35.5 | 0.200 / 0.198 | 32.8 / 32.7 | 0.170 / 0.170 |
| forearm L / R | 29.2 / 29.5 | 0.163 / 0.165 | 28.4 / 28.3 | 0.151 / 0.153 |
| thigh L / R | 60.5 / 60.2 | 0.338 / 0.336 | 54.9 / 54.7 | 0.303 / 0.301 |
| calf L / R | 41.6 / 41.5 | 0.233 / 0.232 | 38.6 / 38.5 | 0.216 / 0.215 |

The pipeline and SHAPY disagree by 8–12 cm on torso girths. The pipeline follows the **clothed** silhouette, while SHAPY regresses a shape from appearance. Which one is closer to the body cannot be decided without tape (§6).

### 3.4 Processing time and resources (RTX 4070 Laptop 8 GB, torch 2.5.1+cu121)

| Stage | s |
|---|---|
| decode (129 frames) | 7.2 |
| detect | 14.9 |
| capture stats (new) | 2.3 |
| HMR2 | 26.1 |
| masks | 13.7 |
| fit (incl. 4-fold CV + keypoint-only comparison) | 179.9 |
| export | ~16 |
| **pipeline total** | **260** |
| SHAPY | 12.6 |
| CLI wall time (model load, SHAPY, report) | 314 |

Peak GPU memory: 3569 MiB allocated, 6042 MiB reserved. Process peak RSS is recorded from this release on; this run predates that fix and shows `None`. The 4-fold cross-validation takes about 60% of the fit time. It is validation, not reconstruction, and could be made optional for production speed. It was kept, per "do not sacrifice accuracy for speed".

## 4. Before / after

**Same recording, same fit code: baseline run `9e126c35` (before Phase 4) vs `88ae7ffc` (after).**

| | Before | After |
|---|---|---|
| Frames selected | 24 | 24 (identical frame set) |
| Held-out IoU (fitted shape) | 0.8155 | 0.8155 |
| Fit IoU / keypoint error | 0.8848 / 0.673% | 0.8847 / 0.676% |
| Girths | — | within 0.1 cm of before at every site (GPU non-determinism) |
| Capture-quality report | none | good, 7/7 checks |
| Reconstruction verdict | none | consistent |
| Command | upload via UI/API, then `run_reconstruction.py <id>`, then height via API, then SHAPY via API | `scan_360.py VIDEO --height-cm 179 --shapy` |
| Pipeline time | 245 s | 260 s (+2.3 s capture stats; the rest is run-to-run variation) |

This is the expected result for a clean capture: Phase 4 neither regresses nor changes the mesh.

**Degraded capture: first 45% of the same video (half turn, `96dc03ec`).**

| | Before Phase 4 (equivalent) | After |
|---|---|---|
| Feedback | two warnings: "too few right_side frames", "heading change 177° < 270°" | capture **insufficient**: "Only 177° of rotation was measured: turn a full 360° …"; "No usable view within a 183° arc around the right side …" |
| Reconstruction verdict | would have looked fine (fit metrics all good: held-out IoU 0.822, protrusion 1.6%) | **unreliable** (view_coverage fails) |
| Girths | — | within 0.3 cm of the full turn |

The last row matters. Missing half the body did not change the girths, because SMPL's 10 shape coefficients are nearly symmetric, so the unseen side is *inferred* from the seen side. Every internal-consistency metric stays green. That is why view coverage now gates the reconstruction verdict: fit quality alone cannot detect an unobserved side.

## 5. Known limitations

1. **No independent accuracy validation.** No real tape or 3D-scan data exists for these recordings (the `tape_validation.json` entries in earlier runs are TEST data). Every cm value is unvalidated.
2. **Scale comes from supplied height only.** There is no camera calibration or reference object, and the CLI never invents one. A 1 cm height error moves every girth by about 0.6%.
3. **Clothed silhouette.** Loose clothing inflates girths (hip and thigh especially); silhouettes cannot see under clothing.
4. **Shape space.** Ten SMPL shape coefficients, no per-vertex refinement. Asymmetry and local detail (for example abdomen shape) are only approximated, and an unseen side is inferred.
5. **Camera intrinsics** are HMR2's assumed focal length; `checkerboard_calibration.py` exists but is not used in the reconstruction.
6. **Capture thresholds** are engineering defaults checked on one setup (one subject, one room). They have not been tuned on a dataset of good and bad captures.
7. **The heading-flip rejection** has not fired on a real recording yet (the test recording has none). It is covered by unit tests only.
8. **Concurrency.** A reconstruction started from the CLI is owned by that process. If the web server reads its `status.json` while it is still running, the server marks it "interrupted" (pre-existing behaviour, also true of `run_reconstruction.py`). Run CLI scans while the server is idle, or start them through the UI.
9. **Speed.** About 4–5 min per scan on a laptop GPU, mostly cross-validation.

## 6. Recommended next phase

1. **Tape validation study (blocking for any accuracy claim).** About 10 subjects in minimal or tight clothing, 2 recordings each, ISO 8559-style tape by a trained measurer (templates exist in `docs/validation/`). Decide between the pipeline, SHAPY and the HMR2-prior settings by error against tape, not by IoU.
2. **Calibrated scale.** Use a reference of known size (an A3 checkerboard on the floor, or a floor mat) to get scale without relying on stated height, and use calibrated intrinsics in the fit (`checkerboard_calibration.py`).
3. **Live capture guidance.** Run `rotation_trajectory` and the coverage and lighting checks on the phone or browser preview while recording, so the operator is told to slow down or keep turning before stopping.
4. **Shape detail beyond 10 coefficients.** SMPL with 300 coefficients, or per-vertex displacement constrained by the visual hull (`scripts/visual_hull_girths.py` is the upper-bound check). Only worth doing once tape data shows where the error is.
5. **Production speed.** Make 4-fold CV optional (validation mode vs scan mode); a 2-fold option would cut the fit time by about half.
