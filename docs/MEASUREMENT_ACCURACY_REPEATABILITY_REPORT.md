# Measurement accuracy and repeatability report

Date: 18 Nov 2026 · Scope: body circumferences from the unified 360° reconstruction · Status: **repeatability measured; accuracy against tape still pending (no tape data yet)**

## 1. Summary

| Question | Answer from this study |
|---|---|
| Are the girths centimetres? | Only after a supplied barefoot height sets the scale (`POST /runs/{id}/scale`). A single uncalibrated camera can't recover absolute size. |
| How repeatable are they between two independent recordings of one person? | Mean CV 2.2% (range 1.3–3.4% per site); recording-to-recording difference 0.5–3.5 cm at 176 cm. |
| Is the pipeline itself deterministic? | Yes for practical purposes: re-running a saved fit reproduces the shape to ≤ 0.11 SD and the girths to about 0.1%. |
| What was the largest accuracy problem? | Loose clothing inflated girths. With no shape prior, the body grew to fill the garments: hip 129 cm and thigh 70 cm at 176 cm for a medium-build subject. The fitted body also stuck out 7–8% beyond its own clothed outline, which is physically impossible. |
| What changed? | A shape prior toward the multi-frame HMR2 shape is now on by default (`RECON_BETA_PRIOR=0.2`). Protrusion halves (7.5% → 3.6%), shape coefficients stay below 1.7 SD, and girths become plausible (hip about 100 cm, thigh about 50 cm). |
| Is it validated against tape? | **No.** The tape-entry workflow exists (`POST /runs/{id}/tape`, `scripts/tape_validation_report.py`), but no tape values have been recorded yet. Accuracy in cm can't be stated until they are. |

## 2. Measurement locations (anatomical definitions)

Each girth is a plane section of the final mesh, posed in a standard A-pose (arms 45° down, legs 5° apart). The plane is placed from SMPL's own joints, and "tape" is the convex-hull perimeter of the section enclosing the reference point. That hull is what a tape spans across concavities. Rings are exported in `measurement_rings.obj` and drawn on the avatar.

| Site | Model definition | Tape protocol (operator) |
|---|---|---|
| Chest | Maximum horizontal girth from the spine3 joint up to halfway to the shoulder joints (below the armpits) | Horizontal under the armpits at the fullest part of the chest, arms relaxed, end of a normal exhale |
| Waist | Minimum horizontal girth between the spine1 and spine3 joints | Horizontal at the narrowest part between the lowest rib and the hip bone |
| Abdomen | Horizontal girth at spine1 height (umbilical proxy) | Horizontal at navel level |
| Hip | Maximum horizontal girth from the pelvis joint down 40% toward the knees, enclosing both buttocks | Horizontal around the fullest part of the buttocks, feet together |
| Upper arm L/R | Perpendicular to the humerus at the shoulder–elbow midpoint | Perpendicular, halfway between shoulder tip and elbow, arm hanging relaxed |
| Forearm L/R | Maximum perpendicular girth 12–45% from elbow to wrist | Fullest part just below the elbow, hand relaxed |
| Thigh L/R | Perpendicular to the femur at the hip–knee joint midpoint | Halfway between groin crease and top of kneecap, weight on both feet |
| Calf L/R | Maximum perpendicular girth 15–50% from knee to ankle | Fullest part of the lower leg, weight on both feet |

A section is rejected rather than reported if it doesn't close, if more than one loop encloses the reference point, or if another body part crosses it (for example an arm touching the torso).

## 3. Data and method

- **Recordings:** two distinct source videos of the same person (content hashes `f8c4825b…` = A, `b5e73b7e…` = B). Medium build; loose T-shirt and loose cargo trousers; shoes on. The other nine video IDs on disk are byte-identical re-uploads of A or B, so they are **not** counted as repeat measurements.
- **Offline refits:** `scripts/measurement_accuracy_study.py` reloads each run's saved fit inputs (keypoints, HMR2 poses, cameras, subject masks) and repeats the shape fit under different settings, without re-running video decoding, HMR2 or segmentation. Runs used: `38895355` (A) and `4fe46ed4` (B).
- **Error measures available without tape:**
  - *Protrusion*: the share of the fitted body's silhouette that lies outside the observed clothed silhouette. A body can't stick out of its own clothes, so this is an objective error (mask noise sets a floor).
  - *Uncovered*: the share of the clothed outline the body doesn't fill. This is expected to be large under loose clothing, plus hair and shoes.
  - *Held-out IoU*: 4-fold cross-validated silhouette IoU on views not used for the shape. This is scored against the **clothed** outline, so it rewards a body that fills loose garments. It is reported, but it is not a girth-accuracy measure.
  - *Between-recording spread*: the CV and range of each girth across recordings A and B.

## 4. Results

### 4.1 Effect of the fit setting (both recordings)

| Setting | Protrusion | Uncovered | Held-out IoU | Max shape coeff. | Mean CV A↔B |
|---|---|---|---|---|---|
| Previous default (no shape prior) | 7.2% / 7.8% | 5.8% / 6.5% | 0.838 / 0.825 | 2.8 / 2.6 SD | 1.98% |
| Coverage term ×0.25, no prior | 6.4% / 5.6% | 9.5% / 11.6% | 0.831 / 0.814 | 2.9 / 2.3 SD | 0.87% |
| **Shape prior 0.2 (new default)** | **3.9% / 3.4%** | 14.0% / 14.5% | 0.796 / 0.782 | **1.7 / 1.6 SD** | 2.23% |
| Shape prior 1.0 | 3.9% / 3.4% | 14.2% / 14.8% | — | 1.7 / 1.7 SD | 2.12% |

Lowering only the coverage term made the result more repeatable but left the hip and thighs inflated (hip/height 0.71–0.73). Any shape prior of 0.05 or more removed most of the inflation, and the girths then barely changed between prior weights of 0.05 and 1.0.

### 4.2 Girths at 176 cm (recording A / B, mean, range)

| Site | Previous default | New default (prior 0.2) | A↔B range (new) | CV (new) |
|---|---|---|---|---|
| Chest | 117.5 / 119.2 | 100.2 / 102.3 (101.3) | 2.1 cm | 1.5% |
| Waist | 103.6 / 105.9 | 83.1 / 86.4 (84.8) | 3.4 cm | 2.8% |
| Abdomen | 109.2 / 111.1 | 85.7 / 89.2 (87.5) | 3.5 cm | 2.8% |
| Hip | 126.8 / 130.7 | 98.2 / 101.0 (99.6) | 2.9 cm | 2.0% |
| Upper arm L / R | 39.0 / 40.8 · 38.6 / 40.3 | 28.0 / 29.4 · 28.0 / 29.3 | 1.4 · 1.3 cm | 3.4 · 3.2% |
| Forearm L / R | 30.6 / 31.3 · 30.9 / 31.8 | 26.2 / 26.7 · 26.6 / 27.0 | 0.5 · 0.5 cm | 1.4 · 1.3% |
| Thigh L / R | 68.5 / 71.0 · 68.4 / 70.8 | 49.4 / 51.2 · 48.7 / 50.5 | 1.8 · 1.9 cm | 2.5 · 2.7% |
| Calf L / R | 45.9 / 46.8 · 45.6 / 46.7 | 36.3 / 37.1 · 36.3 / 37.1 | 0.8 · 0.8 cm | 1.6 · 1.6% |

The cm values use a single uniform scale, girth ÷ height × 176. An error of ±1 cm in the supplied height moves every value by ±0.57%.

### 4.3 Pipeline determinism and history

- Refitting run `38895355` from its saved inputs reproduced its shape coefficients to within 0.05 SD, and the girths to the third decimal of girth ÷ height.
- Runs `18e840e1` and `38895355` (same recording, same code) differ by ≤ 0.3% at every site.
- Run `eb02e080` (same recording as A, earlier code) gave hip ÷ height 0.581, against 0.720 later the same day. That 24% jump came from a fitting-method change, not from the recording.

## 5. Factors

| Factor | Evidence here | Status |
|---|---|---|
| Clothing | The dominant error. Loose clothing plus a symmetric silhouette fit inflated the hip by about 29 cm and the thigh by about 20 cm. The shape prior removes most of it, but loose garments can still bias girths. | Partly addressed. **Needs** a minimal or tight-clothing recording plus tape to quantify the remaining bias. |
| Posture | Both recordings use the same A-pose turn. The mesh is re-posed into a fixed measurement pose, so pose affects girths only through the fit. | **Not tested.** Needs recordings with arms lower or higher and slight slouching. |
| Camera distance | Both recordings are at a similar distance (subject about 80% of frame height). | **Not tested.** Needs recordings at, for example, 2 m / 3 m / 4 m. |
| Absolute scale | Supplied stature only. A checkerboard calibration tool exists (`backend/tools/calibrate_camera.py`), but it gives camera intrinsics, not the subject's distance, so it doesn't fix size by itself. | Height scale in place; a known-size reference in the scene would be an independent check. |

## 6. What is still needed to finish validation

1. **Tape measurements** of the subject at the sites in section 2, taken by one trained measurer, two readings each, entered per run (`POST /api/reconstruction/runs/{id}/tape`) together with the stadiometer height. Then run `scripts/tape_validation_report.py`, which gives bias, SD, MAE and maximum error per site, split by clothing.
2. **At least 3 recordings per condition** (minimal, tight and loose clothing; 2 m and 4 m; relaxed and A-pose), so repeatability (RC = 2.77 × within-subject SD) rests on more than two samples.
3. **More subjects** (≥ 5, varied body types) before quoting accuracy. The shape prior pulls toward HMR2's estimate, which tends toward average bodies, so very lean or very large subjects may be under-represented.
4. Re-run this study with the new recordings:
   `.venv\Scripts\python.exe scripts/measurement_accuracy_study.py RUN_A RUN_B ... --same-subject --height-cm <H> --cv --output outputs/accuracy_study/<name>.json`

## 7. Display

The completed-run panel now opens with a **Body Avatar** view (`frontend/src/components/BodyAvatarPanel.jsx`):

- The mesh is coloured by body segment (arms, legs, trunk, head). Segment boundaries come from SMPL skinning weights (`scripts/export_smpl_segments.py`).
- Callout cards for the left and right arms, trunk, and left and right legs show the primary girth, a ± uncertainty, and the secondary girths. Leader lines run to the point on each measurement ring nearest the viewer, and dots are dimmed when the limb is behind the trunk.
- Front, Side and Back presets, drag to turn, and mirror view. Cards follow the side on which their limb appears.
- **Compare** against an earlier scan: the mesh morphs between the two scans, with Replay and change chips. Re-runs of the same video are labelled, because they aren't new measurements.
- Colour key: in single-scan mode, measurement status (measured, tape-checked within ±2 cm, left/right differ by more than 5%, section not measurable). In compare mode, change versus the 95% repeatability coefficient (smaller, within repeatability, larger).
- An accuracy table shows each girth, its ±1 SD (height scale combined with repeatability), the 95% repeatability, and the tape error where one was recorded.

Colours describe measurement status or change, never body composition. Nothing in this pipeline measures fat or muscle.

## 8. Update: three recordings, outside term, validation tooling (18 Nov 2026, afternoon)

### 8.1 Data hygiene
- The tape records in runs `44c10580` and `2e0b075f` were **test entries** (confirmed by the user), not real tape measurements. They are now marked `excluded` and are ignored by every report and by the avatar. **No real tape data exists yet.**
- A third distinct recording was found: C, video `f304bffd`, run `2e0b075f`.

### 8.2 Repeatability across three recordings (identical settings, offline refits)
Settings: shape prior 0.2 and coverage weight 1.0, with the "outside" term in either form. Cm values are at 176 cm.

| Site | Mean (cm) | Range A/B/C (cm) | CV % |
|---|---|---|---|
| Chest | 101.1 | 2.1 | 1.0 |
| Waist | 84.5 | 3.4 | 2.0 |
| Abdomen | 87.2 | 3.5 | 2.0 |
| Hip | 99.8 | 2.8 | 1.5 |
| Upper arm L / R | 28.6 / 28.6 | 1.4 / 1.3 | 2.4 / 2.3 |
| Forearm L / R | 26.4 / 26.7 | 0.5 / 0.5 | 1.0 / 0.9 |
| Thigh L / R | 50.3 / 49.7 | 1.8 / 1.8 | 1.8 / 1.9 |
| Calf L / R | 36.7 / 36.7 | 0.8 / 0.8 | 1.1 / 1.1 |

Mean CV is 1.6%. The avatar's repeatability figures (`REPEATABILITY_SD_PCT`) now use these values.

### 8.3 "Stay inside the outline" term
The fit-code session (poc-mesh-81) replaced the saturating, diluted outside penalty with a non-saturating one (`RECON_SIL_OUTSIDE=charbonnier`). On all three recordings, with prior 0.2, protrusion falls from **3.4–4.1% to 1.3–1.4%**, girths change by at most 0.2 cm, and repeatability is unchanged (CV 1.58% vs 1.59%). That is an objective improvement.

### 8.4 Open question: how strongly to hold the shape to HMR2
poc-mesh-81 chose new defaults by held-out IoU: shape prior off, a population prior of 0.05, and per-view body pose fitted. That raises the hip to about 0.656 × height (about 115 cm at 176 cm), against 0.574 (about 101 cm) with prior 0.2. Held-out IoU is scored against the clothed outline, so it favours a body that fills loose clothing. In profile views the new setting fills 96% of the clothed depth; prior 0.2 fills 80%. Both are physically possible under a loose T-shirt. **None of the image-based metrics can decide which girths are correct; only real tape measurements can.** Each run records its exact fit settings (`multiview_optimisation.fit_config`), so tape values can be compared against each setting.

### 8.5 Validation tooling (steps: repeatability, clothing / distance / posture, tape)
- `POST /api/reconstruction/runs/{id}/conditions` (and a form under "Accuracy & method" in the avatar) tags a recording with a subject code, clothing, camera distance and posture.
- `GET /api/reconstruction/validation/report` and `scripts/validation_report.py` produce a full report:
  - one row per distinct recording, with re-uploads and re-runs collapsed by file content;
  - only runs made with the same fit settings;
  - repeatability computed only between recordings with the same conditions;
  - each condition's effect compared with the repeatability coefficient;
  - tape accuracy (bias, SD, MAE, worst error) excluding test entries.
- `scripts/validation_report.py --exclude-tape RUN …` marks test tape records; `--tag RUN --subject S01 …` tags old runs.

**Agreed between the two working sessions (18 Nov 2026):** the corrected outside term is an objective fix (2–4× less protrusion). The shape-prior setting is **not** settled and neither girth set is claimed as more accurate. When real tape values exist, compare `RECON_SIL_OUTSIDE=charbonnier`, `RECON_POP_PRIOR=0.05`, with `RECON_BETA_PRIOR` = 0 and 0.2 on the same recordings, and keep whichever is closer to tape (`scripts/validation_report.py`; runs are told apart by their recorded `fit_config`).
