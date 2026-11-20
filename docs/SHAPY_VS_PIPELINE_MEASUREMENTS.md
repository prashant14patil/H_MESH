# SHAPY vs the pipeline: what each measures and how to decide

2026-11-20, branch `feature/shapy-as-default-measurements`. Every scan can give girths from two models. Neither model
has been checked against a tape measure. This note collects the evidence so far and the 10-minute check that settles it.

| | Pipeline (default fit) | SHAPY |
|---|---|---|
| What it is | HMR2 start, then one SMPL shape fitted to the person's outline in every view | Single-image network ([Choutas et al., CVPR 2022](https://shapy.is.tue.mpg.de/)), trained on model-agency photos paired with height/chest/waist/hip measurements plus crowd-sourced shape words; median over the run's frames and their mirror images |
| Strength | Specific to this person: uses the full 360° outline | Can partly see through clothing; repeated recordings agree within 0.2–0.5 cm |
| Weakness | Measures the clothed outline, so loose clothing reads as body | Never sees this subject's outline; pulls toward an average body, so it may also under-report real change over time |

## Evidence so far: one recording (video `777541cd`), 179 cm, no tape

| cm | chest | waist | abdomen | hip |
|---|---|---|---|---|
| Pipeline, current fit | 114.3 | 100.3 | 104.0 | 117.1 |
| SHAPY | 106.1 | 88.5 | 90.8 | 105.0 |
| Clothed visual hull, an upper bound with no body model ([investigation §5.1](RECONSTRUCTION_ACCURACY_INVESTIGATION.md)) | n/a (arms) | 114.3 | 117.0 | 122.7 |
| Estimated real tape, reasoned and not measured (§5.4) | 106–114 | 95–103 | 98–106 | 110–116 |

- **The pipeline is inside or at the edge of every estimated range.** Its hip is at the top end because it carries
  ease from the loose cargo trousers.
- **SHAPY is below the range at waist, abdomen and hip by 5–7 cm,** and at the bottom edge for chest. Per §5.3, its
  88.5 cm waist would need a near-circular ~28 cm body section inside a 38 × 33 cm clothed section.
- **Two learned estimates agree with each other, not with the outline.** The pipeline with its shape pinned to HMR2's
  own estimate (prior 0.2, §5) gave 104.0 / 87.9 / 90.8 / 102.7, close to SHAPY. The investigation judged that shape
  a generic, shallow torso.
- **An earlier "8 runs, pipeline ~6 cm larger" summary is not independent evidence.** It counted six refits of this
  same video, which agree to 0.1 cm, plus runs from before the 18 Nov silhouette fix.

On current evidence the pipeline is the better estimate for this subject, with the hip probably 1–7 cm high. That is
reasoning, not a measurement.

## Where each model is used

- **`scripts/scan_360.py`** runs SHAPY by default after the reconstruction (`--no-shapy` skips it; the old `--shapy`
  flag is still accepted). It prints both models side by side. A SHAPY failure never fails the scan.
- **`scripts/shapy_compare.py RUN --height-cm H --tape chest=.. waist=.. hip=..`** scores both models against tape
  with the app's own `compare_to_tape()`. Without `--tape` it uses the run's `tape_validation.json`.
- **The app** shows both in the measurement panel and in the HTML scan report. Body Avatar, assessment history/BCM
  and the tape-validation API use the pipeline only.
- **Default view in the measurement panel:** Pipeline.

## How to decide (about 10 minutes)

1. **Measure the subject with a tape on the day of a scan.** Use the `TAPE_PROTOCOL` wording in
   `backend/services/body_measurements.py`. Take chest, waist, abdomen and hip, if possible `thigh_left` and
   `calf_left`, and barefoot height.
2. **Run** `.venv/Scripts/python.exe scripts/shapy_compare.py <run_id> --height-cm <measured> --tape chest=.. waist=.. abdomen=.. hip=..`
3. **Repeat once more,** with a second person or with the same person in fitted clothing.
4. **Make the default whichever model is closer by 2 cm or more both times.**
   - To make SHAPY the default, move Body Avatar and assessment history to SHAPY in the same change and recompute
     history. Otherwise the change of method appears as a 10 cm change in the body.
   - If the pipeline wins, ask subjects to wear fitted clothing. That removes its main error.
