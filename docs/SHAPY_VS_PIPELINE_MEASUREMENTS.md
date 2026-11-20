# SHAPY vs the pipeline: what each measures and which to use

2026-11-20, branch `feature/shapy-as-default-measurements`. Every scan gives girths from two models, and the
measurement panel shows both side by side. In the first tape check, SHAPY was about 2 cm outside the tape ranges.
The pipeline was 4–14 cm too large with its old setting, and about 3 cm out with the HMR2-prior setting that became
its default on 2026-11-20.

| | Pipeline | SHAPY |
|---|---|---|
| What it is | HMR2 start, then one SMPL shape fitted to the person's outline in every view, held close to HMR2's own shape estimate (prior 0.2) | Single-image network ([Choutas et al., CVPR 2022](https://shapy.is.tue.mpg.de/)), trained on model-agency photos paired with height/chest/waist/hip measurements plus crowd-sourced shape words; median over the run's frames and their mirror images |
| Strength | Specific to this person: uses the full 360° outline and the person's pose | Can partly see through clothing; repeated recordings agree within 0.2–0.5 cm |
| Weakness | The outline includes clothing. Without the prior (old default, prior 0) loose clothing became girth, 4–14 cm too large | Never sees this subject's outline; pulls toward an average body, so it may under-report real change over time |

## Tape check, 2026-11-20

Subject of recordings `777541cd` and `35e61e9b`, who wore a loose T-shirt and cargo trousers in the videos. The
tape was self-measured, as ranges over repeated readings. Values are in cm at the 179 cm height the runs used, for
recording `777541cd`.

| | waist | abdomen | hip | mean error vs midpoint | mean distance outside range |
|---|---|---|---|---|---|
| **Tape** | **88–90** | **95–100** | **98–103** | | |
| SHAPY (same on both recordings) | 88.5 | 90.8 | 105.0 | 3.9 | 2.1 |
| Pipeline, old default (prior 0) | 100.3 | 104.0 | 117.1 | 11.5 | 9.5 |
| Pipeline, HMR2 prior 0.2 (run `0f761c80`; default from 2026-11-20) | 87.7 | 90.6 | 103.1 | 3.6 | 1.6 |
| Clothed visual hull ([investigation §5.1](RECONSTRUCTION_ACCURACY_INVESTIGATION.md)) | 114.3 | 117.0 | 122.7 | | |
| Earlier reasoned estimate (§5.4), now shown wrong | 95–103 | 98–106 | 110–116 | | |

Across all three recordings, the fit/accuracy session's study (investigation §5.8, `scripts/tape_config_study.py`,
`outputs/accuracy_study/tape_config_study.json`) gives this mean distance outside the tape range. Height is fixed
at 179 cm for every row.

| Setting | cm |
|---|---|
| Pipeline, old default (prior 0) | 8.3 |
| Pipeline, body kept inside the outline, no prior | 7.3 |
| Pipeline, inside-only + HMR2 prior 0.2 | 3.5 |
| **Pipeline, HMR2 prior 0.2 (new default; male prior changes it by < 0.5 cm)** | **2.9** |
| Pipeline, prior 0.2 + surface detail | 1.6–1.7 (not primary: it works by filling the garment and overshoots the hip on outline-only variants) |
| SHAPY (the two recordings it has run on) | 2.1–2.2 |

- **Clothing, not the fit maths, caused the old error.** Loose clothes added about 25 cm to the outline at waist and
  hip (hull 114 / 123 against tape ~89 / ~100), and a fit that fills the outline turns that into girth.
- **Held-out IoU against a clothed outline picked the worst setting.** It rewards filling the clothes, so it must not
  choose shape settings.
- **Remaining error:**
  - The abdomen is 4–8 cm under the tape in every learned variant, SHAPY included. Ten shape coefficients cannot
    hold this belly.
  - The pipeline's waist is 2–4 cm under on 2 of 3 recordings.
- **Limits:**
  - One person, in loose clothing.
  - A 2 cm error in the supplied height moves every value by about 1.1%.
  - Chest and limbs were not taped.

## What this changes

- **Default view:** the measurement panel opens on **Compare both** (`ModelAvatarCompare.jsx`). SHAPY and the
  pipeline appear as Body Avatars side by side (stacked on phones), turning together, each with its own girth cards.
  - With a tape check recorded for the run, each body marks the sites within ±2 cm of the tape, and a table lists
    both models' errors.
  - SHAPY's body is a different mesh (SMPL-X, 10,475 vertices). It borrows its segment colours from the pipeline body
    (nearest vertex, same pose), and its measurement rings come from `shapy_measurement_rings.obj` (SHAPY result
    version 3).
  - The panel remembers each viewer's tab under the key `measurementModelView.v3`, so everyone starts on the
    comparison once.
- **Everything else stays on the pipeline:** Body Avatar, assessment history/BCM, the tape-validation API and the
  size recommendation.
  - With the new default, new runs land within about 3 cm of the tape.
  - Runs made with the old default keep their inflated numbers until re-run. Re-run them before comparing history.
- **Checks still to do:**
  1. Barefoot height and chest/thigh tape for the same subject.
  2. A scan in fitted clothing.
  3. A second person.
  4. The abdomen shortfall.

## Tools

- **`scripts/scan_360.py`** runs SHAPY by default after the reconstruction (`--no-shapy` skips it; the old `--shapy`
  flag is still accepted) and prints both models, SHAPY first. A SHAPY failure never fails the scan.
- **`scripts/shapy_compare.py RUN --height-cm H --tape waist=88-90 abdomen=95-100 hip=98-103`** scores both models
  with the app's own `compare_to_tape()`, against the midpoint and the range. Without `--tape` it uses the run's
  `tape_validation.json`. Take tape readings with the `TAPE_PROTOCOL` wording in `backend/services/body_measurements.py`.
