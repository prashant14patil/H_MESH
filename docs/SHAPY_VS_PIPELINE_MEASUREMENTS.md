# SHAPY vs the pipeline: what each measures and which to use

2026-11-20, branch `feature/shapy-as-default-measurements`. Every scan gives girths from two models. In the first tape
check, SHAPY was within a few cm and the pipeline was 4–14 cm too large, so **SHAPY is now the default view**.

| | Pipeline (default fit) | SHAPY |
|---|---|---|
| What it is | HMR2 start, then one SMPL shape fitted to the person's outline in every view | Single-image network ([Choutas et al., CVPR 2022](https://shapy.is.tue.mpg.de/)), trained on model-agency photos paired with height/chest/waist/hip measurements plus crowd-sourced shape words; median over the run's frames and their mirror images |
| Strength | Specific to this person: uses the full 360° outline | Can partly see through clothing; repeated recordings agree within 0.2–0.5 cm |
| Weakness | Measures the clothed outline. In loose clothing it was 4–14 cm too large | Never sees this subject's outline; pulls toward an average body, so it may under-report real change over time |

## Tape check, 2026-11-20

Subject of recordings `777541cd` and `35e61e9b`, who wore a loose T-shirt and cargo trousers in the videos. The
tape was self-measured, as ranges over repeated readings. All values are in cm at the 179 cm height the runs used.

| | waist | abdomen | hip | mean error vs midpoint | mean distance outside range |
|---|---|---|---|---|---|
| **Tape** | **88–90** | **95–100** | **98–103** | | |
| SHAPY (same on both recordings) | 88.5 | 90.8 | 105.0 | 3.9 | 2.1 |
| Pipeline, current default fit | 100.3 | 104.0 | 117.1 | 11.5 | 9.5 |
| Pipeline with a strong pull to HMR2's own shape (prior 0.2, run `0f761c80`) | 87.7 | 90.6 | 103.1 | 3.6 | 1.6 |
| Clothed visual hull ([investigation §5.1](RECONSTRUCTION_ACCURACY_INVESTIGATION.md)) | 114.3 | 117.0 | 122.7 | | |
| Earlier reasoned estimate (§5.4), now shown wrong | 95–103 | 98–106 | 110–116 | | |

- **SHAPY:** waist inside the range, hip 2 cm above, abdomen 4 cm below.
- **The default pipeline fit is above the range at every site.** Clothing added about 25 cm to the outline at waist
  and hip (hull 114 / 123 against tape ~89 / ~100). That is far more than the investigation assumed, and the
  silhouette fit fills it.
- **The investigation's rejected setting matched the tape.** The fit pulled strongly toward HMR2's own shape was
  rejected because its held-out IoU against the clothed outline was lower, yet it matched the tape as well as
  SHAPY. Held-out IoU against a clothed outline rewards filling the clothes; it should not choose shape settings.
- **The abdomen is low in both learned estimates** (about 91 against 95–100). Ten shape coefficients cannot hold
  this belly; the trunk surface-detail stage may help there.
- **Limits of this check:**
  - One person, in loose clothing.
  - A 2 cm error in the supplied height moves every value by about 1.1%, which is not enough to change the result.
  - Chest and limbs were not taped.

## What this changes, and what is next

- **Default:** the measurement panel opens on **Compare both** (`ModelAvatarCompare.jsx`): SHAPY and the pipeline as
  Body Avatars side by side (stacked on phones), turning together, each with its own girth cards.
  - With a tape check recorded for the run, each body marks the sites within ±2 cm of the tape, and a table lists
    both models' errors.
  - SHAPY's body is a different mesh (SMPL-X, 10,475 vertices). It borrows its segment colours from the pipeline body
    (nearest vertex, same pose), and its measurement rings come from `shapy_measurement_rings.obj` (SHAPY result
    version 3).
  - `scripts/scan_360.py` prints SHAPY first.
  - The panel remembers each viewer's tab under the key `measurementModelView.v3`, so everyone starts on the
    comparison once.
- **Still on the pipeline:** Body Avatar cards, assessment history/BCM and the tape-validation API, which today show
  values 4–14 cm too large in loose clothing.
  - Move them to SHAPY in one change and recompute history. Otherwise the change of method appears as a 10 cm
    change in the body.
  - These files are being edited by other sessions, so coordinate first.
- **Fit hypothesis (not a result):** keep the term that keeps the body inside the outline (clothing never makes the
  outline smaller than the body), stop rewarding a body that fills the outline, and use a learned shape (SHAPY or
  HMR2) as a strong prior. Test it against this tape before adopting it.
- **Checks still to do:**
  1. Barefoot height and chest/thigh tape for the same subject.
  2. A scan in fitted clothing. If the pipeline then matches the tape, its error is purely clothing.
  3. A second person.

## Tools

- **`scripts/scan_360.py`** runs SHAPY by default after the reconstruction (`--no-shapy` skips it; the old `--shapy`
  flag is still accepted) and prints both models. A SHAPY failure never fails the scan.
- **`scripts/shapy_compare.py RUN --height-cm H --tape waist=88-90 abdomen=95-100 hip=98-103`** scores both models
  with the app's own `compare_to_tape()`, against the midpoint and the range. Without `--tape` it uses the run's
  `tape_validation.json`. Take tape readings with the `TAPE_PROTOCOL` wording in `backend/services/body_measurements.py`.
