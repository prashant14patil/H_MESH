# Video frame quality investigation and V2 decision

## Result

**No V2 mesh batch was generated.** All 30 previously extracted frames retain the extractor's `is_blurry: true` flag, and therefore none meets this investigation's conservative selection criterion of a readable, detected, full-body frame without that warning. This is a screening decision, not a definitive determination that every frame is unusable. No blur threshold was lowered and the V1 outputs were preserved.

Only two of 262 decoded source frames exceeded the extractor's current default native-resolution whole-frame Laplacian-variance threshold of 100: indices 109 and 110 at 3.615 and 3.647 seconds, respectively. Their scores were 100.24 and 100.08. They are adjacent frames, not a diverse set of views; their person-region sharpness and pose were not independently validated. A new 8–12-view quality batch cannot be justified by these two marginal threshold passes alone.

## Video and blur investigation

- The source WebM reports 1280×720, 1000 FPS and 8687 frames in its OpenCV header. These FPS/count fields are implausible for this recording.
- The existing extraction metadata reports 8.687 seconds and approximately 30.16 FPS from a timestamp-based measured walk. This analysis decoded 262 frames. The measured FPS is an estimate, not a proof of constant frame rate.
- All 30 extracted JPEGs were readable and measured 1280×720.
- The extractor computes grayscale whole-frame Laplacian variance on decoded frames before JPEG encoding. Its current default `BLUR_THRESHOLD` is 100 and scores below that threshold are flagged blurry. The historical process's actual environment value was not persisted, so the stored flags are authoritative for that run; 100 is used here only as the current-default diagnostic comparison.
- Stored extracted-frame scores range from 40.60 to 94.09, median 77.955. Recomputed JPEG scores range from approximately 41.06 to 92.97, median 77.39. The whole decoded video's native-resolution scores range from 34.26 to 100.24, median 79.19.
- Resizing decoded frames to 640×360 with area interpolation produced scores from 143.45 to 313.11, median 251.82. This demonstrates scale sensitivity: comparing the resized scores against the native-resolution threshold would be invalid and does not make the underlying footage sharper.
- Laplacian variance also depends on background edges, texture, noise and compression. The landmark-derived person-region scores in the inventory are exploratory; no threshold has been validated for those crops.

The inspected subject crops show limited fine detail, but the scaled preview does **not** establish whether motion blur, focus, compression or limited subject resolution is the cause. Motion blur is therefore **undetermined**, not confirmed. An independent visual assessment of native frames and a controlled sharpness reference would be needed to validate or recalibrate the screening criterion.

## Complete frame analysis and ranking

- `outputs/video_to_mesh/quality_analysis_ecff78_002/frame_quality.csv`
- `outputs/video_to_mesh/quality_analysis_ecff78_002/frame_quality.json`
- `outputs/video_to_mesh/quality_analysis_ecff78_001/subject_crops.jpg` — four-frame qualitative comparison.

The CSV and JSON contain all 30 frames, timestamps, stored blur scores and flags, recomputed JPEG and resized scores, diagnostic landmark-region scores, detection status, landmark visibility data, full-body visibility, duplicate-comparison status, review reasons, V1 membership, visibility-first review rank and sharpness-only rank.

The pose processor's `detection_confidence` is the **mean of tracked landmark visibility values**, not a calibrated person-detection probability; the analysis records the latter as unavailable. Its single-pose output cannot exclude additional people. `full_body_visible` means all tracked landmarks passed the pose visibility threshold, not a guarantee that the entire physical body is unobstructed. Accepted frames were not rejected as duplicates by the extractor; the first frame has no prior comparison. No candidate duplicate rejections were recorded in this capture.

The highest stored native whole-frame scores were:

| Sharpness rank | Extracted frame | Timestamp (s) | Score | All tracked landmarks visible |
|---:|---|---:|---:|---|
| 1 | frame_020 | 5.518 | 94.09 | No |
| 2 | frame_004 | 0.896 | 91.47 | Yes |
| 3 | frame_017 | 4.656 | 90.30 | Yes |
| 4 | frame_013 | 3.489 | 88.98 | No |
| 5 | frame_021 | 5.793 | 87.51 | No |
| 6 | frame_019 | 5.215 | 87.22 | No |
| 7 | frame_025 | 6.975 | 87.02 | Yes |
| 8 | frame_029 | 8.112 | 86.36 | Yes |
| 9 | frame_030 | 8.414 | 85.71 | Yes |
| 10 | frame_006 | 1.455 | 84.02 | Yes |

These are **relative review candidates**, not accepted sharp frames. The separate review ranking prioritizes eligibility, full-body visibility and mean landmark visibility before sharpness; it does not prove viewpoint diversity. Across all extracted frames, the existing pose summary reports a person in 30 and all tracked landmarks visible in 17. Viewpoint labels remain unvalidated.

## V1 versus V2

| Measure | V1 | V2 |
|---|---|---|
| Selected extracted frames | 10 | 0 — deferred |
| Frames flagged blurry | 10/10 | No selection |
| Mean stored extractor Laplacian variance | 77.69 | Not applicable |
| Successful independent meshes | 10 | 0 — no inference submitted |
| Batch elapsed | 10.797 s | Not applicable |
| Geometry | Each OBJ structurally checked: 6,890 vertices and 13,776 faces | No OBJ generated |
| Visual mesh improvement | Not established | Cannot compare |

A higher relative score alone would not establish a better reconstruction. No V2 previews exist and no claim of improved mesh appearance can be made. The existing V1 meshes remain independent single-image predictions, not a unified 360° mesh or validated body measurements.

## Recommendations for a new recording

1. Use a fixed, stable camera with sufficient light; avoid low-light auto-exposure and inspect focus on the subject before recording. If camera controls permit it, use a shorter exposure to reduce motion smear while maintaining adequate illumination.
2. Have one subject rotate slowly and steadily, with the entire body inside the frame and no nearby bystanders or severe occlusions. Avoid subject movement toward or away from the camera.
3. Fill more of the image with the subject while retaining head-to-toe margins. Record at the intended capture resolution and settings; avoid digital enlargement as a substitute for detail.
4. Review several native-resolution frames from front, back, both profiles and intermediate positions for visible sharpness before inference. Compare against a stationary, in-focus reference captured with the same camera.
5. Validate a person-region sharpness measure and any thresholds against representative labelled footage before changing them. Whole-frame Laplacian variance can be dominated by the background.
6. Re-run extraction and this inventory on the new video. Only submit a diverse selection when quality and visibility are supported by the data.

## Reproducibility and tests

Analysis command:

```powershell
python scripts/analyze_video_frame_quality.py --capture data/production_validation/ecff78fe-773e-4eb8-a785-c356e3be4d19 --output outputs/video_to_mesh/new_quality_run --v1 outputs/video_to_mesh/ecff78_batch_001/results.json
```

The output directory is created exclusively. The script reads existing video, extraction and pose data and does not submit inference. `--diagnostic-threshold` is explicitly a native-resolution source-video comparison; it does not change stored frame flags or production settings.

The analysis script compiled successfully. The three existing mocked batch-client tests passed; they cover partial failure, OBJ validation and no-overwrite behavior. The quality measurements above are from the real video, not synthetic tests. No additional HMR2 inference or V2 geometry test was run. The generated frame inventories and crops contain personal data and should be handled under the applicable privacy policy.
