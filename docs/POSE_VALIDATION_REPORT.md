# MediaPipe validation report

## Environment

Windows 11, Python 3.12.10. The existing HMR2 environment has NumPy 1.26.4, SciPy 1.12.0, OpenCV 4.10.0.84 and Torch 2.5.1+cu121. Fresh-process imports of NumPy, Torch, Lightning and chumpy succeeded, but a fresh HMR2 inference was not tested.

The earlier unconstrained MediaPipe installation selected newer transitive dependencies and upgraded NumPy; NumPy and SciPy were restored. A later `--no-deps` installation added MediaPipe and JAX to the existing environment, leaving missing dependencies. `pip check` reports missing ml-dtypes, matplotlib, opencv-contrib-python and sounddevice, and pandas 3.1.1 requires NumPy >=2.0.2. MediaPipe import there fails because matplotlib is missing. Do not assume the existing environment is healthy or upgrade its NumPy to 2.x.

An isolated `.venv-pose` was created with Python 3.12 and NumPy 1.26.4, SciPy 1.12.0, MediaPipe 0.10.14, JAX/JAXlib 0.4.30 and opencv-contrib-python 4.10.0.84. `pip check` passed. MediaPipe imported and the existing PoseLandmarker and ImageSegmenter loaded on CPU. TensorFlow Lite feedback and protobuf deprecation warnings were emitted.

Recreate the isolated test environment on Windows with `py -3.12 -m venv .venv-pose`, then `.\.venv-pose\Scripts\python.exe -m pip install -r requirements-pose-validation.txt` and `.\.venv-pose\Scripts\python.exe -m pip check`. The requirements file pins direct dependencies, not every transitive dependency; revalidate the resolved environment. This environment does not include the HMR2 runtime.

## Real video API test

The test used a local 1280×720 browser-recorded WebM showing a person rotating. A separate test server on port 18763 hosted the existing capture and pose routers with a separate output directory, not the HMR2 app.

- Upload and extraction completed. The container reported 1000 FPS and 8687 frames; measured timing reported 8.687 seconds and approximately 30.16 FPS.
- Of 30 candidates, only 2 were accepted; 28 failed to read. Both retained frames were flagged blurry.
- MediaPipe completed on both retained frames: 2/2 persons detected, 13 tracked landmarks each, 2/2 marked full-body visible, 0 requiring review.
- The viewpoint estimator labelled both retained frames front and reported insufficient coverage. Its labels are heuristic.
- Pose visualization, segmentation mask and metadata endpoints returned HTTP 200.

The initial run was **not a successful full-rotation extraction**. Sequential decoding reads 262 frames over 8.687 seconds, while the old extractor sought each target using `CAP_PROP_POS_MSEC` and received 28 read failures. The second extracted frame visually appeared front-facing despite its nominal 3.475-second timestamp. A quantitative closest-frame comparison failed due to shell quoting; that mismatch was not independently measured.

## Retest after forward-only sampling change

The extractor was changed to decode sequentially and record decoded timestamps rather than repeatedly seeking. A fresh isolated API server processed the same original WebM end to end:

- Upload succeeded; 30 candidates were sampled, 30 extracted and none rejected. Timestamps ranged from 0.000 to 8.414 seconds. The measured average FPS was 30.16.
- MediaPipe detected a person in all 30 frames and returned 13 tracked landmarks per frame. Seventeen frames were marked full-body visible and five required review; pose processing took 3.812 seconds.
- Viewpoint output counted 14 front, 6 back, 10 intermediate and no left or right sides; coverage remained insufficient. The labels are heuristic and are not validated as accurate.
- Pose visualization, segmentation mask and metadata endpoints returned HTTP 200.
- Three sampling regression tests passed: decoded timestamps/no seeking, end-of-file accounting and missing-timestamp fallback (`python -m unittest scripts.test_sequential_sampling -v` in the isolated environment).

This validates the capture and pose routers in the isolated test server, not the combined production backend or main application upload flow. The FPS-based timestamp fallback is approximate when presentation timestamps are unavailable.

## Sequential-frame diagnostic

The source contact sheet visibly shows front, side, back and intermediate positions. To isolate pose from extraction, nine frames sampled sequentially at approximately one-second intervals were sent directly to the existing pose processor. This bypasses the production extractor and is not an end-to-end test.

MediaPipe detected a person and returned 13 tracked landmarks on all 9 frames. It marked 5 full-body visible and 1 requiring review. The heuristic reported 4 front, 2 back and 3 intermediate, with zero left/right side and insufficient coverage. Visually side-facing frames around 2 and 6 seconds were labelled ambiguous intermediate. Front/back labels are not validated as reliable classifications.

## Frontend

A separate Vite test page rendered the existing FrameGallery component against real isolated-server API results. The two thumbnails, timing, rejection counts, blur badges and insufficient-coverage warning displayed. The frame detail displayed the real pose overlay and landmark visibility table. This did not test the main application upload interaction or the production combined backend.

## Remaining blockers

1. Extend WebM sampling regression tests to additional codecs, timestamp discontinuities, variable frame rates and independently verify timestamp accuracy.
2. Run the combined production upload-to-gallery workflow with all views retained.
3. Validate or replace the viewpoint heuristic before using coverage as evidence of full rotation.
4. Establish a reproducible supported production dependency solution without upgrading NumPy in the HMR2 environment; resolve the existing environment conflicts and run a fresh single-photo HMR2 inference.
5. Browser-test the main application workflow. The isolated test server does not load HMR2.
6. Validation artifacts under `data/pose_validation` contain personal video frames and pose results; handle them according to the applicable privacy policy.

No HMR2 inference logic was modified. Multi-view reconstruction was not started.
