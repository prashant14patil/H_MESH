"""
scripts/_generate_phase_b_plan.py
----------------------------------
One-shot generator for docs/PHASE_B_PLAN.md.

This generator script itself is assembled across a few small edits
(to stay under per-call output limits), but docs/PHASE_B_PLAN.md is
produced by a SINGLE terminal-based file write: running this script
once performs one write_text() call with the fully assembled content,
then reads it back to verify completeness. No diff editor /
replace_in_file is used on docs/PHASE_B_PLAN.md itself.

Does not touch any application code.
"""

from pathlib import Path

DOC_PATH = Path("docs/PHASE_B_PLAN.md")

PART_01 = r"""# Phase B — Pipeline Hardening Implementation Plan
**Tanita Human Mesh Reconstruction POC**

**Status:** Planning document only — no application code modified as part of producing this plan.
**Context:** SMPL model file is still being transferred; no inference has been run yet. This plan is based on static code review of the existing local HMR2 pipeline, consistent with `docs/AUDIT.md`.

**Constraints honored:**
- Read-only review — no code changes made while producing this plan.
- Preserves the existing local HMR2 inference architecture (no Colab/ngrok/external services reintroduced).
- Preserves the existing `/api/inference` endpoint structure (`POST /submit`, `GET /status/{id}`, `GET /mesh/{id}`).
- No new dependencies proposed unless explicitly called out and justified.
- No BCM/body-measurement work included — out of scope for this plan.
- All confirmed findings from `docs/AUDIT.md` (Critical Bugs #1-#10, Compatibility section, Prioritized Plan) are preserved and cross-referenced below rather than re-litigated.

**Legend:** CONFIRMED = Confirmed via direct code read · INFERRED = Inference from code behavior, not yet runtime-verified (pending SMPL file + live test)

---

## 1. Image Validation & Preprocessing

### 1.1 File type / MIME validation
- **Existing:** CONFIRMED `backend/utils/validators.py::_check_extension()` + `_check_magic_bytes()`. Allowlist = `.jpg/.jpeg/.png`; magic-byte check cross-validates extension vs real content.
- **Already handled:** Extension spoofing, non-image uploads, mismatched extension/content.
- **Missing:** Nothing. GIF/WebP/BMP rejected by design — acceptable for POC scope.
- **Proposed change:** None required.
- **Test cases:** N/A.
- **Acceptance criteria:** N/A — already correct.
- **Priority:** None.

### 1.2 Upload size limits
- **Existing:** CONFIRMED `_check_size()`, `MAX_UPLOAD_SIZE_MB` env var (default 10MB). Check happens after `await image.read()` — full file buffered before size check.
- **Already handled:** Rejects oversized files with 400 after the fact.
- **Missing:** No limit enforced before the body is fully read (relates to `docs/AUDIT.md` Critical Bug #6).
- **Proposed minimal change:** Add an early `Content-Length` header check as defense-in-depth; keep the post-read check authoritative.
- **Test cases:** Honest large header -> rejected fast; forged small header with large body -> still caught by the post-read check.
- **Acceptance criteria:** No single request forces unbounded memory buffering.
- **Priority:** P2.

### 1.3 Corrupted image handling
- **Existing:** INFERRED No corruption check in `validate_image()` — only magic bytes inspected. Decode happens later in `_run_inference()`.
- **Already handled:** Decode failure doesn't crash the server; caught generically in `submit_job`'s worker thread, job marked `failed`.
- **Missing:** Job briefly shows `"processing"` before failing — misleading; `detail` is a raw exception string, not a validated/clean message.
- **Proposed minimal change:** In `validate_image()`, call `PIL.Image.open(...).verify()` and raise `ValidationError` on failure, so corruption is caught synchronously at submit time.
- **Test cases:** Truncated JPEG; corrupt PNG CRC.
- **Acceptance criteria:** Corrupted images rejected synchronously with HTTP 400, never reach `"processing"` state.
- **Priority:** P1.

### 1.4 EXIF orientation correction
- **Existing:** CONFIRMED absent — no `exif_transpose()` call anywhere in the codebase.
- **Already handled:** Nothing.
- **Missing:** Phone photos with EXIF rotation tags aren't auto-corrected; combined with the full-image bbox fallback (see 2.1), sideways photos silently produce a garbage mesh.
- **Proposed minimal change:** Call `PIL.ImageOps.exif_transpose(img)` right after `PILImage.open(...)` in `_run_inference()`.
- **Test cases:** Photo with EXIF orientation tag=6 vs. a manually pre-rotated equivalent — outputs should match.
- **Acceptance criteria:** EXIF-rotated input produces equivalent output to the upright version.
- **Priority:** P1.

### 1.5 Decompression-bomb protection
- **Existing:** CONFIRMED absent — only compressed file size is checked, not decoded pixel dimensions (per `docs/AUDIT.md` Critical Bug #6).
- **Already handled:** Nothing.
- **Missing:** A small PNG can declare a huge resolution -> multi-gigabyte allocation on decode.
- **Proposed minimal change:** After `.verify()` (see 1.3), check `img.size` against a ceiling (e.g. 8000px per side); set `PIL.Image.MAX_IMAGE_PIXELS` explicitly rather than relying on Pillow's default.
- **Test cases:** Minimal PNG header declaring a huge resolution.
- **Acceptance criteria:** No upload can trigger multi-GB allocation.
- **Priority:** P1.

### 1.6 Safe filenames
- **Existing:** INFERRED Filename is never used to construct filesystem paths — jobs are keyed by server-generated UUID; the client-supplied filename only appears in logs and error-message text.
- **Already handled:** Path traversal via filename is not exploitable today.
- **Missing:** Filename is reflected unsanitized into `ValidationError` message strings returned to the client.
- **Proposed minimal change:** Strip control characters and cap length when embedding the filename in any error message.
- **Test cases:** Filename containing traversal sequences, null bytes, or extreme length.
- **Acceptance criteria:** No traversal possible (already true today); all error text is bounded and control-character-free.
- **Priority:** P3.
"""

PART_02 = r"""

---

## 2. Human Detection & Input Quality

### 2.1 No-person detection
- **Existing:** CONFIRMED `InferenceEngine._detect_person_box()` returns the full-image bounding box unconditionally whenever `self.detector is None`, which is the default behavior when `USE_DETECTRON2` is unset/true but detectron2 fails to import (expected on this Windows machine). This is `docs/AUDIT.md` Critical Bug #7.
- **Already handled:** Nothing — the fallback is intentional but silent.
- **Missing:** No-person photos silently produce a meaningless mesh instead of an explicit error.
- **Proposed minimal change:** Requires detectron2 to be actually loaded (intentionally disabled by default on Windows for build-complexity reasons, per audit). When the detector IS active: raise an explicit, user-facing error on zero person detections. When the detector is NOT active (current default on this machine): document as an accepted, labelled POC limitation rather than building an unreliable local heuristic as a stand-in.
- **Test cases:** (detectron2 enabled) zero-person image -> explicit `ValidationError`/failure with a clear message.
- **Acceptance criteria:** Explicit failure when the detector is active; documented limitation (not silent success) when it is not.
- **Priority:** P2.

### 2.2 Multiple-person ambiguity
- **Existing:** CONFIRMED The detector branch in `_detect_person_box()` already picks the largest-area box deterministically (`boxes[np.argmax(areas)]`) when multiple person detections exist.
- **Already handled:** No crash, no random/non-deterministic behavior.
- **Missing:** No signal is returned to the client that more than one person was detected and one was arbitrarily chosen.
- **Proposed minimal change:** Return the detection count from `_detect_person_box()` and surface it as `metadata.multi_person_detected` in the completed-job response (detectron2-only; no change needed to the fallback path).
- **Test cases:** Multi-person photo with the detector enabled.
- **Acceptance criteria:** Ambiguity is surfaced in response metadata without blocking completion.
- **Priority:** P3.

### 2.3 Partial-body detection
- **Existing:** INFERRED No check exists — SMPL always outputs a full 6890-vertex body regardless of how much of the subject is actually framed in the photo.
- **Already handled:** Nothing.
- **Missing:** No signal that the input was a partial-body (e.g. waist-up) photo, which the model will still "complete" into a full body with potentially large error.
- **Proposed minimal change:** Heuristic on the detected/fallback bbox aspect ratio (e.g. height/width < ~1.3 -> `metadata.partial_body_warning: true`). Explicitly advisory-only, never blocking — false-positive risk for legitimately wide framing is accepted and documented.
- **Test cases:** Waist-up photo vs. full-body photo at the same framing width.
- **Acceptance criteria:** Advisory metadata only; never prevents job completion.
- **Priority:** P2.

### 2.4 Small person / poor image quality
- **Existing:** INFERRED No check exists — tiny subjects within a large frame are processed without complaint or warning.
- **Already handled:** Nothing.
- **Missing:** No minimum-resolution floor; "small subject in otherwise large frame" case needs an actual detector to solve properly.
- **Proposed minimal change:** Reject images below e.g. 200px in either dimension in `validate_image()`. The harder "small subject, large frame" case is deferred and documented as a limitation requiring a working detector.
- **Test cases:** 50x50px image rejected; a normal full-size photo passes unaffected.
- **Acceptance criteria:** Trivially tiny images rejected; harder case explicitly documented as a known limitation.
- **Priority:** P2.

### 2.5 Occlusion warnings
- **Existing:** CONFIRMED absent — only `pred_vertices` is consumed from the model output; no confidence or visibility signal is extracted anywhere.
- **Already handled:** Nothing.
- **Missing:** No occlusion-warning capability at all.
- **Proposed minimal change:** Deferred until live model output can actually be inspected (blocked on the SMPL file arriving and a first successful run) — do not implement speculatively against unverified output shapes.
- **Test cases:** Deferred pending live run.
- **Acceptance criteria:** Deferred pending live run.
- **Priority:** P3.
"""

PART_03 = r"""

---

## 3. Inference Job Management

### 3.1 Job lifecycle and status transitions
- **Existing:** CONFIRMED Clean `queued -> processing -> completed|failed` state machine in `InferenceEngine.submit_job()` / `_worker()`.
- **Already handled:** Happy path and generic-exception failure path both correct.
- **Missing:** No `created_at`/`updated_at` timestamps on job records.
- **Proposed minimal change:** Add timestamps at each state transition — purely additive, no API contract change.
- **Test cases:** Inspect the job dict at each stage of a submission.
- **Acceptance criteria:** Every job carries creation and last-update timestamps.
- **Priority:** P2.

### 3.2 Timeouts and error handling
- **Existing:** CONFIRMED No timeout on the worker thread — a model hang leaves a job in `processing` forever.
- **Already handled:** Generic exceptions inside `_run_inference()` are caught correctly and mapped to `failed`.
- **Missing:** No wall-clock watchdog for jobs that never raise but also never return.
- **Proposed minimal change:** Add an `INFERENCE_TIMEOUT_SEC` env var and a watchdog that marks a job `failed` with a clear "timed out" reason after the threshold. Cannot forcibly cancel a running CUDA kernel without process isolation — this provides detection, not true cancellation.
- **Test cases:** Simulated slow job via a monkeypatched sleep in the worker.
- **Acceptance criteria:** Stuck jobs are eventually marked `failed` with an explicit timeout reason.
- **Priority:** P1.

### 3.3 Temporary-file cleanup
- **Existing:** CONFIRMED OBJ files in `outputs/meshes/` and job dict entries are never deleted — unbounded growth over the life of the process.
- **Already handled:** Nothing.
- **Missing:** No eviction logic of any kind.
- **Proposed minimal change:** Background sweep evicting job records/files older than a new `JOB_RETENTION_HOURS` env var; never touch jobs still in `queued`/`processing`.
- **Test cases:** A backdated completed job gets swept; an in-progress job is left untouched regardless of age.
- **Acceptance criteria:** No unbounded growth of `outputs/meshes/` or the in-memory job dict under long-running operation.
- **Priority:** P2.

### 3.4 Duplicate job protection
- **Existing:** CONFIRMED No dedup — identical images submitted twice create two fully independent jobs.
- **Already handled:** Nothing.
- **Missing:** No content-hash dedup.
- **Proposed minimal change:** Not recommended for Phase B. The frontend already guards against duplicate submission in its state machine, and adding a backend cache changes response semantics (e.g. "is this a fresh job or a cached one?") without a clear POC-stage justification. Explicitly deferred.
- **Test cases:** N/A (deferred).
- **Acceptance criteria:** N/A (deferred).
- **Priority:** P3 (deferred).

### 3.5 Concurrency and GPU-memory management
- **Existing:** CONFIRMED Unbounded thread spawning in `submit_job()` — every submitted job gets its own `threading.Thread`, so concurrent submissions all hit the GPU simultaneously with no coordination.
- **Already handled:** The single-job case is correct by design; `self.jobs[job_id]` dict writes from multiple threads are simple key assignments and don't corrupt each other.
- **Missing:** No concurrency ceiling — multiple simultaneous submissions risk CUDA out-of-memory on the GPU (or extreme slowdown/thrashing on the CPU fallback path), especially relevant for the target low-spec Intel-integrated-graphics deployment machine called out in `docs/AUDIT.md`.
- **Proposed minimal change:** Introduce a new `MAX_CONCURRENT_JOBS` env var (default `1`, matching the current effective single-GPU reality) and guard the model forward pass in `_run_inference()` with a `threading.Semaphore(MAX_CONCURRENT_JOBS)`. Jobs beyond the limit simply wait their turn inside the worker thread rather than being rejected — status stays `queued`/`processing` as appropriate, no new status value or API contract change needed.
- **Test cases:** Three rapid submissions with `MAX_CONCURRENT_JOBS=1` all eventually complete without CUDA OOM, and are observably serialized (job 2 only enters the model forward pass after job 1 finishes it).
- **Acceptance criteria:** At most `MAX_CONCURRENT_JOBS` inference forward-passes execute on the GPU/CPU at any instant; no OOM under rapid concurrent submission in testing.
- **Priority:** P1.
"""
