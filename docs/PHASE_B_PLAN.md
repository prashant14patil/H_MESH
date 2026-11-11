# Phase B — Pipeline Hardening Implementation Plan
**Tanita Human Mesh Reconstruction POC**

**Status:** Planning document only — no code modified as part of producing this plan.
**Context:** SMPL model file is still being transferred; no inference has been run yet. This plan is based on static code review of the existing local HMR2 pipeline.

**Constraints honored:**
- Read-only review — no code changes made while producing this plan.
- Preserves the existing local HMR2 inference architecture (no Colab/ngrok/external services reintroduced).
- Preserves the existing `/api/inference` endpoint structure (`POST /submit`, `GET /status/{id}`, `GET /mesh/{id}`).
- No new dependencies proposed unless explicitly called out and justified.
- No BCM/body-measurement work included — out of scope for this plan.

**Legend:** 🟢 Confirmed via direct code read · 🟡 Inference from code behavior, not yet runtime-verified (pending SMPL file + live test)

---

## 1. Image Validation & Preprocessing

### 1.1 File type / MIME validation
- **Existing:** 🟢 `backend/utils/validators.py::_check_extension()` + `_check_magic_bytes()`. Allowlist = `.jpg/.jpeg/.png`; magic-byte check cross-validates extension vs real content.
- **Already handled:** Extension spoofing, non-image uploads, mismatched extension/content.
- **Missing:** Nothing. GIF/WebP/BMP rejected by design — acceptable for POC scope.
- **Proposed change:** None required.
- **Test cases:** N/A.
- **Acceptance criteria:** N/A — already correct.
- **Priority:** None.

### 1.2 Upload size limits
- **Existing:** 🟢 `_check_size()`, `MAX_UPLOAD_SIZE_MB` env var (default 10MB). Check happens **after** `await image.read()` — full file buffered before size check.
- **Already handled:** Rejects oversized files with 400 after the fact.
- **Missing:** No limit enforced before the body is fully read.
- **Proposed minimal change:** Add an early `Content-Length` header check as defense-in-depth; keep post-read check authoritative.
- **Test cases:** Honest large header → rejected fast; forged small header with large body → still caught.
- **Acceptance criteria:** No single request forces unbounded memory buffering.
- **Priority:** P2.

### 1.3 Corrupted image handling
- **Existing:** 🟡 No corruption check in `validate_image()` — only magic bytes inspected. Decode happens later in `_run_inference()`.
- **Already handled:** Decode failure doesn't crash server; caught generically, job marked `failed`.
- **Missing:** Job shows "processing" before failing — misleading; `detail` is a raw PIL exception string.
- **Proposed minimal change:** In `validate_image()`, call `PIL.Image.open(...).verify()` and raise `ValidationError` on failure.
- **Test cases:** Truncated JPEG; corrupt PNG CRC.
- **Acceptance criteria:** Corrupted images rejected synchronously with 400.
- **Priority:** P1.

### 1.4 EXIF orientation correction
- **Existing:** 🟢 Confirmed absent — no `exif_transpose()` anywhere.
- **Already handled:** Nothing.
- **Missing:** Phone photos with EXIF rotation tags aren't auto-corrected; combined with full-image bbox fallback, sideways photos silently produce garbage.
- **Proposed minimal change:** Call `PIL.ImageOps.exif_transpose(img)` right after `PILImage.open(...)` in `_run_inference()`.
- **Test cases:** Photo with EXIF orientation=6 vs manually-rotated equivalent.
- **Acceptance criteria:** EXIF-rotated input produces equivalent output to upright version.
- **Priority:** P1.

### 1.5 Decompression-bomb protection
- **Existing:** 🟢 Confirmed absent — only compressed file size checked, not decoded pixel dimensions.
- **Already handled:** Nothing.
- **Missing:** Small PNG can declare huge resolution → multi-GB allocation on decode.
- **Proposed minimal change:** After `.verify()` (1.3), check `img.size` against a ceiling (e.g. 8000px); set `PIL.Image.MAX_IMAGE_PIXELS` explicitly.
- **Test cases:** Minimal PNG declaring huge resolution.
- **Acceptance criteria:** No upload triggers multi-GB allocation.
- **Priority:** P1.

### 1.6 Safe filenames
- **Existing:** 🟡 Filename never used in filesystem paths — jobs keyed by UUID; only appears in logs/error text.
- **Already handled:** Path traversal not exploitable.
- **Missing:** Filename reflected unsanitized in error `detail` strings.
- **Proposed minimal change:** Strip control chars, cap length when embedding filename in messages.
- **Test cases:** Filename with traversal sequences, null bytes, extreme length.
- **Acceptance criteria:** No traversal possible (already true); bounded error text.
- **Priority:** P3.

---

## 2. Human Detection & Input Quality

### 2.1 No-person detection
- **Existing:** 🟢 `_detect_person_box()` returns full-image bbox unconditionally when detector is `None` (default `USE_DETECTRON2=false`). `docs/AUDIT.md` Critical Bug #7.
- **Already handled:** Nothing.
- **Missing:** No-person photos silently produce a meaningless mesh.
- **Proposed minimal change:** Requires detectron2 (intentionally disabled by default for Windows build reasons). When enabled: raise explicit error on zero detections. When disabled: document as accepted limitation. Do not build an unreliable local heuristic.
- **Test cases:** (detectron2 enabled) zero-person image → explicit failure.
- **Acceptance criteria:** Explicit failure when detector active; documented limitation otherwise.
- **Priority:** P2.

### 2.2 Multiple-person ambiguity
- **Existing:** 🟢 Detector branch already picks largest-area box deterministically.
- **Already handled:** No crash/random behavior.
- **Missing:** No signal of ambiguity returned.
- **Proposed minimal change:** Return detection count, surface as `metadata.multi_person_detected` (detectron2-only).
- **Test cases:** Multi-person photo with detector enabled.
- **Acceptance criteria:** Ambiguity surfaced in metadata.
- **Priority:** P3.

### 2.3 Partial-body detection
- **Existing:** 🟡 No check — SMPL always outputs a full 6890-vertex body regardless of framing.
- **Already handled:** Nothing.
- **Missing:** No partial-body signal.
- **Proposed minimal change:** Heuristic on bbox aspect ratio (height/width < ~1.3 → `metadata.partial_body_warning: true`), advisory only, never blocking (false-positive risk for wide framing).
- **Test cases:** Waist-up photo vs full-body photo.
- **Acceptance criteria:** Advisory metadata only, never blocks completion.
- **Priority:** P2.

### 2.4 Small person / poor image quality
- **Existing:** 🟡 No check — tiny subjects processed without complaint.
- **Already handled:** Nothing.
- **Missing:** No minimum-resolution floor.
- **Proposed minimal change:** Reject images below e.g. 200px in either dimension in `validate_image()`. "Small subject in large frame" deferred (needs real detector).
- **Test cases:** 50×50px image rejected; normal photo passes.
- **Acceptance criteria:** Trivial-size images rejected; harder case documented as limitation.
- **Priority:** P2.

### 2.5 Occlusion warnings
- **Existing:** 🟢 Confirmed absent — only `pred_vertices` consumed; no confidence/visibility extraction.
- **Already handled:** Nothing.
- **Missing:** No occlusion warning capability.
- **Proposed minimal change:** Deferred until live model output is inspected (needs SMPL file first) — do not implement speculatively.
- **Test cases:** Deferred.
- **Acceptance criteria:** Deferred.
- **Priority:** P3.

---

## 3. Inference Job Reliability

### 3.1 Job lifecycle and status transitions
- **Existing:** 🟢 Clean `queued → processing → completed|failed` state machine.
- **Already handled:** Happy path and generic-exception failure path both correct.
- **Missing:** No `created_at`/`updated_at` timestamps.
- **Proposed minimal change:** Add timestamps at each transition — purely additive, no API contract change.
- **Test cases:** Inspect job dict at each stage.
- **Acceptance criteria:** Every job has creation/update times.
- **Priority:** P2.

### 3.2 Timeouts and error handling
- **Existing:** 🟢 No timeout on the worker thread — a hang leaves a job `processing` forever.
- **Already handled:** Generic exceptions caught correctly.
- **Missing:** No wall-clock watchdog.
- **Proposed minimal change:** Add `INFERENCE_TIMEOUT_SEC` (new env var) watchdog; mark `failed` on timeout. Cannot force-kill a running CUDA kernel without process isolation — this provides detection, not cancellation.
- **Test cases:** Simulated slow job via monkeypatched sleep.
- **Acceptance criteria:** Stuck jobs marked `failed` with clear reason.
- **Priority:** P1.

### 3.3 Temporary-file cleanup
- **Existing:** 🟢 OBJ files and job dict entries never deleted — unbounded growth.
- **Already handled:** Nothing.
- **Missing:** No eviction logic.
- **Proposed minimal change:** Background sweep evicting records/files older than `JOB_RETENTION_HOURS` (new env var); never touch in-progress jobs.
- **Test cases:** Backdated job swept; in-progress job untouched.
- **Acceptance criteria:** No unbounded growth under long-running operation.
- **Priority:** P2.

### 3.4 Duplicate job protection
- **Existing:** 🟢 No dedup — identical images create independent jobs each time.
- **Already handled:** Nothing.
- **Missing:** No content-hash dedup.
- **Proposed minimal change:** Not recommended for Phase B — frontend already guards submission; backend cache changes response semantics without clear justification. Defer.
- **Test cases:** N/A.
- **Acceptance criteria:** N/A.
- **Priority:** P3 (deferred).

### 3.5 Concurrency and GPU-memory management
- **Existing:** 🟢 Unbounded thread spawning — concurrent submissions all hit GPU simultaneously.
- **Already handled:** Single-job case correct by design.
- **Missing:** No concurrency limit — CUDA OOM risk under load.
- **Proposed minimal change:** `threading.Semaphore(MAX_CONCURRENT_JOBS)` (new env var, default 1) around the model forward pass.
- **Test cases:** 3 rapid submissions all complete without OOM.
- **Acceptance criteria:** At most `MAX_CONCURREN
