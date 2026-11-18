"""Reliability hardening: upload decoding, EXIF orientation, job timeouts, orphaned runs (no model needed)."""
import io
import json
import struct
import threading
import time
import zlib

import pytest
from PIL import Image

from backend.services import body_reconstruction as br
from backend.services import inference_engine as ie
from backend.utils.validators import MIN_PHOTO_SIDE_PX, ValidationError, validate_image


def _jpeg(w=300, h=400, exif_orientation=None):
    buf = io.BytesIO()
    img = Image.new("RGB", (w, h), (200, 50, 50))
    img.paste((20, 200, 20), (0, 0, w, h // 4))  # green band marks the top
    if exif_orientation:
        exif = Image.Exif()
        exif[0x0112] = exif_orientation
        img.save(buf, "JPEG", exif=exif.tobytes())
    else:
        img.save(buf, "JPEG")
    return buf.getvalue()


def _png_header_only(w, h):
    """A tiny PNG whose header declares w x h pixels (decompression-bomb shape)."""
    def chunk(kind, data):
        return struct.pack(">I", len(data)) + kind + data + struct.pack(">I", zlib.crc32(kind + data) & 0xFFFFFFFF)
    ihdr = struct.pack(">IIBBBBB", w, h, 8, 2, 0, 0, 0)
    return b"\x89PNG\r\n\x1a\n" + chunk(b"IHDR", ihdr) + chunk(b"IDAT", zlib.compress(b"\x00" * 10)) + chunk(b"IEND", b"")


# ------------------------------------------------------------ validation
def test_valid_photo_passes():
    validate_image("ok.jpg", _jpeg())


def test_truncated_jpeg_is_rejected_synchronously():
    data = _jpeg()
    with pytest.raises(ValidationError, match="corrupt or truncated"):
        validate_image("cut.jpg", data[: len(data) // 2])


def test_decompression_bomb_is_rejected_from_the_header():
    with pytest.raises(ValidationError, match="maximum|unsafe"):
        validate_image("bomb.png", _png_header_only(30000, 30000))


def test_tiny_photo_upload_is_rejected_but_frames_keep_their_own_rules():
    with pytest.raises(ValidationError, match="at least"):
        validate_image("tiny.jpg", _jpeg(50, 60), min_side_px=MIN_PHOTO_SIDE_PX)
    validate_image("frame.jpg", _jpeg(50, 60))  # frame/crop callers do not opt in


def test_png_with_broken_pixel_data_is_rejected():
    with pytest.raises(ValidationError, match="corrupt or truncated"):
        validate_image("bad.png", _png_header_only(400, 400))


# ------------------------------------------------------------- EXIF
def test_exif_orientation_is_applied_before_inference():
    upright = ie.decode_upright_rgb(_jpeg(300, 400))
    assert upright.shape == (400, 300, 3)
    rotated = ie.decode_upright_rgb(_jpeg(400, 300, exif_orientation=6))  # stored sideways, display rotated 90
    assert rotated.shape == (400, 300, 3)


# ---------------------------------------------------------- job timeout
def _engine():
    eng = object.__new__(ie.InferenceEngine)  # skip model loading
    eng._lock, eng._jobs_lock = threading.Lock(), threading.Lock()
    eng._ready, eng._loading, eng._load_error, eng.jobs = True, False, None, {}
    return eng


def test_hung_job_is_reported_failed_and_late_result_discarded(monkeypatch):
    monkeypatch.setattr(ie, "_INFERENCE_TIMEOUT_SEC", 0.2)
    release = threading.Event()
    eng = _engine()

    def slow(image_bytes, job_id):
        release.wait(5)
        return {"status": "completed", "mesh_path": "x.obj"}
    monkeypatch.setattr(eng, "_run_inference", slow)
    eng.submit_job("j1", b"img")
    time.sleep(0.05)
    assert eng.get_job("j1")["status"] == "processing" and eng.get_job("j1")["started_at"]
    time.sleep(0.3)
    job = eng.get_job("j1")
    assert job["status"] == "failed" and job["timed_out"] and "did not finish" in job["detail"]
    release.set()
    time.sleep(0.1)
    assert eng.get_job("j1")["status"] == "failed"  # not resurrected by the late result


def test_normal_job_completes_with_timestamps(monkeypatch):
    eng = _engine()
    monkeypatch.setattr(eng, "_run_inference", lambda b, j: {"status": "completed", "mesh_path": "x.obj"})
    eng.submit_job("j2", b"img")
    for _ in range(50):
        if eng.get_job("j2")["status"] == "completed":
            break
        time.sleep(0.01)
    job = eng.get_job("j2")
    assert job["status"] == "completed" and job["created_at"] <= job["updated_at"]


# --------------------------------------------------------- orphaned runs
def test_orphaned_runs_are_marked_interrupted_at_startup(tmp_path, monkeypatch):
    monkeypatch.setattr(br, "_ROOT", tmp_path)
    live = "00000000-0000-4000-8000-0000000000a1"
    orphan = "00000000-0000-4000-8000-0000000000a2"
    done = "00000000-0000-4000-8000-0000000000a3"
    monkeypatch.setattr(br, "_RUNS", {live: {"run_id": live, "status": "running"}})
    for rid, status in ((live, "running"), (orphan, "awaiting_person_selection"), (done, "completed")):
        (tmp_path / rid).mkdir()
        (tmp_path / rid / "status.json").write_text(json.dumps({"run_id": rid, "status": status, "stage": "fit"}))
    assert br.mark_orphaned_runs() == [orphan]
    read = lambda rid: json.loads((tmp_path / rid / "status.json").read_text())  # noqa: E731
    assert read(orphan)["status"] == "failed" and read(orphan)["error"]["code"] == "interrupted_by_backend_restart"
    assert read(live)["status"] == "running" and read(done)["status"] == "completed"


# ------------------------------------------------- Phase B: remaining items
def test_filenames_in_errors_are_sanitised_and_bounded():
    nasty = "../../etc/" + "\x00\x1b[31m" + "x" * 500 + ".exe"
    with pytest.raises(ValidationError) as err:
        validate_image(nasty, b"data")
    msg = str(err.value)
    assert "\x00" not in msg and "\x1b" not in msg and len(msg) < 300


def test_gpu_slot_serialises_forward_passes(monkeypatch):
    sem = threading.BoundedSemaphore(1)
    monkeypatch.setattr(ie, "_GPU_SLOTS", sem)
    active, peak = [0], [0]
    lock = threading.Lock()

    def work():
        with ie.gpu_slot():
            with lock:
                active[0] += 1
                peak[0] = max(peak[0], active[0])
            time.sleep(0.05)
            with lock:
                active[0] -= 1
    threads = [threading.Thread(target=work) for _ in range(4)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert peak[0] == 1


def test_finished_job_records_expire_but_running_jobs_and_files_stay(monkeypatch, tmp_path):
    monkeypatch.setattr(ie, "_JOB_RETENTION_SEC", 60)
    monkeypatch.setattr(ie, "_OUTPUT_DIR", tmp_path)
    monkeypatch.setattr(ie, "_DELETE_EXPIRED_FILES", False)
    eng = _engine()
    old = time.time() - 3600
    eng.jobs = {"done": {"status": "completed", "updated_at": old},
                "busy": {"status": "processing", "updated_at": old, "started_at": time.time()},
                "new": {"status": "failed", "updated_at": time.time()}}
    (tmp_path / "done.obj").write_text("v 0 0 0\n")
    assert eng._evict_expired(time.time()) == ["done"]
    assert set(eng.jobs) == {"busy", "new"} and (tmp_path / "done.obj").exists()
    eng.jobs["done2"] = {"status": "completed", "updated_at": old}
    (tmp_path / "done2.obj").write_text("v 0 0 0\n")
    monkeypatch.setattr(ie, "_DELETE_EXPIRED_FILES", True)
    eng._evict_expired(time.time())
    assert not (tmp_path / "done2.obj").exists()


def test_upload_route_rejects_declared_oversize_before_reading(monkeypatch):
    from fastapi import FastAPI
    from fastapi.testclient import TestClient
    from backend.routes import inference as route
    app = FastAPI()
    app.include_router(route.router, prefix="/api/inference")
    client = TestClient(app)
    monkeypatch.setattr(route, "MAX_UPLOAD_BYTES", 1000)
    big = b"\xff\xd8\xff" + b"0" * 200_000
    r = client.post("/api/inference/submit", files={"image": ("big.jpg", big, "image/jpeg")})
    assert r.status_code == 413
    # An understated header cannot bypass the limit: at most limit+1 bytes are read and then refused.
    monkeypatch.setattr(route, "_MULTIPART_OVERHEAD", 10_000_000)
    r = client.post("/api/inference/submit", files={"image": ("big.jpg", big, "image/jpeg")})
    assert r.status_code == 413 and "limit" in r.json()["detail"]
