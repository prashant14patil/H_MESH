"""
scripts/test_integration.py
----------------------------
End-to-end integration test for the Tanitas Human Digital Twin — LOCAL
GPU POC.

Tests the complete chain:
    React/CLI client  <-->  Local FastAPI backend  <-->  HMR2 model (local GPU)

Unlike the original POC, there is no Google Colab dependency here — the
FastAPI backend performs inference in-process.

Prerequisites:
    1. The HMR2 model has been downloaded (python scripts/download_models.py)
       and HMR2_CHECKPOINT is set in .env.
    2. FastAPI backend is running:
           uvicorn backend.main:app --reload --port 8000
    3. Wait a few seconds after starting the backend for the model to
       finish loading (check with `curl http://localhost:8000/health`).

Usage:
    python scripts/test_integration.py
    python scripts/test_integration.py --image path/to/photo.jpg
    python scripts/test_integration.py --backend http://localhost:8000

Exit codes:  0 = all tests passed,  1 = one or more tests failed.
"""
import argparse, os, sys, time
import requests
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
DEFAULT_BACKEND = os.getenv("BACKEND_URL", "http://localhost:8000").rstrip("/")
POLL_INTERVAL   = int(os.getenv("POLL_INTERVAL_SEC", "3"))
MAX_POLLS       = int(os.getenv("MAX_POLL_ATTEMPTS", "60"))

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _ok(label):
    print(f"  PASS  {label}")

def _fail(label, detail=""):
    print(f"  FAIL  {label}" + (f": {detail}" if detail else ""))
    return False

def _step(n, title):
    print(f"\n[{n}] {title}")

# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------
def test_backend_health(base: str) -> bool:
    _step(1, "Backend health check")
    try:
        r = requests.get(f"{base}/health", timeout=5)
        if r.status_code == 200:
            data = r.json()
            _ok(f"GET /health -> {data}")
            if data.get("model_ready") is False:
                print("       NOTE: model_ready=false — the HMR2 model is still "
                      "loading, or failed to load (see model_load_error).")
            return True
        return _fail("GET /health", f"HTTP {r.status_code}")
    except requests.RequestException as e:
        return _fail("GET /health", str(e))


def test_submit(base: str, image_path: str) -> str | None:
    _step(2, f"Submit image: {image_path}")
    if not Path(image_path).exists():
        _fail("Image file not found", image_path)
        return None
    try:
        with open(image_path, "rb") as f:
            r = requests.post(
                f"{base}/api/inference/submit",
                files={"image": (Path(image_path).name, f)},
                timeout=30,
            )
        if r.status_code in (200, 202):
            data = r.json()
            job_id = data.get("job_id")
            _ok(f"POST /api/inference/submit -> job_id={job_id}, status={data.get('status')}")
            return job_id
        return _fail("POST /api/inference/submit", f"HTTP {r.status_code}: {r.text[:200]}")
    except requests.RequestException as e:
        _fail("POST /api/inference/submit", str(e))
        return None


def test_poll(base: str, job_id: str) -> dict | None:
    _step(3, f"Poll status for job {job_id}")
    for attempt in range(1, MAX_POLLS + 1):
        try:
            r = requests.get(f"{base}/api/inference/status/{job_id}", timeout=15)
            data = r.json()
            current = data.get("status", "unknown")
            print(f"       attempt {attempt:2d}/{MAX_POLLS}: {current}")
            if current == "completed":
                _ok(f"Job completed  mesh_url={data.get('mesh_url')}")
                return data
            if current in ("failed", "error"):
                _fail("Job failed", data.get("detail", "unknown error"))
                return None
        except requests.RequestException as e:
            print(f"       attempt {attempt:2d}: request error ({e})")
        time.sleep(POLL_INTERVAL)
    _fail("Polling timeout", f"Job did not complete in {MAX_POLLS * POLL_INTERVAL}s")
    return None


def test_download_mesh(base: str, status_data: dict) -> bool:
    _step(4, "Download and validate OBJ mesh")
    mesh_url = status_data.get("mesh_url", "")
    if not mesh_url:
        return _fail("No mesh_url in status response")

    url = f"{base}{mesh_url}" if mesh_url.startswith("/") else mesh_url
    try:
        r = requests.get(url, timeout=60)
        if r.status_code != 200:
            return _fail("GET mesh", f"HTTP {r.status_code}")
        obj_bytes = r.content
        lines     = obj_bytes.decode("utf-8", errors="replace").splitlines()
        verts     = sum(1 for l in lines if l.startswith("v "))
        faces     = sum(1 for l in lines if l.startswith("f "))
        _ok(f"Downloaded {len(obj_bytes):,} bytes | {verts:,} vertices | {faces:,} faces")
        if verts < 100 or faces < 100:
            return _fail("OBJ validation", f"Too few verts/faces ({verts}/{faces})")
        # Save locally for manual inspection
        out = Path("outputs/meshes/integration_test.obj")
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_bytes(obj_bytes)
        _ok(f"Saved to {out}")
        return True
    except requests.RequestException as e:
        return _fail("GET mesh", str(e))


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description="End-to-end integration test")
    parser.add_argument("--backend", default=DEFAULT_BACKEND,
                        help="FastAPI base URL (default: BACKEND_URL env or http://localhost:8000)")
    parser.add_argument("--image", default="",
                        help="Path to a JPEG/PNG test image (default: auto-detect)")
    args = parser.parse_args()

    # Find a test image if not specified
    image_path = args.image
    if not image_path:
        candidates = list(Path(".").glob("**/*.jpg")) + list(Path(".").glob("**/*.png"))
        candidates = [p for p in candidates if "outputs" not in str(p) and ".git" not in str(p)]
        if candidates:
            image_path = str(candidates[0])
            print(f"Auto-selected test image: {image_path}")
        else:
            print("ERROR: No test image found. Use --image path/to/photo.jpg")
            sys.exit(1)

    print(f"\nTanitas Human Digital Twin -- Local GPU Integration Test")
    print(f"Backend : {args.backend}")
    print(f"Image   : {image_path}")
    print("=" * 50)

    results = []

    ok1 = test_backend_health(args.backend)
    results.append(ok1)
    if not ok1:
        print("\nBackend is offline. Start it with:")
        print("  uvicorn backend.main:app --reload --port 8000")
        sys.exit(1)

    job_id = test_submit(args.backend, image_path)
    results.append(job_id is not None)
    if not job_id:
        sys.exit(1)

    status_data = test_poll(args.backend, job_id)
    results.append(status_data is not None)
    if not status_data:
        sys.exit(1)

    ok4 = test_download_mesh(args.backend, status_data)
    results.append(ok4)

    print("\n" + "=" * 50)
    passed = sum(1 for r in results if r)
    total  = len(results)
    if passed == total:
        print(f"ALL {total} TESTS PASSED")
        sys.exit(0)
    else:
        print(f"{total - passed}/{total} TESTS FAILED")
        sys.exit(1)


if __name__ == "__main__":
    main()
