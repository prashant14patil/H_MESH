"""Validate a fresh HMR2 inference through the real backend API.

Run: python scripts/validate_hmr2_regression.py --backend http://127.0.0.1:18765 --image path/to/person.jpg

The backend may retain the submitted image and generated mesh in its configured
output directory. This script does not overwrite an existing mesh.
"""
import argparse
import math
import time
from pathlib import Path

import requests


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--backend", required=True)
    parser.add_argument("--image", type=Path, required=True)
    parser.add_argument("--timeout", type=float, default=300)
    args = parser.parse_args()
    base = args.backend.rstrip("/")

    health = requests.get(f"{base}/health", timeout=15)
    health.raise_for_status()
    print("Health:", health.json(), flush=True)
    if not health.json().get("model_ready"):
        raise RuntimeError("HMR2 model is not ready")

    with args.image.open("rb") as fh:
        response = requests.post(
            f"{base}/api/inference/submit",
            files={"image": (args.image.name, fh, "image/jpeg")},
            timeout=60,
        )
    response.raise_for_status()
    submission = response.json()
    job_id = submission["job_id"]
    print("Submit:", submission, flush=True)

    deadline = time.monotonic() + args.timeout
    while time.monotonic() < deadline:
        response = requests.get(f"{base}/api/inference/status/{job_id}", timeout=15)
        response.raise_for_status()
        status = response.json()
        if status.get("status") == "completed":
            break
        if status.get("status") in ("failed", "error"):
            raise RuntimeError(f"Inference failed: {status}")
        time.sleep(2)
    else:
        raise TimeoutError(f"Inference job {job_id} did not finish")

    print("Completed:", status, flush=True)
    mesh_url = status.get("mesh_url")
    if not mesh_url or not mesh_url.startswith("/api/inference/"):
        raise RuntimeError(f"Unexpected mesh URL: {mesh_url!r}")
    response = requests.get(base + mesh_url, timeout=60)
    response.raise_for_status()
    vertices = []
    faces = []
    for line in response.text.splitlines():
        fields = line.split()
        if not fields:
            continue
        if fields[0] == "v":
            coords = [float(x) for x in fields[1:4]]
            if len(coords) != 3 or not all(math.isfinite(x) for x in coords):
                raise ValueError("Invalid vertex coordinates")
            vertices.append(coords)
        elif fields[0] == "f":
            indices = [int(x.split("/")[0]) for x in fields[1:]]
            if len(indices) < 3:
                raise ValueError("Invalid face")
            faces.append(indices)
    if len(vertices) < 100 or len(faces) < 100:
        raise RuntimeError(f"Insufficient mesh geometry: {len(vertices)} vertices, {len(faces)} faces")
    if any(index == 0 or index > len(vertices) or index < -len(vertices)
           for face in faces for index in face):
        raise RuntimeError("Mesh contains out-of-range face indices")
    print(f"Mesh: HTTP {response.status_code}, {len(response.content)} bytes, "
          f"{len(vertices)} vertices, {len(faces)} faces; geometry validated", flush=True)


if __name__ == "__main__":
    main()
