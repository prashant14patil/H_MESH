"""Verify the video mesh API using explicitly authorized existing video frames."""
import argparse
import time

import requests

from video_to_mesh import validate_obj


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--backend", required=True)
    parser.add_argument("--video-id", required=True)
    parser.add_argument("--frames", nargs="+", required=True)
    parser.add_argument("--timeout", type=float, default=300)
    args = parser.parse_args()
    base = args.backend.rstrip("/")
    health = requests.get(base + "/health", timeout=10)
    health.raise_for_status()
    if not health.json().get("model_ready"):
        raise RuntimeError("Inference model is not ready")
    response = requests.post(base + "/api/video-mesh/runs", json={
        "video_id": args.video_id, "frames": args.frames,
    }, timeout=30)
    response.raise_for_status()
    run = response.json()
    print("Run:", run["run_id"], flush=True)
    deadline = time.monotonic() + args.timeout
    while time.monotonic() < deadline:
        response = requests.get(base + "/api/video-mesh/runs/" + run["run_id"], timeout=15)
        response.raise_for_status()
        run = response.json()
        if run["status"] != "processing":
            break
        time.sleep(1)
    else:
        raise TimeoutError("Batch did not finish")
    print("Status:", run["status"], flush=True)
    for frame in run["frames"]:
        print(frame["filename"], frame["status"], frame.get("detail", ""), flush=True)
        if frame["status"] == "completed":
            response = requests.get(base + frame["mesh_url"], timeout=30)
            response.raise_for_status()
            vertices, faces = validate_obj(response.content)
            print("  validated OBJ:", len(vertices), "vertices,", len(faces), "faces", flush=True)
    if run["status"] != "completed" or any(frame["status"] != "completed" for frame in run["frames"]):
        raise RuntimeError("Batch did not fully succeed")


if __name__ == "__main__":
    main()
