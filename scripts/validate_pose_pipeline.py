"""Exercise the capture and pose HTTP APIs with a supplied video.

Example:
    python scripts/validate_pose_pipeline.py data/captures/ID/original.webm --base http://127.0.0.1:18763

The video and derived images are uploaded to and retained by the server.
Viewpoint labels are heuristic and need manual comparison with source frames.
"""
import argparse
import json
import mimetypes
import time
from pathlib import Path

import requests


def request(method, url, **kwargs):
    response = requests.request(method, url, timeout=30, **kwargs)
    response.raise_for_status()
    return response


def wait_for_job(base, stage, video_id, timeout):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        status = request("GET", f"{base}/api/{stage}/status/{video_id}").json()
        if status["status"] == "completed":
            return status
        if status["status"] == "failed":
            raise RuntimeError(f"{stage} failed: {status}")
        time.sleep(1)
    raise TimeoutError(f"{stage} did not finish within {timeout} seconds")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("video", type=Path)
    parser.add_argument("--base", default="http://127.0.0.1:8000")
    parser.add_argument("--timeout", type=int, default=300)
    args = parser.parse_args()
    base = args.base.rstrip("/")
    mime = mimetypes.guess_type(args.video.name)[0] or "application/octet-stream"
    with args.video.open("rb") as video:
        upload = request(
            "POST", f"{base}/api/capture/upload",
            files={"video": (args.video.name, video, mime)},
        ).json()
    video_id = upload["video_id"]
    print("Upload:", json.dumps(upload), flush=True)

    print("Extraction start:", request("POST", f"{base}/api/capture/process/{video_id}").json(), flush=True)
    print("Extraction status:", wait_for_job(base, "capture", video_id, args.timeout), flush=True)
    frames = request("GET", f"{base}/api/capture/frames/{video_id}").json()
    print("Video info:", json.dumps(frames.get("video_info")), flush=True)
    print("Frame counts:", json.dumps({
        key: frames.get(key) for key in ("candidate_frame_count", "accepted_frame_count", "rejected_frame_count")
    }), flush=True)
    if not frames.get("frames"):
        raise RuntimeError("No extracted frames are available for pose processing")

    print("Pose start:", request("POST", f"{base}/api/pose/process/{video_id}").json(), flush=True)
    print("Pose status:", wait_for_job(base, "pose", video_id, args.timeout), flush=True)
    pose = request("GET", f"{base}/api/pose/results/{video_id}").json()
    coverage = request("GET", f"{base}/api/pose/viewpoints/{video_id}").json()
    print("Pose summary:", json.dumps({
        key: pose.get(key) for key in (
            "total_frames", "frames_with_person_detected", "frames_full_body_visible",
            "frames_requiring_review", "processing_time_sec",
        )
    }), flush=True)
    print("Landmark counts:", json.dumps({f["frame"]: len(f.get("landmarks", {})) for f in pose["frames"]}), flush=True)
    print("Viewpoints:", json.dumps(coverage.get("frames")), flush=True)
    print("Coverage:", json.dumps(coverage.get("coverage")), flush=True)
    sample = pose["frames"][0]["frame"]
    for asset in ("pose_visualization.jpg", "segmentation_mask.png", "metadata.json"):
        response = request("GET", f"{base}/api/pose/frame/{video_id}/{sample}/{asset}")
        print(f"Asset {sample}/{asset}: HTTP {response.status_code}, {len(response.content)} bytes", flush=True)
    if pose.get("frames_with_person_detected", 0) == 0:
        raise RuntimeError("MediaPipe did not detect a person in any extracted frame")
    print("API workflow completed. Validate viewpoint labels against the source video manually.", flush=True)


if __name__ == "__main__":
    main()
