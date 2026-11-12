"""Test independent capture and pose jobs through an existing main backend."""
import argparse
import concurrent.futures
import hashlib
import time
from pathlib import Path

import requests


def poll(base, stage, job_id, timeout):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        response = requests.get(f"{base}/api/{stage}/status/{job_id}", timeout=15)
        response.raise_for_status()
        data = response.json()
        if data["status"] == "completed":
            return data
        if data["status"] == "failed":
            raise RuntimeError(f"{stage} {job_id}: {data}")
        time.sleep(1)
    raise TimeoutError(f"{stage} {job_id}")


def process(base, video, timeout):
    with video.open("rb") as fh:
        response = requests.post(f"{base}/api/capture/upload",
                                 files={"video": (video.name, fh, "video/webm")}, timeout=60)
    response.raise_for_status()
    job_id = response.json()["video_id"]
    print(f"{job_id}: uploaded HTTP {response.status_code}", flush=True)
    response = requests.post(f"{base}/api/capture/process/{job_id}", timeout=15)
    response.raise_for_status()
    print(f"{job_id}: extraction start HTTP {response.status_code}", flush=True)
    print(f"{job_id}: extraction {poll(base, 'capture', job_id, timeout)}", flush=True)
    frames = requests.get(f"{base}/api/capture/frames/{job_id}", timeout=15)
    frames.raise_for_status()
    response = requests.post(f"{base}/api/pose/process/{job_id}", timeout=15)
    response.raise_for_status()
    print(f"{job_id}: pose start HTTP {response.status_code}", flush=True)
    print(f"{job_id}: pose {poll(base, 'pose', job_id, timeout)}", flush=True)
    results = requests.get(f"{base}/api/pose/results/{job_id}", timeout=15)
    results.raise_for_status()
    viewpoints = requests.get(f"{base}/api/pose/viewpoints/{job_id}", timeout=15)
    viewpoints.raise_for_status()
    metadata, pose, coverage = frames.json(), results.json(), viewpoints.json()
    if metadata["video_id"] != job_id or pose["video_id"] != job_id or coverage["video_id"] != job_id:
        raise AssertionError("Cross-job result mismatch")
    if not metadata["frames"] or not pose["frames"]:
        raise AssertionError("No frames or pose results")
    print(f"{job_id}: {len(metadata['frames'])} frames, "
          f"{pose['frames_with_person_detected']} persons, coverage={coverage['coverage']}", flush=True)
    return job_id, metadata, pose


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--base", required=True)
    parser.add_argument("--video", type=Path, required=True)
    parser.add_argument("--capture-root", type=Path, required=True)
    parser.add_argument("--timeout", type=int, default=300)
    args = parser.parse_args()
    base = args.base.rstrip("/")
    invalid = requests.post(f"{base}/api/capture/upload",
                            files={"video": ("invalid.webm", b"not a video", "video/webm")},
                            timeout=15)
    print(f"Invalid upload: HTTP {invalid.status_code}, {invalid.text}", flush=True)
    if invalid.status_code != 400:
        raise AssertionError("Invalid upload should be rejected")
    with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:
        futures = [pool.submit(process, base, args.video, args.timeout) for _ in range(2)]
        results = [future.result() for future in futures]
    ids = [result[0] for result in results]
    if len(set(ids)) != 2:
        raise AssertionError("Upload IDs are not unique")
    expected_hash = hashlib.sha256(args.video.read_bytes()).hexdigest()
    for job_id, metadata, pose in results:
        directory = args.capture_root / job_id
        source = directory / "original.webm"
        if not source.is_file() or hashlib.sha256(source.read_bytes()).hexdigest() != expected_hash:
            raise AssertionError(f"Incorrect stored video for {job_id}")
        if not (directory / "extracted_frames" / metadata["frames"][0]["filename"]).is_file():
            raise AssertionError(f"Missing extracted frame for {job_id}")
        if not (directory / "processed_frames" / "processing_summary.json").is_file():
            raise AssertionError(f"Missing pose summary for {job_id}")
        print(f"{job_id}: verified independent directory and source SHA-256", flush=True)
    print("PASS: two concurrent uploads processed independently", flush=True)


if __name__ == "__main__":
    main()
