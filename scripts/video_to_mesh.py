"""Generate independent meshes using the existing single-image inference API."""
import argparse
import json
import math
import re
import shutil
import time
import uuid
from pathlib import Path

import cv2
import numpy as np
import requests


def validate_obj(data):
    vertices, faces = [], []
    for line in data.decode("utf-8").splitlines():
        fields = line.split()
        if not fields:
            continue
        if fields[0] == "v":
            if len(fields) < 4:
                raise ValueError("Invalid vertex")
            vertex = tuple(float(x) for x in fields[1:4])
            if not all(math.isfinite(x) for x in vertex):
                raise ValueError("Non-finite vertex")
            vertices.append(vertex)
        elif fields[0] == "f":
            face = [int(x.split("/")[0]) for x in fields[1:]]
            if len(face) < 3:
                raise ValueError("Invalid face")
            faces.append(face)
    if len(vertices) < 100 or len(faces) < 100:
        raise ValueError("Insufficient mesh geometry")
    if any(i == 0 or i > len(vertices) or i < -len(vertices) for face in faces for i in face):
        raise ValueError("Face index out of range")
    return np.asarray(vertices), faces


def save_preview(frame_path, vertices, path, title):
    """A simple orthographic vertex preview, not a surface-quality assessment."""
    image = cv2.imread(str(frame_path))
    if image is None:
        raise ValueError("Source image unreadable")
    h, w = 600, 400
    scale = min(w / image.shape[1], h / image.shape[0])
    source = cv2.resize(image, (round(image.shape[1]*scale), round(image.shape[0]*scale)))
    panel = np.full((h, w, 3), 245, dtype=np.uint8)
    y, x = (h-source.shape[0])//2, (w-source.shape[1])//2
    panel[y:y+source.shape[0], x:x+source.shape[1]] = source
    mesh = np.full((h, w, 3), (35, 35, 35), dtype=np.uint8)
    # Match the existing viewer's X-axis 180-degree correction in a front projection.
    xy = vertices[:, :2].copy()
    span = np.maximum(np.ptp(xy, axis=0), 1e-9)
    factor = min((w-80)/span[0], (h-100)/span[1])
    center = (xy.max(axis=0)+xy.min(axis=0))/2
    xy = (xy-center)*factor + [w/2, h/2]
    for px, py in np.rint(xy).astype(int):
        if 0 <= px < w and 0 <= py < h:
            cv2.circle(mesh, (px, py), 1, (170, 210, 80), -1)
    cv2.putText(mesh, "OBJ vertices (not a surface render)", (10, 25), cv2.FONT_HERSHEY_SIMPLEX, .45, (255,255,255), 1)
    combined = np.hstack((panel, mesh))
    cv2.rectangle(combined, (0, h-32), (combined.shape[1]-1, h-1), (0, 0, 0), -1)
    cv2.putText(combined, title[:95], (8, h-10), cv2.FONT_HERSHEY_SIMPLEX, .42, (255,255,255), 1)
    if not cv2.imwrite(str(path), combined):
        raise OSError("Unable to write preview")


def run_batch(capture_dir, manifest, backend, output_root=Path("outputs/video_to_mesh"),
              run_id=None, timeout=300, session=None):
    capture_dir, output_root = Path(capture_dir), Path(output_root)
    selections = manifest["frames"]
    if not selections or len({s["filename"] for s in selections}) != len(selections):
        raise ValueError("Select unique frames")
    for entry in selections:
        name = entry["filename"]
        if Path(name).name != name or not re.fullmatch(r"frame_\d+\.(jpg|jpeg|png)", name, re.I):
            raise ValueError(f"Invalid extracted frame filename: {name}")
        if not entry.get("reason") or not re.fullmatch(r"[a-z0-9_]+", entry.get("label", "")):
            raise ValueError("Each frame needs a safe label and selection reason")
    identifier = run_id or uuid.uuid4().hex
    if not re.fullmatch(r"[a-zA-Z0-9_-]+", identifier):
        raise ValueError("Invalid run ID")
    output_root.mkdir(parents=True, exist_ok=True)
    root = output_root / identifier
    root.mkdir(exist_ok=False)
    for name in ("selected_frames", "meshes", "previews"):
        (root / name).mkdir()
    client = session or requests.Session()
    results = {"capture_dir": str(capture_dir), "extracted_frame_count": manifest.get("extracted_frame_count"),
               "selected_frame_count": len(selections), "frames": [], "status": "processing",
               "limitations": "Independent single-image predictions; not a unified mesh or metric measurement."}
    started = time.monotonic()

    def persist():
        results["elapsed_sec"] = round(time.monotonic()-started, 3)
        (root / "results.json").write_text(json.dumps(results, indent=2, allow_nan=False), encoding="utf-8")

    persist()
    for entry in selections:
        record = dict(entry, status="failed", processing_time_sec=None)
        results["frames"].append(record)
        frame_start = time.monotonic()
        try:
            source = capture_dir / "extracted_frames" / entry["filename"]
            if not source.is_file():
                raise ValueError("Source frame missing")
            image = cv2.imread(str(source))
            if image is None:
                raise ValueError("Source frame unreadable")
            copied = root / "selected_frames" / source.name
            shutil.copyfile(source, copied)
            mime = "image/png" if source.suffix.lower() == ".png" else "image/jpeg"
            with copied.open("rb") as image_file:
                response = client.post(backend.rstrip("/") + "/api/inference/submit",
                                       files={"image": (copied.name, image_file, mime)}, timeout=60)
            response.raise_for_status()
            job = response.json()["job_id"]
            record["job_id"] = job
            deadline = time.monotonic() + timeout
            while time.monotonic() < deadline:
                response = client.get(backend.rstrip("/") + "/api/inference/status/" + job, timeout=15)
                response.raise_for_status()
                status = response.json()
                if status.get("status") == "completed":
                    break
                if status.get("status") in ("failed", "error"):
                    raise RuntimeError(status.get("detail", "Inference failed"))
                time.sleep(1)
            else:
                raise TimeoutError(f"Job {job} timed out")
            mesh_url = status.get("mesh_url", "")
            if mesh_url != f"/api/inference/mesh/{job}":
                raise ValueError("Unexpected mesh URL")
            response = client.get(backend.rstrip("/") + mesh_url, timeout=60)
            response.raise_for_status()
            vertices, faces = validate_obj(response.content)
            stem = f"{source.stem}_{entry['label']}"
            mesh_path = root / "meshes" / f"{stem}.obj"
            mesh_path.write_bytes(response.content)
            record.update(status="completed", mesh_path=str(mesh_path), vertices=len(vertices), faces=len(faces))
            try:
                preview_path = root / "previews" / f"{stem}.jpg"
                save_preview(copied, vertices, preview_path, f"{source.stem} | {entry['label']} | confidence {entry.get('confidence')}")
                record["preview_path"] = str(preview_path)
            except (ValueError, OSError, cv2.error) as exc:
                record["preview_error"] = str(exc)
        except (OSError, ValueError, KeyError, requests.RequestException, RuntimeError) as exc:
            record["error"] = str(exc)
        finally:
            record["processing_time_sec"] = round(time.monotonic() - frame_start, 3)
            persist()
    results["status"] = "completed"
    results["successful_meshes"] = sum(f["status"] == "completed" for f in results["frames"])
    persist()
    lines = ["# Independent video-frame mesh batch", "",
             f"Extracted: {results['extracted_frame_count']}; selected: {len(selections)}; successful: {results['successful_meshes']}.",
             f"Elapsed: {results['elapsed_sec']} seconds.", "",
             "Each mesh is an independent HMR2 prediction; none are fused or metrically validated.", ""]
    for frame in results["frames"]:
        lines.append(f"- {frame['filename']} ({frame['label']}, confidence {frame.get('confidence')}): "
                     f"{frame['status']}; {frame.get('mesh_path', frame.get('error'))}. "
                     f"Reason: {frame['reason']}")
    (root / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return root


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--capture", type=Path, required=True)
    parser.add_argument("--selection", type=Path, required=True)
    parser.add_argument("--backend", default="http://127.0.0.1:8000")
    parser.add_argument("--output-root", type=Path, default=Path("outputs/video_to_mesh"))
    parser.add_argument("--run-id")
    parser.add_argument("--timeout", type=float, default=300)
    args = parser.parse_args()
    manifest = json.loads(args.selection.read_text(encoding="utf-8"))
    print(run_batch(args.capture, manifest, args.backend, args.output_root, args.run_id, args.timeout))


if __name__ == "__main__":
    main()
