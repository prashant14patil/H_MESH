"""Browser + real review routes on temporary synthetic data; never starts inference."""
import hashlib
import json
import socket
import sys
import tempfile
import threading
import time
import uuid
from dataclasses import asdict
from pathlib import Path

from PIL import Image
from playwright.sync_api import sync_playwright
import uvicorn
from fastapi import FastAPI

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from backend.routes.capture import router as capture_router
from backend.routes.capture_quality import router as quality_router
from backend.routes.subject_review import router as review_router
from backend.services import capture_quality_job, subject_review, video_processor
from backend.services.capture_quality import QualityConfig


def main():
    with tempfile.TemporaryDirectory() as temp:
        root = Path(temp)
        video_id, run_id = str(uuid.uuid4()), str(uuid.uuid4())
        video_processor._OUTPUT_ROOT = root / "captures"
        capture_quality_job.ROOT = root / "quality"
        subject_review.ROOT = root / "reviews"
        video_dir = video_processor._OUTPUT_ROOT / video_id
        frame_dir = video_dir / "extracted_frames"
        frame_dir.mkdir(parents=True)
        (video_dir / "original.mp4").write_bytes(b"synthetic video, not a real recording")
        frames, quality_frames = [], []
        for i in range(1, 14):
            name = f"frame_{i:03d}.jpg"
            path = frame_dir / name
            Image.new("RGB", (160, 240), (100, 120, 140)).save(path)
            digest = hashlib.sha256(path.read_bytes()).hexdigest()
            frames.append({"filename": name, "timestamp_sec": float(i), "is_blurry": False, "blur_score": 150})
            quality_frames.append({
                "filename": name, "source_sha256": digest, "status": "review", "warnings": ["subject_association_unknown"],
                "rejections": [], "person_detected": True, "person_count_observed": 0, "detector_evidence": [],
                "landmark_bbox_normalized": [.2, .2, .8, .8], "viewpoint": {"label": "unknown"},
                "extractor_quality": {"is_blurry": False},
                "checks": {key: "pass" for key in ("readability", "person_presence", "sharpness", "lighting", "full_body")},
            })
        metadata = {"video_id": video_id, "status": "completed", "frames": frames, "rejected_frames": [], "video_info": {}}
        (video_dir / "metadata.json").write_text(json.dumps(metadata), encoding="utf-8")
        report = {"run_id": run_id, "video_id": video_id, "frames": quality_frames, "thresholds": asdict(QualityConfig()),
                  "summary": {"extracted_frames": 13, "extractor_non_blurry": 13, "extractor_blurry": 0,
                              "rejected_duplicates": 0, "full_body_visible": 13, "possible_multiple_people": 0,
                              "recommended_frames": []},
                  "viewpoint_coverage": {"status": "unconfirmed", "note": "Synthetic fixture", "counts": {}},
                  "rejected_candidates": []}
        report_dir = capture_quality_job.ROOT / run_id
        report_dir.mkdir(parents=True)
        (report_dir / "report.json").write_text(json.dumps(report), encoding="utf-8")
        app = FastAPI()
        app.include_router(capture_router, prefix="/api/capture")
        app.include_router(quality_router, prefix="/api/capture-quality")
        app.include_router(review_router, prefix="/api/subject-review")
        sock = socket.socket()
        sock.bind(("127.0.0.1", 0))
        sock.listen(128)
        port = sock.getsockname()[1]
        server = uvicorn.Server(uvicorn.Config(app, log_level="error", lifespan="off"))
        thread = threading.Thread(target=server.run, kwargs={"sockets": [sock]}, daemon=True)
        thread.start()
        try:
            deadline = time.monotonic() + 10
            while not server.started and time.monotonic() < deadline:
                time.sleep(.02)
            assert server.started
            with sync_playwright() as playwright:
                browser = playwright.chromium.launch(executable_path=r"C:\Program Files\Google\Chrome\Application\chrome.exe", headless=True)
                page = browser.new_page(viewport={"width": 1440, "height": 900})
                errors = []
                page.on("pageerror", lambda error: errors.append(str(error)))

                def forward(route):
                    url = route.request.url
                    path = url[url.index("/api/"):]
                    if path.startswith("/api/video-mesh/"):
                        route.abort()
                    else:
                        response = route.fetch(url=f"http://127.0.0.1:{port}{path}")
                        route.fulfill(response=response)

                page.route("**/api/**", forward)
                try:
                    page.goto("http://127.0.0.1:5191/", wait_until="domcontentloaded")
                    page.get_by_role("button", name="360° Video Capture").click()
                    page.get_by_label("Recording ID", exact=True).fill(video_id)
                    page.get_by_label("Quality report ID", exact=True).fill(run_id)
                    form = page.locator("form").filter(has=page.get_by_role("heading", name="Reopen saved capture"))
                    assert form.get_by_role("button", name="Open saved capture").is_disabled()
                    form.get_by_label("I am authorized to process this recording and generate meshes.").check()
                    form.get_by_label("Everyone depicted has consented to this processing and mesh generation.").check()
                    form.get_by_role("button", name="Open saved capture").click()
                    panel = page.locator("#subject-review")
                    panel.wait_for(timeout=15000)
                    cards = page.locator("section").filter(has_text="Step 3 — 3D Mesh Reconstruction").locator("article")
                    assert cards.count() == 13
                    assert cards.nth(0).locator('input[type="checkbox"]').first.is_disabled()
                    image = panel.locator("img")
                    image.scroll_into_view_if_needed()
                    rect = image.bounding_box()
                    page.mouse.move(rect["x"] + rect["width"] * .1, rect["y"] + rect["height"] * .1)
                    page.mouse.down()
                    page.mouse.move(rect["x"] + rect["width"] * .9, rect["y"] + rect["height"] * .9, steps=5)
                    page.mouse.up()
                    panel.get_by_role("button", name="Select reference subject").click()
                    panel.get_by_text("Reference subject selected", exact=False).wait_for()
                    assert cards.nth(0).locator('input[type="checkbox"]').first.is_disabled()
                    sessions = list(subject_review.ROOT.glob("*/session.json"))
                    assert len(sessions) == 1
                    session = json.loads(sessions[0].read_text(encoding="utf-8"))
                    assert session["video_id"] == video_id
                    for i, frame in enumerate(frames):
                        panel.locator("select").select_option(frame["filename"])
                        image.scroll_into_view_if_needed()
                        rect = image.bounding_box()
                        page.mouse.move(rect["x"] + rect["width"] * .1, rect["y"] + rect["height"] * .1)
                        page.mouse.down()
                        page.mouse.move(rect["x"] + rect["width"] * .9, rect["y"] + rect["height"] * .9, steps=3)
                        page.mouse.up()
                        panel.get_by_role("button", name="Confirm subject").click()
                        panel.locator("p").filter(has_text="Subject association: PASS").wait_for()
                        card = cards.nth(i)
                        checkbox = card.locator('input[type="checkbox"]').first
                        if i < 12:
                            assert checkbox.is_enabled()
                            checkbox.check()
                        else:
                            assert checkbox.is_disabled()
                    # The thirteenth card must be disabled at the limit.
                    assert cards.nth(12).locator('input[type="checkbox"]').first.is_disabled()
                    assert cards.nth(12).get_by_text("12-frame selection limit", exact=False).count() == 1
                    panel.locator("select").select_option(frames[0]["filename"])
                    assert cards.nth(0).locator('input[type="checkbox"]').first.is_checked()
                    decision_path = subject_review.ROOT / session["session_id"] / "decisions" / (frames[0]["filename"] + ".json")
                    decision = json.loads(decision_path.read_text(encoding="utf-8"))
                    assert decision["recording_id"] == video_id and decision["frame_sha256"] == quality_frames[0]["source_sha256"]
                    assert not errors, errors
                    print("PASS: real review routes and persisted synthetic decisions; navigation and 12-frame limit; no inference")
                finally:
                    browser.close()
        finally:
            server.should_exit = True
            thread.join(timeout=10)
            sock.close()


if __name__ == "__main__":
    main()
