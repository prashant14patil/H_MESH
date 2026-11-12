"""Live browser-to-backend video mesh smoke test. No mocked requests."""
import json
import time
from pathlib import Path
from urllib.parse import urlparse

import requests
from playwright.sync_api import sync_playwright

from video_to_mesh import validate_obj


def main():
    video = Path("data/production_validation/ecff78fe-773e-4eb8-a785-c356e3be4d19/original.webm").resolve()
    errors = []
    events = []
    with sync_playwright() as playwright:
        browser = playwright.chromium.launch(
            executable_path=r"C:\Program Files\Google\Chrome\Application\chrome.exe",
            headless=True, args=["--enable-unsafe-swiftshader"])
        page = browser.new_page(viewport={"width": 1440, "height": 900})
        page.on("pageerror", lambda error: errors.append(str(error)))
        page.on("response", lambda response: events.append(
            (response.request.method, urlparse(response.url).path, response.status))
            if "/api/" in response.url else None)
        try:
            page.goto("http://127.0.0.1:5173/", wait_until="domcontentloaded")
            page.get_by_role("button", name="360° Video Capture").click()
            page.locator('input[type="file"]').set_input_files(str(video))
            page.get_by_role("button", name="Process Video").click()
            page.get_by_text("Processing completed.").wait_for(timeout=180000)
            panel = page.locator("section").filter(has_text="Step 3 — 3D Mesh Reconstruction")
            panel.wait_for(timeout=15000)
            boxes = panel.locator('input[type="checkbox"]')
            if boxes.count() < 2:
                raise AssertionError("Fewer than two extracted frames")
            for i in range(boxes.count()):
                if boxes.nth(i).is_checked():
                    boxes.nth(i).uncheck()
            boxes.nth(0).check()
            boxes.nth(1).check()
            with page.expect_response(
                lambda r: r.request.method == "POST" and "/api/video-mesh/runs" in r.url,
                timeout=30000,
            ) as response_info:
                panel.get_by_role("button", name="Generate 3D Meshes").click()
            response = response_info.value
            print("POST", response.url, response.status, response.text(), flush=True)
            if response.status != 202:
                raise AssertionError("Batch POST failed")
            payload = response.request.post_data_json
            run = response.json()
            run_id = run["run_id"]
            print("payload", json.dumps(payload), "run_id", run_id, flush=True)
            if run["video_id"] != payload["video_id"] or len(payload["frames"]) != 2:
                raise AssertionError("Request and run do not match")
            base = "http://127.0.0.1:5173"
            deadline = time.monotonic() + 300
            while time.monotonic() < deadline:
                status = requests.get(base + "/api/video-mesh/runs/" + run_id, timeout=15)
                status.raise_for_status()
                run = status.json()
                if run["status"] != "processing":
                    break
                time.sleep(1)
            if run["status"] != "completed":
                raise AssertionError(f"Batch did not complete: {run}")
            panel.get_by_text("2/2 processed", exact=False).wait_for(timeout=30000)
            panel.locator("canvas").wait_for(timeout=30000)
            page.wait_for_timeout(3000)
            mesh_events = [event for event in events if "/api/video-mesh/runs/" in event[1] and "/mesh/" in event[1]]
            for frame in run["frames"]:
                mesh = requests.get(base + frame["mesh_url"], timeout=30)
                mesh.raise_for_status()
                vertices, faces = validate_obj(mesh.content)
                print("OBJ", frame["filename"], len(vertices), len(faces), flush=True)
            panel.locator("select").select_option(run["frames"][1]["filename"])
            page.wait_for_timeout(3000)
            mesh_events = [event for event in events if "/api/video-mesh/runs/" in event[1] and "/mesh/" in event[1]]
            if len([event for event in mesh_events if event[2] == 200]) < 2:
                raise AssertionError(f"Viewer did not fetch both meshes: {mesh_events}")
            print("mesh events", mesh_events, "page errors", errors, flush=True)
            if errors:
                raise AssertionError(errors)
            print("PASS: live upload, extraction, batch polling and two OBJ fetches", flush=True)
        finally:
            browser.close()


if __name__ == "__main__":
    main()
