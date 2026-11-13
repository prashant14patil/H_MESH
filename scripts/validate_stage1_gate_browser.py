"""Real browser capture/quality gate smoke test; no API mocking."""
import json
from pathlib import Path

import requests
from playwright.sync_api import sync_playwright

VIDEO = Path("data/production_validation/ecff78fe-773e-4eb8-a785-c356e3be4d19/original.webm").resolve()


def main():
    errors = []
    responses = []
    with sync_playwright() as playwright:
        browser = playwright.chromium.launch(
            executable_path=r"C:\Program Files\Google\Chrome\Application\chrome.exe",
            headless=True, args=["--enable-unsafe-swiftshader"])
        page = browser.new_page(viewport={"width": 1440, "height": 900})
        page.on("pageerror", lambda error: errors.append(str(error)))
        page.on("response", lambda response: responses.append((response.request.method, response.url, response.status))
                if "/api/" in response.url else None)
        try:
            page.goto("http://127.0.0.1:5189/", wait_until="domcontentloaded")
            page.get_by_role("button", name="360° Video Capture").click()
            page.locator('input[type="file"]').set_input_files(str(VIDEO))
            page.get_by_role("button", name="Process Video").click()
            page.get_by_text("Processing completed.").wait_for(timeout=180000)
            panel = page.locator("section").filter(has_text="Capture quality diagnostics")
            with page.expect_response(
                lambda r: r.request.method == "POST" and "/api/capture-quality/runs" in r.url,
                timeout=30000,
            ) as start_info:
                panel.get_by_role("button", name="Analyze capture quality").click()
            start = start_info.value
            if start.status != 202:
                raise AssertionError(f"Quality start failed: {start.status} {start.text()}")
            job = start.json()
            with page.expect_response(
                lambda r: r.request.method == "GET" and f"/api/capture-quality/runs/{job['run_id']}/report" in r.url,
                timeout=180000,
            ) as report_info:
                panel.get_by_text("Viewpoint coverage:", exact=False).wait_for(timeout=180000)
            response = report_info.value
            if response.status != 200:
                raise AssertionError(f"Report failed: {response.status} {response.text()}")
            report = response.json()
            mesh_panel = page.locator("section").filter(has_text="Step 3 — 3D Mesh Reconstruction")
            mesh_panel.get_by_text("Frames with missing or conflicting identity evidence", exact=False).wait_for()
            if mesh_panel.get_by_role("button", name="Generate 3D Meshes").is_enabled():
                raise AssertionError("Mesh submission unexpectedly enabled")
            frame = report["frames"][0]
            name = frame["filename"]
            panel.get_by_role("button", name=f"{name}: {frame['status']}").click()
            panel.get_by_text("Unknown identity cannot be overridden", exact=False).wait_for()
            payload = {"video_id": job["video_id"], "frames": [name],
                       "quality_run_id": job["run_id"], "reviewed_frames": [name]}
            blocked = requests.post("http://127.0.0.1:5189/api/video-mesh/runs", json=payload, timeout=30)
            print("capture_id:", job["video_id"], "quality_run:", job["run_id"], flush=True)
            print("summary:", json.dumps(report["summary"]), flush=True)
            print("gate:", blocked.status_code, blocked.text, flush=True)
            print("page_errors:", errors, flush=True)
            if blocked.status_code != 400 or "identity" not in blocked.text.lower():
                raise AssertionError("Direct request did not fail closed")
            if errors:
                raise AssertionError(errors)
            print("PASS: real upload, extraction, quality report and blocked ambiguous submission", flush=True)
        finally:
            browser.close()


if __name__ == "__main__":
    main()
