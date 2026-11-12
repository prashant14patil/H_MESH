"""Exercise the actual frontend with local video and photo fixtures.

Requires Playwright in an isolated test environment and an installed Chrome.
Run against a main backend and frontend configured to use it. Uploaded images
and videos are retained by the backend in its configured output directories.
"""
import argparse
from pathlib import Path
from urllib.parse import urlparse

from playwright.sync_api import sync_playwright


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--url", default="http://127.0.0.1:3000/")
    parser.add_argument("--video", required=True)
    parser.add_argument("--image", required=True)
    parser.add_argument("--chrome", default=r"C:\Program Files\Google\Chrome\Application\chrome.exe")
    args = parser.parse_args()
    events = []
    errors = []
    with sync_playwright() as playwright:
        browser = playwright.chromium.launch(executable_path=args.chrome, headless=True,
                                             args=["--enable-unsafe-swiftshader"])
        page = browser.new_page(viewport={"width": 1440, "height": 900})
        page.on("pageerror", lambda error: errors.append(str(error)))
        page.on("response", lambda response: events.append(
            (response.request.method, urlparse(response.url).path, response.status))
            if "/api/" in response.url else None)
        try:
            page.goto(args.url, wait_until="domcontentloaded")
            page.get_by_text("Backend online").wait_for(timeout=30000)

            page.get_by_role("button", name="360° Video Capture").click()
            page.locator('input[type="file"]').set_input_files(str(Path(args.video).resolve()))
            page.get_by_role("button", name="Process Video").click()
            page.get_by_text("Processing completed.").wait_for(timeout=180000)
            page.get_by_text("Insufficient viewpoint coverage.").wait_for(timeout=10000)
            frames = page.locator('img[src*="/api/capture/frame/"]')
            print("Gallery thumbnails:", frames.count(), flush=True)
            if frames.count() < 2:
                raise AssertionError("Expected multiple gallery thumbnails")
            frames.first.click()
            pose_image = page.locator('img[alt="pose visualization"]')
            pose_image.wait_for(timeout=15000)
            pose_image.evaluate("(img) => img.decode()")
            print("Pose visualization decoded:", pose_image.get_attribute("src"), flush=True)
            page.get_by_text("Landmark", exact=True).wait_for(timeout=10000)

            required = ("/api/capture/upload", "/api/capture/process/", "/api/capture/status/",
                        "/api/capture/frames/", "/api/pose/process/", "/api/pose/status/",
                        "/api/pose/results/", "/api/pose/viewpoints/", "/api/pose/frame/")
            for path in required:
                matches = [event for event in events if path in event[1]]
                if not matches or not any(200 <= event[2] < 300 for event in matches):
                    raise AssertionError(f"Missing successful request {path}: {matches}")
                print(path, matches[-1], flush=True)

            page.get_by_role("button", name="Single Photo").click()
            page.locator('input[type="file"]').set_input_files(str(Path(args.image).resolve()))
            page.get_by_role("button", name="Generate 3D Mesh").click()
            page.get_by_text("Step 3 — Mesh results").wait_for(timeout=180000)
            page.wait_for_timeout(5000)
            mesh = [event for event in events if "/api/inference/mesh/" in event[1]]
            print("Mesh response:", mesh, flush=True)
            if not any(event[2] == 200 for event in mesh):
                raise AssertionError("Browser did not fetch mesh successfully")
            if errors:
                raise AssertionError(f"Browser page errors: {errors}")
            print("PASS: main UI video and photo workflows completed", flush=True)
        finally:
            browser.close()


if __name__ == "__main__":
    main()
