"""Live Stage 1.1 blocked-quality path; no mocked APIs."""
from pathlib import Path

from playwright.sync_api import sync_playwright

VIDEO = Path("data/production_validation/ecff78fe-773e-4eb8-a785-c356e3be4d19/original.webm").resolve()


def main():
    errors = []
    with sync_playwright() as playwright:
        browser = playwright.chromium.launch(
            executable_path=r"C:\Program Files\Google\Chrome\Application\chrome.exe",
            headless=True, args=["--enable-unsafe-swiftshader"])
        page = browser.new_page(viewport={"width": 1440, "height": 900})
        page.on("pageerror", lambda error: errors.append(str(error)))
        try:
            page.goto("http://127.0.0.1:5190/", wait_until="domcontentloaded")
            page.get_by_role("button", name="360° Video Capture").click()
            page.locator('input[type="file"]').set_input_files(str(VIDEO))
            page.get_by_role("button", name="Process Video").click()
            page.get_by_text("Processing completed.").wait_for(timeout=180000)
            quality = page.locator("section").filter(has_text="Capture quality diagnostics")
            with page.expect_response(lambda r: "/api/capture-quality/runs" in r.url and r.request.method == "POST") as start:
                quality.get_by_role("button", name="Analyze capture quality").click()
            assert start.value.status == 202, (start.value.status, start.value.text())
            quality.get_by_text("Viewpoint coverage:", exact=False).wait_for(timeout=180000)
            panel = page.locator("section").filter(has_text="Subject-targeted manual review")
            panel.wait_for()
            image = panel.locator("img")
            image.wait_for()
            image.scroll_into_view_if_needed()
            rect = image.bounding_box()
            assert rect
            def draw():
                page.mouse.move(rect["x"] + rect["width"] * .30, rect["y"] + rect["height"] * .01)
                page.mouse.down()
                page.mouse.move(rect["x"] + rect["width"] * .62, rect["y"] + rect["height"] * .96, steps=8)
                page.mouse.up()
            draw()
            with page.expect_response(lambda r: "/api/subject-review/sessions" in r.url and r.request.method == "POST") as session:
                panel.get_by_role("button", name="Select reference subject").click()
            assert session.value.status == 201, (session.value.status, session.value.text())
            panel.get_by_text("Suggestion:", exact=False).wait_for(timeout=15000)
            draw()
            with page.expect_response(lambda r: "/decisions" in r.url and r.request.method == "POST") as review:
                panel.get_by_role("button", name="Confirm subject").click()
            print("review POST:", review.value.status, review.value.text(), flush=True)
            assert review.value.status == 400
            assert "mandatory quality" in review.value.text().lower()
            mesh_panel = page.locator("section").filter(has_text="Step 3 — 3D Mesh Reconstruction")
            assert mesh_panel.get_by_role("button", name="Generate 3D Meshes").is_disabled()
            assert not errors, errors
            print("PASS: live upload, extraction, pose, report, reference box and quality-blocked review", flush=True)
        finally:
            browser.close()


if __name__ == "__main__":
    main()
