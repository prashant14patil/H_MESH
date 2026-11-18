"""Open an existing experimental mesh run in the real frontend, without inference."""
import tempfile
from pathlib import Path

from playwright.sync_api import sync_playwright

VIDEO = "5528415e-6f8b-4335-a543-1edfa5ff2eb0"
REPORT = "676c50af-07c0-4f52-bb4f-7d49bd231754"
RUN = "d9ae1553-d1f4-4ccc-9ee9-b9702b692dc3"


def main():
    errors = []
    assets = []
    with sync_playwright() as playwright:
        browser = playwright.chromium.launch(
            executable_path=r"C:\Program Files\Google\Chrome\Application\chrome.exe",
            headless=True,
            args=["--enable-unsafe-swiftshader"],
        )
        try:
            page = browser.new_page(viewport={"width": 1440, "height": 900})
            page.on("pageerror", lambda error: errors.append(str(error)))
            page.on("response", lambda response: assets.append((response.url, response.status))
                    if f"/runs/{RUN}/mesh/" in response.url else None)
            page.goto("http://127.0.0.1:5174/", wait_until="domcontentloaded")
            page.get_by_role("button", name="360° Video Capture").click()
            page.get_by_label("Recording ID", exact=True).fill(VIDEO)
            page.get_by_label("Quality report ID", exact=True).fill(REPORT)
            page.get_by_role("button", name="Open using existing authorization").click()
            page.get_by_role("heading", name="Experimental Video-to-Mesh", exact=False).wait_for(timeout=30000)
            page.get_by_label("Open an existing mesh run by ID").fill(RUN)
            page.get_by_role("button", name="Open run", exact=True).click()
            page.get_by_text("14 successful · 0 failed · 14 total", exact=False).wait_for(timeout=15000)
            canvas = page.locator("canvas").last
            canvas.wait_for(timeout=30000)
            page.wait_for_timeout(5000)
            canvas.scroll_into_view_if_needed()
            page.wait_for_timeout(1000)
            path = Path(tempfile.gettempdir()) / "experimental-video-mesh-viewer.png"
            page.screenshot(path=str(path))
            assert assets and assets[0][1] == 200, assets
            assert canvas.is_visible(), "Viewer canvas is not visible"
            assert page.get_by_text("Mesh viewer:", exact=False).count() == 0
            assert not errors, errors
            print("Opened frame_001.jpg OBJ in existing MeshViewer; canvas visible.")
            print("Mesh response:", assets[0])
            print("Screenshot:", path)
        finally:
            browser.close()


if __name__ == "__main__":
    main()
