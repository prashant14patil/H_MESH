r"""Real end-to-end UI check of the unified reconstruction (real backend, real HMR2, real recording).

Usage (backend + Vite dev server already running):
    .venv\Scripts\python.exe scripts/validate_unified_reconstruction_browser.py \
        --url http://localhost:5174 --video <recording id> --report <quality report id> --out <dir>

Opens the saved recording with its EXISTING authorization (no new consent
attestation), starts the reconstruction, waits for completion and checks the
final OBJ was fetched and parsed by the 3D viewer. Saves screenshots.
"""
import argparse
import json
import time
from pathlib import Path

from playwright.sync_api import sync_playwright


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", default="http://localhost:5174")
    ap.add_argument("--video", required=True)
    ap.add_argument("--report", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--channel", default="msedge")
    a = ap.parse_args()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    console, obj_responses = [], []
    with sync_playwright() as p:
        browser = p.chromium.launch(channel=a.channel, headless=True, args=["--use-angle=swiftshader", "--enable-unsafe-swiftshader"])
        page = browser.new_page(viewport={"width": 1500, "height": 1300})
        page.on("console", lambda m: console.append(f"{m.type}: {m.text}"))
        page.on("response", lambda r: obj_responses.append((r.url, r.status)) if r.url.endswith(".obj") else None)
        deadline = time.time() + 600
        while time.time() < deadline:  # wait for HMR2 to finish loading in the backend
            h = page.request.get(f"{a.url}/health")
            if h.ok and h.json().get("model_ready"):
                break
            time.sleep(3)
        page.goto(a.url)
        page.get_by_role("button", name="360° Video Capture").click()
        page.get_by_label("Recording ID").fill(a.video)
        page.get_by_label("Quality report ID").fill(a.report)
        page.get_by_role("button", name="Open using existing authorization").click()
        panel = page.get_by_test_id("unified-reconstruction")
        panel.get_by_role("button", name="Reconstruct 3D body").click(timeout=60000)
        t0 = time.time()
        page.wait_for_timeout(4000)
        panel.screenshot(path=str(out / "ui_progress.png"))
        panel.get_by_text("Loaded in viewer").wait_for(timeout=900000)
        page.get_by_text("Yes (13776 faces)").wait_for(timeout=120000)
        elapsed = time.time() - t0
        page.wait_for_timeout(2500)
        panel.screenshot(path=str(out / "ui_completed.png"))
        canvas = panel.locator("canvas")
        box = canvas.bounding_box()
        page.mouse.move(box["x"] + box["width"] / 2, box["y"] + box["height"] / 2)
        page.mouse.down()
        page.mouse.move(box["x"] + box["width"] / 2 + 220, box["y"] + box["height"] / 2, steps=12)
        page.mouse.up()
        page.wait_for_timeout(1500)
        canvas.screenshot(path=str(out / "ui_rotated.png"))
        page.get_by_title("Reset camera").click()
        page.wait_for_timeout(1500)
        canvas.screenshot(path=str(out / "ui_reset.png"))
        export = panel.get_by_role("link", name="Export OBJ")
        href = export.get_attribute("href")
        dl = page.request.get(a.url + href if href.startswith("/") else href)
        summary = {
            "elapsed_sec": round(elapsed, 1),
            "obj_responses": obj_responses,
            "export_obj_status": dl.status, "export_obj_bytes": len(dl.body()),
            "export_obj_vertex_lines": sum(1 for l in dl.text().splitlines() if l.startswith("v ")),
            "export_obj_face_lines": sum(1 for l in dl.text().splitlines() if l.startswith("f ")),
            "stats_text": panel.locator("dl").inner_text(),
            "console_errors": [c for c in console if c.startswith("error")],
        }
        browser.close()
    (out / "ui_summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
