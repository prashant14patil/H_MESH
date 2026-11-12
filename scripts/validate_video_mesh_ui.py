"""Browser smoke test with mocked APIs and an existing OBJ fixture."""
import json
from pathlib import Path

from playwright.sync_api import sync_playwright

VIDEO = Path("data/production_validation/ecff78fe-773e-4eb8-a785-c356e3be4d19/original.webm").resolve()
IMAGE = Path("data/production_validation/ecff78fe-773e-4eb8-a785-c356e3be4d19/extracted_frames/frame_003.jpg").resolve()
OBJ = Path("outputs/video_to_mesh/ecff78_batch_001/meshes/frame_003_front_candidate.obj").resolve()
VIDEO_ID = "11111111-1111-4111-8111-111111111111"
RUN_ID = "22222222-2222-4222-8222-222222222222"
NAMES = ["frame_001.jpg", "frame_002.jpg", "frame_003.jpg"]
errors = []
mesh_fetches = []
posted = []


def main():
    metadata = {"frames": [{"filename": name, "timestamp_sec": float(i),
                           "blur_score": 50, "is_blurry": True} for i, name in enumerate(NAMES)],
                "video_info": {"duration_sec": 3, "fps": 30, "width": 1280, "height": 720},
                "candidate_frame_count": 3, "accepted_frame_count": 3, "rejected_frame_count": 0}
    poses = {"frames": [{"frame": name[:-4], "person_detected": True,
                         "full_body_visible": True, "landmarks": {}} for name in NAMES]}
    coverage = {"frames": [{"frame": name[:-4], "viewpoint": "unknown", "confidence": 0.0}
                            for name in NAMES],
                "coverage": {"counts": {}, "missing_buckets": [], "sufficient_rotation_coverage": False}}
    finished = {"run_id": RUN_ID, "status": "completed_with_errors", "frames": [
        {"filename": NAMES[0], "status": "completed", "metadata": {"vertices": 6890, "faces": 13776}},
        {"filename": NAMES[1], "status": "failed", "detail": "synthetic failure"},
        {"filename": NAMES[2], "status": "completed", "metadata": {"vertices": 6890, "faces": 13776}},
    ]}

    def route_api(route):
        request = route.request
        path = request.url.split("?", 1)[0]
        if path.endswith("/health"):
            return route.fulfill(json={"status": "ok"})
        if "/api/capture/upload" in path:
            return route.fulfill(status=201, json={"video_id": VIDEO_ID})
        if "/api/capture/process/" in path or "/api/pose/process/" in path:
            return route.fulfill(status=202, json={"status": "processing"})
        if "/api/capture/status/" in path or "/api/pose/status/" in path:
            return route.fulfill(json={"status": "completed"})
        if "/api/capture/frames/" in path:
            return route.fulfill(json=metadata)
        if "/api/pose/viewpoints/" in path:
            return route.fulfill(json=coverage)
        if "/api/pose/results/" in path:
            return route.fulfill(json=poses)
        if "/api/capture/frame/" in path or "/frame/frame_" in path:
            return route.fulfill(path=str(IMAGE), content_type="image/jpeg")
        if "/api/video-mesh/runs" in path and request.method == "POST":
            posted.append(request.post_data_json)
            return route.fulfill(status=202, json={"run_id": RUN_ID, "status": "processing",
                                 "frames": [{"filename": n, "status": "queued"} for n in NAMES]})
        if "/api/video-mesh/runs/" in path and "/mesh/" in path:
            mesh_fetches.append(path)
            return route.fulfill(path=str(OBJ), content_type="application/octet-stream")
        if "/api/video-mesh/runs/" in path:
            return route.fulfill(json=finished)
        return route.fulfill(status=404, json={"detail": path})

    with sync_playwright() as playwright:
        browser = playwright.chromium.launch(
            executable_path=r"C:\Program Files\Google\Chrome\Application\chrome.exe",
            headless=True, args=["--enable-unsafe-swiftshader"])
        page = browser.new_page(viewport={"width": 1440, "height": 900})
        page.on("pageerror", lambda err: errors.append(str(err)))
        page.route("**/api/**", route_api)
        page.route("**/health", route_api)
        try:
            page.goto("http://127.0.0.1:5178/", wait_until="domcontentloaded")
            page.get_by_role("button", name="360° Video Capture").click()
            page.locator('input[type="file"]').set_input_files(str(VIDEO))
            page.get_by_role("button", name="Process Video").click()
            page.get_by_text("Processing completed.").wait_for(timeout=15000)
            panel = page.locator("section").filter(has_text="Step 3 — 3D Mesh Reconstruction")
            panel.get_by_text("0 selected", exact=False).wait_for()
            checks = panel.locator('input[type="checkbox"]')
            for i in range(3):
                checks.nth(i).check()
            panel.get_by_text("All selected frames are blurry", exact=False).wait_for()
            panel.get_by_role("button", name="Generate 3D Meshes").click()
            panel.get_by_text("synthetic failure").wait_for(timeout=15000)
            panel.locator("canvas").wait_for(timeout=15000)
            page.wait_for_timeout(4000)
            panel.locator("select").select_option(NAMES[2])
            page.wait_for_timeout(3000)
            assert posted and posted[0]["frames"] == NAMES, posted
            assert len(mesh_fetches) >= 2, mesh_fetches
            assert not errors, errors
            print(json.dumps({"posted": posted, "mesh_fetches": mesh_fetches,
                              "page_errors": errors, "result": "PASS"}, indent=2))
        finally:
            browser.close()


if __name__ == "__main__":
    main()
