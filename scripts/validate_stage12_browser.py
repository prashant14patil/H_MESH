"""Isolated Stage 1.2 UI validation. All APIs and images are synthetic fixtures."""
import json
import os
import tempfile

from playwright.sync_api import sync_playwright

VIDEO = "11111111-1111-4111-8111-111111111111"
RUN = "22222222-2222-4222-8222-222222222222"
SESSION = "33333333-3333-4333-8333-333333333333"
NAMES = ["frame_001.jpg", "frame_002.jpg", "frame_003.jpg"]
BOXES = [{"box": [.1, .1, .8, .9], "score": .9},
         {"box": [.12, .12, .78, .88], "score": .7}]
frames = [{"filename": name, "timestamp_sec": i + .25, "is_blurry": False} for i, name in enumerate(NAMES)]
quality_frames = [{
    "filename": name, "timestamp_sec": i + .25, "source_sha256": "hash" + str(i),
    "status": "review", "warnings": ["possible_multiple_people", "subject_association_ambiguous",
                                        "unusuallylongreasoncode" * 8],
    "rejections": [], "person_detected": True, "person_count_observed": 2,
    "checks": {"readability": "pass", "person_presence": "pass", "sharpness": "pass",
               "lighting": "pass", "full_body": "pass"},
    "viewpoint": {"label": "unknown"}, "extractor_quality": {"is_blurry": False},
    "detector_evidence": BOXES,
    "detector_nms": {"raw_count": 2, "filtered_count": 1, "retained_indices": [0],
                     "suppressed": [{"index": 1, "retained_index": 0, "iou": .82}],
                     "invalid_indices": []},
} for i, name in enumerate(NAMES)]
REPORT = {
    "run_id": RUN, "video_id": VIDEO, "frames": quality_frames,
    "summary": {"extracted_frames": 3, "extractor_non_blurry": 3, "extractor_blurry": 0,
                "rejected_duplicates": 0, "full_body_visible": 3, "possible_multiple_people": 3,
                "recommended_frames": []},
    "viewpoint_coverage": {"status": "unconfirmed", "note": "Unvalidated", "counts": {}},
    "rejected_candidates": [],
}
SVG = b'<svg xmlns="http://www.w3.org/2000/svg" width="400" height="600"><rect width="400" height="600" fill="#ddd"/><rect x="100" y="60" width="200" height="480" fill="#777"/></svg>'


def main():
    errors = []
    calls = []
    saved_decisions = {}
    with sync_playwright() as playwright:
        browser = playwright.chromium.launch(executable_path=r"C:\Program Files\Google\Chrome\Application\chrome.exe", headless=True)
        page = browser.new_page(viewport={"width": 1440, "height": 900})
        page.on("pageerror", lambda error: errors.append(str(error)))

        def route_api(route):
            request = route.request
            path = request.url.split("?", 1)[0]
            calls.append((request.method, path))
            if "/api/capture/frame/" in path:
                route.fulfill(status=200, content_type="image/svg+xml", body=SVG)
                return
            status = 200
            if path.endswith("/api/capture/upload"):
                data = {"video_id": VIDEO, "status": "uploaded"}
            elif path.endswith("/authorization") and request.method == "GET":
                data = {"video_id": VIDEO, "authorized": True, "consent_confirmed": True}
            elif path.endswith("/authorization") and request.method == "POST":
                payload = request.post_data_json
                assert payload == {"authorized": True, "consent": True}
                data = {"video_id": VIDEO, "authorized": True, "consent_confirmed": True}
            elif "/api/capture/process/" in path or "/api/pose/process/" in path:
                data = {"status": "processing"}
            elif "/api/capture/status/" in path or "/api/pose/status/" in path:
                data = {"status": "completed"}
            elif "/api/capture/frames/" in path:
                data = {"video_id": VIDEO, "status": "completed", "frames": frames, "rejected_frames": [], "video_info": {}}
            elif "/api/pose/results/" in path or "/api/pose/viewpoints/" in path:
                data = {"frames": []}
            elif path.endswith("/api/capture-quality/runs") and request.method == "POST":
                data = {"run_id": RUN, "status": "processing"}
            elif path.endswith("/report"):
                data = REPORT
            elif "/api/capture-quality/runs/" in path:
                data = {"run_id": RUN, "status": "completed"}
            elif path.endswith("/api/video-mesh/readiness") and request.method == "POST":
                payload = request.post_data_json
                data = ({"ready": True, "blockers": [], "unevaluated": []} if payload.get("frames")
                        else {"ready": False, "blockers": ["Select between 1 and 12 frames", "A matching subject review session and reference subject are required"],
                              "unevaluated": ["Per-frame checks require a valid frame selection"]})
            elif path.endswith("/api/subject-review/sessions") and request.method == "POST":
                payload = request.post_data_json
                data = {"session_id": SESSION, "quality_run_id": RUN, "video_id": VIDEO,
                        "reference_frame": payload["reference_frame"], "reference_box": payload["bbox"]}
            elif path.endswith("/resume"):
                data = {"session": {"session_id": SESSION, "quality_run_id": RUN, "video_id": VIDEO,
                                    "reference_frame": NAMES[0], "reference_box": BOXES[0]["box"]},
                        "decisions": saved_decisions, "unavailable_frames": []}
            elif "/suggestions/" in path:
                name = path.rsplit("/", 1)[-1]
                data = {"filename": name, "identity_status": "identity_ambiguous",
                        "suggestion": {"status": "ambiguous", "anchor_frame": NAMES[0],
                                       "candidates": [{"box": BOXES[0]["box"], "overlap": .8, "center_distance": .1}]}}
            elif path.endswith("/decisions") and request.method == "POST":
                payload = request.post_data_json
                index = NAMES.index(payload["filename"])
                data = {"session_id": SESSION, "quality_run_id": RUN, "frame_id": payload["filename"],
                        "frame_sha256": "hash" + str(index), "reviewer_decision": payload["decision"],
                        "identity_status": "rejected" if payload["decision"] == "reject" else "human_confirmed_not_biometric",
                        "selected_subject_box": payload["bbox"]}
                saved_decisions[payload["filename"]] = data
            else:
                status, data = 404, {"detail": "Unexpected fixture request"}
            route.fulfill(status=status, content_type="application/json", body=json.dumps(data))

        page.route("**/api/**", route_api)
        try:
            page.goto("http://127.0.0.1:5191/", wait_until="domcontentloaded")
            page.get_by_role("button", name="360° Video Capture").click()
            page.locator('input[type="file"]').set_input_files({"name": "fixture.mp4", "mimeType": "video/mp4", "buffer": b"fixture"})
            assert page.get_by_role("button", name="Process Video").is_disabled()
            upload = page.locator("div").filter(has=page.get_by_role("heading", name="Authorization and consent for selected video")).last
            upload.get_by_label("I am authorized to process this recording and generate meshes.").check()
            assert page.get_by_role("button", name="Process Video").is_disabled()
            upload.get_by_label("Everyone depicted has consented to this processing and mesh generation.").check()
            page.get_by_role("button", name="Process Video").click()
            page.get_by_text("Processing completed.").wait_for(timeout=20000)
            page.get_by_role("button", name="Analyze capture quality").click()
            panel = page.locator("#subject-review")
            panel.wait_for(timeout=15000)
            mesh = page.locator("section").filter(has_text="Step 3 — 3D Mesh Reconstruction")
            cards = mesh.locator("article")
            assert cards.count() == 3
            mesh.get_by_role("button", name="Check server readiness").click()
            mesh.get_by_text("Select between 1 and 12 frames").wait_for()
            assert mesh.get_by_text("A matching subject review session and reference subject are required").count() == 1
            assert mesh.get_by_role("region", name="Final inference confirmation").count() == 0
            assert cards.nth(0).get_by_text("After NMS: 1", exact=False).count() == 1
            assert cards.nth(0).locator('[data-suppressed="true"]').count() == 1
            for width in (1440, 390):
                page.set_viewport_size({"width": width, "height": 900})
                page.wait_for_timeout(200)
                for card in cards.all():
                    card.locator("summary").click()
                    metrics = card.evaluate("""el => {
                      const r = el.getBoundingClientRect();
                      return {scroll: el.scrollWidth, client: el.clientWidth, left: r.left, right: r.right,
                        wrap: getComputedStyle(el).overflowWrap};
                    }""")
                    assert metrics["scroll"] <= metrics["client"] + 1, (width, metrics)
                    assert metrics["left"] >= -1 and metrics["right"] <= width + 1, (width, metrics)
                    assert metrics["wrap"] == "anywhere", metrics
                rect = cards.nth(0).locator("img").bounding_box()
                box = cards.nth(0).locator('[data-detection-index="0"]').bounding_box()
                assert rect and box and abs((box["x"] - rect["x"]) / rect["width"] - .1) < .02
                assert abs(box["width"] / rect["width"] - .7) < .02
                cards.nth(0).scroll_into_view_if_needed()
                page.screenshot(path=os.path.join(tempfile.gettempdir(), f"stage12-cards-{width}.png"))
                for card in cards.all():
                    card.locator("summary").click()
            page.set_viewport_size({"width": 1440, "height": 900})
            image = panel.locator("img")
            image.scroll_into_view_if_needed()
            rect = image.bounding_box()
            assert rect
            page.mouse.move(rect["x"] + rect["width"] * .1, rect["y"] + rect["height"] * .1)
            page.mouse.down()
            page.mouse.move(rect["x"] + rect["width"] * .8, rect["y"] + rect["height"] * .9, steps=5)
            page.mouse.up()
            panel.get_by_role("button", name="Select reference subject").click()
            panel.get_by_text("Reference subject selected", exact=False).wait_for()
            assert panel.locator("p").filter(has_text="Subject association: WARNING").count() == 1
            panel.get_by_role("button", name="Use suggested box 1", exact=False).click()
            panel.get_by_role("button", name="Confirm subject").click()
            panel.locator("p").filter(has_text="Subject association: PASS").wait_for()
            panel.get_by_role("button", name="Use suggested box 1", exact=False).click()
            with page.expect_request(lambda request: request.url.endswith("/decisions") and request.method == "POST") as reassignment:
                panel.get_by_role("button", name="Reassign subject").click()
            assert reassignment.value.post_data_json["decision"] == "reassign"
            panel.locator("p").filter(has_text="Subject association: PASS").wait_for()
            assert cards.nth(0).locator('input[type="checkbox"]').first.is_enabled()
            cards.nth(0).locator('input[type="checkbox"]').first.check()
            assert mesh.get_by_role("region", name="Final inference confirmation").count() == 0
            cards.nth(0).get_by_label("Review frame_001.jpg").check()
            mesh.get_by_role("button", name="Check server readiness").click()
            confirmation = mesh.get_by_role("region", name="Final inference confirmation")
            confirmation.wait_for()
            assert confirmation.get_by_role("button", name="Confirm and generate 3D meshes").is_disabled()
            confirmation.get_by_label("I confirm these selected frames and understand that real HMR2 inference will start.").check()
            assert confirmation.get_by_role("button", name="Confirm and generate 3D meshes").is_enabled()
            cards.nth(0).get_by_label("Review frame_001.jpg").uncheck()
            assert confirmation.count() == 0
            cards.nth(1).get_by_role("button", name="Review this frame and detection boxes").click()
            assert panel.locator("select").input_value() == NAMES[1]
            assert panel.locator("p").filter(has_text="Subject association: WARNING").count() == 1
            panel.get_by_role("button", name="Reject frame").click()
            panel.locator("p").filter(has_text="Subject association: FAIL").wait_for()
            assert cards.nth(1).locator('input[type="checkbox"]').first.is_disabled()
            page.get_by_label("Recording ID", exact=True).fill(VIDEO)
            page.get_by_label("Quality report ID", exact=True).fill(RUN)
            assert page.get_by_role("button", name="Open saved capture").is_disabled()
            saved = page.locator("form").filter(has=page.get_by_role("heading", name="Reopen saved capture"))
            saved.get_by_label("I am authorized to process this recording and generate meshes.").check()
            assert page.get_by_role("button", name="Open saved capture").is_disabled()
            saved.get_by_label("Everyone depicted has consented to this processing and mesh generation.").check()
            page.get_by_role("button", name="Open saved capture").click()
            panel.get_by_label("Existing review session ID").fill(SESSION)
            panel.get_by_role("button", name="Resume session").click()
            panel.get_by_text("Session restored.", exact=False).wait_for()
            assert cards.nth(0).locator('input[type="checkbox"]').first.is_enabled()
            assert not cards.nth(0).locator('input[type="checkbox"]').first.is_checked()
            assert mesh.get_by_role("region", name="Final inference confirmation").count() == 0
            cards.nth(0).locator('input[type="checkbox"]').first.check()
            assert mesh.get_by_role("region", name="Final inference confirmation").count() == 0
            assert not any(path.endswith("/api/video-mesh/runs") and method == "POST" for method, path in calls)
            assert not errors, errors
            print("PASS: fixture UI at 1440px and 390px; overlays, wrapping, reference, confirm, reject, acknowledgment and blocking")
        finally:
            browser.close()


if __name__ == "__main__":
    main()
