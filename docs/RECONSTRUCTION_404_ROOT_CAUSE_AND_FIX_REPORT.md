# Reconstruction 404: root cause and fix

## Diagnosis

Two separate problems were observed:

1. The running Uvicorn process on `127.0.0.1:18901` was started before the video-mesh router was registered. A direct **POST** to `http://127.0.0.1:18901/api/video-mesh/runs` returned HTTP **404**, `server: uvicorn`, body `{"detail":"Not Found"}`. Its live OpenAPI contained no `/api/video-mesh` paths, even though the current `backend/main.py` registers `backend.routes.video_mesh` under `/api/video-mesh`. This establishes a route-level 404 from the running backend, not a missing video/frame response. The original user's browser request was not captured, so this direct reproduction is not a browser network log.
2. The Vite development proxy in `frontend/vite.config.js` targeted `127.0.0.1:8000`, not the expected backend on `18901`. While 8000 was unavailable, a proxied POST through `http://127.0.0.1:5173/api/video-mesh/runs` returned HTTP 500. This was a separate connectivity/configuration error, not the reproduced 18901 404.

The frontend handler calls `startMeshBatch(videoId, selected)`; `frontend/src/services/videoMeshApi.js` sends `POST /api/video-mesh/runs` with JSON `{ "video_id": "...", "frames": ["frame_001.jpg", ...] }`. Current backend source accepts that schema at the same route and delegates to the existing video mesh batch service. No duplicate inference implementation was needed.

## Fix and runtime

Changed `frontend/vite.config.js` so both `/api` and `/health` proxy to `http://127.0.0.1:18901` by default. Set `VITE_PROXY_TARGET` before starting Vite to select another backend. A separately configured `VITE_API_BASE_URL` bypasses the Vite proxy and must point to the intended backend origin.

The old backend was found to be a non-reloading Uvicorn process launched before route registration. A guarded stop was attempted but refused because its process identity had changed; no process was forcibly stopped. Port 18901 subsequently became unavailable, then a separate attempted startup failed with port-in-use. A later probe found a Uvicorn process running the current app at 18901 with `--reload`; its health response showed CUDA and `model_ready: true`, and a bodyless POST returned **422**, `{"detail":[{"type":"missing","loc":["body"],"msg":"Field required","input":null}]}`. That is schema validation, confirming the POST route is loaded. The exact external restart mechanism is not established.

Vite was started on port 5173 after the configuration change. The same bodyless POST through Vite returned the backend's 422 response, confirming that the proxy reached the registered route. The correct request is:

```text
POST http://127.0.0.1:5173/api/video-mesh/runs
  -> Vite proxy -> http://127.0.0.1:18901/api/video-mesh/runs
Content-Type: application/json
{"video_id":"<capture UUID>","frames":["frame_001.jpg","frame_002.jpg"]}
```

The API returns HTTP 202 with `run_id`, `video_id`, status and frame records. Poll `GET /api/video-mesh/runs/{run_id}`; completed frame records provide run-scoped `mesh_url` values. Existing validation and per-frame failure handling remain in place.

## Live browser verification

`scripts/validate_live_video_mesh.py` used actual browser requests without interception. It uploaded the existing 8.687-second WebM, completed extraction, manually selected `frame_001.jpg` and `frame_002.jpg`, clicked **Generate 3D Meshes**, and captured:

- Browser request: **POST** `http://127.0.0.1:5173/api/video-mesh/runs`, HTTP **202**.
- Capture ID `d11b6001-3735-41b0-a65a-5c3bc4b7c503`; payload frame references matched the response.
- Run ID `0a3c9fe1-5231-4ecd-bf5b-ac56b86254bd`; polling reached `completed`.
- Both frames completed. Each OBJ downloaded through the proxy and passed the existing finite-vertex/face-index structural validator: **6,890 vertices, 13,776 faces**.
- The browser created a viewer canvas and fetched both run-scoped OBJ URLs with HTTP 200 after switching the selector. No page errors were reported. This confirms mesh requests and canvas presence, not independent visual/anatomical assessment or manual rotate/zoom/pan testing.

The selected test frames were flagged blurry and were used to verify routing, not to claim reconstruction quality. The files are independent predictions, not a fused 360° mesh.

**Separate pose issue:** the live `.venv` backend reports `No module named 'matplotlib'` when importing MediaPipe. A direct import reproduced this. `pip check` also reported missing `ml-dtypes`, `opencv-contrib-python`, `sounddevice` and a pandas/NumPy conflict. The existing UI proceeds after pose failure and allowed manual selection, so pose analysis did **not** succeed in this live test. No dependency was installed or upgraded. Resolve the combined runtime in a clean, compatible environment before relying on pose/viewpoint analysis.

## Regressions

- **35 backend tests passed**: video mesh API/worker, existing video-to-mesh client, calibration, geometry, shared-shape and sequential extraction tests.
- `npm --prefix frontend run build` passed; Vite warned about a JavaScript chunk over 500 kB.
- Existing single-photo inference through `http://127.0.0.1:5173` completed and returned HTTP 200 OBJ with 6,890 vertices and 13,776 faces.
- The V1 mesh directory still contained ten OBJ files after testing. Presence was checked; a byte-for-byte before/after comparison was not performed.
- Batch outputs use unique run IDs; no existing run was intentionally overwritten.

## Files changed

- `frontend/vite.config.js` — configurable proxy target, default 18901.
- `scripts/validate_live_video_mesh.py` — unmocked browser/network smoke test.
- `docs/RECONSTRUCTION_404_ROOT_CAUSE_AND_FIX_REPORT.md` — this report.

No HMR2 inference internals or single-photo route were changed. The browser test generated a new capture and batch; these contain personal data and must be handled under the applicable privacy policy.
