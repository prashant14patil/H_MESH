"""
backend/main.py
---------------
FastAPI application entry point for the Tanitas Human Digital Twin —
LOCAL GPU backend.

Responsibilities:
- Initialise the FastAPI app.
- Register API routers.
- Configure CORS so the React frontend can call the backend.
- Configure structured logging.
- Expose a health-check endpoint that reports model/engine readiness.
- Kick off HMR2 model loading in a background thread at startup so the
  (potentially slow, multi-second) checkpoint load doesn't block the
  server from accepting the health-check immediately.

Unlike the original POC, this backend runs HMR2 inference directly,
in-process, on the local machine's GPU (CUDA) or CPU. There is no
Google Colab, Flask tunnel, or ngrok involved anywhere in this stack.
"""

import logging
import threading

from dotenv import load_dotenv
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from backend.routes.inference import router as inference_router
from backend.services.inference_engine import get_engine

# ---------------------------------------------------------------------------
# Environment
# ---------------------------------------------------------------------------
load_dotenv()

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# App
# ---------------------------------------------------------------------------
app = FastAPI(
    title="Tanitas Human Digital Twin – Local GPU Backend",
    description=(
        "Receives uploaded body images from the React frontend and runs "
        "HMR2 3D body-mesh reconstruction directly on the local GPU — "
        "no Google Colab, Flask tunnel, or ngrok required."
    ),
    version="0.1.0",
)

# ---------------------------------------------------------------------------
# CORS
# Allow the React frontend (Vite default port 5173, fallback 3000) to call
# this backend. Tighten origins in production.
# ---------------------------------------------------------------------------
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:5173",
        "http://127.0.0.1:5173",
        "http://localhost:3000",
        "http://127.0.0.1:3000",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---------------------------------------------------------------------------
# Routers
# ---------------------------------------------------------------------------
app.include_router(inference_router, prefix="/api/inference", tags=["Inference"])


# ---------------------------------------------------------------------------
# Health check
# ---------------------------------------------------------------------------
@app.get("/health", tags=["Health"])
async def health_check() -> dict:
    """
    Liveness probe that also reports whether the HMR2 model has finished
    loading onto the GPU/CPU and is ready to accept inference jobs.
    """
    engine = get_engine()
    return {
        "status": "ok",
        "service": "tanitas-backend",
        "model_ready": engine.is_ready,
        "device": engine.device,
        "model_load_error": engine.load_error,
    }


# ---------------------------------------------------------------------------
# Startup / shutdown events
# ---------------------------------------------------------------------------
@app.on_event("startup")
async def on_startup() -> None:
    logger.info("Backend starting up.")
    logger.info("Loading HMR2 model in background thread (this may take a while)...")

    def _load_in_background():
        engine = get_engine()
        if engine.is_ready:
            logger.info("HMR2 model loaded successfully on device: %s", engine.device)
        else:
            logger.error(
                "HMR2 model failed to load: %s. "
                "The /api/inference/submit endpoint will return 503 until "
                "this is fixed (check HMR2_REPO_PATH / HMR2_CHECKPOINT in .env).",
                engine.load_error,
            )

    threading.Thread(target=_load_in_background, daemon=True).start()


@app.on_event("shutdown")
async def on_shutdown() -> None:
    logger.info("Backend shutting down.")
