"""
frontend/app.py
----------------
Streamlit web application for the Tanita Human Digital Twin POC.

Responsibilities:
- Provide a simple web UI for uploading a full-body RGB photograph.
- Send the image to the local FastAPI backend.
- Poll the backend for inference status.
- Parse the returned OBJ mesh and display it as an interactive 3D Plotly chart.
- Offer a download button for the raw OBJ file.

Environment variables (set in .env or shell):
    BACKEND_URL       – Base URL of the local FastAPI backend
                        (default: http://localhost:8000).
    POLL_INTERVAL_SEC – Seconds between status polls (default: 3).
    MAX_POLL_ATTEMPTS – Maximum poll attempts before giving up (default: 40).

Architecture note:
    This frontend never communicates with any external cloud service.
    Inference runs in-process inside the local FastAPI backend,
    directly on the local machine's GPU/CPU.

Usage:
    streamlit run frontend/app.py
"""

import io
import os
import time

import numpy as np
import plotly.graph_objects as go
import requests
import streamlit as st
from dotenv import load_dotenv
from PIL import Image

load_dotenv()

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
BACKEND_URL: str = os.getenv("BACKEND_URL", "http://localhost:8000").rstrip("/")
POLL_INTERVAL_SEC: int = int(os.getenv("POLL_INTERVAL_SEC", "3"))
MAX_POLL_ATTEMPTS: int = int(os.getenv("MAX_POLL_ATTEMPTS", "40"))

# ---------------------------------------------------------------------------
# Page configuration
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="Tanita Human Digital Twin",
    page_icon="🧬",
    layout="wide",
    initial_sidebar_state="expanded",
)


# ---------------------------------------------------------------------------
# Backend helpers
# ---------------------------------------------------------------------------
def check_backend_health() -> bool:
    """Return True if the backend health endpoint responds OK."""
    try:
        resp = requests.get(f"{BACKEND_URL}/health", timeout=5)
        return resp.status_code == 200
    except requests.RequestException:
        return False


def submit_image(image_bytes: bytes, filename: str) -> dict:
    """POST the image to the backend and return the parsed JSON response."""
    resp = requests.post(
        f"{BACKEND_URL}/api/inference/submit",
        files={"image": (filename, image_bytes)},
        timeout=30,
    )
    resp.raise_for_status()
    return resp.json()


def poll_status(job_id: str) -> dict:
    """GET the current status for a job from the backend."""
    resp = requests.get(
        f"{BACKEND_URL}/api/inference/status/{job_id}",
        timeout=15,
    )
    resp.raise_for_status()
    return resp.json()


def fetch_mesh_bytes(mesh_url: str) -> bytes:
    """Download the OBJ file from the backend mesh endpoint."""
    # mesh_url may be a relative path like /api/inference/mesh/<id>
    if mesh_url.startswith("/"):
        url = f"{BACKEND_URL}{mesh_url}"
    else:
        url = mesh_url
    resp = requests.get(url, timeout=60)
    resp.raise_for_status()
    return resp.content


# ---------------------------------------------------------------------------
# OBJ parser
# ---------------------------------------------------------------------------
def parse_obj(obj_bytes: bytes) -> tuple[np.ndarray, np.ndarray]:
    """
    Parse a Wavefront OBJ file and return (vertices, faces).

    Parameters
    ----------
    obj_bytes : bytes
        Raw OBJ file contents.

    Returns
    -------
    vertices : np.ndarray, shape (N, 3)  – x, y, z coordinates.
    faces    : np.ndarray, shape (M, 3)  – 0-based vertex indices.
    """
    vertices = []
    faces = []
    for line in obj_bytes.decode("utf-8", errors="replace").splitlines():
        line = line.strip()
        if line.startswith("v "):
            parts = line.split()
            vertices.append([float(parts[1]), float(parts[2]), float(parts[3])])
        elif line.startswith("f "):
            parts = line.split()[1:]
            # Handle v, v/vt, v/vt/vn, v//vn formats – take the vertex index only.
            idx = [int(p.split("/")[0]) - 1 for p in parts]
            if len(idx) == 3:
                faces.append(idx)
            elif len(idx) == 4:
                # Triangulate the quad
                faces.append([idx[0], idx[1], idx[2]])
                faces.append([idx[0], idx[2], idx[3]])
    return np.array(vertices, dtype=np.float32), np.array(faces, dtype=np.int32)


# ---------------------------------------------------------------------------
# 3D viewer
# ---------------------------------------------------------------------------
def build_mesh_figure(vertices: np.ndarray, faces: np.ndarray) -> go.Figure:
    """
    Build a Plotly 3D Mesh figure from SMPL vertices and faces.

    The mesh is centred and displayed with equal axis scaling so the body
    proportions are not distorted.
    """
    x, y, z = vertices[:, 0], vertices[:, 1], vertices[:, 2]
    i, j, k = faces[:, 0], faces[:, 1], faces[:, 2]

    # Centre the mesh
    x = x - x.mean()
    y = y - y.mean()
    z = z - z.mean()

    # Compute a shared range for equal-axis scaling
    max_range = max(x.ptp(), y.ptp(), z.ptp()) / 2.0
    mid_x, mid_y, mid_z = x.mean(), y.mean(), z.mean()

    mesh = go.Mesh3d(
        x=x, y=y, z=z,
        i=i, j=j, k=k,
        color="#6fa8dc",
        opacity=0.85,
        flatshading=False,
        lighting=dict(ambient=0.4, diffuse=0.8, specular=0.3, roughness=0.5),
        lightposition=dict(x=100, y=200, z=150),
        name="SMPL mesh",
        showscale=False,
    )

    fig = go.Figure(data=[mesh])
    fig.update_layout(
        scene=dict(
            xaxis=dict(range=[mid_x - max_range, mid_x + max_range],
                       showbackground=False, title=""),
            yaxis=dict(range=[mid_y - max_range, mid_y + max_range],
                       showbackground=False, title=""),
            zaxis=dict(range=[mid_z - max_range, mid_z + max_range],
                       showbackground=False, title=""),
            aspectmode="cube",
            bgcolor="rgb(15,15,25)",
        ),
        paper_bgcolor="rgb(15,15,25)",
        margin=dict(l=0, r=0, t=0, b=0),
        height=560,
    )
    return fig


# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------
def render_sidebar() -> None:
    with st.sidebar:
        st.markdown("## 🧬 Tanita POC")
        st.markdown(
            """
            Upload a full-body photograph to reconstruct
            a 3D body mesh using **HMR 2.0 / 4D-Humans**
            running locally on your machine's GPU/CPU.

            ---
            **Disclaimer**

            This is an experimental POC.
            HMR2 estimates external body geometry and pose
            from a monocular photograph.
            Results do **not** measure body fat, muscle mass,
            or internal body composition.
            """
        )
        st.markdown("---")
        st.markdown("### Backend Status")
        if check_backend_health():
            st.success("✅ Backend online")
        else:
            st.error(f"❌ Backend offline\n`{BACKEND_URL}`")

        st.markdown("---")
        st.markdown("### Settings")
        st.caption(f"Backend: `{BACKEND_URL}`")
        st.caption(f"Poll interval: {POLL_INTERVAL_SEC}s")
        st.caption(f"Max attempts: {MAX_POLL_ATTEMPTS}")


# ---------------------------------------------------------------------------
# Step 1 – Upload
# ---------------------------------------------------------------------------
def render_upload_section() -> tuple[bytes | None, str | None]:
    """Render the upload widget and return (bytes, filename) or (None, None)."""
    st.markdown("## Step 1 – Upload a Full-Body Photograph")
    st.markdown(
        "Upload a **front-facing, full-body** RGB photograph (JPEG or PNG).  \n"
        "For best results: stand straight, arms slightly apart, plain background."
    )

    uploaded_file = st.file_uploader(
        label="Choose an image",
        type=["jpg", "jpeg", "png"],
        help="Maximum file size: 10 MB. JPEG or PNG only.",
    )

    if uploaded_file is not None:
        col1, col2 = st.columns([1, 2])
        with col1:
            st.markdown("**Preview**")
            image = Image.open(uploaded_file)
            st.image(image, caption=uploaded_file.name, use_column_width=True)
            st.caption(
                f"Size: {uploaded_file.size / 1024:.1f} KB  |  "
                f"Format: {image.format or 'unknown'}  |  "
                f"{image.width} × {image.height} px"
            )
        uploaded_file.seek(0)
        return uploaded_file.read(), uploaded_file.name

    return None, None


# ---------------------------------------------------------------------------
# Step 2 – Reconstruct
# ---------------------------------------------------------------------------
def render_inference_section(image_bytes: bytes, filename: str) -> None:
    """Handle submission and polling for a given image."""
    st.markdown("## Step 2 – Run 3D Mesh Reconstruction")

    if st.button("🚀 Reconstruct 3D Mesh", type="primary"):
        # Submit
        with st.spinner("Submitting image to inference service …"):
            try:
                submit_resp = submit_image(image_bytes, filename)
            except requests.HTTPError as exc:
                st.error(f"Submission failed: {exc.response.text}")
                return
            except requests.RequestException as exc:
                st.error(f"Could not reach the backend: {exc}")
                return

        job_id = submit_resp.get("job_id")
        st.info(f"Job submitted. ID: `{job_id}`")

        # Poll
        status_placeholder = st.empty()
        progress_bar = st.progress(0)

        for attempt in range(1, MAX_POLL_ATTEMPTS + 1):
            progress = min(attempt / MAX_POLL_ATTEMPTS, 0.95)
            progress_bar.progress(progress)

            try:
                status_resp = poll_status(job_id)
            except requests.RequestException as exc:
                status_placeholder.warning(f"Polling error (attempt {attempt}): {exc}")
                time.sleep(POLL_INTERVAL_SEC)
                continue

            current_status = status_resp.get("status", "unknown")
            status_placeholder.info(
                f"Status: **{current_status}** (attempt {attempt}/{MAX_POLL_ATTEMPTS})"
            )

            if current_status == "completed":
                progress_bar.progress(1.0)
                st.success("✅ Reconstruction complete!")
                render_result_section(status_resp)
                return

            if current_status in ("failed", "error"):
                st.error(
                    f"Inference failed: {status_resp.get('detail', 'Unknown error')}"
                )
                return

            time.sleep(POLL_INTERVAL_SEC)

        st.warning(
            "Inference did not complete within the expected time.  \n"
            f"Job ID `{job_id}` – refresh the page to retry."
        )


# ---------------------------------------------------------------------------
# Step 3 – View result
# ---------------------------------------------------------------------------
def render_result_section(status_resp: dict) -> None:
    """Display the 3D mesh result once inference is complete."""
    st.markdown("## Step 3 – View 3D Body Mesh")

    mesh_url = status_resp.get("mesh_url")
    metadata = status_resp.get("metadata", {})

    # Metrics
    if metadata:
        c1, c2, c3 = st.columns(3)
        c1.metric("Vertices", metadata.get("vertices", "6,890"))
        c2.metric("Faces", metadata.get("faces", "13,776"))
        c3.metric("Body Model", metadata.get("body_model", "SMPL"))

    if not mesh_url:
        st.warning("No mesh URL returned by the inference service.")
        return

    # Download OBJ bytes from backend
    with st.spinner("Downloading mesh …"):
        try:
            obj_bytes = fetch_mesh_bytes(mesh_url)
        except requests.RequestException as exc:
            st.error(f"Could not download mesh: {exc}")
            return

    # Download button
    st.download_button(
        label="⬇️ Download OBJ file",
        data=obj_bytes,
        file_name="body_mesh.obj",
        mime="application/octet-stream",
    )

    # Parse and render
    st.markdown("### Interactive 3D Mesh Viewer")
    st.caption(
        "Drag to rotate · Scroll to zoom · Right-click drag to pan.  \n"
        "Reconstruction shows estimated external body shape and pose. "
        "It does not represent internal body composition."
    )

    try:
        vertices, faces = parse_obj(obj_bytes)
        if len(vertices) == 0:
            st.warning("OBJ file was downloaded but contains no vertex data.")
            return

        fig = build_mesh_figure(vertices, faces)
        st.plotly_chart(fig, use_container_width=True)

        st.caption(
            f"Mesh: {len(vertices):,} vertices · {len(faces):,} faces · "
            "Estimated external geometry only (HMR2 / SMPL body model)"
        )

    except Exception as exc:
        st.error(f"Could not render 3D viewer: {exc}")
        st.info("The OBJ file was downloaded successfully. Open it in MeshLab or Blender.")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
def main() -> None:
    st.title("🧬 Tanita Human Digital Twin")
    st.markdown(
        "Reconstruct a **3D human body mesh** from a single RGB photograph "
        "using HMR 2.0 / 4D-Humans, running entirely on your local GPU."
    )
    st.divider()

    render_sidebar()

    image_bytes, filename = render_upload_section()

    if image_bytes is not None:
        st.divider()
        render_inference_section(image_bytes, filename)
    else:
        st.markdown("---")
        st.markdown("## Step 2 – Run 3D Mesh Reconstruction")
        st.info("⬆️ Upload a photograph above to continue.")


if __name__ == "__main__":
    main()
