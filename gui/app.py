"""
SatQuery AI — Interactive GUI (Phase 6)
=========================================
Streamlit-based web interface for the SatQuery AI system.

Features:
  - File upload for GeoTIFF/TIFF (single image, optical-SAR pair, bi-temporal pair)
  - Text query input with smart suggestions
  - Agentic router dispatches to the correct specialist tool
  - Results displayed with answer text + overlay images
  - Loading states and error handling

Run:  streamlit run gui/app.py
"""

import io
import os
import sys
import time
import logging
from pathlib import Path

# ── Project path setup ──────────────────────────────────────────────────────
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_PROJECT_ROOT))

import streamlit as st
import numpy as np
from PIL import Image

# Project imports
from router.router import (
    route_query,
    IMAGE_MODE_SINGLE,
    IMAGE_MODE_BITEMPORAL,
    IMAGE_MODE_SAR_OPTICAL,
    TASK_VQA,
    TASK_CLASSIFY,
    TASK_CHANGE,
    TASK_SAR,
    TASK_DESCRIPTIONS,
)
from tools.geotiff_utils import load_geotiff

logger = logging.getLogger(__name__)

# ── Page config ─────────────────────────────────────────────────────────────

st.set_page_config(
    page_title="SatQuery AI — Remote Sensing Assistant",
    page_icon="🛰️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Custom CSS ──────────────────────────────────────────────────────────────

st.markdown("""
<style>
/* ── Import Google Font ── */
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&display=swap');

/* ── Global Font (excluding icon fonts) ── */
html, body, p, div, span, label, input, button {
    font-family: 'Inter', system-ui, -apple-system, sans-serif;
}

/* ── Hide default header/footer ── */
#MainMenu { visibility: hidden; }
header { visibility: hidden; }
footer { visibility: hidden; }

/* ── Main background ── */
.stApp {
    background: linear-gradient(135deg, #0E1117 0%, #151B28 50%, #1A1230 100%);
}

/* ── Hero header ── */
.hero-header {
    text-align: center;
    padding: 1.5rem 0 1rem;
    margin-bottom: 1rem;
}
.hero-header h1 {
    font-size: 2.5rem;
    font-weight: 800;
    background: linear-gradient(135deg, #6C63FF 0%, #A78BFA 50%, #EC4899 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    margin-bottom: 0.3rem;
    letter-spacing: -0.02em;
}
.hero-header p {
    color: #9CA3AF;
    font-size: 1.05rem;
    font-weight: 400;
    margin: 0;
}

/* ── Card containers ── */
.glass-card {
    background: rgba(26, 31, 46, 0.7);
    backdrop-filter: blur(20px);
    border: 1px solid rgba(108, 99, 255, 0.15);
    border-radius: 16px;
    padding: 1.25rem;
    margin-bottom: 1rem;
}

/* ── Task badge ── */
.task-badge {
    display: inline-block;
    padding: 0.35rem 0.9rem;
    border-radius: 20px;
    font-size: 0.78rem;
    font-weight: 600;
    letter-spacing: 0.04em;
    text-transform: uppercase;
}
.task-vqa        { background: rgba(59,130,246,0.2); color: #60A5FA; border: 1px solid rgba(59,130,246,0.3); }
.task-classify   { background: rgba(16,185,129,0.2); color: #34D399; border: 1px solid rgba(16,185,129,0.3); }
.task-change     { background: rgba(245,158,11,0.2); color: #FBBF24; border: 1px solid rgba(245,158,11,0.3); }
.task-sar_fusion { background: rgba(236,72,153,0.2); color: #F472B6; border: 1px solid rgba(236,72,153,0.3); }

/* ── Stat box ── */
.stat-box {
    background: rgba(108, 99, 255, 0.08);
    border: 1px solid rgba(108, 99, 255, 0.2);
    border-radius: 12px;
    padding: 1rem;
    text-align: center;
}
.stat-box .stat-value {
    font-size: 1.5rem;
    font-weight: 700;
    color: #A78BFA;
}
.stat-box .stat-label {
    font-size: 0.78rem;
    color: #9CA3AF;
    text-transform: uppercase;
    letter-spacing: 0.05em;
    margin-top: 0.2rem;
}

/* ── Answer box ── */
.answer-box {
    background: linear-gradient(135deg, rgba(108,99,255,0.12), rgba(167,139,250,0.06));
    border: 1px solid rgba(108, 99, 255, 0.3);
    border-radius: 14px;
    padding: 1.5rem;
    margin: 1rem 0;
}
.answer-box h3 {
    color: #A78BFA;
    font-size: 0.85rem;
    text-transform: uppercase;
    letter-spacing: 0.06em;
    margin-bottom: 0.6rem;
    font-weight: 600;
}
.answer-box p {
    color: #E5E7EB;
    font-size: 1.05rem;
    line-height: 1.7;
    margin: 0;
}

/* ── Sidebar styling ── */
section[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #151B28 0%, #1A1230 100%);
    border-right: 1px solid rgba(108, 99, 255, 0.15);
}

/* ── Smooth image corners ── */
img {
    border-radius: 10px;
}
</style>
""", unsafe_allow_html=True)


# ── Helper functions ────────────────────────────────────────────────────────

def _load_uploaded_image(uploaded_file) -> Image.Image:
    """Load an uploaded file into a PIL Image, handling GeoTIFF properly."""
    suffix = Path(uploaded_file.name).suffix.lower()

    if suffix in {'.tif', '.tiff'}:
        # Write to a temp file so rasterio can open it
        import tempfile
        tmp = Path(tempfile.mkdtemp()) / uploaded_file.name
        tmp.write_bytes(uploaded_file.getvalue())
        loaded = load_geotiff(str(tmp))
        return loaded.pil_image, str(tmp)
    else:
        pil = Image.open(uploaded_file).convert("RGB")
        import tempfile
        tmp = Path(tempfile.mkdtemp()) / uploaded_file.name
        pil.save(str(tmp))
        return pil, str(tmp)


def _task_badge(task: str) -> str:
    """Return HTML for a coloured task badge."""
    label_map = {
        TASK_VQA: "Visual QA",
        TASK_CLASSIFY: "Classification",
        TASK_CHANGE: "Change Detection",
        TASK_SAR: "SAR Fusion",
    }
    label = label_map.get(task, task)
    return f'<span class="task-badge task-{task}">{label}</span>'


EXAMPLE_QUERIES = {
    IMAGE_MODE_SINGLE: [
        "What do you see in this image?",
        "How many buildings are visible?",
        "Is there a river in this image?",
        "Classify this scene",
        "What type of land cover is this?",
        "Describe this satellite image",
    ],
    IMAGE_MODE_BITEMPORAL: [
        "What changed between these two images?",
        "Has any construction occurred?",
        "Describe the differences between the before and after images",
        "Has vegetation decreased?",
        "Are there signs of urban expansion?",
    ],
    IMAGE_MODE_SAR_OPTICAL: [
        "Compare the optical and SAR images",
        "What features are visible in both modalities?",
        "Analyze the SAR and optical pair",
        "What does the radar reveal that optical doesn't?",
    ],
}


# ── Sidebar ─────────────────────────────────────────────────────────────────

with st.sidebar:
    st.markdown("""
    <div style="text-align:center; margin-bottom:1.5rem;">
        <div style="font-size:2.5rem; margin-bottom:0.3rem;">🛰️</div>
        <div style="font-size:1.1rem; font-weight:700; color:#A78BFA; letter-spacing:0.03em;">SatQuery AI</div>
        <div style="font-size:0.75rem; color:#6B7280; margin-top:0.2rem;">SIH 2026 · ISRO PS-26167</div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("---")

    # ── Input mode ──
    st.markdown("##### 📡 Analysis Mode")
    input_mode = st.radio(
        "Select input type",
        options=[
            IMAGE_MODE_SINGLE,
            IMAGE_MODE_BITEMPORAL,
            IMAGE_MODE_SAR_OPTICAL,
        ],
        format_func=lambda x: {
            IMAGE_MODE_SINGLE: "🖼️  Single Image (VQA / Classification)",
            IMAGE_MODE_BITEMPORAL: "🔄  Bi-Temporal Pair (Change Detection)",
            IMAGE_MODE_SAR_OPTICAL: "📡  Optical + SAR Pair (Fusion)",
        }[x],
        index=0,
        label_visibility="collapsed",
    )

    st.markdown("---")

    # ── File uploads ──
    st.markdown("##### 📂 Upload Images")
    st.caption("Supports GeoTIFF, TIFF, PNG, JPG")

    if input_mode == IMAGE_MODE_SINGLE:
        uploaded_1 = st.file_uploader(
            "Upload satellite image",
            type=["tif", "tiff", "png", "jpg", "jpeg", "bmp", "webp"],
            key="upload_single",
        )
        uploaded_2 = None

    elif input_mode == IMAGE_MODE_BITEMPORAL:
        uploaded_1 = st.file_uploader(
            "Before image (T1)",
            type=["tif", "tiff", "png", "jpg", "jpeg", "bmp", "webp"],
            key="upload_before",
        )
        uploaded_2 = st.file_uploader(
            "After image (T2)",
            type=["tif", "tiff", "png", "jpg", "jpeg", "bmp", "webp"],
            key="upload_after",
        )

    else:  # SAR_OPTICAL
        uploaded_1 = st.file_uploader(
            "Optical image",
            type=["tif", "tiff", "png", "jpg", "jpeg", "bmp", "webp"],
            key="upload_optical",
        )
        uploaded_2 = st.file_uploader(
            "SAR image",
            type=["tif", "tiff", "png", "jpg", "jpeg", "bmp", "webp"],
            key="upload_sar",
        )

    st.markdown("---")

    # ── Advanced settings ──
    with st.expander("⚙️ Advanced Settings"):
        timeout = st.slider("VLM timeout (seconds)", 30, 300, 120, step=10)
        max_retries = st.slider("Max retries per model", 1, 5, 2)
        st.caption("Higher timeout = more patience for slow VLM responses")

    # ── Ollama Status ──
    st.markdown("---")
    st.markdown("##### Status")
    try:
        import ollama as _ollama_check
        _ollama_check.list()
        st.markdown('<div style="display:flex;align-items:center;gap:0.5rem;">' 
                    '<div style="width:8px;height:8px;border-radius:50%;background:#34D399;"></div>'
                    '<span style="color:#34D399;font-size:0.85rem;font-weight:500;">Ollama connected</span></div>',
                    unsafe_allow_html=True)
    except Exception:
        st.markdown('<div style="display:flex;align-items:center;gap:0.5rem;">'
                    '<div style="width:8px;height:8px;border-radius:50%;background:#F87171;"></div>'
                    '<span style="color:#F87171;font-size:0.85rem;font-weight:500;">Ollama offline</span>'
                    '<span style="color:#6B7280;font-size:0.75rem;"> (CNN still works)</span></div>',
                    unsafe_allow_html=True)

    # ── About ──
    st.markdown("---")
    with st.expander("About SatQuery AI"):
        st.markdown("""
        **SatQuery AI** is an interactive vision-language
        assistant for multimodal remote sensing image
        analysis.

        **Architecture:**
        - Agentic router dispatches queries
        - 4 specialist tools: VQA, Classification,
          Change Detection, SAR Fusion
        - VLM: qwen2.5vl:7b via Ollama
        - CNN: ResNet18 (EuroSAT) for classification

        **Built for:** SIH 2026 | ISRO Problem 26167
        """)


# ── Main content ────────────────────────────────────────────────────────────

# Hero header
st.markdown("""
<div class="hero-header">
    <h1>SatQuery AI</h1>
    <p>Interactive Vision-Language Assistant for Remote Sensing Image Analysis</p>
</div>
""", unsafe_allow_html=True)


# ── Image preview ───────────────────────────────────────────────────────────

images_loaded = []
image_paths = []

if uploaded_1 is not None:
    try:
        pil_1, path_1 = _load_uploaded_image(uploaded_1)
        images_loaded.append(pil_1)
        image_paths.append(path_1)
    except Exception as e:
        st.error(f"Failed to load image: {e}")
        images_loaded = []

if uploaded_2 is not None:
    try:
        pil_2, path_2 = _load_uploaded_image(uploaded_2)
        images_loaded.append(pil_2)
        image_paths.append(path_2)
    except Exception as e:
        st.error(f"Failed to load second image: {e}")

# Show image previews
if images_loaded:
    if len(images_loaded) == 1:
        col_img, col_space = st.columns([2, 1])
        with col_img:
            st.markdown('<div class="glass-card">', unsafe_allow_html=True)
            st.image(images_loaded[0], caption=uploaded_1.name, width='stretch')
            st.markdown('</div>', unsafe_allow_html=True)
    else:
        col1, col2 = st.columns(2)
        labels = {
            IMAGE_MODE_BITEMPORAL: ("Before (T1)", "After (T2)"),
            IMAGE_MODE_SAR_OPTICAL: ("Optical", "SAR"),
        }
        l1, l2 = labels.get(input_mode, ("Image 1", "Image 2"))
        with col1:
            st.markdown('<div class="glass-card">', unsafe_allow_html=True)
            st.image(images_loaded[0], caption=f"{l1} — {uploaded_1.name}", width='stretch')
            st.markdown('</div>', unsafe_allow_html=True)
        with col2:
            st.markdown('<div class="glass-card">', unsafe_allow_html=True)
            st.image(images_loaded[1], caption=f"{l2} — {uploaded_2.name}", width='stretch')
            st.markdown('</div>', unsafe_allow_html=True)


# ── Query input ─────────────────────────────────────────────────────────────

st.markdown("---")

# Example queries
examples = EXAMPLE_QUERIES.get(input_mode, EXAMPLE_QUERIES[IMAGE_MODE_SINGLE])

col_query, col_btn = st.columns([5, 1])

with col_query:
    query = st.text_input(
        "🔍 Ask a question about your satellite image",
        placeholder="e.g., " + examples[0],
        key="query_input",
    )

with col_btn:
    st.markdown("<br>", unsafe_allow_html=True)  # vertical align
    submit = st.button("Analyze", width='stretch', type="primary")

# Example query chips — use callback to set before widget renders
def _set_example(text):
    st.session_state["query_input"] = text

st.markdown("**Quick examples:**")
chip_cols = st.columns(min(len(examples), 4))
for i, example in enumerate(examples[:4]):
    with chip_cols[i]:
        st.button(example, key=f"example_{i}", width='stretch',
                  on_click=_set_example, args=(example,))


# ── Processing ──────────────────────────────────────────────────────────────

def _check_inputs():
    """Validate that we have the right inputs for the selected mode."""
    if not query or not query.strip():
        st.warning("Please enter a question or instruction.")
        return False

    if input_mode == IMAGE_MODE_SINGLE and not images_loaded:
        st.warning("Please upload an image first.")
        return False

    if input_mode in (IMAGE_MODE_BITEMPORAL, IMAGE_MODE_SAR_OPTICAL):
        if len(images_loaded) < 2:
            st.warning("This mode requires two images. Please upload both.")
            return False

    return True


if submit and _check_inputs():
    # Show processing state
    st.markdown("---")

    with st.spinner("🛰️ Analyzing satellite imagery..."):
        progress = st.progress(0, text="Routing query to specialist tool...")
        t_start = time.perf_counter()

        try:
            # Route and dispatch
            progress.progress(10, text="Classifying query intent...")

            result = route_query(
                query=query,
                images=image_paths,
                image_mode=input_mode,
                timeout=timeout,
                max_retries=max_retries,
            )

            elapsed_total = time.perf_counter() - t_start
            progress.progress(100, text="Analysis complete!")
            time.sleep(0.3)
            progress.empty()

        except Exception as exc:
            progress.empty()
            st.error(f"An error occurred during analysis: {exc}")
            st.stop()

    # ── Display results ─────────────────────────────────────────────────

    task = result.get("task", "")
    status = result.get("status", "error")

    # Routing info bar
    st.markdown(f"""
    <div style="display:flex; align-items:center; gap:1rem; margin-bottom:1rem; flex-wrap:wrap;">
        {_task_badge(task)}
        <span style="color:#6B7280; font-size:0.85rem;">
            Routed via <strong style="color:#9CA3AF">{result.get('routing_method', 'N/A')}</strong>
            in {result.get('routing_elapsed_s', 0):.2f}s
        </span>
    </div>
    """, unsafe_allow_html=True)

    if status == "error":
        st.error(f"**Analysis failed:** {result.get('error', 'Unknown error')}")
    else:
        # ── Answer box ──
        answer = result.get("answer", "No answer generated.")
        # Convert newlines to <br> for proper HTML rendering
        answer_html = answer.replace('\n', '<br>')
        st.markdown(f"""
        <div class="answer-box">
            <h3>🛰️ Analysis Result</h3>
            <p>{answer_html}</p>
        </div>
        """, unsafe_allow_html=True)

        # Track in session history
        if "history" not in st.session_state:
            st.session_state["history"] = []
        st.session_state["history"].append({
            "query": query,
            "task": task,
            "answer": answer[:200],
            "time": f"{result.get('elapsed_s', 0):.1f}s",
        })

        # ── Metrics row ──
        metric_cols = st.columns(4)

        with metric_cols[0]:
            st.markdown(f"""
            <div class="stat-box">
                <div class="stat-value">{result.get('model_used', 'N/A')}</div>
                <div class="stat-label">Model Used</div>
            </div>
            """, unsafe_allow_html=True)

        with metric_cols[1]:
            st.markdown(f"""
            <div class="stat-box">
                <div class="stat-value">{result.get('elapsed_s', 0):.1f}s</div>
                <div class="stat-label">Processing Time</div>
            </div>
            """, unsafe_allow_html=True)

        with metric_cols[2]:
            conf = result.get("confidence", None)
            conf_display = f"{conf:.1%}" if conf and conf > 0 else "—"
            st.markdown(f"""
            <div class="stat-box">
                <div class="stat-value">{conf_display}</div>
                <div class="stat-label">Confidence</div>
            </div>
            """, unsafe_allow_html=True)

        with metric_cols[3]:
            backend = result.get("backend", result.get("model_used", "—"))
            st.markdown(f"""
            <div class="stat-box">
                <div class="stat-value">{backend}</div>
                <div class="stat-label">Backend</div>
            </div>
            """, unsafe_allow_html=True)

        # ── Task-specific outputs ───────────────────────────────────────

        # Classification: show class probabilities
        if task == TASK_CLASSIFY:
            all_classes = result.get("all_classes", [])
            if all_classes:
                st.markdown("#### 📊 Class Probabilities")
                # Show top 5 as a bar chart
                top5 = all_classes[:5]
                import pandas as pd
                df = pd.DataFrame(top5)
                df = df.rename(columns={"class": "Class", "probability": "Probability"})
                st.bar_chart(df.set_index("Class"), color="#6C63FF")

        # Change detection: show overlay images
        if task == TASK_CHANGE:
            st.markdown("#### 🗺️ Change Detection Maps")
            change_map = result.get("change_map")
            diff_image = result.get("diff_image")
            change_mask = result.get("change_mask")
            stats = result.get("stats", {})

            if stats:
                sc1, sc2, sc3 = st.columns(3)
                with sc1:
                    st.metric("Changed Area", f"{stats.get('change_pct', 0):.1f}%")
                with sc2:
                    st.metric("Changed Pixels", f"{stats.get('changed_pixels', 0):,}")
                with sc3:
                    st.metric("Threshold", f"{stats.get('threshold', 0):.0f}")

            img_cols = st.columns(3)
            if change_map:
                with img_cols[0]:
                    st.image(change_map, caption="Change Overlay", width='stretch')
            if diff_image:
                with img_cols[1]:
                    st.image(diff_image, caption="Difference Map", width='stretch')
            if change_mask:
                with img_cols[2]:
                    st.image(change_mask, caption="Change Mask", width='stretch')

        # SAR fusion: show composite and edge comparison
        if task == TASK_SAR:
            st.markdown("#### 🛰️ Optical-SAR Analysis Maps")
            composite = result.get("composite")
            edge_comp = result.get("edge_comparison")
            stats = result.get("stats", {})

            if stats:
                sc1, sc2, sc3, sc4 = st.columns(4)
                with sc1:
                    st.metric("Correlation", f"{stats.get('correlation', 0):.3f}")
                with sc2:
                    st.metric("Edge Agreement", f"{stats.get('edge_agreement_pct', 0):.1f}%")
                with sc3:
                    st.metric("Edge IoU", f"{stats.get('edge_overlap_iou', 0):.3f}")
                with sc4:
                    st.metric("SSIM (approx)", f"{stats.get('ssim_approx', 0):.3f}")

            img_cols = st.columns(2)
            if composite:
                with img_cols[0]:
                    st.image(composite, caption="False-Colour Composite (R=Optical, G=SAR, B=Optical)",
                             width='stretch')
            if edge_comp:
                with img_cols[1]:
                    st.image(edge_comp,
                             caption="Edge Comparison (Green=Optical, Red=SAR, Yellow=Both)",
                             width='stretch')

        # ── Raw output expander ──
        with st.expander("🔧 Debug / Raw Output"):
            st.json({
                k: (str(v) if isinstance(v, Image.Image) else v)
                for k, v in result.items()
                if k not in ("change_map", "diff_image", "change_mask",
                             "composite", "edge_comparison")
            })

elif not submit:
    # ── Welcome state ───────────────────────────────────────────────────
    if not images_loaded:
        st.markdown("""
        <div style="text-align:center; padding:3rem 0; color:#6B7280;">
            <div style="font-size:4rem; margin-bottom:1rem; opacity:0.5;">🛰️</div>
            <h3 style="color:#9CA3AF; font-weight:500; margin-bottom:0.5rem;">Upload an image to get started</h3>
            <p style="max-width:500px; margin:0 auto; line-height:1.6;">
                Select your analysis mode in the sidebar, upload satellite imagery,
                and ask questions about what you see.
            </p>
        </div>
        """, unsafe_allow_html=True)

        # Feature cards
        st.markdown("<br>", unsafe_allow_html=True)
        fc1, fc2, fc3, fc4 = st.columns(4)

        features = [
            ("💬", "Visual QA", "Ask questions about satellite images and get concise answers"),
            ("🏷️", "Scene Classification", "Identify land-cover types using EuroSAT categories"),
            ("🔄", "Change Detection", "Analyze bi-temporal pairs to detect and describe changes"),
            ("📡", "SAR Fusion", "Joint analysis of optical and SAR image pairs"),
        ]

        for col, (icon, title, desc) in zip([fc1, fc2, fc3, fc4], features):
            with col:
                st.markdown(f"""
                <div class="glass-card" style="text-align:center; min-height:180px;">
                    <div style="font-size:2rem; margin-bottom:0.5rem;">{icon}</div>
                    <div style="font-weight:600; color:#E5E7EB; margin-bottom:0.4rem;">{title}</div>
                    <div style="font-size:0.82rem; color:#9CA3AF; line-height:1.5;">{desc}</div>
                </div>
                """, unsafe_allow_html=True)

# ── Footer ──────────────────────────────────────────────────────────────────
st.markdown("---")
st.markdown("""
<div style="text-align:center; padding:1rem 0; color:#4B5563; font-size:0.78rem;">
    SatQuery AI v1.0 &middot; SIH 2026 &middot; ISRO Problem Statement 26167<br>
    Powered by qwen2.5vl:7b &middot; EuroSAT ResNet18 &middot; OpenCV &middot; Streamlit
</div>
""", unsafe_allow_html=True)
