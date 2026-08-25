"""
Streamlit Web Application for AI Image Captioning System.
CodSoft Artificial Intelligence Internship - Task 3.
"""

import streamlit as st
from PIL import Image
from utils import get_device, get_device_name, validate_and_load_image
from caption_generator import ImageCaptioner

# ---------------------------------------------------------
# 1. Page Configuration & Modern Minimalist Styling
# ---------------------------------------------------------
st.set_page_config(
    page_title="AI Image Captioning | CodSoft AI",
    page_icon="🖼️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Professional, modern, and uncluttered UI styling
st.markdown(
    """
    <style>
    /* Global layout adjustments */
    .main .block-container {
        padding-top: 2rem;
        padding-bottom: 3rem;
        max-width: 1200px;
    }

    /* Header section */
    .app-header {
        border-bottom: 1px solid rgba(255, 255, 255, 0.08);
        padding-bottom: 1.5rem;
        margin-bottom: 2rem;
    }
    .app-title {
        font-size: 2.2rem;
        font-weight: 700;
        letter-spacing: -0.025em;
        color: #ffffff;
        margin-bottom: 0.35rem;
    }
    .app-subtitle {
        font-size: 1.05rem;
        color: #94a3b8;
        font-weight: 400;
        margin-bottom: 1rem;
        line-height: 1.5;
    }
    .badge-row {
        display: flex;
        gap: 0.5rem;
        flex-wrap: wrap;
        margin-top: 0.5rem;
    }
    .tech-badge {
        display: inline-flex;
        align-items: center;
        background: #18181b;
        color: #e4e4e7;
        border: 1px solid #27272a;
        padding: 0.25rem 0.75rem;
        border-radius: 6px;
        font-size: 0.8rem;
        font-weight: 500;
        letter-spacing: 0.01em;
    }
    .tech-badge-highlight {
        display: inline-flex;
        align-items: center;
        background: rgba(37, 99, 235, 0.12);
        color: #60a5fa;
        border: 1px solid rgba(59, 130, 246, 0.3);
        padding: 0.25rem 0.75rem;
        border-radius: 6px;
        font-size: 0.8rem;
        font-weight: 500;
    }

    /* Content cards */
    .panel-card {
        background: #0f172a;
        border: 1px solid #1e293b;
        border-radius: 12px;
        padding: 1.5rem;
        margin-bottom: 1.25rem;
    }
    .panel-title {
        font-size: 1.1rem;
        font-weight: 600;
        color: #f1f5f9;
        margin-bottom: 1rem;
        display: flex;
        align-items: center;
        gap: 0.5rem;
    }

    /* Caption display box */
    .caption-container {
        background: #090d16;
        border: 1px solid #2563eb;
        border-radius: 10px;
        padding: 1.5rem;
        margin-top: 1rem;
        margin-bottom: 1.25rem;
    }
    .caption-label {
        font-size: 0.8rem;
        text-transform: uppercase;
        letter-spacing: 0.05em;
        color: #60a5fa;
        font-weight: 600;
        margin-bottom: 0.5rem;
    }
    .caption-text {
        font-size: 1.35rem;
        font-weight: 600;
        color: #ffffff;
        line-height: 1.5;
    }

    /* Metric stats row */
    .stats-grid {
        display: grid;
        grid-template-columns: repeat(auto-fit, minmax(140px, 1fr));
        gap: 0.75rem;
        margin-top: 1rem;
    }
    .stat-box {
        background: #18181b;
        border: 1px solid #27272a;
        border-radius: 8px;
        padding: 0.75rem 1rem;
    }
    .stat-label {
        font-size: 0.75rem;
        color: #71717a;
        text-transform: uppercase;
        letter-spacing: 0.04em;
        margin-bottom: 0.25rem;
    }
    .stat-value {
        font-size: 0.95rem;
        font-weight: 600;
        color: #e4e4e7;
    }

    /* Image preview wrapper */
    .image-preview-frame {
        border-radius: 10px;
        overflow: hidden;
        border: 1px solid #27272a;
        background: #09090b;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


# ---------------------------------------------------------
# 2. Cached Model Loader
# ---------------------------------------------------------
@st.cache_resource(show_spinner=False)
def load_captioning_model():
    """
    Loads and caches the BLIP model into memory once across sessions.
    """
    captioner = ImageCaptioner()
    captioner.load_model()
    return captioner


# ---------------------------------------------------------
# 3. Sidebar Configuration & Controls
# ---------------------------------------------------------
with st.sidebar:
    st.markdown("### System & Controls")

    # Detect device
    detected_device = get_device()
    detected_device_name = get_device_name(detected_device)

    # Hardware & Model Info Card
    st.markdown(
        f"""
        <div style="background: #18181b; border: 1px solid #27272a; border-radius: 8px; padding: 1rem; margin-bottom: 1.5rem;">
            <div style="font-size: 0.75rem; color: #71717a; text-transform: uppercase; letter-spacing: 0.04em; margin-bottom: 0.25rem;">Model Architecture</div>
            <div style="font-size: 0.9rem; font-weight: 600; color: #e4e4e7; margin-bottom: 0.75rem;">Salesforce BLIP Base</div>
            <div style="font-size: 0.75rem; color: #71717a; text-transform: uppercase; letter-spacing: 0.04em; margin-bottom: 0.25rem;">Compute Accelerator</div>
            <div style="font-size: 0.9rem; font-weight: 600; color: #34d399;">{detected_device_name}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown("#### Generation Settings")

    caption_mode = st.radio(
        "Mode",
        options=["Standard (Unconditional)", "Guided Prefix (Conditional)"],
        index=0,
        help="Standard mode describes the entire image. Guided mode completes a sentence prefix.",
    )

    prefix_prompt = None
    if caption_mode == "Guided Prefix (Conditional)":
        prefix_prompt = st.text_input(
            "Prompt Prefix:",
            value="A photograph of",
            help="Provide a starting phrase that the AI will complete based on the image.",
        )

    with st.expander("Advanced Inference Parameters", expanded=False):
        num_beams = st.slider(
            "Beam Search Width",
            min_value=1,
            max_value=5,
            value=3,
            help="Higher values evaluate multiple sentence candidates for higher precision.",
        )
        max_new_tokens = st.slider(
            "Max Output Tokens",
            min_value=20,
            max_value=100,
            value=50,
            help="Maximum length of the generated description.",
        )

    st.markdown("---")
    st.markdown(
        """
        <div style="font-size: 0.8rem; color: #71717a; line-height: 1.4;">
            <strong>CodSoft AI Internship</strong><br>
            Task 3: Image Captioning AI<br>
            PyTorch • Transformers • Vision ViT
        </div>
        """,
        unsafe_allow_html=True,
    )


# ---------------------------------------------------------
# 4. Header & Branding Section
# ---------------------------------------------------------
st.markdown(
    f"""
    <div class="app-header">
        <div class="app-title">AI Image Captioning System</div>
        <div class="app-subtitle">
            Deep learning vision-language pipeline for automated natural language scene description
        </div>
        <div class="badge-row">
            <span class="tech-badge-highlight">CodSoft Task 3</span>
            <span class="tech-badge">Salesforce BLIP (ViT + Transformer)</span>
            <span class="tech-badge">PyTorch</span>
            <span class="tech-badge">⚡ {detected_device_name}</span>
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)


# ---------------------------------------------------------
# 5. Model Loading Initialization
# ---------------------------------------------------------
try:
    with st.spinner("Initializing deep learning model weights..."):
        captioner = load_captioning_model()
except Exception as e:
    st.error(
        f"Unable to initialize model. Please check internet connection for initial weights download.\n\nError: {e}"
    )
    st.stop()


# ---------------------------------------------------------
# 6. Image Upload Section
# ---------------------------------------------------------
uploaded_file = st.file_uploader(
    "Choose an image to analyze",
    type=["jpg", "jpeg", "png", "webp"],
    help="Supported image formats: JPG, JPEG, PNG, WEBP",
)

if uploaded_file is not None:
    # Validate and load image
    pil_image, error_msg = validate_and_load_image(uploaded_file)

    if error_msg:
        st.error(f"Image Error: {error_msg}")
    else:
        # Two-column layout for image and result
        col_img, col_out = st.columns([1, 1], gap="large")

        with col_img:
            st.markdown("#### Input Image")
            # Using width="stretch" to comply with current Streamlit API
            st.image(
                pil_image,
                width="stretch",
                caption=f"{uploaded_file.name} — {pil_image.width} × {pil_image.height} px ({pil_image.format or 'RGB'})",
            )

        with col_out:
            st.markdown("#### Caption Output")
            st.write("Analyze visual features and autoregressively decode natural language caption.")

            generate_btn = st.button("Generate Caption", type="primary", width="stretch")

            # Check if button was clicked or caption was already stored in session state for this file
            session_key = f"caption_{uploaded_file.name}_{uploaded_file.size}_{caption_mode}_{num_beams}"

            if generate_btn:
                with st.spinner("Processing visual features with Vision Transformer..."):
                    try:
                        result = captioner.generate_caption(
                            image=pil_image,
                            prefix_prompt=prefix_prompt if caption_mode == "Guided Prefix (Conditional)" else None,
                            num_beams=num_beams,
                            max_new_tokens=max_new_tokens,
                        )
                        st.session_state[session_key] = result
                    except Exception as err:
                        st.error(f"Inference Failure: {str(err)}")

            # Display results if available in session
            if session_key in st.session_state:
                res = st.session_state[session_key]

                # Main Caption Box
                st.markdown(
                    f"""
                    <div class="caption-container">
                        <div class="caption-label">Generated Description</div>
                        <div class="caption-text">"{res['caption']}"</div>
                    </div>
                    <div class="stats-grid">
                        <div class="stat-box">
                            <div class="stat-label">Inference Time</div>
                            <div class="stat-value">{res['latency_seconds']} s</div>
                        </div>
                        <div class="stat-box">
                            <div class="stat-label">Compute Device</div>
                            <div class="stat-value">{res['device']}</div>
                        </div>
                        <div class="stat-box">
                            <div class="stat-label">Beam Width</div>
                            <div class="stat-value">{num_beams}</div>
                        </div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

                st.markdown("<div style='height: 0.75rem;'></div>", unsafe_allow_html=True)
                st.text_input("Copy Caption", value=res["caption"], key=f"copy_{session_key}")

else:
    st.info("Upload an image in JPG, JPEG, PNG, or WEBP format above to begin caption generation.")
