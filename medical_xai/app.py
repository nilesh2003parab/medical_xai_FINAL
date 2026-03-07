"""
Real-Time Explainable Deep Learning Framework for Medical Image Classification
MSc/PhD Level Project — Streamlit Interface

Supported Kaggle Dataset:
    Chest X-Ray Images (Pneumonia)
    https://www.kaggle.com/datasets/paultimothymooney/chest-xray-pneumonia
    Classes: NORMAL / PNEUMONIA
"""

# ── Standard Library ──────────────────────────────────────────────────────────
import csv
import os
import sys
import time
import traceback
from datetime import datetime
from pathlib import Path

# ── Ensure folders exist on Streamlit Cloud ───────────────────────────────────
os.makedirs("records", exist_ok=True)
os.makedirs("weights", exist_ok=True)

# ── Matplotlib backend BEFORE any other matplotlib/pyplot import ──────────────
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ── Streamlit (imported early so we can show errors on screen) ────────────────
import streamlit as st

# ── Remaining third-party ─────────────────────────────────────────────────────
try:
    import numpy as np
    import torch
    from PIL import Image
    from models.fusion_model import FusionModel
    from utils.preprocessing import get_transform
    from utils.feedback_dataset import doctor_feedback
    from explainability.gradcam import generate_gradcam
    from explainability.lime_exp import run_lime
    from explainability.xray_annotator import annotate_xray
    from utils.treatment_protocol import get_treatment_plan, get_dominant_severity
    from utils.report_generator import generate_pdf_report
    from explainability.shap_exp import run_shap
    from evaluation.escore import e_score
    from utils.image_quality import check_image_quality, annotate_quality_issues, quality_gauge_figure
except Exception as _import_err:
    st.set_page_config(page_title="MedXAI", page_icon="🫁", layout="wide")
    st.error(f"❌ Startup failed: {_import_err}")
    st.code(traceback.format_exc())
    st.stop()

# ─────────────────────────────────────────────────────────────────────────────
# PAGE CONFIG
# ─────────────────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="MedXAI — Explainable Medical Image Analysis",
    page_icon="🫁",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─────────────────────────────────────────────────────────────────────────────
# CUSTOM CSS — Dark clinical theme
# ─────────────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');

    html, body, [class*="css"] {
        font-family: 'Inter', 'Segoe UI', sans-serif;
    }

    /* PAGE — white */
    .stApp,
    [data-testid="stAppViewContainer"],
    [data-testid="stHeader"],
    .main, .block-container,
    [data-testid="stMainBlockContainer"] {
        background-color: #ffffff !important;
    }

    /* SIDEBAR — white with sky blue right border */
    section[data-testid="stSidebar"],
    section[data-testid="stSidebar"] > div {
        background-color: #ffffff !important;
        border-right: 2px solid #29b6f6 !important;
    }
    section[data-testid="stSidebar"] label,
    section[data-testid="stSidebar"] p,
    section[data-testid="stSidebar"] span {
        color: #0277bd !important;
        font-weight: 500 !important;
    }

    /* BUTTONS — sky blue */
    .stButton > button {
        background-color: #29b6f6 !important;
        color: #ffffff !important;
        border: none !important;
        border-radius: 6px !important;
        font-weight: 600 !important;
        padding: 8px 22px !important;
    }
    .stButton > button:hover {
        background-color: #0288d1 !important;
    }

    /* FILE UPLOADER — white box, sky blue dashed border */
    [data-testid="stFileUploader"],
    [data-testid="stFileUploader"] > div,
    [data-testid="stFileUploader"] section,
    [data-testid="stFileUploaderDropzone"],
    [data-testid="stFileUploaderDropzoneInstructions"] {
        background-color: #ffffff !important;
        border: 2px dashed #29b6f6 !important;
        border-radius: 8px !important;
        color: #0277bd !important;
    }
    [data-testid="stFileUploader"] button,
    [data-testid="stFileUploaderDropzone"] button {
        background-color: #29b6f6 !important;
        color: #ffffff !important;
        border: none !important;
        border-radius: 5px !important;
    }
    [data-testid="stFileUploader"] span,
    [data-testid="stFileUploader"] p,
    [data-testid="stFileUploader"] small {
        color: #0277bd !important;
    }

    /* TABS — sky blue active, white inactive */
    .stTabs [data-baseweb="tab-list"] {
        background-color: #e1f5fe !important;
        border-radius: 8px;
        padding: 3px;
    }
    .stTabs [data-baseweb="tab"] {
        color: #0277bd !important;
        font-weight: 500;
        background: transparent !important;
        border-radius: 6px;
    }
    .stTabs [aria-selected="true"] {
        background-color: #29b6f6 !important;
        color: #ffffff !important;
        font-weight: 600 !important;
    }

    /* INPUTS — white with sky blue border */
    input, textarea, select {
        background-color: #ffffff !important;
        border: 1.5px solid #29b6f6 !important;
        border-radius: 6px !important;
        color: #01579b !important;
    }
    [data-testid="stSelectbox"] > div > div,
    [data-baseweb="select"] > div {
        background-color: #ffffff !important;
        border: 1.5px solid #29b6f6 !important;
        color: #01579b !important;
    }
    [data-testid="stNumberInput"] input {
        background-color: #ffffff !important;
        border: 1.5px solid #29b6f6 !important;
        color: #01579b !important;
    }

    /* METRICS */
    [data-testid="stMetricValue"] {
        color: #0288d1 !important;
        font-weight: 700 !important;
    }
    [data-testid="stMetricLabel"] {
        color: #29b6f6 !important;
    }

    /* EXPANDER */
    .streamlit-expanderHeader,
    [data-testid="stExpander"] summary {
        background-color: #e1f5fe !important;
        color: #0277bd !important;
        border-radius: 6px !important;
        font-weight: 500 !important;
    }

    /* TEXT */
    p, li, h1, h2, h3, h4, label, span {
        color: #01579b !important;
    }
    .stMarkdown p { color: #01579b !important; }
    .stCaption { color: #29b6f6 !important; }

    /* DIVIDER */
    hr { border-color: #29b6f6 !important; opacity: 0.3; }

    /* CARDS */
    .xai-card {
        background: #ffffff !important;
        border: 1.5px solid #29b6f6;
        border-radius: 8px;
        padding: 14px 18px;
        margin: 8px 0;
        box-shadow: 0 1px 6px rgba(41,182,246,0.12);
    }
    .xai-card-title {
        font-size: 0.92rem;
        font-weight: 600;
        color: #0288d1;
        margin-bottom: 6px;
    }

    /* SECTION HEADER */
    .section-header {
        font-size: 1rem;
        font-weight: 600;
        color: #0288d1;
        border-left: 4px solid #29b6f6;
        padding-left: 10px;
        margin: 16px 0 10px 0;
    }

    /* TOP HEADER BAR */
    .top-header {
        background: linear-gradient(90deg, #0288d1, #29b6f6);
        border-radius: 8px;
        padding: 16px 22px;
        margin-bottom: 20px;
        display: flex;
        align-items: center;
        gap: 14px;
    }

    /* BADGES */
    .pred-badge-pneumonia {
        display: inline-block;
        background: #e53935;
        color: #ffffff;
        font-weight: 600;
        font-size: 0.95rem;
        padding: 8px 20px;
        border-radius: 6px;
    }
    .pred-badge-normal {
        display: inline-block;
        background: #2e7d32;
        color: #ffffff;
        font-weight: 600;
        font-size: 0.95rem;
        padding: 8px 20px;
        border-radius: 6px;
    }

    /* SCORE CHIP */
    .score-chip {
        display: inline-block;
        background: #e1f5fe;
        border: 1px solid #29b6f6;
        border-radius: 20px;
        padding: 3px 12px;
        font-size: 0.82rem;
        color: #0288d1;
        font-weight: 500;
    }

</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────────────────────
# DEVICE & MODEL LOADING
# ─────────────────────────────────────────────────────────────────────────────
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


@st.cache_resource(show_spinner=False)
def load_model():
    m = FusionModel().to(device)
    weights_path = Path("weights/resnet18_pneumonia_classifier.pth")
    if weights_path.exists():
        try:
            state_dict = torch.load(str(weights_path), map_location=device)
            # Filter out final FC keys (shape mismatch due to fine-tuned head)
            cnn_keys = {k.replace("cnn.", ""): v for k, v in state_dict.items() if k.startswith("cnn.")}
            if cnn_keys:
                m.cnn.load_state_dict(cnn_keys, strict=False)
            else:
                m.load_state_dict(state_dict, strict=False)
        except Exception as err:
            st.warning(f"⚠️ Custom weights not loaded — using ImageNet pretrained: {err}")
    m.eval()
    return m

# ─────────────────────────────────────────────────────────────────────────────
# CACHED INFERENCE FUNCTIONS
# ─────────────────────────────────────────────────────────────────────────────

@st.cache_data(show_spinner=False)
def cached_prediction(_model, _img_tensor, threshold=0.35):
    """
    Predict with adjustable decision threshold.
    threshold: if P(Pneumonia) >= threshold → predict Pneumonia
    Lowered from 0.50 to 0.35 to reduce false negatives (missed pneumonia).
    """
    _model.eval()
    with torch.no_grad():
        logits = _model(_img_tensor)
        probs  = torch.softmax(logits, dim=1)
    pneumonia_prob = probs[0, 1].item()
    # Apply custom threshold instead of argmax
    if pneumonia_prob >= threshold:
        pred_class = 1
        confidence = pneumonia_prob
        label = "Pneumonia"
    else:
        pred_class = 0
        confidence = probs[0, 0].item()
        label = "Normal"
    return pred_class, confidence, label, probs[0].tolist()

# No cache on lime/shap — ensures fresh results with updated code
def cached_lime(_model, image_np, _transform):
    return run_lime(_model, image_np, _transform)

# No cache — always recompute with latest code
def cached_shap(_model, _img_tensor):
    fig, score = run_shap(_model, _img_tensor)
    # Convert figure to PNG bytes so it survives cache serialization
    import io
    buf = io.BytesIO()
    fig.savefig(buf, format="png", bbox_inches="tight", dpi=130, facecolor="white")
    buf.seek(0)
    plt.close(fig)
    return buf.getvalue(), score

@st.cache_data(show_spinner=False)
def cached_gradcam(_model, _img_tensor, _image):
    return generate_gradcam(_model, _img_tensor, _image)

@st.cache_data(show_spinner=False)
def cached_escore(_model, _img_tensor, label):
    return e_score(_model, _img_tensor, label)

# ─────────────────────────────────────────────────────────────────────────────
# UTILITY
# ─────────────────────────────────────────────────────────────────────────────

def progress_bar(placeholder, label: str, steps: int = 5, delay: float = 0.05):
    prog = placeholder.progress(0, text=label)
    for i in range(0, 101, 100 // steps):
        prog.progress(min(i, 100), text=label)
        time.sleep(delay)
    prog.progress(100, text="")
    placeholder.empty()

# ─────────────────────────────────────────────────────────────────────────────
# SIDEBAR — Patient Information
# ─────────────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 🏥 Patient Information")
    st.caption("Fill in patient details before analysis")
    st.markdown("---")

    patient_id   = st.text_input("Patient ID",   placeholder="e.g. PT-2024-0042")
    patient_name = st.text_input("Patient Name",  placeholder="Full name")
    patient_age  = st.number_input("Age (years)", min_value=0, max_value=120, value=35)
    patient_sex  = st.selectbox("Sex", ["Male", "Female", "Other"])
    ward         = st.text_input("Ward / Department", placeholder="e.g. Radiology")

    st.markdown("---")
    st.markdown("#### 🩺 Medical History")
    disease_present = st.radio("Symptoms Present?", ["Yes", "No"], horizontal=True)
    major_surgeries = st.text_area("Surgeries / Procedures", placeholder="None / List here", height=70)

    st.markdown("#### 💊 Comorbidities")
    col_a, col_b = st.columns(2)
    with col_a:
        diabetes   = st.checkbox("Diabetes")
        thyroid    = st.checkbox("Thyroid")
        asthma     = st.checkbox("Asthma")
    with col_b:
        bp         = st.checkbox("Hypertension")
        cholesterol= st.checkbox("Dyslipidemia")
        copd       = st.checkbox("COPD")

    st.markdown("---")
    st.markdown("#### ⚙️ Analysis Settings")
    lime_samples   = st.slider("LIME Samples",   min_value=100, max_value=1000, value=500, step=100)
    lime_features  = st.slider("LIME Regions",   min_value=5,   max_value=20,   value=10)
    shap_background= st.slider("SHAP Background", min_value=5,  max_value=50,   value=20)

    st.markdown("---")
    st.markdown("#### 🎯 Prediction Sensitivity")
    pneumonia_threshold = st.slider(
        "Pneumonia Threshold",
        min_value=0.10, max_value=0.70, value=0.35, step=0.05,
        help="Lower = catches more Pneumonia (fewer missed cases). Higher = more conservative. Default 0.35 recommended."
    )
    # Sensitivity label
    if pneumonia_threshold <= 0.25:
        sens_label = "🔴 Very High Sensitivity"
        sens_note  = "May over-detect pneumonia"
    elif pneumonia_threshold <= 0.40:
        sens_label = "🟢 Recommended"
        sens_note  = "Balanced — reduces missed cases"
    elif pneumonia_threshold <= 0.55:
        sens_label = "🟡 Conservative"
        sens_note  = "May miss borderline cases"
    else:
        sens_label = "🔴 Very Conservative"
        sens_note  = "High risk of missing pneumonia"
    st.caption(f"{sens_label} — {sens_note}")

    st.markdown("---")
    st.caption(f"🖥️ Device: `{str(device).upper()}`")
    st.caption("Model: ResNet18 + Fusion Head")
    st.caption("Dataset: Chest X-Ray (Kaggle)")


# ─────────────────────────────────────────────────────────────────────────────
# HEADER
# ─────────────────────────────────────────────────────────────────────────────
st.markdown("""
<div class="top-header">
    <span style="font-size:2.5rem;">🫁</span>
    <div>
        <h1 style="font-size:1.6rem;margin:0;color:#ffffff;">MedXAI — Explainable Medical Image Classification</h1>
        <p style="margin:2px 0 0 0;color:#5b9ec9;font-size:0.85rem;">
            Real-Time Grad-CAM · LIME · SHAP · E-Score &nbsp;|&nbsp; Pneumonia Detection from Chest X-Rays
        </p>
    </div>
</div>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────────────────────
# IMAGE UPLOAD
# ─────────────────────────────────────────────────────────────────────────────
st.markdown('<div class="section-header">📤 Upload Chest X-Ray Image</div>', unsafe_allow_html=True)

uploaded = st.file_uploader(
    "Drag & drop or click to upload",
    type=["jpg", "jpeg", "png"],
    help="Upload a chest X-ray image (JPEG or PNG). Kaggle dataset: chest-xray-pneumonia",
    label_visibility="collapsed",
)

# ─────────────────────────────────────────────────────────────────────────────
# HOW TO USE — shown when no image uploaded
# ─────────────────────────────────────────────────────────────────────────────
if uploaded is None:
    st.markdown("""
    <div class="xai-card" style="margin-top:10px;">
        <div class="xai-card-title">📋 How to Use This Application</div>
        <ol style="color:#1a3a52;line-height:1.9;margin:0;padding-left:20px;">
            <li><strong>Fill in Patient Information</strong> in the left sidebar</li>
            <li><strong>Upload a Chest X-Ray</strong> image above (JPG/PNG)</li>
            <li>The model will <strong>classify</strong> the image (Normal / Pneumonia)</li>
            <li>Review <strong>Grad-CAM, LIME, SHAP</strong> explainability visualizations</li>
            <li>Check the <strong>E-Score</strong> — explainability quality metric</li>
            <li>Validate AI results using the <strong>clinical feedback checkboxes</strong></li>
            <li>Click <strong>Save & Submit</strong> to record the case</li>
        </ol>
    </div>
    """, unsafe_allow_html=True)
    st.stop()

# ─────────────────────────────────────────────────────────────────────────────
# LOAD MODEL
# ─────────────────────────────────────────────────────────────────────────────
with st.spinner("Loading model..."):
    model = load_model()

# ─────────────────────────────────────────────────────────────────────────────
# IMAGE DISPLAY + PREPROCESSING
# ─────────────────────────────────────────────────────────────────────────────
image = Image.open(uploaded).convert("RGB")
transform = get_transform()

# Preprocess
img_tensor = transform(image).unsqueeze(0).to(device)

# ─────────────────────────────────────────────────────────────────────────────
# IMAGE QUALITY & OUTLIER CHECK  (runs BEFORE prediction)
# ─────────────────────────────────────────────────────────────────────────────
st.markdown('<div class="section-header">🔍 Image Quality & Outlier Detection</div>', unsafe_allow_html=True)

with st.spinner("Analysing image quality..."):
    quality_report = check_image_quality(image)
    quality_annotated = annotate_quality_issues(image, quality_report)

# Quality banner
qc_colors = {"Good": "#1a7f4b", "Acceptable": "#e67e22", "Poor": "#c0392b", "Unusable": "#6c3483"}
qc_color  = qc_colors.get(quality_report.overall_quality, "#555")
qc_icons  = {"Good": "✅", "Acceptable": "⚠️", "Poor": "❌", "Unusable": "🚫"}
qc_icon   = qc_icons.get(quality_report.overall_quality, "❓")

st.markdown(f"""
<div style="background:{qc_color};padding:10px 18px;border-radius:8px;margin-bottom:12px;
            display:flex;align-items:center;justify-content:space-between;flex-wrap:wrap;gap:8px;">
    <span style="color:white;font-weight:600;font-size:0.95rem;">
        {qc_icon}  Image Quality: {quality_report.overall_quality}
        &nbsp;|&nbsp; Score: {quality_report.quality_score:.0%}
        &nbsp;|&nbsp; Issues: {len(quality_report.issues)}
    </span>
    <span style="color:white;font-size:0.88rem;">
        {"✓ AI result reliable" if quality_report.is_reliable else "⚠ AI result may be UNRELIABLE — verify manually"}
    </span>
</div>
""", unsafe_allow_html=True)

if not quality_report.is_reliable:
    st.markdown("""
    <div style="background:#fff0f0;border:2px solid #e53935;border-radius:8px;
                padding:14px 18px;margin-bottom:12px;">
        <div style="color:#e74c3c;font-weight:700;font-size:1rem;margin-bottom:6px;">
            🚨 WARNING: AI Result May Be INCORRECT Due to Image Quality Issues
        </div>
        <div style="color:#b71c1c;font-size:0.88rem;line-height:1.6;">
            Critical image quality issues were detected. The model prediction and all
            explainability outputs may be wrong. Fix the issues below and re-upload
            a clean X-ray before relying on any AI findings.
        </div>
    </div>
    """, unsafe_allow_html=True)

# Side-by-side: original + annotated quality image
qc_col1, qc_col2 = st.columns(2)
with qc_col1:
    st.image(image, caption="Original uploaded image", use_container_width=True)
with qc_col2:
    st.image(quality_annotated, caption="Annotated: quality issues highlighted", use_container_width=True)

# Gauge + issues list
gauge_col, issues_col = st.columns([1, 2])
with gauge_col:
    gauge_fig = quality_gauge_figure(quality_report.quality_score)
    st.pyplot(gauge_fig, clear_figure=True)
    plt.close(gauge_fig)

with issues_col:
    st.markdown("#### 🔎 Detected Issues")
    if not quality_report.issues:
        st.markdown("""
        <div style="padding:10px 14px;background:rgba(27,127,75,0.15);border-left:3px solid #28a745;
                    border-radius:4px;color:#1a3a52;">
            ✅ No issues detected. Image is clean and suitable for AI analysis.
        </div>
        """, unsafe_allow_html=True)
    else:
        sev_colors = {"Critical": "#e74c3c", "Warning": "#e67e22", "Info": "#17a2b8"}
        sev_icons  = {"Critical": "🔴", "Warning": "🟠", "Info": "🔵"}
        for issue in quality_report.issues:
            c  = sev_colors.get(issue.severity, "#aaa")
            ic = sev_icons.get(issue.severity, "⚪")
            region_txt = (f"&nbsp; Region: ({issue.region[0]},{issue.region[1]}) {issue.region[2]}×{issue.region[3]}px"
                          if issue.region else "")
            value_txt  = f"&nbsp; Measured: {issue.value:.2f}" if issue.value else ""
            st.markdown(f"""
            <div style="border-left:3px solid {c};padding:8px 12px;margin:5px 0;
                        background:#f0f8ff;border-radius:4px;">
                <div style="display:flex;align-items:center;gap:8px;margin-bottom:3px;flex-wrap:wrap;">
                    <span>{ic}</span>
                    <span style="color:{c};font-weight:700;">{issue.issue_type}</span>
                    <span style="background:{c};color:white;font-size:0.72rem;padding:1px 8px;
                                 border-radius:10px;">{issue.severity}</span>
                    <span style="color:#5b9ec9;font-size:0.78rem;font-family:monospace;">{region_txt}{value_txt}</span>
                </div>
                <div style="color:#5b9ec9;font-size:0.83rem;">{issue.description}</div>
            </div>
            """, unsafe_allow_html=True)

# How to fix
if quality_report.issues:
    with st.expander("🛠️ How to Fix These Issues", expanded=False):
        fix_map = {
            "Severe Blur":            "Patient moved during exposure. Ask patient to hold completely still and hold breath. Use shorter exposure time.",
            "Mild Blur":              "Slight motion. Ask patient to hold breath. Verify X-ray machine focus and collimation alignment.",
            "Underexposed":           "Increase kVp (kilovoltage) or mAs settings on the X-ray machine. Check patient positioning.",
            "Overexposed":            "Reduce kVp or mAs. Check detector calibration. Verify patient body habitus settings.",
            "Low Contrast":           "Adjust window/level settings. Ensure correct exposure parameters. Use post-processing contrast enhancement.",
            "High Noise":             "Increase mAs slightly for better signal. Use noise-reduction post-processing. Clean the detector plate.",
            "Bright Artifact / Spot": "Remove all jewellery, piercings and metallic objects before imaging. Clean the X-ray detector plate.",
            "Dark Spot / Dent":       "Inspect cassette/detector for physical damage or dents. Clean with approved screen cleaner. Replace if damaged.",
            "Linear Artifacts / Scratches": "Check detector/screen for scratches. Verify anti-scatter grid alignment. Check image file for compression issues.",
            "Edge Obstruction (Top edge)":    "Reposition patient. Ensure arms and shoulders are clear. Check lead apron placement.",
            "Edge Obstruction (Bottom edge)": "Reposition patient lower. Ensure clothing or equipment is not in the beam path.",
            "Edge Obstruction (Left edge)":   "Reposition patient — possible arm or lateral clothing obstruction.",
            "Edge Obstruction (Right edge)":  "Reposition patient — possible arm or lateral clothing obstruction.",
            "Central Obstruction":    "Check nothing is blocking the chest area. Verify beam is not occluded by equipment.",
        }
        for issue in quality_report.issues:
            key = issue.issue_type
            fix = fix_map.get(key, "Retake the X-ray under controlled diagnostic conditions with a calibrated machine.")
            icon = "🔴" if issue.severity == "Critical" else "🟠"
            st.markdown(f"{icon} **{issue.issue_type}:** {fix}")

st.markdown("---")

col_img, col_info = st.columns([1, 1.6])
with col_img:
    st.markdown('<div class="section-header">🩻 Input Image</div>', unsafe_allow_html=True)
    st.image(image, use_container_width=True, caption=f"Uploaded: {uploaded.name}")
    st.caption(f"Original size: {image.size[0]} × {image.size[1]} px")

with col_info:
    # ── PREDICTION ────────────────────────────────────────────────────────
    st.markdown('<div class="section-header">🩺 Model Prediction</div>', unsafe_allow_html=True)

    with st.spinner("Running inference..."):
        pred_class, confidence, label, all_probs = cached_prediction(model, img_tensor, threshold=pneumonia_threshold)

    badge_class = "pred-badge-pneumonia" if label == "Pneumonia" else "pred-badge-normal"
    icon = "⚠️" if label == "Pneumonia" else "✅"
    st.markdown(f'<div class="{badge_class}">{icon} {label}</div>', unsafe_allow_html=True)
    st.markdown(f'<div class="score-chip">Confidence: {confidence:.1%}</div>', unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # Probability bar chart
    fig_prob, ax_prob = plt.subplots(figsize=(5.5, 2.2), facecolor="white")
    colors = ["#27ae60", "#e74c3c"]
    bars = ax_prob.barh(["Normal", "Pneumonia"], all_probs, color=colors, height=0.4)
    ax_prob.set_xlim(0, 1)
    ax_prob.set_facecolor("white")
    ax_prob.tick_params(colors='#1a4a6e', labelsize=10)
    for spine in ax_prob.spines.values():
        spine.set_edgecolor("#c8e6f8")
    for bar, prob in zip(bars, all_probs):
        ax_prob.text(bar.get_width() + 0.01, bar.get_y() + bar.get_height() / 2,
                     f"{prob:.3f}", va="center", ha="left", color="#333333", fontsize=10)
    ax_prob.set_title("Class Probabilities", color="#2c6e9e", fontsize=10, pad=8)
    plt.tight_layout()
    st.pyplot(fig_prob, clear_figure=True)
    plt.close(fig_prob)

    # Patient summary
    if patient_name or patient_id:
        st.markdown(f"""
        <div class="xai-card" style="padding:12px 16px;margin-top:4px;">
            <div style="display:flex;gap:24px;flex-wrap:wrap;font-size:0.85rem;color:#2c6e9e;">
                <span>👤 <strong style='color:#1a3a52'>{patient_name or "—"}</strong></span>
                <span>🆔 {patient_id or "—"}</span>
                <span>🎂 {patient_age} yrs</span>
                <span>⚧ {patient_sex}</span>
            </div>
        </div>
        """, unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────────────────────
# EXPLAINABILITY SECTION
# ─────────────────────────────────────────────────────────────────────────────
st.markdown("---")
st.markdown('<div class="section-header">🔬 Explainability Analysis</div>', unsafe_allow_html=True)

tab_gradcam, tab_lime, tab_shap = st.tabs(["🔥 Grad-CAM", "🧩 LIME", "📊 SHAP"])

# ── Tab 1: Grad-CAM ─────────────────────────────────────────────────────────
with tab_gradcam:
    st.markdown("""
    <div class="xai-card">
        <div class="xai-card-title">Gradient-weighted Class Activation Mapping (Grad-CAM)</div>
        <p style="color:#2c6e9e;font-size:0.85rem;margin:0;">
        Highlights the most discriminative regions of the X-ray for the model's decision.
        Red regions = high activation; Blue = low activation.
        </p>
    </div>
    """, unsafe_allow_html=True)
    prog_ph = st.empty()
    with st.spinner("Generating Grad-CAM..."):
        progress_bar(prog_ph, "Computing Grad-CAM heatmap...")
        try:
            gradcam_img, gradcam_score = cached_gradcam(model, img_tensor, image)
            st.image(gradcam_img, use_container_width=True)
            st.markdown(f'<div class="score-chip">Grad-CAM Activation Score: {gradcam_score:.4f}</div>', unsafe_allow_html=True)
        except Exception as e:
            st.error(f"Grad-CAM failed: {e}")
            gradcam_score = 0.0

# ── Tab 2: LIME ──────────────────────────────────────────────────────────────
with tab_lime:
    st.markdown("""
    <div class="xai-card">
        <div class="xai-card-title">Local Interpretable Model-agnostic Explanations (LIME)</div>
        <p style="color:#2c6e9e;font-size:0.85rem;margin:0;">
        Perturbs the input image and trains a local surrogate model to explain the prediction.
        Yellow boundaries = superpixels contributing most to the decision.
        </p>
    </div>
    """, unsafe_allow_html=True)
    prog_ph2 = st.empty()
    with st.spinner("Running LIME (this takes ~20 seconds)..."):
        progress_bar(prog_ph2, "Running LIME perturbations...", steps=8, delay=0.2)
        try:
            lime_img, lime_score = cached_lime(model, np.array(image), transform)
            st.image(lime_img, use_container_width=True)
            st.markdown(f'<div class="score-chip">LIME Contribution Score: {lime_score:.4f}</div>', unsafe_allow_html=True)
        except Exception as e:
            st.error(f"LIME failed: {e}")
            lime_score = 0.0

# ── Tab 3: SHAP ──────────────────────────────────────────────────────────────
with tab_shap:
    st.markdown("""
    <div class="xai-card">
        <div class="xai-card-title">SHapley Additive exPlanations (SHAP)</div>
        <p style="color:#2c6e9e;font-size:0.85rem;margin:0;">
        Uses game-theoretic Shapley values to attribute each pixel's contribution.
        Red pixels increase prediction confidence; Blue pixels decrease it.
        </p>
    </div>
    """, unsafe_allow_html=True)
    prog_ph3 = st.empty()
    with st.spinner("Computing SHAP values..."):
        progress_bar(prog_ph3, "Computing SHAP values...", steps=6, delay=0.15)
        try:
            shap_bytes, shap_score = cached_shap(model, img_tensor)
            if shap_bytes is not None:
                st.image(shap_bytes, use_container_width=True)
            st.markdown(f'<div class="score-chip">SHAP Mean Abs. Score: {shap_score:.4f}</div>', unsafe_allow_html=True)
        except Exception as e:
            st.error(f"SHAP failed: {e}")
            shap_score = 0.0

# ─────────────────────────────────────────────────────────────────────────────
# E-SCORE
# ─────────────────────────────────────────────────────────────────────────────
st.markdown("---")
st.markdown('<div class="section-header">📏 E-Score — Explainability Quality Metric</div>', unsafe_allow_html=True)

with st.spinner("Computing E-Score..."):
    try:
        escore_value = cached_escore(model, img_tensor, label)
    except Exception as e:
        st.error(f"E-Score error: {e}")
        escore_value = 0.0

col_e1, col_e2, col_e3, col_e4 = st.columns(4)
with col_e1:
    st.metric("E-Score", f"{escore_value:.4f}", help="Combined explainability quality score (0–1)")
with col_e2:
    st.metric("Confidence", f"{confidence:.1%}")
with col_e3:
    st.metric("Predicted Class", label)
with col_e4:
    quality = "Excellent" if escore_value > 0.75 else ("Good" if escore_value > 0.55 else "Fair")
    st.metric("Explanation Quality", quality)

st.markdown("""
<div class="xai-card" style="margin-top:10px;">
    <div class="xai-card-title">E-Score Formula</div>
    <code style="color:#0288d1;font-size:0.9rem;">
    E = 0.40 × Confidence + 0.30 × Focus Score + 0.30 × Gradient Score
    </code>
    <p style="color:#2c6e9e;font-size:0.82rem;margin:8px 0 0 0;">
    Focus Score = inverse entropy of last-layer activations &nbsp;|&nbsp;
    Gradient Score = normalized L₂ norm of input gradients
    </p>
</div>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────────────────────
# ANNOTATED X-RAY — Abnormal Region Detection
# ─────────────────────────────────────────────────────────────────────────────
st.markdown("---")
st.markdown('<div class="section-header">🔎 X-Ray Abnormality Detection & Annotation</div>', unsafe_allow_html=True)

st.markdown("""
<div class="xai-card" style="margin-bottom:10px;">
    <div class="xai-card-title">AI-Annotated Chest X-Ray</div>
    <p style="color:#2c6e9e;font-size:0.85rem;margin:0;">
    Lung zones are automatically analysed using Grad-CAM activation maps.
    Abnormal regions are highlighted with coloured bounding boxes.
    <b style="color:#e74c3c;">Red = Severe</b> &nbsp;·&nbsp;
    <b style="color:#e67e22;">Orange = Moderate</b> &nbsp;·&nbsp;
    <b style="color:#f39c12;">Yellow = Mild</b>
    </p>
</div>
""", unsafe_allow_html=True)

with st.spinner("Detecting abnormal regions and annotating X-ray..."):
    try:
        annotated_img, findings = annotate_xray(
            model, img_tensor, image, label, confidence
        )
    except Exception as e:
        st.error(f"Annotation failed: {e}")
        annotated_img = image
        findings = []

ann_col1, ann_col2 = st.columns([1.2, 1])
with ann_col1:
    st.image(annotated_img, caption="AI-Annotated X-Ray", use_container_width=True)

with ann_col2:
    st.markdown("#### 📋 Detected Findings")
    if findings:
        for f in sorted(findings, key=lambda x: x["activation"], reverse=True):
            sev = f["severity"]
            colors_map = {"Severe": "#e74c3c", "Moderate": "#e67e22", "Mild": "#f39c12", "Normal": "#27ae60"}
            c = colors_map.get(sev, "#6c757d")
            st.markdown(f"""
            <div style="border-left:3px solid {c};padding:6px 10px;margin:5px 0;
                        background:#f0f8ff;border-radius:4px;">
                <span style="color:{c};font-weight:700;">{sev}</span>
                <span style="color:#1a3a52;font-size:0.9rem;"> — {f['zone']}</span><br>
                <span style="color:#5b9ec9;font-size:0.8rem;font-family:monospace;">
                Max activation: {f['activation']:.4f} | Mean: {f['mean_act']:.4f}
                </span>
            </div>
            """, unsafe_allow_html=True)
    else:
        if label == "Normal":
            st.markdown("""
            <div style="border-left:3px solid #28a745;padding:10px;background:#f0f8ff;border-radius:4px;">
                <span style="color:#27ae60;font-weight:700;">✅ No abnormalities detected</span><br>
                <span style="color:#2c6e9e;font-size:0.85rem;">Lung fields appear clear in all zones.</span>
            </div>
            """, unsafe_allow_html=True)
        else:
            st.info("No zones exceeded detection threshold. Check Grad-CAM tab for activation details.")

    # Dominant severity for treatment
    dominant_severity = get_dominant_severity(findings) if findings else ("Moderate" if label == "Pneumonia" else "Normal")
    st.markdown(f"""
    <div style="margin-top:12px;padding:8px 12px;background:#e1f3fc;border-radius:8px;border:1px solid #c8e6f8;">
        <span style="color:#2c6e9e;font-size:0.82rem;">Dominant Severity:</span>
        <span style="font-weight:700;margin-left:8px;color:#0288d1;">{dominant_severity}</span>
        <span style="color:#2c6e9e;font-size:0.82rem;margin-left:16px;">Zones flagged:</span>
        <span style="font-weight:700;margin-left:8px;color:#0288d1;">{len(findings)}/6</span>
    </div>
    """, unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────────────────────
# TREATMENT PROTOCOL
# ─────────────────────────────────────────────────────────────────────────────
st.markdown("---")
st.markdown('<div class="section-header">💊 Treatment Protocol & Clinical Management</div>', unsafe_allow_html=True)

treatment_plan = get_treatment_plan(label, dominant_severity)

badge_color = "#c0392b" if label == "Pneumonia" else "#1a7f4b"
st.markdown(f"""
<div style="background:{badge_color};padding:10px 18px;border-radius:8px;margin-bottom:14px;display:inline-block;">
    <span style="color:white;font-weight:600;font-size:0.95rem;">
        {treatment_plan.condition} — {treatment_plan.severity}
    </span>
    <span style="color:white;font-size:0.88rem;margin-left:14px;">
        {treatment_plan.prognosis}
    </span>
</div>
""", unsafe_allow_html=True)

tx_tab1, tx_tab2, tx_tab3, tx_tab4, tx_tab5 = st.tabs([
    "⚡ Immediate", "💊 Medications", "🔬 Investigations", "🌿 Lifestyle", "📅 Follow-Up & Red Flags"
])

with tx_tab1:
    st.markdown("#### Immediate Actions Required")
    for action in treatment_plan.immediate_actions:
        priority_color = "#e74c3c" if "URGENT" in action.upper() or "ICU" in action.upper() else "#5cb8f0"
        st.markdown(f"""
        <div style="display:flex;align-items:flex-start;gap:8px;margin:6px 0;
                    padding:8px 12px;background:#f0f8ff;border-radius:6px;
                    border-left:3px solid {priority_color};">
            <span style="color:{priority_color};">▶</span>
            <span style="color:#1a3a52;font-size:0.9rem;">{action}</span>
        </div>
        """, unsafe_allow_html=True)

with tx_tab2:
    st.markdown("#### First-Line Medications")
    for med in treatment_plan.medications:
        st.markdown(f"""
        <div class="xai-card" style="padding:10px 14px;margin:6px 0;">
            <div style="display:flex;justify-content:space-between;align-items:center;flex-wrap:wrap;gap:8px;">
                <span style="color:#0288d1;font-weight:700;font-size:0.95rem;">💊 {med['name']}</span>
                <span style="background:#e1f3fc;padding:3px 10px;border-radius:12px;
                             color:#0277bd;font-family:monospace;font-size:0.82rem;">{med['dose']}</span>
                <span style="color:#5b9ec9;font-size:0.82rem;">⏱ {med['duration']}</span>
            </div>
        </div>
        """, unsafe_allow_html=True)
    st.caption("⚠️ Prescriptions must be issued by a qualified clinician. Do not self-medicate.")

with tx_tab3:
    st.markdown("#### Investigations to Order")
    inv_col1, inv_col2 = st.columns(2)
    half = len(treatment_plan.investigations) // 2
    with inv_col1:
        for inv in treatment_plan.investigations[:half+1]:
            st.markdown(f"🔬 {inv}")
    with inv_col2:
        for inv in treatment_plan.investigations[half+1:]:
            st.markdown(f"🔬 {inv}")

with tx_tab4:
    st.markdown("#### Lifestyle & Supportive Care")
    for item in treatment_plan.lifestyle:
        st.markdown(f"""
        <div style="padding:6px 12px;margin:4px 0;background:#f0fff4;
                    border-left:2px solid #43a047;border-radius:4px;color:#1a3a52;font-size:0.88rem;">
            🌿 {item}
        </div>
        """, unsafe_allow_html=True)

with tx_tab5:
    fu_col, rf_col = st.columns(2)
    with fu_col:
        st.markdown("#### 📅 Follow-Up Plan")
        for item in treatment_plan.follow_up:
            st.markdown(f"📌 {item}")
    with rf_col:
        st.markdown("#### 🚨 Red Flags — Escalate Immediately")
        for item in treatment_plan.red_flags:
            st.markdown(f"""
            <div style="padding:5px 10px;margin:4px 0;background:#fff0f0;
                        border-left:2px solid #e53935;border-radius:4px;
                        color:#b71c1c;font-size:0.85rem;">
                ⚠️ {item}
            </div>
            """, unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────────────────────
# DOCTOR FEEDBACK
# ─────────────────────────────────────────────────────────────────────────────
st.markdown("---")
st.markdown('<div class="section-header">👨‍⚕️ Clinical Summary Card</div>', unsafe_allow_html=True)

try:
    feedback_img, feedback_text = doctor_feedback(label)
    col_fb1, col_fb2 = st.columns([1, 2])
    with col_fb1:
        st.image(feedback_img, use_container_width=True)
    with col_fb2:
        for line in feedback_text.split("\n")[1:]:
            if line.strip():
                st.markdown(f"• {line}", unsafe_allow_html=True)
except Exception:
    st.info("Clinical feedback unavailable.")

# ─────────────────────────────────────────────────────────────────────────────
# CLINICAL VALIDATION
# ─────────────────────────────────────────────────────────────────────────────
st.markdown("---")
st.markdown('<div class="section-header">✅ Clinical Validation Checklist</div>', unsafe_allow_html=True)

st.markdown("""
<div class="xai-card" style="margin-bottom:12px;">
    <div class="xai-card-title">Radiologist / Clinician Review</div>
    <p style="color:#2c6e9e;font-size:0.85rem;margin:0;">
    Review the AI-generated explanations and confirm whether they align with clinical knowledge.
    </p>
</div>
""", unsafe_allow_html=True)

val_col1, val_col2 = st.columns(2)
with val_col1:
    v1 = st.checkbox("Grad-CAM highlights clinically relevant lung regions", key="val_gradcam")
    v2 = st.checkbox("LIME superpixels correspond to pathological areas", key="val_lime")
    v3 = st.checkbox("Annotated zones match visible X-ray abnormalities", key="val_annotation")
with val_col2:
    v4 = st.checkbox("SHAP top features are medically meaningful", key="val_shap")
    v5 = st.checkbox("Treatment protocol appropriate for this case", key="val_treatment")
    v6 = st.checkbox("AI explanation supports (not replaces) clinical diagnosis", key="val_support")

clinician_notes = st.text_area(
    "Clinician Notes / Override",
    placeholder="Enter clinical observations, disagreements, or additional context here...",
    height=90,
)

# ─────────────────────────────────────────────────────────────────────────────
# SAVE & SUBMIT + PDF DOWNLOAD
# ─────────────────────────────────────────────────────────────────────────────
st.markdown("---")

save_col, pdf_col, _ = st.columns([1, 1, 2])
with save_col:
    save_btn = st.button("💾 Save & Submit Analysis", use_container_width=True)
with pdf_col:
    gen_pdf_btn = st.button("📄 Generate PDF Report", use_container_width=True)

if save_btn:
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    os.makedirs("records", exist_ok=True)

    record_row = [
        timestamp, patient_id, patient_name, patient_age, patient_sex,
        ward, disease_present, major_surgeries,
        diabetes, bp, thyroid, cholesterol, asthma, copd,
        uploaded.name, label, f"{confidence:.4f}", f"{escore_value:.4f}",
        dominant_severity, len(findings),
    ]
    with open("records/patient_records.csv", "a", newline="") as f:
        csv.writer(f).writerow(record_row)

    val_row = [
        timestamp, patient_id, label, v1, v2, v3, v4, v5, v6, clinician_notes,
    ]
    with open("records/validation_feedback.csv", "a", newline="") as f:
        csv.writer(f).writerow(val_row)

    st.success(f"""
    ✅ Analysis saved!  **Patient:** {patient_name or patient_id or "Unknown"}  |
    **Prediction:** {label} ({confidence:.1%})  |  **Severity:** {dominant_severity}  |
    **E-Score:** {escore_value:.4f}  |  **Zones flagged:** {len(findings)}
    """)

if gen_pdf_btn:
    with st.spinner("Generating PDF report..."):
        try:
            patient_info = {
                "id":   patient_id or "—",
                "name": patient_name or "—",
                "age":  patient_age,
                "sex":  patient_sex,
                "ward": ward or "—",
                "comorbidities": ", ".join(filter(None, [
                    "Diabetes" if diabetes else "",
                    "Hypertension" if bp else "",
                    "Thyroid" if thyroid else "",
                    "Dyslipidemia" if cholesterol else "",
                    "Asthma" if asthma else "",
                    "COPD" if copd else "",
                ])) or "None",
            }
            xai_scores = {
                "gradcam": gradcam_score if "gradcam_score" in dir() else 0.0,
                "lime":    lime_score    if "lime_score"    in dir() else 0.0,
                "shap":    shap_score    if "shap_score"    in dir() else 0.0,
            }
            pdf_bytes = generate_pdf_report(
                patient_info=patient_info,
                label=label,
                confidence=confidence,
                escore=escore_value,
                findings=findings,
                treatment_plan=treatment_plan,
                annotated_img=annotated_img,
                xai_scores=xai_scores,
            )
            fname = f"MedXAI_Report_{patient_id or 'PT'}_{datetime.now().strftime('%Y%m%d_%H%M')}.pdf"
            st.download_button(
                label="⬇️ Download PDF Report",
                data=pdf_bytes,
                file_name=fname,
                mime="application/pdf",
                use_container_width=True,
            )
            st.success("✅ PDF report ready! Click above to download.")
        except Exception as e:
            st.error(f"PDF generation failed: {e}")
            st.info("Make sure `reportlab` is installed: `pip install reportlab`")

# ─────────────────────────────────────────────────────────────────────────────
# FOOTER
# ─────────────────────────────────────────────────────────────────────────────
st.markdown("---")
st.markdown("""
<div style="text-align:center;color:#5b9ec9;font-size:0.78rem;padding:8px 0;">
    MedXAI — MSc/PhD Research Project &nbsp;·&nbsp;
    ResNet18 · Grad-CAM · LIME · SHAP · E-Score · PDF Reports &nbsp;·&nbsp;
    Dataset: Chest X-Ray Pneumonia (Kaggle) &nbsp;·&nbsp;
    ⚠️ For Research Use Only — Not a Medical Device
</div>
""", unsafe_allow_html=True)