import streamlit as st
import tensorflow as tf
import numpy as np
import json
from pathlib import Path
from PIL import Image
import matplotlib.cm as mpl_cm

IMG_SIZE = (224, 224)
STRUCTURES = ["Decks", "Pavements", "Walls"]
MODEL_DIR = Path(__file__).parent

st.set_page_config(
    page_title="Structural Defect Detector",
    page_icon="🏗️",
    layout="wide",
)

# ── Custom CSS ───────────────────────────────────────────────────────────────
st.markdown("""
<style>
    .verdict-defect {
        background: #ff4b4b22;
        border: 2px solid #ff4b4b;
        border-radius: 12px;
        padding: 20px;
        text-align: center;
        font-size: 1.6rem;
        font-weight: 700;
        color: #ff4b4b;
    }
    .verdict-ok {
        background: #21c35422;
        border: 2px solid #21c354;
        border-radius: 12px;
        padding: 20px;
        text-align: center;
        font-size: 1.6rem;
        font-weight: 700;
        color: #21c354;
    }
    .stage-label {
        font-size: 0.75rem;
        text-transform: uppercase;
        letter-spacing: 0.08em;
        color: #888;
        margin-bottom: 2px;
    }
</style>
""", unsafe_allow_html=True)


@st.cache_resource(show_spinner="Loading models…")
def load_models():
    missing = [
        f for f in [
            "structure_model.keras",
            "defect_model_decks.keras",
            "defect_model_pavements.keras",
            "defect_model_walls.keras",
            "thresholds.json",
        ]
        if not (MODEL_DIR / f).exists()
    ]
    if missing:
        st.error(f"Missing model files: {missing}. Run pipeline_train.ipynb first.")
        st.stop()

    struct_model = tf.keras.models.load_model(MODEL_DIR / "structure_model.keras")
    defect_models = {
        s: tf.keras.models.load_model(MODEL_DIR / f"defect_model_{s.lower()}.keras")
        for s in STRUCTURES
    }
    with open(MODEL_DIR / "thresholds.json") as f:
        thresholds = json.load(f)
    return struct_model, defect_models, thresholds


def preprocess(img_pil: Image.Image):
    img = img_pil.convert("RGB").resize((IMG_SIZE[1], IMG_SIZE[0]))
    arr = np.array(img, dtype=np.float32)
    return arr, np.expand_dims(arr, 0)


def gradcam_overlay(model, img_tensor, orig_arr):
    backbone = next(
        (l for l in model.layers if hasattr(l, "layers") and len(l.layers) > 5),
        None,
    )
    if backbone is None:
        return None
    last_conv = next(
        (l.name for l in reversed(backbone.layers) if isinstance(l, tf.keras.layers.Conv2D)),
        None,
    )
    if last_conv is None:
        return None
    try:
        grad_model = tf.keras.Model(
            inputs=model.input,
            outputs=[backbone.get_layer(last_conv).output, model.output],
        )
        with tf.GradientTape() as tape:
            conv_out, pred = grad_model(img_tensor)
            loss = pred[:, 0]
        grads = tape.gradient(loss, conv_out)
        pooled = tf.reduce_mean(grads, axis=(0, 1, 2))
        heatmap = tf.squeeze(conv_out[0] @ pooled[..., tf.newaxis])
        heatmap = tf.maximum(heatmap, 0) / (tf.reduce_max(heatmap) + 1e-8)
        heatmap = heatmap.numpy()

        h, w = orig_arr.shape[:2]
        hm_resized = tf.image.resize(heatmap[..., np.newaxis], (h, w)).numpy()[..., 0]
        colored = mpl_cm.jet(hm_resized)[..., :3]
        overlay = np.clip(0.55 * orig_arr / 255.0 + 0.45 * colored, 0, 1)
        return (overlay * 255).astype(np.uint8)
    except Exception:
        return None


def run_pipeline(img_pil: Image.Image):
    orig_arr, img_tensor = preprocess(img_pil)

    struct_probs = struct_model.predict(img_tensor, verbose=0)[0]
    struct_idx   = int(np.argmax(struct_probs))
    struct_name  = STRUCTURES[struct_idx]
    struct_conf  = float(struct_probs[struct_idx])

    defect_model = defect_models[struct_name]
    threshold    = thresholds[struct_name]
    defect_prob  = float(defect_model.predict(img_tensor, verbose=0)[0][0])
    is_defect    = defect_prob >= threshold

    return struct_probs, struct_name, struct_conf, defect_model, threshold, defect_prob, is_defect, orig_arr, img_tensor


def show_results(img_pil, struct_probs, struct_name, struct_conf,
                 defect_model, threshold, defect_prob, is_defect, orig_arr, img_tensor):

    col_img, col_pipe, col_verdict = st.columns([1.2, 1.2, 1], gap="large")

    with col_img:
        st.markdown('<p class="stage-label">Input image</p>', unsafe_allow_html=True)
        st.image(img_pil, use_container_width=True)

    with col_pipe:
        st.markdown('<p class="stage-label">Stage 1 — Structure type</p>', unsafe_allow_html=True)
        for i, s in enumerate(STRUCTURES):
            bar_val = float(struct_probs[i])
            is_top = (s == struct_name)
            label = f"**{s}**" if is_top else s
            st.markdown(label)
            st.progress(bar_val, text=f"{bar_val:.0%}")

        st.divider()

        st.markdown('<p class="stage-label">Stage 2 — Defect probability</p>', unsafe_allow_html=True)
        st.markdown(f"Model: `defect_model_{struct_name.lower()}`")
        st.progress(min(defect_prob, 1.0), text=f"{defect_prob:.0%}")
        st.caption(f"Threshold: {threshold} · recall-optimised")

    with col_verdict:
        st.markdown('<p class="stage-label">Verdict</p>', unsafe_allow_html=True)
        if is_defect:
            st.markdown(
                '<div class="verdict-defect">⚠️<br>DEFECT<br>DETECTED</div>',
                unsafe_allow_html=True,
            )
        else:
            st.markdown(
                '<div class="verdict-ok">✅<br>NO<br>DEFECT</div>',
                unsafe_allow_html=True,
            )
        st.markdown(f"<br>Predicted structure: **{struct_name}** ({struct_conf:.0%})", unsafe_allow_html=True)

    # Grad-CAM
    with st.expander("🔍 Grad-CAM attention heatmap", expanded=False):
        st.caption(
            "Shows which regions the defect model focused on. "
            "**Not ground-truth localisation** — dataset has no bounding-box annotations."
        )
        with st.spinner("Generating heatmap…"):
            overlay = gradcam_overlay(defect_model, img_tensor, orig_arr)
        if overlay is not None:
            st.image(overlay, caption="Red = high model attention", use_container_width=True)
        else:
            st.warning("Grad-CAM could not be generated for this model.")


# ── Header ───────────────────────────────────────────────────────────────────

st.title("🏗️ Structural Defect Detector")
st.caption(
    "Two-stage CNN pipeline: **Stage 1** classifies structure type (Deck / Pavement / Wall) · "
    "**Stage 2** detects defects, optimised for high recall (minimise missed cracks)."
)

struct_model, defect_models, thresholds = load_models()

# ── Input tabs ───────────────────────────────────────────────────────────────

tab_camera, tab_upload = st.tabs(["📷 Live Camera", "📁 Upload Image"])

img_pil = None

with tab_camera:
    st.info(
        "Point your camera at a **wall, deck, or pavement** surface and click **Take Photo**. "
        "The model will classify the structure type and check for cracks."
    )
    camera_image = st.camera_input("Take a photo", label_visibility="collapsed")
    if camera_image is not None:
        img_pil = Image.open(camera_image)

with tab_upload:
    uploaded = st.file_uploader(
        "Upload an image", type=["jpg", "jpeg", "png"], label_visibility="collapsed"
    )
    if uploaded is not None:
        img_pil = Image.open(uploaded)

# ── Run pipeline ─────────────────────────────────────────────────────────────

if img_pil is None:
    st.stop()

st.divider()

with st.spinner("Running pipeline…"):
    results = run_pipeline(img_pil)

show_results(img_pil, *results)
