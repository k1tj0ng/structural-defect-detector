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
    layout="centered",
)


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


# ── UI ──────────────────────────────────────────────────────────────────────

st.title("🏗️ Structural Defect Detector")
st.caption(
    "Upload a photo of a **deck, pavement, or wall**. "
    "The model classifies the structure type then detects whether a defect (crack) is present."
)

struct_model, defect_models, thresholds = load_models()

uploaded = st.file_uploader(
    "Upload an image", type=["jpg", "jpeg", "png"], label_visibility="collapsed"
)

if uploaded is None:
    st.info("Upload an image to get started.")
    st.stop()

img_pil = Image.open(uploaded)
orig_arr, img_tensor = preprocess(img_pil)

col_img, col_results = st.columns([1, 1], gap="large")

with col_img:
    st.image(img_pil, caption="Uploaded image", use_container_width=True)

with col_results:
    with st.spinner("Running…"):
        # Stage 1 — structure
        struct_probs = struct_model.predict(img_tensor, verbose=0)[0]
        struct_idx   = int(np.argmax(struct_probs))
        struct_name  = STRUCTURES[struct_idx]
        struct_conf  = float(struct_probs[struct_idx])

        # Stage 2 — defect
        defect_model = defect_models[struct_name]
        threshold    = thresholds[struct_name]
        defect_prob  = float(defect_model.predict(img_tensor, verbose=0)[0][0])
        is_defect    = defect_prob >= threshold

    st.subheader("Results")

    # Structure
    st.markdown("**Structure type**")
    bar_cols = st.columns(len(STRUCTURES))
    for i, (col, s) in enumerate(zip(bar_cols, STRUCTURES)):
        col.metric(s, f"{struct_probs[i]:.0%}")
    st.success(f"Predicted: **{struct_name}** ({struct_conf:.0%} confidence)")

    st.divider()

    # Defect verdict
    st.markdown("**Defect status**")
    if is_defect:
        st.error(f"⚠️ DEFECT DETECTED  ({defect_prob:.0%} probability)")
    else:
        st.success(f"✅ NO DEFECT  ({defect_prob:.0%} probability)")

    st.caption(
        f"Decision threshold: {threshold} · "
        f"Optimised for high recall (minimise missed defects)"
    )

# ── Grad-CAM ────────────────────────────────────────────────────────────────

with st.expander("🔍 Model attention heatmap (Grad-CAM)", expanded=False):
    st.caption(
        "Shows which regions the defect model focused on. "
        "**Not ground-truth localisation** — the dataset has no bounding-box annotations."
    )
    with st.spinner("Generating heatmap…"):
        overlay = gradcam_overlay(defect_model, img_tensor, orig_arr)
    if overlay is not None:
        st.image(overlay, caption="Grad-CAM overlay (red = high attention)", use_container_width=True)
    else:
        st.warning("Grad-CAM could not be generated for this model.")
