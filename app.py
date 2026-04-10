import streamlit as st
import random
import json
import os
import warnings

# Suppress visual terminal warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
warnings.filterwarnings('ignore', category=UserWarning, module='tensorflow')
warnings.filterwarnings('ignore', category=UserWarning, module='keras')

from PIL import Image
from utils.predict import load_tflite_model, predict
from utils.gradcam import generate_gradcam, get_last_conv_layer

# Constants
MODEL_PATH = "model/aksharai_model_tf.tflite"
KERAS_MODEL_PATH = "model/aksharai_final.keras"
LABELS_PATH = "data/modi_labels.json"
IDX_TO_CLASS_PATH = "data/idx_to_class.json"

st.set_page_config(page_title="AksharAI", layout="wide")

# Custom CSS for Modern UI
css = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;600;800&display=swap');

html, body, [class*="css"] {
    font-family: 'Inter', sans-serif;
}

/* Background Gradient */
.stApp {
    background: linear-gradient(135deg, #1f0b02 0%, #0d0400 100%);
    color: #f8fafc;
}

/* File Uploader Hover Styling */
[data-testid="stFileUploadDropzone"] {
    border: 2px dashed #ea580c !important;
    border-radius: 12px;
    background-color: rgba(234, 88, 12, 0.05);
    transition: all 0.3s ease;
}
[data-testid="stFileUploadDropzone"]:hover {
    background-color: rgba(234, 88, 12, 0.1);
    border-color: #f97316 !important;
}

/* Glassmorphism Prediction Card */
.prediction-card {
    background: rgba(255, 255, 255, 0.05);
    backdrop-filter: blur(12px);
    border: 1px solid rgba(255, 255, 255, 0.1);
    border-radius: 16px;
    padding: 2.5rem;
    text-align: center;
    box-shadow: 0 10px 15px -3px rgba(0, 0, 0, 0.2), 0 4px 6px -2px rgba(0, 0, 0, 0.1);
    margin: 2rem 0;
}
@media (max-width: 768px) {
    .prediction-label {
        font-size: 3rem;
    }
}
.prediction-title {
    font-size: 1rem;
    color: #94a3b8;
    text-transform: uppercase;
    letter-spacing: 0.15em;
    margin-bottom: 0.5rem;
    font-weight: 600;
}
.prediction-label {
    font-size: 5rem;
    font-weight: 800;
    background: linear-gradient(to right, #fed7aa, #f97316, #ea580c);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    margin: 0;
    line-height: 1.6;
}

/* Custom Progress Bars */
.conf-container {
    margin-bottom: 1.2rem;
}
.conf-header {
    display: flex;
    justify-content: space-between;
    margin-bottom: 0.4rem;
    font-weight: 600;
    color: #e2e8f0;
    font-size: 1.05rem;
}
.conf-bar-bg {
    background-color: rgba(255, 255, 255, 0.1);
    border-radius: 999px;
    height: 12px;
    width: 100%;
    overflow: hidden;
    box-shadow: inset 0 2px 4px rgba(0, 0, 0, 0.2);
}
.conf-bar-fill {
    background: linear-gradient(90deg, #fda4af, #ea580c);
    height: 100%;
    border-radius: 999px;
    transition: width 0.8s ease-in-out;
}
/* Character Grid Library */
.library-section {
    margin-bottom: 2rem;
}
.library-title {
    font-size: 1.5rem;
    font-weight: 600;
    color: #f8fafc;
    margin-bottom: 1rem;
    border-bottom: 2px solid rgba(255,255,255,0.1);
    padding-bottom: 0.5rem;
}
.char-grid {
    display: grid;
    grid-template-columns: repeat(6, 1fr);
    gap: 12px;
}
@media (max-width: 1024px) {
    .char-grid { grid-template-columns: repeat(4, 1fr); }
}
@media (max-width: 600px) {
    .char-grid { grid-template-columns: repeat(3, 1fr); }
}

.char-card-ui {
    background: rgba(255, 255, 255, 0.05);
    border: 1px solid rgba(255, 255, 255, 0.1);
    border-radius: 12px;
    padding: 1.2rem;
    text-align: center;
    position: relative;
    overflow: hidden;
    min-height: 160px;
    display: flex;
    flex-direction: column;
    justify-content: center;
    align-items: center;
    cursor: pointer;
    transition: all 0.4s cubic-bezier(0.25, 0.8, 0.25, 1);
}
.char-card-ui:hover {
    background: rgba(255, 255, 255, 0.1);
    transform: translateY(-5px);
    box-shadow: 0 10px 20px -3px rgba(0, 0, 0, 0.4);
    border-color: rgba(249, 115, 22, 0.5);
}

.char-main-txt {
    font-size: 3.5rem;
    font-weight: 800;
    color: #fb923c;
    margin: 0;
    position: absolute; 
    top: 50%;
    left: 50%;
    transform: translate(-50%, -50%) scale(1);
    transition: all 0.4s cubic-bezier(0.25, 0.8, 0.25, 1);
    line-height: 1;
    transform-origin: center center;
}

.char-card-ui:hover .char-main-txt {
    top: 25px;
    transform: translate(-50%, -50%) scale(0.45); 
    color: #ea580c;
}

.char-details-ui {
    position: absolute;
    top: 55px; 
    left: 0;
    width: 100%;
    opacity: 0;
    padding: 0 12px;
    box-sizing: border-box;
    transform: translateY(15px);
    transition: all 0.4s cubic-bezier(0.25, 0.8, 0.25, 1);
    display: flex;
    flex-direction: column;
    align-items: center;
    pointer-events: none;
}

.char-card-ui:hover .char-details-ui {
    opacity: 1;
    transform: translateY(0);
}

.cd-name { font-weight: bold; color: #f8fafc; margin-bottom: 2px; font-size: 0.9rem; }
.cd-pron { font-style: italic; color: #94a3b8; font-size: 0.75rem; margin-bottom: 6px; }
.cd-ex { color: #fb923c; font-weight: 600; font-size: 0.8rem; margin-bottom: 4px; }
.cd-note { font-size: 0.65rem; color: #64748b; line-height: 1.3; }
</style>
"""
st.markdown(css, unsafe_allow_html=True)

st.markdown("<h1 style='text-align: center; font-weight: 800; font-size: 3rem;'>AksharAI <span style='color: #ea580c;'>✨</span></h1>", unsafe_allow_html=True)

tab1, tab2, tab3, tab4 = st.tabs([
    "⌕ Recognize",
    "✎ Learn",
    "◉ Quiz",
    "⊞ Library"
])



@st.cache_resource
def initialize_system():
    if not os.path.exists(MODEL_PATH) or not os.path.exists(LABELS_PATH) or not os.path.exists(IDX_TO_CLASS_PATH):
        return None, {}, {}
    return load_tflite_model(MODEL_PATH, LABELS_PATH, IDX_TO_CLASS_PATH)

@st.cache_resource
def load_keras_model():
    if not os.path.exists(KERAS_MODEL_PATH):
        return None
    import tensorflow as tf
    try:
        return tf.keras.models.load_model(KERAS_MODEL_PATH)
    except Exception as e:
        return e

keras_model_result = load_keras_model()
if isinstance(keras_model_result, Exception):
    st.warning(f"Grad-CAM disabled: Could not load the `.keras` model ({keras_model_result}).")
    keras_model = None
else:
    keras_model = keras_model_result

interpreter, modi_labels, idx_to_class = initialize_system()

if interpreter is None:
    st.error(f"Cannot find model or data files. Please ensure `{MODEL_PATH}`, `{LABELS_PATH}`, and `{IDX_TO_CLASS_PATH}` exist.")

with tab1:
    with st.container(): 
        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown("<p style='text-align: center; color: #94a3b8; font-size: 1.15rem; margin-bottom: 2rem;'>Upload an image of a Modi script character for instant recognition.</p>", unsafe_allow_html=True)
        uploaded_file = st.file_uploader("Upload Image", type=["jpg", "jpeg", "png"])

        if uploaded_file is None:
            st.info("↑ Upload a Modi character image to begin.")

        if uploaded_file is not None:
            image = Image.open(uploaded_file)
            
            if interpreter is not None and modi_labels:
                with st.spinner("Predicting and generating Grad-CAM..."):
                    try:
                        # Prediction
                        predictions = predict(image, interpreter, modi_labels, idx_to_class)
                        
                        # Grad-CAM
                        try:
                            last_layer = get_last_conv_layer(keras_model) if keras_model else None
                            if last_layer and keras_model:
                                overlay_img = generate_gradcam(image, keras_model, last_layer)
                            else:
                                overlay_img = image # fallback 
                        except Exception as e:
                            st.warning(f"Grad-CAM could not be generated: {e}")
                            overlay_img = image
                        
                        # Display Images Side-by-Side
                        with st.container():
                            spacer1, col1, col2, spacer2 = st.columns([1, 2, 2, 1])
                            with col1:
                                st.image(image, caption="Original Image", width='stretch')
                            with col2:
                                st.image(overlay_img, caption="Grad-CAM Visualization", width='stretch')

                        # Display the best prediction in a styled glassmorphism card
                        best_pred = predictions[0]
                        label_display = best_pred['devanagari']
                        st.markdown(f"""
                        <div class="prediction-card">
                            <div class="prediction-title">Top Prediction: {best_pred['english_name']}</div><br>
                            <div class="prediction-label">{label_display}</div>
                        </div>
                        """, unsafe_allow_html=True)
                        
                        # Show top 3 predictions with gradient confidence bars
                        st.markdown("<h3 style='color: #f8fafc; font-weight: 600; margin-top: 2rem; margin-bottom: 1rem;'>Confidence Scores</h3>", unsafe_allow_html=True)
                        
                        with st.expander("≡ View Confidence Scores", expanded=True):
                            for pred in predictions:
                                # Ensure conf is a percentage
                                conf = pred['confidence'] if pred['confidence'] > 1 else pred['confidence'] * 100
                                label = f"{pred['devanagari']} ({pred['english_name']})"

                                st.markdown(f"""
                                <div class="conf-container">
                                    <div class="conf-header">
                                        <span>{label}</span>
                                        <span>{conf:.1f}%</span>
                                    </div>
                                    <div class="conf-bar-bg">
                                        <div class="conf-bar-fill" style="width: {conf}%;"></div>
                                    </div>
                                </div>
                                """, unsafe_allow_html=True)


                    except Exception as e:
                        st.error(f"Error during prediction or visualization: {e}")
            else:
                # Display the uploaded image if model/labels are missing
                st.image(image, caption="Uploaded Image", width='stretch')

with tab2:
    st.header("✎ Learn Modi Script")
    st.markdown("<br>", unsafe_allow_html=True)

    st.subheader("𑘀 Vowels")
    st.info("अ, आ, इ, ई, उ, ऊ")

    st.subheader("𑘎 Consonants")
    st.info("क, ख, ग, घ, ङ")

    st.subheader("𑙑 Numerals")
    st.info("१, २, ३, ४, ५")



with tab3:
    st.header("◉ Quiz Mode")
    st.markdown("<br>", unsafe_allow_html=True)

    if "score" not in st.session_state:
        st.session_state.score = 0

    st.metric("Score", st.session_state.score)

    if st.button("Next Question"):
        st.write("Question coming soon...")

with tab4:
    st.header("⊞ Character Library")
    st.markdown("<p style='color: #94a3b8;'>Hover over any character card to see details.</p>", unsafe_allow_html=True)

    groups = [
        ("𑙑 Numbers", [
            ['zero','one','two','three','four','five','six','seven','eight','nine']
        ]),
        ("𑘀 Vowels", [
            ['a','aa','i','ii','u','oo'],
            ['e','ai','o','ou','am','ah']
        ]),
        ("𑘎 Consonants", [
            ['k','kh','g','gh'],
            ['ch','chh','ja','jh'],
            ['t','tha','da','dha','nn'],
            ['ta','th','d','dh','n'],
            ['p','ph','b','bh','m'],
            ['y','r','l','v'],
            ['sh','s','h','lh'],
            ['ksh','dyn','shr','tr']
        ])
    ]

    html_parts = []

    for group_name, row_list in groups:
        html_parts.append(f"<div class='library-section'><div class='library-title'>{group_name}</div>")
        
        for keys in row_list:
            valid_keys = [k for k in keys if k in modi_labels]
            if not valid_keys:
                continue

            html_parts.append("<div class='char-grid' style='margin-bottom: 15px;'>")

            for k in valid_keys:
                info = modi_labels.get(k, {})
                html_parts.append(
                    f"<div class='char-card-ui'>"
                    f"<div class='char-main-txt'>{info.get('modi','')}</div>"
                    f"<div class='char-details-ui'>"
                    f"<div class='cd-name'>{info.get('devanagari', '')} ({info.get('english_name', '')})</div>"
                    f"<div class='cd-pron'>{info.get('pronunciation','')}</div>"
                    f"<div class='cd-ex'>{info.get('example_word','')}</div>"
                    f"<div class='cd-note'>{info.get('historical_note','')}</div>"
                    f"</div></div>"
                )

            html_parts.append("</div>") # End char-grid row
        html_parts.append("</div>") # End library-section

    final_html = "".join(html_parts)

    st.markdown(final_html, unsafe_allow_html=True)