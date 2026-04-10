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

/* Fix Selectbox Cursor */
div[data-baseweb="select"], div[data-baseweb="select"] * {
    cursor: pointer !important;
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

    if not modi_labels:
        st.warning("Data not loaded to start the quiz. Please check if `modi_labels.json` exists.")
    else:
        # Category Filter UI
        category_options = {
            "All": None,
            "Vowels": ["vowel", "vowel_modifier"],
            "Consonants": ["consonant", "compound"],
            "Numbers": ["numeral"]
        }
        
        selected_cat = st.selectbox("Select Category", list(category_options.keys()))

        # Reset state if category changes
        if "quiz_category" not in st.session_state or st.session_state.quiz_category != selected_cat:
            st.session_state.quiz_category = selected_cat
            st.session_state.score = 0
            st.session_state.total_questions = 0
            st.session_state.asked_questions = []
            if "quiz_q" in st.session_state:
                del st.session_state.quiz_q

        if "score" not in st.session_state:
            st.session_state.score = 0
            st.session_state.total_questions = 0
            st.session_state.asked_questions = []

        MAX_QUESTIONS = 10

        st.markdown("<br>", unsafe_allow_html=True)
        m1, m2, m3 = st.columns(3)
        with m1:
            st.metric("Score", st.session_state.score)
        with m2:
            accuracy = (st.session_state.score / st.session_state.total_questions * 100) if st.session_state.total_questions > 0 else 0
            st.metric("Accuracy", f"{accuracy:.1f}%")
        with m3:
            st.metric("Progress", f"{st.session_state.total_questions} / {MAX_QUESTIONS}")

        st.progress(st.session_state.total_questions / MAX_QUESTIONS if st.session_state.total_questions <= MAX_QUESTIONS else 1.0)

        # Quiz Finished State
        if st.session_state.total_questions >= MAX_QUESTIONS:
            st.success(f"🎉 Quiz Complete! Your final score is {st.session_state.score} out of {MAX_QUESTIONS}.")
            if st.button("Restart Quiz", type="primary"):
                st.session_state.score = 0
                st.session_state.total_questions = 0
                st.session_state.asked_questions = []
                if "quiz_q" in st.session_state:
                    del st.session_state.quiz_q
                st.rerun()
        else:
            if "quiz_q" not in st.session_state:
                valid_types = category_options[selected_cat]
                if valid_types is None:
                    pool = list(modi_labels.keys())
                else:
                    pool = [k for k, v in modi_labels.items() if v.get("character_type") in valid_types]
                
                # Failsafe if pool < 4
                if len(pool) < 4:
                    pool = list(modi_labels.keys())

                # Prevent repetition
                available_pool = [k for k in pool if k not in st.session_state.asked_questions]
                if not available_pool:
                    # Reset history if all questions in pool have been asked
                    st.session_state.asked_questions = []
                    available_pool = pool

                correct_key = random.choice(available_pool)
                st.session_state.asked_questions.append(correct_key)

                wrong_pool = [k for k in pool if k != correct_key]
                
                # Failsafe if wrong_pool < 3
                if len(wrong_pool) < 3:
                    wrong_pool = [k for k in modi_labels.keys() if k != correct_key]
                    
                wrong_keys = random.sample(wrong_pool, 3)
                
                options_keys = [correct_key] + wrong_keys
                random.shuffle(options_keys)

                available_q_types = ["modi_to_english", "english_to_modi"]
                # Enable audio type if corresponding audio file exists
                if os.path.exists(f"audio files/{modi_labels[correct_key]['devanagari']}.mp3"):
                    available_q_types.append("audio_to_modi")

                q_type = random.choice(available_q_types)

                st.session_state.quiz_q = {
                    "correct_key": correct_key,
                    "options": [modi_labels[k] for k in options_keys],
                    "q_type": q_type,
                    "answered": False,
                    "selected_opt": None
                }

            q = st.session_state.quiz_q
            correct_info = modi_labels[q["correct_key"]]

            st.markdown("<hr style='border-color: rgba(255,255,255,0.1); margin: 2rem 0;'>", unsafe_allow_html=True)
            
            # Progress question number display
            st.markdown(f"<p style='color: #94a3b8; font-weight: 600; font-size: 1.1rem;'>Question {st.session_state.total_questions + 1} of {MAX_QUESTIONS}</p>", unsafe_allow_html=True)

            if q["q_type"] == "modi_to_english":
                st.subheader("Which character is this?")
                st.markdown(f"<h1 style='text-align: center; font-size: 5rem; color: #fb923c;'>{correct_info['modi']}</h1>", unsafe_allow_html=True)
            elif q["q_type"] == "english_to_modi":
                st.subheader("What is the Modi script for this character?")
                st.markdown(f"<h2 style='text-align: center; font-size: 2.5rem; color: #f8fafc; margin-top: 1rem; margin-bottom: 2rem;'>{correct_info['devanagari']} ({correct_info['english_name']})</h2>", unsafe_allow_html=True)
            else:
                st.subheader("Listen to the pronunciation and identify the Modi character:")
                audio_path = f"audio files/{correct_info['devanagari']}.mp3"
                st.audio(audio_path, format="audio/mp3")
                st.markdown("<br>", unsafe_allow_html=True)

            st.markdown("<br>", unsafe_allow_html=True)

            col1, col2 = st.columns(2)
            
            for i, opt in enumerate(q["options"]):
                c = col1 if i % 2 == 0 else col2
                
                if q["q_type"] == "modi_to_english":
                    btn_label = f"{opt['devanagari']} ({opt['english_name']})"
                else:
                    btn_label = f"{opt['modi']} "

                with c:
                    if q["answered"]:
                        is_correct = (opt == correct_info)
                        if is_correct:
                            bg_color = "rgba(34, 197, 94, 0.1)"
                            border_color = "rgba(34, 197, 94, 0.5)"
                            text_color = "#4ade80"
                        elif q["selected_opt"] == opt:
                            bg_color = "rgba(239, 68, 68, 0.1)"
                            border_color = "rgba(239, 68, 68, 0.5)"
                            text_color = "#f87171"
                        else:
                            bg_color = "transparent"
                            border_color = "rgba(250, 250, 250, 0.2)"
                            text_color = "rgba(250, 250, 250, 0.5)"

                        st.markdown(f'''
                        <div style="
                            display: flex;
                            align-items: center;
                            justify-content: center;
                            font-weight: 400;
                            padding: 0.25rem 0.75rem;
                            border-radius: 0.5rem;
                            min-height: 38.4px;
                            margin: 0 0 1rem 0;
                            line-height: 1.6;
                            color: {text_color};
                            width: 100%;
                            background-color: {bg_color};
                            border: 1px solid {border_color};
                            box-sizing: border-box;
                        ">
                            {btn_label}
                        </div>
                        ''', unsafe_allow_html=True)
                    else:
                        if st.button(btn_label, key=f"opt_{i}", use_container_width=True):
                            st.session_state.quiz_q["answered"] = True
                            st.session_state.quiz_q["selected_opt"] = opt
                            st.session_state.total_questions += 1
                            if opt == correct_info:
                                st.session_state.score += 1
                            st.rerun()

            st.markdown("<br>", unsafe_allow_html=True)

            if q["answered"]:
                is_correct = (q["selected_opt"] == correct_info)
                if is_correct:
                    st.success("Correct! 🎉")
                    # Party Pops animation exactly once
                    if "balloons_shown" not in st.session_state.quiz_q:
                        import streamlit.components.v1 as components
                        import base64
                        
                        audio_tag = ""
                        try:
                            with open("audio files/PARTY_SOUND.mp3", "rb") as f:
                                audio_b64 = base64.b64encode(f.read()).decode()
                                audio_tag = f'<audio autoplay src="data:audio/mp3;base64,{audio_b64}"></audio>'
                        except:
                            pass

                        components.html(
                            audio_tag + """
                            <script src="https://cdn.jsdelivr.net/npm/canvas-confetti@1.6.0/dist/confetti.browser.min.js"></script>
                            <script>
                                try {
                                    const parentDoc = window.parent.document;
                                    const canvas = parentDoc.createElement('canvas');
                                    canvas.style.position = 'fixed';
                                    canvas.style.top = '0';
                                    canvas.style.left = '0';
                                    canvas.style.width = '100vw';
                                    canvas.style.height = '100vh';
                                    canvas.style.pointerEvents = 'none';
                                    canvas.style.zIndex = '999999';
                                    parentDoc.body.appendChild(canvas);

                                    var myConfetti = confetti.create(canvas, {
                                        resize: true,
                                        useWorker: true
                                    });

                                    myConfetti({
                                        particleCount: 150,
                                        spread: 80,
                                        origin: { y: 0.6 }
                                    });

                                    setTimeout(() => {
                                        if (parentDoc.body.contains(canvas)) {
                                            parentDoc.body.removeChild(canvas);
                                        }
                                    }, 3500);
                                } catch(e) {
                                    console.error(e);
                                }
                            </script>
                            """,
                            height=0, 
                            width=0
                        )
                        st.session_state.quiz_q["balloons_shown"] = True
                else:
                    st.error("Wrong! 😕")
                
                st.info(f"**Correct Answer:** {correct_info['modi']} - {correct_info['devanagari']} ({correct_info['english_name']})\n\n"
                        f"**Example Word:** {correct_info.get('example_word', 'N/A')}")

                if st.button("Next Question", type="primary"):
                    del st.session_state.quiz_q
                    st.rerun()

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