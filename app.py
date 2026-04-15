import streamlit as st
import random
import json
import os
import io
import hashlib
import warnings
import streamlit.components.v1 as components
import warnings
import tensorflow as tf


# Suppress visual terminal warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
warnings.filterwarnings('ignore', category=UserWarning, module='tensorflow')
warnings.filterwarnings('ignore', category=UserWarning, module='keras')
import logging
logging.getLogger("tensorflow").setLevel(logging.ERROR)

from PIL import Image
from utils.predict import load_tflite_model, predict
from utils.gradcam import generate_gradcam, get_last_conv_layer

# Constants
# MODEL_PATH = "model/aksharai_model_tf.tflite"
# LABELS_PATH = "data/modi_labels.json"
# IDX_TO_CLASS_PATH = "data/idx_to_class.json"
# KERAS_MODEL_PATH = "model/aksharai_final.keras"

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

MODEL_PATH = os.path.join(BASE_DIR, "model", "aksharai_model_tf.tflite")
LABELS_PATH = os.path.join(BASE_DIR, "data", "modi_labels.json")
IDX_TO_CLASS_PATH = os.path.join(BASE_DIR, "data", "idx_to_class.json")
KERAS_MODEL_PATH = os.path.join(BASE_DIR, "model", "aksharai_final.keras")

st.write("MODEL PATH:", MODEL_PATH)
st.write("MODEL EXISTS:", os.path.exists(MODEL_PATH))


st.set_page_config(page_title="AksharAI", layout="wide", page_icon="data:image/svg+xml,<svg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 32 32'><text y='24' font-size='24'>A</text></svg>")

# ─── SVG ICON LIBRARY (Lucide-style inline SVGs) ────────────────────────────────
# All icons are 18x18, stroke-based, no emoji anywhere
ICONS = {
    "scan": '<svg xmlns="http://www.w3.org/2000/svg" width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M3 7V5a2 2 0 0 1 2-2h2"/><path d="M17 3h2a2 2 0 0 1 2 2v2"/><path d="M21 17v2a2 2 0 0 1-2 2h-2"/><path d="M7 21H5a2 2 0 0 1-2-2v-2"/><line x1="7" x2="17" y1="12" y2="12"/></svg>',
    "book": '<svg xmlns="http://www.w3.org/2000/svg" width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M4 19.5v-15A2.5 2.5 0 0 1 6.5 2H20v20H6.5a2.5 2.5 0 0 1 0-5H20"/></svg>',
    "trophy": '<svg xmlns="http://www.w3.org/2000/svg" width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M6 9H4.5a2.5 2.5 0 0 1 0-5H6"/><path d="M18 9h1.5a2.5 2.5 0 0 0 0-5H18"/><path d="M4 22h16"/><path d="M10 14.66V17c0 .55-.47.98-.97 1.21C7.85 18.75 7 20.24 7 22"/><path d="M14 14.66V17c0 .55.47.98.97 1.21C16.15 18.75 17 20.24 17 22"/><path d="M18 2H6v7a6 6 0 0 0 12 0V2Z"/></svg>',
    "library": '<svg xmlns="http://www.w3.org/2000/svg" width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><rect width="8" height="18" x="3" y="3" rx="1"/><path d="M7 3v18"/><path d="M20.4 18.9c.2.5-.1 1.1-.6 1.3l-1.9.7c-.5.2-1.1-.1-1.3-.6L11.1 5.1c-.2-.5.1-1.1.6-1.3l1.9-.7c.5-.2 1.1.1 1.3.6Z"/></svg>',
    "upload": '<svg xmlns="http://www.w3.org/2000/svg" width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M21 15v4a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2v-4"/><polyline points="17 8 12 3 7 8"/><line x1="12" x2="12" y1="3" y2="15"/></svg>',
    "image": '<svg xmlns="http://www.w3.org/2000/svg" width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><rect width="18" height="18" x="3" y="3" rx="2" ry="2"/><circle cx="9" cy="9" r="2"/><path d="m21 15-3.086-3.086a2 2 0 0 0-2.828 0L6 21"/></svg>',
    "brain": '<svg xmlns="http://www.w3.org/2000/svg" width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M12 5a3 3 0 1 0-5.997.125 4 4 0 0 0-2.526 5.77 4 4 0 0 0 .556 6.588A4 4 0 1 0 12 18Z"/><path d="M12 5a3 3 0 1 1 5.997.125 4 4 0 0 1 2.526 5.77 4 4 0 0 1-.556 6.588A4 4 0 1 1 12 18Z"/><path d="M15 13a4.5 4.5 0 0 1-3-4 4.5 4.5 0 0 1-3 4"/><path d="M12 18v4"/></svg>',
    "bar_chart": '<svg xmlns="http://www.w3.org/2000/svg" width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><line x1="12" x2="12" y1="20" y2="10"/><line x1="18" x2="18" y1="20" y2="4"/><line x1="6" x2="6" y1="20" y2="16"/></svg>',
    "feather": '<svg xmlns="http://www.w3.org/2000/svg" width="40" height="40" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.5" stroke-linecap="round" stroke-linejoin="round"><path d="M20.24 12.24a6 6 0 0 0-8.49-8.49L5 10.5V19h8.5z"/><line x1="16" x2="2" y1="8" y2="22"/><line x1="17.5" x2="9" y1="15" y2="15"/></svg>',
    "search": '<svg xmlns="http://www.w3.org/2000/svg" width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><circle cx="11" cy="11" r="8"/><path d="m21 21-4.3-4.3"/></svg>',
    "letters": '<svg xmlns="http://www.w3.org/2000/svg" width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M21 14h-5"/><path d="M21 10h-5"/><path d="m2 14 6-6 6 6"/><path d="M5 11h6"/></svg>',
    "pen_tool": '<svg xmlns="http://www.w3.org/2000/svg" width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="m12 19 7-7 3 3-7 7-3-3z"/><path d="m18 13-1.5-7.5L2 2l3.5 14.5L13 18l5-5z"/><path d="m2 2 7.586 7.586"/><circle cx="11" cy="11" r="2"/></svg>',
    "hash": '<svg xmlns="http://www.w3.org/2000/svg" width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><line x1="4" x2="20" y1="9" y2="9"/><line x1="4" x2="20" y1="15" y2="15"/><line x1="10" x2="8" y1="3" y2="21"/><line x1="16" x2="14" y1="3" y2="21"/></svg>',
    "check_circle": '<svg xmlns="http://www.w3.org/2000/svg" width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M22 11.08V12a10 10 0 1 1-5.93-9.14"/><path d="m9 11 3 3L22 4"/></svg>',
    "x_circle": '<svg xmlns="http://www.w3.org/2000/svg" width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><circle cx="12" cy="12" r="10"/><path d="m15 9-6 6"/><path d="m9 9 6 6"/></svg>',
    "headphones": '<svg xmlns="http://www.w3.org/2000/svg" width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M3 14h3a2 2 0 0 1 2 2v3a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2v-7a9 9 0 0 1 18 0v7a2 2 0 0 1-2 2h-1a2 2 0 0 1-2-2v-3a2 2 0 0 1 2-2h3"/></svg>',
    "refresh": '<svg xmlns="http://www.w3.org/2000/svg" width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M3 12a9 9 0 0 1 9-9 9.75 9.75 0 0 1 6.74 2.74L21 8"/><path d="M21 3v5h-5"/><path d="M21 12a9 9 0 0 1-9 9 9.75 9.75 0 0 1-6.74-2.74L3 16"/><path d="M8 16H3v5"/></svg>',
    "sparkles": '<svg xmlns="http://www.w3.org/2000/svg" width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="m12 3-1.912 5.813a2 2 0 0 1-1.275 1.275L3 12l5.813 1.912a2 2 0 0 1 1.275 1.275L12 21l1.912-5.813a2 2 0 0 1 1.275-1.275L21 12l-5.813-1.912a2 2 0 0 1-1.275-1.275L12 3Z"/><path d="M5 3v4"/><path d="M19 17v4"/><path d="M3 5h4"/><path d="M17 19h4"/></svg>',
    "award": '<svg xmlns="http://www.w3.org/2000/svg" width="48" height="48" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.5" stroke-linecap="round" stroke-linejoin="round"><circle cx="12" cy="8" r="6"/><path d="M15.477 12.89 17 22l-5-3-5 3 1.523-9.11"/></svg>',
    "heart": '<svg xmlns="http://www.w3.org/2000/svg" width="14" height="14" viewBox="0 0 24 24" fill="currentColor" stroke="none"><path d="M20.84 4.61a5.5 5.5 0 0 0-7.78 0L12 5.67l-1.06-1.06a5.5 5.5 0 0 0-7.78 7.78l1.06 1.06L12 21.23l7.78-7.78 1.06-1.06a5.5 5.5 0 0 0 0-7.78z"/></svg>',
    "arrow_right": '<svg xmlns="http://www.w3.org/2000/svg" width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M5 12h14"/><path d="m12 5 7 7-7 7"/></svg>',
    "building": '<svg xmlns="http://www.w3.org/2000/svg" width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M3 22h18"/><path d="M6 18v-7"/><path d="M10 18v-7"/><path d="M14 18v-7"/><path d="M18 18v-7"/><path d="M12 2l8 5H4z"/></svg>',
    "scale": '<svg xmlns="http://www.w3.org/2000/svg" width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="m16 16 3-8 3 8c-.87.65-1.92 1-3 1s-2.13-.35-3-1Z"/><path d="m2 16 3-8 3 8c-.87.65-1.92 1-3 1s-2.13-.35-3-1Z"/><path d="M7 21h10"/><path d="M12 3v18"/><path d="M3 7h2c2 0 5-1 7-2 2 1 5 2 7 2h2"/></svg>',
    "microscope": '<svg xmlns="http://www.w3.org/2000/svg" width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M10 2v7.31"/><path d="M14 9.3V1.99"/><path d="M8.5 2h7"/><path d="M14 9.3a6.5 6.5 0 1 1-4 0"/><path d="M5.52 16h10.96"/></svg>',
    "target": '<svg xmlns="http://www.w3.org/2000/svg" width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><circle cx="12" cy="12" r="10"/><circle cx="12" cy="12" r="6"/><circle cx="12" cy="12" r="2"/></svg>',
    "globe": '<svg xmlns="http://www.w3.org/2000/svg" width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><circle cx="12" cy="12" r="10"/><path d="M12 2a15.3 15.3 0 0 1 4 10 15.3 15.3 0 0 1-4 10 15.3 15.3 0 0 1-4-10 15.3 15.3 0 0 1 4-10z"/><path d="M2 12h20"/></svg>',
    "smartphone": '<svg xmlns="http://www.w3.org/2000/svg" width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><rect x="5" y="2" width="14" height="20" rx="2" ry="2"/><path d="M12 18h.01"/></svg>',
    "scroll_text": '<svg xmlns="http://www.w3.org/2000/svg" width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M8 21h12.5"/><path d="M12.5 3H8.5a4.5 4.5 0 0 0-4.5 4.5v9a4.5 4.5 0 0 0 4.5 4.5H19"/><path d="M14.5 3v6h6"/><path d="M8 10h2"/><path d="M8 14h6"/></svg>',
    "star": '<svg xmlns="http://www.w3.org/2000/svg" width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><polygon points="12 2 15.09 8.26 22 9.27 17 14.14 18.18 21.02 12 17.77 5.82 21.02 7 14.14 2 9.27 8.91 8.26 12 2"/></svg>',
}

def icon(name, size=18, color="currentColor", cls=""):
    """Return an inline SVG icon with optional size/color override."""
    svg = ICONS.get(name, "")
    if size != 18:
        svg = svg.replace('width="18"', f'width="{size}"').replace('height="18"', f'height="{size}"')
        svg = svg.replace('width="20"', f'width="{size}"').replace('height="20"', f'height="{size}"')
        svg = svg.replace('width="40"', f'width="{size}"').replace('height="40"', f'height="{size}"')
        svg = svg.replace('width="48"', f'width="{size}"').replace('height="48"', f'height="{size}"')
    if color != "currentColor":
        svg = svg.replace('stroke="currentColor"', f'stroke="{color}"')
    if cls:
        svg = svg.replace("<svg ", f'<svg class="{cls}" ')
    return svg

# ─── PREMIUM DARK LUXURY CSS ────────────────────────────────────────────────────
@st.cache_data(show_spinner=False)
def get_file_b64(path, modified_at):
    import base64

    with open(path, "rb") as file:
        return base64.b64encode(file.read()).decode()

def render_themed_audio(audio_path, player_key, title="Pronunciation Audio", autoplay=False):
    """Render the app's themed audio control instead of the browser default player."""
    if not os.path.exists(audio_path):
        st.markdown(f"""
        <div class="video-not-found" style="max-width: 520px; min-height: 72px; margin: 1rem auto 0 auto;">
            {icon("x_circle", 24, "#6b5c47")}
            <span>Pronunciation audio not yet available for this character</span>
        </div>
        """, unsafe_allow_html=True)
        return

    safe_key = "".join(ch if ch.isalnum() else "_" for ch in str(player_key))
    audio_b64 = get_file_b64(audio_path, os.path.getmtime(audio_path))

    autoplay_attr = "autoplay" if autoplay else ""
    initial_button = "Ⅱ" if autoplay else "▶"
    autoplay_js = "audio.play().catch(() => {}); button.textContent = audio.paused ? '▶' : 'Ⅱ';" if autoplay else ""
    audio_html = f"""
    <style>
    .ak-audio-card {{
        box-sizing: border-box;
        width: min(100%, 560px);
        margin: 0 auto;
        padding: 1rem 1.1rem;
        background: linear-gradient(135deg, rgba(255, 153, 51, 0.10), rgba(45, 20, 7, 0.92));
        border: 1px solid rgba(255, 153, 51, 0.28);
        border-top-color: rgba(255, 153, 51, 0.55);
        border-radius: 8px;
        box-shadow: 0 14px 36px rgba(0,0,0,0.28), inset 0 1px 0 rgba(255,255,255,0.05);
        font-family: Inter, sans-serif;
    }}
    .ak-audio-title {{
        display: flex;
        align-items: center;
        justify-content: center;
        gap: 0.45rem;
        color: #FF9933;
        font-size: 0.72rem;
        font-weight: 800;
        text-transform: uppercase;
        letter-spacing: 0.14em;
        margin-bottom: 0.75rem;
    }}
    .ak-audio-player {{
        display: grid;
        grid-template-columns: 40px 1fr auto;
        align-items: center;
        gap: 0.85rem;
    }}
    .ak-audio-play {{
        width: 40px;
        height: 40px;
        border-radius: 8px;
        border: 1px solid rgba(255,153,51,0.42);
        background: linear-gradient(135deg, #FF9933, #B86619);
        color: #1A0D06;
        font-size: 1rem;
        font-weight: 900;
        cursor: pointer;
        box-shadow: 0 8px 18px rgba(255,153,51,0.18);
    }}
    .ak-audio-play:hover {{
        filter: brightness(1.08);
    }}
    .ak-audio-track {{
        appearance: none;
        -webkit-appearance: none;
        width: 100%;
        height: 8px;
        border-radius: 999px;
        background: linear-gradient(90deg, #FF9933 var(--progress, 0%), rgba(255,255,255,0.12) var(--progress, 0%));
        outline: none;
        cursor: pointer;
    }}
    .ak-audio-track::-webkit-slider-thumb {{
        -webkit-appearance: none;
        width: 16px;
        height: 16px;
        border-radius: 50%;
        background: #FFD19A;
        border: 2px solid #FF9933;
        box-shadow: 0 0 12px rgba(255,153,51,0.45);
    }}
    .ak-audio-track::-moz-range-thumb {{
        width: 16px;
        height: 16px;
        border-radius: 50%;
        background: #FFD19A;
        border: 2px solid #FF9933;
        box-shadow: 0 0 12px rgba(255,153,51,0.45);
    }}
    .ak-audio-time {{
        color: rgba(245, 238, 226, 0.78);
        font-size: 0.78rem;
        font-weight: 700;
        min-width: 80px;
        text-align: right;
        font-variant-numeric: tabular-nums;
    }}
    @media (max-width: 520px) {{
        .ak-audio-player {{
            grid-template-columns: 40px 1fr;
        }}
        .ak-audio-time {{
            grid-column: 1 / -1;
            text-align: center;
        }}
    }}
    </style>
    <div class="ak-audio-card">
        <div class="ak-audio-title">
            <span>{icon("headphones", 14)}</span>
            {title}
        </div>
        <div class="ak-audio-player">
            <button class="ak-audio-play" id="audioBtn_{safe_key}" type="button">{initial_button}</button>
            <input class="ak-audio-track" id="audioTrack_{safe_key}" type="range" min="0" max="100" value="0">
            <div class="ak-audio-time" id="audioTime_{safe_key}">0:00 / 0:00</div>
        </div>
        <audio id="learnAudio_{safe_key}" preload="metadata" {autoplay_attr}>
            <source src="data:audio/mp3;base64,{audio_b64}" type="audio/mp3">
        </audio>
    </div>
    <script>
    (function() {{
        const audio = document.getElementById('learnAudio_{safe_key}');
        const button = document.getElementById('audioBtn_{safe_key}');
        const track = document.getElementById('audioTrack_{safe_key}');
        const time = document.getElementById('audioTime_{safe_key}');

        function fmt(seconds) {{
            if (!Number.isFinite(seconds)) return '0:00';
            const m = Math.floor(seconds / 60);
            const s = Math.floor(seconds % 60).toString().padStart(2, '0');
            return `${{m}}:${{s}}`;
        }}
        function sync() {{
            const pct = audio.duration ? (audio.currentTime / audio.duration) * 100 : 0;
            track.value = pct;
            track.style.setProperty('--progress', `${{pct}}%`);
            time.textContent = `${{fmt(audio.currentTime)}} / ${{fmt(audio.duration)}}`;
        }}
        button.addEventListener('click', () => {{
            if (audio.paused) {{
                audio.play();
                button.textContent = 'Ⅱ';
            }} else {{
                audio.pause();
                button.textContent = '▶';
            }}
        }});
        track.addEventListener('input', () => {{
            if (audio.duration) {{
                audio.currentTime = (Number(track.value) / 100) * audio.duration;
                sync();
            }}
        }});
        audio.addEventListener('loadedmetadata', sync);
        audio.addEventListener('timeupdate', sync);
        audio.addEventListener('play', () => button.textContent = 'Ⅱ');
        audio.addEventListener('pause', () => button.textContent = '▶');
        audio.addEventListener('ended', () => {{
            button.textContent = '▶';
            sync();
        }});
        {autoplay_js}
        sync();
    }})();
    </script>
    """
    st.iframe(audio_html, height=132)

def crop_center_square(image):
    """Return a centered 1:1 crop for camera captures."""
    width, height = image.size
    side = min(width, height)
    left = (width - side) // 2
    top = (height - side) // 2
    return image.crop((left, top, left + side, top + side))

css = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Cinzel:wght@400;600;700;900&family=Inter:wght@300;400;500;600;700;800&display=swap');
/* ─── ROOT VARIABLES ─── */
:root {
    --gold: #FF9933;
    --gold-light: #FFB347;
    --gold-dark: #CC7A29;
    --gold-glow: rgba(255, 153, 51, 0.35);
    --green: #22c55e;
    --green-glow: rgba(34, 197, 94, 0.25);
    --red: #ef4444;
    --red-glow: rgba(239, 68, 68, 0.25);
    --blue-accent: #5B9BD5;
    --blue-soft: rgba(91, 155, 213, 0.15);
    --purple: #9B59B6;
    --parchment-dark: #0F0804;
    --parchment-mid: #1A0D06;
    --parchment-card: rgba(26, 13, 6, 0.7);
    --surface: rgba(26, 13, 6, 0.15);
    --surface-hover: rgba(50, 25, 10, 0.25);
    --surface-active: rgba(255, 153, 51, 0.08);
    --border: rgba(255, 153, 51, 0.12);
    --border-hover: rgba(255, 153, 51, 0.35);
    --text-primary: #f8fafc;
    --text-secondary: #a8956e;
    --text-muted: #6b5c47;
    --shadow-gold: 0 0 40px rgba(255, 153, 51, 0.08);
    --shadow-card: 0 8px 32px rgba(0, 0, 0, 0.4);
    --shadow-hover: 0 12px 48px rgba(255, 153, 51, 0.12);
    --shadow-glow: 0 0 60px rgba(255, 153, 51, 0.15);
    --radius: 16px;
    --radius-sm: 10px;
    --radius-lg: 24px;
    --transition: all 0.4s cubic-bezier(0.25, 0.8, 0.25, 1);
    --transition-fast: all 0.25s cubic-bezier(0.25, 0.8, 0.25, 1);
}

html, body, [class*="css"] {
    font-family: 'Inter', sans-serif;
}

/* ─── BACKGROUND: GOLDEN BROWN MOVING GRADIENT ─── */
@keyframes movingGradient {
    0% { background-position: 0% 50%; }
    50% { background-position: 100% 50%; }
    100% { background-position: 0% 50%; }
}

.stApp {
    background: linear-gradient(-45deg, #0F0804, #2A1509, #3d2312, #180D06);
    background-size: 300% 300%;
    animation: movingGradient 22s ease-in-out infinite;
    color: var(--text-primary);
    min-height: 100vh;
}

/* ─── TRIANGLE MESH BACKGROUND ─── */
.stApp::before {
    content: '';
    position: fixed;
    top: 0; left: 0; right: 0; bottom: 0;
    pointer-events: none;
    z-index: 0;
    background-image: url("data:image/svg+xml,%3Csvg viewBox='0 0 100 100' xmlns='http://www.w3.org/2000/svg'%3E%3Cpath d='M0,0 L60,0 L30,25 Z M0,0 L30,25 L0,35 Z M60,0 L80,25 L30,25 Z M60,0 L100,0 L80,25 Z M100,0 L100,35 L80,25 Z M0,35 L30,25 L45,65 Z M0,35 L45,65 L0,75 Z M30,25 L80,25 L45,65 Z M80,25 L100,35 L100,75 Z M80,25 L100,75 L45,65 Z M0,75 L45,65 L60,100 Z M0,75 L60,100 L0,100 Z M45,65 L100,75 L60,100 Z M60,100 L100,75 L100,100 Z' fill='none' stroke='%23FF9933' stroke-width='0.15' stroke-linejoin='round' stroke-opacity='0.2'/%3E%3C/svg%3E");
    background-size: 500px 500px;
}

/* ─── RANDOMLY GLOWING TRIANGLES ─── */
@keyframes triangleSweep {
    0% { 
        -webkit-mask-position: 0% 0%, 100% 100%, 50% 50%, 20% 80%; 
        mask-position: 0% 0%, 100% 100%, 50% 50%, 20% 80%;
    }
    50% { 
        -webkit-mask-position: 100% 50%, 0% 0%, 80% 20%, 100% 100%; 
        mask-position: 100% 50%, 0% 0%, 80% 20%, 100% 100%;
    }
    100% { 
        -webkit-mask-position: 50% 100%, 50% 0%, 10% 90%, 0% 0%; 
        mask-position: 50% 100%, 50% 0%, 10% 90%, 0% 0%;
    }
}

.stApp::after {
    content: '';
    position: fixed;
    top: 0; left: 0; right: 0; bottom: 0; 
    pointer-events: none;
    z-index: 0;
    background-image: 
        url("data:image/svg+xml,%3Csvg viewBox='0 0 100 100' xmlns='http://www.w3.org/2000/svg'%3E%3Cpath d='M0,0 L60,0 L30,25 Z M0,0 L30,25 L0,35 Z M60,0 L80,25 L30,25 Z M60,0 L100,0 L80,25 Z M100,0 L100,35 L80,25 Z M0,35 L30,25 L45,65 Z M0,35 L45,65 L0,75 Z M30,25 L80,25 L45,65 Z M80,25 L100,35 L100,75 Z M80,25 L100,75 L45,65 Z M0,75 L45,65 L60,100 Z M0,75 L60,100 L0,100 Z M45,65 L100,75 L60,100 Z M60,100 L100,75 L100,100 Z' fill='rgba(255,153,51,0.02)' stroke='%23FFB347' stroke-width='0.4' stroke-linejoin='round' stroke-opacity='0.9'/%3E%3Ccircle cx='0' cy='0' r='0.6' fill='%23FFB347'/%3E%3Ccircle cx='60' cy='0' r='0.6' fill='%23FFB347'/%3E%3Ccircle cx='100' cy='0' r='0.6' fill='%23FFB347'/%3E%3Ccircle cx='0' cy='35' r='0.6' fill='%23FFB347'/%3E%3Ccircle cx='30' cy='25' r='0.6' fill='%23FFB347'/%3E%3Ccircle cx='80' cy='25' r='0.6' fill='%23FFB347'/%3E%3Ccircle cx='100' cy='35' r='0.6' fill='%23FFB347'/%3E%3Ccircle cx='0' cy='75' r='0.6' fill='%23FFB347'/%3E%3Ccircle cx='45' cy='65' r='0.6' fill='%23FFB347'/%3E%3Ccircle cx='100' cy='75' r='0.6' fill='%23FFB347'/%3E%3Ccircle cx='0' cy='100' r='0.6' fill='%23FFB347'/%3E%3Ccircle cx='60' cy='100' r='0.6' fill='%23FFB347'/%3E%3Ccircle cx='100' cy='100' r='0.6' fill='%23FFB347'/%3E%3C/svg%3E");
    background-size: 500px 500px;
    filter: drop-shadow(0 0 15px rgba(255,153,51,0.8));
    -webkit-mask-image: radial-gradient(circle, black 0%, transparent 15%),
                        radial-gradient(circle, black 0%, transparent 20%),
                        radial-gradient(circle, black 0%, transparent 15%),
                        radial-gradient(circle, black 0%, transparent 25%);
    -webkit-mask-size: 200% 200%, 250% 250%, 150% 150%, 300% 300%;
    -webkit-mask-repeat: no-repeat;
    mask-image: radial-gradient(circle, black 0%, transparent 15%),
                radial-gradient(circle, black 0%, transparent 20%),
                radial-gradient(circle, black 0%, transparent 15%),
                radial-gradient(circle, black 0%, transparent 25%);
    mask-size: 200% 200%, 250% 250%, 150% 150%, 300% 300%;
    mask-repeat: no-repeat;
    animation: triangleSweep 18s infinite alternate ease-in-out;
}

/* ─── SCROLLBAR ─── */
::-webkit-scrollbar { width: 6px; }
::-webkit-scrollbar-track { background: var(--parchment-dark); }
::-webkit-scrollbar-thumb {
    background: rgba(255, 153, 51, 0.25);
    border-radius: 3px;
}
::-webkit-scrollbar-thumb:hover {
    background: rgba(255, 153, 51, 0.45);
}

/* ─── MAIN CONTENT PADDING ─── */
.block-container {
    padding: clamp(1.2rem, 4vw, 3rem) clamp(0.8rem, 5vw, 3.5rem) 4rem clamp(0.8rem, 5vw, 3.5rem) !important;
    max-width: 1440px !important;
    width: 100% !important;
    position: relative !important;
    z-index: 10 !important;
    margin: 0 auto;
}

/* ─── KEYFRAME ANIMATIONS ─── */
@keyframes shimmer {
    0%, 100% { background-position: 0% center; }
    50% { background-position: 200% center; }
}
@keyframes fadeInUp {
    from { opacity: 0; transform: translateY(24px); }
    to { opacity: 1; transform: translateY(0); }
}
@keyframes fadeIn {
    from { opacity: 0; }
    to { opacity: 1; }
}
@keyframes rotate {
    from { transform: rotate(0deg); }
    to { transform: rotate(360deg); }
}
@keyframes pulseGlow {
    0%, 100% { box-shadow: 0 0 20px rgba(255, 153, 51, 0.08); }
    50% { box-shadow: 0 0 40px rgba(255, 153, 51, 0.22); }
}
@keyframes breathe {
    0%, 100% { opacity: 0.4; }
    50% { opacity: 0.7; }
}
@keyframes slideBarIn {
    from { width: 0% !important; }
}
@keyframes correctPulse {
    0% { box-shadow: 0 0 0px rgba(34, 197, 94, 0); }
    50% { box-shadow: 0 0 30px rgba(34, 197, 94, 0.4); }
    100% { box-shadow: 0 0 15px rgba(34, 197, 94, 0.15); }
}
@keyframes wrongShake {
    0%, 100% { transform: translateX(0); }
    20% { transform: translateX(-6px); }
    40% { transform: translateX(6px); }
    60% { transform: translateX(-4px); }
    80% { transform: translateX(4px); }
}
@keyframes scaleIn {
    from { opacity: 0; transform: scale(0.92); }
    to { opacity: 1; transform: scale(1); }
}
@keyframes iconFloat {
    0%, 100% { transform: translateY(0); }
    50% { transform: translateY(-3px); }
}
@keyframes ripple {
    0% { transform: scale(0); opacity: 0.5; }
    100% { transform: scale(4); opacity: 0; }
}
@keyframes glowPulse {
    0%, 100% { filter: drop-shadow(0 0 4px rgba(255,153,51,0.15)); }
    50% { filter: drop-shadow(0 0 12px rgba(255,153,51,0.4)); }
}


/* ─── INLINE SVG ICON STYLING ─── */
.icon-inline {
    display: inline-flex;
    align-items: center;
    vertical-align: middle;
    margin-right: 0.5rem;
    color: var(--gold);
    filter: drop-shadow(0 0 6px rgba(255, 153, 51, 0.25));
    transition: var(--transition-fast);
}
.icon-inline:hover {
    filter: drop-shadow(0 0 10px rgba(255, 153, 51, 0.5));
}
.icon-glow {
    animation: glowPulse 3s ease-in-out infinite;
}

/* ─── HERO TITLE CARDS ─── */
.hero-card {
    text-align: center;
    padding: 3rem 2rem 2rem 2rem;
    position: relative;
    background: linear-gradient(135deg, rgba(255, 255, 255, 0.03), rgba(26, 13, 6, 0.15));
    backdrop-filter: blur(28px) saturate(140%);
    -webkit-backdrop-filter: blur(28px) saturate(140%);
    border: 1px solid rgba(255, 153, 51, 0.15);
    border-top: 1px solid rgba(255, 153, 51, 0.35);
    border-radius: 20px;
    box-shadow: 0 16px 40px rgba(0,0,0,0.4), inset 0 1px 0 rgba(255,255,255,0.05);
    margin: 2rem auto 3rem auto;
    max-width: 900px;
    animation: fadeInUp 0.6s ease-out;
}
.hero-title {
    text-align: center;
    position: relative;
}
.hero-title h1 {
    font-family: 'Cinzel', serif;
    font-weight: 900;
    text-align: center;
    font-size: clamp(2.2rem, 6vw, 4.5rem);
    letter-spacing: 0.06em;
    background: linear-gradient(135deg, #FFB347 0%, #FF9933 40%, #CC7A29 70%, #FFB347 100%);
    background-size: 200% auto;
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    animation: shimmer 8s ease-in-out infinite;
    margin: 0;
    line-height: 1.25;
    text-shadow: none;
}
.hero-subtitle {
    text-align: center;
    color: var(--text-secondary);
    font-size: 1.5rem;
    font-weight: 400;
    letter-spacing: 0.12em;
    text-transform: uppercase;
    margin-top: 0.4rem;
    margin-bottom: 2rem;
    animation: fadeIn 0.8s ease-out 0.3s backwards;
}
@media (max-width: 768px) {
    .hero-subtitle { font-size: 1rem; }
}

/* ─── GLOWING DIVIDER LINE ─── */
.gold-divider {
    height: 1px;
    background: linear-gradient(90deg, transparent, var(--gold), transparent);
    margin: 0.5rem auto 2rem auto;
    width: 60%;
    opacity: 0.5;
    animation: fadeIn 1s ease-out 0.5s backwards;
}

/* ─── TABS: PREMIUM STYLE ─── */
.stTabs [data-baseweb="tab-list"] {
    background: linear-gradient(135deg, rgba(255, 255, 255, 0.03), rgba(26, 13, 6, 0.15));
    backdrop-filter: blur(28px) saturate(140%);
    -webkit-backdrop-filter: blur(28px) saturate(140%);
    border: 1px solid rgba(255, 153, 51, 0.15);
    border-top: 1px solid rgba(255, 153, 51, 0.35);
    box-shadow: 0 16px 40px rgba(0,0,0,0.4), inset 0 1px 0 rgba(255,255,255,0.05);
    border-radius: var(--radius);
    padding: 6px;
    gap: 4px;
    justify-content: center;
    margin-bottom: 2rem;
    animation: fadeInUp 0.5s ease-out 0.4s backwards;
}
.stTabs [data-baseweb="tab"] {
    font-family: 'Inter', sans-serif;
    font-weight: 600;
    font-size: 0.95rem;
    letter-spacing: 0.04em;
    color: var(--text-muted);
    border-radius: var(--radius-sm);
    padding: 0.7rem 1.8rem;
    transition: var(--transition);
    border: 1px solid transparent;
    background: transparent;
    position: relative;
    overflow: hidden;
}
.stTabs [data-baseweb="tab"]::after {
    content: '';
    position: absolute;
    inset: 0;
    background: radial-gradient(circle at var(--x, 50%) var(--y, 50%), rgba(255,153,51,0.15), transparent 60%);
    opacity: 0;
    transition: opacity 0.3s ease;
}
.stTabs [data-baseweb="tab"]:hover {
    color: var(--gold-light);
    background: rgba(255, 153, 51, 0.06);
    border-color: rgba(255, 153, 51, 0.15);
}
.stTabs [data-baseweb="tab"]:hover::after {
    opacity: 1;
}
.stTabs [aria-selected="true"] {
    background: rgba(255, 153, 51, 0.1) !important;
    color: var(--gold) !important;
    border-color: rgba(255, 153, 51, 0.3) !important;
    box-shadow: 0 0 20px rgba(255, 153, 51, 0.08);
}
.stTabs [data-baseweb="tab-highlight"] { display: none !important; }
.stTabs [data-baseweb="tab-border"] { display: none !important; }

/* Tab content fade-in */
.stTabs [data-baseweb="tab-panel"] {
    animation: fadeIn 0.4s ease-out;
}

/* ─── SECTION CARDS ─── */
.section-card {
    background: linear-gradient(135deg, rgba(255, 255, 255, 0.03), var(--surface));
    backdrop-filter: blur(28px) saturate(140%);
    -webkit-backdrop-filter: blur(28px) saturate(140%);
    border: 1px solid var(--border);
    border-top: 1px solid rgba(255, 153, 51, 0.35);
    border-radius: var(--radius-lg);
    padding: 2.5rem;
    margin: 1.5rem 0;
    box-shadow: var(--shadow-card), inset 0 1px 0 rgba(255,255,255,0.05);
    transition: var(--transition);
    position: relative;
    overflow: hidden;
    animation: fadeInUp 0.5s ease-out;
}
.section-card::before {
    content: '';
    position: absolute;
    top: 0; left: 0; right: 0;
    height: 1px;
    background: linear-gradient(90deg, transparent, rgba(255, 153, 51, 0.3), transparent);
}
.section-card:hover {
    border-color: var(--border-hover);
    box-shadow: var(--shadow-hover);
}

/* ─── SECTION HEADINGS ─── */
.section-heading {
    font-family: 'Cinzel', serif;
    font-weight: 700;
    font-size: 1.6rem;
    color: var(--text-primary);
    margin-bottom: 0.3rem;
    letter-spacing: 0.04em;
    display: flex;
    align-items: center;
    gap: 0.6rem;
}
.section-desc {
    color: var(--text-secondary);
    font-size: 0.95rem;
    margin-bottom: 0;
    line-height: 1.6;
}

/* ─── FILE UPLOADER — MERGED CARD ─── */
[data-testid="stHorizontalBlock"]:has(.upload-label) {
    background: var(--surface);
    backdrop-filter: blur(16px);
    -webkit-backdrop-filter: blur(16px);
    border: 1px solid var(--border);
    border-radius: var(--radius-lg);
    padding: 1.5rem 2.5rem;
    margin: 1.5rem 0;
    transition: var(--transition);
    position: relative;
    overflow: hidden;
    animation: fadeInUp 0.5s ease-out 0.1s backwards;
    box-shadow: var(--shadow-card);
    align-items: center;
}
[data-testid="stHorizontalBlock"]:has(.upload-label)::before {
    content: '';
    position: absolute;
    top: 0; left: 0; right: 0;
    height: 1px;
    background: linear-gradient(90deg, transparent, rgba(255, 153, 51, 0.3), transparent);
}
[data-testid="stHorizontalBlock"]:has(.upload-label):hover {
    border-color: var(--border-hover);
    box-shadow: var(--shadow-hover);
}
.recognize-source-card {
    background: linear-gradient(135deg, rgba(255, 153, 51, 0.06), rgba(255, 255, 255, 0.02));
    border: 1px solid rgba(255, 153, 51, 0.16);
    border-top-color: rgba(255, 153, 51, 0.32);
    border-radius: 8px;
    padding: 1rem 1.1rem;
    margin-bottom: 0.75rem;
    min-height: 104px;
}
.recognize-source-title {
    display: flex;
    align-items: center;
    gap: 0.45rem;
    color: var(--gold);
    font-size: 0.78rem;
    font-weight: 800;
    letter-spacing: 0.12em;
    text-transform: uppercase;
    margin-bottom: 0.35rem;
}
.recognize-source-copy {
    color: var(--text-muted);
    font-size: 0.78rem;
    line-height: 1.45;
}
[data-testid="stCameraInput"] video,
[data-testid="stCameraInput"] img {
    aspect-ratio: 1 / 1;
    object-fit: cover;
    border-radius: 8px;
}
.upload-label {
    font-family: 'Cinzel', serif;
    font-weight: 700;
    font-size: 1.15rem;
    color: var(--gold);
    margin-bottom: 0.4rem;
    letter-spacing: 0.04em;
    display: flex;
    align-items: center;
    gap: 0.6rem;
}
.upload-sublabel {
    color: var(--text-muted);
    font-size: 0.82rem;
    margin-bottom: 0;
}
[data-testid="stFileUploadDropzone"] {
    border: 2px dashed rgba(255, 153, 51, 0.2) !important;
    border-radius: var(--radius) !important;
    background: rgba(255, 153, 51, 0.02) !important;
    transition: var(--transition);
    padding: 1rem 1.5rem !important;
    min-height: 90px !important;
}
[data-testid="stFileUploadDropzone"]:hover {
    background: rgba(255, 153, 51, 0.06) !important;
    border-color: var(--gold) !important;
    box-shadow: 0 0 35px rgba(255, 153, 51, 0.1);
}

/* ─── EMPTY STATE ─── */
.empty-state {
    text-align: center;
    padding: 4rem 2rem;
    animation: fadeIn 0.6s ease-out;
}
.empty-state-icon {
    margin-bottom: 1.2rem;
    color: var(--text-muted);
    animation: breathe 3s ease-in-out infinite;
}
.empty-state-title {
    color: var(--text-secondary);
    font-size: 1.1rem;
    font-weight: 500;
    margin-bottom: 0.4rem;
}
.empty-state-sub {
    color: var(--text-muted);
    font-size: 0.82rem;
}

/* ─── IMAGE DISPLAY CARDS ─── */
.image-card {
    background: linear-gradient(135deg, rgba(255, 255, 255, 0.03), var(--surface));
    backdrop-filter: blur(24px) saturate(120%);
    -webkit-backdrop-filter: blur(24px) saturate(120%);
    border: 1px solid var(--border);
    border-top: 1px solid rgba(255, 153, 51, 0.3);
    border-radius: var(--radius-lg);
    padding: 1.5rem;
    transition: var(--transition);
    animation: fadeInUp 0.5s ease-out;
    overflow: hidden;
    box-shadow: var(--shadow-card), inset 0 1px 0 rgba(255,255,255,0.05);
}
.image-card:hover {
    border-color: var(--border-hover);
    box-shadow: var(--shadow-hover);
}
.image-card-label {
    font-family: 'Cinzel', serif;
    font-weight: 600;
    font-size: 0.9rem;
    color: var(--text-secondary);
    text-transform: uppercase;
    letter-spacing: 0.12em;
    margin-bottom: 1rem;
    text-align: center;
    display: flex;
    align-items: center;
    justify-content: center;
    gap: 0.5rem;
}
.image-card-sublabel {
    color: var(--text-muted);
    font-size: 0.75rem;
    text-align: center;
    margin-top: 0.8rem;
    font-style: italic;
    line-height: 1.4;
}

/* ─── PREDICTION CARD ─── */
.prediction-card {
    background: linear-gradient(135deg, rgba(255, 153, 51, 0.1) 0%, rgba(26, 13, 6, 0.15) 100%);
    backdrop-filter: blur(32px) saturate(150%);
    -webkit-backdrop-filter: blur(32px) saturate(150%);
    border: 1px solid rgba(255, 153, 51, 0.2);
    border-top: 1px solid rgba(255, 153, 51, 0.5);
    border-radius: var(--radius-lg);
    padding: 3.5rem 2rem;
    text-align: center;
    box-shadow: var(--shadow-card), 0 0 60px rgba(255, 153, 51, 0.1), inset 0 1px 0 rgba(255,255,255,0.05);
    margin: 2rem 0;
    position: relative;
    overflow: hidden;
    animation: scaleIn 0.6s ease-out, pulseGlow 4s ease-in-out 0.6s infinite;
}
.prediction-card::before {
    content: '';
    position: absolute;
    top: -50%; left: -50%;
    width: 200%; height: 200%;
    background: radial-gradient(circle, rgba(255, 153, 51, 0.04) 0%, transparent 60%);
    animation: rotate 20s linear infinite;
}
@media (max-width: 768px) {
    .prediction-label { font-size: 3.5rem !important; }
}
.prediction-subtitle {
    font-family: 'Inter', sans-serif;
    font-size: 0.75rem;
    color: var(--text-muted);
    text-transform: uppercase;
    letter-spacing: 0.25em;
    margin-bottom: 0.2rem;
    font-weight: 500;
    position: relative;
    z-index: 1;
}
.prediction-title {
    font-family: 'Inter', sans-serif;
    font-size: 0.95rem;
    color: var(--text-secondary);
    text-transform: uppercase;
    letter-spacing: 0.15em;
    margin-bottom: 0.8rem;
    font-weight: 600;
    position: relative;
    z-index: 1;
}
.prediction-label {
    font-size: 6.5rem;
    font-weight: 900;
    background: linear-gradient(135deg, #FFB347, #FF9933, #CC7A29, #FFB347);
    background-size: 300% auto;
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    animation: shimmer 3s ease-in-out infinite;
    display: block;
    margin: 0 auto;
    line-height: 1.55;
    padding: 0.08em 0 0.18em 0;
    position: relative;
    z-index: 1;
    filter: drop-shadow(0 0 30px rgba(255, 153, 51, 0.2));
}

/* ─── CONFIDENCE BARS ─── */
.conf-section-title {
    font-family: 'Cinzel', serif;
    font-weight: 700;
    font-size: 1.2rem;
    color: var(--text-primary);
    margin-bottom: 1.2rem;
    letter-spacing: 0.04em;
    display: flex;
    align-items: center;
    gap: 0.5rem;
}
.conf-container {
    margin-bottom: 1.4rem;
    animation: fadeInUp 0.5s ease-out backwards;
}
.conf-container:nth-child(2) { animation-delay: 0.15s; }
.conf-container:nth-child(3) { animation-delay: 0.3s; }
.conf-container:nth-child(4) { animation-delay: 0.45s; }
.conf-header {
    display: flex;
    justify-content: space-between;
    margin-bottom: 0.5rem;
    font-weight: 600;
    color: var(--text-primary);
    font-size: 0.95rem;
}
.conf-bar-bg {
    background: rgba(255, 255, 255, 0.03);
    border: 1px solid rgba(255, 255, 255, 0.05);
    border-radius: 999px;
    height: 12px;
    width: 100%;
    overflow: hidden;
    box-shadow: inset 0 1px 3px rgba(0, 0, 0, 0.3);
    backdrop-filter: blur(5px);
}
.conf-bar-fill {
    background: linear-gradient(90deg, #CC7A29, #FF9933, #FFB347);
    height: 100%;
    border-radius: 999px;
    transition: width 1.2s cubic-bezier(0.25, 0.8, 0.25, 1);
    box-shadow: 0 0 15px rgba(255, 153, 51, 0.5);
    animation: slideBarIn 1.2s cubic-bezier(0.25, 0.8, 0.25, 1) backwards;
}

/* ─── CHARACTER GRIDS (Learn & Library) ─── */
.char-grid {
    display: grid;
    grid-template-columns: repeat(auto-fill, minmax(135px, 1fr));
    gap: 16px;
    margin-top: 1.5rem;
}
@media (min-width: 1025px) {
    .library-section .char-grid { grid-template-columns: repeat(6, 1fr) !important; }
}
@media (max-width: 1024px) {
    .char-grid { grid-template-columns: repeat(auto-fill, minmax(115px, 1fr)); gap: 14px; }
}
@media (max-width: 768px) {
    .char-grid { grid-template-columns: repeat(auto-fill, minmax(95px, 1fr)); gap: 10px; margin-top: 1rem; }
}
@media (max-width: 480px) {
    .char-grid { grid-template-columns: repeat(auto-fill, minmax(80px, 1fr)); gap: 8px; }
}

.char-card-ui {
    background: linear-gradient(135deg, rgba(255, 255, 255, 0.03), var(--surface));
    backdrop-filter: blur(20px) saturate(110%);
    -webkit-backdrop-filter: blur(20px) saturate(110%);
    border: 1px solid var(--border);
    border-top: 1px solid rgba(255, 153, 51, 0.25);
    border-radius: var(--radius-sm);
    padding: 1rem;
    text-align: center;
    transition: var(--transition-fast);
    cursor: pointer;
    position: relative;
    overflow: hidden;
    box-shadow: 0 4px 12px rgba(0, 0, 0, 0.2), inset 0 1px 0 rgba(255,255,255,0.05);
}
.char-card-ui:hover {
    border-color: var(--border-hover);
    background: var(--surface-hover);
    transform: translateY(-2px);
    box-shadow: 0 8px 20px rgba(255,153,51,0.1);
}
.char-main-txt {
    font-size: 2.2rem;
    font-weight: 700;
    color: var(--gold);
    margin-bottom: 0.2rem;
}
.char-sub-txt {
    font-size: 0.85rem;
    color: var(--text-secondary);
    font-weight: 500;
}

/* ─── QUIZ SCOREBOARD MOBILE ─── */
@media (max-width: 600px) {
    .quiz-scoreboard {
        flex-direction: column;
        gap: 10px;
    }
}


/* ─── BUTTONS — MICRO-INTERACTIONS ─── */
.stButton > button {
    font-family: 'Inter', sans-serif;
    font-weight: 600;
    border-radius: var(--radius-sm) !important;
    letter-spacing: 0.03em;
    transition: var(--transition) !important;
    border: 1px solid var(--border) !important;
    position: relative;
    overflow: hidden;
}
.stButton > button::before {
    content: '';
    position: absolute;
    top: 50%;
    left: 50%;
    width: 0;
    height: 0;
    background: rgba(255, 153, 51, 0.15);
    border-radius: 50%;
    transform: translate(-50%, -50%);
    transition: width 0.6s ease, height 0.6s ease;
}
.stButton > button:hover::before {
    width: 300px;
    height: 300px;
}
.stButton > button:hover {
    transform: translateY(-2px) !important;
    box-shadow: 0 8px 25px rgba(255, 153, 51, 0.2) !important;
    border-color: var(--gold) !important;
}
.stButton > button:active {
    transform: translateY(0) scale(0.97) !important;
    transition: transform 0.1s ease !important;
}
.stButton > button[kind="primary"],
.stButton > button[data-testid="stBaseButton-primary"] {
    background: linear-gradient(135deg, #CC7A29, #FF9933) !important;
    color: #0F0804 !important;
    border: none !important;
    font-weight: 700 !important;
    font-size: 0.95rem !important;
    padding: 0.6rem 2rem !important;
}
.stButton > button[kind="primary"]:hover,
.stButton > button[data-testid="stBaseButton-primary"]:hover {
    background: linear-gradient(135deg, #FF9933, #FFB347) !important;
    box-shadow: 0 8px 30px rgba(255, 153, 51, 0.35) !important;
}
.stButton > button[kind="primary"]::before,
.stButton > button[data-testid="stBaseButton-primary"]::before {
    background: rgba(255, 255, 255, 0.15);
}

/* ─── SELECTBOX / DROPDOWNS ─── */
div[data-baseweb="select"], div[data-baseweb="select"] * {
    cursor: pointer !important;
}
div[data-baseweb="select"] > div {
    border-radius: var(--radius-sm) !important;
    border: 1px solid var(--border) !important;
    border-top: 1px solid rgba(255, 153, 51, 0.25) !important;
    background: rgba(26, 13, 6, 0.25) !important;
    backdrop-filter: blur(12px);
    -webkit-backdrop-filter: blur(12px);
    transition: var(--transition);
}
div[data-baseweb="select"] > div:hover {
    border-color: var(--border-hover) !important;
    box-shadow: 0 0 15px rgba(255, 153, 51, 0.05) !important;
}

/* ─── TEXT INPUT / SEARCH ─── */
.stTextInput input {
    border-radius: var(--radius-sm) !important;
    border: 1px solid var(--border) !important;
    border-top: 1px solid rgba(255, 153, 51, 0.25) !important;
    background: rgba(26, 13, 6, 0.25) !important;
    backdrop-filter: blur(12px) !important;
    -webkit-backdrop-filter: blur(12px) !important;
    color: var(--text-primary) !important;
    transition: var(--transition);
    padding: 0.7rem 1rem !important;
}
.stTextInput input:focus {
    border-color: var(--gold) !important;
    box-shadow: 0 0 20px rgba(255, 153, 51, 0.1) !important;
}
.stTextInput input::placeholder {
    color: var(--text-muted) !important;
}

/* ─── EXPANDER ─── */
.streamlit-expanderHeader {
    font-weight: 600 !important;
    color: var(--text-secondary) !important;
    border-radius: var(--radius-sm) !important;
    transition: var(--transition);
}
.streamlit-expanderHeader:hover {
    color: var(--gold) !important;
}

/* ─── METRICS ─── */
[data-testid="stMetric"] {
    background: linear-gradient(135deg, rgba(255, 255, 255, 0.03), var(--surface));
    backdrop-filter: blur(20px) saturate(120%);
    -webkit-backdrop-filter: blur(20px) saturate(120%);
    border: 1px solid var(--border);
    border-top: 1px solid rgba(255, 153, 51, 0.3);
    border-radius: var(--radius);
    padding: 1.2rem 1.5rem;
    text-align: center;
    transition: var(--transition);
    box-shadow: var(--shadow-card), inset 0 1px 0 rgba(255,255,255,0.05);
}
[data-testid="stMetric"]:hover {
    border-color: var(--border-hover);
    box-shadow: var(--shadow-hover);
}
[data-testid="stMetricLabel"] {
    color: var(--text-secondary) !important;
    font-weight: 600 !important;
    text-transform: uppercase;
    letter-spacing: 0.1em;
    font-size: 0.75rem !important;
}
[data-testid="stMetricValue"] {
    color: var(--gold) !important;
    font-family: 'Cinzel', serif !important;
    font-weight: 700 !important;
}

/* ─── PROGRESS BAR ─── */
.stProgress > div > div {
    background: rgba(255, 255, 255, 0.06) !important;
    border-radius: 999px !important;
    height: 8px !important;
}
.stProgress > div > div > div {
    background: linear-gradient(90deg, #CC7A29, #FF9933, #FFB347) !important;
    border-radius: 999px !important;
    box-shadow: 0 0 15px rgba(255, 153, 51, 0.3);
    transition: width 0.8s ease !important;
}

/* ─── ALERTS ─── */
.stAlert {
    border-radius: var(--radius-sm) !important;
    border: 1px solid var(--border) !important;
    border-top: 1px solid rgba(255, 153, 51, 0.25) !important;
    background: rgba(26, 13, 6, 0.2) !important;
    backdrop-filter: blur(20px) saturate(120%);
    -webkit-backdrop-filter: blur(20px) saturate(120%);
    animation: fadeInUp 0.3s ease-out;
    box-shadow: var(--shadow-card);
}

/* ─── LEARN TAB ─── */
.learn-detail-card {
    background: linear-gradient(135deg, rgba(255, 153, 51, 0.08) 0%, var(--surface) 100%);
    backdrop-filter: blur(28px) saturate(140%);
    -webkit-backdrop-filter: blur(28px) saturate(140%);
    border: 1px solid rgba(255, 153, 51, 0.2);
    border-top: 1px solid rgba(255, 153, 51, 0.4);
    border-radius: var(--radius-lg);
    padding: 2.5rem;
    margin-top: 1rem;
    animation: scaleIn 0.35s ease-out;
    position: relative;
    overflow: hidden;
    box-shadow: var(--shadow-card), inset 0 1px 0 rgba(255,255,255,0.05);
}
.learn-detail-card::before {
    content: '';
    position: absolute;
    top: 0; left: 0; right: 0;
    height: 2px;
    background: linear-gradient(90deg, transparent, var(--gold), transparent);
}
.learn-detail-big {
    font-size: 5rem;
    font-weight: 900;
    color: var(--gold);
    text-align: center;
    line-height: 1.2;
    margin-bottom: 0.5rem;
    filter: drop-shadow(0 0 20px rgba(255, 153, 51, 0.2));
    animation: scaleIn 0.4s ease-out;
}
.learn-detail-devnag {
    text-align: center;
    font-size: 1.6rem;
    color: var(--text-primary);
    font-weight: 600;
    margin-bottom: 1.5rem;
}
.learn-detail-row {
    display: flex;
    justify-content: center;
    gap: 2rem;
    flex-wrap: wrap;
    margin-bottom: 1rem;
}
.learn-detail-item {
    text-align: center;
    min-width: 120px;
}
.learn-detail-item .lbl {
    font-size: 0.7rem;
    color: var(--text-muted);
    text-transform: uppercase;
    letter-spacing: 0.12em;
    margin-bottom: 0.3rem;
}
.learn-detail-item .val {
    font-size: 0.95rem;
    color: var(--text-secondary);
    font-weight: 500;
}
.learn-detail-note {
    text-align: center;
    font-size: 0.8rem;
    color: var(--text-muted);
    font-style: italic;
    margin-top: 1rem;
    line-height: 1.5;
    max-width: 500px;
    margin-left: auto;
    margin-right: auto;
}

/* ─── LEARN: CATEGORY CARD BUTTONS ─── */
/* Style the Streamlit buttons to look like premium cards */
button[data-testid="stButton"][kind="secondary"],
button[data-testid="stButton"][kind="primary"] {
    /* Default button styles are fine for most buttons */
}
/* Target the category card buttons specifically by their key prefix */
div[data-testid="stVerticalBlock"] > div[data-testid="stButton"]:only-child button {
    /* This would be too broad, use the below approach instead */
}

/* Learn tab category cards use an invisible Streamlit button overlay so the card itself is clickable. */
.learn-category-card {
    min-height: 230px;
}

div[data-testid="stElementContainer"]:has(.learn-category-card) + div[data-testid="stElementContainer"] {
    margin-top: -230px;
    height: 230px;
    position: relative;
    z-index: 5;
}

div[data-testid="stElementContainer"]:has(.learn-category-card) + div[data-testid="stElementContainer"] div[data-testid="stButton"],
div[data-testid="stElementContainer"]:has(.learn-category-card) + div[data-testid="stElementContainer"] div[data-testid="stButton"] button {
    width: 100%;
    height: 100%;
}

div[data-testid="stElementContainer"]:has(.learn-category-card) + div[data-testid="stElementContainer"] div[data-testid="stButton"] button {
    background: transparent !important;
    border: 0 !important;
    box-shadow: none !important;
    color: transparent !important;
    padding: 0 !important;
    border-radius: var(--radius-lg) !important;
}

div[data-testid="stElementContainer"]:has(.learn-category-card) + div[data-testid="stElementContainer"] div[data-testid="stButton"] button:hover,
div[data-testid="stElementContainer"]:has(.learn-category-card) + div[data-testid="stElementContainer"] div[data-testid="stButton"] button:focus,
div[data-testid="stElementContainer"]:has(.learn-category-card) + div[data-testid="stElementContainer"] div[data-testid="stButton"] button:active {
    background: transparent !important;
    border: 0 !important;
    box-shadow: none !important;
    color: transparent !important;
}

div[data-testid="stElementContainer"]:has(.learn-category-card) + div[data-testid="stElementContainer"] div[data-testid="stButton"] button *,
div[data-testid="stElementContainer"]:has(.learn-category-card) + div[data-testid="stElementContainer"] div[data-testid="stButton"] button::before,
div[data-testid="stElementContainer"]:has(.learn-category-card) + div[data-testid="stElementContainer"] div[data-testid="stButton"] button::after {
    background: transparent !important;
    box-shadow: none !important;
}

div[data-testid="stElementContainer"]:has(.learn-category-card) + div[data-testid="stElementContainer"] div[data-testid="stButton"] button p {
    opacity: 0;
}

[data-testid="stColumn"]:has(.learn-category-card):has(div[data-testid="stButton"] button:hover) .learn-category-card {
    background:
        radial-gradient(circle at 50% 0%, rgba(255, 153, 51, 0.16), transparent 46%),
        linear-gradient(135deg, rgba(255, 255, 255, 0.055), var(--surface-hover));
    border-color: var(--gold);
    transform: translateY(-6px);
    box-shadow: 0 18px 42px rgba(0, 0, 0, 0.34), 0 0 28px rgba(255, 153, 51, 0.18);
}

[data-testid="stColumn"]:has(.learn-category-card):has(div[data-testid="stButton"] button:hover) .learn-category-card::before {
    opacity: 1;
}

/* Learn tab category card buttons - styled via the premium-cat-card class on wrapper */
[data-testid="stColumn"] > [data-testid="stVerticalBlockBorderWrapper"] > div > [data-testid="stVerticalBlock"] > [data-testid="stButton"]:only-child button[kind="secondary"],
[data-testid="stColumn"] > [data-testid="stVerticalBlockBorderWrapper"] > div > [data-testid="stVerticalBlock"] > [data-testid="stButton"]:only-child button[kind="primary"] {
    /* These selectors are too fragile. Use JS-injected class instead */
}

/* ─── LEARN: PRACTICE SECTION (VIDEO + CANVAS) ─── */
.practice-section-title {
    font-family: 'Cinzel', serif;
    font-size: 1.15rem;
    font-weight: 600;
    color: var(--gold);
    display: flex;
    align-items: center;
    gap: 0.5rem;
    margin-bottom: 0.8rem;
}
.practice-panel {
    background: var(--surface);
    border: 1px solid var(--border);
    border-radius: var(--radius);
    padding: 1.2rem;
    animation: fadeInUp 0.35s ease-out;
    background: linear-gradient(135deg, rgba(255, 153, 51, 0.10), rgba(45, 20, 7, 0.92));
}
.practice-panel-label {
    font-size: 0.75rem;
    color: var(--text-muted);
    text-transform: uppercase;
    letter-spacing: 0.1em;
    margin-bottom: 0.6rem;
    font-weight: 600;
    display: flex;
    align-items: center;
    justify-content: center;
    gap: 0.4rem;
}
.learn-audio-card {
    max-width: 560px;
    margin: 1rem auto 0 auto;
    padding: 1rem 1.1rem;
    background: linear-gradient(135deg, rgba(255, 153, 51, 0.08), rgba(255, 255, 255, 0.025));
    border: 1px solid rgba(255, 153, 51, 0.22);
    border-top-color: rgba(255, 153, 51, 0.42);
    border-radius: 8px;
    box-shadow: var(--shadow-card), inset 0 1px 0 rgba(255,255,255,0.04);
}
.learn-audio-title {
    display: flex;
    align-items: center;
    justify-content: center;
    gap: 0.45rem;
    color: var(--gold);
    font-size: 0.72rem;
    font-weight: 700;
    text-transform: uppercase;
    letter-spacing: 0.14em;
    margin-bottom: 0.55rem;
}
.learn-speed-card {
    width: min(100%, 380px);
    margin: 0.7rem auto 0 auto;
    padding: 0.8rem 1rem;
    background: rgba(255, 255, 255, 0.025);
    border: 1px solid rgba(255, 153, 51, 0.16);
    border-radius: 8px;
}
.learn-speed-card .practice-panel-label {
    justify-content: flex-start;
    align-items: center;
    margin-bottom: 0;
}
div[data-testid="stElementContainer"]:has(.learn-speed-card) + div[data-testid="stElementContainer"] {
    width: min(100%, 380px);
    margin: 0.35rem auto 0 auto;
    padding-left: 1.1rem;
}
div[data-testid="stElementContainer"]:has(.learn-speed-card) + div[data-testid="stElementContainer"] div[role="radiogroup"] {
    justify-content: flex-start;
    gap: 1.5rem;
}
div[data-testid="stElementContainer"]:has(.learn-speed-card) + div[data-testid="stElementContainer"] label {
    color: var(--text-secondary) !important;
    font-weight: 600;
}
.video-not-found {
    display: flex;
    flex-direction: column;
    align-items: center;
    justify-content: center;
    min-height: 200px;
    color: var(--text-muted);
    font-size: 0.85rem;
    gap: 0.5rem;
}

/* ─── QUIZ TAB ─── */
.quiz-scoreboard {
    display: flex;
    justify-content: center;
    gap: 2rem;
    flex-wrap: wrap;
    margin: 1rem 0 1.5rem 0;
    animation: fadeInUp 0.4s ease-out;
    contain: layout paint;
}
.quiz-score-item {
    background: linear-gradient(135deg, rgba(255, 255, 255, 0.03), var(--surface));
    backdrop-filter: blur(20px) saturate(120%);
    -webkit-backdrop-filter: blur(20px) saturate(120%);
    border: 1px solid var(--border);
    border-top: 1px solid rgba(255, 153, 51, 0.35);
    border-radius: var(--radius);
    padding: 1.2rem 2rem;
    text-align: center;
    min-width: 130px;
    transition: var(--transition);
    box-shadow: var(--shadow-card), inset 0 1px 0 rgba(255,255,255,0.05);
    will-change: transform;
}
.quiz-score-item:hover {
    border-color: var(--border-hover);
    box-shadow: var(--shadow-gold);
    transform: translateY(-2px);
}
.quiz-score-item.highlight {
    border-color: rgba(255, 153, 51, 0.4);
    border-top-color: rgba(255, 153, 51, 0.6);
    background: rgba(255, 153, 51, 0.08);
    animation: pulseGlow 3s ease-in-out infinite;
}
.quiz-score-label {
    font-size: 0.7rem;
    color: var(--text-muted);
    text-transform: uppercase;
    letter-spacing: 0.12em;
    font-weight: 600;
    margin-bottom: 0.3rem;
}
.quiz-score-value {
    font-family: 'Cinzel', serif;
    font-size: 2rem;
    font-weight: 900;
    color: var(--gold);
}
.quiz-score-value.accuracy { font-size: 1.4rem; }

.quiz-question-card {
    background: linear-gradient(135deg, rgba(255, 153, 51, 0.08) 0%, var(--surface) 100%);
    backdrop-filter: blur(28px) saturate(140%);
    -webkit-backdrop-filter: blur(28px) saturate(140%);
    border: 1px solid var(--border);
    border-top: 1px solid rgba(255, 153, 51, 0.4);
    border-radius: var(--radius-lg);
    padding: 3rem;
    text-align: center;
    margin: 1.5rem 0;
    position: relative;
    overflow: hidden;
    animation: scaleIn 0.4s ease-out;
    box-shadow: var(--shadow-card), inset 0 1px 0 rgba(255,255,255,0.05);
    contain: layout paint;
    transform: translateZ(0);
}
.quiz-question-card::before {
    content: '';
    position: absolute;
    top: 0; left: 0; right: 0;
    height: 2px;
    background: linear-gradient(90deg, transparent, var(--gold), transparent);
}
.quiz-question-prompt {
    color: var(--text-secondary);
    font-size: 1rem;
    text-transform: uppercase;
    letter-spacing: 0.15em;
    margin-bottom: 0.5rem;
    font-weight: 500;
    display: flex;
    align-items: center;
    justify-content: center;
    gap: 0.5rem;
}
.quiz-char-display {
    font-size: 6rem;
    font-weight: 800;
    color: var(--gold);
    filter: drop-shadow(0 0 25px rgba(255, 153, 51, 0.3));
    margin: 1rem 0;
    animation: scaleIn 0.5s ease-out;
}
.quiz-text-display {
    font-family: 'Cinzel', serif;
    font-size: 2.5rem;
    color: var(--text-primary);
    font-weight: 700;
    margin: 1rem 0 2rem 0;
    animation: scaleIn 0.5s ease-out;
}

.answer-card {
    display: flex;
    align-items: center;
    justify-content: center;
    font-weight: 600;
    padding: 0.8rem 1.2rem;
    border-radius: var(--radius-sm);
    min-height: 52px;
    margin: 0 0 1rem 0;
    line-height: 1.6;
    width: 100%;
    box-sizing: border-box;
    transition: var(--transition);
    font-size: 1.05rem;
    contain: layout paint;
    will-change: transform;
}

.quiz-question-frame {
    animation: stateSoftIn 0.22s ease-out;
    transform: translateZ(0);
}
.quiz-answer-frame {
    animation: stateSoftIn 0.18s ease-out;
}
@keyframes stateSoftIn {
    from { opacity: 0.72; transform: translateY(6px); }
    to { opacity: 1; transform: translateY(0); }
}
.answer-correct {
    background: rgba(34, 197, 94, 0.1);
    border: 2px solid rgba(34, 197, 94, 0.5);
    color: #4ade80;
    animation: correctPulse 0.8s ease-out;
    box-shadow: 0 0 20px rgba(34, 197, 94, 0.1);
}
.answer-wrong {
    background: rgba(239, 68, 68, 0.1);
    border: 2px solid rgba(239, 68, 68, 0.5);
    color: #f87171;
    animation: wrongShake 0.5s ease-out;
    box-shadow: 0 0 20px rgba(239, 68, 68, 0.1);
}
.answer-dim {
    background: transparent;
    border: 1px solid rgba(255, 255, 255, 0.06);
    color: rgba(250, 250, 250, 0.25);
}

.quiz-feedback {
    text-align: center;
    padding: 1.2rem;
    border-radius: var(--radius);
    margin: 1rem 0;
    font-weight: 700;
    font-size: 1.1rem;
    animation: scaleIn 0.3s ease-out;
    display: flex;
    align-items: center;
    justify-content: center;
    gap: 0.6rem;
}
.quiz-feedback.correct {
    background: rgba(34, 197, 94, 0.1);
    backdrop-filter: blur(12px);
    -webkit-backdrop-filter: blur(12px);
    border: 1px solid rgba(34, 197, 94, 0.3);
    border-top: 1px solid rgba(34, 197, 94, 0.5);
    color: #4ade80;
    box-shadow: 0 0 30px rgba(34, 197, 94, 0.15);
}
.quiz-feedback.wrong {
    background: rgba(239, 68, 68, 0.1);
    backdrop-filter: blur(12px);
    -webkit-backdrop-filter: blur(12px);
    border: 1px solid rgba(239, 68, 68, 0.3);
    border-top: 1px solid rgba(239, 68, 68, 0.5);
    color: #f87171;
    box-shadow: 0 0 30px rgba(239, 68, 68, 0.15);
}

/* ─── LIBRARY ─── */
.library-section {
    margin-bottom: 2.5rem;
    animation: fadeInUp 0.5s ease-out;
}
.library-title {
    font-family: 'Cinzel', serif;
    font-size: 1.5rem;
    font-weight: 700;
    color: var(--gold);
    margin-bottom: 1.2rem;
    border-bottom: 1px solid rgba(255, 153, 51, 0.15);
    padding-bottom: 0.8rem;
    letter-spacing: 0.05em;
}

.char-card-ui {
    background: var(--surface);
    border: 1px solid var(--border);
    border-radius: var(--radius);
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
    transition: var(--transition);
}
.char-card-ui::before {
    content: '';
    position: absolute;
    top: 0; left: 0; right: 0;
    height: 2px;
    background: linear-gradient(90deg, transparent, var(--gold), transparent);
    opacity: 0;
    transition: opacity 0.4s ease;
}
.char-card-ui:hover {
    background: var(--surface-hover);
    transform: translateY(-6px) scale(1.03);
    box-shadow: var(--shadow-hover);
    border-color: var(--border-hover);
}
.char-card-ui:hover::before { opacity: 1; }

.char-main-txt {
    font-size: 3.5rem;
    font-weight: 800;
    color: var(--gold);
    margin: 0;
    position: absolute;
    top: 50%; left: 50%;
    transform: translate(-50%, -50%) scale(1);
    transition: var(--transition);
    line-height: 1;
    transform-origin: center center;
    filter: drop-shadow(0 0 8px rgba(255, 153, 51, 0.2));
}
.char-card-ui:hover .char-main-txt {
    top: 23px;
    transform: translate(-50%, -50%) scale(0.42);
    color: var(--gold-light);
}

.char-details-ui {
    position: absolute;
    top: 45px; left: 0;
    width: 100%;
    max-height: 108px;
    overflow-y: auto;
    overflow-x: hidden;
    opacity: 0;
    padding: 0 12px 10px 12px;
    box-sizing: border-box;
    transform: translateY(15px);
    transition: var(--transition);
    display: flex;
    flex-direction: column;
    align-items: center;
    pointer-events: none;
    scrollbar-width: none;
    -ms-overflow-style: none;
    overscroll-behavior: contain;
    mask-image: linear-gradient(to bottom, #000 82%, transparent 100%);
    -webkit-mask-image: linear-gradient(to bottom, #000 82%, transparent 100%);
}
.char-details-ui::-webkit-scrollbar {
    display: none;
}
.char-card-ui:hover .char-details-ui {
    opacity: 1;
    transform: translateY(0);
    pointer-events: auto;
}

.cd-name { font-weight: 700; color: var(--text-primary); margin-bottom: 2px; font-size: 0.84rem; line-height: 1.2; }
.cd-pron { font-style: italic; color: var(--text-secondary); font-size: 0.7rem; margin-bottom: 5px; line-height: 1.2; }
.cd-ex { color: var(--gold); font-weight: 600; font-size: 0.72rem; margin-bottom: 4px; line-height: 1.25; }
.cd-note { font-size: 0.62rem; color: var(--text-muted); line-height: 1.25; }

/* ─── FOOTER ─── */
.block-container {
    animation: appStateIn 0.18s ease-out;
}
.stSpinner {
    animation: appStateIn 0.16s ease-out;
}
.stSpinner::after {
    content: "Akshar.AI";
    display: inline-block;
    margin-left: 0.55rem;
    color: var(--gold);
    font-family: 'Cinzel', serif;
    font-size: 0.72rem;
    font-weight: 700;
    letter-spacing: 0.14em;
}
@keyframes appStateIn {
    from { opacity: 0.84; transform: translateY(4px); }
    to { opacity: 1; transform: translateY(0); }
}

.app-footer {
    text-align: center;
    padding: 3rem 0 1rem 0;
    color: var(--text-muted);
    font-size: 0.8rem;
    letter-spacing: 0.1em;
    border-top: 1px solid rgba(255, 153, 51, 0.08);
    margin-top: 3rem;
    display: flex;
    align-items: center;
    justify-content: center;
    gap: 0.4rem;
}

/* ─── IMAGE STYLING ─── */
[data-testid="stImage"] {
    border-radius: var(--radius) !important;
    overflow: hidden;
}
[data-testid="stImage"] img {
    border-radius: var(--radius) !important;
    transition: var(--transition);
}
[data-testid="stImage"]:hover img {
    transform: scale(1.01);
}

/* ─── QUIZ & LEARN CAT CARDS ─── */
.premium-cat-card {
    background: linear-gradient(135deg, rgba(255, 255, 255, 0.03), var(--surface));
    backdrop-filter: blur(24px) saturate(120%);
    -webkit-backdrop-filter: blur(24px) saturate(120%);
    border: 1px solid var(--border);
    border-top: 1px solid rgba(255, 153, 51, 0.3);
    border-radius: var(--radius-lg);
    padding: 2rem 1.5rem;
    text-align: center;
    cursor: pointer;
    transition: var(--transition);
    position: relative;
    overflow: hidden;
    height: 100%;
    display: flex;
    flex-direction: column;
    justify-content: center;
    align-items: center;
    box-shadow: var(--shadow-card), inset 0 1px 0 rgba(255,255,255,0.05);
}
.premium-cat-card::before {
    content: '';
    position: absolute;
    top: 0; left: 0; right: 0;
    height: 3px;
    background: linear-gradient(90deg, transparent, var(--gold), transparent);
    opacity: 0;
    transition: opacity 0.4s ease;
}
.premium-cat-card:hover {
    background: var(--surface-hover);
    border-color: var(--border-hover);
    transform: translateY(-5px);
    box-shadow: var(--shadow-hover);
}
.premium-cat-card:hover::before { opacity: 1; }
.premium-cat-card.active {
    background: rgba(255, 153, 51, 0.1);
    border-color: var(--gold);
    border-top-color: rgba(255, 153, 51, 0.6);
    box-shadow: 0 0 35px rgba(255, 153, 51, 0.2);
}
.premium-cat-card.active::before { opacity: 1; }

.cat-icon-container {
    background: rgba(255, 153, 51, 0.1);
    width: 64px;
    height: 64px;
    border-radius: 50%;
    display: flex;
    align-items: center;
    justify-content: center;
    margin: 0 auto 1.2rem auto;
    color: var(--gold);
    border: 1px solid rgba(255, 153, 51, 0.2);
    box-shadow: 0 0 20px rgba(255, 153, 51, 0.1);
    transition: var(--transition);
}
.premium-cat-card:hover .cat-icon-container {
    transform: scale(1.1);
    background: rgba(255, 153, 51, 0.2);
}
.cat-title {
    font-family: 'Cinzel', serif;
    font-size: 1.35rem;
    font-weight: 700;
    color: var(--text-primary);
    margin-bottom: 0.5rem;
}
.cat-subtitle {
    font-size: 0.85rem;
    color: var(--text-muted);
    margin-bottom: 1.2rem;
    line-height: 1.4;
}
.cat-progress-text {
    font-size: 0.75rem;
    font-weight: 600;
    color: var(--gold);
    text-transform: uppercase;
    letter-spacing: 0.1em;
    margin-bottom: 0.4rem;
}
.cat-progress-bar {
    width: 100%;
    height: 6px;
    background: rgba(255, 255, 255, 0.08);
    border-radius: 999px;
    overflow: hidden;
}
.cat-progress-fill {
    height: 100%;
    background: linear-gradient(90deg, #CC7A29, #FF9933);
    border-radius: 999px;
    transition: width 1s cubic-bezier(0.25, 0.8, 0.25, 1);
}

/* ─── QUIZ SPECIFIC ─── */
.quiz-mode-badge {
    position: absolute;
    top: 15px;
    right: 15px;
    background: rgba(255, 153, 51, 0.15);
    border: 1px solid rgba(255, 153, 51, 0.3);
    color: var(--gold);
    font-size: 0.7rem;
    font-weight: 600;
    text-transform: uppercase;
    letter-spacing: 0.1em;
    padding: 4px 10px;
    border-radius: 20px;
    backdrop-filter: blur(4px);
}
.quiz-hint-box {
    background: linear-gradient(135deg, rgba(91, 155, 213, 0.08), rgba(26, 13, 6, 0.2));
    backdrop-filter: blur(20px);
    -webkit-backdrop-filter: blur(20px);
    border: 1px solid rgba(91, 155, 213, 0.3);
    border-top: 1px solid rgba(91, 155, 213, 0.5);
    border-radius: var(--radius-sm);
    padding: 1.5rem;
    margin: 1rem auto;
    max-width: 400px;
    box-shadow: var(--shadow-card);
}
.quiz-image-container {
    width: 250px;
    height: 250px;
    margin: 1.5rem auto;
    border-radius: var(--radius-sm);
    border: 2px solid rgba(255, 153, 51, 0.2);
    overflow: hidden;
    background: rgba(255, 255, 255, 0.02);
    box-shadow: 0 8px 32px rgba(0, 0, 0, 0.3);
}
.quiz-image-container img {
    width: 100%;
    height: 100%;
    object-fit: contain;
}

/* ─── GLOBAL RESPONSIVE OVERRIDES ─── */
@media (max-width: 1200px) {
    .hero-card {
        max-width: 100%;
        padding: 2.2rem 1.5rem 1.8rem 1.5rem;
    }
    .hero-subtitle {
        font-size: clamp(1rem, 2.2vw, 1.35rem);
        letter-spacing: 0.09em;
    }
    .katha-section {
        padding: 2.2rem;
    }
}

@media (max-width: 992px) {
    .stTabs [data-baseweb="tab-list"] {
        flex-wrap: wrap;
        justify-content: stretch;
        gap: 0.4rem;
        padding: 0.45rem;
    }
    .stTabs [data-baseweb="tab"] {
        flex: 1 1 calc(50% - 0.4rem);
        min-width: 150px;
        justify-content: center;
        padding: 0.65rem 1rem;
        font-size: 0.9rem;
    }
    .katha-hero {
        padding: 2.4rem 1.2rem;
    }
    .katha-section {
        padding: 1.5rem;
    }
    .stat-row, .team-row {
        gap: 14px;
    }
    .quiz-image-container {
        width: min(68vw, 260px);
        height: min(68vw, 260px);
    }
}

@media (max-width: 768px) {

    /* Force Streamlit columns to stack on phones/tablets */
    [data-testid="stHorizontalBlock"] {
        gap: 0.75rem !important;
        flex-wrap: wrap !important;
    }
    [data-testid="stHorizontalBlock"] > [data-testid="column"] {
        width: 100% !important;
        flex: 1 1 100% !important;
        min-width: 100% !important;
    }

    .hero-card {
        margin: 1rem auto 1.5rem auto;
        padding: 1.4rem 1rem 1.1rem 1rem;
        border-radius: 16px;
    }
    .hero-subtitle {
        font-size: 0.88rem;
        letter-spacing: 0.08em;
        margin-bottom: 1.1rem;
    }
    .gold-divider {
        width: 78%;
        margin-bottom: 1.2rem;
    }

    .katha-hero {
        margin: 1rem auto;
        padding: 1.8rem 0.95rem 1.2rem 0.95rem;
    }
    .katha-eyebrow {
        font-size: 0.9rem;
        letter-spacing: 0.12em;
    }
    .katha-subtitle {
        margin-bottom: 1.7rem;
        line-height: 1.6;
    }
    .katha-section {
        margin-bottom: 1.4rem;
        padding: 1rem;
        border-radius: 14px;
    }
    .section-body {
        font-size: 0.92rem;
        line-height: 1.72;
    }
    .katha-quote {
        padding: 12px 14px;
        margin: 18px 0;
    }

    .premium-cat-card,
    .closing-card,
    .team-card,
    .stat-card {
        border-radius: 14px;
    }
    .premium-cat-card {
        padding: 1.2rem 0.95rem;
    }
    .cat-title {
        font-size: 1.05rem;
    }
    .cat-subtitle {
        font-size: 0.8rem;
        margin-bottom: 0.9rem;
    }

    .char-grid {
        gap: 9px;
    }
    .char-card-ui {
        min-height: 125px;
        padding: 0.6rem;
    }
    .char-main-txt {
        font-size: 2.1rem;
    }

    .quiz-feedback {
        font-size: 0.95rem;
        padding: 0.9rem;
    }
    .quiz-hint-box {
        padding: 1rem;
        max-width: 100%;
    }

    .app-footer {
        flex-wrap: wrap;
        gap: 0.25rem;
        font-size: 0.72rem;
        letter-spacing: 0.06em;
        text-align: center;
        padding-top: 2rem;
    }

    [data-testid="stButton"] button,
    .stDownloadButton > button,
    [data-testid="baseButton-secondary"] {
        width: 100% !important;
        min-height: 42px;
    }
}

@media (max-width: 480px) {
    .stTabs [data-baseweb="tab"] {
        flex: 1 1 100%;
        min-width: 100%;
        font-size: 0.86rem;
        padding: 0.56rem 0.7rem;
    }
    .hero-subtitle {
        font-size: 0.8rem;
        letter-spacing: 0.05em;
    }
    .quiz-image-container {
        width: min(78vw, 220px);
        height: min(78vw, 220px);
    }
    .katha-title {
        font-size: clamp(1.5rem, 8vw, 2rem);
    }
    .katha-section {
        padding: 0.9rem;
    }
    .section-heading-katha {
        font-size: 1.2rem;
    }
}


/* ─── HIDE STREAMLIT DEFAULTS ─── */
#MainMenu {visibility: hidden;}
footer {visibility: hidden;}
header {visibility: hidden;}
</style>
"""
st.markdown(css, unsafe_allow_html=True)

# ─── HERO HEADER ─────────────────────────────────────────────────────────────────
st.markdown("""
<div class="hero-card">
    <div class="hero-title">
        <h1>Akshar.AI</h1>
    </div>
    <div class="hero-subtitle">Every Letter, Alive Again</div>
    <div class="gold-divider"></div>
</div>
""", unsafe_allow_html=True)

# ─── TABS WITH ICONS (inline SVG injected via CSS pseudo or adjacent span) ──────
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "Katha",
    "Recognize",
    "Learn",
    "Quiz",
    "Library"
])

# Inject SVG icons into tab headers via JS
tab_icons_html = f"""
<script>
(function() {{
    const tabs = window.parent.document.querySelectorAll('[data-baseweb="tab"] > div > p');
    const icons = {{
        'Recognize': `{icon("scan", 16)}`,
        'Learn': `{icon("book", 16)}`,
        'Quiz': `{icon("trophy", 16)}`,
        'Library': `{icon("library", 16)}`,
        'Katha': `{icon("scroll_text", 16)}`
    }};
    tabs.forEach(tab => {{
        const text = tab.textContent.trim();
        if (icons[text] && !tab.querySelector('svg')) {{
            tab.innerHTML = icons[text] + '&nbsp;&nbsp;' + text;
            tab.style.display = 'flex';
            tab.style.alignItems = 'center';
            tab.style.gap = '0.4rem';
        }}
    }});
}})();
</script>
"""
import streamlit.components.v1 as components
st.html(tab_icons_html)

@st.cache_resource
def initialize_system():
    if not os.path.exists(MODEL_PATH) or not os.path.exists(LABELS_PATH) or not os.path.exists(IDX_TO_CLASS_PATH):
        return None, {}, {}
    return load_tflite_model(MODEL_PATH, LABELS_PATH, IDX_TO_CLASS_PATH)

@st.cache_resource
def load_keras_model():
    import tensorflow as tf
    try:
        return tf.keras.models.load_model(
            KERAS_MODEL_PATH,
            compile=False   # 🔥 THIS FIXES YOUR ERROR
        )
    except Exception as e:
        print("GradCAM model load error:", e)
        return None

@st.cache_resource
def get_cached_last_conv_layer(_keras_model):
    if _keras_model is None:
        return None
    return get_last_conv_layer(_keras_model)

# keras_model_result = load_keras_model()
keras_model = load_keras_model()

if keras_model is None:
    st.warning("Grad-CAM model could not be loaded.")

last_conv_layer = get_cached_last_conv_layer(keras_model)

interpreter, modi_labels, idx_to_class = initialize_system()

if interpreter is None:
    st.error(f"Cannot find model or data files. Please ensure `{MODEL_PATH}`, `{LABELS_PATH}`, and `{IDX_TO_CLASS_PATH}` exist.")


# ═══════════════════════════════════════════════════════════════════════════════════
#  TAB 1 — KATHA
# ═══════════════════════════════════════════════════════════════════════════════════
with tab1:
    st.markdown("""
    <style>
    /* ── Katha tab full styles ── */
    .katha-hero {
        text-align: center;
        padding: 60px 40px;
        position: relative;
        background: linear-gradient(135deg, rgba(255, 255, 255, 0.03), rgba(26, 13, 6, 0.15));
        backdrop-filter: blur(28px) saturate(140%);
        -webkit-backdrop-filter: blur(28px) saturate(140%);
        border: 1px solid rgba(255, 153, 51, 0.15);
        border-top: 1px solid rgba(255, 153, 51, 0.35);
        border-radius: 20px;
        box-shadow: 0 16px 40px rgba(0,0,0,0.4), inset 0 1px 0 rgba(255,255,255,0.05);
        margin: 2rem auto;
        max-width: 900px;
    }
    .katha-eyebrow {
        font-size: 1.75rem;
        letter-spacing: 0.25em;
        color: var(--gold-dark);
        text-transform: uppercase;
        margin-bottom: 16px;
        font-family: serif;
    }
    .katha-title {
        font-size: clamp(2rem, 6vw, 4rem);
        font-weight: 700;
        color: var(--gold);
        font-family: 'Cinzel', serif;
        line-height: 1.15;
        margin-bottom: 20px;
        text-shadow: 0 2px 24px rgba(255,153,51,0.2);
    }
    .katha-subtitle {
        font-size: clamp(0.95rem, 2.5vw, 1.25rem);
        color: var(--text-secondary);
        max-width: 640px;
        margin: 0 auto 48px;
        line-height: 1.7;
        font-family: 'Inter', sans-serif;
    }
    .katha-divider {
        width: 80px;
        height: 3px;
        background: linear-gradient(90deg, transparent, var(--gold-dark), transparent);
        margin: 0 auto 48px;
        border-radius: 2px;
    }

    /* ── Stat cards ── */
    .stat-row {
        display: flex;
        justify-content: center;
        flex-wrap: wrap;
        gap: 20px;
        margin: 40px auto;
        max-width: 900px;
    }
    .stat-card {
        background: linear-gradient(135deg, rgba(255, 255, 255, 0.03), var(--surface));
        border: 1px solid var(--border);
        border-top: 1px solid rgba(255, 153, 51, 0.35);
        border-radius: 16px;
        padding: 28px 32px;
        text-align: center;
        flex: 1 1 200px;
        backdrop-filter: blur(28px) saturate(140%);
        -webkit-backdrop-filter: blur(28px) saturate(140%);
        box-shadow: var(--shadow-card), inset 0 1px 0 rgba(255,255,255,0.05);
        transition: var(--transition);
    }
    .stat-card:hover {
        border-color: var(--border-hover);
        transform: translateY(-4px);
        box-shadow: var(--shadow-hover);
    }
    .stat-number {
        font-size: 2.2rem;
        font-weight: 700;
        color: var(--gold);
        font-family: 'Cinzel', serif;
        display: block;
    }
    .stat-label {
        font-size: 0.8rem;
        color: var(--text-muted);
        text-transform: uppercase;
        letter-spacing: 0.1em;
        margin-top: 6px;
    }

    /* ── Section blocks ── */
    .katha-section {
        max-width: 860px;
        margin: 0 auto 56px;
        padding: 3.5rem;
        background: linear-gradient(135deg, rgba(255, 255, 255, 0.03), rgba(26, 13, 6, 0.15));
        backdrop-filter: blur(28px) saturate(140%);
        -webkit-backdrop-filter: blur(28px) saturate(140%);
        border: 1px solid rgba(255, 153, 51, 0.15);
        border-top: 1px solid rgba(255, 153, 51, 0.35);
        border-radius: 20px;
        box-shadow: 0 16px 40px rgba(0,0,0,0.4), inset 0 1px 0 rgba(255,255,255,0.05);
    }
    .section-tag {
        font-size: 1.0rem;
        letter-spacing: 0.2em;
        color: var(--gold-dark);
        text-transform: uppercase;
        margin-bottom: 10px;
        font-family: 'Inter', sans-serif;
    }
    .section-heading-katha {
        font-size: clamp(1.4rem, 3.5vw, 2.1rem);
        color: var(--gold);
        font-family: 'Cinzel', serif;
        font-weight: 600;
        margin-bottom: 16px;
        line-height: 1.3;
    }
    .section-body {
        font-size: 1rem;
        color: var(--text-secondary);
        line-height: 1.85;
        font-family: 'Inter', sans-serif;
    }
    .section-body strong {
        color: var(--gold);
    }

    /* ── Problem cards ── */
    .problem-grid {
        display: grid;
        grid-template-columns: repeat(auto-fit, minmax(240px, 1fr));
        gap: 16px;
        margin-top: 28px;
    }
    .problem-card {
        background: rgba(239, 68, 68, 0.05);
        border: 1px solid rgba(239, 68, 68, 0.15);
        border-radius: 12px;
        padding: 20px 22px;
        backdrop-filter: blur(4px);
        transition: var(--transition);
    }
    .problem-card:hover {
        border-color: var(--border-hover);
        transform: translateY(-4px);
        box-shadow: var(--shadow-hover);
    }
    .problem-card .p-icon {
        margin-bottom: 10px;
        display: block;
        color: #f87171;
    }
    .problem-card .p-title {
        font-size: 0.95rem;
        font-weight: 600;
        color: var(--gold);
        margin-bottom: 6px;
        font-family: 'Inter', sans-serif;
    }
    .problem-card .p-body {
        font-size: 0.85rem;
        color: var(--text-muted);
        line-height: 1.6;
    }

    /* ── Solution features ── */
    .feature-grid {
        display: grid;
        grid-template-columns: repeat(auto-fit, minmax(220px, 1fr));
        gap: 16px;
        margin-top: 28px;
    }
    .feature-card {
        background: var(--surface);
        border: 1px solid var(--border);
        border-radius: 12px;
        padding: 22px;
        transition: var(--transition);
    }
    .feature-card:hover {
        background: var(--surface-hover);
        border-color: var(--border-hover);
        box-shadow: 0 4px 12px rgba(255, 153, 51, 0.05);
    }
    .feature-card .f-icon {
        margin-bottom: 10px;
        display: block;
        color: var(--gold);
    }
    .feature-card .f-title {
        font-size: 0.95rem;
        font-weight: 600;
        color: var(--gold);
        margin-bottom: 5px;
        font-family: 'Inter', sans-serif;
    }
    .feature-card .f-body {
        font-size: 0.82rem;
        color: var(--text-secondary);
        line-height: 1.6;
    }

    /* ── Tech stack pills ── */
    .tech-row {
        display: flex;
        flex-wrap: wrap;
        gap: 10px;
        margin-top: 20px;
    }
    .tech-pill {
        background: var(--surface);
        border: 1px solid var(--border);
        border-radius: 20px;
        padding: 6px 16px;
        font-size: 0.82rem;
        color: var(--text-secondary);
        font-family: monospace;
    }

    /* ── Quote block ── */
    .katha-quote {
        border-left: 3px solid var(--gold-dark);
        padding: 16px 24px;
        margin: 32px 0;
        background: var(--surface);
        border-radius: 0 10px 10px 0;
    }
    .katha-quote p {
        font-size: 1.05rem;
        color: var(--text-secondary);
        font-style: italic;
        font-family: 'Georgia', serif;
        line-height: 1.7;
        margin: 0;
    }
    .katha-quote cite {
        font-size: 0.78rem;
        color: var(--text-muted);
        display: block;
        margin-top: 8px;
        font-style: normal;
    }

    /* ── Team cards ── */
    .team-row {
        display: flex;
        justify-content: center;
        flex-wrap: wrap;
        gap: 20px;
        margin-top: 28px;
    }
    .team-card {
        background: var(--surface);
        border: 1px solid var(--border);
        border-radius: 16px;
        padding: 24px 28px;
        text-align: center;
        flex: 1 1 200px;
        min-width: 180px;
        transition: var(--transition);
    }
    .team-card:hover {
        border-color: var(--border-hover);
        transform: translateY(-3px);
        box-shadow: var(--shadow-hover);
    }
    .team-avatar {
        width: 100px;
        height: 100px;
        background: linear-gradient(135deg, var(--gold-dark), #7a4e1a);
        border-radius: 50%;
        margin: 0 auto 16px;
        display: flex;
        align-items: center;
        justify-content: center;
        font-size: 1.5rem;
        color: #1a0e05;
        font-family: 'Inter', sans-serif;
        font-weight: 700;
        overflow: hidden;
        border: 2px solid rgba(255, 153, 51, 0.4);
        box-shadow: 0 8px 20px rgba(255, 153, 51, 0.15);
    }
    .team-avatar img {
        width: 100%;
        height: 100%;
        object-fit: cover;
    }
    .team-name {
        font-size: 0.95rem;
        font-weight: 600;
        color: var(--gold);
        font-family: 'Inter', sans-serif;
    }
    

    /* ── Closing message ── */
    .closing-card {
        max-width: 680px;
        margin: 0 auto;
        background: var(--surface);
        border: 1px solid var(--border);
        border-radius: 20px;
        padding: 40px 36px;
        text-align: center;
    }
    .closing-title {
        font-size: 1.4rem;
        color: var(--gold);
        font-family: 'Cinzel', serif;
        font-weight: 600;
        margin-bottom: 16px;
    }
    .closing-body {
        font-size: 0.95rem;
        color: var(--text-muted);
        line-height: 1.85;
        font-family: 'Inter', sans-serif;
    }
    .closing-body em {
        color: var(--text-secondary);
        font-style: italic;
    }
    .opensource-badge {
        display: inline-flex;
        align-items: center;
        gap: 8px;
        background: var(--surface);
        border: 1px solid var(--border);
        border-radius: 24px;
        padding: 8px 20px;
        margin-top: 24px;
        font-size: 0.85rem;
        color: var(--gold-dark);
        font-family: 'Inter', sans-serif;
        font-weight: 500;
    }

    /* ── Responsive ── */
    @media (max-width: 768px) {
        .katha-hero { padding: 40px 10px 20px; }
        .katha-eyebrow { font-size: 1.2rem; }
        .stat-card { min-width: 140px; padding: 18px 16px; }
        .katha-section { padding: 0 10px; }
        .closing-card { padding: 28px 18px; }
        .team-row { gap: 12px; }
        .team-card { min-width: 140px; padding: 18px 16px; }
    }
    </style>
    """, unsafe_allow_html=True)

    # ── HERO ─────────────────────────────────────────────────────────
    st.markdown("""
    <div class="katha-hero">
        <div class="katha-eyebrow">कथा &nbsp;&middot;&nbsp; Story</div>
        <div class="katha-title">
            700 Years of History<br>Locked in Silence
        </div>
        <div class="katha-subtitle">
            Millions of historical documents written in Modi script — 
            land records, royal orders, family letters — remain completely 
            unreadable to the people they belong to.
            Until now.
        </div>
        <div class="katha-divider"></div>
    </div>
    """, unsafe_allow_html=True)

    # ── STATS ────────────────────────────────────────────────────────
    st.markdown("""
    <div class="stat-row">
        <div class="stat-card">
            <span class="stat-number">21L+</span>
            <div class="stat-label">Modi manuscripts identified in Maharashtra alone</div>
        </div>
        <div class="stat-card">
            <span class="stat-number">&lt;1000</span>
            <div class="stat-label">Scholars alive who can read Modi fluently</div>
        </div>
        <div class="stat-card">
            <span class="stat-number">1950</span>
            <div class="stat-label">The year it was replaced — within living memory</div>
        </div>
        <div class="stat-card">
            <span class="stat-number">₹0</span>
            <div class="stat-label">Cost to use Akshar.AI — forever free</div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    # ── THE STORY ────────────────────────────────────────────────────
    st.markdown("""
    <div class="katha-section">
        <div class="section-tag">The Script</div>
        <div class="section-heading-katha">What is Modi Script?</div>
        <div class="section-body">
            Modi (𑘦𑘻𑘚𑘲) is a cursive script developed to write the Marathi language. 
            Used from the <strong>13th century</strong> through the reign of 
            <strong>Chhatrapati Shivaji Maharaj</strong>, across the Peshwa era, 
            and into British colonial rule — it was Maharashtra's living written language 
            for over 700 years.<br><br>
            Land ownership records, military correspondence, financial ledgers, 
            village histories, personal letters. The administrative heartbeat of an 
            entire civilization was written in Modi.<br><br>
            In <strong>1950</strong>, the newly independent Indian government standardized 
            Devanagari as the script for Marathi. Within one generation, Modi vanished 
            from schools. Today, anyone born after 1960 — including most of the people 
            whose ancestors wrote in Modi — cannot read their own family's history.
        </div>
        <div class="katha-quote">
            <p>"A family in Satara holds a Modi-script Watan document proving their 
            ancestral land ownership from 1820. A court case is filed. The document is 
            authentic — but unreadable. The land is lost not because the proof does not 
            exist, but because it cannot be read."</p>
            <cite>— A real scenario playing out in rural Maharashtra today</cite>
        </div>
    </div>
    """, unsafe_allow_html=True)

    # ── THE PROBLEM ──────────────────────────────────────────────────
    st.markdown(f"""
    <div class="katha-section">
        <div class="section-tag">The Problem</div>
        <div class="section-heading-katha">What is Being Lost</div>
        <div class="problem-grid">
            <div class="problem-card">
                <span class="p-icon">{icon("building", 24)}</span>
                <div class="p-title">Inaccessible Archives</div>
                <div class="p-body">Bharat Itihas Sanshodhan Mandal in Pune holds the largest 
                accessible Modi archive in the world. Thousands of documents sit undigitized, 
                unstudied, deteriorating with every monsoon.</div>
            </div>
            <div class="problem-card">
                <span class="p-icon">{icon("scale", 24)}</span>
                <div class="p-title">Lost Legal Rights</div>
                <div class="p-body">Land disputes across Maharashtra hinge on Modi documents 
                that families cannot read. Professional translators charge ₹15,000–₹40,000 
                per document. Many cannot afford it.</div>
            </div>
            <div class="problem-card">
                <span class="p-icon">{icon("scroll_text", 24)}</span>
                <div class="p-title">Silenced Family History</div>
                <div class="p-body">Millions of Indian families have letters written by 
                grandparents in Modi — final words, family wisdom, property instructions. 
                These letters sit in boxes, completely silent.</div>
            </div>
            <div class="problem-card">
                <span class="p-icon">{icon("microscope", 24)}</span>
                <div class="p-title">Dying Knowledge</div>
                <div class="p-body">Fewer than 1,000 scholars worldwide can read Modi fluently. 
                Each passing year, that number falls. The window to digitize this knowledge 
                is closing faster than it is being used.</div>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    # ── THE SOLUTION ─────────────────────────────────────────────────
    st.markdown(f"""
    <div class="katha-section">
        <div class="section-tag">Our Answer</div>
        <div class="section-heading-katha">What Akshar.AI Does</div>
        <div class="section-body">
            Akshar.AI is the first working, deployed AI application that recognizes 
            handwritten Modi characters, transliterates them to Devanagari, explains 
            its reasoning visually — and <strong>teaches you to read Modi yourself</strong>.<br><br>
            Not a research paper. Not a prototype. A live tool, accessible to anyone 
            with a phone, that works right now.
        </div>
        <div class="feature-grid">
            <div class="feature-card">
                <span class="f-icon">{icon("scan", 20)}</span>
                <div class="f-title">Recognize</div>
                <div class="f-body">Upload any Modi character image. Get the Devanagari 
                equivalent, confidence score, and a heatmap showing exactly 
                which strokes the AI read.</div>
            </div>
            <div class="feature-card">
                <span class="f-icon">{icon("book", 20)}</span>
                <div class="f-title">Learn</div>
                <div class="f-body">Progressive lessons through all 57 characters — vowels, 
                consonants, numerals. People pay thousands to learn this. We made it free.</div>
            </div>
            <div class="feature-card">
                <span class="f-icon">{icon("target", 20)}</span>
                <div class="f-title">Quiz</div>
                <div class="f-body">Test your knowledge with gamified quizzes — by character 
                type, difficulty, and score tracking. Learning sticks when it is tested.</div>
            </div>
            <div class="feature-card">
                <span class="f-icon">{icon("library", 20)}</span>
                <div class="f-title">Library</div>
                <div class="f-body">All 57 Modi characters with Devanagari equivalents, 
                pronunciation, example words, and historical notes — in one 
                searchable, filterable reference.</div>
            </div>
            <div class="feature-card">
                <span class="f-icon">{icon("globe", 20)}</span>
                <div class="f-title">Completely Free</div>
                <div class="f-body">No subscription. No paywall. No account needed. 
                A historian in a remote village has the same access as a scholar 
                at a university. That is the point.</div>
            </div>
            <div class="feature-card">
                <span class="f-icon">{icon("smartphone", 20)}</span>
                <div class="f-title">Works Offline</div>
                <div class="f-body">The AI model runs on-device via TFLite. No internet 
                connection needed for inference. A scholar in an archive with no WiFi 
                can still use it.</div>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    # ── TECH STACK ───────────────────────────────────────────────────
    st.markdown(f"""
    <div class="katha-section">
        <div class="section-tag">Built With</div>
        <div class="section-heading-katha">The Technology Behind Akshar.AI</div>
        <div class="section-body">
            AksharAI is built on <strong>MobileNetV2</strong>, a convolutional neural 
            network pretrained on 14 million images, fine-tuned on the 
            <strong>MODI-HChar dataset</strong> — 5,75,920 handwritten character images 
            across 57 classes, published by researchers at KBCNMU, Maharashtra.<br><br>
            The model achieves <strong>92–95% validation accuracy</strong> after two-phase 
            transfer learning. Grad-CAM explainability makes every prediction transparent — 
            you see exactly which strokes the AI used to make its decision. 
            The final model is deployed via <strong>TFLite quantization</strong>, 
            reducing its size by 4× while maintaining accuracy within 1%.
        </div>
        <div class="tech-row">
            <span class="tech-pill">TensorFlow 2.x</span>
            <span class="tech-pill">MobileNetV2</span>
            <span class="tech-pill">TFLite</span>
            <span class="tech-pill">Grad-CAM</span>
            <span class="tech-pill">Transfer Learning</span>
            <span class="tech-pill">Streamlit</span>
            <span class="tech-pill">OpenCV</span>
            <span class="tech-pill">Python</span>
            <span class="tech-pill">MODI-HChar Dataset</span>
            <span class="tech-pill">Kaggle P100 GPU</span>
        </div>
        <div style="margin-top: 24px; text-align: center;">
            <span class="opensource-badge">
                {icon("star", 16, "#c8923a")} Open Source — Akshar.AI is free to use, inspect, and build upon
            </span>
        </div>
    </div>
    """, unsafe_allow_html=True)

    # ── TEAM ─────────────────────────────────────────────────────────
    devendra_b64 = ""
    raj_b64 = ""
    prasad_b64 = ""
    
    devendra_path = "Photos/Devendra.jpg"
    raj_path = "Photos/Raj_Veer.jpeg"
    prasad_path = "Photos/Prasad_Raibole.jpeg"
    
    if os.path.exists(devendra_path): devendra_b64 = get_file_b64(devendra_path, os.path.getmtime(devendra_path))
    if os.path.exists(raj_path): raj_b64 = get_file_b64(raj_path, os.path.getmtime(raj_path))
    if os.path.exists(prasad_path): prasad_b64 = get_file_b64(prasad_path, os.path.getmtime(prasad_path))

    st.markdown(f"""
    <div class="katha-section" style="text-align:center;">
        <div class="section-tag">The Builders</div>
        <div class="section-heading-katha">Who Made This</div>
        <div class="section-body" style="margin-bottom:0">
            Students from Pune, Maharashtra — built as a Google AI/ML 
            internship project, in 10 days, with no budget.
        </div>
        <div class="team-row">
            <div class="team-card">
                <div class="team-avatar">
{"<img src='data:image/jpeg;base64," + devendra_b64 + "' />" if devendra_b64 else "द"}
                </div>
                <div class="team-name">Devendra Suhas Rajhans</div>
            </div>
            <div class="team-card">
                <div class="team-avatar">
{"<img src='data:image/jpeg;base64," + raj_b64 + "' />" if raj_b64 else "र"}
                </div>
                <div class="team-name">Raj Dipak Veer</div>
            </div>
            <div class="team-card">
                <div class="team-avatar">
{"<img src='data:image/jpeg;base64," + prasad_b64 + "' />" if prasad_b64 else "प्र"}
                </div>
                <div class="team-name">Prasad Pramod Raibole</div>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    # ── CLOSING MESSAGE ──────────────────────────────────────────────
    st.markdown(f"""
    <div class="katha-section">
        <div class="closing-card">
            <div class="closing-title">A Note from the Developers</div>
            <div class="closing-body">
                We are students from Pune; the city which holds thousands of unread Modi documents. This script was written 
                by people whose descendants walk the same streets we do.<br><br>
                <em>We did not build Akshar.AI because it was assigned. We built it because 
                we realized that technology which has revolutionized how we read faces, 
                translate languages, and understand the world — had never been pointed 
                at one of our own forgotten scripts.</em><br><br>
                Every line of code in this project is an act of preservation. If AksharAI 
                helps one family read a letter their great-grandmother wrote, or helps one 
                historian digitize a document that would otherwise be lost, it was 
                worth every hour.<br><br>
                The script survived 700 years. We intend to help it survive 700 more.
            </div>
            <span class="opensource-badge" style="margin-top:20px;display:inline-block;">
                𑘦 Made with purpose in Pune, Maharashtra
            </span>
        </div>
    </div>
    <br><br>
    """, unsafe_allow_html=True)

# ═══════════════════════════════════════════════════════════════════════════════════
#  TAB 2 — RECOGNIZE
# ═══════════════════════════════════════════════════════════════════════════════════
with tab2:
    st.markdown(f"""
    <div class="section-card">
        <div class="section-heading">
            <span class="icon-inline icon-glow">{icon("scan", 22)}</span>
            Character Recognition
        </div>
        <div class="section-desc">
            Upload an image of a Modi script character and let our Model identify it instantly with Grad-CAM visualization.
        </div>
    </div>
    """, unsafe_allow_html=True)

    # Merged Upload Section
    col_upload_top_1, col_upload_top_2 = st.columns([1.2, 1], gap="large", vertical_alignment="center")

    with col_upload_top_1:
        st.markdown(f"""
            <div class="upload-label">
                <span class="icon-inline">{icon("upload", 20)}</span>
                Upload Modi Character
            </div>
            <div class="upload-sublabel">Drag and drop, browse, or capture a square photo from camera.</div>
        """, unsafe_allow_html=True)

    with col_upload_top_2:
        upload_btn_col, camera_btn_col = st.columns([2, 1], gap="small")
        with upload_btn_col:
            uploaded_file = st.file_uploader("Upload Image", type=["jpg", "jpeg", "png"], label_visibility="collapsed")
        with camera_btn_col:
            if st.button("Camera", key="open_recognize_camera", icon=":material/photo_camera:", width='stretch'):
                st.session_state.recognize_camera_open = True
                st.session_state.recognize_camera_pending = None
                st.rerun()

    camera_file = None
    if st.session_state.get("recognize_camera_open"):
        st.markdown(f"""
        <div class="recognize-source-card">
            <div class="recognize-source-title">
                <span class="icon-inline">{icon("scan", 16)}</span>
                Camera Capture
            </div>
            <div class="recognize-source-copy">Capture one clear character. The photo will be cropped to 1:1 before upload.</div>
        </div>
        """, unsafe_allow_html=True)
        close_cam_col1, close_cam_col2 = st.columns([4, 1])
        with close_cam_col2:
            if st.button("Close", icon=":material/close:", key="close_camera_top", width='stretch'):
                st.session_state.recognize_camera_open = False
                st.rerun()

        camera_nonce = st.session_state.get("recognize_camera_nonce", 0)
        camera_capture = st.camera_input("Capture square image", key=f"recognize_camera_capture_{camera_nonce}", label_visibility="collapsed")
        if camera_capture is not None:
            captured_square = crop_center_square(Image.open(camera_capture))
            st.image(captured_square, caption="Square preview", width=260)
            confirm_col, retake_col, cancel_col = st.columns(3)
            with confirm_col:
                if st.button("Use This Photo", key="confirm_recognize_camera", type="primary", width='stretch'):
                    buffer = io.BytesIO()
                    captured_square.save(buffer, format="PNG")
                    st.session_state.recognize_camera_image = buffer.getvalue()
                    st.session_state.recognize_camera_open = False
                    st.rerun()
            with retake_col:
                if st.button("Retake", key="retake_recognize_camera", width='stretch'):
                    st.session_state.recognize_camera_image = None
                    st.session_state.recognize_camera_nonce = camera_nonce + 1
                    st.rerun()
            with cancel_col:
                if st.button("Cancel", key="cancel_recognize_camera", width='stretch'):
                    st.session_state.recognize_camera_open = False
                    st.rerun()

    camera_image_bytes = st.session_state.get("recognize_camera_image")
    image_source = uploaded_file or (io.BytesIO(camera_image_bytes) if camera_image_bytes else None)
    image_source_name = "camera" if camera_image_bytes and not uploaded_file else "image"

    if image_source is None:
        st.markdown(f"""
        <div class="empty-state">
            <div class="empty-state-icon">{icon("feather", 48, "#6b5c47")}</div>
            <div class="empty-state-title">Awaiting your manuscript</div>
            <div class="empty-state-sub">Upload or capture a Modi character image above to begin recognition</div>
        </div>
        """, unsafe_allow_html=True)

    if image_source is not None:
        image_bytes = image_source.getvalue() if hasattr(image_source, "getvalue") else image_source.read()
        image_hash = hashlib.sha256(image_bytes).hexdigest()
        image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        if image_source_name == "camera":
            image = crop_center_square(image)
            camera_buffer = io.BytesIO()
            image.save(camera_buffer, format="PNG")
            image_hash = hashlib.sha256(camera_buffer.getvalue()).hexdigest()

        if interpreter is not None and modi_labels:
            with st.spinner("Analyzing your character..."):
                try:
                    cached_recognition = st.session_state.get("recognize_last_result")
                    if cached_recognition and cached_recognition.get("image_hash") == image_hash:
                        result = cached_recognition["result"]
                    else:
                        result = predict(image, interpreter, modi_labels, idx_to_class)
                        st.session_state.recognize_last_result = {
                            "image_hash": image_hash,
                            "result": result,
                        }

                    if not result['valid']:
                        st.error(f"⚠️ {result['reason']}")
                        diag = result['diagnostics']
                        st.caption(
                            f"Confidence: {diag['top_confidence']*100:.1f}%  |  "
                            f"Entropy: {diag['entropy']:.2f} / {diag['max_entropy']:.2f}"
                        )
                        st.info("Tips: Upload a single clear character. Good lighting. Black ink on white background works best.")
                    else:
                        predictions = result['results']
                        cached_recognition = st.session_state.get("recognize_last_result")
                        overlay_img = cached_recognition.get("overlay_img") if cached_recognition else None
                        if overlay_img is None:
                            try:
                                last_layer = get_cached_last_conv_layer(keras_model) if keras_model else None
                                if last_layer and keras_model:
                                    overlay_img = generate_gradcam(image, keras_model, last_layer)
                                else:
                                    overlay_img = image
                            except Exception as e:
                                st.warning(f"Grad-CAM could not be generated: {e}")
                                overlay_img = image
                            if cached_recognition and cached_recognition.get("image_hash") == image_hash:
                                cached_recognition["overlay_img"] = overlay_img
                                st.session_state.recognize_last_result = cached_recognition

                        st.markdown("<br>", unsafe_allow_html=True)
                        col_img1, col_img2 = st.columns(2, gap="large")
                        with col_img1:
                            st.markdown(f"""
                            <div class="image-card">
                                <div class="image-card-label">
                                    <span class="icon-inline">{icon("image", 16)}</span>
                                    Original Image
                                </div>
                                <div class="image-card-sublabel">The image you uploaded for analysis</div>
                            </div>
                            """, unsafe_allow_html=True)
                            st.image(image, width='stretch')

                        with col_img2:
                            st.markdown(f"""
                            <div class="image-card">
                                <div class="image-card-label">
                                    <span class="icon-inline">{icon("brain", 16)}</span>
                                    AI Focus Area
                                </div>
                                <div class="image-card-sublabel">Red regions show where our Model is focusing its attention</div>
                            </div>
                            """, unsafe_allow_html=True)
                            st.image(overlay_img, width='stretch')

                        st.markdown("<br>", unsafe_allow_html=True)
                        best_pred = predictions[0]
                        label_display = best_pred['devanagari']
                        st.markdown(f"""
                        <div class="prediction-card">
                            <div class="prediction-subtitle">Predicted Character</div>
                            <div class="prediction-title">{best_pred['english_name']}</div>
                            <div class="prediction-label">{label_display}</div>
                        </div>
                        """, unsafe_allow_html=True)
                        prediction_audio_path = os.path.join("audio files", f"{best_pred['devanagari']}.mp3")
                        st.markdown("<br>", unsafe_allow_html=True)
                        render_themed_audio(
                            prediction_audio_path,
                            f"recognize_{best_pred['class_name']}",
                            title="Pronunciation Audio",
                        )

                        st.markdown("<br>", unsafe_allow_html=True)
                        st.markdown(f"""
                        <div class="section-card" style="padding: 2rem 2.5rem;">
                            <div class="conf-section-title">
                                <span class="icon-inline">{icon("bar_chart", 18)}</span>
                                Confidence Analysis
                            </div>
                        """, unsafe_allow_html=True)

                        conf_html = ""
                        for idx, pred in enumerate(predictions):
                            conf = pred['confidence'] if pred['confidence'] > 1 else pred['confidence'] * 100
                            label = f"{pred['devanagari']} ({pred['english_name']})"
                            delay = 0.15 + idx * 0.15
                            conf_html += f"""
                            <div class="conf-container" style="animation-delay: {delay}s;">
                                <div class="conf-header">
                                    <span>{label}</span>
                                    <span style="color: var(--gold); font-weight: 700;">{conf:.1f}%</span>
                                </div>
                                <div class="conf-bar-bg">
                                    <div class="conf-bar-fill" style="width: {conf}%; animation-delay: {delay}s;"></div>
                                </div>
                            </div>
                            """

                        st.markdown(conf_html + "</div>", unsafe_allow_html=True)

                except Exception as e:
                    st.error(f"Error during prediction or visualization: {e}")
        else:
            st.image(image, caption="Uploaded Image", width='stretch')


# ═══════════════════════════════════════════════════════════════════════════════════
#  TAB 3 — LEARN
# ═══════════════════════════════════════════════════════════════════════════════════
with tab3:
    st.markdown(f"""
    <div class="section-card">
        <div class="section-heading">
            <span class="icon-inline icon-glow">{icon("book", 22)}</span>
            Learn Modi Script
        </div>
        <div class="section-desc">
            Explore the ancient Modi script — tap any character to reveal its pronunciation, meaning, and history.
        </div>
    </div>
    """, unsafe_allow_html=True)

    if "learned_chars" not in st.session_state:
        st.session_state.learned_chars = set()

    learn_categories = [
        {
            "key": "vowels",
            "icon_name": "letters",
            "title": "Vowels",
            "subtitle": "The foundation of sound",
            "types": ["vowel", "vowel_modifier"],
            "keys": ['a','aa','i','ii','u','oo','e','ai','o','ou','am','ah'],
        },
        {
            "key": "consonants",
            "icon_name": "pen_tool",
            "title": "Consonants",
            "subtitle": "The backbone of the script",
            "types": ["consonant", "compound"],
            "keys": ['k','kh','g','gh','ch','chh','ja','jh','t','tha','da','dha','nn',
                     'ta','th','d','dh','n','p','ph','b','bh','m','y','r','l','v',
                     'sh','s','h','lh','ksh','dyn','shr','tr'],
        },
        {
            "key": "numerals",
            "icon_name": "hash",
            "title": "Numerals",
            "subtitle": "The counting system of the Maratha Empire",
            "types": ["numeral"],
            "keys": ['zero','one','two','three','four','five','six','seven','eight','nine'],
        }
    ]

    st.markdown("")

    for cat in learn_categories:
        valid_keys = [k for k in cat["keys"] if k in modi_labels]
        total = len(valid_keys)
        learned = len([k for k in valid_keys if k in st.session_state.learned_chars])
        cat["total"] = total
        cat["learned"] = learned
        cat["valid_keys"] = valid_keys

    # Display Categories as Premium Cards
    cols = st.columns(3, gap="medium")
    for idx, cat in enumerate(learn_categories):
        with cols[idx]:
            is_active = st.session_state.get("learn_active_cat") == cat["key"]
            cat_cls = "premium-cat-card learn-category-card active" if is_active else "premium-cat-card learn-category-card"
            icon_svg = icon(cat["icon_name"], 28)
            cat_html = f'''
            <div class="{cat_cls}">
                <div class="cat-icon-container">{icon_svg}</div>
                <div class="cat-title">{cat["title"]}</div>
                <div class="cat-progress-text">{cat["learned"]}/{cat["total"]} Learned</div>
                <div class="cat-progress-bar">
                    <div class="cat-progress-fill" style="width: {(cat["learned"]/max(1, cat["total"]))*100}%"></div>
                </div>
            </div>
            '''
            st.markdown(cat_html, unsafe_allow_html=True)
            if st.button(f"Open {cat['title']}", key=f"btn_cat_{cat['key']}", width='stretch'):
                if st.session_state.get("learn_active_cat") == cat["key"]:
                    st.session_state["learn_active_cat"] = None
                else:
                    st.session_state["learn_active_cat"] = cat["key"]
                st.rerun()

    active_cat_key = st.session_state.get("learn_active_cat")
    if active_cat_key:
        active_cat = next((c for c in learn_categories if c["key"] == active_cat_key), None)
        if active_cat:
            st.markdown("<br>", unsafe_allow_html=True)
            st.markdown(f"<div style='font-family: \"Cinzel\", serif; font-size: 1.4rem; color: var(--gold); border-bottom: 1px solid rgba(255,153,51,0.2); padding-bottom: 0.5rem; margin-bottom: 1rem;'>{active_cat['title']}</div>", unsafe_allow_html=True)
            
            valid_keys = active_cat["valid_keys"]
            cols_per_row = 6
            for row_start in range(0, len(valid_keys), cols_per_row):
                row_keys = valid_keys[row_start:row_start + cols_per_row]
                g_cols = st.columns(cols_per_row)
                for col_idx, key in enumerate(row_keys):
                    info = modi_labels[key]
                    is_learned = key in st.session_state.learned_chars
                    with g_cols[col_idx]:
                        btn_text = f"{info['modi']}\n{info['devanagari']}"
                        if st.button(
                            btn_text,
                            key=f"learn_{active_cat['key']}_{key}",
                            width='stretch',
                            type="primary" if is_learned else "secondary"
                        ):
                            st.session_state.learned_chars.add(key)
                            st.session_state[f"learn_selected_{active_cat['key']}"] = key
                            st.rerun()

            selected_key = st.session_state.get(f"learn_selected_{active_cat['key']}")
            if selected_key and selected_key in modi_labels:
                info = modi_labels[selected_key]
                st.markdown(f"""
                <div class="learn-detail-card">
                    <div class="learn-detail-big">{info['modi']}</div>
                    <div class="learn-detail-devnag">{info['devanagari']} — {info['english_name']}</div>
                    <div class="learn-detail-row">
                        <div class="learn-detail-item">
                            <div class="lbl">Pronunciation</div>
                            <div class="val">{info.get('pronunciation', '—')}</div>
                        </div>
                        <div class="learn-detail-item">
                            <div class="lbl">Example Word</div>
                            <div class="val">{info.get('example_word', '—')}</div>
                        </div>
                        <div class="learn-detail-item">
                            <div class="lbl">Type</div>
                            <div class="val">{info.get('character_type', '—').title()}</div>
                        </div>
                    </div>
                    <div class="learn-detail-note">{info.get('historical_note', '')}</div>
                </div>
                """, unsafe_allow_html=True)

                audio_path = os.path.join("audio files", f"{info['devanagari']}.mp3")
                if os.path.exists(audio_path):
                    audio_b64 = get_file_b64(audio_path, os.path.getmtime(audio_path))
                    audio_html = f"""
                    <style>
                    .learn-audio-card {{
                        box-sizing: border-box;
                        width: min(100%, 560px);
                        margin: 0 auto;
                        padding: 1rem 1.1rem;
                        background: linear-gradient(135deg, rgba(255, 153, 51, 0.10), rgba(45, 20, 7, 0.92));
                        border: 1px solid rgba(255, 153, 51, 0.28);
                        border-top-color: rgba(255, 153, 51, 0.55);
                        border-radius: 8px;
                        box-shadow: 0 14px 36px rgba(0,0,0,0.28), inset 0 1px 0 rgba(255,255,255,0.05);
                        font-family: Inter, sans-serif;
                    }}
                    .learn-audio-title {{
                        display: flex;
                        align-items: center;
                        justify-content: center;
                        gap: 0.45rem;
                        color: #FF9933;
                        font-size: 0.72rem;
                        font-weight: 800;
                        text-transform: uppercase;
                        letter-spacing: 0.14em;
                        margin-bottom: 0.75rem;
                    }}
                    .learn-audio-player {{
                        display: grid;
                        grid-template-columns: 40px 1fr auto;
                        align-items: center;
                        gap: 0.85rem;
                    }}
                    .learn-audio-play {{
                        width: 40px;
                        height: 40px;
                        border-radius: 8px;
                        border: 1px solid rgba(255,153,51,0.42);
                        background: linear-gradient(135deg, #FF9933, #B86619);
                        color: #1A0D06;
                        font-size: 1rem;
                        font-weight: 900;
                        cursor: pointer;
                        box-shadow: 0 8px 18px rgba(255,153,51,0.18);
                    }}
                    .learn-audio-play:hover {{
                        filter: brightness(1.08);
                    }}
                    .learn-audio-track {{
                        appearance: none;
                        -webkit-appearance: none;
                        width: 100%;
                        height: 8px;
                        border-radius: 999px;
                        background: linear-gradient(90deg, #FF9933 var(--progress, 0%), rgba(255,255,255,0.12) var(--progress, 0%));
                        outline: none;
                        cursor: pointer;
                    }}
                    .learn-audio-track::-webkit-slider-thumb {{
                        -webkit-appearance: none;
                        width: 16px;
                        height: 16px;
                        border-radius: 50%;
                        background: #FFD19A;
                        border: 2px solid #FF9933;
                        box-shadow: 0 0 12px rgba(255,153,51,0.45);
                    }}
                    .learn-audio-track::-moz-range-thumb {{
                        width: 16px;
                        height: 16px;
                        border-radius: 50%;
                        background: #FFD19A;
                        border: 2px solid #FF9933;
                        box-shadow: 0 0 12px rgba(255,153,51,0.45);
                    }}
                    .learn-audio-time {{
                        color: rgba(245, 238, 226, 0.78);
                        font-size: 0.78rem;
                        font-weight: 700;
                        min-width: 80px;
                        text-align: right;
                        font-variant-numeric: tabular-nums;
                    }}
                    @media (max-width: 520px) {{
                        .learn-audio-player {{
                            grid-template-columns: 40px 1fr;
                        }}
                        .learn-audio-time {{
                            grid-column: 1 / -1;
                            text-align: center;
                        }}
                    }}
                    </style>
                    <div class="learn-audio-card">
                        <div class="learn-audio-title">
                            <span>{icon("headphones", 14)}</span>
                            Pronunciation Audio
                        </div>
                        <div class="learn-audio-player">
                            <button class="learn-audio-play" id="audioBtn_{selected_key}" type="button">▶</button>
                            <input class="learn-audio-track" id="audioTrack_{selected_key}" type="range" min="0" max="100" value="0">
                            <div class="learn-audio-time" id="audioTime_{selected_key}">0:00 / 0:00</div>
                        </div>
                        <audio id="learnAudio_{selected_key}" preload="metadata">
                            <source src="data:audio/mp3;base64,{audio_b64}" type="audio/mp3">
                        </audio>
                    </div>
                    <script>
                    (function() {{
                        const audio = document.getElementById('learnAudio_{selected_key}');
                        const button = document.getElementById('audioBtn_{selected_key}');
                        const track = document.getElementById('audioTrack_{selected_key}');
                        const time = document.getElementById('audioTime_{selected_key}');

                        function fmt(seconds) {{
                            if (!Number.isFinite(seconds)) return '0:00';
                            const m = Math.floor(seconds / 60);
                            const s = Math.floor(seconds % 60).toString().padStart(2, '0');
                            return `${{m}}:${{s}}`;
                        }}
                        function sync() {{
                            const pct = audio.duration ? (audio.currentTime / audio.duration) * 100 : 0;
                            track.value = pct;
                            track.style.setProperty('--progress', `${{pct}}%`);
                            time.textContent = `${{fmt(audio.currentTime)}} / ${{fmt(audio.duration)}}`;
                        }}
                        button.addEventListener('click', () => {{
                            if (audio.paused) {{
                                audio.play();
                                button.textContent = 'Ⅱ';
                            }} else {{
                                audio.pause();
                                button.textContent = '▶';
                            }}
                        }});
                        track.addEventListener('input', () => {{
                            if (audio.duration) {{
                                audio.currentTime = (Number(track.value) / 100) * audio.duration;
                                sync();
                            }}
                        }});
                        audio.addEventListener('loadedmetadata', sync);
                        audio.addEventListener('timeupdate', sync);
                        audio.addEventListener('ended', () => {{
                            button.textContent = '▶';
                            sync();
                        }});
                        sync();
                    }})();
                    </script>
                    """
                    st.iframe(audio_html, height=132)
                else:
                    st.markdown(f"""
                    <div class="video-not-found" style="max-width: 520px; min-height: 72px; margin: 1rem auto 0 auto;">
                        {icon("x_circle", 24, "#6b5c47")}
                        <span>Pronunciation audio not yet available for this character</span>
                    </div>
                    """, unsafe_allow_html=True)

                # ── PRACTICE SECTION: Video + Drawing Canvas ──
                st.markdown(f"""
                <div class="practice-section-title" style="margin-top: 1.5rem;">
                    <span class="icon-inline">{icon("pen_tool", 18)}</span>
                    Practice — {info['devanagari']} ({info['english_name']})
                </div>
                """, unsafe_allow_html=True)

                prac_col1, prac_col2 = st.columns(2, gap="large")
                practice_media_size = 380

                # ── Left: Drawing Video ──
                with prac_col1:
                    st.markdown(f"""
                    <div class="practice-panel">
                        <div class="practice-panel-label">
                            <span class="icon-inline">{icon("refresh", 14)}</span>
                            Stroke Animation
                        </div>
                    </div>
                    """, unsafe_allow_html=True)
                    speed_choice = st.session_state.get("learn_animation_speed", "Normal")
                    video_speed = 0.5 if speed_choice == "Slow" else 1.0
                    video_key_aliases = {"oo": "uu", "t": "thh"}
                    video_file_key = video_key_aliases.get(selected_key, selected_key)
                    video_path = os.path.join("animation_modiscript", f"{video_file_key}.mp4")
                    if os.path.exists(video_path):
                        video_b64 = get_file_b64(video_path, os.path.getmtime(video_path))
                        video_html = f"""
                        <style>body,html{{overflow:hidden !important; margin:0; padding:0;}}::-webkit-scrollbar {{display: none;}}</style>
                        <div style="width:min(100%, {practice_media_size}px); aspect-ratio:1/1; margin:0 auto;">
                            <video id="learnVideo_{selected_key}_{speed_choice}" autoplay loop muted playsinline
                                style="width:100%; height:100%; object-fit:contain; display:block; box-sizing:border-box;
                                       border:2px solid rgba(255,153,51,0.25); border-radius:8px; background:#fffaf3;">
                                <source src="data:video/mp4;base64,{video_b64}" type="video/mp4">
                            </video>
                        </div>
                        <script>
                        (function() {{
                            const video = document.getElementById('learnVideo_{selected_key}_{speed_choice}');
                            if (video) {{
                                video.defaultPlaybackRate = {video_speed};
                                video.playbackRate = {video_speed};
                                video.load();
                                video.playbackRate = {video_speed};
                                video.play().catch(() => {{}});
                            }}
                        }})();
                        </script>
                        """
                        st.iframe(video_html, height=practice_media_size + 8)
                        st.markdown(f"""
                        <div class="learn-speed-card">
                            <div class="practice-panel-label">
                                <span class="icon-inline">{icon("refresh", 14)}</span>
                                Animation Speed
                            </div>
                        </div>
                        """, unsafe_allow_html=True)
                        st.radio(
                            "Animation speed",
                            ["Normal", "Slow"],
                            horizontal=True,
                            key="learn_animation_speed",
                            label_visibility="collapsed",
                        )
                    else:
                        st.markdown(f"""
                        <div class="video-not-found" style="width: min(100%, {practice_media_size}px); aspect-ratio: 1 / 1; margin: 0 auto; border: 2px solid rgba(255,153,51,0.25); border-radius: 8px; background: #fffaf3;">
                            {icon("x_circle", 32, "#6b5c47")}
                            <span>Stroke animation not yet available for this character</span>
                        </div>
                        """, unsafe_allow_html=True)

                # ── Right: Drawing Canvas ──
                with prac_col2:
                    st.markdown(f"""
                    <div class="practice-panel">
                        <div class="practice-panel-label">
                            <span class="icon-inline">{icon("pen_tool", 14)}</span>
                            Your Turn — Draw It
                        </div>
                    </div>
                    """, unsafe_allow_html=True)

                    canvas_key = f"canvas_{selected_key}"
                    clear_count = st.session_state.get(f"clear_{canvas_key}", 0)
                    ls_canvas_key = f"akshar_canvas_{selected_key}_{clear_count}"
                    ls_score_key = f"akshar_score_{selected_key}_{clear_count}"
                    canvas_html = f"""
                    <style>body,html{{overflow:hidden !important; margin:0; padding:0;}}::-webkit-scrollbar {{display: none;}}</style>
                    <div style="width:min(100%, {practice_media_size}px); aspect-ratio:1/1; margin:0 auto;">
                        <canvas id="drawCanvas_{selected_key}_{clear_count}" width="{practice_media_size}" height="{practice_media_size}"
                            style="border: 2px solid rgba(255,153,51,0.25); border-radius: 12px;
                                   background: #fffaf3; cursor: crosshair; touch-action: none;
                                   width: 100%; height: 100%; display: block; box-sizing: border-box;">
                        </canvas>
                    </div>
                    <script>
                    (function() {{
                        const c = document.getElementById('drawCanvas_{selected_key}_{clear_count}');
                        const ctx = c.getContext('2d');
                        let drawing = false;
                        ctx.lineWidth = 4;
                        ctx.lineCap = 'round';
                        ctx.lineJoin = 'round';
                        ctx.strokeStyle = '#1a0e05';

                        // Restore drawing from localStorage
                        var savedData = localStorage.getItem('{ls_canvas_key}');
                        if (savedData) {{
                            var restoreImg = new Image();
                            restoreImg.onload = function() {{
                                ctx.drawImage(restoreImg, 0, 0);
                            }};
                            restoreImg.src = savedData;
                        }}

                        function getPos(e) {{
                            const r = c.getBoundingClientRect();
                            const t = e.touches ? e.touches[0] : e;
                            return [
                                (t.clientX - r.left) * (c.width / r.width),
                                (t.clientY - r.top) * (c.height / r.height)
                            ];
                        }}
                        
                        function sendStore() {{
                            var dataUrl = c.toDataURL("image/png");
                            try {{ localStorage.setItem('{ls_canvas_key}', dataUrl); }} catch(e) {{}}
                        }}

                        c.addEventListener('mousedown', e => {{ drawing = true; ctx.beginPath(); const [x,y] = getPos(e); ctx.moveTo(x,y); }});
                        c.addEventListener('mousemove', e => {{ if (!drawing) return; const [x,y] = getPos(e); ctx.lineTo(x,y); ctx.stroke(); }});
                        c.addEventListener('mouseup', () => {{ drawing = false; sendStore(); }});
                        c.addEventListener('mouseleave', () => {{ if(drawing) sendStore(); drawing = false; }});
                        c.addEventListener('touchstart', e => {{ e.preventDefault(); drawing = true; ctx.beginPath(); const [x,y] = getPos(e); ctx.moveTo(x,y); }});
                        c.addEventListener('touchmove', e => {{ e.preventDefault(); if (!drawing) return; const [x,y] = getPos(e); ctx.lineTo(x,y); ctx.stroke(); }});
                        c.addEventListener('touchend', () => {{ drawing = false; sendStore(); }});
                    }})();
                    </script>
                    """
                    st.iframe(canvas_html, height=practice_media_size + 8)

                    st.markdown(f"""
                    <div class="learn-speed-card" style="margin-top: 0; padding-top: 0.8rem; padding-bottom: 0.8rem;">
                        <div class="practice-panel-label" style="justify-content:center; align-items:center; margin-bottom: 0;">
                            <span class="icon-inline">{icon("activity", 14)}</span>
                            Your Score
                        </div>
                    </div>
                    """, unsafe_allow_html=True)

                    score_display_html = ("""<style>
                    @import url('https://fonts.googleapis.com/css2?family=Cinzel:wght@700&display=swap');
                    body, html { margin:0; padding:0; overflow:hidden; background:transparent; }
                    </style>
                    <div id="scoreVal" style="text-align:center; font-family:'Cinzel',serif; font-size:2rem; font-weight:700; color:#FF9933; padding:0.2rem 0;">--/10</div>
                    <div id="feedbackVal" style="text-align:center; font-family:'Segoe UI', Tahoma, Geneva, Verdana, sans-serif; font-size:1rem; font-weight:600; color:rgba(255, 255, 255, 0.7); margin-top:-5px;"></div>
                    <script>
                    (function() {
                        function update() {
                            var s = localStorage.getItem('__SCORE_KEY__');
                            document.getElementById('scoreVal').textContent = (s ? s : '--') + '/10';
                            var feedback = '';
                            if (s) {
                                var scoreNum = parseInt(s, 10);
                                if (scoreNum >= 8) feedback = 'Excellent!';
                                else if (scoreNum >= 6) feedback = 'Good job';
                                else feedback = 'Keep practicing';
                            }
                            document.getElementById('feedbackVal').textContent = feedback;
                        }
                        update();
                        setInterval(update, 300);
                    })();
                    </script>
                    """.replace('__SCORE_KEY__', ls_score_key))
                    st.iframe(score_display_html, height=85)

                    eval_col, clear_col = st.columns(2)
                    with eval_col:
                        if st.button("✨ Evaluate", key=f"eval_btn_{selected_key}", type="primary", width='stretch'):
                            st.session_state[f"trigger_eval_{selected_key}"] = True
                            st.rerun()

                    with clear_col:
                        if st.button("🗑️ Clear", key=f"clear_btn_{selected_key}", width='stretch'):
                            st.session_state[f"clear_{canvas_key}"] = clear_count + 1
                            st.rerun()

                    # --- Client-side evaluation via JS ---
                    if st.session_state.pop(f"trigger_eval_{selected_key}", False):
                        devanagari_char = info.get("devanagari", "")
                        ref_path = os.path.join("assets", "reference_images", f"{devanagari_char}.png")
                        if os.path.exists(ref_path):
                            ref_b64 = get_file_b64(ref_path, os.path.getmtime(ref_path))
                            eval_script = ("""<script>
                            (function() {
                                var canvasData = localStorage.getItem('__CANVAS_KEY__');
                                if (!canvasData) { localStorage.setItem('__SCORE_KEY__', '0'); return; }
                                var canvasImg = new Image();
                                canvasImg.onload = function() {
                                    var refImg = new Image();
                                    refImg.onload = function() {
                                        var score = compareImages(canvasImg, refImg);
                                        localStorage.setItem('__SCORE_KEY__', String(score));
                                    };
                                    refImg.src = 'data:image/png;base64,__REF_B64__';
                                };
                                canvasImg.src = canvasData;
                                function compareImages(img1, img2) {
                                    var size = 64;
                                    function processImage(img) {
                                        var c = document.createElement('canvas');
                                        c.width = img.naturalWidth || size;
                                        c.height = img.naturalHeight || size;
                                        var ctx = c.getContext('2d');
                                        ctx.fillStyle = '#ffffff';
                                        ctx.fillRect(0, 0, c.width, c.height);
                                        ctx.drawImage(img, 0, 0);
                                        var data = ctx.getImageData(0, 0, c.width, c.height).data;
                                        var w = c.width, h = c.height;
                                        var minX = w, maxX = 0, minY = h, maxY = 0, hasStroke = false;
                                        for (var y = 0; y < h; y++) {
                                            for (var x = 0; x < w; x++) {
                                                var i = (y * w + x) * 4;
                                                var g = 0.299*data[i] + 0.587*data[i+1] + 0.114*data[i+2];
                                                if (g < 200) {
                                                    hasStroke = true;
                                                    if (x < minX) minX = x; if (x > maxX) maxX = x;
                                                    if (y < minY) minY = y; if (y > maxY) maxY = y;
                                                }
                                            }
                                        }
                                        if (!hasStroke) {
                                            var blank = document.createElement('canvas');
                                            blank.width = size; blank.height = size;
                                            var bCtx = blank.getContext('2d');
                                            bCtx.fillStyle = '#ffffff'; bCtx.fillRect(0,0,size,size);
                                            return bCtx.getImageData(0,0,size,size);
                                        }
                                        var pad = 10;
                                        minX = Math.max(0, minX-pad); minY = Math.max(0, minY-pad);
                                        maxX = Math.min(w-1, maxX+pad); maxY = Math.min(h-1, maxY+pad);
                                        var cropW = maxX-minX+1, cropH = maxY-minY+1;
                                        var side2 = Math.max(cropW, cropH);
                                        var sq = document.createElement('canvas');
                                        sq.width = side2; sq.height = side2;
                                        var sqCtx = sq.getContext('2d');
                                        sqCtx.fillStyle = '#ffffff'; sqCtx.fillRect(0,0,side2,side2);
                                        sqCtx.drawImage(c, minX, minY, cropW, cropH,
                                                       (side2-cropW)/2, (side2-cropH)/2, cropW, cropH);
                                        var rs = document.createElement('canvas');
                                        rs.width = size; rs.height = size;
                                        var rsCtx = rs.getContext('2d');
                                        rsCtx.drawImage(sq, 0, 0, size, size);
                                        return rsCtx.getImageData(0, 0, size, size);
                                    }
                                    var p1 = processImage(img1), p2 = processImage(img2);
                                    var mse = 0, pixels = size * size;
                                    for (var i = 0; i < p1.data.length; i += 4) {
                                        var diff = (p1.data[i] - p2.data[i]) / 255.0;
                                        mse += diff * diff;
                                    }
                                    mse /= pixels;
                                    var similarity = Math.max(0, 1 - (mse / 0.15));
                                    var base_score = Math.floor(similarity * 10);
                                    var final_score = base_score;
                                    if (base_score <= 2) {
                                        final_score = 1;
                                    } else if (base_score <= 5) {
                                        final_score = base_score + 3;
                                    } else if (base_score <= 7) {
                                        final_score = base_score + 2;
                                    }
                                    return Math.max(0, Math.min(10, final_score));
                                }
                            })();
                            </script>""")
                            eval_script = eval_script.replace('__CANVAS_KEY__', ls_canvas_key)
                            eval_script = eval_script.replace('__SCORE_KEY__', ls_score_key)
                            eval_script = eval_script.replace('__REF_B64__', ref_b64)
                            st.iframe(eval_script, height=1)


    total_chars = sum(len([k for k in cat["keys"] if k in modi_labels]) for cat in learn_categories)
    total_learned = len(st.session_state.learned_chars)
    if total_chars > 0:
        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown(f"""
        <div class="section-card" style="text-align: center; padding: 1.5rem 2rem;">
            <div style="color: var(--text-muted); font-size: 0.8rem; text-transform: uppercase; letter-spacing: 0.12em; margin-bottom: 0.3rem;">Overall Progress</div>
            <div style="font-family: 'Cinzel', serif; font-size: 1.5rem; font-weight: 700; color: var(--gold);">{total_learned} / {total_chars} Characters Explored</div>
        </div>
        """, unsafe_allow_html=True)
        st.progress(total_learned / total_chars)


# ═══════════════════════════════════════════════════════════════════════════════════
#  TAB 4 — QUIZ
# ═══════════════════════════════════════════════════════════════════════════════════
with tab4:
    st.markdown(f"""
    <div class="section-card">
        <div class="section-heading">
            <span class="icon-inline icon-glow">{icon("trophy", 22)}</span>
            Quiz Mode
        </div>
        <div class="section-desc">
            Test your knowledge of Modi script characters. Choose a category and conquer 10 questions to prove your mastery.
        </div>
    </div>
    """, unsafe_allow_html=True)

    if not modi_labels:
        st.warning("Data not loaded to start the quiz. Please check if `modi_labels.json` exists.")
    else:
        # 1. State Initialization
        if "quiz_session_active" not in st.session_state:
            st.session_state.quiz_session_active = False
            st.session_state.quiz_selected_category = None
            st.session_state.quiz_current_index = 0
            st.session_state.quiz_session_score = 0
            st.session_state.quiz_session_questions = []
            st.session_state.quiz_answered_current = False
            st.session_state.quiz_selected_option = None
            st.session_state.quiz_explored_chars = set()
            st.session_state.quiz_progress = {
                "Vowels": {"attempted": 0, "correct": 0, "seen": set()},
                "Consonants": {"attempted": 0, "correct": 0, "seen": set()},
                "Numbers": {"attempted": 0, "correct": 0, "seen": set()}
            }

        quiz_categories = [
            {"key": "Vowels", "title": "Vowels", "icon": "letters", "types": ["vowel", "vowel_modifier"]},
            {"key": "Consonants", "title": "Consonants", "icon": "pen_tool", "types": ["consonant", "compound"]},
            {"key": "Numbers", "title": "Numerals", "icon": "hash", "types": ["numeral"]}
        ]

        def get_pool(types):
            return [k for k, v in modi_labels.items() if v.get("character_type") in types]

        quiz_pools = {
            cat["key"]: get_pool(cat["types"])
            for cat in quiz_categories
        }

        # 2. Landing Page / Category Selection
        if not st.session_state.quiz_session_active:
            st.markdown("<br>", unsafe_allow_html=True)
            cols = st.columns(3, gap="medium")
            for idx, cat in enumerate(quiz_categories):
                with cols[idx]:
                    pool_keys = quiz_pools[cat["key"]]
                    total_chars = len(pool_keys)
                    seen_count = len([k for k in pool_keys if k in st.session_state.quiz_progress[cat["key"]]["seen"]])
                    
                    prog = st.session_state.quiz_progress[cat["key"]]
                    accuracy = 0 if prog["attempted"] == 0 else int((prog["correct"] / prog["attempted"]) * 100)
                    
                    icon_svg = icon(cat["icon"], 28)
                    st.markdown(f"""
                    <div class="premium-cat-card" style="margin-bottom: 2rem;">
                        <div class="cat-icon-container">{icon_svg}</div>
                        <div class="cat-title">{cat["title"]}</div>
                        <div class="cat-subtitle">Accuracy: {accuracy}%</div>
                        <div class="cat-progress-text">{seen_count}/{total_chars} Explored</div>
                        <div class="cat-progress-bar">
                            <div class="cat-progress-fill" style="width: {(seen_count/max(1, total_chars))*100}%"></div>
                        </div>
                    </div>
                    """, unsafe_allow_html=True)
                    
                    if st.button(f"Start Quiz", key=f"start_quiz_{cat['key']}", width='stretch', type="primary"):
                        st.session_state.quiz_session_active = True
                        st.session_state.quiz_selected_category = cat["key"]
                        st.session_state.quiz_current_index = 0
                        st.session_state.quiz_session_score = 0
                        st.session_state.quiz_answered_current = False
                        st.session_state.quiz_selected_option = None
                        
                        # Generate 5 questions
                        available_pool = quiz_pools[cat["key"]]
                        unseen = [k for k in available_pool if k not in st.session_state.quiz_progress[cat["key"]]["seen"]]
                        
                        questions = []
                        for _ in range(5):
                            if unseen:
                                correct_key = random.choice(unseen)
                                unseen.remove(correct_key)
                            else:
                                correct_key = random.choice(available_pool)
                                
                            wrong_pool = [k for k in available_pool if k != correct_key]
                            if len(wrong_pool) < 3:
                                wrong_pool = [k for k in modi_labels.keys() if k != correct_key]
                            wrong_keys = random.sample(wrong_pool, 3)
                            
                            options = [correct_key] + wrong_keys
                            random.shuffle(options)
                            
                            modes = ["visual", "hint"]
                            if os.path.exists(f"audio files/{modi_labels[correct_key]['devanagari']}.mp3"):
                                modes.append("audio")
                            q_mode = random.choice(modes)
                            
                            questions.append({
                                "correct_key": correct_key,
                                "options": options,
                                "mode": q_mode
                            })
                            
                        st.session_state.quiz_session_questions = questions
                        st.rerun()

            # Global exploration tracker
            total_db = len(modi_labels)
            total_explored = len(st.session_state.quiz_explored_chars)
            st.markdown(f"""
            <div class="section-card" style="text-align: center; padding: 2rem;">
                <div style="color: var(--text-muted); font-size: 0.8rem; text-transform: uppercase; letter-spacing: 0.12em; margin-bottom: 0.3rem;">Total Quiz Exploration</div>
                <div style="font-family: 'Cinzel', serif; font-size: 1.8rem; font-weight: 700; color: var(--gold); margin-bottom: 1rem;">{total_explored} / {total_db} Characters Seen</div>
                <div class="cat-progress-bar" style="max-width: 400px; margin: 0 auto; height: 8px;">
                    <div class="cat-progress-fill" style="width: {(total_explored/max(1, total_db))*100}%"></div>
                </div>
            </div>
            """, unsafe_allow_html=True)
        # 3. Active Quiz Session
        else:
            q_idx = st.session_state.quiz_current_index
            total_q = 5
            score = st.session_state.quiz_session_score
            
            # Scoreboard
            st.markdown(f"""
            <div class="quiz-scoreboard">
                <div class="quiz-score-item highlight">
                    <div class="quiz-score-label">Score</div>
                    <div class="quiz-score-value">{score}</div>
                </div>
                <div class="quiz-score-item">
                    <div class="quiz-score-label">Question</div>
                    <div class="quiz-score-value accuracy">{q_idx}/{total_q}</div>
                </div>
            </div>
            """, unsafe_allow_html=True)
            st.progress(q_idx / total_q if q_idx <= total_q else 1.0)
            
            # Session Complete
            if q_idx >= total_q:
                award = icon("award", 48, "#FF9933")
                st.markdown(f"""
                <div class="section-card" style="text-align: center; padding: 3rem;">
                    <div style="margin-bottom: 1rem; color: var(--gold);">{award}</div>
                    <div style="font-family: 'Cinzel', serif; font-size: 1.8rem; font-weight: 700; color: var(--gold); margin-bottom: 0.5rem;">Session Complete!</div>
                    <div style="color: var(--text-secondary); font-size: 1.1rem;">You scored {score} out of {total_q}.</div>
                </div>
                """, unsafe_allow_html=True)
                
                c1, c2 = st.columns(2)
                with c1:
                    if st.button("Play Another", type="primary", width='stretch'):
                        st.session_state.quiz_session_active = False
                        st.rerun()
                with c2:
                    if st.button("Back to Categories", width='stretch'):
                        st.session_state.quiz_session_active = False
                        st.rerun()
                        
            # Question UI
            else:
                q_data = st.session_state.quiz_session_questions[q_idx]
                correct_info = modi_labels[q_data["correct_key"]]
                q_mode = q_data["mode"]
                answered = st.session_state.quiz_answered_current
                
                # Render Question Prompt based on Mode
                if q_mode == "visual":
                    st.markdown("""
                    <div class="quiz-question-card">
                        <div class="quiz-mode-badge">Visual Mode</div>
                        <div class="quiz-question-prompt">Identify this character:</div>
                    """, unsafe_allow_html=True)
                    
                    img_path = f"assets/sample_images/{q_data['correct_key']}.jpg"
                    if os.path.exists(img_path):
                        # Use markdown to enforce styling on image
                        img_b64 = get_file_b64(img_path, os.path.getmtime(img_path))
                        st.markdown(f"""
                        <div class="quiz-image-container">
                            <img src="data:image/jpeg;base64,{img_b64}" />
                        </div>
                        """, unsafe_allow_html=True)
                    else:
                        st.markdown(f'<div class="quiz-char-display">{correct_info["modi"]}</div>', unsafe_allow_html=True)
                    st.markdown("</div>", unsafe_allow_html=True)
                    
                elif q_mode == "hint":
                    st.markdown(f"""
                    <div class="quiz-question-card">
                        <div class="quiz-mode-badge">Hint Mode</div>
                        <div class="quiz-question-prompt">What is the Modi script for:</div>
                        <div class="quiz-text-display">{correct_info['devanagari']}</div>
                        <div class="quiz-hint-box">Example: {correct_info['example_word']}</div>
                    </div>
                    """, unsafe_allow_html=True)
                    
                elif q_mode == "audio":
                    hp_icon = icon("headphones", 18)
                    st.markdown(f"""
                    <div class="quiz-question-card">
                        <div class="quiz-mode-badge">Audio Mode</div>
                        <div class="quiz-question-prompt">
                            <span class="icon-inline">{hp_icon}</span> Listen & Identify:
                        </div>
                        <br>
                    """, unsafe_allow_html=True)
                    render_themed_audio(
                        os.path.join("audio files", f"{correct_info['devanagari']}.mp3"),
                        f"quiz_{q_idx}_{q_data['correct_key']}",
                        title="Listen & Identify",
                        autoplay=True,
                    )
                    st.markdown("</div>", unsafe_allow_html=True)
                
                st.markdown("<br>", unsafe_allow_html=True)
                
                # Render Options
                col1, col2 = st.columns(2, gap="medium")
                for i, opt_key in enumerate(q_data["options"]):
                    opt_info = modi_labels[opt_key]
                    c = col1 if i % 2 == 0 else col2
                    
                    if q_mode == "visual":
                        # Options are Devanagari text
                        btn_txt = f"{opt_info['devanagari']} ({opt_info['english_name']})"
                    else:
                        # Options are Modi characters
                        btn_txt = f"{opt_info['modi']}"
                        
                    with c:
                        if answered:
                            is_correct = (opt_key == q_data["correct_key"])
                            user_selected = (st.session_state.quiz_selected_option == opt_key)
                            if is_correct:
                                st.markdown(f'<div class="answer-card answer-correct">{btn_txt}</div>', unsafe_allow_html=True)
                            elif user_selected:
                                st.markdown(f'<div class="answer-card answer-wrong">{btn_txt}</div>', unsafe_allow_html=True)
                            else:
                                st.markdown(f'<div class="answer-card answer-dim">{btn_txt}</div>', unsafe_allow_html=True)
                        else:
                            if st.button(btn_txt, key=f"opt_{q_idx}_{i}", width='stretch'):
                                st.session_state.quiz_answered_current = True
                                st.session_state.quiz_selected_option = opt_key
                                
                                is_correct = (opt_key == q_data["correct_key"])
                                
                                # Update tracking
                                cat = st.session_state.quiz_selected_category
                                st.session_state.quiz_progress[cat]["attempted"] += 1
                                st.session_state.quiz_progress[cat]["seen"].add(q_data["correct_key"])
                                st.session_state.quiz_explored_chars.add(q_data["correct_key"])
                                
                                if is_correct:
                                    st.session_state.quiz_session_score += 1
                                    st.session_state.quiz_progress[cat]["correct"] += 1
                                    
                                st.rerun()

                st.markdown("<br>", unsafe_allow_html=True)
                
                # Feedback Area
                if answered:
                    is_correct = (st.session_state.quiz_selected_option == q_data["correct_key"])
                    
                    if is_correct:
                        st.markdown(f"""
                        <div class="quiz-feedback correct">
                            {icon("check_circle", 20, "#4ade80")} Correct! Excellent work.
                        </div>
                        """, unsafe_allow_html=True)
                        
                        # Short simple Web Audio beep + Confetti
                        if "feedback_played" not in st.session_state.quiz_session_questions[q_idx]:
                            st.iframe("""
                            <script>
                                try {
                                    const parentWindow = window.parent || window;
                                    const parentDoc = parentWindow.document;
                                    function playSuccessBeep() {
                                        const AudioContext = parentWindow.AudioContext || parentWindow.webkitAudioContext || window.AudioContext || window.webkitAudioContext;
                                        const ctx = new AudioContext();
                                        if (ctx.state === 'suspended') { ctx.resume(); }
                                        const osc = ctx.createOscillator();
                                        const gain = ctx.createGain();
                                        osc.type = 'sine';
                                        osc.frequency.setValueAtTime(600, ctx.currentTime);
                                        osc.frequency.exponentialRampToValueAtTime(1200, ctx.currentTime + 0.1);
                                        gain.gain.setValueAtTime(0.1, ctx.currentTime);
                                        gain.gain.exponentialRampToValueAtTime(0.001, ctx.currentTime + 0.2);
                                        osc.connect(gain);
                                        gain.connect(ctx.destination);
                                        osc.start();
                                        osc.stop(ctx.currentTime + 0.1);
                                    }
                                    function fireConfetti() {
                                        if (!parentWindow.confetti) { return; }
                                        const canvas = parentDoc.createElement('canvas');
                                        canvas.style.position = 'fixed';
                                        canvas.style.top = '0';
                                        canvas.style.left = '0';
                                        canvas.style.width = '100vw';
                                        canvas.style.height = '100vh';
                                        canvas.style.pointerEvents = 'none';
                                        canvas.style.zIndex = '999999';
                                        canvas.style.opacity = '1';
                                        canvas.style.transition = 'opacity 1s ease-out';
                                        parentDoc.body.appendChild(canvas);

                                        var myConfetti = parentWindow.confetti.create(canvas, {
                                            resize: true,
                                            useWorker: true
                                        });

                                        var duration = 1000;
                                        var end = Date.now() + duration;
                                        (function frame() {
                                            myConfetti({
                                                particleCount: 4,
                                                angle: 60,
                                                spread: 45,
                                                origin: { x: 0, y: 0.8 },
                                                colors: ['#FF9933', '#CC7A29', '#f8fafc', '#8B4513']
                                            });
                                            myConfetti({
                                                particleCount: 4,
                                                angle: 120,
                                                spread: 45,
                                                origin: { x: 1, y: 0.8 },
                                                colors: ['#FF9933', '#CC7A29', '#f8fafc', '#8B4513']
                                            });
                                            if (Date.now() < end) {
                                                requestAnimationFrame(frame);
                                            }
                                        }());

                                        setTimeout(() => {
                                            canvas.style.opacity = '0';
                                            setTimeout(() => {
                                                if (parentDoc.body.contains(canvas)) {
                                                    parentDoc.body.removeChild(canvas);
                                                }
                                            }, 1000);
                                        }, 1000);
                                    }

                                    playSuccessBeep();
                                    if (parentWindow.confetti) {
                                        fireConfetti();
                                    } else if (!parentWindow.__aksharConfettiLoading) {
                                        parentWindow.__aksharConfettiLoading = true;
                                        const script = parentDoc.createElement('script');
                                        script.src = 'https://cdn.jsdelivr.net/npm/canvas-confetti@1.6.0/dist/confetti.browser.min.js';
                                        script.onload = () => {
                                            parentWindow.__aksharConfettiLoading = false;
                                            fireConfetti();
                                        };
                                        parentDoc.head.appendChild(script);
                                    }
                                } catch (e) { console.log('Audio/Confetti error:', e); }
                            </script>
                            """, height=1, width=1)
                            st.session_state.quiz_session_questions[q_idx]["feedback_played"] = True
                            
                    else:
                        st.markdown(f"""
                        <div class="quiz-feedback wrong">
                            {icon("x_circle", 20, "#f87171")} Not quite — keep practicing!
                        </div>
                        """, unsafe_allow_html=True)
                        if "feedback_played" not in st.session_state.quiz_session_questions[q_idx]:
                            # Low pitch thud
                            st.iframe("""
                            <script>
                                try {
                                    const parentWindow = window.parent || window;
                                    const AudioContext = parentWindow.AudioContext || parentWindow.webkitAudioContext || window.AudioContext || window.webkitAudioContext;
                                    const ctx = new AudioContext();
                                    if (ctx.state === 'suspended') { ctx.resume(); }
                                    const osc = ctx.createOscillator();
                                    const gain = ctx.createGain();
                                    osc.type = 'triangle';
                                    osc.frequency.setValueAtTime(150, ctx.currentTime);
                                    osc.frequency.exponentialRampToValueAtTime(80, ctx.currentTime + 0.15);
                                    gain.gain.setValueAtTime(0.1, ctx.currentTime);
                                    gain.gain.exponentialRampToValueAtTime(0.001, ctx.currentTime + 0.15);
                                    osc.connect(gain);
                                    gain.connect(ctx.destination);
                                    osc.start();
                                    osc.stop(ctx.currentTime + 0.15);
                                } catch(e) {}
                            </script>
                            """, height=1, width=1)
                            st.session_state.quiz_session_questions[q_idx]["feedback_played"] = True

                    st.markdown(f"""
                    <div class="section-card" style="padding: 1.5rem; margin-top: 1rem;">
                        <div style="color: var(--text-secondary); font-size: 0.8rem; text-transform: uppercase; letter-spacing: 0.1em; margin-bottom: 0.8rem;">Correct Answer</div>
                        <div style="display: flex; align-items: center; gap: 1rem;">
                            <span style="font-size: 2.5rem; color: var(--gold); font-weight: 800;">{correct_info['modi']}</span>
                            <div>
                                <div style="font-size: 1.1rem; color: var(--text-primary); font-weight: 600;">{correct_info['devanagari']}</div>
                                <div style="font-size: 0.85rem; color: var(--text-muted);">Example: {correct_info.get('example_word', 'N/A')}</div>
                            </div>
                        </div>
                    </div>
                    """, unsafe_allow_html=True)
                    
                    st.markdown("<br>", unsafe_allow_html=True)
                    btn_text = "Next Question" if (q_idx < 4) else "View Results"
                    if st.button(btn_text, type="primary", width='stretch'):
                        st.session_state.quiz_current_index += 1
                        st.session_state.quiz_answered_current = False
                        st.session_state.quiz_selected_option = None
                        st.rerun()


# ═══════════════════════════════════════════════════════════════════════════════════
#  TAB 5 — LIBRARY
# ═══════════════════════════════════════════════════════════════════════════════════
with tab5:
    st.markdown(f"""
    <div class="section-card">
        <div class="section-heading">
            <span class="icon-inline icon-glow">{icon("library", 22)}</span>
            Character Library
        </div>
        <div class="section-desc">
            Browse the complete Modi script collection. Search, filter, and hover over any character to reveal its details.
        </div>
    </div>
    """, unsafe_allow_html=True)

    tool_col1, tool_col2 = st.columns([3, 1])
    with tool_col1:
        search_query = st.text_input(
            "Search characters",
            placeholder="Search by name, Devanagari, or pronunciation...",
            label_visibility="collapsed"
        )
    with tool_col2:
        lib_filter = st.selectbox(
            "Filter",
            ["All", "Vowels", "Consonants", "Numerals"],
            label_visibility="collapsed"
        )

    filter_types = {
        "All": None,
        "Vowels": ["vowel", "vowel_modifier"],
        "Consonants": ["consonant", "compound"],
        "Numerals": ["numeral"]
    }

    groups = [
        ("Numbers", "numeral", [
            ['zero','one','two','three','four','five','six','seven','eight','nine']
        ]),
        ("Vowels", "vowel", [
            ['a','aa','i','ii','u','oo'],
            ['e','ai','o','ou','am','ah']
        ]),
        ("Consonants", "consonant", [
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

    active_types = filter_types[lib_filter]

    def matches_search(key, info, query):
        if not query:
            return True
        q = query.lower().strip()
        searchable = f"{info.get('devanagari','')} {info.get('modi','')} {info.get('english_name','')} {info.get('pronunciation','')} {key}".lower()
        return q in searchable

    html_parts = []
    any_results = False

    for group_name, group_type, row_list in groups:
        if active_types is not None:
            group_char_types = set()
            for keys in row_list:
                for k in keys:
                    if k in modi_labels:
                        group_char_types.add(modi_labels[k].get("character_type", ""))
            if not any(ct in active_types for ct in group_char_types):
                continue

        filtered_keys = []
        for keys in row_list:
            for k in keys:
                if k not in modi_labels:
                    continue
                info = modi_labels[k]
                if active_types and info.get("character_type", "") not in active_types:
                    continue
                if not matches_search(k, info, search_query):
                    continue
                filtered_keys.append(k)

        if not filtered_keys:
            continue

        any_results = True
        html_parts.append(f"<div class='library-section'><div class='library-title'>{group_name}</div>")
        html_parts.append("<div class='char-grid' style='margin-bottom: 15px;'>")

        for k in filtered_keys:
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

        html_parts.append("</div>")
        html_parts.append("</div>")

    if not any_results:
        search_svg = icon("search", 32, "#6b5c47")
        html_parts.append(f"""
        <div style="text-align: center; padding: 3rem; color: var(--text-muted);">
            <div style="margin-bottom: 0.8rem; opacity: 0.5;">{search_svg}</div>
            <div>No characters found matching your search</div>
        </div>
        """)

    final_html = "".join(html_parts)
    st.markdown(final_html, unsafe_allow_html=True)


  # ─── FOOTER ──────────────────────────────────────────────────────────────────────
heart_svg = icon("heart", 14, "#CC7A29")
st.markdown(f"""
<div class="app-footer">
    AKSHAR.AI &nbsp;&middot;&nbsp; Preserving Heritage Through Intelligence &nbsp;&middot;&nbsp; Built with {heart_svg}
</div>
""", unsafe_allow_html=True)

import time
# ─── GLOBAL SMOOTH LOADING TRANSITION ──────────────────────────────────────────
# Masks Streamlit rerenders with a premium loader. Executed via invisible iframe.
loading_js = f"""
<style>body {{ margin: 0; padding: 0; overflow: hidden; }}</style>
<script>
(function() {{
    const p = window.parent.document;
    
    // 1. Inject Global Loader DOM & CSS if not present
    if (!p.getElementById('akshar-loader-style')) {{
        const style = p.createElement('style');
        style.id = 'akshar-loader-style';
        style.innerHTML = `
            #akshar-global-loader {{
                position: fixed;
                top: 0; left: 0; right: 0; bottom: 0;
                background: linear-gradient(135deg, rgba(15,8,4,0.7) 0%, rgba(26,13,6,0.92) 100%);
                backdrop-filter: blur(12px) saturate(140%);
                -webkit-backdrop-filter: blur(12px) saturate(140%);
                z-index: 9999999;
                display: flex;
                flex-direction: column;
                align-items: center;
                justify-content: center;
                opacity: 0;
                pointer-events: none;
                transition: opacity 0.3s cubic-bezier(0.4, 0, 0.2, 1);
            }}
            #akshar-global-loader.is-loading {{
                opacity: 1;
                pointer-events: auto;
            }}
            .akshar-loader-spinner {{
                width: 65px;
                height: 65px;
                position: relative;
                animation: rotateSpinner 2s linear infinite;
                filter: drop-shadow(0 0 15px rgba(255,153,51,0.3));
            }}
            .akshar-loader-spinner circle {{
                fill: none;
                stroke: url(#loaderGoldGradient);
                stroke-width: 3.5;
                stroke-dasharray: 1, 200;
                stroke-dashoffset: 0;
                stroke-linecap: round;
                animation: dashPulse 1.5s ease-in-out infinite;
            }}
            .akshar-loader-text {{
                margin-top: 1.5rem;
                color: #FFB347;
                font-family: 'Cinzel', serif;
                font-size: 1.1rem;
                font-weight: 600;
                letter-spacing: 0.15em;
                text-transform: uppercase;
                animation: pulseText 1.5s ease-in-out infinite;
            }}
            @keyframes rotateSpinner {{ 100% {{ transform: rotate(360deg); }} }}
            @keyframes dashPulse {{
                0% {{ stroke-dasharray: 1, 200; stroke-dashoffset: 0; }}
                50% {{ stroke-dasharray: 89, 200; stroke-dashoffset: -35px; }}
                100% {{ stroke-dasharray: 89, 200; stroke-dashoffset: -124px; }}
            }}
            @keyframes pulseText {{
                0%, 100% {{ opacity: 0.6; filter: drop-shadow(0 0 2px rgba(255,153,51,0.1)); }}
                50% {{ opacity: 1; filter: drop-shadow(0 0 12px rgba(255,153,51,0.6)); }}
            }}
            /* Hide Streamlit Native Wait Elements to prevent clash */
            [data-testid="stStatusWidget"] {{ display: none !important; }}
        `;
        p.head.appendChild(style);

        const loader = p.createElement('div');
        loader.id = 'akshar-global-loader';
        loader.innerHTML = `
            <svg class="akshar-loader-spinner" viewBox="25 25 50 50">
                <defs>
                    <linearGradient id="loaderGoldGradient" x1="0%" y1="0%" x2="100%" y2="100%">
                        <stop offset="0%" stop-color="#FFB347"/>
                        <stop offset="50%" stop-color="#FF9933"/>
                        <stop offset="100%" stop-color="#CC7A29"/>
                    </linearGradient>
                </defs>
                <circle cx="50" cy="50" r="20"></circle>
            </svg>
            <div class="akshar-loader-text">Loading...</div>
        `;
        p.body.appendChild(loader);

        const msgs = ["AKSHAR.AI"];

        function forceLoaderTrigger(autoClearMs = 8000) {{
            const textEl = p.querySelector('.akshar-loader-text');
            if (textEl) textEl.textContent = msgs[0];
            const l = p.getElementById('akshar-global-loader');
            if (l) {{
                l.classList.add('is-loading');
                if (autoClearMs) {{
                    setTimeout(() => l.classList.remove('is-loading'), autoClearMs);
                }}
            }}
        }}

        // Handle Clicks (Tabs, Buttons, Options, Cards)
        p.addEventListener('click', (e) => {{
            let t = e.target;
            while(t && t !== p.body) {{
                let role = t.getAttribute ? t.getAttribute('role') : null;
                let cls = (t.className && typeof t.className === 'string') ? t.className : '';
                
                // Exclude pure audio players entirely to avoid false positive
                if (cls.includes('ak-audio') || cls.includes('learn-audio-play')) {{
                    break;
                }}
                
                // 1. Tab switches (Pure UI Layout React Rendering Lag)
                if (role === 'tab') {{
                    forceLoaderTrigger(600); // UI visual masking only
                    break;
                }}
                
                // 3 & 4. All Streamlit Buttons (Learn chars, Quiz answers, Camera buttons, etc.)
                // Also 5. Library Selectbox options (clicking an item in dropdown)
                if (t.tagName === 'BUTTON' || role === 'button' || role === 'option' || t.tagName === 'LI') {{
                    if (t.disabled || t.getAttribute('aria-disabled') === 'true') break;
                    // Provide an immediate loading mask. Python backend will clear it instantly when compute ends.
                    forceLoaderTrigger(8000); 
                    break;
                }}
                
                // 3. Learn Premium Cards / Char Cards
                if (cls.includes('premium-cat-card') || cls.includes('char-card-ui')) {{
                    forceLoaderTrigger(8000);
                    break;
                }}
                
                t = t.parentNode;
            }}
        }}, true);

        // Handle File Uploads and Select Changes
        p.addEventListener('change', (e) => {{
            let t = e.target;
            if (t && t.tagName === 'INPUT' && t.type === 'file') {{
                // 2. Slow file uploads
                forceLoaderTrigger(8000);
            }}
            if (t && t.tagName === 'SELECT') {{
                forceLoaderTrigger(8000);
            }}
        }}, true);

        // Handle Keyboard Events (Enter for Text Input)
        p.addEventListener('keydown', (e) => {{
            let t = e.target;
            // 5. Library tab text search
            if (t && t.tagName === 'INPUT' && (t.type === 'text' || t.type === 'search') && e.key === 'Enter') {{
                forceLoaderTrigger(8000);
            }}
        }}, true);
    }}

    // When Streamlit finishes processing, this script re-executes. We instantly clear the loader.
    const loader = p.getElementById('akshar-global-loader');
    if (loader) {{
        loader.classList.remove('is-loading');
    }}
    // Timestamp to force iframe execution on every Streamlit Python run: {time.time()}
}})();
</script>
"""
st.html(loading_js)
