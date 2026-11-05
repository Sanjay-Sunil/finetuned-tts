"""Streamlit demo: compare the released Pocket TTS with the Hinglish fine-tune.

    streamlit run app.py

The fine-tuned model loads from the private Hugging Face repo (set HF_TOKEN),
or from a local checkpoint when HINGLISH_CHECKPOINT points at one:

    HINGLISH_CHECKPOINT=runs/hinglish_24l_local/keep/checkpoint_00004000.pt streamlit run app.py
"""

import os
import random

import streamlit as st
import streamlit.components.v1 as components
import torch
from pocket_tts import TTSModel

from hinglish_tts.decision_pages import PAGES, page_html
from hinglish_tts.samples import SAMPLES
from hinglish_tts.text_norm import normalize

HF_CONFIG = "hf://zxnjxd/pocket-tts-finetuned/config.yaml"
LOCAL_CONFIG = "configs/inference_local.yaml"
VOICE = "voices/indictts_hi_female.wav"

st.set_page_config(page_title="Hinglish Pocket TTS", layout="wide")


@st.cache_resource(show_spinner="Loading the released and fine-tuned models...")
def load_models() -> dict:
    device = "cuda" if torch.cuda.is_available() else "cpu"
    before = TTSModel.load_model(language="english")
    checkpoint = os.environ.get("HINGLISH_CHECKPOINT")
    if checkpoint:
        after = TTSModel.load_model(config=LOCAL_CONFIG)
        after.load_training_checkpoint(checkpoint)
    else:
        after = TTSModel.load_model(config=HF_CONFIG)
    models = {}
    for name, model in (("before", before), ("after", after)):
        model.to(device)
        models[name] = (model, model.get_state_for_audio_prompt(VOICE))
    return models


def speak(name: str, text: str):
    model, state = load_models()[name]
    with torch.no_grad():
        audio = model.generate_audio(state, normalize(text))
    return audio.cpu().numpy(), model.sample_rate


def pick_sample():
    choices = [s for s in SAMPLES if s != st.session_state.get("text")]
    st.session_state.text = random.choice(choices)
    st.session_state.pop("audio", None)


st.title("Hinglish Pocket TTS")
st.caption(
    "Kyutai Pocket TTS (24-layer) fine-tuned for Hindi + English code-switching. "
    "Both models clone the same Indian voice prompt."
)
load_models()

try_tab, how_tab = st.tabs(["Try it", "How it was built"])

with try_tab:
    if "text" not in st.session_state:
        st.session_state.text = SAMPLES[0]
    col_text, col_button = st.columns([5, 1], vertical_alignment="bottom")
    with col_text:
        text = st.text_area(
            "Text (Hindi in Devanagari, English in Latin script)", key="text", height=90
        )
    with col_button:
        st.button("Random sample", on_click=pick_sample, width="stretch")
    st.caption(f"Model input after normalization: {normalize(text)}")

    audio = st.session_state.setdefault("audio", {})
    before_col, after_col = st.columns(2)
    for col, name, title, note in (
        (before_col, "before", "Before", "released Pocket TTS (English)"),
        (after_col, "after", "After", "Hinglish fine-tune, step 4000"),
    ):
        with col, st.container(border=True):
            st.subheader(title)
            st.caption(note)
            if st.button("Play", key=f"play_{name}", type="primary", disabled=not text.strip()):
                with st.spinner("Generating..."):
                    audio[name] = (text, *speak(name, text))
            if name in audio and audio[name][0] == text:
                _, wav, sr = audio[name]
                st.audio(wav, sample_rate=sr, autoplay=True)

with how_tab:
    st.caption("The key decisions behind the model, drawn live in your browser.")
    for page in PAGES:
        with st.container(border=True):
            components.html(page_html(page), height=720)
