import altair as alt
import pandas as pd
import streamlit as st
import torch
from pathlib import Path
from transformers import AutoModelForSequenceClassification, AutoTokenizer

# ---------------- PAGE CONFIG ----------------
st.set_page_config(
    page_title="Emotive — feelings, decoded",
    page_icon="💌",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ---------------- MODEL CONFIG ----------------
MODEL_PATH = Path(__file__).resolve().parent / "bert-emotion-model"
HF_FALLBACK_MODEL = "Shivdikar23/emotive-bert-emotion"
DEFAULT_EMOTIONS = ["sadness", "joy", "love", "anger", "fear", "surprise"]
EMOJI = {
    "sadness": "🥹", "joy": "☀️", "love": "💗",
    "anger": "🌶️", "fear": "🫣", "surprise": "✨"
}
OWN_TEXT = "Write your own..."
EXAMPLE_TEXTS = [
    "Just got the internship I was hoping for!!!!",
    "I miss the people I used to talk to every day.",
    "I feel so lucky to have you in my life.",
    "I cannot believe they treated me that way.",
    "I am worried this presentation will go badly.",
    "I did not expect that news at all!",
    "Can't stop smiling after that message.",
    "This caption is not it and I'm tired.",
]


def has_model_weights(model_path: Path) -> bool:
    return (
        (model_path / "model.safetensors").exists()
        or (model_path / "pytorch_model.bin").exists()
        or any(model_path.glob("model*.safetensors"))
    )


def emotions_from_config(model):
    id2label = getattr(model.config, "id2label", None) or {}
    if not id2label:
        return DEFAULT_EMOTIONS
    labels = []
    for i in range(len(id2label)):
        value = id2label.get(i, id2label.get(str(i), DEFAULT_EMOTIONS[i] if i < len(DEFAULT_EMOTIONS) else str(i)))
        value = str(value).lower().strip()
        if value.startswith("label_") and i < len(DEFAULT_EMOTIONS):
            value = DEFAULT_EMOTIONS[i]
        labels.append(value)
    return labels


# ---------------- VISUAL DESIGN ----------------
st.markdown(r"""
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;600;700&family=Nunito:wght@500;600;700;800;900&display=swap');
:root {
  --ink:#34324f; --muted:#85829f; --purple:#8277d8; --purple-dark:#6256b7;
  --line:#eae6f4; --white:#fffdfd; --pink:#ffe6ef; --blue:#e7f2ff;
  --mint:#e2f6ed; --yellow:#fff3cf;
}
html, body, [class*="css"] { font-family:'DM Sans',sans-serif; }
.stApp {
  background:
    radial-gradient(ellipse at 6% 8%, rgba(255,218,232,.72) 0, transparent 25%),
    radial-gradient(ellipse at 93% 12%, rgba(211,231,255,.78) 0, transparent 28%),
    radial-gradient(ellipse at 85% 88%, rgba(221,247,235,.55) 0, transparent 24%),
    #fbfaff;
  color:var(--ink);
}
.block-container { max-width:1160px; padding:1.6rem 2rem 2.5rem; }
header[data-testid="stHeader"] { background:transparent; }
footer { visibility:hidden; }
#MainMenu { visibility:hidden; }
.brandbar { display:flex; align-items:center; justify-content:space-between; gap:16px; margin-bottom:28px; }
.brand-left { display:flex; align-items:center; gap:12px; }
.logo { width:48px; height:48px; display:flex; align-items:center; justify-content:center; border-radius:17px; background:linear-gradient(145deg,#b9adff,#8c9ce9); color:white; font-size:25px; box-shadow:0 9px 22px #a69ae94d; transform:rotate(-5deg); }
.brand-name { color:#3f3b68; font:900 26px 'Nunito',sans-serif; letter-spacing:-.8px; line-height:1.05; }
.brand-tag { color:#a09ab9; font-size:10px; letter-spacing:2px; font-weight:800; margin-top:5px; }
.status-pill { background:#fff; border:1px solid var(--line); border-radius:999px; padding:9px 13px; color:#777391; font-size:12px; font-weight:700; box-shadow:0 4px 15px #635b8610; }
.hero { position:relative; padding:20px 0 22px; }
.hero-kicker { display:inline-flex; gap:7px; align-items:center; background:#f0edff; color:#7669c5; border:1px solid #e5defe; border-radius:999px; padding:7px 12px; font-size:11px; font-weight:800; letter-spacing:1px; text-transform:uppercase; }
.hero h1 { color:#383653; font:900 clamp(34px,4.5vw,54px)/1.08 'Nunito',sans-serif; letter-spacing:-1.8px; margin:17px 0 12px; max-width:800px; }
.hero h1 .highlight { color:#8878d6; }
.hero p { color:#85829f; font-size:15px; line-height:1.75; max-width:660px; margin:0; }
.hero-art { float:right; margin:0 4% 0 20px; width:142px; height:142px; border-radius:40% 60% 58% 42% / 45% 42% 58% 55%; background:linear-gradient(145deg,#ffe6f0,#e7e3ff 55%,#e2f3ff); display:flex; align-items:center; justify-content:center; font-size:62px; box-shadow:inset 0 0 0 1px #fff9,0 18px 36px #9e91c51c; animation:bob 5s ease-in-out infinite; }
@keyframes bob { 0%,100%{transform:translateY(0) rotate(3deg)} 50%{transform:translateY(-8px) rotate(-3deg)} }
.reaction-strip { display:flex; align-items:center; flex-wrap:wrap; gap:8px; padding:14px 0 22px; }
.reaction { border:1px solid #ece8f5; background:#ffffffc9; border-radius:999px; padding:8px 12px; font-size:12px; font-weight:700; color:#77728f; box-shadow:0 4px 12px #5c52790a; }
.reaction:nth-child(1){background:#fff0f5}.reaction:nth-child(2){background:#fff6db}.reaction:nth-child(3){background:#eaf5ff}.reaction:nth-child(4){background:#e8f7ef}.reaction:nth-child(5){background:#f0edff}.reaction:nth-child(6){background:#fff0e9}
.section-card { background:rgba(255,255,255,.88); border:1px solid #eae6f4; border-radius:24px; padding:24px; box-shadow:0 12px 34px #5c52790a; }
.section-title { color:#403d62; font:800 20px 'Nunito',sans-serif; margin:0 0 5px; }
.section-copy { color:#9691ad; font-size:13px; line-height:1.65; margin-bottom:15px; }
.input-label { color:#6f6a8c; font-size:12px; font-weight:800; letter-spacing:.4px; margin:4px 0 8px; }
div[data-testid="stTextArea"] textarea { background:#fffefe; border:1px solid #e6e0f2; border-radius:16px; color:#403d62; font-size:15px; padding:15px; line-height:1.65; }
div[data-testid="stTextArea"] textarea:focus { border-color:#b9adef; box-shadow:0 0 0 3px #b9adef2b; }
""", unsafe_allow_html=True)

# ---------------- MODEL + OCR ----------------
@st.cache_resource
def load_model(model_path: str):
    model_dir = Path(model_path)
    local_ok = (
        model_dir.is_dir()
        and (model_dir / "config.json").exists()
        and has_model_weights(model_dir)
    )
    source = str(model_dir) if local_ok else HF_FALLBACK_MODEL
    tokenizer = AutoTokenizer.from_pretrained(source, local_files_only=local_ok, use_fast=True)
    model = AutoModelForSequenceClassification.from_pretrained(source, local_files_only=local_ok)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()
    return tokenizer, model, device, local_ok


@st.cache_resource
def load_ocr():
    from rapidocr import RapidOCR
    return RapidOCR()


def extract_text_from_image(file_bytes: bytes) -> str:
    result = load_ocr()(file_bytes)
    texts = getattr(result, "txts", None)
    if not texts:
        return ""
    return "\n".join(line.strip() for line in texts if line and str(line).strip())


try:
    tokenizer, model, device, used_local = load_model(str(MODEL_PATH))
except ImportError as e:
    st.error("The installed Transformers/PyTorch versions are incompatible. Check your project environment.")
    st.exception(e)
    st.stop()
except Exception as e:
    st.error("Could not load the emotion model.")
    if not has_model_weights(MODEL_PATH):
        st.info(f"Place the trained model files in `bert-emotion-model`, or allow downloading `{HF_FALLBACK_MODEL}`.")
    st.exception(e)
    st.stop()

EMOTIONS = emotions_from_config(model)

# ---------------- HEADER ----------------
st.markdown("""
<div class="brandbar">
  <div class="brand-left">
    <div class="logo">✳</div>
    <div><div class="brand-name">emotive.</div><div class="brand-tag">YOUR LITTLE FEELINGS LAB</div></div>
  </div>
  <div class="status-pill">✦ BERT-powered · 6 emotions</div>
</div>
<div class="hero">
  <div class="hero-art">૮ ˶ᵔ ᵕ ᵔ˶ ა</div>
  <div class="hero-kicker">✦ a tiny bit of emotional insight</div>
  <h1>What's the <span class="highlight">feeling</span><br>behind the words?</h1>
  <p>Drop in a caption, chat, or late-night thought. We'll read the words and find the emotion hiding between the lines.</p>
</div>
<div class="reaction-strip">
  <span class="reaction">☀️ joy energy</span><span class="reaction">🥹 soft hours</span>
  <span class="reaction">💌 in my feelings</span><span class="reaction">🌶️ a little spicy</span>
  <span class="reaction">🫣 nervous era</span><span class="reaction">✨ plot twist</span>
</div>
""", unsafe_allow_html=True)

if not used_local:
    st.info(f"Using the Hugging Face model `{HF_FALLBACK_MODEL}` because local model weights were not found.")
# ---------------- INPUT STATE + INPUT MODES ----------------
if "input_text" not in st.session_state:
    st.session_state.input_text = ""
if "_last_example" not in st.session_state:
    st.session_state._last_example = OWN_TEXT

mode = st.segmented_control(
    "Input method",
    options=["Paste text", "Upload screenshot"],
    default="Paste text",
    required=True,
    label_visibility="collapsed",
)

if mode == "Upload screenshot":
    uploaded = st.file_uploader(
        "Upload a screenshot of a message, caption, or post",
        type=["png", "jpg", "jpeg", "webp"],
        help="Text will be extracted from the image before emotion analysis.",
    )
    if uploaded is not None:
        file_bytes = uploaded.getvalue()
        file_id = f"{uploaded.name}-{uploaded.size}"
        st.image(file_bytes, caption=uploaded.name, width=300)
        if st.session_state.get("_ocr_file_id") != file_id:
            try:
                with st.spinner("Reading the text in your screenshot..."):
                    extracted = extract_text_from_image(file_bytes)
                st.session_state.input_text = extracted
                st.session_state._ocr_file_id = file_id
                st.session_state._last_example = OWN_TEXT
                if not extracted:
                    st.warning("I couldn't find readable text. Try a clearer screenshot or type the text below.")
            except Exception as e:
                st.error("Screenshot text extraction failed. Check that RapidOCR is installed.")
                st.exception(e)
        else:
            st.caption("Text extracted — check it below and edit if needed.")
else:
    example = st.selectbox("Pick a mood sample", [OWN_TEXT] + EXAMPLE_TEXTS)
    if example != st.session_state._last_example:
        st.session_state.input_text = "" if example == OWN_TEXT else example
        st.session_state._last_example = example

# ---------------- ANALYSIS FORM ----------------
st.markdown('<div class="section-card"><div class="section-title">💌 Your message</div><div class="section-copy">No judgement here. Paste a post, a text, or whatever you want to understand.</div>', unsafe_allow_html=True)
with st.form("emotion_form"):
    text = st.text_area(
        "Your text",
        key="input_text",
        placeholder="e.g. can't stop smiling after that message...",
        height=155,
        max_chars=5000,
        label_visibility="collapsed",
    )
    submit_col, hint_col = st.columns([1, 1.5], vertical_alignment="center")
    with submit_col:
        submitted = st.form_submit_button("✦  Read the vibe", use_container_width=True)
    with hint_col:
        st.markdown('<div class="small-note">A sentence is enough. Longer text works too — max 5,000 characters.</div>', unsafe_allow_html=True)
st.markdown('</div>', unsafe_allow_html=True)
# ---------------- PREDICTION ----------------
if submitted:
    if not text.strip():
        st.warning("Add a little text first, then tap Read the vibe ✨")
    else:
        try:
            with st.spinner("Reading the vibe... ✨"):
                inputs = tokenizer(text.strip(), return_tensors="pt", truncation=True, padding=True, max_length=128)
                inputs = {k: v.to(device) for k, v in inputs.items()}
                with torch.inference_mode():
                    outputs = model(**inputs)
                    probs = torch.softmax(outputs.logits.float(), dim=-1)[0]
                probs = probs.detach().cpu().numpy()

            predicted_id = int(probs.argmax())
            if predicted_id >= len(EMOTIONS):
                raise RuntimeError(f"Model predicted class {predicted_id}, but only {len(EMOTIONS)} labels are configured.")
            predicted_emotion = EMOTIONS[predicted_id].lower()
            confidence = float(probs[predicted_id])
            emoji = EMOJI.get(predicted_emotion, "💬")

            st.markdown("<hr>", unsafe_allow_html=True)
            st.markdown('<div class="hero-kicker">✦ your vibe check is ready</div>', unsafe_allow_html=True)
            st.markdown('<h2 style="font-family:Nunito,sans-serif;color:#403d62;font-weight:900;margin:12px 0 18px">The feeling behind your words</h2>', unsafe_allow_html=True)
            result_col, score_col = st.columns([1.45, 1], gap="large")
            with result_col:
                st.markdown(f"""
                <div class="result-card">
                  <div class="result-sparkle">✧</div>
                  <div class="result-eyebrow">MOST LIKELY EMOTION</div>
                  <div class="result-name">{emoji} {predicted_emotion.title()}</div>
                  <div class="result-copy">That's the emotion your text most strongly signals.</div>
                </div>
                """, unsafe_allow_html=True)
            with score_col:
                st.markdown(f"""
                <div class="score-card">
                  <div class="score-label">Model confidence</div>
                  <div class="score-number">{confidence:.1%}</div>
                  <div class="small-note">The model's estimated probability for this prediction.</div>
                </div>
                """, unsafe_allow_html=True)
                st.write("")
                st.progress(confidence)
                st.markdown('<div class="small-note">Confidence is not a guarantee that the prediction is correct.</div>', unsafe_allow_html=True)

            result_df = pd.DataFrame({
                "Emotion": [e.title() for e in EMOTIONS],
                "Probability": [float(p) for p in probs[:len(EMOTIONS)]],
            }).sort_values("Probability", ascending=False)

            st.markdown('<div class="section-card"><div class="section-title">🎀 The full reaction mix</div><div class="section-copy">A peek at how strongly the model considered each emotion.</div>', unsafe_allow_html=True)
            chart = (
                alt.Chart(result_df)
                .mark_bar(cornerRadiusEnd=8, color="#9b8ce5", height=22)
                .encode(
                    x=alt.X("Probability:Q", scale=alt.Scale(domain=[0, 1]), axis=alt.Axis(format="%", title=None, grid=False)),
                    y=alt.Y("Emotion:N", sort="-x", title=None, axis=alt.Axis(labelColor="#77728f", ticks=False, domain=False)),
                    tooltip=[alt.Tooltip("Emotion:N"), alt.Tooltip("Probability:Q", format=".1%")],
                )
                .properties(height=255, background="transparent", padding={"left": 4, "right": 12, "top": 8, "bottom": 4})
                .configure_view(strokeOpacity=0)
                .configure_axis(gridColor="#f0edf7", labelFont="DM Sans", titleFont="DM Sans")
            )
            st.altair_chart(chart, use_container_width=True)
            with st.expander("See exact emotion scores"):
                display_df = result_df.copy()
                display_df["Probability"] = display_df["Probability"].map(lambda x: f"{x:.2%}")
                st.dataframe(display_df, hide_index=True, use_container_width=True)
            st.markdown('</div>', unsafe_allow_html=True)
            st.markdown('<div class="small-note">A gentle reminder: this tool recognises language patterns. It can’t know someone’s true feelings for certain.</div>', unsafe_allow_html=True)
        except Exception as e:
            st.error("The prediction couldn't be completed.")
            st.exception(e)
# ---------------- FOOTER ----------------
st.markdown("""
<div class="footer">made with 💗 and a little machine learning &nbsp; · &nbsp; EMOTIVE</div>
""", unsafe_allow_html=True)
