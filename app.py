import re
import numpy as np
import streamlit as st
import streamlit.components.v1 as components
import tensorflow as tf

imdb = tf.keras.datasets.imdb
sequence = tf.keras.preprocessing.sequence
load_model = tf.keras.models.load_model

# ============================================================================
# PAGE CONFIG
# ============================================================================
st.set_page_config(
    page_title="CineSentiment — The Tug of War",
    page_icon="🎬",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ============================================================================
# MODEL / VOCAB LOADING (cached so it only happens once)
# ============================================================================
@st.cache_resource(show_spinner="🎞️ Waking up the review-reading brain...")
def load_assets():
    word_index = imdb.get_word_index()
    reverse_word_index = {value: key for key, value in word_index.items()}
    model = load_model("simple_rnn_imdb_model.h5")
    return word_index, reverse_word_index, model

try:
    word_index, reverse_word_index, model = load_assets()
    ASSETS_OK = True
except Exception as e:
    ASSETS_OK = False
    st.error(f"Couldn't load model/vocab: {e}")

def decode_review(encoded_review):
    return " ".join([reverse_word_index.get(i - 3, "?") for i in encoded_review])

# ============================================================================
# ⚠️ PROTECTED SECTION — DO NOT MODIFY THESE TWO FUNCTIONS
# ============================================================================
def preprocess_text(text):
    text = text.lower()
    text = re.sub(r"[^a-zA-Z0-9\s]", "", text)
    words = text.split()
    encoded_review = []
    for word in words:
        index = word_index.get(word)
        if index is None:
            index = 2
        else:
            index = index + 3
            if index >= 10000:
                index = 2
        encoded_review.append(index)
    padded_review = sequence.pad_sequences(
        [encoded_review],
        maxlen=500
    )
    return padded_review


def predict_sentiment(review):
    preprocessed_input = preprocess_text(review)
    prediction = model.predict(
        preprocessed_input,
        verbose=0
    )
    score = float(prediction[0][0])
    sentiment = "Positive" if score >= 0.5 else "Negative"
    return sentiment, score
# ============================================================================
# END PROTECTED SECTION
# ============================================================================


# ============================================================================
# COLOR HELPERS
# ============================================================================
RED = (231, 76, 60)
GREEN = (46, 204, 113)

def mix_color(t, c1=RED, c2=GREEN):
    t = max(0.0, min(1.0, t))
    r = int(c1[0] + (c2[0] - c1[0]) * t)
    g = int(c1[1] + (c2[1] - c1[1]) * t)
    b = int(c1[2] + (c2[2] - c1[2]) * t)
    return f"rgb({r},{g},{b})"

def rgba(t, alpha, c1=RED, c2=GREEN):
    t = max(0.0, min(1.0, t))
    r = int(c1[0] + (c2[0] - c1[0]) * t)
    g = int(c1[1] + (c2[1] - c1[1]) * t)
    b = int(c1[2] + (c2[2] - c1[2]) * t)
    return f"rgba({r},{g},{b},{alpha})"


# ============================================================================
# SESSION STATE
# ============================================================================
if "history" not in st.session_state:
    st.session_state.history = []
if "total_score" not in st.session_state:
    st.session_state.total_score = 0.0
if "count" not in st.session_state:
    st.session_state.count = 0
if "review_text" not in st.session_state:
    st.session_state.review_text = ""
if "last_result" not in st.session_state:
    st.session_state.last_result = None

avg_score = (st.session_state.total_score / st.session_state.count) if st.session_state.count > 0 else 0.5

def set_example(text):
    st.session_state.review_text = text

def reset_battle():
    st.session_state.history = []
    st.session_state.total_score = 0.0
    st.session_state.count = 0
    st.session_state.last_result = None

def analyze():
    review = st.session_state.review_text.strip()
    if not review:
        st.session_state.last_result = "empty"
        return
    sentiment, score = predict_sentiment(review)
    st.session_state.history.insert(0, {
        "review": review,
        "sentiment": sentiment,
        "score": score,
    })
    st.session_state.total_score += score
    st.session_state.count += 1
    st.session_state.last_result = {"review": review, "sentiment": sentiment, "score": score}


# ============================================================================
# GLOBAL CSS — the whole site tints toward the current majority color
# ============================================================================
glow = rgba(avg_score, 0.28)
glow2 = rgba(avg_score, 0.14)
accent = mix_color(avg_score)

st.markdown(f"""
<style>
.stApp {{
    background:
        radial-gradient(circle at 15% 10%, {glow} 0%, transparent 45%),
        radial-gradient(circle at 85% 90%, {glow2} 0%, transparent 50%),
        #0e0e16;
    transition: background 0.8s ease;
}}
h1, h2, h3 {{
    font-family: 'Trebuchet MS', sans-serif;
}}
.title-glow {{
    text-align:center;
    font-size:3rem;
    font-weight:900;
    background: linear-gradient(90deg, #e74c3c, #f1c40f, #2ecc71);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    padding-bottom: 0.1em;
}}
.subtitle {{
    text-align:center;
    color:#aaa;
    font-size:1.05rem;
    margin-bottom: 1.5em;
}}
.mood-banner {{
    text-align:center;
    padding: 14px;
    border-radius: 14px;
    font-size:1.2rem;
    font-weight:700;
    color:white;
    background: linear-gradient(90deg, {mix_color(0)}, {accent}, {mix_color(1)});
    box-shadow: 0 0 25px {glow};
    margin-bottom: 1.5em;
}}
.result-card {{
    border-radius: 16px;
    padding: 22px;
    background: rgba(255,255,255,0.04);
    border: 1px solid rgba(255,255,255,0.08);
    box-shadow: 0 0 30px {rgba(avg_score, 0.08)};
}}
.score-badge {{
    display:inline-block;
    padding: 6px 18px;
    border-radius: 999px;
    font-weight:800;
    font-size:1.1rem;
    color:white;
}}
.stButton>button {{
    border-radius: 10px;
    font-weight:600;
    transition: 0.2s;
}}
.stButton>button:hover {{
    transform: translateY(-2px) scale(1.02);
}}
</style>
""", unsafe_allow_html=True)


# ============================================================================
# TUG-OF-WAR WIDGET (per-review visualization)
# ============================================================================
def tug_of_war_html(score, sentiment):
    knot_pos = score * 100  # 0 = far red (left), 100 = far green (right)
    knot_color = mix_color(score)
    red_shake = round(2 + (1 - score) * 8, 1)   # bigger shake if red is winning
    green_shake = round(2 + score * 8, 1)
    red_speed = round(1.4 - (1 - score) * 0.9, 2)
    green_speed = round(1.4 - score * 0.9, 2)
    winner = "GREEN 🟢" if score >= 0.5 else "RED 🔴"

    html = f"""
    <div style="font-family:'Trebuchet MS',sans-serif; padding:10px 0;">
      <style>
        @keyframes shakeLeft {{
            0%,100% {{ transform: translateX(0) rotate(0deg); }}
            50% {{ transform: translateX(-{red_shake}px) rotate(-3deg); }}
        }}
        @keyframes shakeRight {{
            0%,100% {{ transform: translateX(0) rotate(0deg); }}
            50% {{ transform: translateX({green_shake}px) rotate(3deg); }}
        }}
        .team-red   {{ animation: shakeLeft {red_speed}s ease-in-out infinite; }}
        .team-green {{ animation: shakeRight {green_speed}s ease-in-out infinite; }}
        .rope-track {{
            position:relative; height:70px; margin:20px 0;
            background: repeating-linear-gradient(90deg,
                #6b4423 0px, #6b4423 18px, #543319 18px, #543319 22px);
            border-radius: 10px;
            box-shadow: inset 0 0 12px rgba(0,0,0,0.5);
        }}
        .center-line {{
            position:absolute; left:50%; top:-8px; bottom:-8px;
            width:2px; background: rgba(255,255,255,0.4);
        }}
        .knot {{
            position:absolute; top:50%;
            left: calc({knot_pos}% - 18px);
            transform: translateY(-50%);
            width:36px; height:36px; border-radius:50%;
            background:{knot_color};
            box-shadow: 0 0 18px {knot_color}, 0 0 4px #fff inset;
            transition: left 1.2s cubic-bezier(.34,1.56,.64,1);
            display:flex; align-items:center; justify-content:center;
            font-size:18px;
        }}
      </style>

      <div style="display:flex; justify-content:space-between; align-items:center;">
        <div class="team-red" style="font-size:48px; text-align:center;">
            😠<br><span style="font-size:13px; color:#e74c3c; font-weight:700;">TEAM NEGATIVE</span>
        </div>
        <div style="flex:1; margin:0 18px;">
            <div class="rope-track">
                <div class="center-line"></div>
                <div class="knot">🚩</div>
            </div>
        </div>
        <div class="team-green" style="font-size:48px; text-align:center;">
            😄<br><span style="font-size:13px; color:#2ecc71; font-weight:700;">TEAM POSITIVE</span>
        </div>
      </div>

      <div style="text-align:center; margin-top:6px; color:#ccc; font-size:14px;">
          Rope pulled by <b style="color:{knot_color};">{winner}</b> &nbsp;|&nbsp;
          Pull strength — Red: {round((1-score)*100)}% &nbsp;vs&nbsp; Green: {round(score*100)}%
      </div>
    </div>
    """
    components.html(html, height=200)


# ============================================================================
# HEADER
# ============================================================================
st.markdown('<div class="title-glow">🎬 CineSentiment: The Tug of War</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="subtitle">Type a movie review. Watch Team Positive and Team Negative '
    'battle it out — and slowly decide the mood color of the whole site.</div>',
    unsafe_allow_html=True
)

if st.session_state.count > 0:
    dominant = "GREEN 🟢 (Positive)" if avg_score >= 0.5 else "RED 🔴 (Negative)"
    st.markdown(
        f'<div class="mood-banner">🏆 Current site majority: {dominant} '
        f'&nbsp;·&nbsp; overall mood score {avg_score:.2f} '
        f'&nbsp;·&nbsp; {st.session_state.count} reviews battled so far</div>',
        unsafe_allow_html=True
    )
else:
    st.markdown(
        '<div class="mood-banner">⚖️ The site is perfectly balanced. Submit a review to tip the scales!</div>',
        unsafe_allow_html=True
    )

# ============================================================================
# SIDEBAR — STATS
# ============================================================================
with st.sidebar:
    st.header("📊 Battle Stats")
    pos_count = sum(1 for h in st.session_state.history if h["sentiment"] == "Positive")
    neg_count = st.session_state.count - pos_count

    c1, c2 = st.columns(2)
    c1.metric("🟢 Positive", pos_count)
    c2.metric("🔴 Negative", neg_count)

    st.progress(avg_score, text=f"Site mood: {avg_score*100:.1f}% green")

    st.markdown("---")
    st.subheader("🎯 Try a quick example")
    st.button("😍 Glowing rave review", use_container_width=True,
              on_click=set_example,
              args=("This movie was an absolute masterpiece, the acting was phenomenal and I was moved to tears by the ending.",))
    st.button("🤢 Brutal takedown", use_container_width=True,
              on_click=set_example,
              args=("This was a complete waste of time, terrible acting, awful script, I want my money back.",))
    st.button("😐 Mixed bag", use_container_width=True,
              on_click=set_example,
              args=("The visuals were stunning but the plot was slow and the dialogue felt forced at times.",))

    st.markdown("---")
    st.button("🔄 Reset the battlefield", use_container_width=True, on_click=reset_battle)

# ============================================================================
# MAIN INPUT
# ============================================================================
left, right = st.columns([2, 1])

with left:
    st.text_area(
        "✍️ Write or paste a movie review",
        key="review_text",
        height=150,
        placeholder="e.g. The cinematography was breathtaking but the plot dragged on forever...",
    )
    st.button("⚔️ Send it into battle", type="primary", on_click=analyze, disabled=not ASSETS_OK)

with right:
    st.markdown("#### How it works")
    st.markdown(
        "- Your review is fed to a Simple RNN trained on IMDB data\n"
        "- The model outputs a score from **0 (very negative)** to **1 (very positive)**\n"
        "- Team Red and Team Green tug the rope toward their side\n"
        "- Every review nudges the **whole site's mood color**"
    )

# ============================================================================
# RESULTS
# ============================================================================
result = st.session_state.last_result

if result == "empty":
    st.warning("Please type a review before sending it into battle. ✍️")
elif isinstance(result, dict):
    sentiment = result["sentiment"]
    score = result["score"]
    badge_color = mix_color(score)
    emoji = "🎉" if score >= 0.5 else "💥"

    st.markdown("### 🏟️ Battle Result")
    st.markdown(f"""
    <div class="result-card">
        <p style="font-size:1.05rem; color:#ddd;">Review: <i>"{result['review'][:220]}{'...' if len(result['review'])>220 else ''}"</i></p>
        <span class="score-badge" style="background:{badge_color};">
            {emoji} {sentiment.upper()} &nbsp;|&nbsp; score {score:.4f}
        </span>
    </div>
    """, unsafe_allow_html=True)

    tug_of_war_html(score, sentiment)
    st.progress(score, text=f"Positivity meter: {score*100:.1f}%")

    if score >= 0.85:
        st.balloons()
    elif score <= 0.15:
        st.snow()

# ============================================================================
# BATTLE LOG
# ============================================================================
if st.session_state.history:
    with st.expander(f"📜 Battle log ({len(st.session_state.history)} reviews)", expanded=False):
        for h in st.session_state.history:
            c = mix_color(h["score"])
            st.markdown(
                f"<div style='padding:8px 12px; border-left:4px solid {c}; margin-bottom:6px; "
                f"background:rgba(255,255,255,0.03); border-radius:6px;'>"
                f"<b style='color:{c};'>{h['sentiment']}</b> ({h['score']:.3f}) — "
                f"{h['review'][:140]}{'...' if len(h['review'])>140 else ''}</div>",
                unsafe_allow_html=True
            )