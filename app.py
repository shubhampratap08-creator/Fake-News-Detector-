import os
import numpy as np
import pandas as pd
import joblib
import streamlit as st
from train import clean_text

st.set_page_config(page_title="Fake News Detector", page_icon="📰", layout="wide")

st.markdown("""
<style>
.hero {background: linear-gradient(135deg,#4f46e5,#7c3aed); padding: 28px 32px;
       border-radius: 16px; color: white; margin-bottom: 24px;}
.hero h1 {margin: 0; font-size: 2.1rem;}
.hero p {margin: 6px 0 0 0; opacity: .9;}
.result {padding: 22px; border-radius: 14px; text-align: center; margin: 12px 0;}
.result h2 {margin: 0;}
.real {background:#e8f8ee; color:#166534; border:1px solid #86efac;}
.fake {background:#fdecec; color:#991b1b; border:1px solid #fca5a5;}
</style>
<div class="hero">
  <h1>📰 Fake News Detector</h1>
  <p>Paste a headline or article and a machine learning model will estimate whether it looks real or fake.</p>
</div>
""", unsafe_allow_html=True)

EXAMPLES = {
    "Select an example...": "",
    "Example 1 (reads like real news)": "The Senate voted on Tuesday to approve a federal budget bill after weeks of negotiations between party leaders, with the final tally of 68 to 32 sending the measure to the president for signature.",
    "Example 2 (reads like fake news)": "SHOCKING!!! You will not believe what the media is hiding from you! Secret documents LEAKED prove everything you were told is a lie, share before they delete this!",
}


@st.cache_resource
def load():
    return joblib.load("models/model.joblib"), joblib.load("models/vectorizer.joblib")


try:
    model, vectorizer = load()
except FileNotFoundError:
    st.error("Model not found. Run `python train.py` first.")
    st.stop()


def get_confidence(vec):
    if hasattr(model, "predict_proba"):
        return float(model.predict_proba(vec)[0].max())
    if hasattr(model, "decision_function"):
        d = abs(float(model.decision_function(vec)[0]))
        return 1 / (1 + np.exp(-d))
    return None


def top_words(vec, pred, n=8):
    if not hasattr(model, "coef_"):
        return None
    coef = model.coef_[0]
    feats = vectorizer.get_feature_names_out()
    row = vec.tocoo()
    contrib = {feats[j]: v * coef[j] for j, v in zip(row.col, row.data)}
    sign = 1 if pred == 1 else -1
    top = sorted(contrib.items(), key=lambda x: x[1] * sign, reverse=True)[:n]
    return pd.DataFrame({"word": [w for w, _ in top],
                         "influence": [abs(s) for _, s in top]}).set_index("word")


with st.sidebar:
    st.header("About")
    st.write("B.Tech minor project: fake news detection using TF-IDF and machine learning.")
    st.write(f"**Model in use:** {type(model).__name__}")
    st.divider()
    st.write("**Made by:** Shubham, Chirag, Dhruv")
    st.write("**College:** CBS Group of Institutions")
    st.divider()
    st.caption("This is a statistical model, not a fact-checker. It learns writing patterns, "
               "not whether a claim is true.")

tab1, tab2, tab3 = st.tabs(["🔍 Analyze", "📊 Model performance", "ℹ️ How it works"])

with tab1:
    choice = st.selectbox("Try an example", list(EXAMPLES.keys()))
    text = st.text_area("News text", value=EXAMPLES[choice], height=220,
                        placeholder="Paste a news headline or full article here...")
    st.caption(f"{len(text.split())} words")

    if st.button("Analyze", type="primary", use_container_width=True):
        if len(text.split()) < 5:
            st.warning("Please enter at least a few words.")
        else:
            with st.spinner("Analyzing..."):
                vec = vectorizer.transform([clean_text(text)])
                pred = int(model.predict(vec)[0])
                conf = get_confidence(vec)

            if pred == 1:
                st.markdown('<div class="result real"><h2>✅ Looks like REAL news</h2></div>',
                            unsafe_allow_html=True)
            else:
                st.markdown('<div class="result fake"><h2>🚨 Looks like FAKE news</h2></div>',
                            unsafe_allow_html=True)

            if conf is not None:
                st.metric("Confidence", f"{conf:.1%}")
                st.progress(min(conf, 1.0))

            words = top_words(vec, pred)
            if words is not None and not words.empty:
                st.subheader("Words that influenced this result")
                st.bar_chart(words)

with tab2:
    if os.path.exists("outputs/model_comparison.csv"):
        st.subheader("Model comparison")
        st.dataframe(pd.read_csv("outputs/model_comparison.csv"),
                     use_container_width=True, hide_index=True)
    else:
        st.info("Run `python train.py` to generate the comparison table.")
    if os.path.exists("outputs/confusion_matrix.png"):
        st.subheader("Confusion matrix (best model)")
        st.image("outputs/confusion_matrix.png", width=420)

with tab3:
    st.markdown("""
**Pipeline**
1. **Clean** the text: lowercase, remove links, punctuation and stopwords, then lemmatize.
2. **TF-IDF** converts the text into numbers that score how distinctive each word is.
3. **Classifier** trained on thousands of labelled articles predicts real or fake.
4. The app shows the verdict, a confidence score and the words that influenced it.

**Limitations**
- Trained on mostly political news, so accuracy may drop on other topics.
- It detects writing style, not factual correctness.
""")
