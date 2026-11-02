from pathlib import Path
import json
import joblib
import streamlit as st

ROOT = Path(__file__).resolve().parent
MODEL_PATH = ROOT / "models" / "fake_news_pipeline.joblib"
METRICS_PATH = ROOT / "models" / "metrics.json"

st.set_page_config(
    page_title="TruthLens | Fake News Detector",
    page_icon="🔎",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;600;700;800&display=swap');
html, body, [class*="css"] { font-family: 'DM Sans', sans-serif; }
.stApp { background: #0b1020; color: #eef2ff; }
header[data-testid="stHeader"] { background: rgba(11,16,32,0.9); }
.block-container { max-width: 1120px; padding-top: 2rem; padding-bottom: 3rem; }
.hero {
  padding: 2rem 2.2rem; border: 1px solid #263451; border-radius: 22px;
  background: linear-gradient(120deg, #172642 0%, #111a30 55%, #162b3d 100%);
  margin-bottom: 1.4rem;
}
.eyebrow { color: #67e8c2; font-size: .78rem; font-weight: 800; letter-spacing: .15em; }
.hero h1 { color: #f8fafc; font-size: 2.35rem; margin: .35rem 0 .5rem; }
.hero p { color: #b8c5dc; font-size: 1rem; margin: 0; }
.panel {
  background: #121b2e; border: 1px solid #273650; border-radius: 18px;
  padding: 1.2rem 1.35rem; margin: .7rem 0;
}
.metric {
  background: #121b2e; border: 1px solid #273650; border-radius: 15px;
  padding: 1rem 1.1rem;
}
.metric-label { color: #9aabc5; font-size: .82rem; }
.metric-value { color: #f8fafc; font-size: 1.65rem; font-weight: 800; }
.result {
  border-radius: 17px; padding: 1.2rem 1.4rem; margin-top: 1rem;
  border: 1px solid;
}
.fake { background: #351b28; border-color: #a83c58; }
.real { background: #12352f; border-color: #238a70; }
.result h2 { margin: 0 0 .3rem; color: #fff; }
.result p { margin: 0; color: #d4deed; }
.small-note { color: #9aabc5; font-size: .82rem; }
div.stButton > button {
  border-radius: 11px; border: 0; font-weight: 700; min-height: 2.8rem;
}
textarea { border-radius: 12px !important; }
[data-testid="stSidebar"] { background: #0e1729; border-right: 1px solid #263451; }
</style>
""", unsafe_allow_html=True)

@st.cache_resource
def load_model():
    return joblib.load(MODEL_PATH)

def load_metrics():
    if METRICS_PATH.exists():
        with open(METRICS_PATH, "r", encoding="utf-8") as f:
            return json.load(f)
    return {}

with st.sidebar:
    st.markdown("## 🔎 TruthLens")
    st.caption("AI-Based Fake News Detection")
    page = st.radio("NAVIGATION", ["Detect News", "About Project", "Model Details"])
    st.divider()
    st.markdown("**Built with**")
    st.caption("Python · NLP · TF-IDF · Logistic Regression · Streamlit")
    st.caption("Designed for local use on Apple Silicon")

st.markdown("""
<div class="hero">
  <div class="eyebrow">AI-POWERED TEXT CLASSIFICATION</div>
  <h1>TruthLens</h1>
  <p>A simple machine-learning assistant that checks whether a news article resembles
  examples labelled Fake or Real in its training dataset.</p>
</div>
""", unsafe_allow_html=True)

if not MODEL_PATH.exists():
    st.error("The trained model has not been created yet.")
    st.markdown("""
    **First-time setup**
    1. Add `Fake.csv` and `True.csv` to the `dataset` folder.
    2. Open the VS Code terminal and run `python train_model.py`.
    3. Then run `streamlit run app.py`.
    """)
    st.stop()

model = load_model()
metrics = load_metrics()

if page == "Detect News":
    st.markdown("### Analyse a news article")
    st.write("Paste a headline or article below. Longer text usually gives the model more wording to examine.")
    article = st.text_area(
        "News text",
        height=230,
        placeholder="Paste the news headline and article text here...",
        label_visibility="collapsed"
    )
    left, right = st.columns([1, 1])
    with left:
        analyse = st.button("🔍  Analyse News", type="primary", use_container_width=True)
    with right:
        clear = st.button("Clear text", use_container_width=True)

    if clear:
        st.rerun()

    if analyse:
        cleaned = article.strip()
        if len(cleaned) < 30:
            st.warning("Please enter a little more text (at least 30 characters) for a useful prediction.")
        else:
            prediction = model.predict([cleaned])[0]
            probabilities = model.predict_proba([cleaned])[0]
            classes = list(model.classes_)
            confidence = float(probabilities[classes.index(prediction)]) * 100

            if prediction == "Fake":
                st.markdown(f"""
                <div class="result fake">
                  <h2>⚠️ Prediction: Fake</h2>
                  <p>The text resembles patterns labelled Fake in the training dataset.</p>
                </div>""", unsafe_allow_html=True)
            else:
                st.markdown(f"""
                <div class="result real">
                  <h2>✓ Prediction: Real</h2>
                  <p>The text resembles patterns labelled Real in the training dataset.</p>
                </div>""", unsafe_allow_html=True)

            st.markdown("")
            c1, c2 = st.columns(2)
            with c1:
                st.metric("Model confidence", f"{confidence:.1f}%")
            with c2:
                st.metric("Predicted category", prediction)
            st.progress(min(max(confidence / 100, 0.0), 1.0))
            st.info("Important: this percentage is the model's confidence in its learned class, not the probability that the story is factually true. Verify important claims with reliable independent sources.")

    st.markdown("""
    <p class="small-note">Tip: include the complete article rather than only a few words.</p>
    """, unsafe_allow_html=True)

elif page == "About Project":
    st.markdown("### About the project")
    st.markdown("""
    <div class="panel">
      <h3>AI-Based Fake News Detection System</h3>
      This minor project demonstrates supervised machine learning for classifying
      English-language news text. It uses labelled examples from <b>Fake.csv</b> and
      <b>True.csv</b>.
    </div>
    """, unsafe_allow_html=True)
    st.markdown("**Project workflow**")
    steps = [
        ("01", "Dataset", "Load Fake.csv and True.csv and assign labels."),
        ("02", "Preprocessing", "Combine available headline and article text; remove empty and duplicate text."),
        ("03", "Feature extraction", "Convert text into numerical TF-IDF features."),
        ("04", "Classification", "Train a Logistic Regression classifier."),
        ("05", "Evaluation", "Evaluate predictions on a held-out test split."),
        ("06", "Deployment", "Use the saved pipeline in this Streamlit interface.")
    ]
    for number, heading, description in steps:
        st.markdown(f"""
        <div class="panel"><b style="color:#67e8c2">{number}</b> &nbsp; <b>{heading}</b><br>
        <span style="color:#b8c5dc">{description}</span></div>
        """, unsafe_allow_html=True)

elif page == "Model Details":
    st.markdown("### Model performance")
    if metrics:
        accuracy = metrics.get("accuracy", 0)
        a, b, c = st.columns(3)
        with a:
            st.markdown(f'<div class="metric"><div class="metric-label">Test accuracy</div><div class="metric-value">{accuracy*100:.2f}%</div></div>', unsafe_allow_html=True)
        with b:
            st.markdown(f'<div class="metric"><div class="metric-label">Total articles</div><div class="metric-value">{metrics.get("total_articles", 0):,}</div></div>', unsafe_allow_html=True)
        with c:
            st.markdown(f'<div class="metric"><div class="metric-label">Test samples</div><div class="metric-value">{metrics.get("test_articles", 0):,}</div></div>', unsafe_allow_html=True)

        report = metrics.get("classification_report", {})
        rows = []
        for label in ("Fake", "Real"):
            item = report.get(label, {})
            if item:
                rows.append({
                    "Class": label,
                    "Precision": round(item.get("precision", 0), 3),
                    "Recall": round(item.get("recall", 0), 3),
                    "F1-score": round(item.get("f1-score", 0), 3),
                    "Support": int(item.get("support", 0))
                })
        if rows:
            st.markdown("#### Classification report")
            st.dataframe(rows, use_container_width=True, hide_index=True)
        matrix = metrics.get("confusion_matrix")
        if matrix:
            st.markdown("#### Confusion matrix")
            st.caption("Rows = actual labels; columns = predicted labels. Order: Fake, Real.")
            st.dataframe(
                {"Predicted Fake": [matrix[0][0], matrix[1][0]],
                 "Predicted Real": [matrix[0][1], matrix[1][1]]},
                use_container_width=True,
                hide_index=True
            )
    else:
        st.info("Metrics will appear after you train the model.")

    st.markdown("""
    <div class="panel">
      <b>Algorithm:</b> TF-IDF + Logistic Regression<br>
      <b>Evaluation:</b> 80% training / 20% testing with stratified split<br>
      <b>Random state:</b> 42
    </div>
    """, unsafe_allow_html=True)

st.divider()
st.markdown(
    '<p class="small-note">Academic demonstration only · A text classifier is not a fact-checking authority.</p>',
    unsafe_allow_html=True
)
