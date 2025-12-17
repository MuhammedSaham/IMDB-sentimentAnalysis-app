import streamlit as st
import joblib
import re

best_model = joblib.load("imdb_sentiment_model.pkl")
vectorizer = joblib.load("vectorizer.pkl")

def clean_text(text):
    text = str(text).lower()
    text = re.sub(r"<.*?>", " ", text)
    text = re.sub(r"[^a-z0-9\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text

def predict_sentiment(text):
    text_clean = clean_text(text)
    vec = vectorizer.transform([text_clean])
    pred = best_model.predict(vec)[0]
    return "Positive" if pred == 1 else "Negative"

st.set_page_config(
    page_title="IMDB Sentiment Analyzer",
    page_icon="🎬",
    layout="centered"
)

with st.sidebar:
    st.header("ℹ️ About")
    st.write(
        """
        This app uses:
        - TF-IDF Vectorizer
        - LinearSVC Model
        - Trained on IMDB reviews
        """
    )
    st.markdown("---")
    st.write("👨‍💻 Built with Streamlit")


st.title("🎬 IMDB Movie Review Sentiment Analyzer")
st.caption("Analyze movie reviews instantly using Machine Learning")


if "review" not in st.session_state:
    st.session_state.review = ""


review = st.text_area(
    "📝 Enter your movie review:",
    height=200,
    key="review"
)

predict_btn = st.button("🔍 Predict Sentiment")

if predict_btn:
    if review.strip() == "":
        st.warning("⚠️ Please enter a review to analyze.")
    else:
        with st.spinner("Analyzing sentiment..."):
            sentiment = predict_sentiment(review)

        st.markdown("---")
        if sentiment == "Positive":
            st.success(f"✅ **Sentiment:** {sentiment}")
        else:
            st.error(f"❌ **Sentiment:** {sentiment}")

st.markdown(
    """
    ---
    **Disclaimer:** This model is for educational purposes only.
    Predictions may not always be accurate.
    """
)
