
# ==========================================
# SMART PRODUCT REVIEW SENTIMENT ANALYZER
# Streamlit Web Application
# ==========================================

import streamlit as st
import pandas as pd
import joblib
import re
import nltk

from nltk.corpus import stopwords
from pathlib import Path

# ------------------------------------------
# PAGE CONFIGURATION
# ------------------------------------------

st.set_page_config(
    page_title="Product Review Sentiment Analyzer",
    page_icon="📊",
    layout="wide"
)

# ------------------------------------------
# PROJECT PATHS
# ------------------------------------------

PROJECT_FOLDER = Path(__file__).resolve().parent

MODEL_PATH = PROJECT_FOLDER / "models" / "sentiment_model.pkl"
VECTORIZER_PATH = PROJECT_FOLDER / "models" / "tfidf_vectorizer.pkl"

# ------------------------------------------
# LOAD MODEL AND VECTORIZER
# ------------------------------------------

@st.cache_resource
def load_model():

    model = joblib.load(MODEL_PATH)
    vectorizer = joblib.load(VECTORIZER_PATH)

    return model, vectorizer

model, vectorizer = load_model()

# ------------------------------------------
# PREPARE STOPWORDS
# ------------------------------------------

@st.cache_resource
def load_stopwords():

    nltk.download("stopwords", quiet=True)

    words = set(stopwords.words("english"))

    words.discard("not")
    words.discard("no")
    words.discard("nor")

    return words

stop_words = load_stopwords()

# ------------------------------------------
# CLEAN REVIEW TEXT
# ------------------------------------------

def clean_text(text):

    text = str(text).lower()

    text = re.sub(r"[^a-z\s]", " ", text)

    words = text.split()

    words = [
        word for word in words
        if word not in stop_words
    ]

    return " ".join(words)

# ------------------------------------------
# APPLICATION HEADER
# ------------------------------------------

st.title("📊 Smart Product Review Sentiment Analyzer")

st.write(
    "Analyze customer reviews using Natural Language Processing "
    "and Machine Learning."
)

st.divider()

# ------------------------------------------
# NAVIGATION
# ------------------------------------------

tab1, tab2, tab3 = st.tabs([
    "Single Review Analysis",
    "Bulk Review Analysis",
    "About Project"
])

# ==========================================
# TAB 1: SINGLE REVIEW ANALYSIS
# ==========================================

with tab1:

    st.subheader("Analyze a Customer Review")

    review = st.text_area(
        "Enter your product review:",
        placeholder="Example: This product is amazing...",
        height=150
    )

    if st.button("Analyze Sentiment", type="primary"):

        if review.strip() == "":
            st.warning("Please enter a review.")

        else:

            cleaned_review = clean_text(review)

            if not cleaned_review.strip():
                st.warning("Please enter a review containing English words.")

            else:
                review_vector = vectorizer.transform(
                    [cleaned_review]
                )

                prediction = model.predict(review_vector)[0]

                probabilities = model.predict_proba(
                    review_vector
                )[0]

                st.subheader("Prediction Result")

                if prediction == "Positive":
                    st.success("😊 Positive Review")

                elif prediction == "Negative":
                    st.error("😞 Negative Review")

                else:
                    st.info("😐 Neutral Review")

                st.write("**Sentiment confidence:**")

                probability_df = pd.DataFrame({
                    "Sentiment": model.classes_,
                    "Confidence (%)": probabilities * 100
                })

                st.bar_chart(
                    probability_df.set_index("Sentiment")
                )

                st.write(
                    "Predicted sentiment:",
                    prediction
                )

# ==========================================
# TAB 2: BULK REVIEW ANALYSIS
# ==========================================

with tab2:

    st.subheader("Analyze Multiple Reviews")

    st.write(
        "Upload a CSV file containing a column named Text."
    )

    uploaded_file = st.file_uploader(
        "Upload CSV file",
        type=["csv"]
    )

    if uploaded_file is not None:

        batch_df = pd.read_csv(uploaded_file)

        if "Text" not in batch_df.columns:

            st.error(
                "Your CSV file must contain a column named Text."
            )

        else:

            if st.button("Analyze All Reviews", type="primary"):

                batch_df = batch_df.dropna(
                    subset=["Text"]
                ).copy()

                if batch_df.empty:
                    st.warning("No valid reviews found.")

                else:

                    with st.spinner("Analyzing reviews..."):

                        batch_df["Cleaned_Text"] = (
                            batch_df["Text"].apply(clean_text)
                        )

                        batch_df["Sentiment"] = "Unknown"

                        valid_mask = (
                            batch_df["Cleaned_Text"].str.strip() != ""
                        )

                        if valid_mask.any():

                            review_vectors = vectorizer.transform(
                                batch_df.loc[
                                    valid_mask, "Cleaned_Text"
                                ]
                            )

                            batch_df.loc[
                                valid_mask, "Sentiment"
                            ] = model.predict(review_vectors)

                        st.success("Analysis completed!")

                        st.subheader("Analysis Results")

                        st.dataframe(
                            batch_df[["Text", "Sentiment"]],
                            use_container_width=True
                        )

                        st.subheader("Sentiment Distribution")

                        counts = batch_df[
                            batch_df["Sentiment"] != "Unknown"
                        ]["Sentiment"].value_counts()

                        st.bar_chart(counts)

                        csv_data = batch_df.to_csv(
                            index=False
                        ).encode("utf-8")

                        st.download_button(
                            label="Download Results CSV",
                            data=csv_data,
                            file_name="sentiment_results.csv",
                            mime="text/csv"
                        )

# ==========================================
# TAB 3: ABOUT PROJECT
# ==========================================

with tab3:

    st.subheader("About This Project")

    st.write("""
    This project analyzes Amazon product reviews
    and classifies them into three sentiment categories:

    - Positive
    - Negative
    - Neutral

    **Technologies used:**

    - Python
    - Natural Language Processing
    - NLTK
    - Pandas
    - Scikit-learn
    - TF-IDF
    - Logistic Regression
    - Streamlit

    **Dataset:** Amazon Fine Food Reviews

    **Training reviews:** 40,000

    **Testing reviews:** 10,000

    **Model accuracy:** 75.41%
    """)

    st.info(
        "The model was trained using TF-IDF "
        "and Logistic Regression."
    )

# ------------------------------------------
# FOOTER
# ------------------------------------------

st.divider()

st.caption(
    "Smart Product Review Sentiment Analyzer | NLP Project"
)
