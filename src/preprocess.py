
# ==========================================
# NLP PROJECT: PRODUCT REVIEW SENTIMENT ANALYZER
# Data Preprocessing
# ==========================================

import pandas as pd
import re
import nltk
from nltk.corpus import stopwords
from pathlib import Path

# ------------------------------------------
# STEP 1: Set file paths
# ------------------------------------------

PROJECT_FOLDER = Path(__file__).resolve().parent.parent

INPUT_FILE = PROJECT_FOLDER / "dataset" / "Reviews.csv"
OUTPUT_FILE = PROJECT_FOLDER / "dataset" / "processed_reviews.csv"

# ------------------------------------------
# STEP 2: Download English stopwords
# ------------------------------------------

print("Downloading stopwords...")

nltk.download("stopwords", quiet=True)

stop_words = set(stopwords.words("english"))

# Keep important negative words
stop_words.discard("not")
stop_words.discard("no")
stop_words.discard("nor")

# ------------------------------------------
# STEP 3: Convert rating into sentiment
# ------------------------------------------

def get_sentiment(score):

    if score <= 2:
        return "Negative"

    elif score == 3:
        return "Neutral"

    else:
        return "Positive"


# ------------------------------------------
# STEP 4: Clean review text
# ------------------------------------------

def clean_text(text):

    text = str(text)

    # Convert to lowercase
    text = text.lower()

    # Remove numbers and special characters
    text = re.sub(r"[^a-z\s]", " ", text)

    # Split into words
    words = text.split()

    # Remove stopwords
    words = [
        word for word in words
        if word not in stop_words
    ]

    # Join words
    return " ".join(words)


# ------------------------------------------
# STEP 5: Load dataset
# ------------------------------------------

print("\nLoading dataset...")

if not INPUT_FILE.exists():
    raise FileNotFoundError(
        f"Dataset not found: {INPUT_FILE}"
    )

df = pd.read_csv(
    INPUT_FILE,
    usecols=["Text", "Score"]
)

print("Dataset loaded successfully!")
print("Original reviews:", len(df))


# ------------------------------------------
# STEP 6: Remove missing values
# ------------------------------------------

print("\nRemoving missing values...")

df = df.dropna(subset=["Text", "Score"]).copy()

print("Reviews after removing missing values:", len(df))


# ------------------------------------------
# STEP 7: Select balanced reviews
# ------------------------------------------

print("\nSelecting customer reviews...")

df = pd.concat(
    [
        group.sample(
            n=min(len(group), 10000),
            random_state=42
        )
        for _, group in df.groupby("Score")
    ],
    ignore_index=True
)

print("Selected reviews:", len(df))
print("Available columns:", df.columns.tolist())


# ------------------------------------------
# STEP 8: Create sentiment labels
# ------------------------------------------

print("\nCreating sentiment labels...")

df["Sentiment"] = df["Score"].apply(get_sentiment)

print("Sentiment labels created!")


# ------------------------------------------
# STEP 9: Clean review text
# ------------------------------------------

print("\nCleaning review text...")

df["Cleaned_Text"] = df["Text"].apply(clean_text)

print("Text cleaning completed!")


# ------------------------------------------
# STEP 10: Save processed dataset
# ------------------------------------------

print("\nSaving processed dataset...")

df.to_csv(OUTPUT_FILE, index=False)

print("Dataset saved successfully!")


# ------------------------------------------
# STEP 11: Display final results
# ------------------------------------------

print("\n===================================")
print("PREPROCESSING COMPLETED SUCCESSFULLY")
print("===================================")

print("\nTotal processed reviews:", len(df))

print("\nSentiment distribution:")

print(df["Sentiment"].value_counts())

print("\nSample processed reviews:")

print(
    df[["Score", "Sentiment", "Cleaned_Text"]].head()
)

print("\nOutput file:", OUTPUT_FILE)
