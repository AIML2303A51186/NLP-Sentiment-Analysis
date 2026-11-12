
# ==========================================
# NLP PROJECT: PRODUCT REVIEW SENTIMENT ANALYZER
# Model Training using TF-IDF and Logistic Regression
# ==========================================

import pandas as pd
import joblib
import matplotlib.pyplot as plt

from pathlib import Path

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay
)

# ------------------------------------------
# STEP 1: Set project paths
# ------------------------------------------

PROJECT_FOLDER = Path(__file__).resolve().parent.parent

DATA_FILE = PROJECT_FOLDER / "dataset" / "processed_reviews.csv"
MODEL_FOLDER = PROJECT_FOLDER / "models"

MODEL_FOLDER.mkdir(parents=True, exist_ok=True)

# ------------------------------------------
# STEP 2: Load processed dataset
# ------------------------------------------

print("Loading processed dataset...")

df = pd.read_csv(DATA_FILE)

X = df["Cleaned_Text"]
y = df["Sentiment"]

print("Dataset loaded successfully!")
print("Total reviews:", len(df))

# ------------------------------------------
# STEP 3: Split dataset
# ------------------------------------------

print("\nSplitting dataset into training and testing...")

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

print("Training reviews:", len(X_train))
print("Testing reviews:", len(X_test))

# ------------------------------------------
# STEP 4: Convert text into numerical features
# ------------------------------------------

print("\nApplying TF-IDF...")

vectorizer = TfidfVectorizer(
    max_features=30000,
    ngram_range=(1, 2),
    min_df=2,
    sublinear_tf=True
)

X_train_tfidf = vectorizer.fit_transform(X_train)
X_test_tfidf = vectorizer.transform(X_test)

print("TF-IDF conversion completed!")
print("Number of features:", X_train_tfidf.shape[1])

# ------------------------------------------
# STEP 5: Train Logistic Regression model
# ------------------------------------------

print("\nTraining Logistic Regression model...")

model = LogisticRegression(
    max_iter=1000,
    random_state=42
)

model.fit(X_train_tfidf, y_train)

print("Model training completed!")

# ------------------------------------------
# STEP 6: Test the model
# ------------------------------------------

print("\nTesting the model...")

y_pred = model.predict(X_test_tfidf)

accuracy = accuracy_score(y_test, y_pred)

print("\nModel Accuracy:", round(accuracy * 100, 2), "%")

print("\nClassification Report:")

print(classification_report(y_test, y_pred))

# ------------------------------------------
# STEP 7: Display confusion matrix
# ------------------------------------------

print("\nGenerating confusion matrix...")

cm = confusion_matrix(
    y_test,
    y_pred,
    labels=["Negative", "Neutral", "Positive"]
)

display = ConfusionMatrixDisplay(
    confusion_matrix=cm,
    display_labels=["Negative", "Neutral", "Positive"]
)

display.plot(cmap="Blues")

plt.title("Sentiment Analysis Confusion Matrix")
plt.tight_layout()

plt.savefig(
    PROJECT_FOLDER / "confusion_matrix.png"
)

plt.show()

# ------------------------------------------
# STEP 8: Save trained model and vectorizer
# ------------------------------------------

print("\nSaving trained model...")

joblib.dump(
    model,
    MODEL_FOLDER / "sentiment_model.pkl"
)

joblib.dump(
    vectorizer,
    MODEL_FOLDER / "tfidf_vectorizer.pkl"
)

print("Model saved successfully!")

# ------------------------------------------
# FINAL RESULT
# ------------------------------------------

print("\n===================================")
print("MODEL TRAINING COMPLETED SUCCESSFULLY")
print("===================================")

print("Accuracy:", round(accuracy * 100, 2), "%")
print("Model location:", MODEL_FOLDER)
print("Confusion matrix: confusion_matrix.png")
