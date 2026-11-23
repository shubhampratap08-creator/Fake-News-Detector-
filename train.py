"""Train and compare classical ML models for fake news detection.

Expects data/Fake.csv and data/True.csv (Kaggle: "Fake and Real News Dataset", ISOT).
Run:  python train.py
"""
import re
import joblib
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression, PassiveAggressiveClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                             f1_score, confusion_matrix, classification_report)

for pkg in ("stopwords", "wordnet", "omw-1.4"):
    nltk.download(pkg, quiet=True)

STOP = set(stopwords.words("english"))
LEMMA = WordNetLemmatizer()


def clean_text(text: str) -> str:
    text = str(text).lower()
    # remove agency tags like "WASHINGTON (Reuters) -" that leak the label
    text = re.sub(r"^.{0,60}\(reuters\)\s*-?\s*", "", text)
    text = re.sub(r"http\S+|www\S+", " ", text)
    text = re.sub(r"[^a-z\s]", " ", text)
    words = [LEMMA.lemmatize(w) for w in text.split() if w not in STOP and len(w) > 2]
    return " ".join(words)


def load_data() -> pd.DataFrame:
    fake = pd.read_csv("data/Fake.csv")
    real = pd.read_csv("data/True.csv")
    fake["label"] = 0  # 0 = fake
    real["label"] = 1  # 1 = real
    df = pd.concat([fake, real], ignore_index=True)
    df["content"] = df["title"].fillna("") + " " + df["text"].fillna("")
    return df.sample(frac=1, random_state=42).reset_index(drop=True)


def main():
    df = load_data()
    print("Dataset shape:", df.shape)
    print(df["label"].value_counts().rename({0: "fake", 1: "real"}))

    df["clean"] = df["content"].apply(clean_text)
    X_train, X_test, y_train, y_test = train_test_split(
        df["clean"], df["label"], test_size=0.2, random_state=42, stratify=df["label"])

    vectorizer = TfidfVectorizer(max_features=50000, ngram_range=(1, 2), min_df=3)
    Xtr = vectorizer.fit_transform(X_train)
    Xte = vectorizer.transform(X_test)

    models = {
        "Logistic Regression": LogisticRegression(max_iter=1000),
        "Naive Bayes": MultinomialNB(),
        "Passive Aggressive": PassiveAggressiveClassifier(max_iter=1000, random_state=42),
        "Random Forest": RandomForestClassifier(n_estimators=100, n_jobs=-1, random_state=42),
    }

    rows, fitted = [], {}
    for name, model in models.items():
        model.fit(Xtr, y_train)
        pred = model.predict(Xte)
        rows.append({
            "Model": name,
            "Accuracy": accuracy_score(y_test, pred),
            "Precision": precision_score(y_test, pred),
            "Recall": recall_score(y_test, pred),
            "F1": f1_score(y_test, pred),
        })
        fitted[name] = (model, pred)
        print(f"\n=== {name} ===")
        print(classification_report(y_test, pred, target_names=["Fake", "Real"]))

    results = pd.DataFrame(rows).sort_values("F1", ascending=False).round(4)
    print("\nModel comparison:\n", results.to_string(index=False))
    results.to_csv("outputs/model_comparison.csv", index=False)

    best_name = results.iloc[0]["Model"]
    best_model, best_pred = fitted[best_name]
    print("\nBest model:", best_name)

    cm = confusion_matrix(y_test, best_pred)
    plt.figure(figsize=(5, 4))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=["Fake", "Real"], yticklabels=["Fake", "Real"])
    plt.title(f"Confusion Matrix - {best_name}")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.tight_layout()
    plt.savefig("outputs/confusion_matrix.png", dpi=150)

    joblib.dump(best_model, "models/model.joblib")
    joblib.dump(vectorizer, "models/vectorizer.joblib")
    print("Saved model and vectorizer to models/")


if __name__ == "__main__":
    main()
