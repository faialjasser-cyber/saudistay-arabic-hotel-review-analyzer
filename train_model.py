"""Train a Saudi hotel-review sentiment baseline."""
from __future__ import annotations
import argparse
import re
from pathlib import Path

import joblib
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.pipeline import FeatureUnion, Pipeline

ROOT = Path(__file__).resolve().parent
MODEL_PATH = ROOT / "models" / "saudi_hotel_sentiment.joblib"
SAUDI_CITIES = [
    "الرياض", "riyadh", "جدة", "jeddah", "مكة", "مكه", "makkah", "mecca",
    "المدينة المنورة", "المدينه المنوره", "madinah", "الدمام",
    "dammam", "الخبر", "khobar", "الظهران", "dhahran", "الطائف", "taif",
    "أبها", "ابها", "abha", "تبوك", "tabuk", "ينبع", "yanbu", "جازان",
    "jazan", "حائل", "hail", "بريدة", "buraidah", "الأحساء", "الاحساء",
    "الهفوف", "hafuf", "نجران", "najran",
]
ALIASES = {
    "review": ["review", "review_text", "text", "التعليق", "المراجعة"],
    "location": ["location", "city", "hotel_city", "hotel_location", "country", "المدينة", "الموقع"],
    "sentiment": ["polarity_score", "sentiment", "label", "المشاعر", "التصنيف"],
    "rating": ["user_rating", "rating", "review_rating", "تقييم", "التقييم"],
}

def find_column(df, key):
    normalized = {str(c).strip().lower(): c for c in df.columns}
    return next((normalized[a.lower()] for a in ALIASES[key] if a.lower() in normalized), None)

def load_table(path):
    if path.suffix.lower() in {".xlsx", ".xls"}:
        return pd.read_excel(path)
    if path.suffix.lower() == ".csv":
        return pd.read_csv(path, encoding="utf-8-sig")
    raise ValueError("الملف لازم يكون CSV أو Excel.")

def map_label(value):
    if pd.isna(value):
        return None
    labels = {
        "-1": "سلبي", "-1.0": "سلبي", "negative": "سلبي", "neg": "سلبي",
        "0": "محايد", "0.0": "محايد", "neutral": "محايد", "neu": "محايد",
        "1": "إيجابي", "1.0": "إيجابي", "positive": "إيجابي", "pos": "إيجابي",
        "سلبي": "سلبي", "محايد": "محايد", "ايجابي": "إيجابي", "إيجابي": "إيجابي",
    }
    return labels.get(str(value).strip().lower())

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default=str(ROOT / "data" / "balanced_reviews.xlsx"))
    args = parser.parse_args()
    path = Path(args.data)
    if not path.exists():
        raise FileNotFoundError(f"لم أجد الملف: {path}")

    df = load_table(path)
    review_col, location_col = find_column(df, "review"), find_column(df, "location")
    label_col, rating_col = find_column(df, "sentiment"), find_column(df, "rating")
    if not review_col or not location_col:
        raise ValueError(f"لم أتعرف على عمودي المراجعة والموقع. الأعمدة: {list(df.columns)}")

    # Determine country from the hotel location, never the reviewer's nationality.
    pattern = "|".join(re.escape(city) for city in SAUDI_CITIES)
    mask = df[location_col].fillna("").astype(str).str.contains(pattern, case=False, regex=True)
    saudi = df.loc[mask].copy()
    saudi[review_col] = saudi[review_col].fillna("").astype(str).str.strip()
    saudi = saudi[saudi[review_col].str.len() >= 3]
    if label_col:
        saudi["target"] = saudi[label_col].map(map_label)
    elif rating_col:
        rating = pd.to_numeric(saudi[rating_col], errors="coerce")
        saudi["target"] = rating.map(
            lambda x: "سلبي" if x <= 4 else ("محايد" if x <= 6 else "إيجابي")
            if pd.notna(x) else None
        )
    else:
        raise ValueError("يلزم عمود polarity_score/sentiment أو عمود تقييم رقمي.")
    saudi = saudi.dropna(subset=["target"])
    if saudi.empty or saudi["target"].nunique() < 2:
        raise ValueError("لم أجد بيانات سعودية كافية بعد التصفية. راجعي عمود الموقع.")

    print(f"صفوف الملف: {len(df):,}")
    print(f"مراجعات مواقع سعودية: {len(saudi):,}")
    print("المواقع:", saudi[location_col].value_counts().head(20).to_dict())
    print("الفئات:", saudi["target"].value_counts().to_dict())

    X_train, X_test, y_train, y_test = train_test_split(
        saudi[review_col], saudi["target"], test_size=0.2,
        random_state=42, stratify=saudi["target"]
    )
    features = FeatureUnion([
        ("word", TfidfVectorizer(analyzer="word", ngram_range=(1, 2), min_df=2,
                                 max_features=100_000, sublinear_tf=True)),
        ("char", TfidfVectorizer(analyzer="char_wb", ngram_range=(2, 5), min_df=2,
                                 max_features=100_000, sublinear_tf=True)),
    ])
    model = Pipeline([
        ("features", features),
        ("classifier", LogisticRegression(max_iter=1200, class_weight="balanced")),
    ])
    model.fit(X_train, y_train)
    pred = model.predict(X_test)
    print(f"Accuracy: {accuracy_score(y_test, pred):.3f}")
    print(classification_report(y_test, pred, zero_division=0))
    labels = sorted(saudi["target"].unique())
    print("Confusion matrix labels:", labels)
    print(confusion_matrix(y_test, pred, labels=labels))

    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump({"pipeline": model, "labels": labels, "trained_rows": len(saudi)}, MODEL_PATH)
    print(f"تم حفظ النموذج: {MODEL_PATH}")

if __name__ == "__main__":
    main()
