# train.py
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report
from joblib import dump
from utils import basic_clean

def load_csv(path):
    df = pd.read_csv(path)
    df = df.dropna(subset=["text","label"])
    df["text"] = df["text"].astype(str).map(basic_clean)
    df["label"] = df["label"].map(lambda x: 1 if str(x).lower().startswith("phish") else 0)
    return df

if __name__ == "__main__":
    df = load_csv("sample_data/train.csv")
    X_train, X_val, y_train, y_val = train_test_split(
        df["text"], df["label"], test_size=0.2, random_state=42, stratify=df["label"]
    )

    clf = Pipeline([
        ("tfidf", TfidfVectorizer(
            ngram_range=(1,2),
            min_df=2,
            max_features=50000,
            sublinear_tf=True
        )),
        ("lr", LogisticRegression(max_iter=200, n_jobs=None))
    ])

    clf.fit(X_train, y_train)
    preds = clf.predict(X_val)
    print(classification_report(y_val, preds, target_names=["safe","phish"]))

    dump(clf, "model.joblib")
    print("Saved model.joblib")
