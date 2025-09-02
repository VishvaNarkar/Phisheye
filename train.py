import argparse
from pathlib import Path
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score, roc_auc_score
import joblib
import sys
import warnings
import ast
import re

warnings.filterwarnings("ignore", category=UserWarning)

def find_csv_files(base: Path):
    # prefer files that contain "vectoriz" (vectorized datasets from Figshare)
    all_csvs = sorted([p for p in base.rglob("*.csv") if "Zone.Identifier" not in p.name])
    vect = [p for p in all_csvs if "vectoriz" in p.name.lower() or "vectorized" in p.name.lower()]
    return vect if vect else all_csvs

def safe_read_csv(path: Path):
    # try utf-8 then latin1, skip bad lines, set low_memory False to avoid tokenization issues
    for enc in ("utf-8", "latin1", "iso-8859-1"):
        try:
            return pd.read_csv(path, encoding=enc, on_bad_lines="skip", low_memory=False)
        except Exception:
            continue
    raise IOError(f"Unable to read {path} with common encodings")

def detect_label_column(df: pd.DataFrame):
    candidates = ["status", "label", "class", "is_phish", "is_phishing", "phishing", "phish", "is_spam"]
    for c in candidates:
        if c in df.columns:
            return c
    for c in df.columns:
        if df[c].dtype == "object" and df[c].nunique() <= 5:
            return c
    for c in df.columns:
        if pd.api.types.is_numeric_dtype(df[c]) and set(df[c].dropna().unique()).issubset({0,1}):
            return c
    return None

def normalize_label_series(s: pd.Series):
    if pd.api.types.is_numeric_dtype(s):
        return s.astype(int)
    s_lower = s.astype(str).str.lower().str.strip()
    mapped = s_lower.map(lambda v: 1 if ("phish" in v or "malicious" in v or v in {"1","true","yes","spam"}) else (0 if ("legit" in v or "legitimate" in v or v in {"0","false","no","ham","not spam"}) else np.nan))
    return mapped

def parse_vector_value(val):
    if pd.isna(val):
        return None
    if isinstance(val, (list, tuple, np.ndarray)):
        return np.asarray(val, dtype=float)
    s = str(val).strip()
    # try literal eval for JSON-like / Python list
    try:
        obj = ast.literal_eval(s)
        if isinstance(obj, (list, tuple)) and all(isinstance(x, (int, float, np.number)) or (isinstance(x, str) and re.match(r"^-?\d+(\.\d+)?$", x.strip())) for x in obj):
            return np.asarray(obj, dtype=float)
    except Exception:
        pass
    # remove surrounding brackets
    s2 = re.sub(r'^[\[\(\{]+|[\]\)\}]+$', '', s)
    # split on commas or whitespace
    if ',' in s2:
        tokens = [t.strip() for t in s2.split(',') if t.strip() != ""]
    else:
        tokens = [t.strip() for t in s2.split() if t.strip() != ""]
    # if tokens look numeric, convert
    if not tokens:
        return None
    numeric_tokens = []
    for t in tokens:
        # allow scientific notation
        if re.match(r'^-?\d+(\.\d+)?([eE]-?\d+)?$', t):
            numeric_tokens.append(float(t))
        else:
            # not numeric
            return None
    return np.asarray(numeric_tokens, dtype=float)

def expand_vector_columns(df: pd.DataFrame, label_col: str, max_samples=50):
    # find candidate object columns (not label)
    candidates = [c for c in df.columns if c != label_col and df[c].dtype == "object"]
    for col in candidates:
        # sample some non-null values
        sample = df[col].dropna().head(max_samples)
        if sample.empty:
            continue
        parsed = [parse_vector_value(v) for v in sample]
        # require majority parse success and consistent lengths
        parsed_valid = [p for p in parsed if isinstance(p, np.ndarray)]
        if len(parsed_valid) < max(3, len(sample) // 2):
            continue
        lengths = [p.shape[0] for p in parsed_valid]
        if len(set(lengths)) != 1:
            continue
        vec_len = lengths[0]
        # expand full column
        arr = np.zeros((len(df), vec_len), dtype=float)
        for i, v in enumerate(df[col].values):
            pv = parse_vector_value(v)
            if isinstance(pv, np.ndarray) and pv.shape[0] == vec_len:
                arr[i] = pv
            else:
                # leave zeros for missing/unparseable
                pass
        # attach as numeric columns
        new_cols = [f"{col}_f{i}" for i in range(vec_len)]
        arr_df = pd.DataFrame(arr, columns=new_cols, index=df.index)
        df = pd.concat([df.drop(columns=[col]), arr_df], axis=1)
        # after expanding one vector column return (one is usually enough)
        return df
    return df

def load_and_label(path: Path):
    df = safe_read_csv(path)
    label_col = detect_label_column(df)
    if label_col is None:
        raise ValueError(f"Could not detect label column in {path.name}")
    df[label_col] = normalize_label_series(df[label_col])
    df = df[df[label_col].notna()].copy()
    return df, label_col

def prepare_features(df: pd.DataFrame, label_col: str):
    # first try expanding any vector-like object columns into numeric columns
    df = expand_vector_columns(df, label_col)

    non_feature_cols = [c for c in df.columns if c.lower() in {"url","raw","email","message","body","subject","id"}]
    feats = [c for c in df.columns if c != label_col and c not in non_feature_cols]
    numeric_feats = [c for c in feats if pd.api.types.is_numeric_dtype(df[c]) or pd.api.types.is_bool_dtype(df[c])]

    # if still no numeric features, try to coerce remaining object columns to numeric (some CSVs may have numbers stored as strings)
    if not numeric_feats:
        coerced = []
        for c in feats:
            try:
                coerced_col = pd.to_numeric(df[c], errors="coerce")
                if coerced_col.notna().sum() > 0:
                    df[c] = coerced_col
                    coerced.append(c)
            except Exception:
                continue
        numeric_feats = [c for c in feats if pd.api.types.is_numeric_dtype(df[c]) or pd.api.types.is_bool_dtype(df[c])]

    X = df[numeric_feats].copy()
    X[numeric_feats] = X[numeric_feats].fillna(0)
    y = df[label_col].astype(int).copy()
    return X, y

def main(input_dir: str, out_model: str):
    base = Path(input_dir)
    if not base.exists():
        print(f"Input directory not found: {input_dir}", file=sys.stderr)
        return 1

    files = find_csv_files(base)
    if not files:
        print(f"No CSV files found in {input_dir}", file=sys.stderr)
        return 1

    # dedupe file list by dataset stem (avoid raw + vectorized duplicates)
    seen_stems = set()
    chosen_files = []
    for p in files:
        # normalize stem removing common suffixes
        stem = p.stem.lower()
        for suf in ["_vectorized_data", "_vectorized", "vectorized_data", "vectorized"]:
            if stem.endswith(suf):
                stem = stem[: -len(suf)]
        if stem in seen_stems:
            continue
        seen_stems.add(stem)
        chosen_files.append(p)

    frames = []
    label_col_global = None
    for f in chosen_files:
        try:
            df, label_col = load_and_label(f)
            if label_col_global is None:
                label_col_global = label_col
            elif label_col != label_col_global:
                df = df.rename(columns={label_col: label_col_global})
            frames.append(df)
            print(f"Loaded {len(df)} rows from {f.name}")
        except Exception as e:
            print(f"Skipping {f.name}: {e}")

    if not frames:
        print("No usable data loaded.", file=sys.stderr)
        return 1

    combined = pd.concat(frames, ignore_index=True, sort=False)
    combined_before = len(combined)
    combined = combined.drop_duplicates().reset_index(drop=True)
    print(f"Combined dataset rows: {combined_before} -> after drop_duplicates: {len(combined)}, columns: {len(combined.columns)}")

    X, y = prepare_features(combined, label_col_global)

    if X.empty:
        print("No numeric features found after filtering.", file=sys.stderr)
        return 1

    print("Label distribution:\n", y.value_counts().to_string())

    if y.nunique() < 2:
        print("Only one label present in data; cannot train.", file=sys.stderr)
        return 1

    # split with stratify where possible
    try:
        strat = y if min(y.value_counts()) > 1 else None
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=strat)
    except Exception:
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # use class_weight to help with imbalance
    model = RandomForestClassifier(n_estimators=200, random_state=42, n_jobs=-1, class_weight="balanced")
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    print("Accuracy:", accuracy_score(y_test, y_pred))
    try:
        y_prob = model.predict_proba(X_test)[:, 1]
        auc = roc_auc_score(y_test, y_prob)
        print("ROC AUC:", round(auc, 4))
    except Exception:
        pass

    print("\nClassification Report:\n", classification_report(y_test, y_pred, zero_division=0))

    joblib.dump(model, out_model)
    print(f"✅ Model saved as {out_model}")
    return 0

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train phishing RF from multiple CSVs (prefers vectorized files)")
    parser.add_argument("--input-dir", "-i", default="sample_data", help="Directory to search for CSVs (default: sample_data)")
    parser.add_argument("--out", "-o", default="phishing_rf_model.pkl", help="Output model file (default: phishing_rf_model.pkl)")
    args = parser.parse_args()
    sys.exit(main(args.input_dir, args.out))