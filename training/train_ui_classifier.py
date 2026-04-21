"""
train_ui_classifier.py — Train a lightweight UI-element vs content classifier.

Uses features extracted from Rico/Enrico dataset (or synthetic data) to train
a scikit-learn Random Forest that runs in <1ms on CPU at inference time.

Usage:
    python training/download_rico.py   # first, generate features
    python training/train_ui_classifier.py

Output:
    models/ui_classifier.pkl    (serialised sklearn pipeline)
    models/ui_classifier_meta.json  (accuracy metrics)
"""

import csv
import json
import os
import sys

# Add project root to path
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")
MODELS_DIR = os.path.join(PROJECT_ROOT, "models")
FEATURES_CSV = os.path.join(DATA_DIR, "rico_features.csv")
MODEL_PATH = os.path.join(MODELS_DIR, "ui_classifier.pkl")
META_PATH = os.path.join(MODELS_DIR, "ui_classifier_meta.json")

FEATURE_COLUMNS = [
    "x_norm", "y_norm", "w_norm", "h_norm",
    "text_length", "word_count", "is_single_line",
    "has_punctuation", "letter_ratio",
]


def load_data():
    """Load features from CSV."""
    if not os.path.exists(FEATURES_CSV):
        print(f"ERROR: Feature file not found: {FEATURES_CSV}")
        print("Run 'python training/download_rico.py' first.")
        sys.exit(1)

    X = []
    y = []

    with open(FEATURES_CSV, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            features = [float(row[col]) for col in FEATURE_COLUMNS]
            label = int(row["label"])
            X.append(features)
            y.append(label)

    print(f"Loaded {len(X)} samples from {FEATURES_CSV}")
    print(f"  UI elements (1): {sum(1 for v in y if v == 1)}")
    print(f"  Content (0):     {sum(1 for v in y if v == 0)}")
    return X, y


def train():
    """Train and save the classifier."""
    try:
        from sklearn.ensemble import RandomForestClassifier
        from sklearn.model_selection import cross_val_score, train_test_split
        from sklearn.metrics import classification_report, accuracy_score
        from sklearn.preprocessing import StandardScaler
        from sklearn.pipeline import Pipeline
        import joblib
    except ImportError:
        print("ERROR: scikit-learn is required.")
        print("Run: pip install scikit-learn joblib")
        sys.exit(1)

    X, y = load_data()

    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    print(f"\nTraining set: {len(X_train)} samples")
    print(f"Test set:     {len(X_test)} samples")

    # Build pipeline: scaler + random forest
    pipeline = Pipeline([
        ("scaler", StandardScaler()),
        ("clf", RandomForestClassifier(
            n_estimators=100,
            max_depth=12,
            min_samples_leaf=5,
            random_state=42,
            n_jobs=-1,  # use all CPU cores
        )),
    ])

    # Train
    print("\nTraining Random Forest classifier...")
    pipeline.fit(X_train, y_train)

    # Evaluate
    y_pred = pipeline.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)

    print(f"\n{'='*50}")
    print(f"Test Accuracy: {accuracy:.4f} ({accuracy*100:.1f}%)")
    print(f"{'='*50}")
    print("\nClassification Report:")
    print(classification_report(
        y_test, y_pred,
        target_names=["Content", "UI Element"],
    ))

    # Cross-validation
    print("Running 5-fold cross-validation...")
    cv_scores = cross_val_score(pipeline, X, y, cv=5, scoring="accuracy")
    print(f"CV Accuracy: {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")

    # Feature importance
    clf = pipeline.named_steps["clf"]
    importances = list(zip(FEATURE_COLUMNS, clf.feature_importances_))
    importances.sort(key=lambda x: x[1], reverse=True)
    print("\nFeature Importance:")
    for name, imp in importances:
        bar = "#" * int(imp * 50)
        print(f"  {name:18s} {imp:.4f}  {bar}")

    # Save model
    os.makedirs(MODELS_DIR, exist_ok=True)
    joblib.dump(pipeline, MODEL_PATH)
    print(f"\nModel saved to {MODEL_PATH}")

    # Save metadata
    meta = {
        "accuracy": round(accuracy, 4),
        "cv_accuracy_mean": round(cv_scores.mean(), 4),
        "cv_accuracy_std": round(cv_scores.std(), 4),
        "n_train": len(X_train),
        "n_test": len(X_test),
        "feature_columns": FEATURE_COLUMNS,
        "feature_importances": {name: round(imp, 4) for name, imp in importances},
    }
    with open(META_PATH, "w") as f:
        json.dump(meta, f, indent=2)
    print(f"Metadata saved to {META_PATH}")

    return accuracy


if __name__ == "__main__":
    train()
