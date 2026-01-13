import pandas as pd
from pathlib import Path
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, roc_auc_score

BASE_DIR = Path(__file__).resolve().parents[1]
DATA_PATH = BASE_DIR / "data" / "processed"

def run_fusion():
    df = pd.read_csv(DATA_PATH / "provider_with_behavior.csv")

    feature_cols = [
        "TotalClaims",
        "TotalClaimAmount",
        "AvgClaimAmount",
        "MaxClaimAmount",
        "UniquePhysicians",
        "anomaly_score",
        "behavior_risk"
    ]

    X = df[feature_cols].fillna(0)
    y = df["PotentialFraud"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    model = XGBClassifier(
        n_estimators=300,
        max_depth=5,
        learning_rate=0.05,
        subsample=0.9,
        colsample_bytree=0.9,
        eval_metric="logloss",
        random_state=42
    )

    model.fit(X_train, y_train)

    probs = model.predict_proba(X)[:, 1]
    df["fraud_probability"] = probs

    y_pred = model.predict(X_test)
    print(classification_report(y_test, y_pred))
    print("ROC-AUC:", roc_auc_score(y_test, model.predict_proba(X_test)[:, 1]))

    out_file = DATA_PATH / "provider_with_fraud_prob.csv"
    df.to_csv(out_file, index=False)

    print("Fusion complete.")
    print("Saved to:", out_file)
    print(df[["Provider", "fraud_probability"]].head())


if __name__ == "__main__":
    run_fusion()
