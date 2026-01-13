import pandas as pd
from pathlib import Path
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler

BASE_DIR = Path(__file__).resolve().parents[1]
DATA_PATH = BASE_DIR / "data" / "processed"

def run_anomaly_model():
    df = pd.read_csv(DATA_PATH / "provider_features.csv")

    # Keep only numeric behavior features
    feature_cols = [
        "TotalClaims",
        "TotalClaimAmount",
        "AvgClaimAmount",
        "MaxClaimAmount",
        "UniquePhysicians"
    ]

    X = df[feature_cols].fillna(0)

    # Scale features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Train Isolation Forest
    iso = IsolationForest(
        n_estimators=200,
        contamination=0.1,
        random_state=42
    )

    df["anomaly_score"] = -iso.fit_predict(X_scaled)
    # Convert from {-1, 1} to continuous score
    df["anomaly_score"] = iso.decision_function(X_scaled) * -1

    out_file = DATA_PATH / "provider_with_anomaly.csv"
    df.to_csv(out_file, index=False)

    print("Anomaly scores generated.")
    print("Saved to:", out_file)
    print(df[["Provider", "anomaly_score"]].head())


if __name__ == "__main__":
    run_anomaly_model()
