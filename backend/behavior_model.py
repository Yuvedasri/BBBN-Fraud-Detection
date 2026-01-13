import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Masking
from tensorflow.keras.preprocessing.sequence import pad_sequences

BASE_DIR = Path(__file__).resolve().parents[1]
RAW_PATH = BASE_DIR / "data" / "raw"
PROC_PATH = BASE_DIR / "data" / "processed"

MAX_SEQ_LEN = 50

def build_sequences():
    ip = pd.read_csv(RAW_PATH / "Train_Inpatientdata.csv")
    op = pd.read_csv(RAW_PATH / "Train_Outpatientdata.csv")
    labels = pd.read_csv(RAW_PATH / "Train.csv")

    ip["Date"] = pd.to_datetime(ip["AdmissionDt"], errors="coerce")
    op["Date"] = pd.to_datetime(op["ClaimStartDt"], errors="coerce")

    ip = ip[["Provider", "Date", "InscClaimAmtReimbursed"]]
    op = op[["Provider", "Date", "InscClaimAmtReimbursed"]]

    data = pd.concat([ip, op], ignore_index=True).dropna()
    data = data.sort_values(["Provider", "Date"])

    scaler = MinMaxScaler()
    data["Amt"] = scaler.fit_transform(
        data[["InscClaimAmtReimbursed"]]
    )

    seqs = []
    y = []
    providers = []

    for pid, g in data.groupby("Provider"):
        seq = g["Amt"].values.tolist()
        if len(seq) < 5:
            continue
        seqs.append(seq[:MAX_SEQ_LEN])
        label = labels.loc[labels["Provider"] == pid, "PotentialFraud"]
        if len(label) == 0:
            continue
        y.append(1 if label.values[0] == "Yes" else 0)
        providers.append(pid)

    X = pad_sequences(seqs, maxlen=MAX_SEQ_LEN, padding="post", dtype="float32")
    y = np.array(y)

    return X, y, providers


def run_lstm():
    X, y, providers = build_sequences()

    model = Sequential([
        Masking(mask_value=0.0, input_shape=(MAX_SEQ_LEN, 1)),
        LSTM(32),
        Dense(1, activation="sigmoid")
    ])

    model.compile(
        optimizer="adam",
        loss="binary_crossentropy",
        metrics=["accuracy"]
    )

    model.fit(X[..., None], y, epochs=5, batch_size=32, verbose=1)

    probs = model.predict(X[..., None]).flatten()

    df = pd.DataFrame({
        "Provider": providers,
        "behavior_risk": probs
    })

    base = pd.read_csv(PROC_PATH / "provider_with_anomaly.csv")
    merged = base.merge(df, on="Provider", how="left").fillna(0)

    out_file = PROC_PATH / "provider_with_behavior.csv"
    merged.to_csv(out_file, index=False)

    print("Behavior risks generated.")
    print("Saved to:", out_file)
    print(merged[["Provider", "behavior_risk"]].head())


if __name__ == "__main__":
    run_lstm()
