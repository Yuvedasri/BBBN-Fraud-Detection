import pandas as pd
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parents[1]
DATA_PATH = BASE_DIR / "data" / "processed"

def apply_agent_rules():
    df = pd.read_csv(DATA_PATH / "provider_with_fraud_prob.csv")

    def decide(row):
        p = row["fraud_probability"]
        net = row.get("behavior_risk", 0)

        if p > 0.85 and net > 0.5:
            return "BLOCK"
        elif 0.60 <= p <= 0.85:
            return "FLAG"
        else:
            return "APPROVE"

    df["Decision"] = df.apply(decide, axis=1)

    out_file = DATA_PATH / "provider_with_decision.csv"
    df.to_csv(out_file, index=False)

    print("Decisions applied.")
    print("Saved to:", out_file)
    print(df[["Provider", "fraud_probability", "Decision"]].head())


if __name__ == "__main__":
    apply_agent_rules()
