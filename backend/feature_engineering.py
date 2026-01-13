import pandas as pd
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parents[1]
RAW_PATH = BASE_DIR / "data" / "raw"
OUT_PATH = BASE_DIR / "data" / "processed"

def build_provider_features():
    ip = pd.read_csv(RAW_PATH / "Train_Inpatientdata.csv")
    op = pd.read_csv(RAW_PATH / "Train_Outpatientdata.csv")
    ben = pd.read_csv(RAW_PATH / "Train_Beneficiarydata.csv")
    labels = pd.read_csv(RAW_PATH / "Train.csv")

    # Combine inpatient & outpatient
    ip["ClaimType"] = "Inpatient"
    op["ClaimType"] = "Outpatient"
    claims = pd.concat([ip, op], ignore_index=True)

    # Convert numeric columns safely
    num_cols = [
        "InscClaimAmtReimbursed",
        "IPAnnualReimbursementAmt",
        "OPAnnualReimbursementAmt"
    ]
    for c in num_cols:
        if c in claims.columns:
            claims[c] = pd.to_numeric(claims[c], errors="coerce").fillna(0)

    # Provider-level aggregation
    agg = claims.groupby("Provider").agg(
        TotalClaims=("ClaimID", "count"),
        TotalClaimAmount=("InscClaimAmtReimbursed", "sum"),
        AvgClaimAmount=("InscClaimAmtReimbursed", "mean"),
        MaxClaimAmount=("InscClaimAmtReimbursed", "max"),
        UniquePhysicians=("AttendingPhysician", "nunique")
    ).reset_index()

    # Merge labels
    data = agg.merge(labels, on="Provider", how="left")
    data["PotentialFraud"] = data["PotentialFraud"].map({"Yes": 1, "No": 0})

    OUT_PATH.mkdir(exist_ok=True)
    out_file = OUT_PATH / "provider_features.csv"
    data.to_csv(out_file, index=False)

    print("Provider features saved to:", out_file)
    print(data.head())

if __name__ == "__main__":
    build_provider_features()
