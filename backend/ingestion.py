import pandas as pd
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parents[1]
RAW_PATH = BASE_DIR / "data" / "raw"

def load_datasets():
    print("RAW PATH:", RAW_PATH)
    print("Files found:", [p.name for p in RAW_PATH.iterdir()])

    inpatient = pd.read_csv(RAW_PATH / "Train_Inpatientdata.csv")
    outpatient = pd.read_csv(RAW_PATH / "Train_Outpatientdata.csv")
    beneficiary = pd.read_csv(RAW_PATH / "Train_Beneficiarydata.csv")
    labels = pd.read_csv(RAW_PATH / "Train.csv")

    return inpatient, outpatient, beneficiary, labels


if __name__ == "__main__":
    ip, op, ben, y = load_datasets()

    print("Inpatient:", ip.shape)
    print("Outpatient:", op.shape)
    print("Beneficiary:", ben.shape)
    print("Labels:", y.shape)
