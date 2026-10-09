"""Load and prepare the bank churn data."""

from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "Bank_Churn.csv"

TARGET = "Exited"
ID_COLUMNS = ["CustomerId", "Surname"]


def load(path: Path = DATA) -> pd.DataFrame:
    """Read the raw CSV and run basic quality checks."""
    df = pd.read_csv(path)
    assert df["CustomerId"].is_unique, "duplicate customers in the data"
    assert df[TARGET].isin([0, 1]).all(), "target must be 0/1"
    return df


def add_features(df: pd.DataFrame) -> pd.DataFrame:
    """Engineered features from the original notebook, plus a few new ones."""
    out = df.drop(columns=[c for c in ID_COLUMNS if c in df.columns]).copy()
    # From the original notebook
    out["balance_to_income"] = out["Balance"] / out["EstimatedSalary"]
    out["income_v_product"] = out["EstimatedSalary"] / out["NumOfProducts"]
    # New: segment-friendly labels
    out["age_band"] = pd.cut(out["Age"], [17, 29, 39, 49, 59, 120], labels=["18–29", "30–39", "40–49", "50–59", "60+"])
    out["zero_balance"] = (out["Balance"] == 0).astype(int)
    out["products"] = out["NumOfProducts"].astype(str)
    return out


def churn_rate(df: pd.DataFrame, by) -> pd.DataFrame:
    """Customers, churners and churn rate for each group."""
    g = df.groupby(by, observed=True)[TARGET].agg(customers="size", churned="sum").reset_index()
    g["churn_rate"] = g["churned"] / g["customers"]
    return g


def lift_table(y_true, proba, bins: int = 10) -> pd.DataFrame:
    """Customers ranked by predicted risk, split into equal-sized groups (deciles)."""
    d = pd.DataFrame({"y": np.asarray(y_true), "p": np.asarray(proba)}).sort_values("p", ascending=False)
    d["decile"] = np.arange(len(d)) * bins // len(d) + 1
    t = d.groupby("decile").agg(customers=("y", "size"), churners=("y", "sum"), avg_risk=("p", "mean")).reset_index()
    t["churn_rate"] = t["churners"] / t["customers"]
    t["share_of_all_churners"] = t["churners"].cumsum() / t["churners"].sum()
    return t
