"""Export model-ready tables for Tableau and Power BI.

    python -m bi.build_bi

Writes bi/data/*.csv and bi/kpis.json (the headline numbers your dashboard should show,
so you can check a Tableau or Power BI build against this repo).

fact_customer has one row per customer with the engineered bands from the live report and
a churn-risk score from the best model in the same comparison. Customers the model was
trained on are flagged (`in_test_set = 0`): their scores are optimistic, so evaluate the
model on test-set customers only and use the scores elsewhere for ranking and targeting.
"""

import json
from pathlib import Path

import pandas as pd

from analysis import data as d
from analysis import model as m

OUT = Path(__file__).resolve().parent


def tables(raw: pd.DataFrame) -> tuple[dict, dict]:
    df = d.add_features(raw)
    df.insert(0, "CustomerId", raw["CustomerId"].values)
    s = m.split(df.drop(columns=["CustomerId"]))
    comparison, fitted = m.compare(s)
    best_name = comparison.loc[0, "model"]
    best_model, _ = fitted[best_name]

    fact = df[[
        "CustomerId", "Geography", "Gender", "Age", "age_band", "Tenure", "CreditScore",
        "Balance", "zero_balance", "NumOfProducts", "HasCrCard", "IsActiveMember",
        "EstimatedSalary", "balance_to_income", "Exited",
    ]].copy()
    fact["churn_risk"] = best_model.predict_proba(df[m.INPUTS])[:, 1].round(4)
    fact["in_test_set"] = fact.index.isin(s.X_test.index).astype(int)
    fact["risk_band"] = pd.cut(fact["churn_risk"], [-0.01, 0.2, 0.4, 0.6, 1.0], labels=["Low", "Medium", "High", "Very high"])

    test = fact[fact["in_test_set"] == 1]
    lift = d.lift_table(test["Exited"], test["churn_risk"])
    info = {"best_model": best_name, "test_roc_auc": float(comparison.loc[0, "test_roc_auc"]), "lift": lift}
    models = comparison[["model", "cv_roc_auc", "test_roc_auc", "test_pr_auc", "test_accuracy"]].round(4)
    return {"fact_customer": fact, "model_comparison": models, "lift_test_set": lift.round(4)}, info


def kpis(fact: pd.DataFrame, info: dict) -> dict:
    lift = info["lift"]
    top2 = float(lift.loc[lift["decile"] == 2, "share_of_all_churners"].iloc[0])
    return {
        "Customers": int(len(fact)),
        "Churned customers": int(fact["Exited"].sum()),
        "Churn rate": round(float(fact["Exited"].mean()), 4),
        "Balance held by churned customers": round(float(fact.loc[fact["Exited"] == 1, "Balance"].sum()), 0),
        "Churn rate, Germany": round(float(fact.loc[fact["Geography"] == "Germany", "Exited"].mean()), 4),
        "Churn rate, inactive members": round(float(fact.loc[fact["IsActiveMember"] == 0, "Exited"].mean()), 4),
        "Best model": info["best_model"],
        "Best model ROC AUC (held-out)": round(info["test_roc_auc"], 3),
        "Share of churners in riskiest 20% (held-out)": round(top2, 3),
    }


def build(out: Path = OUT) -> dict:
    raw = d.load()
    t, info = tables(raw)
    (out / "data").mkdir(parents=True, exist_ok=True)
    for name, frame in t.items():
        frame.to_csv(out / "data" / f"{name}.csv", index=False)
    k = kpis(t["fact_customer"], info)
    (out / "kpis.json").write_text(json.dumps(k, indent=2) + "\n")
    return {"tables": t, "kpis": k}


if __name__ == "__main__":
    res = build()
    for name, frame in res["tables"].items():
        print(f"{name}: {len(frame):,} rows")
    print(json.dumps(res["kpis"], indent=2))
