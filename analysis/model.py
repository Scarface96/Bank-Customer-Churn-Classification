"""Train, compare and explain churn models."""

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, roc_auc_score, roc_curve
from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from .data import TARGET

SEED = 42

# Raw inputs every model sees (the same fields a bank would have for a customer).
NUMERIC = ["CreditScore", "Age", "Tenure", "Balance", "EstimatedSalary"]
FLAGS = ["HasCrCard", "IsActiveMember"]
CATEGORICAL = ["Geography", "Gender"]
INPUTS = NUMERIC + FLAGS + CATEGORICAL + ["NumOfProducts"]

PRETTY = {
    "CreditScore": "Credit score",
    "Age": "Age",
    "Tenure": "Tenure (years)",
    "Balance": "Balance",
    "EstimatedSalary": "Salary",
    "HasCrCard": "Has credit card",
    "IsActiveMember": "Active member",
    "Geography": "Country",
    "Gender": "Gender",
    "NumOfProducts": "Number of products",
}


class AgeCurve(BaseEstimator, TransformerMixin):
    """Adds (age - 45)^2 / 100 so a linear model can capture risk that peaks in middle age."""

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        a = np.asarray(X, dtype=float).reshape(-1, 1)
        return np.hstack([a, (a - 45) ** 2 / 100])

    def get_feature_names_out(self, input_features=None):
        return np.array(["Age", "Age_curve"])


def baseline_logistic() -> Pipeline:
    """Roughly the original notebook's model: everything linear, products as a number."""
    pre = ColumnTransformer(
        [
            ("num", StandardScaler(), NUMERIC + ["NumOfProducts"] + FLAGS),
            ("cat", OneHotEncoder(drop="first"), CATEGORICAL),
        ]
    )
    return Pipeline([("pre", pre), ("clf", LogisticRegression(max_iter=1000))])


def explainable_logistic() -> Pipeline:
    """Still a transparent logistic regression, but age can curve and product count is a category."""
    pre = ColumnTransformer(
        [
            ("age", Pipeline([("curve", AgeCurve()), ("scale", StandardScaler())]), ["Age"]),
            ("num", StandardScaler(), ["CreditScore", "Tenure", "Balance", "EstimatedSalary"]),
            ("flags", "passthrough", FLAGS),
            ("cat", OneHotEncoder(drop="first", handle_unknown="ignore"), CATEGORICAL + ["NumOfProducts"]),
        ]
    )
    return Pipeline([("pre", pre), ("clf", LogisticRegression(max_iter=2000, C=1.0))])


def _tree_pre():
    return ColumnTransformer(
        [("cat", OneHotEncoder(handle_unknown="ignore"), CATEGORICAL)], remainder="passthrough"
    )


def random_forest() -> Pipeline:
    """The tuned forest from the original notebook (n_estimators reduced for build speed)."""
    clf = RandomForestClassifier(
        n_estimators=400, max_depth=12, min_samples_leaf=5, max_samples=0.6, n_jobs=-1, random_state=SEED
    )
    return Pipeline([("pre", _tree_pre()), ("clf", clf)])


def gradient_boosting() -> Pipeline:
    clf = HistGradientBoostingClassifier(learning_rate=0.05, max_iter=300, max_leaf_nodes=15, l2_regularization=1.0, random_state=SEED)
    return Pipeline([("pre", _tree_pre()), ("clf", clf)])


MODELS = {
    "Logistic regression (original)": baseline_logistic,
    "Explainable logistic regression": explainable_logistic,
    "Random forest (tuned)": random_forest,
    "Gradient boosting": gradient_boosting,
}


@dataclass
class Split:
    X_train: pd.DataFrame
    X_test: pd.DataFrame
    y_train: pd.Series
    y_test: pd.Series


def split(df: pd.DataFrame, test_size: float = 0.2) -> Split:
    X, y = df[INPUTS], df[TARGET]
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=test_size, stratify=y, random_state=SEED)
    return Split(Xtr, Xte, ytr, yte)


def compare(s: Split, folds: int = 5) -> tuple[pd.DataFrame, dict]:
    """Cross-validated ROC AUC on the training set, then a final check on the held-out test set."""
    cv = StratifiedKFold(folds, shuffle=True, random_state=SEED)
    rows, fitted = [], {}
    for name, make in MODELS.items():
        cv_auc = cross_val_score(make(), s.X_train, s.y_train, cv=cv, scoring="roc_auc", n_jobs=-1)
        model = make().fit(s.X_train, s.y_train)
        p = model.predict_proba(s.X_test)[:, 1]
        rows.append(
            {
                "model": name,
                "cv_roc_auc": cv_auc.mean(),
                "cv_std": cv_auc.std(),
                "test_roc_auc": roc_auc_score(s.y_test, p),
                "test_pr_auc": average_precision_score(s.y_test, p),
                "test_accuracy": ((p >= 0.5).astype(int) == s.y_test).mean(),
            }
        )
        fitted[name] = (model, p)
    table = pd.DataFrame(rows).sort_values("test_roc_auc", ascending=False).reset_index(drop=True)
    return table, fitted


def roc_points(y, p) -> pd.DataFrame:
    fpr, tpr, thr = roc_curve(y, p)
    return pd.DataFrame({"fpr": fpr, "tpr": tpr, "threshold": thr})


def importance(model, s: Split, repeats: int = 8) -> pd.DataFrame:
    """How much test ROC AUC drops when each input is shuffled."""
    r = permutation_importance(model, s.X_test, s.y_test, scoring="roc_auc", n_repeats=repeats, random_state=SEED, n_jobs=-1)
    t = pd.DataFrame({"feature": INPUTS, "auc_drop": r.importances_mean, "std": r.importances_std})
    t["label"] = t["feature"].map(PRETTY)
    return t.sort_values("auc_drop", ascending=False).reset_index(drop=True)


def retention_value(y, p, thresholds, value_saved: float, offer_cost: float, success_rate: float) -> pd.DataFrame:
    """Net value of contacting everyone whose risk is at or above each threshold.

    Every contacted customer costs `offer_cost`. A contacted churner is kept with
    probability `success_rate`, which saves `value_saved`.
    """
    y = np.asarray(y)
    p = np.asarray(p)
    rows = []
    for t in thresholds:
        hit = p >= t
        contacted = int(hit.sum())
        caught = int((hit & (y == 1)).sum())
        net = caught * success_rate * value_saved - contacted * offer_cost
        rows.append({"threshold": t, "contacted": contacted, "churners_reached": caught, "net_value": net})
    return pd.DataFrame(rows)


def export_logistic(model: Pipeline) -> dict:
    """Everything the browser needs to score a customer with the explainable model."""
    pre: ColumnTransformer = model.named_steps["pre"]
    clf: LogisticRegression = model.named_steps["clf"]
    coef = clf.coef_[0]
    names = list(pre.get_feature_names_out())
    weights = dict(zip(names, coef))

    age_scale = pre.named_transformers_["age"].named_steps["scale"]
    num_scale = pre.named_transformers_["num"]
    onehot: OneHotEncoder = pre.named_transformers_["cat"]

    def w(name):
        return float(weights[name])

    categories = {}
    for col, cats in zip(CATEGORICAL + ["NumOfProducts"], onehot.categories_):
        categories[col] = {str(c): (w(f"cat__{col}_{c}") if f"cat__{col}_{c}" in weights else 0.0) for c in cats}

    return {
        "intercept": float(clf.intercept_[0]),
        "age": {
            "mean": [float(v) for v in age_scale.mean_],
            "scale": [float(v) for v in age_scale.scale_],
            "weight": [w("age__Age"), w("age__Age_curve")],
        },
        "numeric": {
            col: {"mean": float(m), "scale": float(sc), "weight": w(f"num__{col}")}
            for col, m, sc in zip(["CreditScore", "Tenure", "Balance", "EstimatedSalary"], num_scale.mean_, num_scale.scale_)
        },
        "flags": {col: w(f"flags__{col}") for col in FLAGS},
        "categories": categories,
    }


def score_with_export(spec: dict, row: dict) -> float:
    """Python twin of the in-browser scorer, used to test that the export is exact."""
    a = float(row["Age"])
    z = spec["intercept"]
    for v, m, sc, wt in zip([a, (a - 45) ** 2 / 100], spec["age"]["mean"], spec["age"]["scale"], spec["age"]["weight"]):
        z += wt * (v - m) / sc
    for col, s in spec["numeric"].items():
        z += s["weight"] * (float(row[col]) - s["mean"]) / s["scale"]
    for col, wt in spec["flags"].items():
        z += wt * float(row[col])
    for col, table in spec["categories"].items():
        z += table.get(str(row[col]), 0.0)
    return 1 / (1 + np.exp(-z))
