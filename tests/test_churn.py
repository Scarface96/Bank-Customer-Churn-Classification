import numpy as np
import pandas as pd

from analysis import data, model


def test_data_loads_and_is_clean():
    df = data.load()
    assert len(df) == 10_000
    assert df["Exited"].between(0, 1).all()


def test_engineered_features():
    df = data.add_features(data.load())
    assert {"balance_to_income", "income_v_product", "age_band", "zero_balance"} <= set(df.columns)
    assert df["age_band"].notna().all()


def test_churn_rate_matches_manual_calculation():
    df = pd.DataFrame({"g": ["a", "a", "b", "b", "b"], "Exited": [1, 0, 1, 1, 0]})
    t = data.churn_rate(df, "g").set_index("g")
    assert t.loc["a", "churn_rate"] == 0.5
    assert abs(t.loc["b", "churn_rate"] - 2 / 3) < 1e-9


def test_lift_table_covers_every_customer():
    y = np.array([1, 0] * 50)
    p = np.linspace(1, 0, 100)
    t = data.lift_table(y, p)
    assert t["customers"].sum() == 100
    assert t["share_of_all_churners"].iloc[-1] == 1


def test_retention_value_arithmetic():
    y = np.array([1, 1, 0, 0])
    p = np.array([0.9, 0.6, 0.55, 0.1])
    t = model.retention_value(y, p, [0.5], value_saved=1000, offer_cost=100, success_rate=0.5).iloc[0]
    # 3 contacted, 2 churners reached: 2 * 0.5 * 1000 - 3 * 100
    assert (t["contacted"], t["churners_reached"], t["net_value"]) == (3, 2, 700)


def test_browser_scorer_matches_the_model():
    df = data.add_features(data.load())
    s = model.split(df)
    m = model.explainable_logistic().fit(s.X_train, s.y_train)
    spec = model.export_logistic(m)
    rows = s.X_test.head(200)
    expected = m.predict_proba(rows)[:, 1]
    got = [model.score_with_export(spec, r) for r in rows.to_dict("records")]
    assert np.allclose(expected, got, atol=1e-9)
