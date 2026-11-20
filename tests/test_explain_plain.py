"""
Tests for the plain-language explanations (src/models/explain_plain.py).

The explanations are read by the public, so the key properties are:
  - they are built from the model's real SHAP contributions,
  - the six everyday factors account for the whole push (shares add to ~100),
  - the wording is jargon-free,
  - the direction (raises / lowers) agrees with the model's own SHAP sign.
"""

import os
import sys

import numpy as np
import pandas as pd
import pytest
from xgboost import XGBClassifier

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src", "models"))

import explain_plain
from explain_plain import GROUPS, build_context, explain_rows

FEATURES = [f for _, _, members in GROUPS for f in members]
JARGON = ["lag", "shap value", "log-odds", "logit", "roll3", "roll7", "_lag", "feature", "xgboost"]


@pytest.fixture(scope="module")
def setup():
    rng = np.random.default_rng(0)
    n = 3000
    dates = pd.date_range("2000-01-01", periods=n)
    X = pd.DataFrame({
        "doy_sin": np.sin(2 * np.pi * dates.dayofyear / 365.25), "doy_cos": np.cos(2 * np.pi * dates.dayofyear / 365.25),
        "rainfall_lag1": rng.gamma(1, 4, n), "rainfall_lag2": rng.gamma(1, 4, n), "rainfall_lag3": rng.gamma(1, 4, n),
        "rainfall_roll3": rng.gamma(1, 4, n), "rainfall_roll7": rng.gamma(1, 4, n),
        "runoff_lag1": rng.gamma(1, 3, n), "runoff_roll3": rng.gamma(1, 3, n),
        "soil_moisture_lag1": rng.uniform(0.25, 0.5, n), "rain_soil_interaction_lag1": rng.gamma(1, 2, n),
        "temperature_lag1": rng.normal(22, 2, n), "humidity_lag1": rng.uniform(40, 100, n), "wind_speed_lag1": rng.uniform(0, 4, n),
        "elevation_m": 850.0, "slope_degrees": 13.0, "twi": 3.1,
    })[FEATURES]
    y = (X["runoff_lag1"] + 0.5 * X["rainfall_roll3"] > 7).astype(int).values
    model = XGBClassifier(n_estimators=60, max_depth=3, random_state=0, eval_metric="logloss").fit(X.values, y)
    weather = pd.DataFrame({"date": dates, "runoff_mm": X["runoff_lag1"], "rainfall_mm": X["rainfall_lag1"],
                            "soil_moisture_m3m3": X["soil_moisture_lag1"], "relative_humidity_pct": X["humidity_lag1"]})
    season = weather.assign(flood_risk=y)
    return model, X, dates, build_context(weather, season)


def _explain(setup, rows, estimated=None):
    model, X, dates, ctx = setup
    probs = model.predict_proba(X.values[rows])[:, 1]
    bands = ["LOW" if p < 0.33 else "MODERATE" if p < 0.66 else "HIGH" for p in probs]
    return explain_rows(model, X.values[rows], FEATURES, [dates[i] for i in rows], probs, bands, ctx, estimated), probs


def test_structure_and_shares(setup):
    out, _ = _explain(setup, [0, 1, 2, 3, 4])
    assert len(out) == 5
    for e in out:
        assert e["headline"] and e["method_note"]
        keys = [f["key"] for f in e["factors"]]
        assert sorted(keys) == sorted(k for k, _, _ in GROUPS)
        assert abs(sum(f["strength_pct"] for f in e["factors"]) - 100) <= 4
        assert [f["strength_pct"] for f in e["factors"]] == sorted((f["strength_pct"] for f in e["factors"]), reverse=True)
        assert all(f["direction"] in ("raises", "lowers", "little") for f in e["factors"])


def test_wording_has_no_technical_jargon(setup):
    out, _ = _explain(setup, list(range(0, 300, 7)))
    for e in out:
        text = (e["headline"] + " " + " ".join(f["detail"] + f["title"] for f in e["factors"])).lower()
        for word in JARGON:
            assert word not in text, f"jargon '{word}' leaked into: {text[:200]}"


def test_direction_matches_the_models_own_shap_sign(setup):
    """A factor reported as 'raises' must have a positive summed SHAP value."""
    import shap
    model, X, dates, ctx = setup
    rows = [10, 200, 500, 900]
    out, _ = _explain(setup, rows)
    sv = shap.TreeExplainer(model).shap_values(X.values[rows])
    idx = {n: i for i, n in enumerate(FEATURES)}
    for r, e in enumerate(out):
        for f in e["factors"]:
            members = dict((k, m) for k, _, m in GROUPS)[f["key"]]
            push = sum(sv[r, idx[m]] for m in members)
            if f["direction"] == "raises":
                assert push > 0
            if f["direction"] == "lowers":
                assert push < 0


def test_high_and_low_headlines_follow_the_probability(setup):
    model, X, dates, ctx = setup
    p = model.predict_proba(X.values)[:, 1]
    hi, lo = int(np.argmax(p)), int(np.argmin(p))
    out, _ = _explain(setup, [hi, lo])
    assert out[0]["headline"].startswith("High risk")
    assert out[1]["headline"].startswith("Low risk")


def test_estimated_days_carry_a_caution_note(setup):
    out, _ = _explain(setup, [0, 1], estimated=[True, False])
    assert "estimated" in out[0]["note"]
    assert out[1]["note"] == ""


def test_explainer_is_cached_per_model(setup):
    model = setup[0]
    assert explain_plain._explainer_for(model) is explain_plain._explainer_for(model)
