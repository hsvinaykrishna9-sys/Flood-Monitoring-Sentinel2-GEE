"""
Plain-language explanations of flood-risk predictions, for the web dashboard.

The model uses 14 (whole district) or 17 (per taluk) technical inputs such as
"runoff_lag1" or "rain_soil_interaction_lag1". Nobody visiting the dashboard
should have to read those. This module does three things:

  1. Measures how much each input pushed one prediction up or down, using SHAP
     (TreeExplainer: exact for tree models, and the contributions add up to the
     model's own output, so nothing here is an approximation of the model).
  2. Adds the inputs together into six everyday factors. SHAP values can be
     summed safely because they are additive, so a factor's push is the exact
     total of its inputs' pushes:
        water running off the ground, recent rainfall, how wet the ground is,
        time of year, recent air conditions, and (per taluk) the shape of the land.
  3. Writes one sentence per factor using the real values behind that day,
     compared with what is usual for the same calendar month, plus a one-line
     headline. No technical words (lag, log-odds, SHAP value) appear in it.

The result is a small JSON-friendly dict, built by `explain_rows`.
"""

import numpy as np

MONTHS = ["January", "February", "March", "April", "May", "June", "July",
          "August", "September", "October", "November", "December"]

# (key, title shown to people, the model inputs it is made of)
GROUPS = [
    ("runoff", "Water running off the ground", ["runoff_lag1", "runoff_roll3"]),
    ("rain", "Recent rainfall", ["rainfall_lag1", "rainfall_lag2", "rainfall_lag3", "rainfall_roll3", "rainfall_roll7"]),
    ("soil", "How wet the ground is", ["soil_moisture_lag1", "rain_soil_interaction_lag1"]),
    ("season", "Time of year", ["doy_sin", "doy_cos"]),
    ("weather", "Recent air conditions", ["temperature_lag1", "humidity_lag1", "wind_speed_lag1"]),
    ("terrain", "Shape of the land", ["elevation_m", "slope_degrees", "twi"]),
]

# Short reasons used in the headline sentence (lower-case fragments).
REASON_UP = {
    "runoff": "a lot of water was already running off the ground",
    "rain": "it has been raining heavily over the last few days",
    "soil": "the ground is already soaked and cannot take much more water",
    "season": "this is the time of year when floods are most common",
    "weather": "recent humidity, temperature and wind point towards wet weather",
    "terrain": "the shape of the land in this area adds to the risk",
}
REASON_DOWN = {
    "runoff": "very little water was running off the ground",
    "rain": "there has been little rain lately",
    "soil": "the ground is dry enough to soak up more rain",
    "season": "this is a quiet time of year for floods",
    "weather": "recent humidity, temperature and wind point towards settled weather",
    "terrain": "the shape of the land in this area keeps the risk down",
}

MIN_SHARE = 0.08          # a factor below 8% of the total push is reported as "little effect"
BAND_WORD = {"LOW": "Low", "MODERATE": "Moderate", "HIGH": "High"}

METHOD_NOTE = ("How this is worked out: the prediction model looks at recent rainfall, water running off the ground, "
               "soil wetness, the time of year, air conditions and (for each taluk) the shape of the land. "
               "A method called SHAP measures how much each of these pushed this particular prediction up or down. "
               "We add them together into the everyday factors shown here.")
ESTIMATED_NOTE = ("This day is further ahead, so some of the inputs are estimated from recent trends "
                  "rather than measured. Treat the reasons as a guide.")

_EXPLAINERS = {}


def _explainer_for(model):
    import shap
    key = id(model)
    if key not in _EXPLAINERS:
        _EXPLAINERS[key] = shap.TreeExplainer(model)
    return _EXPLAINERS[key]


# ----------------------------------------------------------------- context

def build_context(weather_df, season_df):
    """Typical values for each calendar month, so a number can be described as
    "higher than usual". `weather_df` supplies the monthly averages (the taluk's
    own history for a taluk, the district history otherwise); `season_df` is the
    long district record used for the share of flood-risk days in each month."""
    w = weather_df.copy()
    w["month"] = w["date"].dt.month
    typical = w.groupby("month").agg(
        runoff=("runoff_mm", "mean"), rain=("rainfall_mm", "mean"), soil=("soil_moisture_m3m3", "mean"),
        humidity=("relative_humidity_pct", "mean"),
    )
    s = season_df.copy()
    s["month"] = s["date"].dt.month
    rate = s.groupby("month")["flood_risk"].mean() * 100
    years = int(s["date"].dt.year.max() - s["date"].dt.year.min())   # same "76 years" the dashboard shows elsewhere
    return {
        "typical": typical.to_dict("index"),
        "flood_rate_by_month": rate.to_dict(),
        "flood_rate_overall": float(s["flood_risk"].mean() * 100),
        "years": years,
    }


# ----------------------------------------------------------------- wording

def _compare_ratio(value, typical):
    if typical <= 0:
        return "about the same as"
    r = value / typical
    if r >= 2: return "much higher than"
    if r >= 1.25: return "higher than"
    if r >= 0.8: return "about the same as"
    if r >= 0.4: return "lower than"
    return "much lower than"


def _compare_points(value, typical):
    d = (value - typical) * 100          # soil water content, percentage points
    if d >= 8: return "much wetter than"
    if d >= 3: return "wetter than"
    if d > -3: return "about as wet as"
    if d > -8: return "drier than"
    return "much drier than"


def _sentence(key, vals, month, ctx, direction):
    m = MONTHS[month - 1]
    typ = ctx["typical"].get(month, {})
    if key == "runoff":
        v, t = vals["runoff_lag1"], typ.get("runoff", 0.0)
        return (f"The day before, about {v:.1f} mm of water ran off the ground, which is "
                f"{_compare_ratio(v, t)} the usual {t:.1f} mm for {m}.")
    if key == "rain":
        v, t = vals["rainfall_roll3"], typ.get("rain", 0.0)
        return (f"Over the three days before, it rained about {v:.1f} mm a day on average, which is "
                f"{_compare_ratio(v, t)} the usual {t:.1f} mm for {m}.")
    if key == "soil":
        v, t = vals["soil_moisture_lag1"], typ.get("soil", 0.0)
        wetter = (v - t) * 100 >= 3
        extra = (" Wet ground soaks up less new rain, so more of it runs into streams and rivers."
                 if direction == "raises" and wetter else "")
        return (f"The soil was holding about {v * 100:.0f}% water, which is "
                f"{_compare_points(v, t)} the usual {t * 100:.0f}% for {m}.{extra}")
    if key == "season":
        rate = ctx["flood_rate_by_month"].get(month, 0.0)
        monsoon = " It falls in the monsoon season." if month in (6, 7, 8, 9) else ""
        return (f"Over {ctx['years']} years of records, {rate:.0f}% of {m} days were flood-risk days, "
                f"compared with {ctx['flood_rate_overall']:.0f}% across the whole year.{monsoon}")
    if key == "weather":
        h, tmp, w = vals["humidity_lag1"], vals["temperature_lag1"], vals["wind_speed_lag1"]
        return (f"The day before, the air was {h:.0f}% humid and {tmp:.0f}°C, with a wind of "
                f"about {w * 3.6:.0f} km/h.")
    if key == "terrain":
        return (f"This area sits about {vals['elevation_m']:.0f} m above sea level with an average slope of "
                f"{vals['slope_degrees']:.0f}°. The shape of the land changes how quickly water drains away.")
    return ""


def _headline(band, prob, factors):
    pct = round(prob * 100)
    raising = [f for f in factors if f["direction"] == "raises"]
    lowering = [f for f in factors if f["direction"] == "lowers"]
    word = BAND_WORD.get(band, band.title())
    if band == "LOW":
        if lowering:
            return f"{word} risk ({pct}%), mainly because {REASON_DOWN[lowering[0]['key']]}."
        return f"{word} risk ({pct}%). No single factor stands out."
    if raising:
        text = f"{word} risk ({pct}%), mainly because {REASON_UP[raising[0]['key']]}"
        if len(raising) > 1 and raising[1]["strength_pct"] >= 20:
            text += f", and {REASON_UP[raising[1]['key']]}"
        text += "."
        if band == "MODERATE" and lowering:
            text += f" On the other hand, {REASON_DOWN[lowering[0]['key']]}."
        return text
    return f"{word} risk ({pct}%). No single factor stands out."


# ----------------------------------------------------------------- main entry

def explain_rows(model, X_rows, feature_names, dates, probs, bands, ctx, estimated=None):
    """Explain several predictions at once. Returns one dict per row:
        {headline, factors: [{key, title, direction, strength_pct, detail}], note, method_note}
    `direction` is "raises", "lowers" or "little". `strength_pct` is the factor's share
    of the total push (all factors together add up to about 100)."""
    X = np.asarray(X_rows, dtype=float).reshape(len(X_rows), -1)
    shap_vals = np.asarray(_explainer_for(model).shap_values(X))
    index = {name: i for i, name in enumerate(feature_names)}
    estimated = estimated or [False] * len(X)
    out = []
    for r in range(len(X)):
        vals = {n: float(X[r, i]) for n, i in index.items()}
        month = int(dates[r].month)
        groups = []
        for key, title, members in GROUPS:
            present = [n for n in members if n in index]
            if present:
                groups.append((key, title, float(sum(shap_vals[r, index[n]] for n in present))))
        total = sum(abs(g[2]) for g in groups) or 1.0
        factors = []
        for key, title, push in groups:
            share = abs(push) / total
            direction = "little" if share < MIN_SHARE else ("raises" if push > 0 else "lowers")
            factors.append({"key": key, "title": title, "direction": direction,
                            "strength_pct": int(round(share * 100)),
                            "detail": _sentence(key, vals, month, ctx, direction)})
        factors.sort(key=lambda f: -f["strength_pct"])
        out.append({
            "headline": _headline(bands[r], float(probs[r]), factors),
            "factors": factors,
            "note": ESTIMATED_NOTE if estimated[r] else "",
            "method_note": METHOD_NOTE,
        })
    return out
