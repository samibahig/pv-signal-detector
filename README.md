---
title: Pharmacovigilance Signal Detector
emoji: ⚕
colorFrom: red
colorTo: pink
sdk: docker
pinned: false
license: mit
short_description: PRR / ROR / IC / EBGM pharmacovigilance signals
---

# ⚕ Pharmacovigilance Signal Detector

A **domain-specific signal detection visualization** for pharmacovigilance that encodes statistical inference, regulatory rules, and ranked safety signal prioritization into a single interactive primitive.

Built with [Dash](https://dash.plotly.com/) + [Plotly](https://plotly.com/) — deployable on **Hugging Face Spaces** (Docker SDK) or any Python environment.

---

## Proposed Plotly Express API

This tool implements a domain-specific primitive that does not yet exist in Plotly Express. The equivalent high-level call, once contributed to the Plotly library, would look like:

```python
import plotly.express as px

fig = px.signal_detection(
    df,
    x="prr",               # signal strength axis: "prr" | "ror" | "ic" | "ebgm"
    y="cases",             # evidence strength axis
    size="ebgm",           # point size encoding: "ebgm" | "inv_ci_width" | "cases"
    color="signal_class",  # color by regulatory class: "signal" | "watch" | "background"
    hover_data=["drug", "event", "prr", "ror", "ic", "ebgm",
                "cases", "lower_ci", "upper_ci"],
    thresholds={
        "prr":   2,    # PRR ≥ 2 → disproportional reporting
        "cases": 3,    # minimum case count
        "ci":    1,    # lower 95% CI > 1
    },
)
fig.show()
```

### What the function does internally (abstracted steps)

| Step | Action |
|------|--------|
| 1 | Ingest raw contingency table `(a, b, c, d)` |
| 2 | Compute PRR, ROR, IC, EBGM, 95% CI |
| 3 | Apply regulatory decision rules |
| 4 | Assign `signal_class` per pair |
| 5 | Generate scatter with threshold lines |
| 6 | Format rich hover tooltips |

Until this function is merged into `plotly.py`, you can use this Dash app directly, or import the standalone functions:

```python
from app import compute_metrics, classify_signals

df = compute_metrics(raw_df)          # adds prr, ror, ic, ebgm, lower_ci, upper_ci
df = classify_signals(df,             # adds signal_class, signal_score
    prr_thresh=2.0,
    n_thresh=3,
    ci_thresh=1.0,
)
```

---

## What this does

Standard scatter plots show correlations. This tool does something different:

| Layer | Meaning |
|-------|---------|
| **Data** | Drug–event contingency counts (a, b, c, d) |
| **Stats** | PRR / ROR / IC / EBGM computed on-the-fly |
| **Rules** | Regulatory thresholds (FDA / EMA standard) |
| **Viz** | Scatter with decision quadrants |
| **Product** | Ranked safety signal prioritization |

---

## Metrics computed

### PRR — Proportional Reporting Ratio
```
PRR = (a / (a+b)) / (c / (c+d))
```
- PRR > 1 → disproportional reporting
- **PRR ≥ 2** → standard regulatory screening threshold

### ROR — Reporting Odds Ratio
```
ROR = (a × d) / (b × c)
```
Symmetric odds ratio, widely used in case–non-case designs.

### Log ROR 95% Confidence Interval
```
log(ROR) ± 1.96 × sqrt(1/a + 1/b + 1/c + 1/d)
```

### IC — Information Component (bits)
```
IC = log₂( P(d,e) / (P(d) × P(e)) )
```
Measures how much drug and event co-occur beyond chance.

### EBGM — Empirical Bayes Geometric Mean
Gamma-Poisson shrinkage approximation:
```
EBGM ≈ (a + 0.5) / (expected + 0.5)
```
Stabilizes rare-event inflation and reduces false positives.

---

## Regulatory Decision Engine

```
IF PRR ≥ 2 AND cases ≥ 3 AND lower_CI > 1  →  🔴 signal
ELSE IF PRR ≥ 2                              →  🟡 watch
ELSE                                          →  ⚫ background
```

All thresholds are adjustable live in the UI.

---

## Input data format

Upload a CSV with these columns:

| Column | Type | Description |
|--------|------|-------------|
| `drug` | string | Drug name |
| `event` | string | Adverse event name |
| `a` | int | Drug + event (cases) |
| `b` | int | Drug + no event |
| `c` | int | Other drugs + event |
| `d` | int | Other drugs + no event |

No pre-computed metrics needed — everything is derived automatically. Laplace smoothing (0.5) applied on zero cells.

---

## Decision Quadrants

```
         │  High PRR
    🟡   │   🔴
  Rare   │  Priority
  Signal │  Signal
─────────┼──────────── PRR = 2
    🟢   │   🟠
  Noise  │  Frequent /
         │  Non-specific
         │
      Low PRR
```

---

## Local development

```bash
pip install -r requirements.txt
python app.py
# → http://localhost:7860
```

---

## Real-world usage

- Pharmacovigilance teams (FDA / EMA workflows)
- Post-market drug safety monitoring
- Signal prioritization committees
- Safety analytics dashboards
- FAERS / EudraVigilance spontaneous reporting analysis

---

## Statistical considerations

- **Underreporting bias**: counts reflect reported cases, not actual incidence
- **Rare event instability**: EBGM shrinkage + minimum count threshold mitigates inflated PRR
- **Multiple testing**: Bayesian shrinkage (EBGM, IC) and CI filtering reduce false discovery
- **Laplace smoothing**: applied automatically when any contingency cell = 0
