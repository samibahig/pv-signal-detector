---
title: Pharmacovigilance Signal Detector
emoji: ⚕
colorFrom: red
colorTo: orange
sdk: docker
pinned: false
license: mit
short_description: PRR / ROR / IC / EBGM signal detection with regulatory decision engine
---

# ⚕ Pharmacovigilance Signal Detector

A **domain-specific signal detection visualization** for pharmacovigilance that encodes statistical inference, regulatory rules, and ranked safety signal prioritization into a single interactive primitive.

Built with [Dash](https://dash.plotly.com/) + [Plotly](https://plotly.com/) — deployable on **Hugging Face Spaces** (Docker SDK) or any Python environment.

---

## What this does

Standard scatter plots show correlations. This tool does something different:

| Layer | Meaning |
|-------|---------|
| **Data** | Drug–event contingency counts (a, b, c, d) |
| **Stats** | PRR / ROR / IC / EBGM computed automatically |
| **Rules** | Regulatory thresholds applied (FDA / EMA) |
| **Viz** | Scatter with decision quadrants |
| **Product** | Ranked signal prioritization system |

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
Bayesian shrinkage estimator using Gamma-Poisson approximation:
```
EBGM ≈ (a + 0.5) / (expected + 0.5)
```
Stabilizes rare event inflation and reduces false positives.

---

## Regulatory Decision Engine

```
IF PRR ≥ 2 AND cases ≥ 3 AND lower_CI > 1  →  🔴 signal
ELSE IF PRR ≥ 2                              →  🟡 watch
ELSE                                          →  ⚫ background
```

All thresholds are adjustable in the UI.

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

No pre-computed metrics needed — everything is derived automatically.

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
- FAERS / EudraVigilance analysis

---

## Statistical considerations

- **Underreporting bias**: counts reflect reported cases, not actual incidence
- **Rare event instability**: EBGM shrinkage + minimum count threshold mitigates inflated PRR for rare events
- **Multiple testing**: Bayesian shrinkage (EBGM, IC) and CI filtering reduce false discovery rate
- **Laplace smoothing**: applied automatically when any contingency cell = 0
