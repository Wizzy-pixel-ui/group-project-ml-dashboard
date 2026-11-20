# GROUP Project ML — Dashboard

**Predicting natural-flow decline and planning artificial lift on the Volve field**

An interactive dashboard built on top of the [GROUP Project ML](https://github.com/Wizzy-pixel-ui/GROUP-project-ML.git) research prototype. It reads 9,143 days of real production history from six North Sea wells, fits Arps decline curves, screens artificial-lift methods, and renders a 3D solution-gas-drive reservoir view.

**Live dashboard:** https://static.teamily.ai/sites/4c4359e0-0d04-4bf2-950a-1f5ac8b27922/webpages/group-project-ml-dashboard_v10/index.html
**Source repository:** https://github.com/Wizzy-pixel-ui/GROUP-project-ML.git

---

## Contents

1. [What this is, in one paragraph](#1-what-this-is-in-one-paragraph)
2. [The ML concept behind the dashboard](#2-the-ml-concept-behind-the-dashboard)
3. [The reservoir engineering concept](#3-the-reservoir-engineering-concept)
4. [Solution gas drive](#4-solution-gas-drive)
5. [Bubble points and dew points](#5-bubble-points-and-dew-points)
6. [Why some datasets lack important variables](#6-why-some-datasets-lack-important-variables)
7. [Every chart in the dashboard, explained](#7-every-chart-in-the-dashboard-explained)
8. [Presentation cheat sheet](#8-presentation-cheat-sheet)
9. [What this project does not do](#9-what-this-project-does-not-do)
10. [Data, licensing and credits](#10-data-licensing-and-credits)

---

## 1. What this is, in one paragraph

We took a petroleum-engineering machine-learning prototype and turned it into a dashboard you can click through. The prototype asks one question: **using a well's production history, can we forecast its oil rate and estimate when natural flow will fall below an economic limit?** The dashboard answers that question visually — it fits a decline curve to each well, projects it forward, marks the month the well crosses 100 barrels per day, and then screens six artificial-lift methods against what we know about the well. It also shows, honestly, that the machine-learning models are **not yet better than a naive baseline** on this data. That honesty is a feature, not a bug: it tells you exactly what work remains.

### The numbers you will be quoting

| Quantity | Value |
|---|---|
| Daily production rows | **9,143** |
| Producing wells | **6** |
| Date range | **2008-02-12 → 2016-09-17** |
| Cumulative oil | **10.03 million bbl** |
| Cumulative gas | **1.47 billion scf** |
| Cumulative water | **15.32 million bbl** |
| Peak field rate | **5,901.84 bbl/d** on **2009-01-08**, from well **15/9-F-12** |
| Water cut, first month | **0.83 %** |
| Water cut, last producing month | **84.46 %** |
| Field GOR | **147 scf/bbl** |
| On-stream fraction | **84.1 %** |
| Zero-rate days | **1,144** (12.5 % of all well-days) |
| Economic limit used | **100 bbl/d** |
| Forecast horizon | **6 months** |

### The six wells

| Well | Rows | First date | Last date | Oil (bbl) | Peak (bbl/d) | Final water cut | Zero-rate days |
|---|---:|---|---|---:|---:|---:|---:|
| 15/9-F-12 | 3,056 | 2008-02-12 | 2016-09-17 | 4,579,610 | 5,901.84 | 88.4 % | 220 |
| 15/9-F-14 | 3,056 | 2008-02-12 | 2016-09-17 | 3,942,233 | 5,644.37 | 96.6 % | 333 |
| 15/9-F-11 | 1,165 | 2013-07-08 | 2016-09-17 | 1,147,849 | 2,064.61 | 84.5 % | 42 |
| 15/9-F-1 C | 744 | 2014-04-08 | 2016-04-21 | 177,187 | 1,549.81 | 72.7 % | 315 |
| 15/9-F-15 D | 978 | 2014-01-12 | 2016-09-17 | 148,519 | 513.12 | 58.6 % | 211 |
| 15/9-F-5 | 144 | 2016-04-12 | 2016-09-17 | 38,433 | 396.80 | 21.0 % | 23 |

Two wells — **15/9-F-12 (45.6 %)** and **15/9-F-14 (39.3 %)** — account for **84.9 %** of all the oil ever produced. That concentration matters: any field-level average is really a statement about those two wells.

---

## 2. The ML concept behind the dashboard

### 2.1 What the models are

There are **two separate models** in this project, and it is important not to confuse them.

**Model A — the regression forecaster.** A `HistGradientBoostingRegressor` from scikit-learn. It predicts a **number** — the oil rate some months ahead. It is trained on engineered features built from each well's own history: recent rate, rolling averages, drawdown, depletion fraction, water cut, gas-oil ratio, choke setting, cumulative production fraction. Twelve features in total:

```
depletion, drawdown, p_over_pb, bhp_over_pb, water_cut, gor,
oil_rate, liq, rate_6m, rate_3m, cum_frac, choke
```

**Model B — the lift-transition classifier.** Also a gradient-boosted tree model, but this one predicts a **probability** — the chance that a well will need artificial lift within the next 6 months. It is trained on the **synthetic** 36-well dataset, because that is the only dataset in the project that has dated lift labels.

**Model C — Arps decline curves.** Not machine learning at all. A classical petroleum-engineering decline curve, fitted by nonlinear least squares. It is the interpretable baseline that the ML models have to beat.

### 2.2 How they were trained

**The regression model** was trained on a **grouped-well split**: entire wells are held out, so the model is tested on wells it has never seen. This is the honest way to test whether a model generalises to a new well, rather than memorising one well's history. For the Arps comparison, the setup is a single fixed-origin forecast:

- **Context window:** February 2008 → December 2014 (83 months)
- **Holdout:** January 2015 → September 2016 (21 months)
- **Well:** 15/9-F-12
- **Arps training points:** 82 positive-rate months (December 2014 is zero and is excluded from curve fitting)
- **HGB training examples:** 118 eligible one-step examples from four *other* Volve wells

**The classifier** was trained on 1,944 examples from 27 synthetic wells and tested on 648 examples from 9 held-out synthetic wells. The positive class — "lift needed within 6 months" — occurs in only **3.9 %** of examples. That is a heavily imbalanced problem, and it explains almost everything about the metrics below.

### 2.3 What the metrics mean

| Metric | What it measures | How to read it |
|---|---|---|
| **R²** | How much better than predicting the mean | **1.0** = perfect. **0.0** = no better than guessing the average. **Negative** = *worse* than guessing the average. |
| **MAE** | Mean absolute error | Average miss, in the target's own units (bbl/d here). Lower is better. |
| **RMSE** | Root mean squared error | Like MAE but punishes big misses harder. Always ≥ MAE. |
| **AUC** | Area under the ROC curve | **0.5** = coin flip. **1.0** = perfect ranking. Measures whether the model *ranks* risky wells above safe ones. |
| **Precision** | Of the wells it flagged, how many were right | Low precision = many false alarms. |
| **Recall** | Of the wells that actually needed lift, how many it caught | Low recall = it misses real events. |
| **F1** | Harmonic mean of precision and recall | A single number when both matter. |

### 2.4 The honest results — and they are poor

**This is the most important table in this README. Do not hide it.**

#### Regression: 21-month holdout on well 15/9-F-12

| Model | MAE (bbl/d) | RMSE (bbl/d) | R² | Verdict |
|---|---:|---:|---:|---|
| **HistGradientBoosting (recursive)** | **207.11** | **261.55** | **−0.35** | Lowest MAE, but still worse than the mean |
| Chronos-Bolt-Small (zero-shot) | 275.04 | 343.65 | −1.34 | Pretrained model, not petroleum-specific |
| **Arps exponential** | **301.68** | **355.72** | **−1.51** | Interpretable engineering baseline |
| Chronos-2 (clipped at zero) | 387.52 | 446.67 | −2.95 | 80 % interval covered only 19 % |
| **Persistence** | **388.97** | **449.20** | **−3.00** | "Tomorrow = today" |
| Chronos-2 (raw) | 507.10 | 526.06 | −4.48 | Predicted negative rates 90.5 % of the time |

**Every single R² is negative.** That means **no model in this table beats simply predicting the holdout mean.** The lowest MAE belongs to the gradient-boosting model, but MAE and R² disagree here because R² is squared-error based and the holdout contains a large operational discontinuity — a zero-rate December 2014 followed by a big January 2015 restart.

**Say this plainly if asked:** *"Our best model has an R² of −0.35. That means it is worse than a naive average. We report it because hiding it would be dishonest, and because it tells us the model is not ready."*

#### Earlier one-step comparisons — the ML model loses to persistence

| Target and evaluation | HGB MAE | Persistence MAE | Winner |
|---|---:|---:|---|
| Oil rate, one held-out well | 293.67 | **104.35** | Persistence, by 2.8× |
| Downhole pressure, one held-out well | 18.21 | **12.29** | Persistence |
| Oil rate, purged later-time holdout | 140.27 | **52.99** | Persistence, by 2.6× |
| Downhole pressure, purged later-time holdout | 19.56 | **13.52** | Persistence |

In the simple one-step setting, **gradient boosting is substantially worse than persistence on every single target.** This is a real and common outcome on small, noisy, non-stationary production data.

#### Classifier: lift-transition prediction (synthetic data)

| Metric | Value | What it means |
|---|---:|---|
| **AUC** | **0.793** | Decent *ranking* — it does put risky wells above safe ones |
| **Accuracy** | **92.9 %** | Misleadingly high — see below |
| **Precision** | **18.2 %** | Of 33 wells it flagged, only 6 truly needed lift |
| **Recall** | **24.0 %** | It caught only 6 of the 25 real lift events |
| **F1** | **0.207** | Weak overall |

Confusion matrix (648 test examples):

|  | Predicted no lift | Predicted lift |
|---|---:|---:|
| **Actually no lift** | 596 (TN) | 27 (FP) |
| **Actually lift** | 19 (FN) | 6 (TP) |

**Why 92.9 % accuracy is a trap.** Only 3.9 % of examples are positive. A model that predicts "no lift" for *everything* would score **96.1 % accuracy** — higher than our model's 92.9 %. So the headline accuracy number is worse than doing nothing. The AUC of 0.793 is the only genuinely encouraging number here: the model does rank risk sensibly, it just sets its threshold badly.

**Say this plainly if asked:** *"Accuracy of 92.9 % sounds good but is worse than always saying no. The model misses 19 of 25 real lift events. Its AUC of 0.793 says the ranking is useful; the threshold is not."*

### 2.5 Feature importance — what actually drives the prediction

Measured as the drop in AUC when each feature is shuffled (ΔAUC):

| Feature | ΔAUC | Reading |
|---|---:|---|
| **water_cut** | **+0.1877** | Dominant by a factor of 3 |
| liq (liquid rate) | +0.0564 | Second |
| bhp_over_pb | +0.0260 | Pressure relative to bubble point |
| oil_rate | +0.0218 | Current rate |
| cum_frac | +0.0090 | How much of the reserve is gone |
| gor | +0.0006 | Negligible |
| rate_3m | −0.0017 | Noise |
| p_over_pb | −0.0019 | Noise |
| choke | −0.0020 | Noise |
| rate_6m | −0.0042 | Noise |
| drawdown | −0.0065 | Noise |
| depletion | −0.0334 | Actively harmful |

**Water cut dominates.** That is physically sensible: rising water cut is the classic signature of a well drowning. But note that **six of the twelve features have negative importance** — they add noise rather than signal. A leaner model would probably do better.

---

## 3. The reservoir engineering concept

### 3.1 Natural flow

A well flows naturally when the pressure in the reservoir is high enough to push fluids all the way up the tubing to surface, against gravity and friction, without any pump. The energy comes from the reservoir itself.

Think of it as a balance between two curves:

- **IPR — Inflow Performance Relationship.** How much fluid the reservoir can deliver into the wellbore at a given bottom-hole pressure. Lower pressure at the sandface = more inflow.
- **VLP — Vertical Lift Performance.** How much pressure you need at the bottom of the well to lift a given rate to surface. Higher rate = more pressure needed.

Where those two curves cross is the **operating point** — the rate and pressure the well will actually settle at. Natural flow works as long as that operating point sits above the economic limit.

### 3.2 Why reservoirs stop flowing naturally

Four reasons, and they often act together:

1. **Pressure depletion.** Production removes fluid, so reservoir pressure falls. Less pressure = less driving energy = lower rate. This is the fundamental one.
2. **Rising water cut.** As water encroaches, the well lifts more and more water per barrel of oil. Water is heavier and gives no revenue, so the same pressure lifts less oil. On Volve, water cut went from **0.83 % to 84.46 %** — the field is now mostly producing water.
3. **Rising gas-oil ratio (in some cases).** Gas breaking out of solution can actually help (gas lift), but excessive free gas causes slugging and reduces pump efficiency.
4. **Mechanical limits.** Sand, scale, wax, or a rising fluid column can overwhelm the available pressure.

### 3.3 Decline curves and Arps

An **Arps decline curve** is an empirical fit to a well's falling production rate. It is not derived from physics — it is a curve that happens to describe decline well. J.J. Arps formalised three forms in 1945:

**Exponential** (b = 0):
$$q(t) = q_i \, e^{-D_i t}$$

**Hyperbolic** (0 < b < 1):
$$q(t) = \frac{q_i}{(1 + b D_i t)^{1/b}}$$

**Harmonic** (b = 1):
$$q(t) = \frac{q_i}{1 + D_i t}$$

Where:
- $$q_i$$ = fitted initial rate (an extrapolated curve intercept, **not** a measured rate)
- $$D_i$$ = nominal decline rate, per month in this project
- $$b$$ = dimensionless decline exponent
- $$t$$ = elapsed time in months

**How we fit it.** Bounded nonlinear least squares on **log-rate** residuals with a soft-L1 loss, so the fit is robust to outliers. Only strictly positive rates enter the fit — a decline curve models producing trend, not shut-ins. Time is kept in calendar months including gaps. The curve form is chosen by **AICc** (small-sample-corrected Akaike Information Criterion) using the training context only, never the holdout.

**There are two different Arps fits in this project — do not mix them up.**

**(a) The research comparison fit** — from `scripts/compare_arps_models.py`, used for the 21-month holdout on 15/9-F-12. AICc selected the **exponential** form:

| Parameter | Value |
|---|---:|
| $$q_i$$ | **7,712.95** bbl/d |
| $$D_i$$ | **0.04789** per month |
| $$b$$ | 0 (exponential) |
| Training points | 82 |
| AICc | −132.49 |

An annualised decline of about **43.7 % per year** ($$1 - e^{-0.04789 \times 12}$$). This is the fit quoted in the repo's `docs/volve_field_results.md` and in the metrics table above.

**(b) The dashboard's per-well fits** — each well is fitted separately over its own full history, and the curve form is chosen per well. These are the numbers the dashboard actually draws and the ones the trigger dates come from:

| Well | Form | $$q_i$$ (bbl/d) | $$D_i$$ /month | $$b$$ | Points | AICc |
|---|---|---:|---:|---:|---:|---:|
| 15/9-F-12 | hyperbolic | 6,884.66 | 0.05483 | 0.2088 | 102 | −111.52 |
| 15/9-F-14 | exponential | 5,531.40 | 0.03759 | 0 | 97 | −187.82 |
| 15/9-F-11 | exponential | 1,174.45 | 0.00983 | 0 | 39 | −51.35 |
| 15/9-F-1 C | harmonic | 951.19 | 0.40104 | 1 | 25 | −1.76 |
| 15/9-F-15 D | harmonic | 272.37 | 0.05083 | 1 | 30 | −56.26 |
| 15/9-F-5 | **no fit** | — | — | — | — | — |

Note that F-12 is **hyperbolic** in the dashboard but **exponential** in the research comparison — different fitting windows give different answers, which is itself a useful lesson about decline curves.

**Two warnings about $$q_i$$.** First, it is an **extrapolated intercept**, not a measured initial rate — F-12's fitted 6,884.66 bbl/d sits above its actual peak of 5,901.84 bbl/d. Second, fitting one smooth curve to a history full of shut-ins, restarts and interventions is a simplification. Treat $$q_i$$ as a curve-fitting artefact, not a reservoir property.

### 3.4 The economic limit

The **economic limit** is the production rate below which a well stops paying for itself — the revenue no longer covers operating costs, maintenance, and the cost of keeping it on stream. Below that rate you either shut the well in or install artificial lift.

**In this project the limit is set to 100 bbl/d.** It is a project assumption, not a universal number. Real economic limits depend on oil price, operating cost, tax, water-handling cost, and contract terms — and they change over time. If you are asked "why 100?", the honest answer is: *"It is a stated project assumption so the analysis is reproducible. A real study would derive it from the field's economics."*

### 3.5 Artificial lift, and how the method is chosen

When natural flow is no longer adequate, you add energy. The five main methods:

| Method | How it works | Best suited to | Main constraints |
|---|---|---|---|
| **Gas lift** | Inject gas into the tubing to lighten the fluid column | Deviated or high-rate wells; offshore; where gas is available | Needs **injection-gas supply** and compression. No gas = not feasible. |
| **ESP** (electric submersible pump) | Multi-stage centrifugal pump on a cable, set downhole | High liquid volumes; high water cut | Free gas at intake, solids, temperature, **power availability**, costly to pull on failure |
| **Sucker-rod pump** (SRP / beam lift) | Surface beam drives a rod string and downhole plunger | Onshore, moderate rates, rod-accessible wells | Depth and rod loading, deviation, **rod-service access** |
| **PCP** (progressing cavity pump) | Rotating elastomer or metal-to-metal positive-displacement pump | Viscous oil, sand, lower/moderate rates | Elastomer temperature and chemistry limits, torque, **surface drive** |
| **Hydraulic / jet** | Surface pump sends high-pressure power fluid downhole | Deep or deviated wells, compact surface footprint | **Power-fluid circuit**, separation capacity, erosion |

**How the choice is made — and how it is *not* made here.** Real lift selection is a nodal-analysis problem: you need IPR and VLP curves, an operating point, equipment envelopes, facility capacity, power and gas availability, intervention cost, and lifecycle economics. This project does **not** do that. What it does is a **deterministic rule-based screen** — a shortlist, not a recommendation.

The screen applies **feasibility gates first**, then qualitative support/caution rules. A method is labelled:

- **`eligible`** — passes the gates and has supporting evidence
- **`conditional`** — plausible, but a hard input is missing
- **`screened_out`** — a documented hard incompatibility
- **`unknown`** — insufficient information

**A missing hard constraint stays `unknown`, never "passed".** A high weighted score can never override a failed gate.

**The Volve result — and why it is unsatisfying.** For well 15/9-F-12 (liquid rate 503.8 bbl/d, water cut 88 %, moderate free gas):

| Method | Status | Supporting evidence | Missing input |
|---|---|---|---|
| Gas lift | `conditional` | Offshore: no downhole rotating unit | **Injection-gas supply** |
| Rod pump (SRP) | `conditional` | Low liquid duty | **Rod-pump access** |
| PCP | `conditional` | Positive displacement suits viscous/emulsified fluid | **Surface drive** |
| Hydraulic / jet | `conditional` | Offshore: compact surface footprint | **Power-fluid circuit** |
| ESP | `conditional` | Water cut 88 % suits high-volume lift | **Power availability** (caution: free gas at intake) |
| Plunger lift | `screened_out` | — | Blocked: oil well, gas-well branch only |

**Every single candidate is `conditional`.** That is the correct answer given the data, and it is the single most important limitation of the lift screen: the Volve workbook does not record whether gas compression, electrical power, or rod service was available. Without those three facts, no method can be confirmed. The screen is honest about it rather than inventing a winner.

---

## 4. Solution gas drive

### 4.1 The mechanism

Oil at reservoir pressure holds dissolved gas, like carbonated drink under pressure. The pressure at which gas starts to come out of solution is the **bubble point**.

**Solution gas drive** (also called depletion drive or internal gas drive) works in three stages:

1. **Above the bubble point.** Reservoir pressure is above $$P_b$$. All gas stays dissolved. Oil expands slightly as pressure falls — this is weak, because oil is nearly incompressible. Recovery is poor at this stage.
2. **At the bubble point.** Pressure reaches $$P_b$$. The first bubbles form.
3. **Below the bubble point.** Gas comes out of solution and **expands dramatically** as pressure continues to fall. Gas is highly compressible, so a small pressure drop releases a large volume. This expanding gas pushes oil toward the wellbore. This is where most of the energy comes from.

### 4.2 Why it matters

Solution gas drive is the **most common primary drive mechanism** in oil reservoirs, and it is typically **inefficient** — recovery factors are often only 5–30 %, because the gas quickly becomes the mobile phase and channels through the oil. It also produces a characteristic signature: **rising GOR** as pressure falls below $$P_b$$, followed by falling oil rate.

### 4.3 How it maps to the 3D view

The 3D reservoir panel is a **schematic driven by the selected well's own data** — not a reservoir simulator. It is a 2D canvas projection (not WebGL), and the layer sizes are scaled from water cut and the ratio of flowing bottom-hole pressure to bubble point, not solved from flow equations.

What it shows:

- **Gas cap (amber)** — grows as pressure falls below the bubble point, with **bubbles rising** through the oil. The base of the gas cap *is* the bubble-point plane.
- **Oil zone (green)** — with droplets migrating toward the wellbore.
- **Water zone (violet)** — grows with water cut, with influx arrows at the base.
- **Bubble-point plane** — a translucent surface with a marker where it crosses the wellbore.
- **Perforated interval** — a lighter casing segment with tick marks at the flowing bottom-hole pressure, plus animated inflow arrows when producing.
- **Live HUD** — free gas %, oil rate, water cut.

**The vertical axis is pressure, not depth.** High pressure at the bottom, low at the top, scaled to each well's own BHP and $$P_b$$. This is the only way the plane stays inside the block when the two are far apart.

**Two contrasting cases, both correct:**

- **Synthetic solution-gas-drive well:** BHP 1,059 psi, $$P_b$$ 2,499 psi. BHP is *below* bubble point, so a **31.7 % gas cap** forms and the label reads **"solution gas drive"**. This is the mechanism in action.
- **Volve 15/9-F-12:** BHP 273 bar, $$P_b$$ 58 bar. BHP is **4.7× above** bubble point, so there is **no free gas** — 0.0 % gas cap — and the label reads **"undersaturated · water drive"**. That is the physics, not a missing feature.

**Say this plainly if asked:** *"The 3D view is a schematic, not a simulator. It shows the direction the reservoir is moving, not absolute volumes. On Volve there is no gas cap because the pressure is still well above bubble point — the field is water-injection supported, not solution-gas driven."*

---

## 5. Bubble points and dew points

### 5.1 Bubble point — physically what it is

The **bubble point pressure** $$P_b$$ is the pressure at which the **first bubble of gas** comes out of solution as pressure is lowered, at a fixed temperature. Below $$P_b$$ the oil is **undersaturated** (single phase); above it, two phases coexist.

It is a property of the **oil and its dissolved gas at a given temperature** — not of the well or the rock. Two wells producing the same fluid have the same $$P_b$$.

**Why it matters:**
- It marks the pressure below which free gas appears — the start of solution gas drive.
- It sets the pressure at which GOR begins to rise.
- For pump selection, it tells you whether you will have **free gas at the pump intake** — a major ESP constraint.
- It is the reference for the `bhp_over_pb` and `p_over_pb` features in the ML model.

### 5.2 Dew point — physically what it is

The **dew point pressure** $$P_d$$ is the mirror image, for **gas-condensate** systems. As pressure is *lowered* through $$P_d$$, liquid **condensate** drops out of the gas phase. Below the dew point you get a two-phase system where liquid is forming in the reservoir — and that liquid is often **immobile**, so it is lost. This is called **condensate banking** and it damages productivity.

**Bubble point is for oil (gas comes out). Dew point is for gas (liquid comes out).** A black-oil reservoir has a bubble point. A gas-condensate reservoir has a dew point. Some volatile-oil systems have both.

### 5.3 How they are estimated — the Standing correlation

When no laboratory PVT report exists, the bubble point can be **estimated** from production data using **Standing's correlation** (1947):

$$P_b = 18.2 \left[ \left(\frac{R_s}{\gamma_g}\right)^{0.83} 10^{\,0.00091\,T - 0.0125\,\text{API}} - 1.4 \right]$$

Where:
- $$R_s$$ = solution gas-oil ratio, scf/bbl
- $$\gamma_g$$ = gas specific gravity (air = 1)
- $$T$$ = temperature, °F
- $$\text{API}$$ = oil gravity in degrees API
- $$P_b$$ = bubble point, **psia**

This is exactly what the dashboard implements in its `standingPb()` function.

**The critical caveat.** Standing's correlation needs $$R_s$$, $$\gamma_g$$, $$T$$ and API. The Volve workbook has **none of these as PVT properties** — it has producing GOR, which is *not* the same as solution GOR once free gas is being produced. So the dashboard's PVT estimate uses **assumed** values: **API 35, gas SG 0.8**. The button is labelled "PVT est." and the tooltip says so. **These are assumptions, not measurements.** If you are asked where the bubble point comes from, the answer is: *"Standing's correlation with assumed fluid properties, because Volve has no PVT report. It is an estimate for illustration, not a measured value."*

### 5.4 Why dew point does not apply to Volve

**Volve is a black-oil field.** It produces conventional crude with a dissolved-gas bubble point. There is no gas-condensate column, so there is no dew point to report.

The dashboard states this explicitly in its data-quality panel:

> `dew_point: "not present in Volve source"`

and for the synthetic set:

> `dew_point: "not present in either dataset"`

**The dew-point series exists in the code** — the PT chart will draw a violet dashed dew-point line **if** an uploaded dataset contains a `dew_point_pressure` column. It is wired up and tested with a gas-condensate test file, but **no dataset in this project has that column**, so on Volve the line never appears. That is correct behaviour, not a bug.

**Say this plainly if asked:** *"Dew point does not apply to Volve because it is a black-oil reservoir. The chart supports dew point for gas-condensate data, but we have none, so the line is absent."*

---

## 6. Why some datasets lack important variables

This is the heart of the project's limitations. Each missing variable removes a specific analysis.

### 6.1 Volve — what is missing and what it costs

| Variable | Status | Consequence |
|---|---|---|
| **Well ID** | ✅ Present (6 wells) | Full per-well analysis possible |
| **Dates** | ✅ Present, daily | Time-series, decline curves, forecasting all possible |
| **Oil / gas / water rates** | ✅ Present, 0 % missing | Full production analysis |
| **BHP (downhole pressure)** | ⚠️ **27.1 % missing** | Pressure analysis is patchy. F-12's last survey is **2010-10** but it produced until **2016-09** — six years with no pressure reading. The 3D view carries the last known value forward and labels it "(last survey)". |
| **Downhole temperature** | ⚠️ **27.1 % missing** | Cannot compute reliable PVT from measured temperature |
| **Wellhead pressure** | ⚠️ 3.1 % missing | Minor |
| **Choke** | ⚠️ 2.6 % missing | Minor |
| **On-stream hours** | ✅ Present, 0 % missing | Lets us separate downtime from decline |
| **GOR** | ✅ Derivable from gas/oil | Present, but producing GOR ≠ solution GOR below $$P_b$$ |
| **Water cut** | ✅ Derivable from water/oil | Present and complete |
| **Bubble point** | ❌ **Absent** | Must be estimated by Standing with assumed fluid properties |
| **Dew point** | ❌ **Absent** | Not applicable (black oil) |
| **Lift status** | ❌ **Absent** | **The single biggest gap** — see below |
| **Reservoir pressure** | ❌ Absent | Only flowing BHP available; cannot compute drawdown properly |

### 6.2 The lift-label gap — the project's central problem

**The Volve workbook does not record whether each well was flowing naturally or on artificial lift at any given time.** There is no `lift_active` column, no lift type, no transition date.

**What this costs, concretely:**

1. **The classifier cannot be trained on Volve.** It is trained on synthetic data instead. So its metrics describe synthetic wells, not the North Sea.
2. **"When does natural flow become inadequate?" cannot be answered from labels.** It can only be *inferred* from a decline curve crossing a threshold — which is a trend extrapolation, not a prediction of a real event.
3. **The lift screen cannot be validated.** We can say "gas lift is conditional", but we cannot check whether gas lift was actually installed, or whether it worked.
4. **The 3D view cannot show a real transition.** It shows the reservoir state, not the moment a pump went in.

**This is why the dashboard's decline predictions are labelled as extrapolations.** The trigger dates come from the repo's own stored Arps fit, not from observed events.

### 6.3 The water-injection complication

Volve was produced with **field-level water injection for pressure support**. This means:

- Pressure decline is **not** purely depletion — injection is propping it up.
- The decline in oil rate mixes **depletion** with **rising water cut** and **operational decisions**.
- Volve is therefore **not a clean solution-gas-drive dataset**, even though solution gas drive is the project's stated focus.

**Say this plainly if asked:** *"Volve is water-injection supported, so its decline is not pure depletion. That is one reason we cannot treat it as a solution-gas-drive validation set."*

### 6.4 The synthetic dataset — what it has and what that is worth

| Variable | Status |
|---|---|
| Wells | 36 |
| Rows | 2,592 (72 months each) |
| Missing values | **0 % in every column** |
| Reservoir pressure | ✅ Present |
| BHP | ✅ Present |
| Bubble point | ✅ Present |
| Initial reservoir pressure | ✅ Present |
| Choke | ✅ Present |
| **Lift labels** | ✅ **Present** (`lift_active`) |
| Dew point | ❌ Absent |

**The synthetic set is the only place lift labels exist.** That is why the classifier is trained there. But it is **generated from simplified formulas, not a reservoir simulator**, and it is **not field evidence**. Its metrics must never be presented as field accuracy.

### 6.5 What each gap would take to fix

| Gap | What would fix it |
|---|---|
| No lift status | Dated well-event records: workover reports, lift installation dates, ESP run-life logs |
| No PVT | A laboratory PVT report, or at minimum measured $$R_s$$, $$\gamma_g$$, API and temperature |
| Sparse BHP | More frequent pressure surveys, or a permanent downhole gauge |
| No reservoir pressure | Static pressure surveys (shut-in), or a material-balance study |
| No dew point | Gas-condensate PVT — only relevant if a condensate well is added |
| Water-injection masking decline | Injection and voidage data, so pressure support can be separated from depletion |

---

## 7. Every chart in the dashboard, explained

The dashboard has **six KPI cards**, **six main charts**, **three rail panels** (decline prediction, PDF report, 3D reservoir) and a **credits panel**. Here is each one.

### 7.1 The KPI strip (six cards)

| Card | What it shows | How to read it |
|---|---|---|
| **Wells** | 6, with "9,143 daily rows" underneath | Scope of the dataset |
| **Lift triggers** | How many wells cross 100 bbl/d on the Arps fit, out of 6 | **4 of 6** on Volve |
| **Earliest window** | The earliest trigger date and which well | **2016-05**, well 15/9-F-1 C |
| **Field oil now** | Oil rate in the last *producing* month | Not the last calendar month — the final month is often a shut-in |
| **Water cut** | Water cut at that same month | **84.5 %** |
| **On-stream** | Fraction of days the field was producing, plus zero-rate day count | **84.1 %**, 1,144 zero days |

### 7.2 Lift window (`cLift`) — the main decline chart

**What it shows.** One well's oil rate over time, with the Arps decline curve fitted through it, the forward forecast, the 100 bbl/d economic limit, and a marker where the forecast crosses that limit.

**Axes.** X = date (monthly). Y = oil rate, bbl/d.

**Datasets on the chart:**
- **Oil rate** — the actual observed monthly rate
- **Arps fit** — the fitted decline curve over the history
- **Forecast** — the Arps curve extended forward
- **Economic limit** — a horizontal line at 100 bbl/d
- **Trigger** — a point where the forecast crosses the limit

**How to read it.** Where the blue observed line sits above the dashed limit, the well is economic. Where the forecast crosses the limit, that is the predicted month natural flow becomes inadequate.

**Conclusion to draw.** *"Four of six Volve wells cross the 100 bbl/d limit within the forecast horizon. The earliest is 15/9-F-1 C in May 2016 — already in the past. 15/9-F-12 crosses in July 2018."*

**Caveat.** The fit is a smooth curve through a history full of shut-ins. It does not model restarts, interventions, or water injection.

### 7.3 Lift screen (rule-based cards)

**What it shows.** Six lift methods, each with a status badge (`conditional`, `screened_out`, `eligible`, `unknown`), supporting evidence, cautions, and **missing inputs**.

**How to read it.** Green/amber/red status tells you whether the method is plausible. The "missing" line tells you what fact would settle it.

**Conclusion to draw.** *"All five oil-well methods are conditional because gas supply, power and rod access are unknown. Plunger lift is screened out because it is a gas-well method."*

### 7.4 PT — pressure and temperature (`cPT`)

**What it shows.** Pressure against time for the selected well.

- **On Volve:** measured flowing BHP, wellhead pressure, and a bubble-point guide line (from Standing, with assumed fluid properties).
- **On synthetic:** reservoir pressure, flowing BHP, bubble point, and dew point if present.

**Axes.** X = date. Y = pressure (bar on Volve, psi on synthetic — the axis is unit-aware).

**How to read it.** Watch the gap between reservoir pressure and flowing BHP — that gap is the drawdown, the energy actually being used to produce. Watch where BHP falls relative to the bubble-point line: crossing it means free gas starts appearing.

**Conclusion to draw.** *"On 15/9-F-12, BHP stays well above the bubble point throughout — the well is undersaturated, so there is no free gas and no solution-gas-drive energy."*

**Caveat.** BHP is 27.1 % missing. The line has gaps, and the last survey for F-12 is 2010.

### 7.5 Pressure state (`cPS`)

**What it shows.** A focused view of the pressure state relative to bubble point.

- **On Volve:** pressure state — where BHP sits against $$P_b$$.
- **On synthetic:** water cut and GOR on dual axes.

**Axes.** X = date. Y = pressure (Volve) or water cut % / GOR scf/bbl (synthetic, dual axis).

**How to read it.** If BHP is above $$P_b$$, the well is undersaturated — single-phase oil, no free gas. Below $$P_b$$, gas is breaking out.

**Conclusion to draw.** *"Volve wells are undersaturated. The synthetic solution-gas-drive wells are not — that is the contrast the 3D view visualises."*

### 7.6 Model quality — ROC curve (`cRoc`)

**What it shows.** The **receiver operating characteristic** curve for the lift classifier, plus the chance diagonal.

**Axes.** X = false positive rate (0→1). Y = true positive rate (0→1).

**How to read it.** The further the curve bows toward the top-left corner, the better the model ranks risky wells above safe ones. The dashed diagonal is a coin flip.

**Conclusion to draw.** *"AUC 0.793 — the ranking is genuinely better than chance. But the operating threshold is badly set: precision 18.2 %, recall 24.0 %. The model finds only 6 of 25 real lift events."*

**Caveat.** This is **synthetic** data. It says nothing about North Sea wells.

### 7.7 Feature importance (`imp`) — bar chart

**What it shows.** ΔAUC for each of the twelve features — how much ranking performance is lost when that feature is shuffled.

**Axes.** Horizontal bars, centred on zero. Positive = useful. Negative = noise.

**How to read it.** Longest bar to the right is the most important feature.

**Conclusion to draw.** *"Water cut dominates at +0.188 — three times the next feature. Six of twelve features are negative, meaning they add noise. A leaner model would likely do better."*

### 7.8 Data quality (`dq`) — table

**What it shows.** Per-column missingness for the active dataset, plus duplicate count, zero-rate count, and the status of lift labels, bubble point and dew point.

**How to read it.** Any column with a non-zero percentage has gaps you must account for.

**Conclusion to draw.** *"Volve: 0 duplicates, 1,144 zero-rate days, BHP and temperature 27.1 % missing, no lift status, no bubble point, no dew point. The data is clean where it exists and absent where it matters."*

### 7.9 Water cut & GOR (`cWC`) — dual axis

**What it shows.** Field water cut (left axis, %) and field GOR (right axis, scf/bbl) over the whole field's life.

**Axes.** X = date. Left Y = water cut %. Right Y = GOR scf/bbl.

**How to read it.** Water cut climbing steeply is the classic water-breakthrough signature. GOR staying flat means the reservoir is not depleting below bubble point.

**Conclusion to draw.** *"Water cut went from 0.83 % to 84.46 % while GOR stayed near 147 scf/bbl. That is water encroachment, not gas breakout — consistent with water-injection pressure support."*

### 7.10 Well ranking (`rankTbl`) — table

**What it shows.** All wells ranked, with oil, liquid rate, water cut, GOR and trigger date.

**Columns (Volve):** Well · Oil (bbl) · Liq b/d · WC · GOR · Trigger
**Columns (synthetic):** Well · Type · Liq b/d · WC · Pred · Actual

**How to read it.** Sort by oil to see contribution. Sort by trigger to see urgency.

**Conclusion to draw.** *"15/9-F-12 and 15/9-F-14 together produce 84.9 % of the field's oil. 15/9-F-1 C already passed its trigger in May 2016."*

### 7.11 Decline prediction panel (`cPred`)

**What it shows.** A dedicated panel for one well: oil history, Arps fit, forward forecast, the 100 bbl/d limit, and a **red diamond** marking the month natural flow becomes inadequate. A **log-scale toggle** is on by default so the tail and the limit are readable against a 5,902 bbl/d peak.

**Axes.** X = date. Y = oil rate, bbl/d (log or linear).

**How to read it.** The red diamond is the answer to "when?". The table below ranks all six wells by trigger date.

**The trigger dates (from the repo's stored Arps fit):**

| Well | Trigger | Rate at trigger |
|---|---|---:|
| 15/9-F-1 C | **2016-05-01** | 86.26 bbl/d |
| 15/9-F-15 D | **2016-11-01** | 99.83 bbl/d |
| 15/9-F-14 | **2017-06-01** | 99.09 bbl/d |
| 15/9-F-12 | **2018-07-01** | 97.87 bbl/d |
| 15/9-F-11 | No trigger | — |
| 15/9-F-5 | No trigger | — |

**Conclusion to draw.** *"Two wells have already crossed the limit, two will within the next two years, and two have no Arps fit — F-5 has only 144 days of history, too short to fit a decline curve."*

**Caveat.** These are **trend extrapolations**, not observed events. The repo's stored value is used as the source of truth; a re-derived value disagreed for F-1 C (2015-12 vs 2016-05) and the repo's number was kept.

### 7.12 Analysis report panel (`tRep`)

**What it shows.** A preview of a five-section report, and a button that builds a **real multi-page A4 PDF in the browser** using jsPDF (MIT licence, vendored locally — no CDN, no API key, no cost). Nothing is uploaded.

**Sections:** Scope · Objective · Decline · Field oil · Water cut · Lift screen · Data quality · Limitations.

**How to read it.** It is a snapshot of the current filter state — change the well or date range and the report changes.

### 7.13 3D reservoir panel (`c3d`)

**What it shows.** An animated 3D block driven by the selected well's data. Drag to rotate, wheel to zoom, scrub the timeline or press Play.

**Elements:** gas cap (amber, with rising bubbles) · oil zone (green, droplets migrating to the wellbore) · water zone (violet, influx arrows) · bubble-point plane (translucent, with a marker where it crosses the wellbore) · perforated interval (tick marks at flowing BHP) · live HUD (free gas %, oil rate, water cut).

**How to read it.** The vertical axis is **pressure**, not depth. The plane is the bubble point. If the perforations sit **above** the plane in pressure terms, there is no free gas. If they sit **below**, gas is breaking out and the gas cap grows.

**Conclusion to draw.** *"On the synthetic solution-gas-drive well, BHP 1,059 psi is below $$P_b$$ 2,499 psi, so a 31.7 % gas cap forms — that is solution gas drive. On Volve F-12, BHP 273 bar is 4.7× above $$P_b$$ 58 bar, so there is no gas cap at all — the field is undersaturated and water-driven."*

**Caveat.** Schematic, not a simulator. 2D canvas projection, not WebGL. Layer sizes scaled from water cut and BHP/$$P_b$$, not solved from flow equations.

### 7.14 Credits panel (`tCred`)

UMaT crest, copyright notice, the five group members with index numbers, and the GitHub link.

---

## 8. Presentation cheat sheet

Short answers to the questions you are most likely to be asked.

**"What does the dashboard actually predict?"**
It forecasts oil rate and estimates the month a well's decline curve crosses 100 bbl/d. It also screens lift methods. It does **not** predict a real lift-transition event, because no dataset has those labels.

**"How accurate is your model?"**
For regression, R² is **negative** — the best model (HGB, MAE 207.11) is worse than predicting the mean. For the classifier, AUC is **0.793** but recall is only **24 %** — it misses 19 of 25 events. We report both honestly.

**"Why is your accuracy 92.9 % but you say it's bad?"**
Because only 3.9 % of cases are positive. Always predicting "no lift" would score 96.1 %. Our 92.9 % is *worse* than doing nothing.

**"Why not use a neural network?"**
With 6 wells and 9,143 rows, a deep model would overfit. Gradient boosting is the right complexity for this data size. The bottleneck is data, not model architecture.

**"Why is Arps still in there if ML is better?"**
Arps is not better — it has the third-lowest MAE. It is in there because it is **interpretable**: an engineer can read $$q_i$$ and $$D_i$$ and sanity-check them. ML gives a number with no physical meaning.

**"What is the bubble point and where did you get it?"**
The pressure at which gas first comes out of solution. Volve has no PVT report, so we estimate it with **Standing's correlation** using **assumed** API 35 and gas SG 0.8. It is an estimate, not a measurement.

**"Why is there no dew point?"**
Volve is a black-oil reservoir — it has a bubble point, not a dew point. The chart supports dew point for gas-condensate data, but we have none.

**"Why does the 3D show no gas cap on Volve?"**
Because BHP (273 bar) is 4.7× above the bubble point (58 bar). The oil is undersaturated, so no free gas exists. That is correct physics.

**"Why are all lift methods 'conditional'?"**
Because the Volve workbook does not record whether injection gas, electrical power, or rod service was available. Without those three facts, no method can be confirmed. We leave them unknown rather than guessing.

**"What is the biggest weakness?"**
No dated lift-transition labels. Everything about "when does natural flow fail" is a decline-curve extrapolation, not an observed event.

**"What would you do next?"**
Get dated lift-status records per well. That single dataset would let us train a real transition classifier, validate the lift screen, and turn the 3D view from a snapshot into a prediction.

---

## 9. What this project does not do

Being explicit about this is part of the deliverable.

- ❌ It does **not** confirm that natural flow is inadequate. A falling rate is not proof.
- ❌ It does **not** solve IPR/VLP/nodal analysis. There is no operating-point calculation.
- ❌ It does **not** recommend a lift method. It produces a **shortlist** for engineering review.
- ❌ It does **not** size equipment, design installations, or estimate cost.
- ❌ It does **not** predict a real lift-transition date. Trigger dates are trend extrapolations.
- ❌ It does **not** simulate the reservoir. The 3D view is a schematic.
- ❌ It does **not** claim field accuracy. The classifier is trained on synthetic data.
- ❌ It does **not** infer drive mechanism from a pressure/bubble-point comparison alone.

**Every one of these is a deliberate boundary, not an oversight.** The repo's `TODO.md` tracks them as open work.

---

## 10. Data, licensing and credits

### Volve field data

Provided by **Equinor** and former Volve licence partners, shared under the [Equinor terms and conditions for use of Volve data](https://www.equinor.com/content/dam/statoil/documents/what-we-do/Equinor-HRS-Terms-and-conditions-for-licence-to-data-Volve.pdf) — based on CC BY 4.0 with an additional no-sale condition.

Source workbook mirror: [f0nzie/volve_eclipse_reservoir](https://github.com/f0nzie/volve_eclipse_reservoir), file `inst/rawdata/Volve production data.xlsx`.

The processed file `data/processed/volve_daily_producers.csv` is an adapted table from the workbook's *Daily Production Data* sheet, retaining only `WELL_TYPE == OP` producer rows. It keeps source units and does not fill missing observations.

### Software licences

| Component | Licence |
|---|---|
| Chart.js | MIT |
| chartjs-plugin-zoom | MIT |
| Hammer.js | MIT |
| jsPDF | MIT |
| Firebase JS SDK | Apache 2.0 |
| scikit-learn | BSD-3-Clause |

### The group

**Petroleum Engineering, University of Mines and Technology (UMaT), Tarkwa, Ghana**

| Name | Index number |
|---|---|
| Precious Agyei | SPE.41.009.025.23 |
| Francis Atiimbik Adiakaasi | SPE.41.009.018.23 |
| Randy Akwesi Nti | SPE.41.009.101.23 |
| Ursula Esi Baffoe | SPE.41.009.060.23 |
| Richard Abekah | SPE.41.009.007.23 |

### Copyright

© 2026 Group Project ML — Petroleum Engineering students, University of Mines and Technology (UMaT), Tarkwa, Ghana. **All rights reserved** for the dashboard, models and analysis.

The Volve dataset remains Equinor's, under its own open-data licence. Third-party libraries remain under their own licences.

### Links

- **Live dashboard:** https://static.teamily.ai/sites/4c4359e0-0d04-4bf2-950a-1f5ac8b27922/webpages/group-project-ml-dashboard_v10/index.html
- **ML source repository:** https://github.com/Wizzy-pixel-ui/GROUP-project-ML.git
- **This repository:** https://github.com/Wizzy-pixel-ui/group-project-ml-dashboard

---

*Every number in this README was read from the project's own data files and reports — `reports/arps_comparison/arps_model_comparison_metrics.csv`, `docs/volve_field_results.md`, and the dashboard's `data/dashboard_data.js`. Nothing is estimated or invented.*
