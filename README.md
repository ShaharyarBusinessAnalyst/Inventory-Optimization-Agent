# Inventory Optimization Agent — Multi-Product Newsvendor

A production-grade inventory optimization pipeline for multi-SKU retail planning, combining **newsvendor theory**, **Sample Average Approximation (SAA)**, and **LLM-generated plain-language explanations** to determine optimal weekly order quantities under budget, storage, and service-level constraints.

Applied to the M5 Walmart Forecasting dataset — 15 SKUs from the HOUSEHOLD_1 department at store CA_1.

---

## Problem

A category manager must commit to weekly order quantities $Q_i$ for each SKU *before* demand is observed. The agent maximises expected profit subject to:

- **Budget constraint** — weekly procurement spend ≤ $3,500 (supplier cost basis)
- **Storage constraint** — total units ordered ≤ 1,200
- **Fill rate floor** — minimum service level per SKU

---

## Architecture

```
M5 Dataset (Walmart sales + prices + calendar)
        │
        ▼
Section 2: Data Pipeline
        │  Filter → CA_1 / HOUSEHOLD_1
        │  Aggregate daily → weekly demand
        │  Activity filter (≥26 non-zero weeks)
        │  Top-15 SKUs by mean training demand
        │
        ▼
Section 3: Distribution Fitting
        │  AIC selection per SKU: Normal / Negative-Binomial / Poisson
        │  Bootstrap 90% CI on mean demand
        │
        ▼
Section 4: Constraint Engine
        │  Declarative CONFIG → scipy constraint dictionaries
        │  Supports: budget (cost basis), storage, fill rate, per-SKU bounds
        │
        ▼
Section 5: SAA Optimisation (SLSQP)
        │  1,000 Monte Carlo scenarios per SKU
        │  Warm start: Q₀ = F⁻¹(CR) per SKU
        │  Validation gate: ≤1% deviation from closed-form Q* (unconstrained)
        │  KKT verification: primal feasibility, dual feasibility,
        │                    complementary slackness, stationarity
        │
        ▼
Section 6–9: Analysis
        │  Constraint audit + shadow prices
        │  Out-of-sample validation (weeks 79–104 hold-out)
        │  Sensitivity analysis (budget, storage, fill rate, stockout penalty)
        │
        ▼
Section 10: LLM Explanation Agent (Claude API)
        │  Translates optimisation output into plain-language recommendations
        │  Structured JSON schema → stakeholder-ready narrative
        │
        ▼
Section 11: Task 2 — Case-Pack Discount Models
           Three purchasing models compared:
           Model A: No discount (SLSQP baseline)
           Model B: All-units two-tier discount, discrete grid
           Model C: Hybrid — discount applied to Model A quantities
```

---

## Newsvendor Formulation

**Per-SKU cost structure:**

$$c_{u,i} = p_i - c_i + s_i \quad \text{(underage: lost margin + goodwill)}$$

$$c_{o,i} = c_i(1 - \text{salvage\_rate}) + h_i \quad \text{(overage: net procurement + holding)}$$

$$CR_i = \frac{c_{u,i}}{c_{u,i} + c_{o,i}} \approx 0.59 \text{ for HOUSEHOLD\_1}$$

**Unconstrained optimum:** $Q^*_i = F_i^{-1}(CR_i)$

**SAA objective (multi-SKU, constrained):**

$$\max_{Q \ge 0} \; \frac{1}{S}\sum_{s=1}^S \sum_{i=1}^n \pi_i\!\left(Q_i, D_i^{(s)}\right) \quad \text{s.t. constraints}$$

**Realised profit (corrected formulation):**

$$\pi_i(Q_i, D_i) = p_i \min(D_i, Q_i) - c_i Q_i - h_i(Q_i - D_i)^+$$

---

## Key Results

| Metric | Value |
|---|---|
| SKUs modelled | 15 (HOUSEHOLD_1, CA_1) |
| Demand distributions | Normal, Negative-Binomial, Poisson (AIC-selected per SKU) |
| Optimisation scenarios | 1,000 Monte Carlo |
| Solver | SLSQP (scipy.optimize) |
| Validation gate | ≤1% deviation from closed-form Q* |
| KKT conditions | All 4 verified |
| Model B/C discount savings | Reported in Section 11 |

---

## Quick Start

```bash
# Install dependencies
pip install numpy pandas scipy matplotlib anthropic

# Upload M5 dataset files to ./
# Required: sales_train_evaluation.csv, sell_prices.csv, calendar.csv
# Available at: https://www.kaggle.com/competitions/m5-forecasting-accuracy/data

# Set Anthropic API key for LLM explanation agent (Section 10)
export ANTHROPIC_API_KEY=sk-ant-...

# Run in Jupyter
jupyter lab inventory_agent_clean.ipynb
```

---

## Project Structure

```
├── inventory_agent_clean.ipynb    # Full pipeline notebook
└── README.md
```

> **Data note:** M5 dataset files are not included (Kaggle competition data). Download from
> [kaggle.com/competitions/m5-forecasting-accuracy](https://www.kaggle.com/competitions/m5-forecasting-accuracy/data)
> and place in the same directory as the notebook.

---

## Tech Stack

| Component | Technology |
|---|---|
| Optimisation | SLSQP (scipy.optimize.minimize) |
| Distribution fitting | scipy.stats (Normal, NB, Poisson) + AIC |
| SAA | NumPy Monte Carlo |
| LLM explanations | Anthropic Claude API |
| Data | M5 Walmart Forecasting Dataset |
| Language | Python 3 |
