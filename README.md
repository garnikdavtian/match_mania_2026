# 🏀 March Machine Learning Mania 2026 — Silver Medal Solution (29th Place)
**Probabilistic Forecasting of NCAA Division I Basketball Tournament Outcomes via Multi-Signal Ensemble Learning**
[![Kaggle Competition](https://img.shields.io/badge/Kaggle-Competition-20BEFF?logo=kaggle)](https://www.kaggle.com/competitions/march-machine-learning-mania-2026)
[![Brier Score](https://img.shields.io/badge/Brier%20Score-0.1209-brightgreen)]()
[![Placement](https://img.shields.io/badge/Placement-29th%20%2F%203400%2B-silver)]()
[![Medal](https://img.shields.io/badge/Medal-🥈%20Silver-c0c0c0)]()
---
## Table of Contents
1. [Competition Overview](#1-competition-overview)
2. [Solution Architecture](#2-solution-architecture)
3. [Feature Engineering Pipeline](#3-feature-engineering-pipeline)
4. [Rating Systems](#4-rating-systems)
5. [Modeling Strategy](#5-modeling-strategy)
6. [Ensemble & Calibration](#6-ensemble--calibration)
7. [Cross-Validation Framework](#7-cross-validation-framework)
8. [Results & Analysis](#8-results--analysis)
9. [Repository Structure](#9-repository-structure)
10. [Reproducibility](#10-reproducibility)
11. [References](#11-references)
---
## 1. Competition Overview
### 1.1 Problem Statement
The [March Machine Learning Mania 2026](https://www.kaggle.com/competitions/march-machine-learning-mania-2026) competition challenges participants to predict the outcomes of all possible matchups in both the **Men's** and **Women's NCAA Division I Basketball Tournaments**. For each hypothetical game between team $A$ (lower ID) and team $B$ (higher ID), the task is to predict $P(\text{Team } A \text{ wins})$.
### 1.2 Evaluation Metric
Submissions are evaluated using the **Brier Score**, defined as the mean squared error between predicted probabilities and actual binary outcomes:
$$\text{Brier Score} = \frac{1}{N} \sum_{i=1}^{N} (f_i - o_i)^2$$
where $f_i \in [0, 1]$ is the predicted probability and $o_i \in \{0, 1\}$ is the realized outcome. A lower Brier score indicates superior probabilistic calibration. Notably, a naïve baseline of $f_i = 0.5$ yields a Brier score of 0.25.
### 1.3 Competition Format
| Aspect | Detail |
|--------|--------|
| **Tournaments** | Men's (68 teams) + Women's (68 teams) |
| **Prediction scope** | All $\binom{68}{2} = 2{,}278$ pairwise matchups per tournament |
| **Historical data** | Regular season & tournament results (1985–2025 Men's, 1998–2025 Women's) |
| **Auxiliary data** | Box scores, Massey Ordinals, conference affiliations, coaching records, game locations |
| **Metric** | Brier Score (lower is better) |
### 1.4 Final Result
| Metric | Value |
|--------|-------|
| **Private Leaderboard Score** | **0.1208707** |
| **Placement** | **29th**  |
| **Medal** | 🥈 **Silver** |
---
## 2. Solution Architecture
The solution implements a **three-stage pipeline** combining domain-driven feature engineering with a heterogeneous model ensemble:
```
┌─────────────────────────────────────────────────────────────────┐
│                    DATA INGESTION LAYER                         │
│  Regular Season Compact ─── Regular Season Detailed ─── Seeds   │
│  Conference Affiliations ── Massey Ordinals (Men's only)        │
└──────────────────────────────┬──────────────────────────────────┘
                               │
                               ▼
┌────────────────────────────────────────────────────────────────┐
│               FEATURE ENGINEERING LAYER                        │
│                                                                │
│  ┌──────────────┐ ┌──────────────┐ ┌────────────────────────┐  │
│  │  Elo Ratings │ │  SRS Ratings │ │ Box-Score Efficiency   │  │
│  │  (w/ HCA,    │ │  (Iterative  │ │ (OffEff, DefEff,       │  │
│  │  MOV-adj,    │ │  Strength of │ │  eFG%, TOV%, ORB%,     │  │
│  │  regression) │ │  Schedule)   │ │  FTR, 3PAr, etc.)      │  │
│  └──────────────┘ └──────────────┘ └────────────────────────┘  │
│                                                                │
│  ┌──────────────┐ ┌──────────────┐ ┌────────────────────────┐  │
│  │  Recent Form │ │  Conference  │ │  Massey Composite      │  │
│  │  (Last 10    │ │  Strength    │ │  (Normalized rank      │  │
│  │  games)      │ │  (Conf SRS)  │ │  aggregation)          │  │
│  └──────────────┘ └──────────────┘ └────────────────────────┘  │
└──────────────────────────────┬─────────────────────────────────┘
                               │ Pairwise Differentials
                               ▼
┌────────────────────────────────────────────────────────────────┐
│                   MODELING LAYER                               │
│                                                                │
│  ┌────────────────────┐  ┌─────────────────────────────────┐   │
│  │ Logistic Regression│  │ Histogram Gradient Boosting     │   │
│  │ (L2, C=1.0)        │  │ (depth=3, lr=0.05, 300 iter)    │   │
│  └────────┬───────────┘  └────────────┬────────────────────┘   │
│           │                           │                        │
│           ▼                           ▼                        │
│  ┌─────────────────────────────────────────────────────────┐   │
│  │            Grid-Searched Weighted Ensemble              │   │
│  │     P = w₁·P_logit + w₂·P_hgb + w₃·P_elo                │   │
│  │     (Optimal weights via Brier Score minimization)      │   │
│  └─────────────────────┬───────────────────────────────────┘   │
│                        │                                       │
└────────────────────────┼───────────────────────────────────────┘
                         │ Clipped to [0.01, 0.99]
                         ▼
                  ┌───────────────┐
                  │ submission.csv│
                  └───────────────┘
```
### 2.1 Key Design Decisions
1. **Symmetrized matchup features**: All features are computed as $\Delta = \text{Feature}_A - \text{Feature}_B$, ensuring the model captures *relative* team strength rather than absolute statistics.
2. **Three complementary predictive signals**: The ensemble combines a parametric model (Logistic Regression), a non-parametric model (Histogram Gradient Boosting), and a pure rating system (Elo), each contributing orthogonal information.
3. **Separate Men's/Women's pipelines**: Independent feature sets and models are trained for each gender, respecting structural differences in the two tournaments.
4. **Conservative probability clipping**: Predictions are clipped to $[0.01, 0.99]$ to avoid catastrophic Brier Score penalties from overconfident predictions on upset outcomes.
---
## 3. Feature Engineering Pipeline
### 3.1 Base Statistics (Compact Results)
From the regular season compact results, we derive per-team season aggregates:
| Feature | Description |
|---------|-------------|
| `WinPct` | Season win percentage |
| `PFpg` / `PApg` | Points for / against per game |
| `Margin` | Average scoring margin |
| `Games` | Total games played |
### 3.2 Advanced Box-Score Metrics (Detailed Results)
Using the Dean Oliver "Four Factors" framework and beyond, we compute tempo-free efficiency metrics:
| Feature | Formula | Interpretation |
|---------|---------|----------------|
| `OffEff` | $100 \times \frac{\text{Points}}{\text{Possessions}}$ | Points scored per 100 possessions |
| `DefEff` | $100 \times \frac{\text{Points Allowed}}{\text{Possessions}}$ | Points allowed per 100 possessions |
| `Tempo` | $\frac{\text{Possessions}}{\text{Games}}$ | Pace of play |
| `eFG%` | $\frac{\text{FGM} + 0.5 \times \text{FGM3}}{\text{FGA}}$ | Effective field goal percentage |
| `TOV%` | $\frac{\text{Turnovers}}{\text{Possessions}}$ | Turnover rate |
| `ORB%` | $\frac{\text{Off. Rebounds}}{\text{Off. Reb. + Opp. Def. Reb.}}$ | Offensive rebounding percentage |
| `FTR` | $\frac{\text{FTA}}{\text{FGA}}$ | Free throw rate |
| `3PAr` | $\frac{\text{3PA}}{\text{FGA}}$ | Three-point attempt rate |
| `AST%` | $\frac{\text{Assists}}{\text{FGM}}$ | Assist ratio |
| `StlRate` | $\frac{\text{Steals}}{\text{Possessions}}$ | Steal rate |
| `BlkRate` | $\frac{\text{Blocks}}{\text{Opp. FGA}}$ | Block rate |
**Possession Estimation** follows the standard formula:
$$\text{Poss} = \text{FGA} - \text{OR} + \text{TO} + 0.475 \times \text{FTA}$$
### 3.3 Massey Composite Ordinal (Men's Only)
The Massey Ordinals aggregate ~150+ rating systems. We:
1. Filter to rankings up to day 133 (pre-tournament)
2. For each system, take the latest available ranking day
3. Normalize ranks to $[0, 1]$ via $\text{RankPct} = \frac{\text{Rank}}{\text{MaxRank}}$
4. Average across all systems: $\text{Massey} = 1 - \overline{\text{RankPct}}$
This provides a consensus "wisdom of crowds" metric that captures information beyond our own features.
### 3.4 Conference Strength
Conference membership is enriched with **Conference SRS** — the mean SRS of all teams within a conference. This captures schedule strength at the league level, penalizing teams in weak conferences.
### 3.5 Seasonal Percentile Rankings
Both SRS and Elo are converted to within-season percentile ranks (`SRSRankPct`, `EloRankPct`), providing scale-invariant relative positioning across seasons with different rating distributions.
### 3.6 Recent Form
The last 10 regular-season games are isolated to compute:
- `RecentWinPct`: Win rate in last 10 games
- `RecentMargin`: Average margin in last 10 games
This captures momentum effects and late-season trajectory.
---
## 4. Rating Systems
### 4.1 Elo Rating System
Our Elo implementation extends the classical chess Elo system with basketball-specific modifications:
**Parameters:**
| Parameter | Value | Rationale |
|-----------|-------|-----------|
| K-factor | 20.0 | Moderate update speed |
| Home Court Advantage | 65.0 Elo points | NCAA-calibrated HCA |
| Season Regression | 0.75 | 75% carryover, 25% regression to mean (1500) |
| Base Rating | 1500.0 | Standard Elo baseline |

**Margin of Victory Multiplier:**
```math
\text{MOV}_{\text{mult}} = \ln(\text{MOV} + 1) \times \frac{2.2}{0.001 \times |\Delta_{\text{Elo}}| + 2.2}
```
This serves dual purposes:
1. **Logarithmic MOV scaling** prevents blowout wins from having disproportionate influence
2. **Auto-correlation correction** reduces the update magnitude when the Elo gap already predicts a large margin, preventing rating inflation
**Update Rule:**
$$\Delta = K \times \text{MOV}_{\text{mult}} \times (S - E)$$
where $E = \frac{1}{1 + 10^{(R_B - R_A - \text{HCA})/400}}$ is the expected score.
### 4.2 Simple Rating System (SRS)
The SRS is an iterative fixed-point algorithm that decomposes team strength into offensive contribution and strength-of-schedule:
1. Initialize: $r_t^{(0)} = \overline{\text{MOV}}_t$
2. Iterate:
```math
r_t^{(k+1)} = \frac{1}{|\mathcal{G}_t|} \sum_{g \in \mathcal{G}_t} (\text{MOV}_g + r_{\text{opp}(g)}^{(k)})
```
3. Converge after 100 iterations
This is equivalent to solving the linear system $\mathbf{r} = \overline{\text{MOV}} + \mathbf{S} \cdot \mathbf{r}$, where $\mathbf{S}$ is the schedule matrix. The SRS is analogous to the system used by Basketball-Reference and captures "who you played" effects.
---
## 5. Modeling Strategy
### 5.1 Logistic Regression
- **Regularization**: L2 penalty with $C = 1.0$
- **Solver**: L-BFGS quasi-Newton method
- **Rationale**: Provides well-calibrated probabilistic outputs; serves as the linear backbone of the ensemble
### 5.2 Histogram Gradient Boosting Classifier
| Hyperparameter | Value | Rationale |
|----------------|-------|-----------|
| `max_depth` | 3 | Shallow trees prevent overfitting on ~2,200 tournament games |
| `learning_rate` | 0.05 | Conservative learning rate for smooth convergence |
| `max_iter` | 300 | Sufficient iterations with low learning rate |
| `l2_regularization` | 0.1 | Mild leaf regularization |
**Why HistGradientBoosting?** Sklearn's native implementation handles missing values natively, requires no hyperparameter tuning for categorical features, and provides competitive performance without external dependencies (XGBoost/LightGBM).
### 5.3 Raw Elo Probability
The third signal is computed directly from end-of-season Elo ratings:
$$P_{\text{Elo}}(\text{A wins}) = \frac{1}{1 + 10^{(R_B - R_A) / 400}}$$
This serves as a **regularizing prior** — a physics-based prediction that doesn't suffer from overfitting.
---
## 6. Ensemble & Calibration
### 6.1 Weighted Ensemble
The final prediction is a convex combination:
$$P_{\text{final}} = w_1 \cdot P_{\text{Logit}} + w_2 \cdot P_{\text{HGB}} + w_3 \cdot P_{\text{Elo}}$$
subject to $w_1 + w_2 + w_3 = 1$, $w_i \geq 0$.
Weights are determined via **grid search** over a simplex with step size 0.05, minimizing the out-of-fold Brier Score from cross-validation.
### 6.2 Probability Clipping
$$P_{\text{clipped}} = \text{clip}(P_{\text{final}}, 0.01, 0.99)$$
This is critical for Brier Score optimization: a single confident incorrect prediction ($P = 0.0$ when outcome = 1) contributes 1.0 to the Brier sum, while clipping limits the worst-case per-game contribution to $0.99^2 \approx 0.98$.
---
## 7. Cross-Validation Framework
### 7.1 GroupKFold by Season
We use **5-fold GroupKFold** with `Season` as the group variable:
- **No temporal leakage**: Each fold validates on entire held-out seasons
- **Simulates the competition setting**: Models must generalize to unseen tournament years
- **Enables honest weight optimization**: Ensemble weights are tuned on OOF predictions
### 7.2 Missing Data Handling
A two-tier imputation strategy:
1. **Season mean imputation**: Missing values filled with the mean of the same feature within the same season
2. **Global mean fallback**: Any remaining NaN values filled with the global feature mean
For prediction-time teams missing from the current season's data, we use their **most recent historical season** as a proxy.
---
## 8. Results & Analysis
### 8.1 Competition Performance
| Metric | Score |
|--------|-------|
| **Brier Score** | **0.1208707** |
| **Placement** | 29th  |
| **Medal** | 🥈 Silver |
| **Percentile** | Top ~3% |
### 8.2 Contextualizing the Score
| Benchmark | Brier Score |
|-----------|-------------|
| Random guessing ($P = 0.5$) | 0.2500 |
| Historical seed-based baseline | ~0.1600 |
| Strong ML solutions | ~0.1200–0.1300 |
| **This solution** | **0.1209** |
| Theoretical perfect | 0.0000 |
The gap between ~0.16 (seeds only) and ~0.12 (top ML solutions) represents the "edge" captured by advanced feature engineering and ensemble methods. The remaining gap to 0.0 is dominated by the **irreducible uncertainty** inherent in single-elimination tournaments (injuries, hot shooting nights, etc.).
### 8.3 Key Strengths
1. **Simplicity**: No external dependencies beyond scikit-learn; no GPU required
2. **Robustness**: Three complementary signals provide stability across tournament structures
3. **Calibration**: Logistic Regression + Elo prior ensures well-calibrated probabilities
4. **Efficiency**: Full pipeline runs in < 2 minutes on commodity hardware
---
## 9. Reproducibility
### 9.1 Requirements
```
python >= 3.9
numpy >= 1.21
pandas >= 1.4
scikit-learn >= 1.0
```
### 9.2 Running the Solution
```bash
# 1. Clone or download the repository
# 2. Place competition data in data/ directory
# 3. Run the solution
python make_submission.py
```
The script will:
1. Build team-level features for Men's and Women's divisions
2. Construct pairwise differential features from historical tournament games
3. Train Logistic Regression and Histogram Gradient Boosting models
4. Optimize ensemble weights via cross-validated Brier Score minimization
5. Generate predictions for all 2026 tournament matchups
6. Save `submission.csv`
### 9.3 Random Seed
All stochastic components use `RANDOM_STATE = 42` for reproducibility.
---
## License
This solution is shared for educational and research purposes under the MIT License.
---
*This document accompanies the competition notebook `march_mania_2026_solution.ipynb` which contains the full executable solution with inline analysis and visualizations.*
