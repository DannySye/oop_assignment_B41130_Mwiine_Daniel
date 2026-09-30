# Assignment 1: UBOS District Population Forecaster & Infrastructure Planner

**Course:** MSCS & MSDS – Object-Oriented Programming with Python  
**Term:** Advent 2026  
**Author:** Daniel Mwiine  
**Student ID:** B41130  
**Registration Number:** S26M25/001  
**Module:** Mini-Project 1 (`assignment_1`)  

---

## 1. Executive Summary & Problem Overview

Public resource allocation in Uganda demands robust demographic forecasting. District planning units frequently must allocate capital expenditure for education infrastructure 5 years in advance. Under-forecasting results in severe classroom congestion, compromised teacher-pupil ratios, and school dropout rates; over-forecasting leads to misallocated fiscal resources and stranded capital assets.

This mini-project models historical demographic trajectories (2015–2024) across five Ugandan districts:
- **Kampala:** Uganda's capital city and commercial epicenter.
- **Wakiso:** High-density peri-urban corridor surrounding Kampala experiencing rapid residential expansion.
- **Gulu:** Northern Uganda's primary economic and urban hub.
- **Mukono:** Major industrial and transport junction in the Greater Kampala Metropolitan Area.
- **Mbarara:** Regional administrative and agricultural hub of Southwestern Uganda.

Using rigorous **Object-Oriented Programming (OOP)**, defensive data validation, comparative statistical analysis, backtesting, and residual bootstrap resampling, this system projects population trajectories for **2025–2029** and translates them into required primary school classroom construction schedules.

---

## 2. Architecture & File Structure

```text
assignment_1/
├── src/
│   ├── __init__.py
│   └── population/
│       ├── __init__.py           # Package entry point exposing public API
│       ├── models.py             # DistrictPopulation domain model with validation & dunder methods
│       ├── stats.py              # Statistical analysis (statistics vs. NumPy ddof divergence)
│       ├── growth.py             # YoY growth rates and CAGR analytics
│       ├── forecasters.py        # Forecaster ABC, LinearTrend, ExponentialCAGR, FibonacciRatio
│       ├── evaluation.py         # Backtesting engine: MAE, RMSE, MAPE, model selection
│       ├── bootstrap.py          # Non-parametric residual bootstrapping (B = 2,000 iterations)
│       └── planning.py           # Primary school classroom infrastructure deployment logic
├── tests/
│   ├── __init__.py
│   └── test_population.py        # Comprehensive test suite (23 unit & integration tests)
├── notebooks/
│   ├── project1_population.ipynb       # Fully executed Jupyter Notebook
│   └── district_population_forecasts.png # 300 DPI multi-panel forecast figure
└── README.md                           # This file
```

---

## 3. Mathematical Foundations & OOP Design

### 3.1 Domain Model: `DistrictPopulation`
The `DistrictPopulation` class encapsulates time series observations with defensive validation:
- **Validation Rules:**
  - Matching dimensions between calendar years and population values.
  - At least 2 observations.
  - Strictly positive calendar years and population values ($P_t > 0$).
  - Strictly ascending chronological order with no duplicate years.
- **Immutability:** Internal NumPy arrays have `writeable = False` to protect data integrity.
- **Dunder Methods:**
  - `__len__`: Returns the number of observations.
  - `__repr__`: Informative string representation detailing name, period, and range.
  - `__getitem__`: Supports both integer indexing `(year, value)` and slice extraction (yielding a sub-`DistrictPopulation`).
  - `__iter__`: Iterates over `(year, value)` tuples.
  - `__eq__`: Verifies equality across names, years, and values.

### 3.2 Statistical Mechanics & The `ddof` Divergence
The standard Python `statistics.variance` and `np.var` diverge by default:
- **`statistics.variance` (Sample Variance, $s^2$):** Assumes the data is a sample drawn from a larger population and applies **Bessel's correction**, dividing by $N - 1$:
  $$s^2 = \frac{1}{N - 1}\sum_{i=1}^N (x_i - \bar{x})^2$$
- **`np.var(..., ddof=0)` (Population Variance, $\sigma^2$):** Divides by $N$:
  $$\sigma^2 = \frac{1}{N}\sum_{i=1}^N (x_i - \bar{x})^2$$
- **Divergence Factor:**
  $$\frac{s^2}{\sigma^2} = \frac{N}{N - 1}$$
  For our $N = 10$ series, $\frac{10}{9} \approx 1.1111$ (an exact $11.11\%$ understatement by default NumPy). Setting `np.var(values, ddof=1)` resolves the discrepancy identically.

### 3.3 Demographic Growth Dynamics
- **Year-on-Year (YoY) Percentage Growth:**
  $$\text{YoY}_t = \left(\frac{P_t - P_{t-1}}{P_{t-1}}\right) \times 100\%$$
- **Compound Annual Growth Rate (CAGR):**
  $$\text{CAGR} = \left(\frac{P_{\text{end}}}{P_{\text{start}}}\right)^{\frac{1}{n}} - 1 \quad (n = 9 \text{ intervals})$$

| Rank | District | Start (2015) | End (2024) | CAGR (%) | Mean YoY (%) | Max YoY (%) |
| :---: | :--- | :---: | :---: | :---: | :---: | :---: |
| **1** | **Wakiso** | 950.0k | 1,670.0k | **6.47%** | 6.50% | 7.41% |
| **2** | **Kampala** | 1,200.0k | 1,800.0k | **4.61%** | 4.63% | 5.63% |
| **3** | **Gulu** | 320.0k | 480.0k | **4.61%** | 4.63% | 5.80% |
| **4** | **Mukono** | 590.0k | 878.0k | **4.52%** | 4.54% | 4.75% |
| **5** | **Mbarara** | 470.0k | 678.0k | **4.16%** | 4.18% | 4.38% |

*Interpretation:* Wakiso is expanding at $6.47\%$ per annum—nearly $40\%$ faster than Kampala—reflecting rapid suburbanization and residential relocation into peri-urban zones.

---

## 4. Forecasting Architecture & Out-of-Sample Backtesting

We implement an abstract base class `Forecaster(ABC)` enforcing `fit(years, values)` and `predict(horizon)` across three concrete subclasses:
1. **`LinearTrendForecaster`**: Fits $\hat{y}(t) = \beta_0 + \beta_1 t$ via OLS (`np.polyfit(deg=1)`).
2. **`ExponentialCAGRForecaster`**: Fits $\ln(y) = \alpha + \beta (t - t_0)$ via log-linear regression, modeling constant geometric compounding.
3. **`FibonacciRatioForecaster`**: Scales successive periods by $\frac{F_{k+1}}{F_k}$. Supported in literal `direct` mode and normalized `calibrated` mode.

### Backtesting Results (2015–2021 Train / 2022–2024 Test)

| District | Model | MAE (k) | RMSE (k) | MAPE (%) | Selected |
| :--- | :--- | :---: | :---: | :---: | :---: |
| **Kampala** | Linear Trend Forecaster | **4.87** | **5.71** | **0.28%** | **BEST** |
| | Exponential / CAGR Forecaster | 14.89 | 15.69 | 0.86% | |
| | Fibonacci-Ratio (Calibrated) | 26.68 | 27.67 | 1.54% | |
| | Fibonacci-Ratio (Direct) | 3,115.93 | 3,745.54 | 179.37% | |
| **Wakiso** | Linear Trend Forecaster | **10.57** | **11.45** | **0.67%** | **BEST** |
| | Exponential / CAGR Forecaster | 19.33 | 20.35 | 1.22% | |
| | Fibonacci-Ratio (Calibrated) | 31.81 | 35.79 | 1.99% | |
| | Fibonacci-Ratio (Direct) | 2,746.42 | 3,302.26 | 175.76% | |
| **Gulu** | Linear Trend Forecaster | **3.80** | **4.13** | **0.84%** | **BEST** |
| | Exponential / CAGR Forecaster | 6.84 | 7.07 | 1.50% | |
| | Fibonacci-Ratio (Calibrated) | 12.39 | 13.06 | 2.70% | |
| | Fibonacci-Ratio (Direct) | 788.19 | 947.67 | 173.34% | |
| **Mukono** | Linear Trend Forecaster | **6.59** | **7.59** | **0.78%** | **BEST** |
| | Exponential / CAGR Forecaster | 11.23 | 12.06 | 1.33% | |
| | Fibonacci-Ratio (Calibrated) | 21.05 | 22.87 | 2.50% | |
| | Fibonacci-Ratio (Direct) | 1,489.26 | 1,790.69 | 177.30% | |
| **Mbarara** | Linear Trend Forecaster | **3.27** | **3.89** | **0.51%** | **BEST** |
| | Exponential / CAGR Forecaster | 6.78 | 7.50 | 1.05% | |
| | Fibonacci-Ratio (Calibrated) | 15.82 | 17.51 | 2.47% | |
| | Fibonacci-Ratio (Direct) | 1,154.91 | 1,388.75 | 178.68% | |

*Summary:* The Linear Trend Forecaster achieved the lowest MAE and MAPE across all five districts (MAPEs well below $1.0\%$), indicating steady linear increments in historical data. Direct Fibonacci scaling fails completely ($\text{MAPE} \approx 175\%$).

---

## 5. 5-Year Out-of-Sample Projections (2025–2029) & Variance Shift

Retraining the best-performing models on the complete 2015–2024 dataset yields the following 5-year outlook:

| District | 2024 Base (k) | 2029 Projection (k) | Net Growth (k) | Hist Var ($s_h^2$) | Forecast Var ($s_f^2$) | Variance Ratio |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Kampala** | 1,800.0 | 2,136.2 | +336.2 | 41,271.1 | 5,556.7 | 0.13 |
| **Wakiso** | 1,670.0 | 2,314.1 | +644.1 | 60,378.9 | 20,417.8 | 0.34 |
| **Gulu** | 480.0 | 569.2 | +89.2 | 2,900.0 | 389.9 | 0.13 |
| **Mukono** | 878.0 | 1,038.5 | +160.5 | 9,842.7 | 1,267.3 | 0.13 |
| **Mbarara** | 678.0 | 808.9 | +130.9 | 5,090.7 | 843.8 | 0.17 |

### Operational Interpretation of the Variance Shift
The empirical variance ratio ($\frac{s_f^2}{s_h^2} \approx 0.13 - 0.34$) reflects a fundamental difference in meaning:
- **Historical Variance ($s_h^2$):** Measures past empirical dispersion across 10 years of observed demographic history, including business cycles, urban migration shocks, and census adjustments.
- **Forecast Variance ($s_f^2$):** Along a deterministic model, this metric merely reflects the mathematical square of the slope ($\beta_1^2 \cdot \text{Var}(t)$). It reflects **zero** stochastic uncertainty.
- **Planning Implication:** Point forecast trajectories create an illusion of certainty. Planners must rely on **prediction intervals** (such as our bootstrap bands) to buffer against real-world stochastic shocks.

---

## 6. Primary School Classroom Infrastructure Plan (2025–2029)

Applying Ministry of Education & Sports (MoES) and UBOS guidelines:
- Primary school cohort proportion: **$18\%$** ($0.18$) of total population.
- Standard classroom capacity: **$53$ pupils per classroom**.
- Integer ceiling rounding: $\lceil \frac{\Delta \text{Pupils}}{53} \rceil$.

| District | Baseline Pop (2024) | Projected Pop (2029) | Net Population Increase | Additional Primary Pupils | Net Classrooms Required |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Kampala** | 1,800,000 | 2,136,242 | 336,242 | 60,524 | **1,142** |
| **Wakiso** | 1,670,000 | 2,314,061 | 644,061 | 115,931 | **2,188** |
| **Gulu** | 480,000 | 569,212 | 89,212 | 16,058 | **303** |
| **Mukono** | 878,000 | 1,038,485 | 160,485 | 28,887 | **546** |
| **Mbarara** | 678,000 | 808,879 | 130,879 | 23,558 | **445** |
| **TOTAL** | — | — | **1,368,879** | **246,458** | **4,624** |

---

## 7. Extension Tasks

### 7.1 Residual Bootstrap Resampling ($B = 2{,}000$ iterations)
Rather than relying on unrealistic Gaussian error assumptions, we implemented a non-parametric residual bootstrap algorithm:
1. Centered in-sample residuals: $\tilde{e}_t = e_t - \bar{e}$.
2. Resampled residuals with replacement across 2,000 iterations to generate synthetic series $y_t^{*(b)} = \hat{y}_t + e_t^{*(b)}$.
3. Refitted model clones on each synthetic series to capture **parameter estimation uncertainty**.
4. Appended resampled innovation errors to simulate **future shock uncertainty**.
5. Calculated empirical $2.5\%$ and $97.5\%$ quantiles to construct rigorous $95\%$ prediction intervals, displayed as shaded bands on all forecast charts.

### 7.2 Critical Demographic Critique of the Fibonacci-Ratio Model
- **Non-Biological Exponential Scale:** Fibonacci ratio scaling multiplies population by $\phi \approx 1.61803$, which equates to a $+61.8\%$ annual growth rate. This would cause human populations to double every $1.44$ years—a biological absurdity.
- **Historical Context:** Fibonacci (1202) modeled immortal, paired rabbits reproducing each month with zero mortality, zero gestation delay, and infinite carrying capacity. Human populations, by contrast, feature 9-month gestation, single births, decades of dependency, and age-specific mortality.
- **Defensible Scope:** Fibonacci scaling is only defensible under:
  1. Synchronous, immortal asexual reproduction (e.g. parthenogenic insects or laboratory cell fission).
  2. Modulated cycle indices where $\frac{F_{k+1}/F_k}{\phi}$ acts as a normalized multiplier around an authentic demographic baseline growth rate.

---

## 8. Findings & Limitations (250 Words)

### Findings
This study demonstrates that demographic expansion in Uganda's central economic corridor is heavily concentrated in peri-urban zones. Wakiso District leads with a Compound Annual Growth Rate (CAGR) of $6.47\%$, significantly outpacing Kampala ($4.61\%$), Gulu ($4.61\%$), Mukono ($4.52\%$), and Mbarara ($4.16\%$). Walk-forward backtesting proved that Linear Trend and Exponential CAGR models deliver superior predictive fidelity (out-of-sample MAPEs $< 1.0\%$), whereas naive Fibonacci scaling produces disastrous errors exceeding $175\%$. 

Between 2024 and 2029, the five surveyed districts will experience a combined net population growth of $1,368,879$ individuals, generating an influx of $246,458$ primary school pupils. Under national standards of 53 pupils per classroom, this necessitates constructing **$4,624$ additional standard classrooms** by 2029. Wakiso alone requires **$2,188$ classrooms** ($47.3\%$ of the total), representing an urgent public infrastructure imperative.

### Limitations
1. **Model Simplicity:** Linear extrapolation assumes constant arithmetic expansion, ignoring density-dependent feedback, zoning saturation, and economic inflection points.
2. **Static Age-Cohort Ratio:** Utilizing a uniform $18\%$ school-age ratio masks regional differences in Total Fertility Rates (TFR), which are higher in Northern Uganda than metropolitan Kampala.
3. **Macroeconomic Shocks:** Deterministic models cannot capture sudden internal displacement, epidemics, or policy shifts that disrupt baseline demographic trajectories.

---

## 9. Verification & Reproduction Instructions

### Running Unit Tests
Execute the full pytest suite from the repository root:
```bash
python -m pytest assignment_1/tests/test_population.py -v
```
All **23 unit and integration tests** will execute and pass in $< 0.5$ seconds.

### Executing the Notebook
Open the notebook in Jupyter and run **Restart & Run All**:
```bash
jupyter notebook assignment_1/notebooks/project1_population.ipynb
```
The notebook executes top-to-bottom without errors and regenerates all figures.

---

## 10. AI Coding Assistant Usage Disclosure

See [README.md](../README.md#ai-use-disclosure) in the repository root for a full account of how AI assistance was used across all five projects.

For this specific project, AI was used to:
- Scaffold the initial `DistrictPopulation` class structure and suggest appropriate dunder methods.
- Help debug a dimension-mismatch error in the bootstrap resampling loop.
- Suggest matplotlib styling for the multi-panel forecast figure.

All mathematical formulations (ddof variance proofs, CAGR derivation, bootstrap algorithm), the backtesting engine design, and the infrastructure planning logic were developed and verified by me independently.

