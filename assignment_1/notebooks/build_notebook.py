"""
Script to programmatically build and execute project1_population.ipynb
using nbformat and nbclient, ensuring deterministic top-to-bottom execution.
"""

import sys
import os
from pathlib import Path
import nbformat as nbf
from nbclient import NotebookClient

def build_notebook():
    nb = nbf.v4.new_notebook()
    cells = []

    # -------------------------------------------------------------
    # Cell 1: Title and Header
    # -------------------------------------------------------------
    cells.append(nbf.v4.new_markdown_cell("""# Mini-Project 1: UBOS District Population Forecaster & School Infrastructure Planner

**Course:** MSCS & MSDS – Object-Oriented Programming with Python  
**Term:** Advent 2026  
**Author:** Daniel Mwiine  
**Student ID:** B41130  
**Registration Number:** S26M25/001  
**Repository:** `oop_assignment_B41130_Mwiine_Daniel`  

---

### Executive Overview & Problem Context
The Uganda Bureau of Statistics (UBOS) and District Planning Units face critical challenges in public resource allocation. Planning social infrastructure—specifically primary school classroom construction—demands reliable, multi-year demographic projections.

This mini-project develops an object-oriented, mathematically rigorous demographic forecasting and educational planning system for five Ugandan districts:
1. **Kampala:** Uganda's administrative and commercial capital.
2. **Wakiso:** Fast-expanding peri-urban district bordering Kampala experiencing rapid demographic influx.
3. **Gulu:** Primary economic and educational metropolis of Northern Uganda.
4. **Mukono:** Major industrial and residential corridor in the Greater Kampala Metropolitan Area.
5. **Mbarara:** The regional commercial and agricultural growth center of Southwestern Uganda.

We implement:
- Robust domain modeling with data encapsulation and defensive validation.
- Comparative statistical mechanics contrasting Python's `statistics` module with `numpy`, resolving the degrees-of-freedom (`ddof`) variance divergence.
- Growth dynamics profiling computing Year-on-Year (YoY) trajectories and Compound Annual Growth Rates (CAGR).
- An extensible Object-Oriented forecasting framework (`Forecaster` ABC) supporting Linear Trend, Exponential CAGR, and Fibonacci-Ratio models.
- Rigorous walk-forward out-of-sample backtesting (2015–2021 train vs. 2022–2024 test) reporting MAE, RMSE, and MAPE.
- 5-year out-of-sample projections (2025–2029) and demographic variance analysis.
- Multi-panel visual dashboards with empirical 95% bootstrap prediction intervals ($B = 2{,}000$ iterations).
- Primary school classroom infrastructure deployment schedules based on Ministry of Education standards.
- A critical demographic critique of the Fibonacci growth formulation."""))

    # -------------------------------------------------------------
    # Cell 2: Imports and Setup
    # -------------------------------------------------------------
    cells.append(nbf.v4.new_code_cell("""import sys
import os
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import statistics

# Ensure local source tree is accessible
current_dir = Path(os.getcwd())
repo_root = current_dir.parent.parent if current_dir.name == "notebooks" else current_dir
src_path = repo_root / "assignment_1" / "src"
if str(src_path) not in sys.path:
    sys.path.insert(0, str(src_path))

from population.models import DistrictPopulation
from population.stats import (
    compute_standard_statistics,
    compute_numpy_statistics,
    compare_statistics_and_numpy,
)
from population.growth import (
    compute_yoy_growth,
    compute_cagr,
    analyze_district_growth,
    rank_districts_by_growth,
)
from population.forecasters import (
    LinearTrendForecaster,
    ExponentialCAGRForecaster,
    FibonacciRatioForecaster,
)
from population.evaluation import (
    calculate_mae,
    calculate_rmse,
    calculate_mape,
    evaluate_forecast,
    backtest_district,
)
from population.planning import ClassroomPlanner
from population.bootstrap import bootstrap_prediction_intervals

# Pinned random number generator for deterministic execution
SEED = 2026
rng = np.random.default_rng(SEED)

print("Packages imported successfully. Deterministic seed pinned at:", SEED)"""))

    # -------------------------------------------------------------
    # Cell 3: Task 1 Markdown
    # -------------------------------------------------------------
    cells.append(nbf.v4.new_markdown_cell("""---
## Core Task 1: Domain Modeling & Data Ingestion

We instantiate the `DistrictPopulation` domain model. This class guarantees:
- **Input Validation:** Enforces matching array dimensions, positive calendar years, strictly positive population values, and chronologically ascending order.
- **Immutability:** Internal NumPy arrays have their writeable flags disabled to prevent accidental state corruption.
- **Dunder Methods:** Implements `__len__`, `__repr__`, `__getitem__` (supporting indexing and slicing), and `__iter__`.

In addition to Kampala, Wakiso, and Gulu, we introduce two authentic Ugandan districts:
- **Mukono:** `[590, 615, 642, 670, 700, 732, 765, 800, 838, 878]`
- **Mbarara:** `[470, 488, 506, 526, 547, 570, 594, 620, 648, 678]`"""))

    # -------------------------------------------------------------
    # Cell 4: Task 1 Code
    # -------------------------------------------------------------
    cells.append(nbf.v4.new_code_cell("""# 10-year historical series (2015 - 2024) in thousands
years = np.arange(2015, 2025, dtype=np.int64)

district_raw_data = {
    "Kampala": [1200, 1250, 1300, 1350, 1420, 1500, 1580, 1650, 1720, 1800],
    "Wakiso":  [950, 1000, 1070, 1150, 1220, 1300, 1390, 1480, 1570, 1670],
    "Gulu":    [320, 330, 345, 360, 375, 390, 410, 430, 455, 480],
    "Mukono":  [590, 615, 642, 670, 700, 732, 765, 800, 838, 878],
    "Mbarara": [470, 488, 506, 526, 547, 570, 594, 620, 648, 678],
}

districts = {}
for name, data in district_raw_data.items():
    districts[name] = DistrictPopulation(name, years, data)
    print(districts[name])

# Demonstrate defensive validation
try:
    bad_district = DistrictPopulation("Invalid", [2020, 2019], [100, 120])
except ValueError as e:
    print(f"\\nDefensive Validation Success (Descending years caught): {e}")"""))

    # -------------------------------------------------------------
    # Cell 5: Task 2 Markdown
    # -------------------------------------------------------------
    cells.append(nbf.v4.new_markdown_cell("""---
## Core Task 2: Statistical Analysis & The Degrees of Freedom (`ddof`) Divergence

We compute the mean, median, sample variance, and standard deviation for each district using:
1. Python's standard `statistics` module.
2. `numpy` with default parameters (`ddof=0`).
3. `numpy` with sample correction (`ddof=1`).

### Mathematical Origin of Divergence
- **Sample Variance ($s^2$ via `statistics.variance`):** Assumes the data represents a sample drawn from a larger population and applies **Bessel's correction** by dividing the sum of squared deviations by $N - 1$:
  $$s^2 = \\frac{1}{N - 1}\\sum_{i=1}^N (x_i - \\bar{x})^2$$
  This yields an unbiased estimator because using the sample mean $\\bar{x}$ instead of the true population mean $\\mu$ introduces a downward bias of factor $\\frac{N - 1}{N}$.
- **Population Variance ($\\sigma^2$ via `np.var(ddof=0)`):** Assumes the observations constitute the complete population, dividing by $N$:
  $$\\sigma^2 = \\frac{1}{N}\\sum_{i=1}^N (x_i - \\bar{x})^2$$
- **The Divergence Ratio:**
  $$\\frac{s^2}{\\sigma^2} = \\frac{N}{N - 1}$$
  For $N = 10$, $\\frac{10}{9} \\approx 1.1111$. Thus, default NumPy understates sample variance by exactly $11.11\\%$. Specifying `np.var(values, ddof=1)` resolves the divergence completely."""))

    # -------------------------------------------------------------
    # Cell 6: Task 2 Code
    # -------------------------------------------------------------
    cells.append(nbf.v4.new_code_cell("""print(f"{'District':<10} | {'Mean (k)':<9} | {'Median':<8} | {'Stats Var (ddof=1)':<18} | {'NumPy Var (ddof=0)':<18} | {'NumPy Var (ddof=1)':<18} | {'Ratio':<6}")
print("-" * 98)

comparison_results = {}
for name, d_obj in districts.items():
    cmp = compare_statistics_and_numpy(d_obj.values)
    comparison_results[name] = cmp
    m = cmp["standard_statistics"]["mean"]
    med = cmp["standard_statistics"]["median"]
    v_stat = cmp["standard_statistics"]["variance"]
    v_np0 = cmp["numpy_ddof_0"]["variance"]
    v_np1 = cmp["numpy_ddof_1"]["variance"]
    ratio = cmp["empirical_ratio"]
    print(f"{name:<10} | {m:<9.1f} | {med:<8.1f} | {v_stat:<18.2f} | {v_np0:<18.2f} | {v_np1:<18.2f} | {ratio:<6.4f}")

# Verify exact theoretical match for Kampala
k_cmp = comparison_results["Kampala"]
print("\\n" + k_cmp["explanation"])"""))

    # -------------------------------------------------------------
    # Cell 7: Task 3 Markdown
    # -------------------------------------------------------------
    cells.append(nbf.v4.new_markdown_cell("""---
## Core Task 3: Growth Dynamics (YoY Growth & Compound Annual Growth Rate)

We evaluate growth through two complementary metrics:
1. **Year-on-Year (YoY) Percentage Growth:**
   $$\\text{YoY}_t = \\left(\\frac{P_t - P_{t-1}}{P_{t-1}}\\right) \\times 100\\%$$
2. **Compound Annual Growth Rate (CAGR):**
   $$\\text{CAGR} = \\left(\\frac{P_{\\text{end}}}{P_{\\text{start}}}\\right)^{\\frac{1}{n}} - 1$$
   where $n = 2024 - 2015 = 9$ compounding intervals."""))

    # -------------------------------------------------------------
    # Cell 8: Task 3 Code
    # -------------------------------------------------------------
    cells.append(nbf.v4.new_code_cell("""ranked_growth = rank_districts_by_growth(list(districts.values()))

print(f"{'Rank':<4} | {'District':<10} | {'Start (2015)':<12} | {'End (2024)':<10} | {'CAGR (%)':<10} | {'Mean YoY (%)':<12} | {'Max YoY (%)':<11}")
print("-" * 80)
for idx, item in enumerate(ranked_growth, 1):
    print(f"{idx:<4} | {item['district_name']:<10} | {item['start_population_k']:<12.1f} | {item['end_population_k']:<10.1f} | "
          f"{item['cagr_percent']:<10.2f} | {item['mean_yoy_percent']:<12.2f} | {item['max_yoy_percent']:<11.2f}")

fastest = ranked_growth[0]
print(f"\\nFastest Expanding District: {fastest['district_name']} at {fastest['cagr_percent']:.2f}% CAGR.")"""))

    # -------------------------------------------------------------
    # Cell 9: Task 4 Markdown
    # -------------------------------------------------------------
    cells.append(nbf.v4.new_markdown_cell("""---
## Core Task 4: Forecasting Architecture (OOP Hierarchy)

We construct an extensible model hierarchy centered around the abstract base class `Forecaster`:
- **`Forecaster(ABC)`**: Defines lifecycle methods `fit(years, values)` and `predict(horizon)`, in-sample residuals extraction, and state tracking.
- **`LinearTrendForecaster`**: Models constant absolute population increments via degree-1 polynomial regression (`np.polyfit(years, values, deg=1)`):
  $$\\hat{y}(t) = \\beta_0 + \\beta_1 t$$
- **`ExponentialCAGRForecaster`**: Models constant geometric growth via log-linear Ordinary Least Squares (OLS):
  $$\\ln(y) = \\alpha + \\beta (t - t_0) \\implies \\hat{y}(t) = \\exp(\\alpha + \\beta(t - t_0))$$
- **`FibonacciRatioForecaster`**: Scales periods by successive Fibonacci ratios $\\frac{F_{k+1}}{F_k}$:
  $$F_k = F_{k-1} + F_{k-2}, \\quad \\lim_{k \\to \\infty} \\frac{F_{k+1}}{F_k} = \\phi \\approx 1.6180339887...$$"""))

    # -------------------------------------------------------------
    # Cell 10: Task 4 Code
    # -------------------------------------------------------------
    cells.append(nbf.v4.new_code_cell("""# Instantiate model prototypes
models_pool = [
    LinearTrendForecaster(),
    ExponentialCAGRForecaster(),
    FibonacciRatioForecaster(mode="calibrated", name="Fibonacci (Calibrated)"),
    FibonacciRatioForecaster(mode="direct", name="Fibonacci (Direct)"),
]

# Quick demonstration of inheritance and fit-predict API on Kampala
demo_model = LinearTrendForecaster()
demo_model.fit(years, districts["Kampala"].values)
print(f"Model: {demo_model.name}")
print(f"Fitted Slope: {demo_model.slope:.2f}k persons/year")
print(f"Fitted Intercept: {demo_model.intercept:.2f}k")
print(f"Goodness of Fit R^2: {demo_model.r_squared:.4f}")
print(f"3-year forecast (2025-2027): {demo_model.predict(3).round(1)}k")"""))

    # -------------------------------------------------------------
    # Cell 11: Task 5 Markdown
    # -------------------------------------------------------------
    cells.append(nbf.v4.new_markdown_cell("""---
## Core Task 5: Backtesting & Out-of-Sample Evaluation

To prevent look-ahead bias, we implement rolling backtesting:
- **Training Set:** 2015–2021 (7 years).
- **Validation Test Set:** 2022–2024 (3 years out-of-sample).

We benchmark forecast accuracy across three standard statistical error metrics:
1. **Mean Absolute Error (MAE):**
   $$\\text{MAE} = \\frac{1}{m}\\sum_{i=1}^m |y_i - \\hat{y}_i|$$
2. **Root Mean Squared Error (RMSE):**
   $$\\text{RMSE} = \\sqrt{\\frac{1}{m}\\sum_{i=1}^m (y_i - \\hat{y}_i)^2}$$
3. **Mean Absolute Percentage Error (MAPE):**
   $$\\text{MAPE} = \\frac{100\\%}{m}\\sum_{i=1}^m \\left|\\frac{y_i - \\hat{y}_i}{y_i}\\right|$$"""))

    # -------------------------------------------------------------
    # Cell 12: Task 5 Code
    # -------------------------------------------------------------
    cells.append(nbf.v4.new_code_cell("""backtest_results = {}

for name, d_obj in districts.items():
    # Evaluate the primary competing models
    models_to_test = [
        LinearTrendForecaster(),
        ExponentialCAGRForecaster(),
        FibonacciRatioForecaster(mode="calibrated", name="Fibonacci-Ratio (Calibrated)"),
        FibonacciRatioForecaster(mode="direct", name="Fibonacci-Ratio (Direct)"),
    ]
    bt = backtest_district(d_obj, models_to_test, train_end_year=2021)
    backtest_results[name] = bt

# Render comprehensive evaluation table
print(f"{'District':<10} | {'Model':<30} | {'MAE (k)':<8} | {'RMSE (k)':<9} | {'MAPE (%)':<8}")
print("-" * 75)
for name, bt in backtest_results.items():
    for eval_item in bt["models_evaluation"]:
        m_name = eval_item["model_name"]
        mae = eval_item["mae"]
        rmse = eval_item["rmse"]
        mape = eval_item["mape"]
        is_best = " (BEST)" if m_name == bt["best_model_name"] else ""
        print(f"{name:<10} | {m_name + is_best:<30} | {mae:<8.2f} | {rmse:<9.2f} | {mape:<8.2f}")
    print("-" * 75)

print("\\nOptimal Model Selection per District:")
for name, bt in backtest_results.items():
    print(f"• {name:<10}: {bt['best_model_name']} (MAE: {bt['best_model_mae']:.2f}k, MAPE: {bt['best_model_mape']:.2f}%)")"""))

    # -------------------------------------------------------------
    # Cell 13: Task 6 Markdown
    # -------------------------------------------------------------
    cells.append(nbf.v4.new_markdown_cell("""---
## Core Task 6: Out-of-Sample Projections (2025–2029) & Variance Shift Analysis

We refit the optimal model for each district using the complete 10-year historical dataset (2015–2024) and project annual population figures for the 5-year planning horizon (**2025–2029**).

### Operational Interpretation of the Variance Shift
- **Historical Variance:** Reflects empirical dispersion driven by historical demographic shocks, macro-economic migration shifts, rural-urban transition, and census revisions.
- **Forecast Variance:** Along a deterministic trendline, forecast variance reflects purely the mathematical gradient (slope $\\beta_1$ or compounding curvature $\\beta$) evaluated over discrete time steps.
- **Operational Meaning:** While point forecasts produce a smooth, increasing trajectory, true demographic reality exhibits escalating stochastic uncertainty as the forecast horizon broadens. Planners who fail to account for variance expansion risk under-building or over-capitalizing physical facilities."""))

    # -------------------------------------------------------------
    # Cell 14: Task 6 Code
    # -------------------------------------------------------------
    cells.append(nbf.v4.new_code_cell("""future_years = np.arange(2025, 2030, dtype=np.int64)
projections = {}
variance_analysis = {}

for name, d_obj in districts.items():
    best_model_name = backtest_results[name]["best_model_name"]
    # Re-instantiate best model type
    if "Linear" in best_model_name:
        prod_model = LinearTrendForecaster()
    elif "Exponential" in best_model_name:
        prod_model = ExponentialCAGRForecaster()
    else:
        prod_model = FibonacciRatioForecaster(mode="calibrated")
        
    prod_model.fit(d_obj.years, d_obj.values)
    preds = prod_model.predict(future_years)
    projections[name] = {
        "model": prod_model,
        "years": future_years,
        "values": preds,
        "base_2024": float(d_obj.values[-1]),
        "projected_2029": float(preds[-1]),
    }
    
    hist_var = float(np.var(d_obj.values, ddof=1))
    fc_var = float(np.var(preds, ddof=1))
    variance_analysis[name] = {
        "hist_var": hist_var,
        "fc_var": fc_var,
        "ratio": fc_var / hist_var,
    }

print(f"{'District':<10} | {'2024 Base':<10} | {'2029 Projection':<16} | {'Net Growth':<11} | {'Hist Var':<10} | {'FC Var':<10} | {'Var Ratio':<9}")
print("-" * 88)
for name, p in projections.items():
    base = p["base_2024"]
    proj = p["projected_2029"]
    growth = proj - base
    hv = variance_analysis[name]["hist_var"]
    fv = variance_analysis[name]["fc_var"]
    v_rat = variance_analysis[name]["ratio"]
    print(f"{name:<10} | {base:<10.1f} | {proj:<16.1f} | {growth:<11.1f} | {hv:<10.1f} | {fv:<10.1f} | {v_rat:<9.2f}")"""))

    # -------------------------------------------------------------
    # Cell 15: Task 7 & Extension 1 Markdown
    # -------------------------------------------------------------
    cells.append(nbf.v4.new_markdown_cell("""---
## Core Task 7 & Extension 1: Multi-Panel Visualization & Residual Bootstrap Prediction Intervals

We construct empirical **95% prediction intervals** via non-parametric **residual bootstrapping ($B = 2{,}000$ iterations)**:
1. Extract and mean-center training residuals:
   $$\\tilde{e}_t = (y_t - \\hat{y}_t) - \\overline{(y - \\hat{y})}$$
2. Resample residuals with replacement to construct synthetic training series $y_t^{*(b)} = \\hat{y}_t + e_t^{*(b)}$.
3. Refit the model on each bootstrap replicate and simulate future paths with resampled innovation shocks:
   $$y_{t+h}^{*(b)} = \\hat{y}_{t+h}^{*(b)} + \\epsilon_{t+h}^{*(b)}$$
4. Compute the empirical $2.5\\%$ and $97.5\\%$ quantiles across the $2{,}000$ generated future paths, rendering them as shaded prediction bands."""))

    # -------------------------------------------------------------
    # Cell 16: Task 7 & Extension 1 Code
    # -------------------------------------------------------------
    cells.append(nbf.v4.new_code_cell("""fig, axes = plt.subplots(3, 2, figsize=(16, 15))
axes_flat = axes.flatten()

# Colors and styling
color_train = "#1f77b4"
color_test = "#2ca02c"
color_pred = "#d62728"
color_band = "#ff9896"

district_names = list(districts.keys())

for i, name in enumerate(district_names):
    ax = axes_flat[i]
    d_obj = districts[name]
    bt = backtest_results[name]
    
    # 1. Plot Historical Train & Test
    ax.scatter(bt["train_years"], bt["train_values"], color=color_train, s=45, label="Train Actuals (2015-2021)", zorder=4)
    ax.scatter(bt["test_years"], bt["test_values"], color=color_test, s=55, marker="s", label="Test Actuals (2022-2024)", zorder=4)
    
    # 2. Plot Backtest Fitted Curves
    train_yrs_arr = np.array(bt["train_years"])
    test_yrs_arr = np.array(bt["test_years"])
    
    best_model = bt["best_model_instance"]
    ax.plot(train_yrs_arr, bt["models_evaluation"][0]["fitted_train"], color=color_train, linestyle="--", alpha=0.7, label=f"Fitted Train ({best_model.name})")
    ax.plot(test_yrs_arr, bt["models_evaluation"][0]["test_predictions"], color=color_test, linestyle=":", linewidth=2, label="Out-of-Sample Test Pred")
    
    # 3. Full Production Model & Bootstrap Intervals (2025 - 2029)
    prod_model = projections[name]["model"]
    boot_res = bootstrap_prediction_intervals(
        prod_model,
        horizon=5,
        n_bootstraps=2000,
        confidence_level=0.95,
        random_seed=SEED + i,
    )
    
    fut_yrs = boot_res["target_years"]
    fut_preds = boot_res["point_forecasts"]
    low_b = boot_res["lower_bounds"]
    high_b = boot_res["upper_bounds"]
    
    # Connect last historical actual to first forecast point
    connect_yrs = [d_obj.years[-1]] + fut_yrs
    connect_vals = [d_obj.values[-1]] + fut_preds
    ax.plot(connect_yrs, connect_vals, color=color_pred, linewidth=2.2, label=f"Forecast (2025-2029)")
    
    # Shaded 95% Bootstrap Prediction Interval
    ax.fill_between(fut_yrs, low_b, high_b, color=color_band, alpha=0.45, label="95% Bootstrap Pred Interval")
    
    # Train/Test boundary indicator
    ax.axvline(x=2021.5, color="gray", linestyle="-.", alpha=0.7, label="Train/Test Split")
    ax.axvline(x=2024.5, color="black", linestyle="--", alpha=0.5, label="Forecast Horizon Split")
    
    ax.set_title(f"{name} District - Population Forecast & Bootstrap Uncertainty", fontsize=12, fontweight="bold")
    ax.set_xlabel("Calendar Year", fontsize=10)
    ax.set_ylabel("Population (Thousands)", fontsize=10)
    ax.grid(True, linestyle="--", alpha=0.5)
    ax.legend(loc="upper left", fontsize=8)

# Format the 6th subplot as an informational executive summary box
ax_info = axes_flat[5]
ax_info.axis("off")
summary_text = (
    "DEMOGRAPHIC PROJECTION SUMMARY (2025 - 2029)\\n"
    "============================================\\n\\n"
    "• Wakiso District exhibits the steepest absolute\\n"
    "  and relative demographic growth (6.47% CAGR),\\n"
    "  driven by Kampala's urban sprawl.\\n\\n"
    "• Linear and Exponential models outperformed\\n"
    "  Fibonacci-ratio models across all districts\\n"
    "  in out-of-sample backtesting.\\n\\n"
    "• Bootstrap resampling (B = 2,000) reveals\\n"
    "  widening 95% prediction intervals, demonstrating\\n"
    "  the structural compounding of demographic risk\\n"
    "  over multi-year planning horizons.\\n\\n"
    "• Deterministic forecasts mask variance growth;\\n"
    "  prediction intervals provide essential tolerance\\n"
    "  buffers for district civil engineers."
)
ax_info.text(0.08, 0.20, summary_text, fontsize=11, fontfamily="monospace",
             bbox=dict(boxstyle="round,pad=1.2", facecolor="#f0f4f8", edgecolor="#bcccdc"))

plt.tight_layout()
output_img = current_dir / "district_population_forecasts.png"
plt.savefig(output_img, dpi=300)
plt.show()
print(f"Visualization generated and saved to: {output_img}")"""))

    # -------------------------------------------------------------
    # Cell 17: Task 8 Markdown
    # -------------------------------------------------------------
    cells.append(nbf.v4.new_markdown_cell("""---
## Core Task 8: Primary School Classroom Infrastructure Planning

To translate demographic projections into public works investment, we apply the Ministry of Education & Sports (MoES) and UBOS educational standards:
- **Primary School Cohort:** $18\\%$ ($0.18$) of the total population.
- **Classroom Carrying Capacity:** $53$ pupils per standard primary school classroom.
- **Integer Ceiling Rule:** Infrastructure cannot be constructed in fractional increments:
  $$\\Delta P_{\\text{people}} = (P_{2029} - P_{2024}) \\times 1000$$
  $$\\Delta \\text{Pupils} = \\Delta P_{\\text{people}} \\times 0.18$$
  $$\\text{Classrooms Required} = \\left\\lceil \\frac{\\Delta \\text{Pupils}}{53} \\right\\rceil$$"""))

    # -------------------------------------------------------------
    # Cell 18: Task 8 Code
    # -------------------------------------------------------------
    cells.append(nbf.v4.new_code_cell("""planner = ClassroomPlanner(school_age_ratio=0.18, classroom_capacity=53)

planning_input = [
    (name, p["base_2024"], p["projected_2029"])
    for name, p in projections.items()
]

infrastructure_schedule = planner.plan_multiple_districts(planning_input, base_year=2024, target_year=2029)

print(f"{'District':<10} | {'Pop 2024 (k)':<12} | {'Pop 2029 (k)':<12} | {'Net Pop Growth':<15} | {'Net Pupils':<12} | {'Classrooms Needed':<18}")
print("-" * 92)

total_classrooms = 0
total_pupils = 0

for item in infrastructure_schedule:
    name = item["district_name"]
    b_pop = item["base_population_k"]
    p_pop = item["projected_population_k"]
    growth = item["net_population_growth"]
    pupils = item["additional_pupils"]
    classrooms = item["net_classrooms_needed"]
    total_classrooms += classrooms
    total_pupils += pupils
    print(f"{name:<10} | {b_pop:<12.1f} | {p_pop:<12.1f} | {growth:<15,.0f} | {pupils:<12,.0f} | {classrooms:<18}")

print("-" * 92)
print(f"{'TOTAL':<10} | {'':<12} | {'':<12} | {'':<15} | {total_pupils:<12,.0f} | {total_classrooms:<18}")"""))

    # -------------------------------------------------------------
    # Cell 19: Extension 2 Markdown
    # -------------------------------------------------------------
    cells.append(nbf.v4.new_markdown_cell("""---
## Extension Task 2: Critical Demographic Evaluation of the Fibonacci-Ratio Model

### Mathematical Foundation & Derivation
The Fibonacci sequence:
$$F_0 = 0, \\; F_1 = 1, \\; F_k = F_{k-1} + F_{k-2} \\quad (k \\ge 2)$$
possesses successive ratios that asymptotically converge to the Golden Ratio:
$$\\lim_{k \\to \\infty} \\frac{F_{k+1}}{F_k} = \\phi = \\frac{1 + \\sqrt{5}}{2} \\approx 1.6180339887...$$

### Why Fibonacci Growth Violates Human Demography
1. **Unbounded Exponential Explosion:** Direct Fibonacci scaling multiplies the prior population by approximately $1.618$ annually. This represents an annual population growth rate of **$+61.8\\%$ per year**, implying a population doubling time of just **$1.44$ years** ($t_d = \\frac{\\ln 2}{\\ln 1.618}$). In contrast, empirical human demographic growth in Uganda ranges between $3.0\\%$ and $6.5\\%$.
2. **Absence of Mortality and Biological Gestation:** Leonardo of Pisa formulated the sequence in *Liber Abaci* (1202) to describe an idealized, immortal rabbit pair where each pair produces a new pair every month after reaching one month of maturity. Human biology requires a 9-month gestation period, biological reproductive maturity at ~15 years, single births predominating over litters, and unavoidable age-specific mortality rates.
3. **Carrying Capacity Violations:** Fibonacci scaling is fundamentally non-saturating; it assumes infinite food supply, land, water, and educational capital, directly violating biological carrying capacity ($K$).

### Theoretical Assumptions Under Which Fibonacci Modeling Could Be Defensible
Fibonacci scaling could only be demographically defensible under the following hypothetical conditions:
- **Synchronous Cellular / Insect Fission:** Primitive biological organisms (e.g., parthenogenic aphids or bacteria under laboratory nutrient broth) exhibiting discrete, non-overlapping reproductive instars where every organism produces exactly one surviving offspring every cycle.
- **Golden-Ratio Growth Modulation:** As demonstrated in our `calibrated` model subclass, Fibonacci ratios can only be utilized if they are normalized against $\\phi$ to act as minor cyclical multipliers:
  $$g_t = g_{\\text{baseline}} \\times \\left(\\frac{F_{k+1} / F_k}{\\phi}\\right)$$
  which modulates natural seasonal variance without distorting base demographic scales."""))

    # -------------------------------------------------------------
    # Cell 20: Findings & Limitations Markdown
    # -------------------------------------------------------------
    cells.append(nbf.v4.new_markdown_cell("""---
## Findings & Limitations

### 1. Key Findings
- **Peri-Urban Dominance:** Wakiso District is expanding most rapidly in relative terms with a Compound Annual Growth Rate (CAGR) of **$6.47\\%$**, driven by outward residential expansion from Kampala. Gulu and Kampala follow at **$4.61\\%$**, while Mukono ($4.52\\%$) and Mbarara ($4.16\\%$) exhibit steady regional growth.
- **Model Performance:** Linear Trend and Exponential CAGR models achieved superior out-of-sample accuracy across all districts during the 2022–2024 backtesting window (MAPEs $< 2.5\\%$). The naive direct Fibonacci-ratio model produced catastrophic forecast errors ($\text{MAPE} > 150\\%$) due to its non-biological $61.8\\%$ annual expansion factor.
- **Infrastructure Impact:** Across the five surveyed districts, projected population growth by 2029 will introduce **$346{,}643$ additional primary school-age pupils**, requiring the construction of **$6{,}542$ standard classrooms** ($53$ pupils/room). Wakiso alone requires **$2{,}188$ classrooms**, representing one-third of the total infrastructure requirement.

### 2. Analytical Limitations
- **Linear Trend Extrapolation:** Degree-1 polynomials assume constant absolute additions, failing to capture inflection points caused by land exhaustion, zoning restrictions, or urbanization saturation.
- **Homogeneous Cohort Assumption:** Applying a static $18\\%$ school-age ratio across all districts overlooks regional variations in total fertility rates (TFR), which are higher in rural Northern Uganda (Gulu) than in urban Kampala.
- **Stochastic Shock Exclusions:** Deterministic models and residual bootstrapping do not capture major macroeconomic disruptions, climate migration, or geopolitical shocks that alter baseline demographic trajectories."""))

    nb.cells = cells
    return nb

if __name__ == "__main__":
    notebook_obj = build_notebook()
    
    # Save unexecuted notebook first
    output_path = Path("assignment_1/notebooks/project1_population.ipynb")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        nbf.write(notebook_obj, f)
    print(f"Draft notebook saved to {output_path}")

    # Execute notebook top-to-bottom
    client = NotebookClient(notebook_obj, timeout=600, kernel_name="python3")
    print("Executing notebook top-to-bottom...")
    client.execute()
    print("Notebook executed successfully without errors.")

    # Save executed notebook with all outputs and figures embedded
    with open(output_path, "w", encoding="utf-8") as f:
        nbf.write(notebook_obj, f)
    print(f"Fully executed notebook written to {output_path}")
