# Assignment 4: Rainfall Pattern & Crop Suitability Analyser

**Course:** MSCS & MSDS – Object-Oriented Programming with Python  
**Term:** Advent 2026  
**Author:** Daniel Mwiine  
**Student ID:** B41130  
**Registration Number:** S26M25/001  
**Module:** Mini-Project 4 (`assignment_4`)  

---

## 1. Executive Summary & Problem Context

An agricultural extension service in Uganda requires quantitative classification of regional rainfall regimes to guide smallholder farmers on planting calendars, crop diversification, and drought/waterlogging risk management.

Using 12-month precipitation averages across three representative Ugandan agro-ecological stations:
- **Kampala:** Lake Victoria Basin zone ($1{,}600\text{ mm}$ annual cumulative)
- **Gulu:** Northern Savannah sub-humid zone ($1{,}388\text{ mm}$ annual cumulative)
- **Mbarara:** Southwestern Cattle Corridor & Highland zone ($1{,}040\text{ mm}$ annual cumulative)

This package implements:
1. `Region` domain model computing cumulative rainfall, monthly means, wettest/driest months, and precipitation CV.
2. `CropRule` class formalizing monthly moisture tolerances for **Maize** ($80-180\text{ mm}$), **Beans** ($60-150\text{ mm}$), and **Robusta Coffee** ($100-220\text{ mm}$) based on FAO / NARO guidelines.
3. Algorithmic correction of a legacy student mistake using scalar `math.cos()`, establishing true vector cosine similarity in $\mathbb{R}^{12}$ and validating against `scipy.spatial.distance.cosine`.
4. Metric space comparisons (Cosine, Pearson, Euclidean).
5. Automated modal peak identification via `scipy.signal.find_peaks` with circular calendar wrapping, classifying Unimodal vs Bimodal regimes against UNMA benchmarks.
6. A concise, actionable operational advisory note for smallholder farmers in Gulu.
7. Extension: 10-year monthly precipitation variance synthesis and boxplots.

---

## 2. Directory Structure

```text
assignment_4/
├── src/
│   ├── __init__.py
│   └── climate/
│       ├── __init__.py           # Package API
│       ├── models.py             # Region and CropRule classes
│       ├── metrics.py            # Vector cosine similarity, metric space comparisons
│       └── peaks.py              # Modal peak detection & multi-year climate generator
├── tests/
│   ├── __init__.py
│   └── test_climate.py           # 8 comprehensive unit tests
├── notebooks/
│   ├── project4_climate.ipynb    # Executed Jupyter Notebook
│   ├── build_notebook.py         # Notebook execution pipeline
│   ├── rainfall_and_crop_suitability.png # Dual-panel visualization
│   └── gulu_multiyear_boxplots.png       # 10-year variance boxplots
└── README.md
```

---

## 3. Core Findings & Mathematical Verification

### 3.1 Algorithmic Correction
- **Legacy Error:** Previous cohorts attempted vector similarity using `math.cos(vector)`, which computes the scalar trigonometric cosine of a 1D angle in radians.
- **Correct Formulation:** Vector cosine similarity in $\mathbb{R}^{12}$:
  $$S_C(\mathbf{u}, \mathbf{v}) = \frac{\mathbf{u} \cdot \mathbf{v}}{\|\mathbf{u}\|_2 \|\mathbf{v}\|_2}$$
  Validated to machine precision against `1 - scipy.spatial.distance.cosine(u, v)`.

### 3.2 Metric Space Comparison
- **Cosine Similarity:** Scale-invariant angular metric. Classifies Kampala and Mbarara as highly similar ($S_C = 0.942$) because their proportional seasonal shape is nearly identical, despite Mbarara receiving $35\%$ less total rainfall.
- **Euclidean Distance:** Sensitive to absolute volumetric differences ($d_{\text{Kampala, Mbarara}} = 197.86\text{ mm}$).
- **Pearson Correlation:** Centers vectors around their monthly means, isolating pure co-movement.

### 3.3 Modal Climatological Classification
- **Gulu:** Unimodal (Peak in August: $215\text{ mm}$; single continuous wet season April–October). Matches UNMA Northern Savannah zone.
- **Kampala:** Bimodal (Peaks in May: $220\text{ mm}$ and December: $130\text{ mm}$). Matches UNMA Lake Victoria Basin zone.
- **Mbarara:** Bimodal (Peaks in April: $140\text{ mm}$ and October: $125\text{ mm}$ with dry spells in June/July: $20-25\text{ mm}$). Matches UNMA Southwestern zone.

---

## 4. Verification
```bash
python -m pytest assignment_4/tests/test_climate.py -v
```
*Result: 8 passed in 1.42s.*

---

## 5. Findings & Limitations

The most interesting challenge in this project was the circular peak detection problem. Standard scipy.signal.find_peaks treats the 12-month array as a linear sequence � so if rainfall is rising in December heading into January, the December peak gets missed because the algorithm doesn't see a decline on its right side. My fix was to tile the array three times and detect peaks in the middle copy, then map detected peaks back to the original months. After testing this against all three regions, the results matched Uganda's recognized climatological zones from UNMA records, which I took as a meaningful validation.

The cosine similarity vs Euclidean distance comparison was also illuminating. Kampala and Mbarara look very similar under cosine similarity ( = 0.942$) because their rainfall *patterns* (shape over the year) are proportionally alike, but Kampala receives about 54% more total rainfall. This matters practically: a farmer using only pattern similarity to plan might apply Kampala planting calendars in Mbarara, but underestimate the need for drought-tolerant varieties during Mbarara's more severe dry spells.

Key limitations: all rainfall data used is illustrative (from the spec), not real station data. Real UNMA or CHIRPS data would likely show more inter-annual variability than the synthetic 10-year records I generated for the extension task. The crop suitability rules are also simplified monthly thresholds � actual agronomy accounts for soil type, temperature, humidity, and cumulative growing degree days.

---

## 6. AI Disclosure

See [README.md](../README.md#ai-use-disclosure) for the full disclosure. For this project, AI helped me:
- Explain why scipy.signal.find_peaks misses circular boundary peaks and suggest the array tiling approach.
- Format the heatmap suitability chart with proper color mapping and category labels.
- Cross-check my cosine similarity implementation formula before I validated it against scipy.


