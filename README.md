# Bias & Fairness Auditor

This project investigates and mitigates potential bias in binary classification models using the **Adult Income dataset**.
The goal is to measure fairness using multiple metrics, experiment with bias mitigation techniques, and analyse the impact on model performance.

## Project Overview

Bias in machine learning models can lead to unequal outcomes across demographic groups.
This project focuses on:

* **Measuring bias** using fairness metrics such as:

  * Statistical Parity Difference
  * Equal Opportunity Difference
  * Calibration
* **Exploring bias sources** through exploratory data analysis (EDA)
* **Applying mitigation strategies** such as *reweighting* to balance outcomes
* **Comparing pre- and post-mitigation results**

## Exploratory Data Analysis

EDA was conducted to examine relationships between **gender**, **race**, **occupation**, **effort level** (hours worked), and **income**.
Key findings include:

* **Blue Collar**: Predominantly male; higher effort correlated with higher income, but women are underrepresented and underpaid.
* **Pink Collar**: Female-dominated at low/normal effort; income gap persists even at high effort.
* **Sales**: Mixed gender distribution, but men’s high-effort roles see greater income gains.
* **Service Protective**: Male-dominated, smaller gap compared to other categories.
* **White Collar**: More balanced gender mix, but men dominate high-effort roles.

These insights informed the **bias score calculation assumptions**, documented in detail in the EDA notebook.

## Bias Score Calculation Assumptions

1. **Clean Dataset** — all missing or inconsistent values handled.
2. **Effort Estimation** — based on hours-per-week, binned into low, normal, and high per occupation using IQR.
3. **Reward Measurement** — reward defined as proportion earning >50K in each gender group.
4. **Occupation Granularity** — categories treated as internally consistent.
5. **Sample Size** — no minimum group size enforced.
6. **Gender Representation** — assumed to reflect real-world patterns.
7. **Interpretation** — bias score < proportion of women indicates underrewarding.

These biases may influence trained ML models by reinforcing existing societal inequalities unless mitigated.

## Technical Approach

1. **Data Processing** (`process_data.py`)

   * Cleaning, encoding categorical variables, and handling missing values.
2. **Model Training** (`model_testing.py`)

   * Logistic Regression baseline.
   * Reweighted Logistic Regression for mitigation.
3. **Bias Mitigation** (`reweighting.py`)

   * Implements instance reweighting to balance protected groups.
4. **Evaluation** (`eval_helper.py`, `fairness_metrics.py`)

   * Accuracy, precision, recall.
   * Fairness metrics before/after reweighting.
5. **Visualization** (`plot_helper.py`)

   * Crosstabs, heatmaps, bias metric comparisons.

## Repository Structure

```
bias_auditor/
│
├── notebooks/              # Jupyter notebooks for EDA & model comparison
├── outputs/                # (Ignored in .gitignore) Model outputs & predictions
├── utils/                  # Helper modules for processing, evaluation & plotting
├── data/                   # Dataset (Adult Income CSV)
├── requirements.txt        # Python dependencies
└── README.md               # Project documentation
```

## Tech Stack

* **Python 3.11+**
* **Pandas, NumPy** — data processing
* **Matplotlib, Seaborn** — visualization
* **scikit-learn** — machine learning models & metrics
* **Jupyter Notebook** — analysis & experimentation
