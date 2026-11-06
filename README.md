# Which Struggling Firms Recover? Predicting the Recovery of Marginal Firms under Economic Shocks

**Team project, Financial Big Data Analytics Bootcamp, Ubion (Jun–Jul 2022) · Team *Gisa-hoesaeng* ("back from the dead"), 4 members · I was team lead** · [한국어](README.ko.md)

When an economic shock pushes companies into distress, which ones climb back out the next year, and do the warning signs differ from calm periods? We studied Korean companies subject to external audit, defined "marginal firms" in two ways, and predicted which of them would recover, comparing three crisis periods with three calm periods.

> **Data notice.** The original analysis used TS2000, a licensed financial-statement database that cannot be redistributed, and the original code was not preserved. This repository is a **reconstruction** from the final presentation and the team's notes. The notebook runs the full pipeline on a **synthetic panel** with the same line items, so every number it produces is simulated. The original results below come from the final presentation.

## Research design

| | |
|---|---|
| Firms | Korean companies subject to external audit (*oegam* firms), excluding financial firms and special-purpose companies |
| Marginal firm | (1) Interest coverage below 1 for two consecutive years (Bank of Korea), or (2) operating loss for two consecutive years (credit guarantee agencies) |
| Recovery (target) | In the third year, interest coverage of at least 1, or a return to operating profit |
| Shock windows | Dot-com bust 2000–01, global financial crisis 2007–08, European debt crisis 2011–12 (two years up to each business-cycle trough) |
| Calm windows | 2014–15, 2015–16, 2016–17 |
| Datasets | 2 definitions × 6 windows = 12 |
| Candidate features | 28: 20 financial ratios (growth, profitability, activity, stability), 5 Bank of Korea vulnerability ratios (leverage, liquidity), 3 dummies (sales tier, industry, asset tier) |
| Feature selection | Logit model per dataset: keep the more significant variable of any pair correlated above 0.8, then backward elimination at p < 0.05 |
| Models | Logistic regression, decision tree, kNN, ANN, random forest, XGBoost, DNN; grid-search tuning; 80/20 split; F1 and ROC-AUC |
| Robustness | Train on one window, test on another window of the same period type |

## Original results (TS2000 data)

Mean F1 / ROC-AUC across the three windows of each period.

| Model | Interest coverage: shock | Interest coverage: calm | Operating loss: shock | Operating loss: calm |
|---|---|---|---|---|
| Logistic | **0.855 / 0.901** | **0.888 / 0.919** | 0.761 / 0.827 | 0.848 / 0.889 |
| Decision Tree | 0.677 / 0.766 | 0.734 / 0.804 | 0.803 / **0.851** | 0.792 / 0.866 |
| kNN | 0.478 / 0.620 | 0.602 / 0.705 | 0.785 / 0.820 | 0.778 / 0.825 |
| ANN | 0.470 / 0.662 | 0.757 / 0.818 | 0.750 / 0.787 | 0.855 / 0.891 |
| Random Forest | 0.757 / 0.843 | 0.814 / 0.861 | 0.803 / 0.834 | 0.841 / 0.875 |
| XGBoost | 0.764 / 0.830 | 0.831 / 0.875 | **0.813** / 0.846 | 0.846 / 0.881 |
| DNN | 0.665 / 0.756 | 0.782 / 0.869 | 0.793 / 0.831 | **0.885 / 0.945** |

Final dataset sizes ranged from 177 to 1,032 firms, with recovery rates of 25–45%.

**Findings**

- Profitability ratios were significant in every dataset; sales, asset and borrowing-related ratios in most.
- The marginal-firm definition changed the selected features more than the period type did.
- Ensemble models (random forest, XGBoost) performed well across all datasets and were the most stable in the cross-window check; neural networks varied the most.
- Policy suggestions: monitor the commonly significant ratios, broaden the criteria used to flag marginal firms, and combine ensemble and deep-learning models.

## What the reconstruction adds

[`notebooks/marginal_firm_recovery.ipynb`](notebooks/marginal_firm_recovery.ipynb) rebuilds the whole workflow in a form that can be rerun on licensed data: computing the ratios from raw line items, labeling marginal and recovered firms, logit-based feature selection, six tuned models and the cross-window check.

It also documents a design rule that matters for any prediction of this kind: **features must come only from the years before the outcome.** In the original analysis, the ratios came from the two marginal years, and the third year was used only to assign the recovery label. The reconstruction follows the same rule. To show why it matters, the notebook also rebuilds the datasets with outcome-year ratios: on the synthetic data, logistic regression's ROC-AUC jumps from about 0.77 to 0.99, because those ratios already describe the outcome.

## Repository structure

```
├── notebooks/
│   └── marginal_firm_recovery.ipynb   # Full pipeline on synthetic data (English)
├── src/
│   └── synthetic_ts2000.py            # Synthetic firm-year panel with TS2000-style line items
├── docs/
│   └── final_presentation_ko.pdf      # Final presentation (Korean)
├── requirements.txt
├── README.md
└── README.ko.md
```

## Reproducing

```bash
pip install -r requirements.txt
```

Run the notebook from the `notebooks/` folder. It generates the synthetic panel itself (fixed seed). To use real data, set `TS2000_PATH` to a firm-year CSV with the columns produced by `src/synthetic_ts2000.py`. Tested with Python 3.13, scikit-learn 1.9, statsmodels 0.15 and XGBoost 3.4.

## Limitations

- Annual year-end statements cannot capture shocks that start mid-year.
- Two-year windows may not fit every crisis, and the lag between a shock and distress or recovery was not modeled.
- Although the defining variables (interest coverage, operating income) were excluded, closely related ratios remained among the features.
- The synthetic panel reproduces the data structure, not the real relationships, so its results should not be compared with the original numbers.

## Team

Four members. I led the project: planning the study design, modeling and analyzing the results. Teammates handled data collection and preprocessing, modeling, robustness checks, code modularization and the presentation materials.
