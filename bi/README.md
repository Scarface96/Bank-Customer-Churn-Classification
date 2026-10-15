# 📊 Bank Customer Churn: Tableau & Power BI kit

Everything needed to rebuild this project as a Power BI and Tableau dashboard in under an hour: model-ready data, ready-to-paste DAX measures and Tableau calculated fields, a matching colour theme, a page-by-page layout, and the exact numbers your finished dashboard should show.

> **Interactive web version:** [scarface96.github.io/Bank-Customer-Churn-Classification](https://scarface96.github.io/Bank-Customer-Churn-Classification/). The kit uses the same cleaned data, so the numbers will match.

## What's in this folder

| File | What it is |
|---|---|
| `data/fact_customer.csv` | one row per customer (10,000): profile, balance, products, activity, `Exited`, plus `churn_risk` from the best model, `risk_band` and `in_test_set` |
| `data/model_comparison.csv` | one row per model: cross-validated and held-out ROC AUC, PR AUC, accuracy |
| `data/lift_test_set.csv` | one row per risk decile (held-out customers): customers, churners, churn rate, cumulative share of all churners |
| `powerbi/measures.dax` | Every DAX measure, with its number format |
| `powerbi/theme.json` | Report theme in the portfolio's teal-and-amber colours |
| `tableau/calculated_fields.md` | Every calculated field, plus how to connect the tables |
| `tableau/Preferences.tps` | The same colours as Tableau palettes |
| `kpis.json` | Headline numbers, for checking your build |
| `build_bi.py` | Regenerates everything here from the repo's own cleaning code: `python -m bi.build_bi` |

## Dashboard design

Canvas 1280 × 720. Light page background, white cards, teal for the main series, amber to highlight, red only for bad news.

### Page 1: Who leaves
*Which customers churn most?*

- KPI cards: Customers · Churned · Churn Rate · Balance Lost to Churn
- Bar charts (each with a constant line at Overall Churn Rate): Churn Rate by Geography, Gender, age_band, NumOfProducts, IsActiveMember
- Matrix heatmap: rows age_band, columns Geography, values Churn Rate

### Page 2: Model results
*How well can we predict it?*

- Bar chart: model_comparison[test_roc_auc] by model
- Line chart: lift_test_set[share_of_all_churners] by decile (the riskiest 20% of customers hold about 62% of churners)
- Column chart: lift_test_set[churn_rate] by decile
- Text box: note that evaluation uses held-out customers only (in_test_set = 1)

### Page 3: Retention targets
*Who should we call this week?*

- Cards: At-Risk Customers · Balance at Risk
- Table: CustomerId, Geography, Age, Balance, NumOfProducts, churn_risk (data bars), filtered to Exited = 0 and risk_band High/Very high, sorted by churn_risk
- Column chart: Customers by risk_band, legend Exited
- Slicers: Geography, risk_band

## Build it in Power BI

1. **Get data → Text/CSV** and load each file in `bi/data/`. Check that date columns come in as *Date* and IDs as *Whole number* or *Text*.
2. **Home → Enter data**, name the table `_Measures`, load it, then add each measure from `powerbi/measures.dax` and set the format given above it.
3. **View → Themes → Browse for themes** → `powerbi/theme.json`.
4. Build the pages above. Add a title text box on each page with the question it answers.
5. Check your cards against **Check your numbers** below.
6. Save as `Bank Customer Churn.pbix` in the repo root.

## Build it in Tableau

1. Copy `tableau/Preferences.tps` to `Documents/My Tableau Repository/` (or merge its palettes into the one there) and restart Tableau.
2. Connect the data as described at the top of `tableau/calculated_fields.md`.
3. Create the calculated fields in that file.
4. Build one sheet per chart above, then a dashboard per page at 1280 × 720 (Fixed size). Use the *Tony Mulunda* palettes for colour.
5. Check your BANs against **Check your numbers** below.
6. **File → Save to Tableau Public**, then save the packaged workbook (`.twbx`) in the repo root.

## Check your numbers

Your finished dashboard (no filters applied) should show:

| Measure | Value |
|---|---|
| Customers | 10,000 |
| Churned customers | 2,037 |
| Churn rate | 20.4% |
| Balance held by churned customers | 185,588,095 |
| Churn rate, Germany | 32.4% |
| Churn rate, inactive members | 26.9% |
| Best model | Gradient boosting |
| Best model ROC AUC (held-out) | 0.863 |
| Share of churners in riskiest 20% (held-out) | 61.7% |

If a number is off, the usual causes are a relationship pointing the wrong way, a date column loaded as text, or a missing join.

## Notes on the data

- `churn_risk` comes from the best model in the same comparison as the live report (gradient boosting). Customers it was trained on have optimistic scores, so judge model quality on `in_test_set = 1` only. For targeting, the scores are fine to rank everyone.

## Publish and show it off

- **Tableau Public:** after saving, copy the share link and add it to the main README next to the live web report.
- **Power BI:** *Publish to web* needs a work or school account. Without one, add a screenshot of each page to `docs/images/` and link the `.pbix` from the README.
- Take a 1280 × 720 screenshot of page 1 and put it at the top of the main README, so it's the first thing a recruiter sees.
