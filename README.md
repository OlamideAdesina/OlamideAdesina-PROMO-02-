# OlamideAdesina-PROMO-02-

# Final year project

# Predicting Health Outcomes Using Environmental, Socioeconomic and Digital Inclusion Data in Sunderland

## 1. What the data actually supports

Six files were supplied. After inspection:

| File | Actual geography | Content |
|---|---|---|
| `SUN2_2024/2025/2026.csv` | **One monitoring station** (Sunderland Silksworth) | Hourly PM10, PM2.5, NO, NO2, NOx, Ozone, Jan 2024–Aug 2026 |
| `File_7_IoD2025_...csv` | **LSOA** (England-wide; ~33,000 areas) | English Indices of Deprivation 2025 — Income, Employment, Education, Health, Crime, Housing Barriers, Living Environment domains, deciles/ranks, population |
| `all-datasets.csv` / `all-datasets-E08.xlsx` | **Local Authority** (England-wide; ~330 areas, same data in long CSV / wide Excel form) | ONS "Explore Local Statistics" — 110 indicators covering population, economy, education, digital connectivity, environment, health & wellbeing, housing, business, 1991–2026 |

**Key implication:** the only dataset with genuine sub-Sunderland (ward/LSOA) spatial detail is the deprivation index. Environmental and digital-inclusion indicators in `all-datasets.csv` are one figure for the whole of Sunderland, and the air-quality file is one physical sensor. Rather than force a fictitious LSOA-level merge of data that isn't spatially resolved, the project uses **two complementary, honestly-scoped models**.

## 2. Model A — National (Local Authority level)

- **Target:** Preventable cardiovascular mortality rate (per 100,000, age-standardised), most recent 3-year period (~2022)
- **Sample:** 292 English Local Authorities (E06 unitary, E07 district, E08 metropolitan borough, E09 London borough — county councils (E10) excluded to avoid double-counting districts they contain); 268 with complete predictor data
- **Predictors:** employment rate, economic inactivity, household income, child poverty, qualifications, fuel poverty, housing affordability, house prices, greenhouse gas emissions, air pollution regulation (ecosystem service), domestic gas/electricity use, 4G/5G coverage, gigabit broadband availability, premises below 30Mbps, adult obesity, smoking prevalence, population density, median age, job density

### Results (held-out 20% test set)

| Model | CV R² (train) | Test MAE | Test RMSE | Test R² |
|---|---|---|---|---|
| Linear Regression | 0.76 | 3.20 | 3.82 | **0.85** |
| Random Forest | 0.74 | 3.45 | 4.30 | 0.81 |
| XGBoost | 0.74 | 3.25 | 4.11 | 0.83 |

**Top predictors** (permutation importance, consistent across models): child poverty rate, household disposable income, house prices/affordability, adult obesity. Digital-inclusion indicators (broadband/4G/5G) show comparatively small importance once income and poverty are accounted for — Sunderland already has near-universal 4G/5G coverage (100%), so it isn't a differentiator nationally.

Sunderland's actual rate (34.6 per 100,000) sits close to but modestly above the national median (~28), consistent with its child poverty rate (~30%, well above the national average) and above-average adult obesity prevalence (43%, one of the highest in the sample).

## 3. Model B — Sunderland deep-dive (LSOA level)

- **Target:** Health Deprivation & Disability score (IoD2025) — a validated composite of years-of-life-lost, illness/disability, acute morbidity and mental health, higher = worse health
- **Sample:** all 185 Sunderland LSOAs, no missing data
- **Predictors:** income, employment, education/skills, crime, housing barriers, and living environment deprivation domain scores, plus % population aged 0–15 and 60+

### Results (held-out 20% test set)

| Model | CV R² (train) | Test MAE | Test RMSE | Test R² |
|---|---|---|---|---|
| Linear Regression | 0.84 | 0.16 | 0.20 | 0.93 |
| Random Forest | 0.84 | 0.18 | 0.20 | 0.92 |
| XGBoost | 0.83 | 0.16 | 0.20 | **0.93** |

**Top predictors:** employment deprivation and income deprivation dominate (consistent with the well-established income/employment → health gradient), followed by education. This mirrors national findings at much finer geographic resolution and identifies specific LSOAs within Sunderland with the greatest health deprivation for targeting.

## 4. Deliverables

```
project/
├── PROJECT_SUMMARY.md          <- this file
├── scripts/                    <- data prep, modelling, EDA pipeline (run in order 01→05)
├── data/                       <- cleaned datasets + map boundary files
├── outputs/                    <- model metrics, feature importances, predictions, static charts
└── app/                        <- self-contained Streamlit dashboard (app.py + requirements.txt + data/outputs copies)
```

**To run the dashboard:**
```
cd app
pip install -r requirements.txt
streamlit run app.py
```

The dashboard has 5 sections: Overview, National comparison (interactive choropleth map, model comparison, feature importance, scatter explorer), Sunderland deep-dive (LSOA choropleth map, model comparison, feature importance, scatter explorer), Air quality context (time trend from Silksworth station), and Data & methodology (full caveats).

## 5. Limitations (see also the app's "Data & methodology" page)

- Deprivation domains are components of the overall IMD, so predictors and target are not fully independent by construction — read results as describing established deprivation associations, not strict causality.
- 6 of 185 Sunderland LSOAs have 2021-Census boundaries with no matching 2011 boundary in the mapping source used, so they don't render on the map (still included in all modelling/tabular results).
- Air quality reflects a single fixed sensor and could not be spatially merged into either model as a numeric predictor.

