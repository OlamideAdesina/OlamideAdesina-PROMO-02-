"""
Streamlit dashboard: Predicting Health Outcomes in Sunderland
Using Environmental, Socioeconomic and Digital Inclusion Data

Run with:  streamlit run app.py
(run from inside the 'app' folder, or adjust DATA_DIR / OUT_DIR below)
"""
import json
import os

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

BASE = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE, "data")
OUT_DIR = os.path.join(BASE, "outputs")

st.set_page_config(page_title="Sunderland Health Outcomes", layout="wide", page_icon="🏥")

# ---------------------------------------------------------------- loaders
@st.cache_data
def load_national():
    df = pd.read_csv(f"{DATA_DIR}/national_la_dataset.csv")
    with open(f"{DATA_DIR}/lad_eng_clean.geojson") as f:
        geo = json.load(f)
    return df, geo


@st.cache_data
def load_sunderland():
    df = pd.read_csv(f"{DATA_DIR}/sunderland_lsoa_dataset.csv")
    with open(f"{DATA_DIR}/lsoa_sunderland_clean.geojson") as f:
        geo = json.load(f)
    return df, geo


@st.cache_data
def load_air_quality():
    daily = pd.read_csv(f"{DATA_DIR}/air_quality_daily.csv", parse_dates=["date"])
    monthly = pd.read_csv(f"{DATA_DIR}/air_quality_monthly.csv")
    return daily, monthly


@st.cache_data
def load_model_outputs(prefix):
    metrics = json.load(open(f"{OUT_DIR}/{prefix}_metrics.json"))
    importances = json.load(open(f"{OUT_DIR}/{prefix}_importances.json"))
    test_preds = json.load(open(f"{OUT_DIR}/{prefix}_test_predictions.json"))
    full_preds = pd.read_csv(f"{OUT_DIR}/{prefix}_predictions_full.csv")
    corr = pd.read_csv(f"{OUT_DIR}/{prefix}_correlation_matrix.csv", index_col=0)
    return metrics, importances, test_preds, full_preds, corr


NAT_FEATURE_LABELS = {
    "employment_rate": "Employment rate (%)",
    "econ_inactivity_rate": "Economic inactivity rate (%)",
    "gdhi_per_head": "Household income per head (£)",
    "child_poverty_pct": "Child poverty, after housing costs (%)",
    "no_qualifications_pct": "No qualifications (%)",
    "level3_plus_qual_pct": "Level 3+ qualifications (%)",
    "fuel_poverty_pct": "Fuel poverty (%)",
    "housing_affordability_ratio": "Housing affordability ratio",
    "avg_house_price": "Average house price (£)",
    "ghg_emissions": "Greenhouse gas emissions",
    "air_pollution_regulating": "Air pollution regulating (ecosystem service)",
    "domestic_gas_consumption": "Domestic gas consumption",
    "domestic_elec_consumption": "Domestic electricity consumption",
    "coverage_4g_pct": "4G coverage (%)",
    "coverage_5g_pct": "5G coverage (%)",
    "gigabit_broadband_pct": "Gigabit broadband availability (%)",
    "premises_below_30mbps_pct": "Premises below 30Mbps (%)",
    "adult_obesity_pct": "Adult obesity prevalence (%)",
    "cigarette_smokers_pct": "Cigarette smokers (%)",
    "population_density": "Population density",
    "median_age": "Median age",
    "job_density": "Job density",
}

SUN_FEATURE_LABELS = {
    "income_score": "Income deprivation score",
    "employment_score": "Employment deprivation score",
    "education_score": "Education, Skills & Training score",
    "crime_score": "Crime score",
    "housing_barriers_score": "Barriers to Housing & Services score",
    "living_environment_score": "Living Environment score",
    "pct_population_0_15": "Population aged 0-15 (%)",
    "pct_population_60_plus": "Population aged 60+ (%)",
}

MODEL_COLORS = {"Linear Regression": "#4C72B0", "Random Forest": "#55A868", "XGBoost": "#C44E52"}

# ---------------------------------------------------------------- sidebar
st.sidebar.title("Sunderland Health Outcomes")
st.sidebar.markdown(
    "Predicting health outcomes using **environmental**, **socioeconomic** and "
    "**digital inclusion** data.\n\n"
    "Data: ONS Explore Local Statistics · English Indices of Deprivation 2025 · "
    "UK-AIR Sunderland Silksworth monitoring station."
)
page = st.sidebar.radio(
    "Section",
    ["Overview", "National comparison (LA-level)", "Sunderland deep-dive (LSOA-level)",
     "Air quality context", "Data & methodology"],
)

# ================================================================== OVERVIEW
if page == "Overview":
    st.title("Predicting Health Outcomes in Sunderland")
    st.markdown(
        """
This dashboard predicts health outcomes using a combination of **environmental**,
**socioeconomic**, and **digital inclusion** indicators, combining national ONS data
with Sunderland-specific deprivation and air quality data.

Because the available data has genuinely different levels of geographic detail
(see *Data & methodology*), the analysis works at **two complementary scales**:
"""
    )
    c1, c2 = st.columns(2)
    with c1:
        st.subheader("🇬🇧 National model")
        st.markdown(
            "**Target:** Preventable cardiovascular mortality rate (per 100,000)\n\n"
            "**Scale:** ~270 English Local Authorities\n\n"
            "**Predictors:** income, employment, education, housing, fuel poverty, "
            "greenhouse gas emissions, air pollution regulation, domestic energy use, "
            "4G/5G/broadband coverage, obesity & smoking prevalence, demographics.\n\n"
            "This gives a large, genuinely varying sample for robust ML model comparison, "
            "with Sunderland highlighted throughout."
        )
    with c2:
        st.subheader("Sunderland deep-dive")
        st.markdown(
            "**Target:** Health Deprivation & Disability score (IoD2025)\n\n"
            "**Scale:** 185 Lower-layer Super Output Areas (LSOAs) within Sunderland\n\n"
            "**Predictors:** income, employment, education, crime, housing barriers, "
            "living environment deprivation, population structure.\n\n"
            "This gives real within-Sunderland, ward-level geographic detail and maps."
        )

    st.divider()
    st.subheader("Model performance at a glance")
    im = st.container()
    colA, colB = im.columns(2)
    with colA:
        st.image(f"{OUT_DIR}/model_comparison_r2.png")
    with colB:
        nat_metrics, *_ = load_model_outputs("national")
        sun_metrics, *_ = load_model_outputs("sunderland")
        best_nat = max(nat_metrics, key=lambda k: nat_metrics[k]["test_r2"])
        best_sun = max(sun_metrics, key=lambda k: sun_metrics[k]["test_r2"])
        st.metric("Best national model", best_nat, f"R² = {nat_metrics[best_nat]['test_r2']:.2f}")
        st.metric("Best Sunderland model", best_sun, f"R² = {sun_metrics[best_sun]['test_r2']:.2f}")
        st.caption(
            "Both tasks achieve strong out-of-sample R² (>0.85), indicating that "
            "socioeconomic conditions are highly predictive of these health outcomes -- "
            "consistent with the wider public health literature on the social "
            "determinants of health."
        )

# ================================================================== NATIONAL
elif page == "National comparison (LA-level)":
    st.title("🇬🇧 National comparison: predicting preventable CVD mortality")
    nat_df, nat_geo = load_national()
    metrics, importances, test_preds, full_preds, corr = load_model_outputs("national")

    tab1, tab2, tab3, tab4 = st.tabs(["Map", "Model comparison", "Feature importance", "Explore relationships"])

    with tab1:
        st.markdown("Predicted vs actual preventable cardiovascular mortality rate, by Local Authority.")
        metric_choice = st.selectbox("Map metric", ["Actual rate", "Predicted rate", "Residual (actual - predicted)"])
        col = {"Actual rate": "cvd_mortality_rate", "Predicted rate": "predicted_cvd_mortality_rate",
               "Residual (actual - predicted)": "residual"}[metric_choice]
        map_df = full_preds.dropna(subset=[col])
        fig = px.choropleth_mapbox(
            map_df, geojson=nat_geo, locations="areacd", featureidkey="properties.areacd",
            color=col, color_continuous_scale="RdBu_r" if col == "residual" else "OrRd",
            mapbox_style="carto-positron", zoom=4.6, center={"lat": 53.0, "lon": -1.5},
            opacity=0.75, hover_name="areanm",
            labels={col: metric_choice},
        )
        fig.update_layout(margin=dict(l=0, r=0, t=0, b=0), height=550)
        st.plotly_chart(fig)
        sund_val = nat_df.loc[nat_df.areanm == "Sunderland", "cvd_mortality_rate"].values
        if len(sund_val):
            st.info(f"Sunderland's actual preventable CVD mortality rate: **{sund_val[0]:.1f} per 100,000** "
                    f"(national median: {nat_df['cvd_mortality_rate'].median():.1f}).")

    with tab2:
        st.subheader("Model comparison")
        met_df = pd.DataFrame(metrics).T
        met_df = met_df.rename(columns={
            "cv_r2_mean": "CV R² (mean)", "cv_r2_std": "CV R² (std)",
            "test_mae": "Test MAE", "test_rmse": "Test RMSE", "test_r2": "Test R²",
        })
        st.dataframe(met_df.style.format("{:.3f}").highlight_max(subset=["Test R²"], color="lightgreen"))

        st.subheader("Predicted vs Actual (held-out test set)")
        tp = pd.DataFrame(test_preds)
        model_pick = st.radio("Model", list(metrics.keys()), horizontal=True, key="nat_model_pick")
        fig2 = px.scatter(tp, x="actual", y=model_pick,
                           labels={"actual": "Actual CVD mortality rate", model_pick: "Predicted"},
                           color_discrete_sequence=[MODEL_COLORS[model_pick]])
        lims = [tp["actual"].min(), tp["actual"].max()]
        fig2.add_trace(go.Scatter(x=lims, y=lims, mode="lines", line=dict(dash="dash", color="grey"), name="Perfect prediction"))
        fig2.update_layout(height=450)
        st.plotly_chart(fig2)

    with tab3:
        st.subheader("Which factors matter most? (permutation importance)")
        model_pick2 = st.radio("Model", list(importances.keys()), horizontal=True, key="nat_imp_model")
        imp = importances[model_pick2]
        imp_df = pd.DataFrame({"feature": list(imp.keys()), "importance": list(imp.values())})
        imp_df["label"] = imp_df["feature"].map(NAT_FEATURE_LABELS)
        imp_df = imp_df.sort_values("importance", ascending=True).tail(15)
        fig3 = px.bar(imp_df, x="importance", y="label", orientation="h",
                      color_discrete_sequence=[MODEL_COLORS[model_pick2]])
        fig3.update_layout(height=550, xaxis_title="Permutation importance (drop in R² when shuffled)", yaxis_title="")
        st.plotly_chart(fig3)
        st.caption(
            "Permutation importance measures how much a model's test-set R² drops when a "
            "feature's values are randomly shuffled -- larger drop = more important. This is "
            "computed consistently across all three model types so importances are directly comparable."
        )

    with tab4:
        st.subheader("Correlation heatmap")
        st.image(f"{OUT_DIR}/eda_national_corr_heatmap.png")
        st.subheader("Scatter explorer")
        x_col = st.selectbox("X variable", list(NAT_FEATURE_LABELS.keys()), format_func=lambda c: NAT_FEATURE_LABELS[c])
        fig4 = px.scatter(nat_df, x=x_col, y="cvd_mortality_rate", hover_name="areanm",
                           labels={x_col: NAT_FEATURE_LABELS[x_col], "cvd_mortality_rate": "CVD mortality rate"},
                           trendline="ols", color_discrete_sequence=["#4C72B0"])
        sund = nat_df[nat_df.areanm == "Sunderland"]
        fig4.add_trace(go.Scatter(x=sund[x_col], y=sund["cvd_mortality_rate"], mode="markers",
                                   marker=dict(color="crimson", size=14), name="Sunderland"))
        st.plotly_chart(fig4)

# ================================================================== SUNDERLAND
elif page == "Sunderland deep-dive (LSOA-level)":
    st.title("Sunderland: Health Deprivation across LSOAs")
    sun_df, sun_geo = load_sunderland()
    metrics, importances, test_preds, full_preds, corr = load_model_outputs("sunderland")

    tab1, tab2, tab3, tab4 = st.tabs(["Map", "Model comparison", "Feature importance", "Explore relationships"])

    with tab1:
        metric_choice = st.selectbox(
            "Map metric",
            ["Actual Health Deprivation score", "Predicted score", "Residual (actual - predicted)",
             "Health Deprivation decile", "Income deprivation score", "Employment deprivation score"],
        )
        col_map = {
            "Actual Health Deprivation score": "health_deprivation_score",
            "Predicted score": "predicted_health_deprivation_score",
            "Residual (actual - predicted)": "residual",
            "Health Deprivation decile": "health_deprivation_decile",
            "Income deprivation score": "income_score",
            "Employment deprivation score": "employment_score",
        }
        col = col_map[metric_choice]
        if col in full_preds.columns:
            map_df = full_preds
        else:
            map_df = sun_df
        fig = px.choropleth_mapbox(
            map_df, geojson=sun_geo, locations="lsoa_code", featureidkey="properties.lsoa_code",
            color=col, color_continuous_scale="RdBu_r" if col == "residual" else "OrRd",
            mapbox_style="carto-positron", zoom=10.6,
            center={"lat": 54.88, "lon": -1.43}, opacity=0.8,
            hover_name="lsoa_name" if "lsoa_name" in map_df.columns else None,
            labels={col: metric_choice},
        )
        fig.update_layout(margin=dict(l=0, r=0, t=0, b=0), height=600)
        st.plotly_chart(fig)
        st.caption("Note: 6 of 185 Sunderland LSOAs use boundaries that changed between the 2011 and 2021 "
                   "Census and are not shown on the map (they remain in all tabular analysis/model training).")

    with tab2:
        st.subheader("Model comparison")
        met_df = pd.DataFrame(metrics).T
        met_df = met_df.rename(columns={
            "cv_r2_mean": "CV R² (mean)", "cv_r2_std": "CV R² (std)",
            "test_mae": "Test MAE", "test_rmse": "Test RMSE", "test_r2": "Test R²",
        })
        st.dataframe(met_df.style.format("{:.3f}").highlight_max(subset=["Test R²"], color="lightgreen"))

        st.subheader("Predicted vs Actual (held-out test set)")
        tp = pd.DataFrame(test_preds)
        model_pick = st.radio("Model", list(metrics.keys()), horizontal=True, key="sun_model_pick")
        fig2 = px.scatter(tp, x="actual", y=model_pick,
                           labels={"actual": "Actual Health Deprivation score", model_pick: "Predicted"},
                           color_discrete_sequence=[MODEL_COLORS[model_pick]])
        lims = [tp["actual"].min(), tp["actual"].max()]
        fig2.add_trace(go.Scatter(x=lims, y=lims, mode="lines", line=dict(dash="dash", color="grey"), name="Perfect prediction"))
        fig2.update_layout(height=450)
        st.plotly_chart(fig2)

    with tab3:
        st.subheader("Which factors matter most?")
        model_pick2 = st.radio("Model", list(importances.keys()), horizontal=True, key="sun_imp_model")
        imp = importances[model_pick2]
        imp_df = pd.DataFrame({"feature": list(imp.keys()), "importance": list(imp.values())})
        imp_df["label"] = imp_df["feature"].map(SUN_FEATURE_LABELS)
        imp_df = imp_df.sort_values("importance", ascending=True)
        fig3 = px.bar(imp_df, x="importance", y="label", orientation="h",
                      color_discrete_sequence=[MODEL_COLORS[model_pick2]])
        fig3.update_layout(height=450, xaxis_title="Permutation importance", yaxis_title="")
        st.plotly_chart(fig3)

    with tab4:
        st.subheader("Scatter explorer")
        x_col = st.selectbox("X variable", list(SUN_FEATURE_LABELS.keys()), format_func=lambda c: SUN_FEATURE_LABELS[c])
        fig4 = px.scatter(sun_df, x=x_col, y="health_deprivation_score", hover_name="lsoa_name",
                           labels={x_col: SUN_FEATURE_LABELS[x_col], "health_deprivation_score": "Health Deprivation score"},
                           trendline="ols", color_discrete_sequence=["#C44E52"])
        st.plotly_chart(fig4)

# ================================================================== AIR QUALITY
elif page == "Air quality context":
    st.title("Air quality context: Sunderland Silksworth monitoring station")
    daily, monthly = load_air_quality()
    st.markdown(
        "Hourly air quality readings from the **UK-AIR Sunderland Silksworth** monitoring "
        "station (Jan 2024 - Aug 2026), aggregated to daily and monthly means. This is a "
        "single fixed monitoring point rather than a spatial (ward/LSOA) dataset, so it is "
        "shown here as environmental context rather than as a model predictor."
    )
    pollutant = st.selectbox("Pollutant", ["pm25", "pm10", "no2", "nox", "no", "ozone"],
                              format_func=lambda p: {"pm25": "PM2.5", "pm10": "PM10", "no2": "Nitrogen dioxide",
                                                      "nox": "Nitrogen oxides", "no": "Nitric oxide", "ozone": "Ozone"}[p])
    fig = px.line(daily, x="date", y=pollutant, title=f"Daily mean {pollutant.upper()} (µg/m³)")
    fig.update_layout(height=400)
    st.plotly_chart(fig)

    c1, c2, c3 = st.columns(3)
    c1.metric("Mean PM2.5", f"{daily['pm25'].mean():.1f} µg/m³", help="WHO interim target 3 (annual): 15 µg/m³")
    c2.metric("Mean NO2", f"{daily['no2'].mean():.1f} µg/m³", help="UK annual objective: 40 µg/m³")
    c3.metric("Days PM2.5 > 15 µg/m³", f"{(daily['pm25']>15).sum()} / {len(daily)}")

    st.image(f"{OUT_DIR}/eda_air_quality_trend.png")

# ================================================================== METHODOLOGY
else:
    st.title("Data & methodology")
    st.markdown(
        """
### Data sources
| Dataset | Geography | Use in this project |
|---|---|---|
| ONS *Explore Local Statistics* (`all-datasets.csv`) | Local Authority (England, ~330 areas) | National predictors + target (CVD mortality) |
| English Indices of Deprivation 2025, File 7 | LSOA (England, ~33,000 areas -> 185 in Sunderland) | Sunderland LSOA-level target + predictors |
| UK-AIR Sunderland Silksworth station (`SUN2_2024/25/26.csv`) | Single monitoring point | Local environmental context (time trend) |

### Why two scales, not one?
The brief asks for a ward/LSOA-level model merging environmental, socioeconomic and
digital-inclusion data for Sunderland. In practice, **only the deprivation (IoD2025) data
is available at LSOA level** for Sunderland; the ONS digital-inclusion and environmental
indicators (4G/5G/broadband, emissions, energy use) are published at Local Authority level
(i.e. one figure for the whole of Sunderland), and the air quality data comes from a single
monitoring station. Rather than fabricate spatial variation that isn't in the source data,
this project:

1. Builds a genuine **national LA-level model** (~270 English authorities) so the full
   range of environmental + socioeconomic + digital-inclusion indicators can be used with
   real cross-sectional variation, and situates Sunderland within that national picture.
2. Builds a genuine **Sunderland LSOA-level model** using the one dataset that *is*
   available at that resolution (IoD2025), predicting the Health Deprivation & Disability
   score from the other deprivation domains -- giving real ward-level maps and detail.
3. Uses the air quality station data as **supporting local context** (EDA/time trend),
   since a single point cannot be meaningfully merged onto 185 different LSOAs.

### Modelling approach
- **Models compared:** Linear Regression, Random Forest, XGBoost
- **Split:** 80/20 train/test, 5-fold cross-validation on the training set
- **Metrics:** MAE, RMSE, R² (test set) + CV R² (mean ± std)
- **Feature importance:** permutation importance on the held-out test set (model-agnostic,
  comparable across all three model types)
- National model: E10 (county council) rows were excluded to avoid double-counting, since
  county areas are aggregates of their constituent district (E07) authorities already in
  the sample.

### Limitations
- LSOA boundaries changed slightly between the 2011 and 2021 Census; 6 of Sunderland's 185
  LSOAs (2021 boundaries) do not have a matching 2011 boundary in the mapping file used, so
  they are excluded from the map (but included in all tabular analysis and modelling).
- The Health Deprivation & Disability score is itself one of seven domains combined into
  the overall Index of Multiple Deprivation, so predictor domains are not fully independent
  of the target by construction -- results should be read as describing established
  deprivation associations, not strictly causal relationships.
- Air quality reflects one fixed location and cannot be spatially disaggregated across
  Sunderland with the data provided.
"""
    )
