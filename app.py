"""
app.py — AC Capacity Predictor
================================
A polished Streamlit web application for predicting required Air
Conditioner tonnage from room / environmental parameters.

Run:
    streamlit run app.py
"""

import json
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st

# --------------------------------------------------------------------------
# PAGE CONFIG & GLOBAL STYLE
# --------------------------------------------------------------------------
st.set_page_config(
    page_title="AC Capacity Predictor",
    page_icon="❄️",
    layout="wide",
    initial_sidebar_state="expanded",
)

PRIMARY = "#0EA5B7"      # teal
PRIMARY_DARK = "#0B7285"
ACCENT = "#F97316"       # warm orange (contrast against cool teal theme)

CUSTOM_CSS = f"""
<style>
    .stApp {{
        background: linear-gradient(180deg, #071019 0%, #0B1622 55%, #0B1622 100%);
    }}
    #MainMenu, footer {{visibility: hidden;}}

    .hero {{
        padding: 1.6rem 2rem;
        border-radius: 18px;
        background: linear-gradient(120deg, {PRIMARY_DARK} 0%, {PRIMARY} 55%, #14b8a6 100%);
        box-shadow: 0 10px 30px rgba(14,165,183,0.25);
        margin-bottom: 1.4rem;
    }}
    .hero h1 {{
        color: white;
        font-size: 2.1rem;
        margin-bottom: 0.2rem;
        font-weight: 800;
        letter-spacing: -0.02em;
    }}
    .hero p {{
        color: rgba(255,255,255,0.9);
        font-size: 1.02rem;
        margin: 0;
    }}

    .metric-card {{
        background: rgba(255,255,255,0.04);
        border: 1px solid rgba(255,255,255,0.08);
        border-radius: 14px;
        padding: 1.1rem 1.3rem;
        text-align: center;
    }}
    .metric-card .label {{
        color: #94A3B8;
        font-size: 0.8rem;
        text-transform: uppercase;
        letter-spacing: 0.08em;
    }}
    .metric-card .value {{
        color: #F1F5F9;
        font-size: 1.6rem;
        font-weight: 700;
        margin-top: 0.2rem;
    }}

    .result-card {{
        background: linear-gradient(135deg, rgba(14,165,183,0.16), rgba(20,184,166,0.06));
        border: 1px solid rgba(14,165,183,0.35);
        border-radius: 18px;
        padding: 1.6rem 1.8rem;
        text-align: center;
    }}
    .result-card .ton-label {{
        color: #94A3B8;
        font-size: 0.85rem;
        text-transform: uppercase;
        letter-spacing: 0.1em;
        margin-top: 0.3rem;
    }}
    .recommend-pill {{
        display: inline-block;
        margin-top: 0.9rem;
        padding: 0.45rem 1.2rem;
        border-radius: 999px;
        background: {ACCENT};
        color: #1a0f00;
        font-weight: 700;
        font-size: 0.95rem;
    }}
    section[data-testid="stSidebar"] {{
        background: #0B1420;
        border-right: 1px solid rgba(255,255,255,0.06);
    }}
</style>
"""
st.markdown(CUSTOM_CSS, unsafe_allow_html=True)

FEATURES = [
    "Room_Area", "Room_Height", "Occupancy", "Outdoor_Temperature",
    "Window_Area", "Equipment_Load", "Insulation_Level", "Sun_Exposure",
]
FEATURE_LABELS = {
    "Room_Area": "Room Area (m²)",
    "Room_Height": "Room Height (m)",
    "Occupancy": "Occupancy (people)",
    "Outdoor_Temperature": "Outdoor Temperature (°C)",
    "Window_Area": "Window Area (m²)",
    "Equipment_Load": "Equipment Load (kW)",
    "Insulation_Level": "Insulation Level (1=poor, 5=excellent)",
    "Sun_Exposure": "Sun Exposure (1=Low, 2=Medium, 3=High)",
}


# --------------------------------------------------------------------------
# DATA / MODEL LOADING (cached)
# --------------------------------------------------------------------------
@st.cache_data
def load_data():
    return pd.read_csv("data/AC_Capacity_Dataset_2000.csv")


@st.cache_resource
def load_models():
    lin_model = joblib.load("model/linear_model.pkl")
    rf_model = joblib.load("model/rf_model.pkl")
    with open("model/artifact.json") as f:
        artifact = json.load(f)
    return lin_model, rf_model, artifact


df = load_data()
lin_model, rf_model, artifact = load_models()


def recommend_size(predicted_ton: float) -> float:
    return max(1.0, min(3.5, round(predicted_ton * 2) / 2))


# --------------------------------------------------------------------------
# SVG GAUGE (no plotly dependency needed)
# --------------------------------------------------------------------------
def render_gauge(value, vmin=0.5, vmax=4.0):
    import math
    pct = max(0.0, min(1.0, (value - vmin) / (vmax - vmin)))
    angle = 180 * pct
    r = 90
    cx, cy = 110, 110
    x = cx + r * math.cos(math.radians(180 - angle))
    y = cy - r * math.sin(math.radians(180 - angle))
    svg = f"""
    <svg width="220" height="130" viewBox="0 0 220 130">
      <path d="M 20 110 A 90 90 0 0 1 200 110" fill="none" stroke="#1E293B" stroke-width="16" stroke-linecap="round"/>
      <path d="M 20 110 A 90 90 0 0 1 {x:.1f} {y:.1f}" fill="none" stroke="{PRIMARY}" stroke-width="16" stroke-linecap="round"/>
      <circle cx="{x:.1f}" cy="{y:.1f}" r="7" fill="{ACCENT}"/>
      <text x="110" y="105" text-anchor="middle" font-size="26" font-weight="800" fill="#F1F5F9">{value:.2f}</text>
      <text x="110" y="124" text-anchor="middle" font-size="11" fill="#94A3B8">TON</text>
    </svg>
    """
    return svg

def stepper_slider(label, min_val, max_val, default, step, key):
    """A slider with -/+ buttons on either side that nudge the value by `step`."""
    if key not in st.session_state:
        st.session_state[key] = default

    def _dec():
        st.session_state[key] = max(min_val, round(st.session_state[key] - step, 2))

    def _inc():
        st.session_state[key] = min(max_val, round(st.session_state[key] + step, 2))

    c_minus, c_slider, c_plus = st.columns([1, 6, 1])
    c_minus.button("-", key=f"{key}_dec", on_click=_dec, use_container_width=True)
    c_plus.button("+", key=f"{key}_inc", on_click=_inc, use_container_width=True)
    c_slider.slider(label, min_val, max_val, step=step, key=key)
    return st.session_state[key]

# --------------------------------------------------------------------------
# HERO HEADER
# --------------------------------------------------------------------------
st.markdown(
    """
    <div class="hero">
        <h1>❄️ AC Capacity Predictor</h1>
        <p>Machine-learning powered HVAC sizing — estimate the right air conditioner
        tonnage for any room from its physical & environmental profile.</p>
    </div>
    """,
    unsafe_allow_html=True,
)

tab_predict, tab_perf, tab_data, tab_about = st.tabs(
    ["🔮  Predict", "📊  Model Performance", "🔍  Dataset Explorer", "ℹ️  About"]
)

# ==========================================================================
# TAB 1 — PREDICT
# ==========================================================================
with tab_predict:
    col_input, col_result = st.columns([1.3, 1], gap="large")

    with col_input:
        st.subheader("Room & Environment Parameters")
        c1, c2 = st.columns(2)
        with c1:
            area = stepper_slider("Room Area (m²)", 5.0, 120.0, 30.0, 0.5, key="area")
            height = stepper_slider("Room Height (m)", 2.2, 4.5, 3.0, 0.5, key="height")
            occupancy = stepper_slider("Occupancy (people)", 1, 15, 4, 1, key="occupancy")
            temperature = stepper_slider("Outdoor Temperature (°C)", 15.0, 48.0, 34.0, 0.5, key="temperature")
        with c2:
            window = stepper_slider("Window Area (m²)", 0.5, 20.0, 6.0, 0.5, key="window")
            equipment = stepper_slider("Equipment Load (kW)", 0.0, 4.0, 1.0, 0.5, key="equipment")
            insulation = stepper_slider("Insulation Level (1-5)", 1, 5, 3, 1, key="insulation")
            sun = stepper_slider("Sun Exposure (1=Low, 2=Med, 3=High)", 1, 3, 2, 1, key="sun")

        st.button("⚡ Predict AC Capacity", use_container_width=True, type="primary")

    new_room = pd.DataFrame({
        "Room_Area": [area], "Room_Height": [height], "Occupancy": [occupancy],
        "Outdoor_Temperature": [temperature], "Window_Area": [window],
        "Equipment_Load": [equipment], "Insulation_Level": [insulation], "Sun_Exposure": [sun],
    })
    prediction = float(lin_model.predict(new_room)[0])
    rf_prediction = float(rf_model.predict(new_room)[0])
    recommended = recommend_size(prediction)

    with col_result:
        st.subheader("Prediction")
        st.markdown(f"""
        <div class="result-card">
            {render_gauge(prediction)}
            <div class="ton-label">Predicted AC Capacity</div>
            <div class="recommend-pill">Recommended: {recommended:.1f} Ton unit</div>
        </div>
        """, unsafe_allow_html=True)

        st.caption(f"Random Forest cross-check: **{rf_prediction:.2f} Ton** "
                    f"(Δ {abs(prediction - rf_prediction):.2f} Ton vs. Linear model)")

    st.divider()
    st.markdown("##### How this number was produced")
    contributions = {
        FEATURE_LABELS[f]: artifact["linear_coefficients"][f] * new_room[f].iloc[0]
        for f in FEATURES
    }
    contrib_df = pd.DataFrame(
        {"Contribution (Ton)": contributions}
    ).sort_values("Contribution (Ton)")

    fig, ax = plt.subplots(figsize=(7, 3.2))
    colors = [ACCENT if v < 0 else PRIMARY for v in contrib_df["Contribution (Ton)"]]
    ax.barh(contrib_df.index, contrib_df["Contribution (Ton)"], color=colors)
    ax.axvline(0, color="#475569", linewidth=0.8)
    ax.set_facecolor("none")
    fig.patch.set_alpha(0)
    ax.tick_params(colors="#CBD5E1", labelsize=8)
    for spine in ax.spines.values():
        spine.set_color("#334155")
    ax.set_xlabel("Contribution to predicted tonnage", color="#94A3B8", fontsize=9)
    st.pyplot(fig, use_container_width=True)
    st.caption(
        "Each bar shows how much that feature (× its learned coefficient) adds to or "
        "subtracts from the base predicted load — this is exactly how the Linear "
        "Regression model arrives at its number, feature by feature."
    )

# ==========================================================================
# TAB 2 — MODEL PERFORMANCE
# ==========================================================================
with tab_perf:
    st.subheader("Model Evaluation")

    lm = artifact["linear_metrics"]
    rm = artifact["rf_metrics"]

    m1, m2, m3, m4 = st.columns(4)
    for col, label, val in zip(
        [m1, m2, m3, m4],
        ["MAE (Linear)", "RMSE (Linear)", "R² (Linear)", "5-Fold CV R²"],
        [lm["MAE"], lm["RMSE"], lm["R2"], lm["CV_R2_mean"]],
    ):
        col.markdown(f"""
        <div class="metric-card">
            <div class="label">{label}</div>
            <div class="value">{val:.3f}</div>
        </div>""", unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown(
        f"For reference, a **Random Forest** benchmark achieves "
        f"MAE **{rm['MAE']:.3f}**, R² **{rm['R2']:.3f}** — used here only to sanity-check "
        f"the simpler, more interpretable Linear Regression model actually used for prediction."
    )

    col_scatter, col_resid = st.columns(2)
    y_test = np.array(artifact["y_test"])
    y_pred = np.array(artifact["y_pred_linear"])

    with col_scatter:
        fig1, ax1 = plt.subplots(figsize=(5, 5))
        ax1.scatter(y_test, y_pred, alpha=0.55, color=PRIMARY, edgecolor="none", s=28)
        lims = [min(y_test.min(), y_pred.min()), max(y_test.max(), y_pred.max())]
        ax1.plot(lims, lims, "--", color=ACCENT, linewidth=1.6, label="Perfect Prediction")
        ax1.set_xlabel("Actual AC Capacity (Ton)", color="#CBD5E1")
        ax1.set_ylabel("Predicted AC Capacity (Ton)", color="#CBD5E1")
        ax1.set_title("Actual vs Predicted", color="#F1F5F9")
        ax1.legend(facecolor="#0F172A", labelcolor="#CBD5E1")
        ax1.tick_params(colors="#94A3B8")
        fig1.patch.set_alpha(0)
        ax1.set_facecolor("none")
        for spine in ax1.spines.values():
            spine.set_color("#334155")
        st.pyplot(fig1, use_container_width=True)

    with col_resid:
        residuals = y_test - y_pred
        fig2, ax2 = plt.subplots(figsize=(5, 5))
        ax2.scatter(y_pred, residuals, alpha=0.55, color="#14b8a6", edgecolor="none", s=28)
        ax2.axhline(0, color=ACCENT, linestyle="--", linewidth=1.6)
        ax2.set_xlabel("Predicted AC Capacity (Ton)", color="#CBD5E1")
        ax2.set_ylabel("Residual (Actual − Predicted)", color="#CBD5E1")
        ax2.set_title("Residual Plot", color="#F1F5F9")
        ax2.tick_params(colors="#94A3B8")
        fig2.patch.set_alpha(0)
        ax2.set_facecolor("none")
        for spine in ax2.spines.values():
            spine.set_color("#334155")
        st.pyplot(fig2, use_container_width=True)

    st.markdown("##### Feature Coefficients (Linear Regression)")
    coef_df = pd.Series(artifact["linear_coefficients"]).rename(index=FEATURE_LABELS)
    coef_df = coef_df.sort_values()
    fig3, ax3 = plt.subplots(figsize=(9, 3.4))
    colors3 = [ACCENT if v < 0 else PRIMARY for v in coef_df]
    ax3.barh(coef_df.index, coef_df.values, color=colors3)
    ax3.axvline(0, color="#475569", linewidth=0.8)
    ax3.tick_params(colors="#CBD5E1", labelsize=8)
    fig3.patch.set_alpha(0)
    ax3.set_facecolor("none")
    for spine in ax3.spines.values():
        spine.set_color("#334155")
    st.pyplot(fig3, use_container_width=True)
    st.caption(
        "Positive bars increase predicted tonnage as the feature increases; negative "
        "bars (e.g. Insulation Level) reduce it — better insulation lowers cooling load."
    )

# ==========================================================================
# TAB 3 — DATASET EXPLORER
# ==========================================================================
with tab_data:
    st.subheader("Dataset Overview")
    c1, c2, c3 = st.columns(3)
    c1.markdown(f'<div class="metric-card"><div class="label">Rows</div><div class="value">{len(df)}</div></div>', unsafe_allow_html=True)
    c2.markdown(f'<div class="metric-card"><div class="label">Features</div><div class="value">{len(FEATURES)}</div></div>', unsafe_allow_html=True)
    c3.markdown(f'<div class="metric-card"><div class="label">Missing Values</div><div class="value">{int(df.isnull().sum().sum())}</div></div>', unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)
    st.dataframe(df.head(20), use_container_width=True, height=280)

    st.markdown("##### Correlation Heatmap")
    fig4, ax4 = plt.subplots(figsize=(8, 6))
    corr = df[FEATURES + ["AC_Capacity_Ton"]].corr()
    sns.heatmap(
        corr, annot=True, fmt=".2f", cmap="mako", ax=ax4,
        cbar_kws={"label": "Correlation"}, linewidths=0.4, linecolor="#0B1622",
    )
    fig4.patch.set_alpha(0)
    ax4.tick_params(colors="#CBD5E1")
    st.pyplot(fig4, use_container_width=True)

    st.markdown("##### Target Distribution")
    fig5, ax5 = plt.subplots(figsize=(8, 3))
    ax5.hist(df["AC_Capacity_Ton"], bins=25, color=PRIMARY, edgecolor="#0B1622")
    ax5.set_xlabel("AC Capacity (Ton)", color="#CBD5E1")
    ax5.set_ylabel("Count", color="#CBD5E1")
    fig5.patch.set_alpha(0)
    ax5.set_facecolor("none")
    ax5.tick_params(colors="#94A3B8")
    for spine in ax5.spines.values():
        spine.set_color("#334155")
    st.pyplot(fig5, use_container_width=True)

# ==========================================================================
# TAB 4 — ABOUT
# ==========================================================================
with tab_about:
    st.subheader("About This Project")
    st.markdown("""
This application predicts the **required Air Conditioner capacity (in Tons of
Refrigeration)** for a room, using a Linear Regression model trained on 2,000
labeled samples describing each room's physical and environmental profile.

**Pipeline**
1. **Data** — `Room_Area, Room_Height, Occupancy, Outdoor_Temperature, Window_Area,
   Equipment_Load, Insulation_Level, Sun_Exposure → AC_Capacity_Ton`
2. **Split** — 80/20 train/test split, `random_state=42` for reproducibility.
3. **Model** — `sklearn.linear_model.LinearRegression`, cross-validated with 5-fold CV.
4. **Benchmark** — a Random Forest Regressor trained alongside it purely to confirm
   the linear model isn't missing strong non-linear structure in the data.
5. **Recommendation logic** — the raw prediction is rounded to the nearest
   0.5-ton increment (how AC units are actually sold) and clamped to the
   commercially available range **[1.0, 3.5] Ton**.

**Why Linear Regression and not a black-box model?**
HVAC sizing is a decision that needs to be explainable — a reviewer or
technician should be able to see *why* a room needs a 2.0-ton unit and not a
1.5-ton one. Linear Regression's coefficients make that reasoning fully
transparent (see the *Model Performance* tab), while the Random Forest
benchmark confirms we aren't sacrificing meaningful accuracy for that
transparency.

**Tech stack:** Python · pandas · scikit-learn · Streamlit · Matplotlib/Seaborn
""")

    st.info(
        "This tool provides an engineering **estimate**, not a certified HVAC load "
        "calculation (e.g. Manual J). Always have final sizing verified by a "
        "qualified HVAC professional before purchase or installation."
    )
