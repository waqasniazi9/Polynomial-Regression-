import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
from pathlib import Path

from train_model import (
    train,
    response,
    optimize_budget,
    get_feature_breakdown,
    generate_synergy_grid,
    FEATURES,
)

# ---------------------------------------------------------
# Page Configuration & Styling
# ---------------------------------------------------------
st.set_page_config(
    page_title="AdVision AI | Polynomial Regression & Marketing Mix Modeling",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom Theme CSS
CUSTOM_CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@300;400;500;600;700;800&display=swap');

html, body, [class*="css"] {
    font-family: 'Plus Jakarta Sans', -apple-system, BlinkMacSystemFont, sans-serif;
}

/* Main Container spacing */
.block-container {
    padding-top: 1.8rem;
    padding-bottom: 2.5rem;
    padding-left: 2.2rem;
    padding-right: 2.2rem;
    max-width: 1400px;
}

/* Hero Banner */
.hero-banner {
    background: linear-gradient(135deg, rgba(30, 41, 59, 0.85) 0%, rgba(15, 23, 42, 0.95) 100%);
    border: 1px solid rgba(99, 102, 241, 0.25);
    border-radius: 16px;
    padding: 1.5rem 1.8rem;
    margin-bottom: 1.5rem;
    box-shadow: 0 10px 25px -5px rgba(0, 0, 0, 0.3), 0 8px 10px -6px rgba(0, 0, 0, 0.3);
}

.hero-title {
    font-size: 2.0rem;
    font-weight: 800;
    background: linear-gradient(90deg, #FFFFFF 0%, #E2E8F0 50%, #818CF8 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    margin: 0;
    line-height: 1.2;
}

.hero-subtitle {
    color: #94A3B8;
    font-size: 0.95rem;
    margin-top: 0.4rem;
    margin-bottom: 0.9rem;
}

.badge-row {
    display: flex;
    flex-wrap: wrap;
    gap: 0.5rem;
}

.badge {
    display: inline-flex;
    align-items: center;
    padding: 0.25rem 0.75rem;
    border-radius: 9999px;
    font-size: 0.75rem;
    font-weight: 600;
    letter-spacing: 0.025em;
}

.badge-primary {
    background: rgba(99, 102, 241, 0.15);
    color: #A5B4FC;
    border: 1px solid rgba(99, 102, 241, 0.35);
}

.badge-success {
    background: rgba(16, 185, 129, 0.15);
    color: #6EE7B7;
    border: 1px solid rgba(16, 185, 129, 0.35);
}

.badge-amber {
    background: rgba(245, 158, 11, 0.15);
    color: #FCD34D;
    border: 1px solid rgba(245, 158, 11, 0.35);
}

/* Metric Cards */
.kpi-card {
    background: linear-gradient(145deg, rgba(30, 41, 59, 0.7) 0%, rgba(15, 23, 42, 0.8) 100%);
    border: 1px solid rgba(255, 255, 255, 0.08);
    border-radius: 14px;
    padding: 1.15rem 1.25rem;
    transition: transform 0.2s ease, border-color 0.2s ease, box-shadow 0.2s ease;
    box-shadow: 0 4px 12px rgba(0, 0, 0, 0.15);
    position: relative;
    overflow: hidden;
}

.kpi-card:hover {
    transform: translateY(-2px);
    border-color: rgba(99, 102, 241, 0.4);
    box-shadow: 0 8px 20px rgba(99, 102, 241, 0.15);
}

.kpi-card-glow-indigo::before {
    content: "";
    position: absolute;
    top: 0; left: 0; right: 0;
    height: 3px;
    background: linear-gradient(90deg, #6366F1, #818CF8);
}

.kpi-card-glow-emerald::before {
    content: "";
    position: absolute;
    top: 0; left: 0; right: 0;
    height: 3px;
    background: linear-gradient(90deg, #10B981, #34D399);
}

.kpi-card-glow-amber::before {
    content: "";
    position: absolute;
    top: 0; left: 0; right: 0;
    height: 3px;
    background: linear-gradient(90deg, #F59E0B, #FBBF24);
}

.kpi-card-glow-cyan::before {
    content: "";
    position: absolute;
    top: 0; left: 0; right: 0;
    height: 3px;
    background: linear-gradient(90deg, #06B6D4, #22D3EE);
}

.kpi-label {
    font-size: 0.8rem;
    font-weight: 600;
    text-transform: uppercase;
    letter-spacing: 0.05em;
    color: #94A3B8;
    margin-bottom: 0.35rem;
}

.kpi-value {
    font-size: 1.85rem;
    font-weight: 800;
    color: #F8FAFC;
    line-height: 1.1;
}

.kpi-subtext {
    font-size: 0.78rem;
    color: #CBD5E1;
    margin-top: 0.35rem;
    display: flex;
    align-items: center;
    gap: 0.3rem;
}

.kpi-delta-positive {
    color: #34D399;
    font-weight: 700;
}

.kpi-delta-neutral {
    color: #94A3B8;
}

.status-box {
    border-radius: 10px;
    padding: 0.85rem 1.1rem;
    margin-bottom: 1.2rem;
    font-size: 0.88rem;
    line-height: 1.45;
}

.status-info {
    background: rgba(30, 58, 138, 0.25);
    border: 1px solid rgba(59, 130, 246, 0.35);
    color: #BFDBFE;
}

.status-warning {
    background: rgba(146, 64, 14, 0.25);
    border: 1px solid rgba(245, 158, 11, 0.35);
    color: #FDE68A;
}

.status-success {
    background: rgba(6, 78, 59, 0.25);
    border: 1px solid rgba(16, 185, 129, 0.35);
    color: #A7F3D0;
}

/* Sidebar Styling */
[data-testid="stSidebar"] {
    background-color: #0F172A;
    border-right: 1px solid rgba(255, 255, 255, 0.06);
}

/* Tab Active Highlights */
button[data-baseweb="tab"] {
    font-weight: 600;
    font-size: 0.92rem;
    padding-top: 0.5rem;
    padding-bottom: 0.5rem;
}

/* Clean Dataframes */
[data-testid="stDataFrame"] {
    border: 1px solid rgba(255, 255, 255, 0.08);
    border-radius: 10px;
}
</style>
"""
st.markdown(CUSTOM_CSS, unsafe_allow_html=True)

# ---------------------------------------------------------
# Load Resources & Model
# ---------------------------------------------------------
@st.cache_resource
def load_and_cache_model():
    return train()

model, linear, report, cv_results, comparison, data, training = load_and_cache_model()

# Min / Max / Medians for input features
limits = {
    ch: {
        "min": float(training[ch].min()),
        "max": float(training[ch].max()),
        "median": float(training[ch].median()),
        "mean": float(training[ch].mean()),
    }
    for ch in FEATURES
}

# ---------------------------------------------------------
# Sidebar Controls & Budget Presets
# ---------------------------------------------------------
with st.sidebar:
    st.markdown("### 🎛️ Advertising Budget Controls")
    st.caption("Adjust spending for each marketing channel to simulate predicted product sales.")

    # Strategy Presets
    preset_choice = st.selectbox(
        "⚡ Quick Budget Scenarios",
        options=[
            "Custom Allocation",
            "📊 Historical Median Plan",
            "🚀 High Growth Campaign",
            "📺 TV-Heavy Brand Focus",
            "📻 Radio-Heavy Pulse",
            "⚖️ Balanced Channel Split",
        ],
        index=0,
        help="Quickly load pre-defined marketing budget allocations.",
    )

    # Preset values dictionary
    default_vals = {
        "Custom Allocation": {
            "TV": limits["TV"]["median"],
            "Radio": limits["Radio"]["median"],
            "Newspaper": limits["Newspaper"]["median"],
        },
        "📊 Historical Median Plan": {
            "TV": limits["TV"]["median"],
            "Radio": limits["Radio"]["median"],
            "Newspaper": limits["Newspaper"]["median"],
        },
        "🚀 High Growth Campaign": {
            "TV": 220.0,
            "Radio": 42.0,
            "Newspaper": 25.0,
        },
        "📺 TV-Heavy Brand Focus": {
            "TV": 250.0,
            "Radio": 15.0,
            "Newspaper": 10.0,
        },
        "📻 Radio-Heavy Pulse": {
            "TV": 80.0,
            "Radio": 45.0,
            "Newspaper": 15.0,
        },
        "⚖️ Balanced Channel Split": {
            "TV": 60.0,
            "Radio": 30.0,
            "Newspaper": 30.0,
        },
    }

    active_defaults = default_vals[preset_choice]

    st.markdown("---")
    st.markdown("#### 📡 Channel Spending (Dataset Units)")

    budgets = {}
    for ch in FEATURES:
        budgets[ch] = st.slider(
            f"{ch} Spend",
            min_value=0.0,
            max_value=float(np.ceil(limits[ch]["max"] * 1.25)),
            value=float(active_defaults[ch]),
            step=1.0,
            key=f"slider_{ch}_{preset_choice}",
            help=f"Observed training range: {limits[ch]['min']:.1f} to {limits[ch]['max']:.1f} units.",
        )

    st.markdown("---")
    total_budget = sum(budgets.values())

    # Mini Budget Breakdown Chart in Sidebar
    if total_budget > 0:
        alloc_data = pd.DataFrame(
            {
                "Channel": FEATURES,
                "Budget": [budgets[ch] for ch in FEATURES],
                "Share": [f"{(budgets[ch]/total_budget)*100:.1f}%" for ch in FEATURES],
            }
        )
        fig_mini = px.pie(
            alloc_data,
            values="Budget",
            names="Channel",
            hole=0.55,
            color="Channel",
            color_discrete_map={"TV": "#6366F1", "Radio": "#10B981", "Newspaper": "#F59E0B"},
        )
        fig_mini.update_layout(
            margin=dict(l=5, r=5, t=10, b=10),
            showlegend=True,
            legend=dict(orientation="h", yanchor="bottom", y=-0.3, xanchor="center", x=0.5),
            paper_bgcolor="rgba(0,0,0,0)",
            height=180,
            font=dict(color="#94A3B8", size=10),
        )
        st.plotly_chart(fig_mini, use_container_width=True)

    # Range Guardian in Sidebar
    extrapolations = [ch for ch in FEATURES if not (limits[ch]["min"] <= budgets[ch] <= limits[ch]["max"])]
    if extrapolations:
        st.markdown(
            f"""
            <div class="status-box status-warning">
                ⚠️ <b>Extrapolation Alert</b><br>
                Spend on <b>{', '.join(extrapolations)}</b> exceeds observed historical range ({', '.join([f"{ch}: max {limits[ch]['max']:.1f}" for ch in extrapolations])}). Polynomial curves outside this boundary may exhibit extreme curvature.
            </div>
            """,
            unsafe_allow_html=True,
        )
    else:
        st.markdown(
            """
            <div class="status-box status-success">
                ✅ <b>In-Sample Interpolation</b><br>
                All channel budgets fall within observed dataset bounds. Polynomial predictions are statistically well-supported.
            </div>
            """,
            unsafe_allow_html=True,
        )

# ---------------------------------------------------------
# Core Model Prediction Calculations
# ---------------------------------------------------------
row = pd.DataFrame([budgets])[FEATURES]
poly_pred = float(model.predict(row)[0])
lin_pred = float(linear.predict(row)[0])
poly_vs_linear_delta = poly_pred - lin_pred
poly_vs_linear_pct = (poly_vs_linear_delta / lin_pred * 100) if lin_pred != 0 else 0.0

# TV Marginal calculation at current budget point
tv_marginal_df = response(model, training, budgets, "TV")
# Interpolate current TV marginal return
curr_tv_marginal = float(np.interp(
    budgets["TV"], tv_marginal_df["Budget"], tv_marginal_df["Marginal sales"]
))

# ---------------------------------------------------------
# Header & Hero Banner
# ---------------------------------------------------------
st.markdown(
    """
    <div class="hero-banner">
        <div class="hero-title">📈 AdVision AI — Marketing Mix Modeling</div>
        <div class="hero-subtitle">
            Predict product sales from TV, Radio, and Newspaper advertising budgets using Polynomial Regression.
            Detect diminishing returns, capture cross-channel synergy, and optimize budget allocation for maximum ROI.
        </div>
        <div class="badge-row">
            <span class="badge badge-primary">🎯 Capstone-03 Project</span>
            <span class="badge badge-success">✨ Polynomial Degree 2 (R² = 95.33%)</span>
            <span class="badge badge-primary">📉 Linear Baseline (R² = 90.59%)</span>
            <span class="badge badge-amber">📊 Kaggle Advertising Dataset (200 Records)</span>
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)

# ---------------------------------------------------------
# Executive KPI Row
# ---------------------------------------------------------
k1, k2, k3, k4 = st.columns(4)

with k1:
    st.markdown(
        f"""
        <div class="kpi-card kpi-card-glow-indigo">
            <div class="kpi-label">Predicted Sales (Poly Degree 2)</div>
            <div class="kpi-value">{poly_pred:.2f} <span style="font-size: 1rem; color: #94A3B8;">units</span></div>
            <div class="kpi-subtext">
                <span class="{'kpi-delta-positive' if poly_vs_linear_delta >= 0 else 'kpi-delta-neutral'}">
                    {'+' if poly_vs_linear_delta >= 0 else ''}{poly_vs_linear_delta:.2f} ({poly_vs_linear_pct:+.1f}%)
                </span>
                vs linear baseline
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

with k2:
    st.markdown(
        f"""
        <div class="kpi-card kpi-card-glow-cyan">
            <div class="kpi-label">Linear Baseline Sales</div>
            <div class="kpi-value">{lin_pred:.2f} <span style="font-size: 1rem; color: #94A3B8;">units</span></div>
            <div class="kpi-subtext">
                Degree 1 Linear Model (MAE: {report['linear']['MAE']:.2f})
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

with k3:
    eff_ratio = (poly_pred / total_budget * 100) if total_budget > 0 else 0.0
    st.markdown(
        f"""
        <div class="kpi-card kpi-card-glow-emerald">
            <div class="kpi-label">Total Advertising Spend</div>
            <div class="kpi-value">{total_budget:.1f} <span style="font-size: 1rem; color: #94A3B8;">budget</span></div>
            <div class="kpi-subtext">
                Efficiency: <b>{eff_ratio:.2f} sales</b> per 100 spend units
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

with k4:
    # Diminishing return status
    if curr_tv_marginal < 0.02:
        tv_status = "⚠️ Severe Saturation"
        tv_status_desc = "TV marginal return is near zero; reallocate budget."
        glow_class = "kpi-card-glow-amber"
    elif curr_tv_marginal < 0.045:
        tv_status = "📉 Diminishing Zone"
        tv_status_desc = "Higher TV spend yields reduced incremental gain."
        glow_class = "kpi-card-glow-amber"
    else:
        tv_status = "🚀 High Growth Zone"
        tv_status_desc = "TV spend is in strong marginal return region."
        glow_class = "kpi-card-glow-emerald"

    st.markdown(
        f"""
        <div class="kpi-card {glow_class}">
            <div class="kpi-label">TV Marginal Efficiency</div>
            <div class="kpi-value" style="font-size: 1.45rem;">{tv_status}</div>
            <div class="kpi-subtext">
                +{curr_tv_marginal:.4f} sales / unit &bull; {tv_status_desc}
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

st.write("")

# Alert if negative prediction occurs
if poly_pred < 0:
    st.error("⚠️ The fitted polynomial predicts negative sales for these inputs; inputs are deeply outside physical reality.")

# ---------------------------------------------------------
# Tab Navigation
# ---------------------------------------------------------
tabs = st.tabs(
    [
        "🎯 Instant Prediction & Scenarios",
        "📉 Diminishing Returns & Saturation",
        "⚡ AI Budget Allocator & Optimizer",
        "🔬 Model Evaluation & Polynomial Diagnostics",
        "📁 Dataset & Market Analysis",
    ]
)

# =========================================================
# TAB 1: INSTANT PREDICTION & SCENARIOS
# =========================================================
with tabs[0]:
    st.subheader("📊 Live Prediction Summary & Sensitivity Scenarios")
    st.markdown(
        "Instant sales prediction based on your configured marketing budget. Compare with a linear regression baseline and test automated 'What-If' sensitivity simulations."
    )

    col_pred_left, col_pred_right = st.columns([1.1, 0.9])

    with col_pred_left:
        st.markdown("#### Current Budget Allocation & Expected Output")
        summary_df = pd.DataFrame(
            [
                {
                    "TV Budget": f"{budgets['TV']:.1f}",
                    "Radio Budget": f"{budgets['Radio']:.1f}",
                    "Newspaper Budget": f"{budgets['Newspaper']:.1f}",
                    "Total Budget": f"{total_budget:.1f}",
                    "Polynomial Sales": f"{poly_pred:.2f}",
                    "Linear Sales": f"{lin_pred:.2f}",
                    "Non-Linear Lift": f"{poly_vs_linear_delta:+.2f} ({poly_vs_linear_pct:+.1f}%)",
                }
            ]
        )
        st.dataframe(summary_df, hide_index=True, use_container_width=True)

        # Download CSV Button
        export_df = row.copy()
        export_df["Total_Budget"] = total_budget
        export_df["Polynomial_Predicted_Sales"] = poly_pred
        export_df["Linear_Baseline_Sales"] = lin_pred
        export_df["Non_Linear_Advantage"] = poly_vs_linear_delta

        st.download_button(
            label="📥 Download Current Prediction Plan (CSV)",
            data=export_df.to_csv(index=False),
            file_name="advision_sales_prediction.csv",
            mime="text/csv",
            help="Export input values and predictions for marketing team review.",
        )

        st.markdown("#### 💡 Strategic Marketing Guidance")
        if budgets["Newspaper"] > 40.0:
            st.warning(
                "📌 **Overspending Warning on Newspaper**: In the quadratic fit, Newspaper has an insignificant coefficient (-0.038) and minimal slope. Spending over 40 units on Newspaper yields virtually zero incremental return. Consider reallocating towards TV or Radio."
            )
        elif budgets["TV"] > 220.0:
            st.info(
                "📌 **TV Diminishing Returns Warning**: TV spending is above 220 units where the negative quadratic term ($TV^2 = -2.69$) creates strong curvature. Incremental sales per dollar spent on TV are flattening."
            )
        else:
            st.success(
                "📌 **Balanced Spend Profile**: Your budget is well positioned within the active growth range of TV and Radio synergy."
            )

    with col_pred_right:
        # Donut Chart with Spend vs Predicted Contribution
        st.markdown("#### Budget Distribution Breakdown")
        fig_donut = px.pie(
            values=[budgets[ch] for ch in FEATURES],
            names=FEATURES,
            color=FEATURES,
            color_discrete_map={"TV": "#6366F1", "Radio": "#10B981", "Newspaper": "#F59E0B"},
            hole=0.6,
        )
        fig_donut.update_layout(
            template="plotly_dark",
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            height=260,
            margin=dict(l=10, r=10, t=10, b=10),
            legend=dict(orientation="h", yanchor="bottom", y=-0.15, xanchor="center", x=0.5),
        )
        st.plotly_chart(fig_donut, use_container_width=True)

    st.markdown("---")
    st.subheader("🧪 What-If Sensitivity Lab")
    st.markdown(
        "Explore how small budget adjustments in specific channels immediately impact overall product sales."
    )

    # Scenarios Table
    scenarios = []

    # Scenario 1: +10% per channel
    for ch in FEATURES:
        up = dict(budgets)
        up[ch] *= 1.10
        extra_b = up[ch] - budgets[ch]
        p_up = float(model.predict(pd.DataFrame([up])[FEATURES])[0])
        scenarios.append(
            {
                "Scenario": f"Boost {ch} by +10%",
                "Channel Affected": ch,
                "Budget Change": f"+{extra_b:.1f}",
                "New Total Budget": f"{sum(up.values()):.1f}",
                "Predicted Sales": f"{p_up:.2f}",
                "Sales Impact": f"{p_up - poly_pred:+.2f}",
                "Marginal ROI (Sales/Budget)": f"{(p_up - poly_pred)/extra_b:.3f}" if extra_b > 0 else "N/A",
            }
        )

    # Scenario 2: Shift 15 units from Newspaper to Radio
    if budgets["Newspaper"] >= 15.0:
        shift_np_to_radio = dict(budgets)
        shift_np_to_radio["Newspaper"] = max(0.0, shift_np_to_radio["Newspaper"] - 15.0)
        shift_np_to_radio["Radio"] += 15.0
        p_shift_r = float(model.predict(pd.DataFrame([shift_np_to_radio])[FEATURES])[0])
        scenarios.append(
            {
                "Scenario": "🔄 Reallocate 15 units: Newspaper ➔ Radio",
                "Channel Affected": "Newspaper to Radio",
                "Budget Change": "0.0 (Zero Extra Cost)",
                "New Total Budget": f"{sum(shift_np_to_radio.values()):.1f}",
                "Predicted Sales": f"{p_shift_r:.2f}",
                "Sales Impact": f"{p_shift_r - poly_pred:+.2f}",
                "Marginal ROI (Sales/Budget)": "Pure Synergy Gain",
            }
        )

    # Scenario 3: Shift 15 units from Newspaper to TV
    if budgets["Newspaper"] >= 15.0:
        shift_np_to_tv = dict(budgets)
        shift_np_to_tv["Newspaper"] = max(0.0, shift_np_to_tv["Newspaper"] - 15.0)
        shift_np_to_tv["TV"] += 15.0
        p_shift_tv = float(model.predict(pd.DataFrame([shift_np_to_tv])[FEATURES])[0])
        scenarios.append(
            {
                "Scenario": "🔄 Reallocate 15 units: Newspaper ➔ TV",
                "Channel Affected": "Newspaper to TV",
                "Budget Change": "0.0 (Zero Extra Cost)",
                "New Total Budget": f"{sum(shift_np_to_tv.values()):.1f}",
                "Predicted Sales": f"{p_shift_tv:.2f}",
                "Sales Impact": f"{p_shift_tv - poly_pred:+.2f}",
                "Marginal ROI (Sales/Budget)": "Pure Synergy Gain",
            }
        )

    st.dataframe(pd.DataFrame(scenarios), hide_index=True, use_container_width=True)

# =========================================================
# TAB 2: DIMINISHING RETURNS & SATURATION ANALYTICS
# =========================================================
with tabs[1]:
    st.subheader("📉 Diminishing Returns, Saturation & Channel Curvature")
    st.markdown(
        """
        In real-world advertising, the first dollar spent produces higher sales returns than the 200th dollar.
        Polynomial regression captures this phenomenon via non-linear quadratic terms ($TV^2 < 0$) and cross-channel synergy ($TV \\times Radio > 0$).
        """
    )

    selected_channel = st.selectbox(
        "Select Channel to Analyze for Diminishing Returns:",
        options=FEATURES,
        index=0,
        help="Analyzes the selected channel while holding other channels fixed at sidebar budgets.",
    )

    curve_data = response(model, training, budgets, selected_channel)

    col_chart1, col_chart2 = st.columns(2)

    with col_chart1:
        st.markdown(f"#### 📈 Total Sales Response Curve ({selected_channel})")
        fig_response = go.Figure()

        # Response Line
        fig_response.add_trace(
            go.Scatter(
                x=curve_data["Budget"],
                y=curve_data["Predicted sales"],
                mode="lines",
                name="Fitted Polynomial Curve",
                line=dict(color="#6366F1", width=3.5),
            )
        )

        # Current User Position Marker
        current_ch_budget = budgets[selected_channel]
        current_ch_sales = float(np.interp(
            current_ch_budget, curve_data["Budget"], curve_data["Predicted sales"]
        ))

        fig_response.add_trace(
            go.Scatter(
                x=[current_ch_budget],
                y=[current_ch_sales],
                mode="markers+text",
                name="Your Current Budget",
                marker=dict(color="#10B981", size=14, symbol="diamond", line=dict(color="#FFFFFF", width=2)),
                text=[f"Current: {current_ch_budget:.1f}"],
                textposition="top left",
            )
        )

        fig_response.update_layout(
            template="plotly_dark",
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(15, 23, 42, 0.4)",
            xaxis_title=f"{selected_channel} Spend (Dataset Units)",
            yaxis_title="Predicted Sales (Units)",
            height=370,
            margin=dict(l=20, r=20, t=30, b=20),
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        )
        st.plotly_chart(fig_response, use_container_width=True)

    with col_chart2:
        st.markdown("#### ⚡ Marginal Sales Curve (First Derivative: $\\frac{\\Delta \\text{Sales}}{\\Delta \\text{Spend}}$)")
        fig_marginal = go.Figure()

        fig_marginal.add_trace(
            go.Scatter(
                x=curve_data["Budget"],
                y=curve_data["Marginal sales"],
                mode="lines",
                name="Marginal Return Rate",
                line=dict(color="#F59E0B", width=3, dash="solid"),
                fill="tozeroy",
                fillcolor="rgba(245, 158, 11, 0.1)",
            )
        )

        # Highlight zero line
        fig_marginal.add_hline(y=0, line_dash="dash", line_color="#EF4444", annotation_text="Zero Return Threshold")

        # Current marginal marker
        current_ch_marginal = float(np.interp(
            current_ch_budget, curve_data["Budget"], curve_data["Marginal sales"]
        ))
        fig_marginal.add_trace(
            go.Scatter(
                x=[current_ch_budget],
                y=[current_ch_marginal],
                mode="markers+text",
                name="Your Current Marginal Return",
                marker=dict(color="#10B981", size=14, symbol="circle", line=dict(color="#FFFFFF", width=2)),
                text=[f"Slope: +{current_ch_marginal:.4f}"],
                textposition="bottom left" if current_ch_marginal > 0.04 else "top right",
            )
        )

        fig_marginal.update_layout(
            template="plotly_dark",
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(15, 23, 42, 0.4)",
            xaxis_title=f"{selected_channel} Spend (Dataset Units)",
            yaxis_title="Marginal Sales per Extra Unit of Budget",
            height=370,
            margin=dict(l=20, r=20, t=30, b=20),
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        )
        st.plotly_chart(fig_marginal, use_container_width=True)

    # Curvature Analysis Box
    curvature_median = float(curve_data["Curvature"].median())
    if curvature_median < -1e-6:
        st.markdown(
            f"""
            <div class="status-box status-warning">
                📉 <b>Diminishing Marginal Returns Confirmed for {selected_channel}</b>:
                The curvature (second derivative) is negative (<b>{curvature_median:.2e}</b>). As spending rises, each additional budget unit generates fewer incremental sales than the previous unit. Spending heavily past the plateau risks overspending.
            </div>
            """,
            unsafe_allow_html=True,
        )
    elif curvature_median > 1e-6:
        st.markdown(
            f"""
            <div class="status-box status-info">
                📈 <b>Super-Linear Response for {selected_channel}</b>:
                The curvature is slightly positive (<b>+{curvature_median:.2e}</b>). This indicates that increased spending still yields strong or accelerating returns, especially when amplified by cross-channel synergy.
            </div>
            """,
            unsafe_allow_html=True,
        )
    else:
        st.markdown(
            f"""
            <div class="status-box status-info">
                ⚖️ <b>Near-Linear Response for {selected_channel}</b>:
                The marginal return is approximately constant across observed spending ranges.
            </div>
            """,
            unsafe_allow_html=True,
        )

    # 3D Synergy Interaction Surface
    st.markdown("---")
    st.subheader("🌐 Cross-Channel Synergy: TV × Radio 3D Response Surface")
    st.markdown(
        "Polynomial regression captures multi-channel synergy! Notice how high TV combined with high Radio creates super-additive sales lift ($TV \\times Radio = +1.39$ coefficient)."
    )

    tv_grid, radio_grid, z_sales = generate_synergy_grid(
        model, training, newspaper_val=budgets["Newspaper"], resolution=25
    )

    fig_3d = go.Figure(
        data=[
            go.Surface(
                z=z_sales,
                x=tv_grid,
                y=radio_grid,
                colorscale="Viridis",
                colorbar=dict(title="Predicted Sales"),
            )
        ]
    )

    # Add marker for current user position on the surface
    fig_3d.add_trace(
        go.Scatter3d(
            x=[budgets["TV"]],
            y=[budgets["Radio"]],
            z=[poly_pred],
            mode="markers+text",
            marker=dict(size=8, color="#EF4444", symbol="diamond"),
            text=["Your Spend"],
            textposition="top center",
            name="Current Budget Position",
        )
    )

    fig_3d.update_layout(
        template="plotly_dark",
        scene=dict(
            xaxis_title="TV Budget",
            yaxis_title="Radio Budget",
            zaxis_title="Predicted Sales",
            xaxis=dict(backgroundcolor="#0F172A"),
            yaxis=dict(backgroundcolor="#0F172A"),
            zaxis=dict(backgroundcolor="#0F172A"),
        ),
        paper_bgcolor="rgba(0,0,0,0)",
        height=480,
        margin=dict(l=10, r=10, t=10, b=10),
    )
    st.plotly_chart(fig_3d, use_container_width=True)

# =========================================================
# TAB 3: AI BUDGET ALLOCATOR & OPTIMIZER
# =========================================================
with tabs[2]:
    st.subheader("⚡ AI Marketing Budget Allocator & ROI Optimizer")
    st.markdown(
        """
        **Goal**: Given a fixed total advertising budget, how should you distribute it across TV, Radio, and Newspaper to **maximize total predicted sales** while eliminating waste and avoiding diminishing returns?
        
        Our mathematical optimizer employs **Sequential Least Squares Programming (SLSQP)** to find the global optimum within empirical training bounds.
        """
    )

    col_opt_input, col_opt_settings = st.columns([1, 1])

    with col_opt_input:
        target_budget = st.number_input(
            "Target Total Marketing Budget to Optimize:",
            min_value=20.0,
            max_value=400.0,
            value=float(total_budget) if total_budget >= 20.0 else 180.0,
            step=5.0,
            help="Total dollars / units available across all three marketing channels.",
        )

    with col_opt_settings:
        min_channel_pct = st.slider(
            "Minimum Channel Allocation Guarantee (% per channel)",
            min_value=0.0,
            max_value=20.0,
            value=0.0,
            step=1.0,
            help="Enforces that every channel receives at least X% of the budget (e.g. for brand presence).",
        )

    min_per_ch = (min_channel_pct / 100.0) * target_budget

    # Compute Optimization
    opt_allocation, opt_predicted_sales = optimize_budget(
        target_budget, model, training, min_per_channel=min_per_ch
    )

    # Current budget scaled to target_budget for fair baseline comparison
    if total_budget > 0:
        scaled_current_budgets = {ch: (budgets[ch] / total_budget) * target_budget for ch in FEATURES}
    else:
        scaled_current_budgets = {ch: target_budget / 3.0 for ch in FEATURES}

    current_baseline_sales = float(
        model.predict(pd.DataFrame([[scaled_current_budgets[ch] for ch in FEATURES]], columns=FEATURES))[0]
    )

    opt_lift = opt_predicted_sales - current_baseline_sales
    opt_lift_pct = (opt_lift / current_baseline_sales * 100) if current_baseline_sales > 0 else 0.0

    st.markdown("---")

    # Optimization Result KPI Cards
    r1, r2, r3 = st.columns(3)
    r1.metric("Optimized Predicted Sales", f"{opt_predicted_sales:.2f} units", delta=f"{opt_lift:+.2f} ({opt_lift_pct:+.1f}%)")
    r2.metric("Total Budget Allocated", f"{target_budget:.1f} units", delta="Exact Spend Match")
    r3.metric(
        "Optimal Efficiency (ROI)",
        f"{(opt_predicted_sales / target_budget * 100):.2f}",
        delta=f"{(opt_lift / target_budget * 100):+.2f} efficiency lift",
    )

    st.markdown("#### ⚖️ Current Allocation vs. AI-Optimized Allocation")

    comp_chart_df = pd.DataFrame(
        {
            "Channel": FEATURES * 2,
            "Allocation (Units)": [scaled_current_budgets[ch] for ch in FEATURES]
            + [opt_allocation[ch] for ch in FEATURES],
            "Strategy": ["Your Allocation (Scaled)"] * 3 + ["AI-Optimized Allocation"] * 3,
        }
    )

    fig_comp = px.bar(
        comp_chart_df,
        x="Channel",
        y="Allocation (Units)",
        color="Strategy",
        barmode="group",
        color_discrete_map={
            "Your Allocation (Scaled)": "#94A3B8",
            "AI-Optimized Allocation": "#10B981",
        },
        text_auto=".1f",
    )
    fig_comp.update_layout(
        template="plotly_dark",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(15, 23, 42, 0.4)",
        height=320,
        margin=dict(l=20, r=20, t=20, b=20),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    )
    st.plotly_chart(fig_comp, use_container_width=True)

    # Detailed Table & Action Plan
    alloc_diff_df = pd.DataFrame(
        [
            {
                "Channel": ch,
                "Current Spend": f"{scaled_current_budgets[ch]:.1f}",
                "Optimized Spend": f"{opt_allocation[ch]:.1f}",
                "Recommended Shift": f"{opt_allocation[ch] - scaled_current_budgets[ch]:+.1f}",
                "Optimized Channel Share": f"{(opt_allocation[ch] / target_budget * 100):.1f}%",
            }
            for ch in FEATURES
        ]
    )
    st.dataframe(alloc_diff_df, hide_index=True, use_container_width=True)

    st.markdown("#### 🚀 Actionable Marketing Reallocation Recommendations:")
    realloc_points = []
    for ch in FEATURES:
        diff = opt_allocation[ch] - scaled_current_budgets[ch]
        if diff > 5.0:
            realloc_points.append(
                f"**Increase {ch} by +{diff:.1f} units**: This channel has strong marginal efficiency and pairs synergistically with other channels."
            )
        elif diff < -5.0:
            realloc_points.append(
                f"**Reduce {ch} by {abs(diff):.1f} units**: Spending here is past the diminishing return threshold or yielding negligible ROI."
            )
        else:
            realloc_points.append(f"**Maintain {ch} spend near current level** (±{abs(diff):.1f} units).")

    for pt in realloc_points:
        st.markdown(f"- {pt}")

# =========================================================
# TAB 4: MODEL EVALUATION & DIAGNOSTICS
# =========================================================
with tabs[3]:
    st.subheader("🔬 Model Architecture, Cross-Validation & Polynomial Diagnostics")
    st.markdown(
        """
        Why Polynomial Regression? Linear regression assumes a fixed, endless linear return on ad spend.
        Polynomial Regression with Degree 2 captures **diminishing returns** and **channel interactions** while guarding against high-degree overfitting.
        """
    )

    # Metrics Summary Cards
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Selected Degree", f"Degree {report['selected_degree']}", "Selected via 5-Fold CV")
    m2.metric("Polynomial R² (Test Set)", f"{report['polynomial']['R2']:.4f}", f"+{report['polynomial']['R2'] - report['linear']['R2']:.4f} vs Linear")
    m3.metric("Polynomial MAE", f"{report['polynomial']['MAE']:.3f} units", f"{report['polynomial']['MAE'] - report['linear']['MAE']:.3f} error reduction")
    m4.metric("Polynomial RMSE", f"{report['polynomial']['RMSE']:.3f} units", f"{report['polynomial']['RMSE'] - report['linear']['RMSE']:.3f} error reduction")

    st.markdown("---")

    col_diag1, col_diag2 = st.columns(2)

    with col_diag1:
        st.markdown("#### 📊 Model Selection: 5-Fold Cross-Validation RMSE")
        cv_chart_df = cv_results.copy()
        cv_chart_df["Degree_Label"] = [f"Degree {d} ({'Linear' if d==1 else 'Quadratic' if d==2 else 'Cubic'})" for d in cv_chart_df["Degree"]]

        fig_cv = px.bar(
            cv_chart_df,
            x="Degree_Label",
            y="CV_RMSE",
            color="Degree_Label",
            error_y="CV_RMSE_SD",
            text_auto=".3f",
            color_discrete_sequence=["#94A3B8", "#10B981", "#EF4444"],
        )
        fig_cv.update_layout(
            template="plotly_dark",
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(15, 23, 42, 0.4)",
            xaxis_title="Polynomial Complexity",
            yaxis_title="Cross-Validation RMSE (Lower is Better)",
            height=320,
            margin=dict(l=20, r=20, t=20, b=20),
            showlegend=False,
        )
        st.plotly_chart(fig_cv, use_container_width=True)
        st.caption("Degree 2 achieved the lowest cross-validation error without the risk of Runge's phenomenon / overfitting found in Degree 3.")

    with col_diag2:
        st.markdown("#### 🎯 Actual vs. Predicted Sales (Test Set: 40 Campaigns)")
        fig_actual_pred = px.scatter(
            comparison,
            x="Actual",
            y="Polynomial",
            hover_data=["TV", "Radio", "Newspaper", "Residual"],
            title="",
            color="Residual",
            color_continuous_scale="Viridis",
        )
        min_val = min(comparison["Actual"].min(), comparison["Polynomial"].min())
        max_val = max(comparison["Actual"].max(), comparison["Polynomial"].max())

        fig_actual_pred.add_shape(
            type="line",
            x0=min_val,
            y0=min_val,
            x1=max_val,
            y1=max_val,
            line=dict(color="#EF4444", dash="dash", width=2),
        )
        fig_actual_pred.update_layout(
            template="plotly_dark",
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(15, 23, 42, 0.4)",
            xaxis_title="Actual Sales (Dataset Units)",
            yaxis_title="Predicted Sales (Polynomial Model)",
            height=320,
            margin=dict(l=20, r=20, t=20, b=20),
        )
        st.plotly_chart(fig_actual_pred, use_container_width=True)

    # Feature Importance & Equation Decomposition
    st.markdown("---")
    st.markdown("#### 📐 Polynomial Equation Feature Breakdown")
    st.markdown(
        "Here is the mathematical decomposition of how each advertising channel and interaction contributes to predicted sales:"
    )

    feat_df = get_feature_breakdown(model)

    col_feat1, col_feat2 = st.columns([1.1, 0.9])

    with col_feat1:
        fig_feat = px.bar(
            feat_df,
            x="Coefficient",
            y="Feature",
            orientation="h",
            color="Type",
            color_discrete_map={
                "Direct Linear Effect": "#6366F1",
                "Non-Linear Curvature": "#EF4444",
                "Synergy Interaction": "#10B981",
            },
            text_auto=".2f",
        )
        fig_feat.update_layout(
            template="plotly_dark",
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(15, 23, 42, 0.4)",
            yaxis=dict(autorange="reversed"),
            height=340,
            margin=dict(l=20, r=20, t=20, b=20),
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        )
        st.plotly_chart(fig_feat, use_container_width=True)

    with col_feat2:
        st.dataframe(feat_df, hide_index=True, use_container_width=True)
        st.info(
            """
            **Key Insights from Weights:**
            - **TV^2 (-2.69)**: The negative quadratic coefficient mathematically verifies the **diminishing return** phenomenon on TV advertising.
            - **TV × Radio (+1.39)**: Positive synergy confirms that running TV alongside Radio creates higher sales than running either channel alone.
            - **Newspaper (-0.04)**: Minimal coefficient confirms negligible contribution.
            """
        )

# =========================================================
# TAB 5: DATASET & MARKET ANALYSIS
# =========================================================
with tabs[4]:
    st.subheader("📁 Kaggle Advertising Dataset Explorer")
    st.markdown(
        "The model is trained on the classic Kaggle Advertising Dataset (200 real-world campaign records)."
    )

    col_data_stat, col_data_corr = st.columns([1, 1])

    with col_data_stat:
        st.markdown("#### 📋 Dataset Statistical Summary")
        st.dataframe(data.describe().round(2), use_container_width=True)

    with col_data_corr:
        st.markdown("#### 🔗 Channel Correlation Heatmap")
        corr_matrix = data.corr().round(2)
        fig_corr = px.imshow(
            corr_matrix,
            text_auto=True,
            aspect="auto",
            color_continuous_scale="RdBu_r",
            zmin=-1,
            zmax=1,
        )
        fig_corr.update_layout(
            template="plotly_dark",
            paper_bgcolor="rgba(0,0,0,0)",
            height=280,
            margin=dict(l=20, r=20, t=20, b=20),
        )
        st.plotly_chart(fig_corr, use_container_width=True)

    st.markdown("#### 📊 Campaign Data Records")
    st.dataframe(data, use_container_width=True, height=260)

    st.markdown("---")
    st.markdown(
        """
        - **Data Source**: [Kaggle Advertising Dataset](https://www.kaggle.com/datasets/ashydv/advertising-dataset)
        - **Total Sample Size**: 200 records (160 Train, 40 Test)
        - **Features**: TV, Radio, Newspaper (Budget units) & Sales (Units sold)
        """
    )
