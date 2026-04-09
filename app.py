"""
app.py  –  AI Climate Data Analyst  –  Streamlit Dashboard
Run:  streamlit run app.py
"""

import sys
import os
import json
import time
import pandas as pd
import streamlit as st
from pathlib import Path
import plotly

sys.path.insert(0, str(Path(__file__).parent))

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="AI Climate Analyst",
    page_icon="🌍",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Custom CSS ────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@300;400;600;700&family=JetBrains+Mono:wght@400;600&display=swap');

html, body, [class*="css"] { font-family: 'Space Grotesk', sans-serif; }

.main { background: #0F1117; }

.metric-card {
    background: linear-gradient(135deg, #1A1E2E 0%, #222740 100%);
    border: 1px solid rgba(0,201,167,0.2);
    border-radius: 12px;
    padding: 18px 22px;
    margin-bottom: 10px;
    transition: border-color 0.2s;
}
.metric-card:hover { border-color: rgba(0,201,167,0.6); }
.metric-label { color: #888; font-size: 12px; text-transform: uppercase; letter-spacing: 1px; }
.metric-value { color: #00C9A7; font-size: 28px; font-weight: 700; line-height: 1.2; }
.metric-delta { color: #FFC75F; font-size: 13px; }

.insight-card {
    background: #1A1E2E;
    border-left: 3px solid #00C9A7;
    border-radius: 0 8px 8px 0;
    padding: 14px 18px;
    margin-bottom: 8px;
    font-size: 14px;
    color: #C8D0E0;
}
.anomaly-card {
    background: #1A1E2E;
    border-left: 3px solid #FF6B6B;
    border-radius: 0 8px 8px 0;
    padding: 14px 18px;
    margin-bottom: 8px;
    font-size: 14px;
    color: #C8D0E0;
}
.rec-card {
    background: #1A1E2E;
    border-left: 3px solid #845EC2;
    border-radius: 0 8px 8px 0;
    padding: 14px 18px;
    margin-bottom: 8px;
    font-size: 14px;
    color: #C8D0E0;
}
.risk-badge-high   { background: #FF6B6B22; color: #FF6B6B; padding: 3px 10px; border-radius: 20px; font-size: 12px; }
.risk-badge-medium { background: #FFC75F22; color: #FFC75F; padding: 3px 10px; border-radius: 20px; font-size: 12px; }
.risk-badge-low    { background: #00C9A722; color: #00C9A7; padding: 3px 10px; border-radius: 20px; font-size: 12px; }

div[data-testid="stSidebar"] { background: #1A1E2E; border-right: 1px solid #2A2E45; }
h1, h2, h3 { color: #E8ECF4 !important; }
p { color: #C8D0E0; }
</style>
""", unsafe_allow_html=True)

# ── Header ────────────────────────────────────────────────────────────────────
st.markdown("""
<div style="padding: 30px 0 10px 0;">
  <h1 style="font-size:2.4rem; font-weight:700; background: linear-gradient(90deg, #00C9A7, #845EC2);
     -webkit-background-clip:text; -webkit-text-fill-color:transparent; margin:0;">
    🌍 AI Climate Data Analyst
  </h1>
  <p style="color:#888; margin-top:6px;">Upload raw climate CSV/JSON → Auto-clean → AI Insights → Interactive Visualizations</p>
</div>
""", unsafe_allow_html=True)
st.divider()

# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("### ⚙️ Configuration")
    api_key = st.text_input("Anthropic API Key", type="password",
                             placeholder="sk-ant-...",
                             help="Leave blank to use ANTHROPIC_API_KEY env var")
    if api_key:
        os.environ["ANTHROPIC_API_KEY"] = api_key

    st.divider()
    st.markdown("### 📂 Data Source")
    data_source = st.radio("Choose input", ["Use sample data", "Upload file"])

    uploaded = None
    if data_source == "Upload file":
        uploaded = st.file_uploader("Drop CSV or JSON", type=["csv", "json"])

    st.divider()
    st.markdown("### 🎛️ Filters")
    run_insights = st.checkbox("Generate AI Insights", value=True)
    show_raw     = st.checkbox("Show raw data preview", value=False)
    show_clean   = st.checkbox("Show cleaned data", value=False)

    st.divider()
    st.caption("Built with Python · Pandas · LangChain · Plotly · Streamlit")


# ── Load & cache data ─────────────────────────────────────────────────────────
@st.cache_data(show_spinner=False)
def load_and_clean(filepath: str | None, raw_bytes: bytes | None,
                   file_ext: str | None) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    from src.data_cleaner import ClimateDataCleaner
    import tempfile

    if raw_bytes is not None:
        suffix = file_ext or ".csv"
        with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
            tmp.write(raw_bytes)
            tmp_path = tmp.name
        cleaner = ClimateDataCleaner(tmp_path)
    else:
        cleaner = ClimateDataCleaner(filepath or "data/climate_raw.csv")

    df_clean = cleaner.run()
    return cleaner.df_raw, df_clean, cleaner.get_report()


@st.cache_data(show_spinner=False)
def get_insights(_df_clean: pd.DataFrame) -> dict:
    from src.insights_engine import InsightsEngine
    engine = InsightsEngine()
    return engine.generate(_df_clean)


# ── Main flow ─────────────────────────────────────────────────────────────────
if data_source == "Use sample data":
    raw_bytes = None
    ext       = None
    fpath     = "data/climate_raw.csv"
else:
    if uploaded is None:
        st.info("📎 Please upload a CSV or JSON file to continue.", icon="📂")
        st.stop()
    raw_bytes = uploaded.read()
    ext       = Path(uploaded.name).suffix
    fpath     = None

# ── Run cleaning ──────────────────────────────────────────────────────────────
with st.spinner("🧹 Auto-cleaning dataset..."):
    df_raw, df_clean, report = load_and_clean(fpath, raw_bytes, ext)

# ── City / season filters ─────────────────────────────────────────────────────
with st.sidebar:
    if "city" in df_clean.columns:
        cities = ["All"] + sorted(df_clean["city"].dropna().unique().tolist())
        sel_city = st.selectbox("City", cities)
    else:
        sel_city = "All"

    if "season" in df_clean.columns:
        seasons = ["All"] + sorted(df_clean["season"].dropna().unique().tolist())
        sel_season = st.selectbox("Season", seasons)
    else:
        sel_season = "All"

df_filtered = df_clean.copy()
if sel_city   != "All": df_filtered = df_filtered[df_filtered["city"]   == sel_city]
if sel_season != "All": df_filtered = df_filtered[df_filtered["season"] == sel_season]

# ── KPI row ───────────────────────────────────────────────────────────────────
st.markdown("#### 📊 Dataset Overview")
k1, k2, k3, k4, k5 = st.columns(5)

def kpi(col, label, value, delta=""):
    col.markdown(f"""
    <div class="metric-card">
      <div class="metric-label">{label}</div>
      <div class="metric-value">{value}</div>
      {'<div class="metric-delta">' + delta + '</div>' if delta else ''}
    </div>""", unsafe_allow_html=True)

kpi(k1, "Total Records",   f"{len(df_filtered):,}",  f"Raw: {len(df_raw):,}")
kpi(k2, "Avg Temperature", f"{df_filtered['temperature_c'].mean():.1f}°C" if 'temperature_c' in df_filtered else "N/A")
kpi(k3, "Avg AQI",         f"{df_filtered['aqi'].mean():.0f}"             if 'aqi'           in df_filtered else "N/A")
kpi(k4, "Avg CO₂",         f"{df_filtered['co2_ppm'].mean():.1f} ppm"     if 'co2_ppm'       in df_filtered else "N/A")
kpi(k5, "Cities",          f"{df_filtered['city'].nunique()}"              if 'city'          in df_filtered else "N/A")

st.divider()

# ── Cleaning report ───────────────────────────────────────────────────────────
with st.expander("🧹 Cleaning Report", expanded=False):
    cols = st.columns(2)
    cols[0].metric("Rows In",  report.get("rows_in",  0))
    cols[1].metric("Rows Out", report.get("rows_out", 0))
    for step in report.get("steps", []):
        st.markdown(f"- **{step['step']}**: {step['detail']}")

if show_raw:
    with st.expander("📄 Raw Data (first 50 rows)"):
        st.dataframe(df_raw.head(50), use_container_width=True)

if show_clean:
    with st.expander("✅ Cleaned Data (first 50 rows)"):
        st.dataframe(df_filtered.head(50), use_container_width=True)

# ── AI Insights ───────────────────────────────────────────────────────────────
if run_insights:
    st.markdown("#### 🤖 AI-Powered Insights")
    with st.spinner("🧠 Generating insights with LangChain + Claude..."):
        insights = get_insights(df_filtered)

    tab_a, tab_b, tab_c = st.tabs(["💡 Key Findings", "⚠️ Anomalies", "✅ Recommendations"])

    with tab_a:
        st.markdown(f"**{insights.get('executive_summary', '')}**")
        st.markdown("")
        for f in insights.get("key_findings", []):
            st.markdown(f'<div class="insight-card">💡 {f}</div>', unsafe_allow_html=True)

        ri = insights.get("risk_indicators", {})
        st.markdown("**Risk Indicators:**")
        rc1, rc2, rc3 = st.columns(3)
        rc1.markdown(f"🏙️ **High-AQI Cities:** {', '.join(ri.get('high_aqi_cities', []))}")
        rc2.markdown(f"🌿 **CO₂ Trend:** {ri.get('co2_trend', 'N/A').capitalize()}")
        rc3.markdown(f"🌡️ **Temp Concern:** {ri.get('temperature_concern', 'N/A').capitalize()}")

    with tab_b:
        for a in insights.get("anomalies", []):
            st.markdown(f'<div class="anomaly-card">⚠️ {a}</div>', unsafe_allow_html=True)

    with tab_c:
        for r in insights.get("recommendations", []):
            st.markdown(f'<div class="rec-card">✅ {r}</div>', unsafe_allow_html=True)

    st.divider()

# ── Visualisations ────────────────────────────────────────────────────────────
st.markdown("#### 📈 Interactive Visualizations")

from src.visualizer import ClimateVisualizer
viz = ClimateVisualizer(df_filtered)
figs = viz.get_all_figures()

# Row 1
c1, c2 = st.columns(2)
with c1:
    st.plotly_chart(figs["temperature_trend"],  use_container_width=True)
with c2:
    st.plotly_chart(figs["aqi_by_city"],        use_container_width=True)

# Row 2
c3, c4 = st.columns(2)
with c3:
    st.plotly_chart(figs["co2_vs_temp"],        use_container_width=True)
with c4:
    st.plotly_chart(figs["pm25_distribution"],  use_container_width=True)

# Row 3
c5, c6 = st.columns(2)
with c5:
    st.plotly_chart(figs["seasonal_heatmap"],   use_container_width=True)
with c6:
    st.plotly_chart(figs["air_quality_pie"],    use_container_width=True)

# Row 4
c7, c8 = st.columns(2)
with c7:
    st.plotly_chart(figs["wind_rainfall"],      use_container_width=True)
with c8:
    st.plotly_chart(figs["correlation_matrix"], use_container_width=True)

# ── Download cleaned data ─────────────────────────────────────────────────────
st.divider()
st.markdown("#### 💾 Export")
csv_bytes = df_filtered.to_csv(index=False).encode()
st.download_button("⬇️ Download Cleaned CSV", csv_bytes,
                   "climate_clean.csv", "text/csv", use_container_width=True)
