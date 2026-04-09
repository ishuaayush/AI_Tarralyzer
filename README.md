![CI workflow status badge for terralyze repository showing build and test results](https://github.com/YOUR_USERNAME/terralyze/actions/workflows/ci.yml/badge.svg)

# 🌍 AI Climate Data Analyst

> **Auto-clean → AI Insights → Interactive Visualisations**
> An end-to-end data analytics system for climate & environmental datasets, powered by Python, LangChain, Claude (Anthropic), and Streamlit.

---

## 📸 System Overview

```
Raw CSV / JSON  ──►  Auto Cleaner  ──►  AI Insights Engine  ──►  Streamlit Dashboard
(climate data)        (Pandas)          (LangChain + Claude)       (Plotly + Charts)
```

---

## 🗂️ Project Structure

```
ai_data_analyst/
│
├── app.py                          # 🖥️  Streamlit dashboard (main entry point)
│
├── data/
│   ├── generate_data.py            # Sample climate data generator (w/ noise)
│   ├── climate_raw.csv             # Generated raw data (510 rows, 13 cols)
│   └── climate_raw.json            # JSON subset (100 records)
│
├── src/
│   ├── __init__.py
│   ├── data_cleaner.py             # 🧹 Auto-cleaning pipeline
│   ├── insights_engine.py          # 🤖 LangChain + Claude insights
│   └── visualizer.py               # 📊 Plotly chart generators
│
├── notebooks/
│   └── analysis_walkthrough.ipynb  # 📓 Step-by-step Jupyter walkthrough
│
├── assets/                         # Auto-generated HTML charts
├── requirements.txt
└── README.md
```

---

## 🚀 Quick Start

### 1. Clone & install
```bash
git clone <repo-url>
cd ai_data_analyst
pip install -r requirements.txt
```

### 2. Generate sample data
```bash
python data/generate_data.py
```

### 3. (Optional) Set your Anthropic API key for AI insights
```bash
export ANTHROPIC_API_KEY=sk-ant-...
```

### 4. Launch the Streamlit dashboard
```bash
streamlit run app.py
```

### 5. Or run the Jupyter notebook
```bash
cd notebooks
jupyter notebook analysis_walkthrough.ipynb
```

---

## 🧹 Auto-Cleaning Pipeline

`src/data_cleaner.py` — `ClimateDataCleaner`

| Step | What it does |
|------|-------------|
| **Load** | Reads CSV or JSON automatically |
| **Dedup** | Removes exact duplicate rows |
| **Date fix** | Coerces malformed dates → `NaT`, drops unfixable rows |
| **Type coercion** | Forces all numeric columns to `float64` |
| **Outlier clipping** | Domain-aware bounds (e.g. temperature: −60 to 60 °C) |
| **Imputation** | Group-median by city, fallback global median |
| **String normalisation** | Strip whitespace, title-case city names |
| **Feature engineering** | Adds `month`, `year`, `season`, `air_quality_category` |

Every step is logged in a structured `report` dict available via `cleaner.get_report()`.

---

## 🤖 AI Insights Engine

`src/insights_engine.py` — `InsightsEngine`

Uses **LangChain** + **Claude (claude-sonnet-4)** to analyse the cleaned dataset and return structured JSON insights:

```json
{
  "executive_summary": "...",
  "key_findings":      ["..."],
  "anomalies":         ["..."],
  "recommendations":   ["..."],
  "risk_indicators": {
    "high_aqi_cities":      ["Mumbai", "Delhi"],
    "co2_trend":            "rising",
    "temperature_concern":  "moderate"
  }
}
```

**Fallback mode**: If no API key is set, a rule-based analytical fallback is used — the dashboard still works fully without a key.

---

## 📊 Visualisations

`src/visualizer.py` — `ClimateVisualizer` — 8 interactive Plotly charts:

| Chart | Description |
|-------|-------------|
| 🌡️ Temperature Trend | Monthly average temperature time-series |
| 🏙️ AQI by City | Horizontal bar — highest-pollution cities |
| 🔵 CO₂ vs Temperature | Scatter plot sized by AQI, coloured by city |
| 🗓️ Seasonal Heatmap | City × Season AQI heatmap |
| 💨 PM2.5 Distribution | Histogram with WHO 35 µg/m³ threshold line |
| 📊 Correlation Matrix | Feature-to-feature correlation heatmap |
| 🌬️ Wind & Rainfall | Dual-axis monthly bar + line chart |
| 🍩 Air Quality Pie | Donut chart of AQI category breakdown |

---

## 🌍 Sample Data Schema

| Column | Type | Unit | Description |
|--------|------|------|-------------|
| `date` | datetime | — | Observation date |
| `city` | string | — | City name (15 global cities) |
| `temperature_c` | float | °C | Air temperature |
| `humidity_pct` | float | % | Relative humidity |
| `co2_ppm` | float | ppm | CO₂ concentration |
| `pm2_5_ugm3` | float | µg/m³ | Fine particulate matter |
| `aqi` | int | — | Air Quality Index (0–500) |
| `rainfall_mm` | float | mm | Daily rainfall |
| `wind_speed_kmh` | float | km/h | Wind speed |
| `uv_index` | float | — | UV index (0–11) |
| `pressure_hpa` | float | hPa | Atmospheric pressure |
| `visibility_km` | float | km | Visibility |
| `solar_radiation_wm2` | float | W/m² | Solar radiation |

**Intentional data quality issues injected for demo:**
- 7 % missing values per numeric column
- 10 duplicate rows
- 15 physical outliers (e.g. temperature 85°C, CO₂ 1200 ppm)
- 5 unparseable date strings

---

## 🛠️ Tech Stack

| Layer | Technology |
|-------|-----------|
| Data processing | Python 3.10+, Pandas 2.x, NumPy |
| AI / LLM | Anthropic Claude (claude-sonnet-4), LangChain |
| Visualisation | Plotly 5.x, Matplotlib |
| Dashboard | Streamlit 1.32+ |
| Notebooks | Jupyter, ipykernel |

---

## ⚙️ Environment Variables

| Variable | Required | Description |
|----------|----------|-------------|
| `ANTHROPIC_API_KEY` | Optional | Enables real AI insights via Claude API |

---
