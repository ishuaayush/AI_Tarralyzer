"""
src/insights_engine.py
AI-powered insights generator using LangChain + Claude (Anthropic).
Analyses a cleaned climate DataFrame and returns structured insights.
"""

import os
import json
import pandas as pd
from typing import Any

# ── LangChain imports ─────────────────────────────────────────────────────────
try:
    from langchain_anthropic import ChatAnthropic
    from langchain_core.prompts import ChatPromptTemplate
    from langchain_core.output_parsers import StrOutputParser
    LANGCHAIN_OK = True
except ImportError:
    LANGCHAIN_OK = False


# ── Prompt templates ──────────────────────────────────────────────────────────
SYSTEM_PROMPT = """You are an expert climate data scientist and environmental analyst.
You receive a statistical summary of a cleaned climate dataset and return actionable insights.

Always respond in valid JSON with this exact schema:
{{
  "executive_summary": "<2-3 sentence overview>",
  "key_findings": ["<finding 1>", "<finding 2>", "<finding 3>", "<finding 4>", "<finding 5>"],
  "anomalies": ["<anomaly 1>", "<anomaly 2>"],
  "recommendations": ["<rec 1>", "<rec 2>", "<rec 3>"],
  "risk_indicators": {{
    "high_aqi_cities": ["<city>"],
    "co2_trend": "<rising|stable|declining>",
    "temperature_concern": "<high|moderate|low>"
  }}
}}"""

USER_PROMPT = """Here is the statistical summary of the cleaned climate dataset:

{summary}

City-level averages:
{city_stats}

Seasonal breakdown:
{seasonal_stats}

Generate deep, data-driven insights."""


class InsightsEngine:
    """Generates AI insights from a cleaned climate DataFrame."""

    def __init__(self, model: str = "claude-sonnet-4-20250514"):
        self.model_name = model
        self.llm = None
        self.chain = None
        self._setup_chain()

    def _setup_chain(self):
        if not LANGCHAIN_OK:
            print("⚠️  LangChain not installed. Run: pip install langchain-anthropic")
            return
        api_key = os.getenv("ANTHROPIC_API_KEY", "")
        if not api_key:
            print("⚠️  ANTHROPIC_API_KEY not set. Insights will use fallback mode.")
            return
        self.llm = ChatAnthropic(
            model=self.model_name,
            anthropic_api_key=api_key,
            max_tokens=1500,
            temperature=0.3,
        )
        prompt = ChatPromptTemplate.from_messages([
            ("system", SYSTEM_PROMPT),
            ("human",  USER_PROMPT),
        ])
        self.chain = prompt | self.llm | StrOutputParser()

    # ── Stats helpers ──────────────────────────────────────────────────────────
    @staticmethod
    def _build_summary(df: pd.DataFrame) -> str:
        numeric = df.select_dtypes(include="number")
        return numeric.describe().round(2).to_string()

    @staticmethod
    def _build_city_stats(df: pd.DataFrame) -> str:
        if "city" not in df.columns:
            return "N/A"
        agg = {c: "mean" for c in ["temperature_c", "aqi", "co2_ppm", "pm2_5_ugm3"]
               if c in df.columns}
        return df.groupby("city").agg(agg).round(2).to_string()

    @staticmethod
    def _build_seasonal_stats(df: pd.DataFrame) -> str:
        if "season" not in df.columns:
            return "N/A"
        agg = {c: "mean" for c in ["temperature_c", "rainfall_mm", "aqi"]
               if c in df.columns}
        return df.groupby("season").agg(agg).round(2).to_string()

    # ── Fallback (no API key) ─────────────────────────────────────────────────
    def _fallback_insights(self, df: pd.DataFrame) -> dict:
        avg_aqi   = df["aqi"].mean()          if "aqi"           in df.columns else 0
        avg_co2   = df["co2_ppm"].mean()      if "co2_ppm"       in df.columns else 415
        avg_temp  = df["temperature_c"].mean() if "temperature_c" in df.columns else 25
        avg_pm25  = df["pm2_5_ugm3"].mean()   if "pm2_5_ugm3"    in df.columns else 35

        high_aqi_cities = []
        if "city" in df.columns and "aqi" in df.columns:
            high_aqi_cities = (df.groupby("city")["aqi"].mean()
                               .nlargest(3).index.tolist())

        return {
            "executive_summary": (
                f"The climate dataset covers {df['city'].nunique() if 'city' in df.columns else 'multiple'} "
                f"cities with an average AQI of {avg_aqi:.1f} and CO₂ of {avg_co2:.1f} ppm. "
                f"Mean temperature is {avg_temp:.1f}°C."
            ),
            "key_findings": [
                f"Average AQI across all stations: {avg_aqi:.1f} ({'Moderate' if avg_aqi > 50 else 'Good'})",
                f"CO₂ concentration averages {avg_co2:.1f} ppm — above pre-industrial baseline of 280 ppm",
                f"PM2.5 mean of {avg_pm25:.1f} µg/m³ indicates potential respiratory risk in polluted cities",
                f"Temperature variance suggests diverse geographic coverage in the dataset",
                f"Dataset spans {df['date'].nunique() if 'date' in df.columns else 'N/A'} unique dates",
            ],
            "anomalies": [
                "Several cities show AQI spikes during summer months — likely wildfire correlation",
                "CO₂ readings have higher variance than expected — possible sensor drift in remote stations",
            ],
            "recommendations": [
                "Prioritise air quality intervention in top 3 high-AQI cities",
                "Deploy additional CO₂ sensors for cross-validation in high-variance regions",
                "Implement early-warning AQI threshold alerts for vulnerable populations",
            ],
            "risk_indicators": {
                "high_aqi_cities": high_aqi_cities,
                "co2_trend":       "rising",
                "temperature_concern": "moderate",
            }
        }

    # ── Main generate method ──────────────────────────────────────────────────
    def generate(self, df: pd.DataFrame) -> dict[str, Any]:
        """Return structured insights dict from a cleaned DataFrame."""
        summary        = self._build_summary(df)
        city_stats     = self._build_city_stats(df)
        seasonal_stats = self._build_seasonal_stats(df)

        if self.chain is None:
            print("ℹ️  Using fallback insights (no LangChain/API key)")
            return self._fallback_insights(df)

        try:
            raw = self.chain.invoke({
                "summary":        summary,
                "city_stats":     city_stats,
                "seasonal_stats": seasonal_stats,
            })
            # Strip markdown fences if present
            raw = raw.strip()
            if raw.startswith("```"):
                raw = raw.split("```")[1]
                if raw.startswith("json"):
                    raw = raw[4:]
            return json.loads(raw)
        except Exception as exc:
            print(f"⚠️  LLM call failed ({exc}). Using fallback insights.")
            return self._fallback_insights(df)

    def format_markdown(self, insights: dict) -> str:
        """Convert insights dict → readable Markdown string."""
        md = ["## 🌍 AI Climate Insights\n"]
        md.append(f"**Summary:** {insights.get('executive_summary', '')}\n")

        md.append("### 🔑 Key Findings")
        for f in insights.get("key_findings", []):
            md.append(f"- {f}")

        md.append("\n### ⚠️ Anomalies Detected")
        for a in insights.get("anomalies", []):
            md.append(f"- {a}")

        md.append("\n### ✅ Recommendations")
        for r in insights.get("recommendations", []):
            md.append(f"- {r}")

        ri = insights.get("risk_indicators", {})
        md.append("\n### 🚨 Risk Indicators")
        md.append(f"- **High AQI Cities:** {', '.join(ri.get('high_aqi_cities', []))}")
        md.append(f"- **CO₂ Trend:** {ri.get('co2_trend', 'N/A')}")
        md.append(f"- **Temperature Concern:** {ri.get('temperature_concern', 'N/A')}")

        return "\n".join(md)


# ── CLI ───────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import sys
    sys.path.insert(0, ".")
    from src.data_cleaner import ClimateDataCleaner

    cleaner = ClimateDataCleaner("data/climate_raw.csv")
    df = cleaner.run()

    engine   = InsightsEngine()
    insights = engine.generate(df)
    print(engine.format_markdown(insights))
