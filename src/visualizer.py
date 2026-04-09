"""
src/visualizer.py
Chart generation for the AI Climate Analyst.
All charts return Plotly figures so they can be embedded in Streamlit
or exported as static images.
"""

import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.io as pio

# ── Brand palette ─────────────────────────────────────────────────────────────
PALETTE = {
    "primary":   "#00C9A7",
    "secondary": "#845EC2",
    "accent":    "#FF6B6B",
    "warn":      "#FFC75F",
    "bg":        "#0F1117",
    "surface":   "#1A1E2E",
    "text":      "#E8ECF4",
}
COLOR_SEQ = [PALETTE["primary"], PALETTE["warn"], PALETTE["accent"],
             PALETTE["secondary"], "#4D8FFF", "#FF9671", "#B0A8B9"]

TEMPLATE = "plotly_dark"


def _fig_layout(fig: go.Figure, title: str) -> go.Figure:
    """Apply consistent dark layout."""
    fig.update_layout(
        title=dict(text=title, font=dict(size=18, color=PALETTE["text"])),
        paper_bgcolor=PALETTE["bg"],
        plot_bgcolor=PALETTE["surface"],
        font=dict(color=PALETTE["text"], family="Inter, sans-serif"),
        margin=dict(l=40, r=40, t=60, b=40),
        legend=dict(bgcolor="rgba(0,0,0,0)", borderwidth=0),
    )
    return fig


class ClimateVisualizer:
    """Generates all dashboard charts from a clean climate DataFrame."""

    def __init__(self, df: pd.DataFrame):
        self.df = df

    # ── 1. Time-series: temperature trend ─────────────────────────────────────
    def temperature_trend(self) -> go.Figure:
        if "date" not in self.df.columns or "temperature_c" not in self.df.columns:
            return go.Figure()
        monthly = (self.df.set_index("date")["temperature_c"]
                   .resample("ME").mean().reset_index())
        fig = px.line(monthly, x="date", y="temperature_c",
                      color_discrete_sequence=[PALETTE["primary"]],
                      template=TEMPLATE)
        fig.update_traces(line_width=2.5)
        return _fig_layout(fig, "🌡️ Monthly Average Temperature (°C)")

    # ── 2. AQI distribution by city ───────────────────────────────────────────
    def aqi_by_city(self) -> go.Figure:
        if "city" not in self.df.columns or "aqi" not in self.df.columns:
            return go.Figure()
        city_aqi = (self.df.groupby("city")["aqi"].mean()
                    .sort_values(ascending=True).reset_index())
        fig = px.bar(city_aqi, x="aqi", y="city", orientation="h",
                     color="aqi",
                     color_continuous_scale=["#00C9A7", "#FFC75F", "#FF6B6B"],
                     template=TEMPLATE)
        fig.update_layout(coloraxis_showscale=False)
        return _fig_layout(fig, "🏙️ Average AQI by City")

    # ── 3. CO₂ vs Temperature scatter ─────────────────────────────────────────
    def co2_vs_temp(self) -> go.Figure:
        cols = ["co2_ppm", "temperature_c", "city", "aqi"]
        missing = [c for c in cols if c not in self.df.columns]
        if missing:
            return go.Figure()
        fig = px.scatter(self.df.sample(min(300, len(self.df))),
                         x="temperature_c", y="co2_ppm",
                         color="city", size="aqi",
                         opacity=0.75,
                         color_discrete_sequence=COLOR_SEQ,
                         template=TEMPLATE)
        return _fig_layout(fig, "🔵 CO₂ vs Temperature (sized by AQI)")

    # ── 4. Seasonal heatmap ───────────────────────────────────────────────────
    def seasonal_heatmap(self) -> go.Figure:
        needed = ["season", "city", "aqi"]
        if any(c not in self.df.columns for c in needed):
            return go.Figure()
        pivot = self.df.pivot_table(values="aqi", index="city",
                                    columns="season", aggfunc="mean").round(1)
        season_order = [s for s in ["Spring", "Summer", "Autumn", "Winter"]
                        if s in pivot.columns]
        pivot = pivot[season_order]
        fig = go.Figure(go.Heatmap(
            z=pivot.values, x=pivot.columns.tolist(), y=pivot.index.tolist(),
            colorscale=[[0, "#00C9A7"], [0.5, "#FFC75F"], [1, "#FF6B6B"]],
            text=pivot.values.round(1), texttemplate="%{text}",
        ))
        return _fig_layout(fig, "🗓️ Seasonal AQI Heatmap (City × Season)")

    # ── 5. PM2.5 histogram ────────────────────────────────────────────────────
    def pm25_distribution(self) -> go.Figure:
        if "pm2_5_ugm3" not in self.df.columns:
            return go.Figure()
        fig = px.histogram(self.df, x="pm2_5_ugm3", nbins=40,
                           color_discrete_sequence=[PALETTE["secondary"]],
                           template=TEMPLATE)
        fig.add_vline(x=35, line_dash="dash", line_color=PALETTE["accent"],
                      annotation_text="WHO limit (35 µg/m³)",
                      annotation_position="top right")
        return _fig_layout(fig, "💨 PM2.5 Distribution (µg/m³)")

    # ── 6. Correlation matrix ─────────────────────────────────────────────────
    def correlation_matrix(self) -> go.Figure:
        num_cols = [c for c in
                    ["temperature_c", "humidity_pct", "co2_ppm",
                     "pm2_5_ugm3", "aqi", "rainfall_mm", "wind_speed_kmh",
                     "uv_index", "pressure_hpa"]
                    if c in self.df.columns]
        corr = self.df[num_cols].corr().round(2)
        fig = go.Figure(go.Heatmap(
            z=corr.values,
            x=corr.columns.tolist(),
            y=corr.index.tolist(),
            colorscale="RdBu",
            zmid=0,
            text=corr.values.round(2),
            texttemplate="%{text}",
            colorbar=dict(title="r"),
        ))
        return _fig_layout(fig, "📊 Feature Correlation Matrix")

    # ── 7. Wind-speed & rainfall dual-axis ────────────────────────────────────
    def wind_rainfall(self) -> go.Figure:
        need = ["date", "wind_speed_kmh", "rainfall_mm"]
        if any(c not in self.df.columns for c in need):
            return go.Figure()
        monthly = (self.df.set_index("date")[["wind_speed_kmh", "rainfall_mm"]]
                   .resample("ME").mean().reset_index())
        fig = make_subplots(specs=[[{"secondary_y": True}]])
        fig.add_trace(go.Bar(x=monthly["date"], y=monthly["rainfall_mm"],
                             name="Rainfall (mm)", marker_color=PALETTE["secondary"],
                             opacity=0.7), secondary_y=False)
        fig.add_trace(go.Scatter(x=monthly["date"], y=monthly["wind_speed_kmh"],
                                 name="Wind Speed (km/h)", mode="lines+markers",
                                 line=dict(color=PALETTE["primary"], width=2.5)),
                      secondary_y=True)
        fig.update_layout(template=TEMPLATE, paper_bgcolor=PALETTE["bg"],
                          plot_bgcolor=PALETTE["surface"],
                          font=dict(color=PALETTE["text"]),
                          title=dict(text="🌬️ Monthly Wind Speed & Rainfall",
                                     font=dict(size=18)))
        return fig

    # ── 8. Air-quality category pie ───────────────────────────────────────────
    def air_quality_pie(self) -> go.Figure:
        col = "air_quality_category"
        if col not in self.df.columns:
            return go.Figure()
        counts = self.df[col].value_counts().reset_index()
        counts.columns = ["category", "count"]
        fig = px.pie(counts, names="category", values="count",
                     color_discrete_sequence=COLOR_SEQ,
                     hole=0.45, template=TEMPLATE)
        return _fig_layout(fig, "🍩 Air Quality Category Breakdown")

    # ── Export all charts ─────────────────────────────────────────────────────
    def get_all_figures(self) -> dict[str, go.Figure]:
        return {
            "temperature_trend":   self.temperature_trend(),
            "aqi_by_city":         self.aqi_by_city(),
            "co2_vs_temp":         self.co2_vs_temp(),
            "seasonal_heatmap":    self.seasonal_heatmap(),
            "pm25_distribution":   self.pm25_distribution(),
            "correlation_matrix":  self.correlation_matrix(),
            "wind_rainfall":       self.wind_rainfall(),
            "air_quality_pie":     self.air_quality_pie(),
        }

    def save_all_html(self, output_dir: str = "assets"):
        import os
        os.makedirs(output_dir, exist_ok=True)
        for name, fig in self.get_all_figures().items():
            path = f"{output_dir}/{name}.html"
            pio.write_html(fig, path)
            print(f"  Saved → {path}")


# ── CLI ───────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import sys
    sys.path.insert(0, ".")
    from src.data_cleaner import ClimateDataCleaner

    cleaner = ClimateDataCleaner("data/climate_raw.csv")
    df = cleaner.run()
    viz = ClimateVisualizer(df)
    viz.save_all_html("assets")
    print("✅ All charts saved to assets/")
