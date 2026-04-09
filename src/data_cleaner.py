"""
src/data_cleaner.py
Auto-cleaning pipeline for climate/environment datasets.
Handles: missing values, duplicates, outliers, type coercion, date parsing.
"""

import pandas as pd
import numpy as np
import json
from pathlib import Path
from datetime import datetime


# ── Column config ─────────────────────────────────────────────────────────────
NUMERIC_BOUNDS = {
    "temperature_c":      (-60, 60),
    "humidity_pct":       (0, 100),
    "co2_ppm":            (300, 600),
    "pm2_5_ugm3":         (0, 400),
    "aqi":                (0, 500),
    "rainfall_mm":        (0, 300),
    "wind_speed_kmh":     (0, 150),
    "uv_index":           (0, 11),
    "pressure_hpa":       (870, 1085),
    "visibility_km":      (0, 50),
    "solar_radiation_wm2":(0, 1200),
}

NUMERIC_COLS  = list(NUMERIC_BOUNDS.keys())
REQUIRED_COLS = ["date", "city"] + NUMERIC_COLS


class ClimateDataCleaner:
    """End-to-end auto-cleaning pipeline for climate CSV/JSON data."""

    def __init__(self, filepath: str):
        self.filepath = Path(filepath)
        self.report: dict = {
            "source_file":  str(filepath),
            "timestamp":    datetime.now().isoformat(),
            "steps":        []
        }
        self.df_raw:   pd.DataFrame | None = None
        self.df_clean: pd.DataFrame | None = None

    # ── I/O ──────────────────────────────────────────────────────────────────
    def load(self) -> "ClimateDataCleaner":
        ext = self.filepath.suffix.lower()
        if ext == ".csv":
            self.df_raw = pd.read_csv(self.filepath, low_memory=False)
        elif ext == ".json":
            with open(self.filepath) as f:
                raw = json.load(f)
            self.df_raw = pd.DataFrame(raw) if isinstance(raw, list) else pd.DataFrame([raw])
        else:
            raise ValueError(f"Unsupported file type: {ext}")

        self._log("load", f"Loaded {len(self.df_raw)} rows × {len(self.df_raw.columns)} cols")
        self.df_clean = self.df_raw.copy()
        return self

    # ── Pipeline steps ────────────────────────────────────────────────────────
    def drop_duplicates(self) -> "ClimateDataCleaner":
        before = len(self.df_clean)
        self.df_clean.drop_duplicates(inplace=True)
        self.df_clean.reset_index(drop=True, inplace=True)
        removed = before - len(self.df_clean)
        self._log("duplicates", f"Removed {removed} duplicate rows")
        return self

    def fix_dates(self) -> "ClimateDataCleaner":
        if "date" not in self.df_clean.columns:
            return self
        original_nulls = self.df_clean["date"].isna().sum()
        self.df_clean["date"] = pd.to_datetime(
            self.df_clean["date"], errors="coerce"
        )
        new_nulls = self.df_clean["date"].isna().sum()
        self._log("dates", f"Coerced dates; {new_nulls - original_nulls} bad dates → NaT")
        return self

    def coerce_numerics(self) -> "ClimateDataCleaner":
        present = [c for c in NUMERIC_COLS if c in self.df_clean.columns]
        for col in present:
            self.df_clean[col] = pd.to_numeric(self.df_clean[col], errors="coerce")
        self._log("types", f"Coerced {len(present)} columns to numeric")
        return self

    def clip_outliers(self) -> "ClimateDataCleaner":
        clipped_total = 0
        for col, (lo, hi) in NUMERIC_BOUNDS.items():
            if col not in self.df_clean.columns:
                continue
            mask = self.df_clean[col].notna()
            out_of_range = ((self.df_clean.loc[mask, col] < lo) |
                            (self.df_clean.loc[mask, col] > hi)).sum()
            self.df_clean[col] = self.df_clean[col].clip(lower=lo, upper=hi)
            clipped_total += out_of_range
        self._log("outliers", f"Clipped {clipped_total} out-of-range values")
        return self

    def impute_missing(self) -> "ClimateDataCleaner":
        imputed = {}
        for col in NUMERIC_COLS:
            if col not in self.df_clean.columns:
                continue
            n_missing = self.df_clean[col].isna().sum()
            if n_missing > 0:
                # Group-median imputation by city when possible
                if "city" in self.df_clean.columns:
                    self.df_clean[col] = self.df_clean.groupby("city")[col].transform(
                        lambda x: x.fillna(x.median())
                    )
                # Fall back to global median
                remaining = self.df_clean[col].isna().sum()
                if remaining > 0:
                    self.df_clean[col].fillna(self.df_clean[col].median(), inplace=True)
                imputed[col] = int(n_missing)
        self._log("imputation", f"Imputed missing values: {imputed}")
        return self

    def standardise_strings(self) -> "ClimateDataCleaner":
        for col in self.df_clean.select_dtypes(include="object").columns:
            self.df_clean[col] = self.df_clean[col].str.strip().str.title()
        self._log("strings", "Stripped and title-cased all string columns")
        return self

    def drop_unusable_rows(self) -> "ClimateDataCleaner":
        """Drop rows where the date is still NaT after fixing."""
        before = len(self.df_clean)
        if "date" in self.df_clean.columns:
            self.df_clean.dropna(subset=["date"], inplace=True)
            self.df_clean.reset_index(drop=True, inplace=True)
        removed = before - len(self.df_clean)
        self._log("unusable_rows", f"Dropped {removed} rows with unparseable dates")
        return self

    def add_derived_features(self) -> "ClimateDataCleaner":
        if "date" in self.df_clean.columns:
            self.df_clean["month"]   = self.df_clean["date"].dt.month
            self.df_clean["season"]  = self.df_clean["month"].map(
                {12: "Winter", 1: "Winter", 2: "Winter",
                 3: "Spring", 4: "Spring", 5: "Spring",
                 6: "Summer", 7: "Summer", 8: "Summer",
                 9: "Autumn", 10: "Autumn", 11: "Autumn"}
            )
            self.df_clean["year"]    = self.df_clean["date"].dt.year
        if "aqi" in self.df_clean.columns:
            self.df_clean["air_quality_category"] = pd.cut(
                self.df_clean["aqi"],
                bins=[0, 50, 100, 150, 200, 300, 500],
                labels=["Good", "Moderate", "Unhealthy for Sensitive",
                        "Unhealthy", "Very Unhealthy", "Hazardous"]
            )
        self._log("features", "Added month, season, year, air_quality_category")
        return self

    # ── Main entry point ──────────────────────────────────────────────────────
    def run(self) -> pd.DataFrame:
        (self
         .load()
         .drop_duplicates()
         .fix_dates()
         .coerce_numerics()
         .clip_outliers()
         .impute_missing()
         .standardise_strings()
         .drop_unusable_rows()
         .add_derived_features())

        self.report["rows_in"]  = len(self.df_raw)
        self.report["rows_out"] = len(self.df_clean)
        self.report["cols_out"] = list(self.df_clean.columns)
        return self.df_clean

    def save(self, output_path: str | None = None) -> "ClimateDataCleaner":
        if output_path is None:
            output_path = self.filepath.parent / (self.filepath.stem + "_clean.csv")
        self.df_clean.to_csv(output_path, index=False)
        self._log("save", f"Clean data saved → {output_path}")
        return self

    def get_report(self) -> dict:
        return self.report

    # ── Internal ──────────────────────────────────────────────────────────────
    def _log(self, step: str, message: str):
        self.report["steps"].append({"step": step, "detail": message})
        print(f"  [{step}] {message}")


# ── CLI usage ─────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import argparse, json as _json
    parser = argparse.ArgumentParser()
    parser.add_argument("--input",  default="data/climate_raw.csv")
    parser.add_argument("--output", default=None)
    args = parser.parse_args()

    cleaner = ClimateDataCleaner(args.input)
    df_clean = cleaner.run()
    cleaner.save(args.output)

    print("\n── Cleaning Report ──────────────────────────")
    print(_json.dumps(cleaner.get_report(), indent=2, default=str))
