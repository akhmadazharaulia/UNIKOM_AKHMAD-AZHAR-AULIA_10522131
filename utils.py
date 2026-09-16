"""
Utility functions shared across pages:
- Column auto-detection (Date, SKU, Item, Quantity)
- Data cleaning & daily aggregation (CRISP-DM Data Preparation)
- SARIMA modeling helpers
- BOM matching helpers
- Weekday/weekend evaluation
"""
import re
import pandas as pd
import numpy as np


# ---------------------------------------------------------------------
# Column auto-detection — POS exports vary in header naming
# ---------------------------------------------------------------------
def _normalize_col(name: str) -> str:
    """Bersihkan BOM, non-breaking space, dan whitespace berlebih dari nama kolom."""
    return re.sub(r"\s+", " ", str(name).replace("\ufeff", "").replace("\xa0", " ")).strip().lower()


def guess_column(df: pd.DataFrame, candidates: list[str]) -> str | None:
    """Cari kolom yang cocok dengan salah satu candidate name.
    Exact match (setelah normalisasi) diprioritaskan dulu, baru fallback ke
    word-boundary partial match berdasarkan urutan candidate — bukan asal
    ketemu kolom pertama yang mengandung substring."""
    norm_map = {_normalize_col(c): c for c in df.columns}

    # 1. Exact match
    for cand in candidates:
        key = _normalize_col(cand)
        if key in norm_map:
            return norm_map[key]

    # 2. Word-boundary partial match, prioritas urutan candidate
    for cand in candidates:
        pattern = r"\b" + re.escape(_normalize_col(cand)) + r"\b"
        for c in df.columns:
            if re.search(pattern, _normalize_col(c)):
                return c

    return None


def auto_detect_pos_columns(df: pd.DataFrame) -> dict:
    return {
        "date": guess_column(df, ["Date", "Tanggal", "Transaction Date"]),
        "sku": guess_column(df, ["SKU", "Kode", "Product Code"]),
        "item": guess_column(df, ["Item", "Menu", "Product", "Item Name"]),
        "qty": guess_column(df, ["Quantity", "Qty", "Jumlah"]),
    }


def auto_detect_bom_columns(df: pd.DataFrame) -> dict:
    return {
        "sku": guess_column(df, ["SKU", "Kode", "Product Code"]),
        "menu": guess_column(df, ["Menu", "Item", "Nama Menu"]),
        "bahan": guess_column(df, ["Bahan Baku", "Resep Bahan", "Bahan", "Ingredient", "Material"]),
        "qty": guess_column(df, ["Qty", "Qty Resep", "Quantity", "Jumlah"]),
        "satuan": guess_column(df, ["Satuan", "Unit"]),
    }


# ---------------------------------------------------------------------
# CRISP-DM: Data Preparation — Cleaning + Daily Transformation
# ---------------------------------------------------------------------
def clean_pos_data(df: pd.DataFrame, col_map: dict) -> pd.DataFrame:
    """Step: Data Selection + Data Cleaning (Section 3.2.4.3 / 4.2 of thesis)."""
    date_c, sku_c, item_c, qty_c = (
        col_map["date"], col_map["sku"], col_map["item"], col_map["qty"]
    )
    work = df[[date_c, sku_c, item_c, qty_c]].copy()
    work.columns = ["Date", "SKU", "Item", "Quantity"]

    # Data Cleaning
    work = work.dropna()
    work = work.drop_duplicates()

    # Robust date parsing: try common explicit formats first (faster, no
    # warnings), then fall back to flexible parsing for anything left.
    raw_dates = work["Date"]
    parsed = pd.to_datetime(raw_dates, errors="coerce", format="mixed")
    work["Date"] = parsed
    work = work.dropna(subset=["Date"])
    work["Quantity"] = pd.to_numeric(work["Quantity"], errors="coerce")
    work = work.dropna(subset=["Quantity"])
    work = work[work["Quantity"] > 0]
    work["SKU"] = work["SKU"].astype(str).str.strip()
    work["Date"] = work["Date"].dt.date

    return work.reset_index(drop=True)


def transform_to_daily(df_clean: pd.DataFrame) -> pd.DataFrame:
    """Step: Data Transformation — aggregate to continuous daily time series."""
    daily = df_clean.groupby("Date")["Quantity"].sum()
    full_range = pd.date_range(start=daily.index.min(), end=daily.index.max(), freq="D")
    daily = daily.reindex(full_range, fill_value=0)
    daily = daily.reset_index()
    daily.columns = ["Date", "Total_Quantity"]
    return daily


def split_train_test(df_daily: pd.DataFrame, test_ratio: float = 0.2):
    """Step: Data Splitting — chronological time-based split."""
    n = len(df_daily)
    split_idx = int(n * (1 - test_ratio))
    train = df_daily.iloc[:split_idx].reset_index(drop=True)
    test = df_daily.iloc[split_idx:].reset_index(drop=True)
    return train, test


# ---------------------------------------------------------------------
# BOM Integration
# ---------------------------------------------------------------------
def integrate_bom(forecast_df: pd.DataFrame, df_bom: pd.DataFrame,
                   bom_col_map: dict, sku_qty_map: dict) -> pd.DataFrame:
    """
    forecast_df: DataFrame with columns [Date, SKU, Predicted_Qty]
    df_bom: raw BOM dataframe
    Returns: DataFrame with Date, SKU, Menu, Bahan Baku, Qty Resep, Satuan, Total Kebutuhan
    """
    sku_c = bom_col_map["sku"]
    menu_c = bom_col_map.get("menu")
    bahan_c = bom_col_map["bahan"]
    qty_c = bom_col_map["qty"]
    satuan_c = bom_col_map.get("satuan")

    cols = [sku_c, bahan_c, qty_c]
    rename = {sku_c: "SKU", bahan_c: "Bahan_Baku", qty_c: "Qty_Resep"}
    if menu_c:
        cols.append(menu_c)
        rename[menu_c] = "Menu"
    if satuan_c:
        cols.append(satuan_c)
        rename[satuan_c] = "Satuan"

    bom_work = df_bom[cols].copy().rename(columns=rename)
    bom_work["SKU"] = bom_work["SKU"].astype(str).str.strip()
    bom_work["Qty_Resep"] = pd.to_numeric(bom_work["Qty_Resep"], errors="coerce")
    bom_work = bom_work.dropna(subset=["Qty_Resep"])

    merged = forecast_df.merge(bom_work, on="SKU", how="inner")
    merged["Total_Kebutuhan"] = merged["Predicted_Qty"] * merged["Qty_Resep"]
    return merged


# ---------------------------------------------------------------------
# Evaluation metrics
# ---------------------------------------------------------------------
def evaluation_metrics(actual: pd.Series, predicted: pd.Series) -> dict:
    actual = np.asarray(actual, dtype=float)
    predicted = np.asarray(predicted, dtype=float)
    mae = np.mean(np.abs(actual - predicted))
    rmse = np.sqrt(np.mean((actual - predicted) ** 2))
    # avoid division by zero in MAPE
    nonzero = actual != 0
    mape = np.mean(np.abs((actual[nonzero] - predicted[nonzero]) / actual[nonzero])) * 100 if nonzero.any() else np.nan
    return {"MAE": round(mae, 2), "RMSE": round(rmse, 2), "MAPE (%)": round(mape, 2)}


def weekday_weekend_split(dates: pd.Series, actual: pd.Series, predicted: pd.Series) -> pd.DataFrame:
    dates = pd.to_datetime(pd.Series(dates))
    df = pd.DataFrame({
        "Date": dates,
        "Actual": actual,
        "Predicted": predicted,
    })
    df["Jenis"] = np.where(df["Date"].dt.dayofweek < 5, "Weekday", "Weekend")
    return df