"""Data layer: A-share OHLCV data fetching and panel construction.

All data cleaning uses **Polars** for performance (lazy scan, streaming pivot,
zero-copy to numpy). Pandas is only used for AkShare API compatibility.

Supports two modes:
1. Load from existing parquet file (factor_investing/data/processed/)
2. Fetch live via AkShare (fallback)
"""

import json
import logging
from pathlib import Path
from typing import Optional

import numpy as np
import polars as pl

try:
    import pandas as pd
    HAS_PANDAS = True
except ImportError:
    HAS_PANDAS = False

try:
    import akshare as ak
    HAS_AKSHARE = True
except ImportError:
    HAS_AKSHARE = False

from config import DATA_CACHE_DIR, DATA_START, DATA_END, DATA_FEATURES


def _save_panel_cache(panel: dict, path: Path):
    """Save panel to disk using numpy .npz (safe, no pickle RCE risk)."""
    arrays = {}
    meta = {}
    for k, v in panel.items():
        if isinstance(v, np.ndarray):
            arrays[k] = v
        elif isinstance(v, list):
            meta[k] = v
        else:
            meta[k] = v
    np.savez_compressed(path.with_suffix(".npz"), **arrays)
    meta_path = path.with_suffix(".meta.json")
    with open(meta_path, "w") as f:
        json.dump(meta, f, default=str)


def _load_panel_cache(path: Path) -> Optional[dict]:
    """Load panel from .npz + .meta.json cache."""
    npz_path = path.with_suffix(".npz")
    meta_path = path.with_suffix(".meta.json")
    if not npz_path.exists():
        return None
    try:
        data = dict(np.load(npz_path, allow_pickle=False))
        if meta_path.exists():
            with open(meta_path, "r") as f:
                meta = json.load(f)
            data.update(meta)
        return data
    except Exception:
        return None

logger = logging.getLogger(__name__)

_FACTOR_INVESTING_DATA = Path(__file__).resolve().parent.parent.parent / "factor_investing" / "data"
_PROCESSED_DAILY = _FACTOR_INVESTING_DATA / "processed" / "daily_20180101_20241231.parquet"
_RAW_DIR = _FACTOR_INVESTING_DATA / "raw"


# ---------------------------------------------------------------------------
# Polars-based data cleaning
# ---------------------------------------------------------------------------

def _pl_pivot_to_numpy(
    lf: pl.LazyFrame,
    index_col: str,
    columns_col: str,
    values_col: str,
    row_order: list,
    col_order: list,
) -> np.ndarray:
    """Pivot a long-form LazyFrame into a (len(col_order), len(row_order)) numpy array.

    Result shape is (M_stocks, T_dates) — stocks on axis 0, dates on axis 1.
    """
    df = (
        lf
        .filter(pl.col(index_col).is_in(row_order))
        .filter(pl.col(columns_col).is_in(col_order))
        .select([index_col, columns_col, values_col])
        .collect()
    )
    pivoted = df.pivot(on=columns_col, index=index_col, values=values_col)
    pivoted = pivoted.sort(index_col)

    present_cols = [c for c in col_order if c in pivoted.columns]
    missing_cols = [c for c in col_order if c not in pivoted.columns]
    arr = pivoted.select(present_cols).to_numpy().astype(np.float64)

    if missing_cols:
        full = np.full((len(col_order), len(row_order)), np.nan)
        col_idx = {c: i for i, c in enumerate(col_order)}
        for j, c in enumerate(present_cols):
            full[col_idx[c], :] = arr[:, j] if j < arr.shape[1] else np.nan
        return full

    return arr


def build_panel_from_parquet(
    parquet_path: Path = _PROCESSED_DAILY,
    max_stocks: Optional[int] = 300,
    min_days: int = 500,
    include_fundamentals: bool = False,
) -> dict[str, np.ndarray]:
    """Build market panel from parquet using Polars for all data cleaning.

    Returns panel dict with keys: open, high, low, close, volume, amount,
    vwap, returns, codes, dates. Optionally adds fundamental fields.
    """
    fund_suffix = "_fund" if include_fundamentals else ""
    cache_key = f"parquet_panel_{max_stocks}_{min_days}{fund_suffix}"
    cache_path = DATA_CACHE_DIR / cache_key
    cached = _load_panel_cache(cache_path)
    if cached is not None:
        logger.info(f"Loaded cached panel (from local data) {cache_path}")
        return cached

    logger.info(f"Using local data (Polars): {parquet_path}")
    if not parquet_path.exists():
        raise FileNotFoundError(f"Local data not found: {parquet_path}")

    lf = pl.scan_parquet(parquet_path)

    # Count trading days per stock
    day_counts = (
        lf
        .group_by("ts_code")
        .agg(pl.col("trade_date").count().alias("n_days"))
        .filter(pl.col("n_days") >= min_days)
        .sort("n_days", descending=True)
        .collect()
    )
    valid_stocks = day_counts["ts_code"].to_list()
    logger.info(f"Stocks with >= {min_days} days: {len(valid_stocks)}")

    if max_stocks and len(valid_stocks) > max_stocks:
        valid_stocks = valid_stocks[:max_stocks]

    # Find common trading dates across selected stocks
    dates_df = (
        lf
        .filter(pl.col("ts_code").is_in(valid_stocks))
        .group_by("trade_date")
        .agg(pl.col("ts_code").n_unique().alias("n_stocks"))
        .filter(pl.col("n_stocks") == len(valid_stocks))
        .sort("trade_date")
        .collect()
    )
    common_dates = dates_df["trade_date"].to_list()
    logger.info(f"Common trading dates: {len(common_dates)}")

    if len(common_dates) < 60:
        raise ValueError(f"Only {len(common_dates)} common dates, need >= 60")

    codes = sorted(valid_stocks)
    M = len(codes)
    T = len(common_dates)
    common_dates_sorted = sorted(common_dates)
    logger.info(f"Building panel (Polars): {M} assets x {T} days")

    # Filter to valid stocks + common dates
    filtered = (
        lf
        .filter(pl.col("ts_code").is_in(codes))
        .filter(pl.col("trade_date").is_in(common_dates_sorted))
    )

    # Pivot each OHLCV field: result is (M, T) with stocks as rows, dates as cols
    field_map = {
        "open": "open",
        "high": "high",
        "low": "low",
        "close": "close",
        "volume": "vol",
        "amount": "amount",
    }

    panel = {}
    for panel_name, col_name in field_map.items():
        df_piv = (
            filtered
            .select(["trade_date", "ts_code", col_name])
            .collect()
            .pivot(on="trade_date", index="ts_code", values=col_name)
        )
        # Ensure consistent row order (sorted codes)
        df_piv = df_piv.sort("ts_code")
        # Extract numeric columns in date-sorted order
        date_cols = sorted([c for c in df_piv.columns if c != "ts_code"])
        arr = df_piv.select(date_cols).to_numpy().astype(np.float64)
        panel[panel_name] = arr

    # Derived fields
    close = panel["close"]
    volume = panel["volume"]
    amount = panel["amount"]

    with np.errstate(divide="ignore", invalid="ignore"):
        vwap = np.where(volume > 0, amount / volume, close)
    panel["vwap"] = vwap

    returns = np.full_like(close, np.nan)
    returns[:, 1:] = close[:, 1:] / close[:, :-1] - 1.0
    panel["returns"] = returns

    clean_codes = [c.split(".")[0] if "." in c else c for c in codes]
    panel["codes"] = clean_codes
    panel["dates"] = np.array(common_dates_sorted)

    if include_fundamentals and _RAW_DIR.exists():
        logger.info("Merging fundamental data (Polars) ...")
        _merge_fundamentals_into_panel_pl(panel, codes, common_dates_sorted)

    _save_panel_cache(panel, cache_path)
    logger.info(f"Panel built: {M} assets x {T} days")
    return panel


def _merge_fundamentals_into_panel_pl(
    panel: dict,
    codes: list[str],
    common_dates: list,
) -> None:
    """Merge balance/income/daily_basic into panel using Polars."""
    M = len(codes)
    T = len(common_dates)

    # --- daily_basic ---
    db_files = sorted(_RAW_DIR.glob("daily_basic*.parquet"))
    if db_files:
        lf = pl.concat([pl.scan_parquet(f) for f in db_files])
        value_cols = [c for c in ["total_mv", "pe", "pb", "dv_ttm"]
                      if c in lf.collect_schema().names()]
        if value_cols:
            dates_int = [int(str(d).replace("-", "")[:8]) for d in common_dates]
            for col in value_cols:
                try:
                    df = (
                        lf
                        .filter(pl.col("ts_code").is_in(codes))
                        .with_columns(pl.col("trade_date").cast(pl.Int64))
                        .filter(pl.col("trade_date").is_in(dates_int))
                        .select(["trade_date", "ts_code", col])
                        .collect()
                        .pivot(on="trade_date", index="ts_code", values=col)
                        .sort("ts_code")
                    )
                    date_str_cols = sorted([c for c in df.columns if c != "ts_code"])
                    arr = df.select(date_str_cols).to_numpy().astype(np.float64)
                    if arr.shape == (M, T):
                        panel[col] = arr
                        logger.info(f"  Merged daily_basic: {col}")
                except Exception as e:
                    logger.warning(f"  Failed daily_basic {col}: {e}")

    # --- quarterly fundamentals (balance, income) ---
    for file_name, date_col, cols in [
        ("batch_balance_all.parquet", "end_date", ["bvps", "roe_pct"]),
        ("batch_income_all.parquet", "end_date", ["basic_eps", "n_income", "revenue"]),
    ]:
        fpath = _RAW_DIR / file_name
        if not fpath.exists():
            continue
        try:
            df = pl.read_parquet(fpath)
            avail_cols = [c for c in cols if c in df.columns]
            if not avail_cols:
                continue
            # Point-in-time: use report_lag_days=45
            _align_quarterly_pl(df, date_col, codes, common_dates, avail_cols, panel, report_lag_days=45)
        except Exception as e:
            logger.warning(f"  Failed {file_name}: {e}")


def _align_quarterly_pl(
    df: pl.DataFrame,
    date_col: str,
    codes: list[str],
    common_dates: list,
    value_cols: list[str],
    panel: dict,
    report_lag_days: int = 45,
) -> None:
    """Align quarterly data to daily dates using Polars asof join (point-in-time)."""
    M = len(codes)
    T = len(common_dates)
    code_to_idx = {c: i for i, c in enumerate(codes)}

    df = df.filter(pl.col("ts_code").is_in(codes))
    df = df.with_columns(
        pl.col(date_col).cast(pl.Utf8).str.slice(0, 8).str.to_datetime("%Y%m%d", strict=False).alias("report_date")
    ).drop_nulls("report_date")

    trade_dates = pl.DataFrame({
        "trade_date": [
            pl.Series([str(d).replace("-", "")[:8] for d in common_dates])
            .str.to_datetime("%Y%m%d", strict=False)
        ][0]
    }).with_columns(
        (pl.col("trade_date") - pl.duration(days=report_lag_days)).alias("cutoff")
    )

    for ts_code in codes:
        sub = (
            df.filter(pl.col("ts_code") == ts_code)
            .select(["report_date"] + value_cols)
            .sort("report_date")
        )
        if sub.is_empty():
            continue
        merged = trade_dates.join_asof(
            sub,
            left_on="cutoff",
            right_on="report_date",
            strategy="backward",
        )
        idx = code_to_idx[ts_code]
        for col in value_cols:
            if col in merged.columns:
                vals = merged[col].to_numpy().astype(np.float64)
                if col not in panel:
                    panel[col] = np.full((M, T), np.nan)
                panel[col][idx, :] = vals

    for col in value_cols:
        if col in panel:
            logger.info(f"  Merged quarterly: {col}")


# ---------------------------------------------------------------------------
# AkShare fallback (uses pandas for API compat, but cleaning still in polars)
# ---------------------------------------------------------------------------

def get_stock_list(index: str = "000300") -> list[str]:
    """Get constituent stock codes for a given index."""
    cache_path = DATA_CACHE_DIR / f"stock_list_{index}.json"
    if cache_path.exists():
        with open(cache_path, "r") as f:
            return json.load(f)

    if not HAS_AKSHARE:
        logger.error("akshare not installed")
        return []

    try:
        df = ak.index_stock_cons_csindex(symbol=index)
        code_col = None
        for col in df.columns:
            if "代码" in str(col) or "code" in str(col).lower() or "成分券代码" in str(col):
                code_col = col
                break
        if code_col is None:
            code_col = df.columns[0]
        codes = df[code_col].astype(str).str.zfill(6).tolist()
        with open(cache_path, "w") as f:
            json.dump(codes, f)
        return codes
    except Exception as e:
        logger.error(f"Failed to fetch stock list: {e}")
        return []


def get_historical_data(
    code: str,
    start: str = DATA_START,
    end: str = DATA_END,
) -> Optional["pd.DataFrame"]:
    """Get historical data for a single stock (AkShare, returns pandas for compat)."""
    if not HAS_PANDAS:
        return None
    cache_path = DATA_CACHE_DIR / f"hist_{code}_{start}_{end}.parquet"
    if cache_path.exists():
        return pd.read_parquet(cache_path)
    if not HAS_AKSHARE:
        return None
    try:
        df = ak.stock_zh_a_hist(
            symbol=code, period="daily",
            start_date=start.replace("-", ""),
            end_date=end.replace("-", ""),
            adjust="qfq",
        )
        if df is None or df.empty:
            return None
        col_map = {}
        for col in df.columns:
            cl = str(col).lower()
            if "日期" in str(col) or "date" in cl: col_map[col] = "date"
            elif "开盘" in str(col) or cl == "open": col_map[col] = "open"
            elif "最高" in str(col) or cl == "high": col_map[col] = "high"
            elif "最低" in str(col) or cl == "low": col_map[col] = "low"
            elif "收盘" in str(col) or cl == "close": col_map[col] = "close"
            elif "成交量" in str(col) or cl == "volume": col_map[col] = "volume"
            elif "成交额" in str(col) or cl == "amount": col_map[col] = "amount"
        df = df.rename(columns=col_map)
        if "date" not in df.columns:
            return None
        df["date"] = pd.to_datetime(df["date"])
        df = df.set_index("date").sort_index()
        for c in ["open", "high", "low", "close", "volume"]:
            if c in df.columns:
                df[c] = pd.to_numeric(df[c], errors="coerce")
        if "amount" not in df.columns:
            df["amount"] = df["close"] * df["volume"]
        df.to_parquet(cache_path)
        return df
    except Exception as e:
        logger.warning(f"Failed {code}: {e}")
        return None


def build_market_panel(
    codes: list[str],
    start: str = DATA_START,
    end: str = DATA_END,
    max_stocks: Optional[int] = None,
) -> dict[str, np.ndarray]:
    """Build panel from per-stock AkShare fetches (fallback path)."""
    cache_key = f"panel_{len(codes)}_{max_stocks}_{start}_{end}"
    cache_path = DATA_CACHE_DIR / cache_key
    cached = _load_panel_cache(cache_path)
    if cached is not None:
        return cached

    if max_stocks:
        codes = codes[:max_stocks]

    all_data = {}
    for code in codes:
        df = get_historical_data(code, start, end)
        if df is not None and len(df) > 100:
            all_data[code] = df

    if not all_data:
        raise ValueError("No valid stock data fetched")

    common_dates = None
    for code, df in all_data.items():
        idx = set(df.index)
        common_dates = idx if common_dates is None else common_dates & idx
    common_dates = sorted(common_dates)
    if len(common_dates) < 60:
        raise ValueError(f"Only {len(common_dates)} common days")

    valid_codes = sorted(all_data.keys())
    M, T = len(valid_codes), len(common_dates)

    panel = {}
    for field in ["open", "high", "low", "close", "volume", "amount"]:
        arr = np.full((M, T), np.nan)
        for i, code in enumerate(valid_codes):
            df = all_data[code]
            if field in df.columns:
                arr[i, :] = df.loc[common_dates, field].values
        panel[field] = arr

    close, volume, amount = panel["close"], panel["volume"], panel["amount"]
    with np.errstate(divide="ignore", invalid="ignore"):
        panel["vwap"] = np.where(volume > 0, amount / volume, close)
    returns = np.full_like(close, np.nan)
    returns[:, 1:] = close[:, 1:] / close[:, :-1] - 1.0
    panel["returns"] = returns
    panel["codes"] = valid_codes
    panel["dates"] = np.array(common_dates)

    _save_panel_cache(panel, cache_path)
    return panel


def calculate_returns(
    panel: dict[str, np.ndarray],
    horizon: int = 1,
) -> np.ndarray:
    """Compute forward returns for IC calculation."""
    close = panel["close"]
    M, T = close.shape
    fwd = np.full((M, T), np.nan)
    if horizon < T:
        fwd[:, :-horizon] = close[:, horizon:] / close[:, :-horizon] - 1.0
    return fwd
