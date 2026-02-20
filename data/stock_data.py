"""Data layer: A-share OHLCV data fetching and panel construction.

Supports two modes:
1. Load from existing parquet file (factor_investing/data/processed/)
2. Fetch live via AkShare (fallback)
"""

import json
import logging
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

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

# Path to existing factor_investing data
_FACTOR_INVESTING_DATA = Path(__file__).resolve().parent.parent.parent / "factor_investing" / "data"
_PROCESSED_DAILY = _FACTOR_INVESTING_DATA / "processed" / "daily_20180101_20241231.parquet"
_RAW_DIR = _FACTOR_INVESTING_DATA / "raw"


def _align_quarterly_to_daily(
    df: pd.DataFrame,
    date_col: str,
    codes: list[str],
    common_dates: list,
    value_cols: list[str],
    report_lag_days: int = 45,
) -> dict[str, np.ndarray]:
    """Align quarterly fundamental data to daily trade dates (point-in-time).

    For each (ts_code, trade_date), use the latest report with
    end_date + report_lag_days <= trade_date (avoids look-ahead).
    Returns dict of arrays (M, T). Uses merge_asof for speed.
    """
    M = len(codes)
    T = len(common_dates)
    result = {col: np.full((M, T), np.nan) for col in value_cols}

    code_to_idx = {c: i for i, c in enumerate(codes)}
    df = df.dropna(subset=[date_col]).copy()
    df[date_col] = pd.to_datetime(df[date_col].astype(str), format="%Y%m%d", errors="coerce")
    df = df.dropna(subset=[date_col])
    df = df[df["ts_code"].isin(codes)].sort_values(date_col)

    trade_df = pd.DataFrame({
        "trade_date": pd.to_datetime([str(d).replace("-", "")[:8] for d in common_dates], format="%Y%m%d"),
    })
    trade_df["_t"] = range(len(trade_df))
    trade_df["trade_date_lag"] = trade_df["trade_date"] - pd.Timedelta(days=report_lag_days)

    trade_sorted = trade_df.sort_values("trade_date_lag").copy()
    for ts_code in codes:
        sub = df[df["ts_code"] == ts_code][[date_col] + value_cols].copy()
        if sub.empty:
            continue
        sub = sub.sort_values(date_col).rename(columns={date_col: "end_date"})
        merged = pd.merge_asof(
            trade_sorted,
            sub,
            left_on="trade_date_lag",
            right_on="end_date",
            direction="backward",
        )
        idx = code_to_idx[ts_code]
        for col in value_cols:
            if col in merged.columns:
                vals = merged[col].values
                vals = np.where(pd.notna(vals) & np.isfinite(vals), vals, np.nan)
                result[col][idx, :] = vals

    return result


def _load_daily_basic_panel(
    codes: list[str],
    common_dates: list,
) -> dict[str, np.ndarray]:
    """Load daily_basic (total_mv, pe, pb, dv_ttm) and align to panel.

    Concatenates all daily_basic_*.parquet files.
    """
    db_files = sorted(_RAW_DIR.glob("daily_basic*.parquet"))
    if not db_files:
        return {}
    dfs = [pd.read_parquet(f) for f in db_files]
    df = pd.concat(dfs, ignore_index=True).drop_duplicates(subset=["ts_code", "trade_date"], keep="last")
    if "trade_date" not in df.columns:
        return {}

    M = len(codes)
    T = len(common_dates)
    value_cols = [c for c in ["total_mv", "pe", "pb", "dv_ttm"] if c in df.columns]
    if not value_cols:
        return {}

    df["trade_date"] = pd.to_numeric(df["trade_date"], errors="coerce")
    df = df[df["ts_code"].isin(codes)]
    dates_int = [int(str(d).replace("-", "")[:8]) for d in common_dates]

    result = {}
    for col in value_cols:
        pt = df.pivot_table(index="trade_date", columns="ts_code", values=col)
        pt = pt.reindex(index=dates_int, columns=codes)
        result[col] = pt.values.T.astype(np.float64)

    return result


def _merge_fundamentals_into_panel(
    panel: dict,
    codes: list[str],
    common_dates: list,
) -> None:
    """Load balance/income/daily_basic from factor_investing and merge into panel."""
    balance_path = _RAW_DIR / "batch_balance_all.parquet"
    income_path = _RAW_DIR / "batch_income_all.parquet"

    if balance_path.exists():
        balance = pd.read_parquet(balance_path)
        bal_cols = [c for c in ["bvps", "roe_pct"] if c in balance.columns]
        if bal_cols:
            bal_arrays = _align_quarterly_to_daily(
                balance, "end_date", codes, common_dates, bal_cols, report_lag_days=45
            )
            for k, arr in bal_arrays.items():
                panel[k] = arr
            logger.info(f"  Merged balance: {list(bal_arrays.keys())}")

    if income_path.exists():
        income = pd.read_parquet(income_path)
        inc_cols = [c for c in ["basic_eps", "n_income", "revenue"] if c in income.columns]
        if inc_cols:
            inc_arrays = _align_quarterly_to_daily(
                income, "end_date", codes, common_dates, inc_cols, report_lag_days=45
            )
            for k, arr in inc_arrays.items():
                panel[k] = arr
            logger.info(f"  Merged income: {list(inc_arrays.keys())}")

    daily_basic = _load_daily_basic_panel(codes, common_dates)
    for k, arr in daily_basic.items():
        panel[k] = arr
    if daily_basic:
        logger.info(f"  Merged daily_basic: {list(daily_basic.keys())}")


def build_panel_from_parquet(
    parquet_path: Path = _PROCESSED_DAILY,
    max_stocks: Optional[int] = 300,
    min_days: int = 500,
    include_fundamentals: bool = False,
) -> dict[str, np.ndarray]:
    """Build market panel directly from the factor_investing processed parquet.

    Args:
        parquet_path: Path to the daily parquet file.
        max_stocks: Maximum number of stocks to include.
        min_days: Minimum trading days a stock must have to be included.
        include_fundamentals: If True, merge balance/income/daily_basic from factor_investing/raw.

    Returns:
        Panel dict with keys: open, high, low, close, volume, amount, vwap, returns, codes, dates.
        When include_fundamentals=True, adds: bvps, roe_pct, basic_eps, n_income, revenue,
        total_mv, pe, pb, dv_ttm (when available).
    """
    fund_suffix = "_fund" if include_fundamentals else ""
    cache_key = f"parquet_panel_{max_stocks}_{min_days}{fund_suffix}"
    cache_path = DATA_CACHE_DIR / cache_key
    cached = _load_panel_cache(cache_path)
    if cached is not None:
        logger.info(f"Loaded cached panel (from local data) {cache_path}")
        return cached

    logger.info(f"Using local data: {parquet_path}")
    if not parquet_path.exists():
        raise FileNotFoundError(f"Local data not found: {parquet_path}. Ensure factor_investing/data/processed/ exists.")
    df = pd.read_parquet(parquet_path)
    logger.info(f"Raw data: {len(df):,} rows, {df['ts_code'].nunique()} stocks")

    # Count days per stock, keep only those with enough data
    day_counts = df.groupby("ts_code")["trade_date"].count()
    valid_stocks = day_counts[day_counts >= min_days].index.tolist()
    logger.info(f"Stocks with >= {min_days} days: {len(valid_stocks)}")

    if max_stocks and len(valid_stocks) > max_stocks:
        # Prefer stocks with most data
        valid_stocks = day_counts.loc[valid_stocks].nlargest(max_stocks).index.tolist()

    df = df[df["ts_code"].isin(valid_stocks)].copy()

    # Get common trading dates across all selected stocks
    dates_per_stock = df.groupby("ts_code")["trade_date"].apply(set)
    common_dates = set.intersection(*dates_per_stock.values)
    common_dates = sorted(common_dates)
    logger.info(f"Common trading dates: {len(common_dates)}")

    if len(common_dates) < 60:
        raise ValueError(f"Only {len(common_dates)} common dates, need >= 60")

    codes = sorted(valid_stocks)
    M = len(codes)
    T = len(common_dates)
    logger.info(f"Building panel: {M} assets x {T} days")

    # Pivot each field
    df_filtered = df[df["trade_date"].isin(common_dates)]

    panel = {}
    field_map = {
        "open": "open",
        "high": "high",
        "low": "low",
        "close": "close",
        "volume": "vol",
        "amount": "amount",
    }

    for panel_name, col_name in field_map.items():
        pivot = df_filtered.pivot(index="trade_date", columns="ts_code", values=col_name)
        pivot = pivot.sort_index()
        pivot = pivot[codes]  # Ensure consistent column order
        panel[panel_name] = pivot.values.T.astype(np.float64)  # (M, T)

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

    # Strip the .SZ/.SH suffix for cleaner display
    clean_codes = [c.split(".")[0] if "." in c else c for c in codes]
    panel["codes"] = clean_codes
    panel["dates"] = np.array(common_dates)

    if include_fundamentals and _RAW_DIR.exists():
        logger.info("Merging fundamental data from factor_investing/raw ...")
        _merge_fundamentals_into_panel(panel, codes, common_dates)

    base_fields = list(field_map.keys()) + ["vwap", "returns"]
    extra = [k for k in panel.keys() if k not in ("codes", "dates") and k not in base_fields]
    logger.info(f"Panel built: {M} assets x {T} days, fields: {base_fields}{extra and [' + ' + str(extra) or '']}")

    _save_panel_cache(panel, cache_path)
    return panel


def get_stock_list(index: str = "000300") -> list[str]:
    """Get constituent stock codes for a given index.

    Args:
        index: Index code. '000300' for CSI 300, '000905' for CSI 500.

    Returns:
        List of 6-digit stock codes (e.g. ['600519', '000858', ...]).
    """
    cache_path = DATA_CACHE_DIR / f"stock_list_{index}.json"
    if cache_path.exists():
        with open(cache_path, "r") as f:
            return json.load(f)

    if not HAS_AKSHARE:
        logger.error("akshare not installed. Install with: pip install akshare")
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

        logger.info(f"Fetched {len(codes)} stocks for index {index}")
        return codes

    except Exception as e:
        logger.error(f"Failed to fetch stock list for {index}: {e}")
        return []


def get_historical_data(
    code: str,
    start: str = DATA_START,
    end: str = DATA_END,
) -> Optional[pd.DataFrame]:
    """Get historical daily OHLCV data for a single stock.

    Args:
        code: 6-digit stock code.
        start: Start date 'YYYY-MM-DD'.
        end: End date 'YYYY-MM-DD'.

    Returns:
        DataFrame with columns [date, open, high, low, close, volume, amount]
        indexed by date, or None on failure.
    """
    cache_path = DATA_CACHE_DIR / f"hist_{code}_{start}_{end}.parquet"
    if cache_path.exists():
        return pd.read_parquet(cache_path)

    if not HAS_AKSHARE:
        logger.error("akshare not installed. Install with: pip install akshare")
        return None

    try:
        start_fmt = start.replace("-", "")
        end_fmt = end.replace("-", "")

        df = ak.stock_zh_a_hist(
            symbol=code,
            period="daily",
            start_date=start_fmt,
            end_date=end_fmt,
            adjust="qfq",
        )

        if df is None or df.empty:
            logger.warning(f"No data for {code}")
            return None

        col_map = {}
        for col in df.columns:
            col_lower = str(col).lower()
            if "日期" in str(col) or "date" in col_lower:
                col_map[col] = "date"
            elif "开盘" in str(col) or col_lower == "open":
                col_map[col] = "open"
            elif "最高" in str(col) or col_lower == "high":
                col_map[col] = "high"
            elif "最低" in str(col) or col_lower == "low":
                col_map[col] = "low"
            elif "收盘" in str(col) or col_lower == "close":
                col_map[col] = "close"
            elif "成交量" in str(col) or col_lower == "volume":
                col_map[col] = "volume"
            elif "成交额" in str(col) or col_lower == "amount":
                col_map[col] = "amount"

        df = df.rename(columns=col_map)

        required = ["date", "open", "high", "low", "close", "volume"]
        if not all(c in df.columns for c in required):
            logger.warning(f"Missing columns for {code}: {df.columns.tolist()}")
            return None

        df["date"] = pd.to_datetime(df["date"])
        df = df.set_index("date").sort_index()

        for col in ["open", "high", "low", "close", "volume"]:
            df[col] = pd.to_numeric(df[col], errors="coerce")

        if "amount" not in df.columns:
            df["amount"] = df["close"] * df["volume"]
        else:
            df["amount"] = pd.to_numeric(df["amount"], errors="coerce")

        df.to_parquet(cache_path)
        return df

    except Exception as e:
        logger.warning(f"Failed to fetch data for {code}: {e}")
        return None


def build_market_panel(
    codes: list[str],
    start: str = DATA_START,
    end: str = DATA_END,
    max_stocks: Optional[int] = None,
) -> dict[str, np.ndarray]:
    """Build the market data panel for factor mining.

    Args:
        codes: List of stock codes.
        start: Start date.
        end: End date.
        max_stocks: Cap on number of stocks (for fast screening).

    Returns:
        Dict mapping feature names to 2D arrays of shape (M, T) where
        M = number of assets and T = number of trading days.
        Also includes 'codes' (list) and 'dates' (array).
    """
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

    logger.info(f"Fetched data for {len(all_data)} / {len(codes)} stocks")

    common_dates = None
    for code, df in all_data.items():
        idx = set(df.index)
        if common_dates is None:
            common_dates = idx
        else:
            common_dates = common_dates & idx

    common_dates = sorted(common_dates)
    if len(common_dates) < 60:
        raise ValueError(f"Only {len(common_dates)} common trading days, need >= 60")

    valid_codes = sorted(all_data.keys())
    M = len(valid_codes)
    T = len(common_dates)
    logger.info(f"Panel: {M} assets x {T} trading days")

    panel = {}
    base_fields = ["open", "high", "low", "close", "volume", "amount"]
    for field in base_fields:
        arr = np.full((M, T), np.nan)
        for i, code in enumerate(valid_codes):
            df = all_data[code]
            vals = df.loc[common_dates, field].values if field in df.columns else np.full(T, np.nan)
            arr[i, :] = vals
        panel[field] = arr

    close = panel["close"]
    volume = panel["volume"]
    amount = panel["amount"]

    with np.errstate(divide="ignore", invalid="ignore"):
        vwap = np.where(volume > 0, amount / volume, close)
    panel["vwap"] = vwap

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
    """Compute forward returns for IC calculation.

    Args:
        panel: Market data panel from build_market_panel.
        horizon: Number of days ahead for return calculation.

    Returns:
        2D array (M, T) of forward returns. Last `horizon` columns are NaN.
    """
    close = panel["close"]
    M, T = close.shape
    fwd = np.full((M, T), np.nan)
    if horizon < T:
        fwd[:, :-horizon] = close[:, horizon:] / close[:, :-horizon] - 1.0
    return fwd
