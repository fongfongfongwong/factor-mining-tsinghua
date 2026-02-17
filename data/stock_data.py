"""Data layer: A-share OHLCV data fetching and panel construction.

Supports two modes:
1. Load from existing parquet file (factor_investing/data/processed/)
2. Fetch live via AkShare (fallback)
"""

import pickle
import logging
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

from config import DATA_CACHE_DIR, DATA_START, DATA_END, DATA_FEATURES

logger = logging.getLogger(__name__)

# Path to existing factor_investing data
_FACTOR_INVESTING_DATA = Path(__file__).resolve().parent.parent.parent / "factor_investing" / "data"
_PROCESSED_DAILY = _FACTOR_INVESTING_DATA / "processed" / "daily_20180101_20241231.parquet"


def build_panel_from_parquet(
    parquet_path: Path = _PROCESSED_DAILY,
    max_stocks: Optional[int] = 300,
    min_days: int = 500,
) -> dict[str, np.ndarray]:
    """Build market panel directly from the factor_investing processed parquet.

    Args:
        parquet_path: Path to the daily parquet file.
        max_stocks: Maximum number of stocks to include.
        min_days: Minimum trading days a stock must have to be included.

    Returns:
        Panel dict with keys: open, high, low, close, volume, amount, vwap, returns, codes, dates
    """
    cache_key = f"parquet_panel_{max_stocks}_{min_days}"
    cache_path = DATA_CACHE_DIR / f"{cache_key}.pkl"
    if cache_path.exists():
        logger.info(f"Loading cached panel from {cache_path}")
        with open(cache_path, "rb") as f:
            return pickle.load(f)

    logger.info(f"Loading parquet from {parquet_path} ...")
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

    logger.info(f"Panel built: {M} assets x {T} days, fields: {list(field_map.keys()) + ['vwap', 'returns']}")

    with open(cache_path, "wb") as f:
        pickle.dump(panel, f)

    return panel


def get_stock_list(index: str = "000300") -> list[str]:
    """Get constituent stock codes for a given index.

    Args:
        index: Index code. '000300' for CSI 300, '000905' for CSI 500.

    Returns:
        List of 6-digit stock codes (e.g. ['600519', '000858', ...]).
    """
    cache_path = DATA_CACHE_DIR / f"stock_list_{index}.pkl"
    if cache_path.exists():
        with open(cache_path, "rb") as f:
            return pickle.load(f)

    try:
        if index == "000300":
            df = ak.index_stock_cons_csindex(symbol="000300")
        elif index == "000905":
            df = ak.index_stock_cons_csindex(symbol="000905")
        else:
            df = ak.index_stock_cons_csindex(symbol=index)

        code_col = None
        for col in df.columns:
            if "代码" in str(col) or "code" in str(col).lower() or "成分券代码" in str(col):
                code_col = col
                break
        if code_col is None:
            code_col = df.columns[0]

        codes = df[code_col].astype(str).str.zfill(6).tolist()

        with open(cache_path, "wb") as f:
            pickle.dump(codes, f)

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
    cache_path = DATA_CACHE_DIR / f"hist_{code}_{start}_{end}.pkl"
    if cache_path.exists():
        with open(cache_path, "rb") as f:
            return pickle.load(f)

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

        with open(cache_path, "wb") as f:
            pickle.dump(df, f)

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
    cache_path = DATA_CACHE_DIR / f"{cache_key}.pkl"
    if cache_path.exists():
        with open(cache_path, "rb") as f:
            return pickle.load(f)

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

    with open(cache_path, "wb") as f:
        pickle.dump(panel, f)

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
