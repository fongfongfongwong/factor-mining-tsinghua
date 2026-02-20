---
description: All data cleaning and ETL must use Polars (not Pandas)
globs: ["**/*.py"]
---

# Data Cleaning: Use Polars Only

All data cleaning, ETL, and data transformation code in this project MUST use **Polars** (`import polars as pl`), NOT Pandas.

## Rules

1. **Read parquet**: Use `pl.scan_parquet()` (lazy) or `pl.read_parquet()` (eager), never `pd.read_parquet()`
2. **GroupBy / Filter / Join**: Use Polars expressions (`pl.col()`, `.group_by()`, `.filter()`, `.join()`)
3. **Pivot**: Use `df.pivot()` in Polars, not `pd.pivot_table()`
4. **To numpy**: Use `.to_numpy()` on Polars DataFrames/Series
5. **Temporal joins**: Use `df.join_asof()` in Polars for point-in-time alignment

## Exceptions

- AkShare API returns Pandas DataFrames — these are acceptable at the API boundary only
- Convert to Polars immediately after receiving: `pl.from_pandas(df)`
- Legacy code in `factor_investing/` may still use Pandas — do not modify that repo

## Why

- Polars is 5-10x faster than Pandas for our workloads (7.4M row parquet scans, group-by pivots)
- Polars lazy execution enables query optimization
- Zero-copy to numpy for downstream factor computation
- M2 Max 96GB unified memory is best utilized with Polars' memory-efficient processing
