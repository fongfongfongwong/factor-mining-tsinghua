# FactorMiner: Project Plan, Design & Documentation

> LLM-Powered Alpha Factor Mining System for Chinese A-Share Market
>
> Inspired by: *FactorMiner: AI-Driven Alpha Factor Discovery* (Tsinghua University, arXiv:2602.14670, 2026)

---

## Table of Contents

1. [Project Overview](#1-project-overview)
2. [Architecture & Design](#2-architecture--design)
3. [Core Algorithm: Ralph Loop](#3-core-algorithm-ralph-loop)
4. [Module Reference](#4-module-reference)
5. [Data Pipeline](#5-data-pipeline)
6. [Expression Engine & Operator Library](#6-expression-engine--operator-library)
7. [Factor Library & Admission Pipeline](#7-factor-library--admission-pipeline)
8. [Experience Memory System](#8-experience-memory-system)
9. [Backtest Engine](#9-backtest-engine)
10. [MASTER Model Evaluator](#10-master-model-evaluator)
11. [Web Dashboard & API](#11-web-dashboard--api)
12. [Deployment](#12-deployment)
13. [Configuration Reference](#13-configuration-reference)
14. [Development & Testing](#14-development--testing)
15. [Project Roadmap](#15-project-roadmap)

---

## 1. Project Overview

### 1.1 Problem Statement

Alpha factor discovery in quantitative investing has traditionally relied on human intuition and laborious manual testing. With thousands of possible operator combinations and data fields, the search space is enormous. Manual approaches suffer from limited exploration breadth, cognitive biases, and inability to systematically learn from past failures.

### 1.2 Solution

FactorMiner automates the entire factor mining lifecycle using an LLM-driven self-evolving agent. The system implements the **Ralph Loop** (Retrieve-Generate-Evaluate-Distill) to continuously discover, validate, and curate an orthogonal factor library for A-share stock trading.

### 1.3 Key Capabilities

| Capability | Description |
|---|---|
| **Self-Evolving Mining** | Ralph Loop drives autonomous factor discovery across multiple sessions |
| **40+ Operators** | Time-series, cross-sectional, and element-wise operators in pure NumPy |
| **Multi-Stage Validation** | Fast IC screening -> full IC validation -> correlation check -> replacement |
| **Orthogonal Library** | Maintains factor diversity via correlation-based deduplication |
| **Experience Memory** | Three-part memory (state, experience, insights) persists across sessions |
| **MASTER Evaluator** | Walk-forward IC evaluation for the Market-Guided Stock Transformer model |
| **Backtest Engine** | Long-short quintile portfolio backtesting with Sharpe, drawdown, turnover |
| **Web Dashboard** | Real-time mining control, factor browser, backtest charts via WebSocket |
| **A-Share Native** | AkShare data, Chinese prompts, domestic market conventions |

### 1.4 Project Statistics

| Metric | Value |
|---|---|
| Total Python files | 19 |
| Total Python lines | ~4,500+ |
| Operator count | 40+ |
| ML model | MASTER (AAAI 2024) |
| Data source | AkShare / Tushare parquet |
| LLM backend | Kimi 2.5 (Moonshot AI) |
| Web framework | FastAPI + WebSocket |
| Frontend | TailwindCSS + ECharts |

### 1.5 Directory Structure

```
factor-miner/
├── api/                            # REST API & WebSocket server
│   ├── main.py                     # FastAPI application (249 lines)
│   └── models.py                   # Pydantic schemas (65 lines)
│
├── backtest/                       # Backtesting engine
│   ├── __init__.py                 # Module exports
│   ├── engine.py                   # Portfolio backtest logic (198 lines)
│   └── metrics.py                  # Performance metrics (153 lines)
│
├── data/                           # Data loading & preprocessing
│   ├── __init__.py                 # Module exports
│   └── stock_data.py              # AkShare integration, panel builder (353 lines)
│
├── evaluator/                      # Model evaluation
│   ├── __init__.py                 # Module marker
│   └── master_evaluator.py        # MASTER model IC evaluator (798 lines)
│
├── factor_mining/                  # Core mining engine
│   ├── __init__.py                 # Module exports
│   ├── miner.py                   # Ralph Loop orchestrator (451 lines)
│   ├── expression_engine.py       # Expression parser & evaluator (394 lines)
│   ├── operators.py               # 40+ operator implementations (369 lines)
│   ├── factor_library.py          # Factor storage & admission (259 lines)
│   └── experience_memory.py       # Three-part memory system (241 lines)
│
├── storage/                        # Persistent state
│   ├── factor_library.json        # Discovered factors
│   └── experience_memory.json     # Mining memory
│
├── templates/                      # Web UI
│   └── index.html                 # Single-page dashboard (625 lines)
│
├── config.py                       # Central configuration (41 lines)
├── run.py                          # CLI entry point (405 lines)
├── evaluate_master.py              # MASTER evaluator CLI (139 lines)
├── test_pipeline.py                # End-to-end pipeline test (271 lines)
├── requirements.txt                # Python dependencies
├── Dockerfile                      # Container definition
├── docker-compose.yml              # Docker orchestration
├── .env / .env.example             # Environment variables
└── README.md                       # Quick-start guide
```

---

## 2. Architecture & Design

### 2.1 System Architecture

```
┌─────────────────────────────────────────────────────────────────────────┐
│                           FactorMiner System                           │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                         │
│  ┌──────────────┐     ┌──────────────┐     ┌──────────────────────┐    │
│  │  LLM Engine  │     │  Ralph Loop  │     │  Web Dashboard       │    │
│  │  (Kimi 2.5)  │◄───►│  Orchestrator│◄───►│  (FastAPI+WebSocket) │    │
│  └──────────────┘     └──────┬───────┘     └──────────────────────┘    │
│                              │                                          │
│         ┌────────────────────┼────────────────────┐                    │
│         ▼                    ▼                    ▼                    │
│  ┌──────────────┐     ┌──────────────┐     ┌──────────────┐          │
│  │  Expression   │     │   Factor     │     │  Experience   │          │
│  │  Engine       │     │   Library    │     │  Memory       │          │
│  │  (Parser +    │     │  (Admission  │     │  (3-Part      │          │
│  │   Evaluator)  │     │   Pipeline)  │     │   System)     │          │
│  └──────┬───────┘     └──────┬───────┘     └──────┬───────┘          │
│         │                    │                    │                    │
│         ▼                    ▼                    ▼                    │
│  ┌──────────────┐     ┌──────────────┐     ┌──────────────┐          │
│  │  Operator     │     │  IC / ICIR   │     │  JSON        │          │
│  │  Library      │     │  Metrics     │     │  Persistence │          │
│  │  (40+ ops)    │     │              │     │              │          │
│  └──────┬───────┘     └──────────────┘     └──────────────┘          │
│         │                                                              │
│         ▼                                                              │
│  ┌──────────────────────────────────────────────────────────────┐      │
│  │                     Data Layer                                │      │
│  │  ┌────────────┐  ┌──────────────┐  ┌──────────────────┐     │      │
│  │  │  AkShare   │  │  Parquet     │  │  Pickle Cache    │     │      │
│  │  │  (Live)    │  │  (Processed) │  │  (Panel Cache)   │     │      │
│  │  └────────────┘  └──────────────┘  └──────────────────┘     │      │
│  └──────────────────────────────────────────────────────────────┘      │
│                                                                         │
│  ┌──────────────────────────────────────────────────────────────┐      │
│  │                   Evaluation Layer                            │      │
│  │  ┌────────────────────┐  ┌──────────────────────────────┐   │      │
│  │  │  Backtest Engine   │  │  MASTER Model Evaluator      │   │      │
│  │  │  (Quintile L/S)    │  │  (Walk-Forward IC Eval)      │   │      │
│  │  └────────────────────┘  └──────────────────────────────┘   │      │
│  └──────────────────────────────────────────────────────────────┘      │
└─────────────────────────────────────────────────────────────────────────┘
```

### 2.2 Design Principles

1. **Separation of Concerns**: Each module has a single responsibility -- parsing, evaluation, storage, mining, API.
2. **Pure NumPy Core**: All operators and metrics run on plain NumPy arrays `(M, T)` without deep learning dependencies.
3. **Panel Data Convention**: All data flows use `(M, T)` 2D arrays where `M` = assets and `T` = time steps.
4. **Stateless Expression Engine**: The parser/evaluator is a pure function from `(expression, data_panel)` to `signal_array`.
5. **Persistent Memory**: Factor library and experience memory survive across sessions via JSON serialization.
6. **Graceful Degradation**: MASTER evaluator falls back to Ridge regression if PyTorch is unavailable.

### 2.3 Data Flow

```
   AkShare / Parquet
         │
         ▼
  ┌──────────────┐
  │ build_panel   │──► Panel Dict: {close: (M,T), open: (M,T), ...}
  └──────────────┘
         │
    ┌────┴────┐
    ▼         ▼
 Fast Panel  Full Panel
 (50 stocks) (300 stocks)
    │         │
    ▼         ▼
 Expression  Expression
 Engine      Engine
    │         │
    ▼         ▼
 Fast IC     Full IC
 Screening   Validation
    │         │
    └────┬────┘
         ▼
  Correlation Check
         │
         ▼
  Factor Library
  (Orthogonal Set)
         │
         ▼
  Backtest Engine
  (Quintile L/S)
```

---

## 3. Core Algorithm: Ralph Loop

The system implements **Algorithm 1** from the FactorMiner paper, named the **Ralph Loop** -- a four-phase cycle that runs continuously:

### 3.1 Phase 1: Retrieve

Assemble context-dependent prompt material from the experience memory:

- **Mining State (S)**: Library size, current round, recent admission rate, domain saturation levels.
- **Structural Experience (P)**: Up to 10 recommended directions (successful patterns) and 10 forbidden directions (high-correlation/low-IC regions).
- **Recent Rejections**: Last 5 rejected candidates with reasons.

This context is injected into the LLM prompt to guide exploration toward productive regions and away from known dead-ends.

### 3.2 Phase 2: Generate

The LLM (Kimi 2.5) receives a structured Chinese prompt containing:

- Available data fields: `open, high, low, close, volume, amount, vwap, returns`
- Available operators: 18 time-series, 4 cross-sectional, 13 element-wise
- Memory context from Phase 1
- Generation constraints (syntax rules, diversity requirements, window ranges 5-60)

The LLM returns a JSON array of `{expression, logic}` pairs. Each expression is validated against the grammar before proceeding.

### 3.3 Phase 3: Evaluate

Multi-stage pipeline with progressive filtering:

```
Candidate Expression
       │
       ▼
[Stage 1] Syntax Validation
       │ validate_expression() checks grammar & operator names
       ▼
[Stage 2] Fast IC Screen (50 stocks)
       │ compute_ic() on fast panel
       │ Reject if |IC_mean| < IC_THRESHOLD (0.02)
       │ Reject if NaN ratio > 50%
       ▼
[Stage 3] Full IC Validation (300 stocks)
       │ compute_ic() on full panel
       │ Reject if |IC_mean| < IC_THRESHOLD
       ▼
[Stage 4] Correlation Check
       │ compute_correlation() vs all existing factors
       │ If max_corr < CORR_THRESHOLD (0.7): ADMIT
       ▼
[Stage 5] Replacement Check (if correlated)
       │ If new IC > existing IC * 1.1: REPLACE
       │ Otherwise: REJECT
       ▼
 ADMITTED to Factor Library
```

### 3.4 Phase 4: Distill

Update the experience memory based on batch results:

1. **Formation** (every round):
   - Add admitted factors to recommended directions
   - Add high-correlation rejections to forbidden directions
   - Update domain saturation counters
   - Track recent rejections

2. **Evolution** (every 5 rounds):
   - Call LLM with batch results to distill 2-3 strategic insights
   - Integrate insights into memory
   - Decay stale entries (> 7 days old)
   - Trim lists to maximum sizes

### 3.5 Convergence Criteria

The mining loop terminates when:
- Library size reaches `TARGET_LIBRARY_SIZE` (default: 100), or
- `MAX_ROUNDS` (default: 50) is exhausted, or
- User manually stops via API/CLI

---

## 4. Module Reference

### 4.1 `config.py` -- Central Configuration

Single source of truth for all system parameters:

| Parameter | Default | Description |
|---|---|---|
| `IC_THRESHOLD` | 0.02 | Minimum absolute IC for admission |
| `CORR_THRESHOLD` | 0.7 | Maximum correlation budget |
| `BATCH_SIZE` | 10 | Candidates per mining round |
| `MAX_ROUNDS` | 50 | Maximum mining iterations |
| `TARGET_LIBRARY_SIZE` | 100 | Target factor count |
| `FAST_SCREEN_ASSETS` | 50 | Stocks for fast IC screen |
| `FULL_UNIVERSE_ASSETS` | 300 | Stocks for full validation |
| `DATA_START` | "2020-01-01" | Data range start |
| `DATA_END` | "2025-12-31" | Data range end |
| `LLM_MODEL` | "kimi-k2.5" | LLM model identifier |

### 4.2 `run.py` -- CLI Entry Point

Subcommands:

```bash
python run.py mine [--rounds N] [--batch N] [--max-stocks N]
python run.py list
python run.py backtest [--max-stocks N]
python run.py server [--port N]
```

When run without arguments, defaults to `mine` with 3 rounds, batch=5 as a quick test.

### 4.3 `test_pipeline.py` -- Pipeline Validator

Tests the full pipeline without requiring an LLM API key by using 16 predefined factor expressions covering:

- Momentum (3 factors)
- Volume-Price (3 factors)
- Volatility (3 factors)
- Price Patterns (3 factors)
- VWAP-Based (2 factors)
- Higher-Order (2 factors)

Validates: data loading, operator evaluation, IC computation, correlation checking, factor admission, experience memory, and backtesting.

---

## 5. Data Pipeline

### 5.1 Data Sources

| Source | Priority | Description |
|---|---|---|
| Parquet file | Primary | Pre-processed daily data from `factor_investing/data/processed/` |
| AkShare | Fallback | Free A-share OHLCV via `akshare` library |

### 5.2 Panel Format

The core data structure is a **Panel Dict** with the following keys:

| Key | Shape | Type | Description |
|---|---|---|---|
| `open` | (M, T) | float64 | Opening prices |
| `high` | (M, T) | float64 | High prices |
| `low` | (M, T) | float64 | Low prices |
| `close` | (M, T) | float64 | Closing prices |
| `volume` | (M, T) | float64 | Trading volume |
| `amount` | (M, T) | float64 | Trading amount (turnover) |
| `vwap` | (M, T) | float64 | Volume-weighted average price |
| `returns` | (M, T) | float64 | Daily returns (close/close[-1] - 1) |
| `codes` | (M,) | list[str] | Stock codes |
| `dates` | (T,) | array | Trading dates |

Where `M` = number of assets and `T` = number of trading days.

### 5.3 Forward Returns

Computed by `calculate_returns(panel, horizon=1)`:

```
fwd_returns[i, t] = close[i, t + horizon] / close[i, t] - 1
```

The last `horizon` columns are NaN (no future data available).

### 5.4 Caching

All panels are cached as pickle files in `storage/data_cache/` with keys like `parquet_panel_100_500.pkl` (100 stocks, 500 min days). Cache hits skip all data processing.

---

## 6. Expression Engine & Operator Library

### 6.1 Expression Syntax

Factor expressions follow a formulaic alpha syntax:

```
cs_rank(ts_corr(close, volume, 10))
neg(cs_rank(div(ts_delta(close, 5), ts_lag(close, 5))))
cs_rank(mul(ts_corr(close, volume, 10), ts_std(returns, 10)))
```

### 6.2 Grammar (BNF)

```
expr      ::= term (('+' | '-') term)*
term      ::= unary (('*' | '/') unary)*
unary     ::= '-' unary | primary
primary   ::= NUMBER | func_call | IDENT | '(' expr ')'
func_call ::= IDENT '(' arg_list ')'
arg_list  ::= expr (',' expr)*
```

### 6.3 Parser Architecture

The expression engine uses a classic **recursive descent parser**:

1. **Tokenizer**: Regex-based lexer produces tokens (NUMBER, IDENT, LPAREN, RPAREN, COMMA, operators).
2. **Parser**: Recursive descent builds an AST from token stream.
3. **Evaluator**: Tree-walk evaluator maps AST nodes to NumPy operations via the operator registry.

AST Node Types:

| Node | Description |
|---|---|
| `NumberNode` | Numeric literal |
| `FieldNode` | Data field reference (close, volume, etc.) |
| `FuncCallNode` | Operator function call with arguments |
| `BinOpNode` | Binary arithmetic (+, -, *, /) |
| `UnaryMinusNode` | Negation |

Recursion is capped at depth 20 (`MAX_RECURSION_DEPTH`) to prevent stack overflow from adversarial LLM outputs.

### 6.4 Operator Reference

#### Time-Series Operators (`ts_*`) -- per asset, rolling along time axis

| Operator | Signature | Description |
|---|---|---|
| `ts_rank` | (x, d) | Percentile rank of current value within past d periods |
| `ts_std` | (x, d) | Rolling standard deviation |
| `ts_mean` | (x, d) | Rolling mean |
| `ts_sum` | (x, d) | Rolling sum |
| `ts_min` | (x, d) | Rolling minimum |
| `ts_max` | (x, d) | Rolling maximum |
| `ts_argmin` | (x, d) | Position of minimum in window |
| `ts_argmax` | (x, d) | Position of maximum in window |
| `ts_delta` | (x, d) | x[t] - x[t-d] |
| `ts_lag` | (x, d) | Lagged value x[t-d] |
| `ts_skew` | (x, d) | Rolling skewness |
| `ts_kurt` | (x, d) | Rolling kurtosis |
| `ts_decay_linear` | (x, d) | Linearly decaying weighted mean |
| `ts_product` | (x, d) | Rolling product |
| `ts_corr` | (x, y, d) | Rolling Pearson correlation |
| `ts_cov` | (x, y, d) | Rolling covariance |
| `ts_rsquare` | (x, y, d) | Rolling R-squared |
| `ts_regression_residual` | (x, y, d) | Rolling OLS residual |

#### Cross-Sectional Operators (`cs_*`) -- across assets at each timestamp

| Operator | Signature | Description |
|---|---|---|
| `cs_rank` | (x) | Percentile rank across assets (0-1) |
| `cs_zscore` | (x) | Z-score normalization |
| `cs_scale` | (x) | Scale so sum(abs) = 1 |
| `cs_demean` | (x) | Subtract cross-sectional mean |

#### Element-Wise Operators

| Operator | Signature | Description |
|---|---|---|
| `abs` | (x) | Absolute value |
| `log` | (x) | Natural log (of abs(x)) |
| `sign` | (x) | Sign function |
| `sqrt` | (x) | Square root (of abs(x)) |
| `neg` | (x) | Negation |
| `inv` | (x) | Reciprocal (1/x) |
| `add` | (x, y) | Addition |
| `sub` | (x, y) | Subtraction |
| `mul` | (x, y) | Multiplication |
| `div` | (x, y) | Division (epsilon-safe) |
| `power` | (x, n) | Power function |
| `max` | (x, y) | Element-wise maximum |
| `min` | (x, y) | Element-wise minimum |

### 6.5 Arity System

The `OPERATOR_REGISTRY` maps each operator name to a `(function, arity)` tuple. Arity types control argument parsing:

| Arity | Arguments | Example |
|---|---|---|
| `ts` | (array, window_int) | `ts_mean(close, 20)` |
| `ts2` | (array, array, window_int) | `ts_corr(close, volume, 10)` |
| `cs` | (array) | `cs_rank(close)` |
| `unary` | (array) | `abs(returns)` |
| `binary` | (array, array) | `add(close, open)` |
| `power` | (array, number) | `power(close, 0.5)` |

---

## 7. Factor Library & Admission Pipeline

### 7.1 Factor Record

Each factor in the library stores:

| Field | Type | Description |
|---|---|---|
| `factor_id` | str | Unique identifier (e.g. "F623854") |
| `expression` | str | Factor expression string |
| `ic_mean` | float | Mean Spearman rank IC |
| `ic_std` | float | IC standard deviation |
| `icir` | float | IC Information Ratio (IC_mean / IC_std) |
| `max_correlation` | float | Maximum correlation with existing factors |
| `turnover` | float | Average signal turnover |
| `logic_description` | str | Chinese description of financial logic |
| `admitted_at` | float | Unix timestamp of admission |
| `mining_round` | int | Round number when discovered |

### 7.2 IC Computation

**Spearman Rank IC** (Equation 2 from paper):

```
IC(t) = SpearmanCorr(signal[:, t], forward_returns[:, t])
```

Computed per timestamp across the cross-section of stocks. Requires at least 10 valid (non-NaN) observations.

**ICIR** (IC Information Ratio):

```
ICIR = mean(IC) / std(IC)
```

### 7.3 Admission Decision Tree

```
check_admission(candidate_signal, candidate_ic_series):

    1. IF valid IC observations < 10: REJECT ("Insufficient IC data")
    2. IF |IC_mean| < IC_THRESHOLD: REJECT ("IC too low")
    3. IF library is empty: ADMIT ("First factor")
    4. Compute max_corr vs all existing factors
    5. IF max_corr < CORR_THRESHOLD:
         ADMIT ("Low correlation, good diversity")
    6. ELSE IF |new_IC| > |existing_IC| * 1.1:
         REPLACE the correlated factor ("Strictly superior")
    7. ELSE: REJECT ("Too correlated and not stronger")
```

### 7.4 Correlation Metric

Time-average cross-sectional Spearman correlation (Equation 3):

```
Corr(A, B) = mean_t[ SpearmanCorr(A[:, t], B[:, t]) ]
```

### 7.5 Current Library

The system has discovered 8 factors across 3 mining rounds:

| # | IC | ICIR | Expression | Logic |
|---|---|---|---|---|
| 1 | -0.060 | -0.298 | `cs_rank(ts_mean(div(sub(high,low),close),15))` | Intraday amplitude ratio |
| 2 | -0.051 | -0.307 | `cs_rank(ts_cov(close,volume,20))` | Price-volume covariance |
| 3 | +0.041 | +0.210 | `cs_rank(neg(ts_mean(returns,5)))` | Short-term reversal |
| 4 | -0.040 | -0.279 | `cs_rank(ts_corr(close,volume,10))` | Price-volume correlation |
| 5 | -0.039 | -0.215 | `cs_rank(div(ts_delta(close,10),ts_std(close,20)))` | Vol-adjusted momentum |
| 6 | -0.030 | -0.180 | `cs_rank(ts_mean(div(sub(close,open),sub(high,low)),20))` | Intraday momentum |
| 7 | -0.028 | -0.207 | `cs_rank(ts_corr(returns,ts_delta(volume,5),20))` | Return-flow correlation |
| 8 | -0.026 | -0.172 | `cs_rank(ts_delta(ts_std(returns,20),5))` | Volatility clustering |

---

## 8. Experience Memory System

### 8.1 Three-Part Structure

The memory system follows Section 3.3 of the FactorMiner paper:

```
┌─────────────────────────────────────────┐
│            Experience Memory             │
├─────────────────────────────────────────┤
│                                          │
│  (S) Mining State                       │
│  ├─ library_size: int                   │
│  ├─ total_candidates: int               │
│  ├─ total_admitted: int                 │
│  ├─ recent_admission_rate: float        │
│  ├─ domain_saturation: {domain: pct}    │
│  └─ current_round: int                  │
│                                          │
│  (P) Structural Experience              │
│  ├─ recommended_directions: [           │
│  │     {direction, success_rate,         │
│  │      domain, added_at}               │
│  │   ]  (max 30)                        │
│  └─ forbidden_directions: [             │
│       {direction, reason, added_at}     │
│     ]  (max 50)                         │
│                                          │
│  (I) Strategic Insights                 │
│  └─ insights: [                         │
│       {insight, added_at}               │
│     ]  (max 20)                         │
│                                          │
│  Recent Rejections (max 30)             │
│  └─ [{expression, reason}]              │
│                                          │
└─────────────────────────────────────────┘
```

### 8.2 Domain Classification

Factors are classified into domains for saturation tracking:

| Domain | Keywords |
|---|---|
| `momentum` | momentum, ts_delta, ts_lag |
| `reversal` | reversal, mean_revert |
| `volatility` | volatil, ts_std, ts_kurt, ts_skew |
| `volume_price` | volume, amount, turnover |
| `correlation` | corr, cov |
| `other` | (fallback) |

### 8.3 Memory Lifecycle

| Event | Action |
|---|---|
| Each round | Update `mining_state`, increment `current_round` |
| After batch | `formation()`: classify results, update recommended/forbidden |
| Every 5 rounds | `evolution()`: LLM distills strategic insights |
| On decay | Remove entries older than 7 days |
| On trim | Cap lists at maximum sizes (30/50/20/30) |

---

## 9. Backtest Engine

### 9.1 Single Factor Backtest

Implements a **long-short quintile portfolio strategy**:

1. At each timestamp `t`, rank all stocks by factor signal.
2. Divide into 5 quintile groups (Q1 = bottom 20%, Q5 = top 20%).
3. Go long Q5, short Q1.
4. Track daily P&L: `ls_return[t] = mean(Q5_returns) - mean(Q1_returns)`.

### 9.2 Library Backtest

Combines multiple factor signals using weighted averaging:

| Method | Weight Formula |
|---|---|
| `equal` | w_i = 1/N |
| `ic_weighted` | w_i = \|IC_mean_i\| / sum(\|IC_mean\|) |
| `icir_weighted` | w_i = \|ICIR_i\| / sum(\|ICIR\|) |

The combined signal is then passed through the same quintile long-short backtest.

### 9.3 Performance Metrics

| Metric | Formula | Description |
|---|---|---|
| **IC Series** | SpearmanCorr(signal, returns) per day | Daily predictive power |
| **ICIR** | mean(IC) / std(IC) | IC stability |
| **Sharpe** | mean(excess) / std(excess) * sqrt(252) | Risk-adjusted return |
| **Max Drawdown** | min((cum - peak) / peak) | Worst peak-to-trough |
| **Annual Return** | (1 + total)^(1/years) - 1 | Annualized compound return |
| **Win Ratio** | count(ret > 0) / count(ret) | Fraction of profitable days |
| **Turnover** | mean(sum(abs(position_change))) | Trading activity |

---

## 10. MASTER Model Evaluator

### 10.1 MASTER Architecture

**MASTER** (Market-Guided Stock Transformer) from AAAI 2024:

```
Input: (n_stocks, seq_len, n_features)
         │
         ▼
  ┌──────────────┐
  │ Input Proj    │  Linear(n_features -> d_model) + Positional Embedding
  └──────┬───────┘
         │
    N x  ▼
  ┌──────────────┐
  │ MASTER Block  │
  │ ├─ Intra-Stock│  Self-attention over time steps (within each stock)
  │ │  Attention   │
  │ ├─ FFN + Norm │
  │ ├─ Inter-Stock│  Self-attention across stocks (within each time step)
  │ │  Attention   │
  │ └─ FFN + Norm │
  └──────┬───────┘
         │
         ▼
  ┌──────────────┐
  │ Market Gate   │  Sigmoid(FC(concat(mean, std))) -- cross-sectional stats
  └──────┬───────┘
         │
         ▼
  ┌──────────────┐
  │ FC Head       │  Linear -> ReLU -> Dropout -> Linear(1)
  └──────────────┘
         │
         ▼
  Output: (n_stocks, 1)  predicted returns
```

### 10.2 Feature Engineering

20 alpha features built from OHLCV panel data:

| # | Feature | Formula | Category |
|---|---|---|---|
| 1 | ret_1d | close / close[-1] - 1 | Returns |
| 2 | ret_5d | close / close[-5] - 1 | Returns |
| 3 | ret_10d | close / close[-10] - 1 | Returns |
| 4 | ret_20d | close / close[-20] - 1 | Returns |
| 5 | vol_ratio_5 | volume / MA(volume, 5) | Volume |
| 6 | vol_ratio_20 | volume / MA(volume, 20) | Volume |
| 7 | volatility_5 | std(returns, 5) | Volatility |
| 8 | volatility_20 | std(returns, 20) | Volatility |
| 9 | vwap_dev | (close - vwap) / vwap | VWAP |
| 10 | hl_range | (high - low) / close | Candle |
| 11 | close_position | (close - low) / (high - low) | Candle |
| 12 | intraday_ret | (close - open) / open | Candle |
| 13 | open_gap | (open - prev_close) / prev_close | Gap |
| 14 | high_dev | (high - close) / close | Tail |
| 15 | low_dev | (close - low) / close | Tail |
| 16 | amount_ratio_5 | amount / MA(amount, 5) | Amount |
| 17 | risk_adj_mom | ret_5d / (vol_20d + eps) | Risk-adjusted |
| 18 | volume_mom_5 | volume / volume[-5] | Volume |
| 19 | vol_volatility | std(vol_ratio_20, 5) | Volume |
| 20 | cs_rank_ret5 | CS_rank(ret_5d) | Cross-sectional |

All features are cross-sectionally z-scored and clipped to [-3, 3].

### 10.3 Walk-Forward Evaluation Protocol

```
Timeline:  |---warmup---|-----train window-----|---test step---|---...
           0          seq_len              train_end        test_end

Fold 1: train [seq_len, seq_len+500)  test [seq_len+500, seq_len+560)
Fold 2: train [60+seq_len, 60+seq_len+500)  test [..., ...+60)
...
```

At each fold:
1. Train MASTER on all cross-sections within the training window.
2. Predict for each test date.
3. Compute Spearman rank IC between predictions and actual forward returns.

### 10.4 IC Evaluation Criteria

| Metric | Threshold | Rating |
|---|---|---|
| **IC Mean** | >= 0.08 | STRONG (30 pts) |
| | >= 0.05 | GOOD (24 pts) |
| | >= 0.03 | MODERATE (16 pts) |
| | >= 0.02 | WEAK (8 pts) |
| | < 0.02 | VERY POOR (0 pts) |
| **ICIR** | >= 1.5 | EXCELLENT (30 pts) |
| | >= 1.0 | GOOD (24 pts) |
| | >= 0.5 | DECENT (16 pts) |
| | >= 0.3 | MEDIOCRE (8 pts) |
| | < 0.3 | UNSTABLE (0 pts) |
| **IC+%** | >= 60% | GOOD (20 pts) |
| | >= 55% | DECENT (14 pts) |
| | >= 50% | MARGINAL (8 pts) |
| | < 50% | POOR (0 pts) |
| **t-stat** | >= 2.58 | Significant at 1% (20 pts) |
| | >= 1.96 | Significant at 5% (14 pts) |
| | < 1.96 | NOT significant (0 pts) |

**Final Verdict** (0-100 score):

| Score | Verdict | Recommendation |
|---|---|---|
| >= 80 | STRONG | Suitable for production use |
| >= 60 | GOOD | Promising, consider further tuning |
| >= 40 | PASS | May work with ensemble/tuning |
| >= 25 | WEAK | Significant improvement needed |
| < 25 | FAIL | Not recommended for live trading |

### 10.5 Usage

```bash
# Quick smoke test (50 stocks, 5 epochs, small model)
python evaluate_master.py --fast

# Default evaluation (100 stocks, 15 epochs, d=64, 2 layers)
python evaluate_master.py

# Thorough evaluation
python evaluate_master.py --stocks 200 --epochs 30 --d-model 128 --n-layers 3
```

### 10.6 Latest Evaluation Results

| Metric | Value | Rating |
|---|---|---|
| IC Mean | +0.0276 | WEAK |
| ICIR | +0.184 | UNSTABLE |
| IC+% | 57.5% | DECENT |
| t-stat | 6.32 (p<0.0001) | SIGNIFICANT |
| **Score** | **42/100** | **PASS** |

**By Year**: IC decays from 0.034 (2020-2022) to 0.005 (2024).
**By Regime**: Performs better in down markets (IC=0.032) than up markets (IC=0.024).

---

## 11. Web Dashboard & API

### 11.1 REST API Endpoints

| Method | Path | Description |
|---|---|---|
| `GET` | `/` | Serve dashboard HTML |
| `POST` | `/api/mining/start` | Start mining with config |
| `POST` | `/api/mining/stop` | Stop mining |
| `GET` | `/api/mining/status` | Mining status |
| `GET` | `/api/factors` | List all factors |
| `GET` | `/api/factors/{id}/detail` | Factor detail with analysis |
| `POST` | `/api/backtest` | Run backtest |
| `GET` | `/api/memory` | Experience memory state |
| `WS` | `/ws/mining` | Real-time mining logs |

### 11.2 Request/Response Models

**MiningConfig** (POST /api/mining/start):
```json
{
  "batch_size": 10,
  "max_rounds": 50,
  "ic_threshold": 0.02,
  "corr_threshold": 0.7,
  "target_size": 100,
  "stock_universe": "000300"
}
```

**BacktestRequest** (POST /api/backtest):
```json
{
  "factor_id": "F623854",
  "method": "ic_weighted"
}
```

### 11.3 Dashboard Features

The single-page dashboard (`templates/index.html`) provides 4 tabs:

1. **Mining Control**: Configuration form, start/stop buttons, real-time WebSocket log stream, admission rate chart.
2. **Factor Library**: Sortable table of all factors with IC, ICIR, correlation, turnover. Click for detailed factor tearsheet.
3. **Backtest**: Settings panel, equity curve chart, quintile group analysis, performance metrics table.
4. **Experience Memory**: Mining state overview, domain saturation bars, recommended/forbidden directions, strategic insights.

Tech stack: TailwindCSS (CDN), ECharts (CDN), vanilla JavaScript.

---

## 12. Deployment

### 12.1 Local Development

```bash
# Install dependencies
pip install -r requirements.txt

# Set environment variables
cp .env.example .env
# Edit .env: set KIMI_API_KEY

# Run pipeline test (no API key needed)
python test_pipeline.py

# Run mining
python run.py mine --rounds 5 --batch 10

# Start web server
python run.py server --port 8000

# Evaluate MASTER model
python evaluate_master.py --fast
```

### 12.2 Docker

```bash
# Build and run
docker compose up

# Or manually
docker build -t factor-miner .
docker run -p 8000:8000 --env-file .env -v ./storage:/app/storage factor-miner
```

**Dockerfile** uses `python:3.11-slim`, installs gcc/g++ for NumPy/SciPy compilation, exposes port 8000.

**docker-compose.yml** mounts `./storage` as a volume for persistence and loads `.env` for API keys.

### 12.3 Dependencies

```
# Core
numpy>=1.24.0          # Array computation
pandas>=2.0.0          # Data manipulation
scipy>=1.11.0          # Statistics (Spearman correlation)
akshare>=1.12.0        # A-share data source (free)

# LLM
openai>=1.0.0          # OpenAI-compatible API client (Kimi)

# Web
fastapi>=0.104.0       # REST API framework
uvicorn>=0.24.0        # ASGI server
websockets>=12.0       # WebSocket support
pydantic>=2.5.0        # Data validation
jinja2>=3.1.0          # Template rendering
aiofiles>=23.2.0       # Async file serving

# Environment
python-dotenv>=1.0.0   # .env file loading

# ML (MASTER Evaluator)
torch>=2.0.0           # PyTorch for MASTER model
loguru>=0.7.0          # Structured logging
scikit-learn>=1.3.0    # Fallback Ridge regression
```

---

## 13. Configuration Reference

### 13.1 Environment Variables (`.env`)

| Variable | Required | Default | Description |
|---|---|---|---|
| `LLM_PROVIDER` | No | `kimi` | LLM provider identifier |
| `KIMI_API_KEY` | Yes | -- | Moonshot AI API key |
| `KIMI_BASE_URL` | No | `https://api.moonshot.cn/v1` | API base URL |
| `KIMI_MODEL` | No | `kimi-k2.5` | Model name |

### 13.2 Mining Tuning Guide

| Goal | Parameters to Adjust |
|---|---|
| **Higher quality factors** | Raise `IC_THRESHOLD` to 0.03-0.05 |
| **More diverse library** | Lower `CORR_THRESHOLD` to 0.5-0.6 |
| **Faster exploration** | Increase `BATCH_SIZE` to 20-30 |
| **Deeper search** | Increase `MAX_ROUNDS` to 100+ |
| **Broader validation** | Increase `FULL_UNIVERSE_ASSETS` to 500 |

---

## 14. Development & Testing

### 14.1 Running Tests

```bash
# Pipeline test (no LLM required)
python test_pipeline.py

# MASTER model evaluation
python evaluate_master.py --fast
```

### 14.2 Adding New Operators

1. Implement the operator function in `factor_mining/operators.py`:
   ```python
   def ts_my_operator(x: np.ndarray, d: int) -> np.ndarray:
       """Description of the operator."""
       M, T = x.shape
       out = np.full_like(x, np.nan)
       for t in range(d - 1, T):
           window = x[:, t - d + 1: t + 1]
           out[:, t] = ...  # Your computation
       return out
   ```

2. Register it in `OPERATOR_REGISTRY`:
   ```python
   "ts_my_operator": (ts_my_operator, "ts"),
   ```

3. The expression engine will automatically support `ts_my_operator(close, 20)` in factor expressions.

### 14.3 Adding New Data Fields

1. Add the field computation in `data/stock_data.py` within `build_panel_from_parquet()` or `build_market_panel()`.
2. Ensure the field is added to the panel dict with shape `(M, T)`.
3. The expression engine will automatically resolve the field name.

---

## 15. Project Roadmap

### Phase 1: Core Mining Engine (Completed)

- [x] Expression parser with recursive descent
- [x] 40+ operator library (time-series, cross-sectional, element-wise)
- [x] Ralph Loop implementation (Retrieve -> Generate -> Evaluate -> Distill)
- [x] Factor library with orthogonal admission pipeline
- [x] Three-part experience memory system
- [x] Multi-stage IC validation (fast screen + full validation)
- [x] CLI runner with mine/list/backtest commands

### Phase 2: Web Dashboard & API (Completed)

- [x] FastAPI REST API with full CRUD
- [x] WebSocket for real-time mining log streaming
- [x] Single-page dashboard (TailwindCSS + ECharts)
- [x] Mining control panel
- [x] Factor library browser
- [x] Backtest visualization
- [x] Experience memory viewer

### Phase 3: Model Evaluation (Completed)

- [x] MASTER (Market-Guided Stock Transformer) integration
- [x] Walk-forward IC evaluation framework
- [x] 20-feature engineering pipeline from OHLCV
- [x] Comprehensive IC report with pass/fail verdict
- [x] Multi-dimensional analysis (yearly, regime, rolling stability)

### Phase 4: IC Improvement & Architecture Upgrade (In Progress)

**Decision Log (2026-02-18):**

#### Problem Statement

Individual formulaic factors from Ralph Loop mining achieve |IC| ~ 0.03-0.06 on 300 A-share stocks (2018-2024). This is below production threshold (|IC| >= 0.05, ICIR >= 0.5). Root cause: individual factors are evaluated in isolation; non-linear factor interactions carry additional alpha.

#### Research Findings

Based on [FactorMiner (arXiv:2602.14670)](https://arxiv.org/abs/2602.14670) and concurrent work:

| Approach | Source | Expected IC | Status |
|----------|--------|-------------|--------|
| AlphaAgent (AST-regularized exploration) | arXiv:2502.16789 | 0.05+ | Reviewed |
| QuantaAlpha (evolutionary operations) | arXiv:2602.07085 | 0.15 (GPT-5.2, CSI300) | Reviewed |
| AlphaPROBE (DAG-based navigation) | arXiv:2602.11917 | 0.06+ | Reviewed |
| Hybrid LLM approach | Frontiers CS 2025 | 0.0515 (+75% over baseline) | Reviewed |
| DoubleEnsemble | Local factor_investing | 0.0521 (Alpha158) | Available |
| HIST (heterogeneous info) | Local factor_investing | 0.0522 (Alpha360) | Available |
| GBDT Factor Combination | Local factor_investing | 0.05-0.10 | **Implemented** |

#### Implemented Changes

1. **Factor Combiner** (`factor_mining/combiner.py`): Walk-forward factor combination using LightGBM, HistGradientBoosting (sklearn), or Ridge. Combines mined formulaic factors + 63 derived features into a high-IC composite signal.

   Results on 300 stocks x 1699 days (67 features, HLZ+OOS screened):
   ```
   Individual factors:  avg |IC| = 0.0363, best |IC| = 0.0661
   Ridge combiner:      IC = +0.0439, ICIR = +0.374, IC+% = 66.4%
   HGBR combiner:       IC = +0.0526, ICIR = +0.500, IC+% = 70.8%
   LightGBM combiner:   IC = +0.0544, ICIR = +0.517, IC+% = 71.2%  ← PRODUCTION CANDIDATE
   ```
   Verdict: **GOOD - Production candidate** (ICIR > 0.5, IC positive > 70%)

2. **HLZ Multiple Testing Correction** (`factor_library.py`): Harvey-Liu-Zhu (2016) adjusted t-statistic threshold. With 71 tested factors, requires |t| > 3.39 for admission. All 8 existing mined factors pass (|t| range: 7.11 to 14.04).

3. **OOS Validation** (`factor_library.py`): Out-of-sample split (60/40 with 5-day purge gap). Rejects factors with > 70% IC degradation OOS. Most factors actually improve OOS (negative degradation = stronger out-of-sample).

4. **Performance Optimizations** (P0):
   - Bottleneck C-compiled operators: 10-100x speedup (ts_mean: 50ms → 0.2ms)
   - AST parse caching in expression engine
   - Signal cache persists across candidates (avoid O(N²) re-evaluation)
   - Vectorized _cs_zscore (~100x)

5. **Architecture Cleanup** (P1-P2):
   - Eliminated run.py mining loop duplication (~200 LOC saved)
   - Consolidated IC metrics to single source (backtest/metrics.py)
   - Unified MASTER model to single definition
   - Shared verdict scoring (evaluator/scoring.py)
   - Config dataclass with env-var overrides
   - AsyncOpenAI for non-blocking LLM calls
   - Encapsulated API state (AppState class)
   - pyproject.toml for proper packaging

6. **Environment**:
   - Homebrew installed at /opt/homebrew (manual tarball, no sudo)
   - libomp 21.1.8 installed for LightGBM/XGBoost OpenMP support
   - LightGBM 4.6.0, XGBoost 2.1.4, scikit-learn 1.6.1 all verified

#### IC Improvement Trajectory

```
Baseline (individual factors):  avg |IC| = 0.0363    ICIR = -0.25   IC+% = 40%
+ Ridge combiner:               IC = +0.0439          ICIR = +0.374  IC+% = 66%
+ HGBR combiner:                IC = +0.0526          ICIR = +0.500  IC+% = 71%
+ LightGBM combiner:            IC = +0.0544          ICIR = +0.517  IC+% = 71%  ← CURRENT
Target:                         |IC| >= 0.08           ICIR >= 1.0    IC+% >= 75%
```

#### Full Pipeline (2000+ predictors)

- **Command**: `python run.py pipeline --max-stocks 500 --max-depth 3 --fundamentals --workers 4`
- **Flow**: Load panel (OHLCV + fundamentals) → generate 36k+ candidates → screen (IC, HLZ, OOS, corr dedup) → keep up to 2000 → LightGBM/HGBR/Ridge combiner → report combined IC.
- **Fundamental fields** (when `--fundamentals`): bvps, roe_pct, basic_eps, n_income, revenue, total_mv, pe, pb, dv_ttm from factor_investing/data/raw (batch_balance_all, batch_income_all, daily_basic).
- **IC target**: In practice, single-factor IC > 0.05 is good, combined IC ≥ 0.08 is a strong target. IC > 0.8 would indicate data leakage; aim for |IC| ≥ 0.08 (8%) for the combined signal.
- **Throughput**: Screening ~1–2 expr/s per worker; use `--workers 4` and `--max-depth 2` for faster runs (~5k candidates).

#### Next Steps (priority order)

- [ ] **Integrate AlphaPROBE** (arXiv:2602.11917, PKU/Zhengren Quant):
  DAG-based alpha mining with Bayesian seed retrieval + DAG-aware generation.
  Reframes factor discovery as navigation on a Directed Acyclic Graph —
  exploits global evolutionary topology for non-redundant, diverse alpha generation.
  Paper: https://arxiv.org/abs/2602.11917 | Code: https://github.com/gta0804/AlphaPROBE
- [ ] Integrate DoubleEnsemble from factor_investing (sample reweighting + feature subspacing)
- [ ] Add regime-aware weighting (MarketStateDetector from factor_investing)
- [ ] Multi-horizon IC (1d, 5d, 10d, 20d) for factor stability assessment
- [ ] Integrate HIST model for concept/relation-based stock embeddings
- [ ] AlphaAgent-style AST regularization to reduce correlated factor generation
- [ ] FactorVAE for disentangled latent factor representations

### Phase 5: Production Readiness (Planned)

- [ ] Transaction cost modeling (A-share: stamp duty 0.1%, commission ~0.03%)
- [ ] Live paper trading integration
- [ ] Factor decay monitoring and auto-refresh
- [ ] Multi-LLM support (Claude, GPT-4, DeepSeek, Kimi)
- [ ] Genetic Programming (GP) factor generation alongside LLM
- [ ] Automated hyperparameter tuning

---

### Key References (Alpha Mining)

| Paper | Year | Method | Key Result |
|-------|------|--------|------------|
| **AlphaPROBE** (arXiv:2602.11917) | 2026 | Bayesian retrieval + DAG-aware evolution | Outperforms 8 baselines on CSI300/500/1000 |
| **QuantaAlpha** (arXiv:2602.07085) | 2026 | Evolutionary LLM-driven mining | IC=0.1501, ARR=27.75% on CSI300 |
| **AlphaForge** (AAAI 2025) | 2024 | Generative-predictive NN + dynamic combination | Enhanced portfolio returns vs baselines |
| **Gu, Kelly, Xiu** (RFS 2020) | 2020 | ML asset pricing (tree + NN) | 2x performance over linear models |
| **WorldQuant 101** | 2015 | 101 formulaic alphas | Avg pairwise corr 15.9% |
| **Harvey, Liu, Zhu** (RFS 2016) | 2016 | Factor Zoo / multiple testing | 27-53% false discovery in 296 factors |

---

*Document updated: 2026-02-20*
*FactorMiner v2.0.0 -- 2211 signals, Sharpe=3.625, Ann Return=72.79%, Max DD=-5.00%*
