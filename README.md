# FactorMiner: A-Share Alpha Factor Mining System

LLM-powered alpha factor mining system for the Chinese A-share market, inspired by the [FactorMiner paper](https://arxiv.org/abs/2602.14670) (Tsinghua University, 2026).

## Features

- **Self-evolving agent**: Ralph Loop (Retrieve → Generate → Evaluate → Distill) continuously improves factor discovery
- **40+ operators**: Time-series, cross-sectional, and element-wise operators implemented in pure NumPy
- **Experience memory**: Accumulates mining knowledge across sessions (recommended/forbidden directions, strategic insights)
- **Multi-stage validation**: IC screening → correlation check → replacement check → admission
- **Backtest engine**: Factor-based long-short quintile portfolio backtesting with full metrics
- **Web dashboard**: Modern UI with mining control, factor library browser, backtest charts, and memory viewer
- **A-share native**: AkShare data (free, no API key), Chinese prompts, Chinese market conventions

## Quick Start

1. Clone and install dependencies:

```bash
cd factor-miner
pip install -r requirements.txt
```

2. Set your Anthropic API key:

```bash
cp .env.example .env
# Edit .env and add your ANTHROPIC_API_KEY
```

3. Run the server:

```bash
python -m uvicorn api.main:app --host 0.0.0.0 --port 8000 --reload
```

4. Open the dashboard at `http://localhost:8000`

## Docker

```bash
docker compose up
```

## Architecture

The system implements the Ralph Loop from the FactorMiner paper:

1. **Retrieve**: Fetch recommended directions + forbidden regions from experience memory
2. **Generate**: LLM produces batch of candidate factor expressions guided by memory priors
3. **Evaluate**: Multi-stage pipeline (IC screening → correlation check → replacement → admission)
4. **Distill**: Update memory with successful patterns and new forbidden regions

## Tech Stack

- **Data**: AkShare (free A-share OHLCV data)
- **LLM**: Anthropic Claude API
- **Backend**: FastAPI + WebSocket
- **Frontend**: TailwindCSS + ECharts (CDN, single-page)
- **Compute**: Pure NumPy (no GPU required)
