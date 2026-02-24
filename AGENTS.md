# AGENTS.md

## Cursor Cloud specific instructions

### Overview

FactorMiner is a single-service Python (3.10+) application — an LLM-powered alpha factor mining system for the Chinese A-share stock market. It has a FastAPI backend + single-page HTML frontend served on port 8000.

### Running the dev server

```bash
python3 -m uvicorn api.main:app --host 0.0.0.0 --port 8000 --reload
```

Dashboard at `http://localhost:8000`. See `README.md` for full Quick Start.

### Key caveats

- **No dedicated linter/test framework**: The project has no ruff/flake8/pytest/mypy configuration. Use `python3 -m py_compile <file>` for syntax checking. `test_pipeline.py` is a standalone script (not pytest), but it requires A-share parquet data files that are not present in this environment.
- **Mining requires external dependencies**: The mining loop (`/api/mining/start`) requires both (1) a valid `KIMI_API_KEY` in `.env` and (2) A-share market data from AkShare (network) or local parquet files. Without these, mining will fail at data loading. All other endpoints (status, factors, memory, dashboard) work without external dependencies.
- **No `python` alias**: Use `python3` (not `python`) to run commands; no `python` symlink exists.
- **pip installs to user site**: Dependencies install to `~/.local/` via pip. Ensure `~/.local/bin` is on `PATH`.
- **Storage is file-based**: Factor library and experience memory persist as JSON in `storage/`. No database needed.
