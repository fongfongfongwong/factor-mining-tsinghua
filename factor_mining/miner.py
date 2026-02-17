"""FactorMiner: main orchestrator implementing Algorithm 1 (Ralph Loop) from the paper.

Cycle: Retrieve -> Generate -> Evaluate -> Distill
"""

import json
import logging
import asyncio
import time
import re
from typing import Optional, Callable, Any

import numpy as np

from config import (
    LLM_API_KEY,
    LLM_BASE_URL,
    LLM_MODEL,
    IC_THRESHOLD,
    CORR_THRESHOLD,
    BATCH_SIZE,
    MAX_ROUNDS,
    TARGET_LIBRARY_SIZE,
    FAST_SCREEN_ASSETS,
    FULL_UNIVERSE_ASSETS,
)
from data.stock_data import build_market_panel, calculate_returns, get_stock_list
from .expression_engine import ExpressionEngine, validate_expression
from .factor_library import FactorLibrary, FactorRecord, compute_ic, compute_icir
from .experience_memory import ExperienceMemory

logger = logging.getLogger(__name__)


GENERATION_PROMPT = """你是一个专业的量化因子研究员。请生成{batch_size}个公式化alpha因子。

## 可用数据字段
open, high, low, close, volume, amount, vwap, returns

## 可用算子
时序算子: ts_rank, ts_std, ts_mean, ts_sum, ts_min, ts_max, ts_argmin, ts_argmax,
         ts_delta, ts_lag, ts_corr, ts_cov, ts_skew, ts_kurt, ts_decay_linear,
         ts_product, ts_rsquare, ts_regression_residual
截面算子: cs_rank, cs_zscore, cs_scale, cs_demean
元素算子: abs, log, sign, sqrt, neg, inv, add, sub, mul, div, power, max, min

## 当前因子库状态
已有因子数: {library_size}
{mining_state_summary}

## 推荐方向（历史高成功率模式）
{recommended_directions}

## 禁止方向（高相关性/低IC区域，请避开）
{forbidden_directions}

## 最近拒绝记录
{recent_rejections}

## 要求
1. 每个因子使用上述算子和字段的嵌套组合
2. 因子之间应体现不同的金融逻辑（动量、反转、波动率、量价关系等）
3. 避免禁止方向中已知的高相关性模式
4. 因子表达式语法示例: cs_rank(ts_corr(close, volume, 10))
5. 窗口参数d应为合理整数（5-60）

请以JSON格式回复:
[{{"expression": "因子表达式", "logic": "金融逻辑描述"}}]"""


INSIGHT_PROMPT = """你是一个量化因子研究专家。基于以下挖掘结果，请总结2-3条战略性发现：

## 本轮结果
录用因子: {admitted_count}/{total_count}
录用率: {admission_rate:.1%}

## 录用的因子
{admitted_factors}

## 被拒绝的因子（含原因）
{rejected_factors}

## 当前已知的战略洞察
{existing_insights}

请以JSON数组格式回复，每条洞察是一个字符串：
["洞察1", "洞察2"]"""


class FactorMiner:
    """Self-evolving factor mining agent implementing the Ralph Loop."""

    def __init__(
        self,
        batch_size: int = BATCH_SIZE,
        max_rounds: int = MAX_ROUNDS,
        ic_threshold: float = IC_THRESHOLD,
        corr_threshold: float = CORR_THRESHOLD,
        target_size: int = TARGET_LIBRARY_SIZE,
        stock_universe: str = "000300",
    ):
        self.batch_size = batch_size
        self.max_rounds = max_rounds
        self.ic_threshold = ic_threshold
        self.corr_threshold = corr_threshold
        self.target_size = target_size
        self.stock_universe = stock_universe

        self.library = FactorLibrary()
        self.memory = ExperienceMemory()

        self._running = False
        self._current_round = 0
        self._status = "idle"
        self._log_callback: Optional[Callable] = None

        self._fast_panel: Optional[dict] = None
        self._full_panel: Optional[dict] = None
        self._fast_engine: Optional[ExpressionEngine] = None
        self._full_engine: Optional[ExpressionEngine] = None
        self._fast_fwd_returns: Optional[np.ndarray] = None
        self._full_fwd_returns: Optional[np.ndarray] = None

    @property
    def is_running(self) -> bool:
        return self._running

    @property
    def status(self) -> str:
        return self._status

    @property
    def current_round(self) -> int:
        return self._current_round

    def set_log_callback(self, callback: Callable):
        self._log_callback = callback

    async def _log(self, msg: str, level: str = "info"):
        log_entry = {"time": time.time(), "level": level, "message": msg, "round": self._current_round}
        getattr(logger, level)(msg)
        if self._log_callback:
            try:
                await self._log_callback(log_entry)
            except Exception:
                pass

    def _prepare_data(self):
        """Prepare fast-screening and full-universe data panels."""
        self._status = "preparing_data"
        codes = get_stock_list(self.stock_universe)
        if not codes:
            raise RuntimeError("Failed to fetch stock list")

        fast_codes = codes[:min(FAST_SCREEN_ASSETS, len(codes))]
        self._fast_panel = build_market_panel(fast_codes, max_stocks=FAST_SCREEN_ASSETS)
        self._fast_engine = ExpressionEngine(self._fast_panel)
        self._fast_fwd_returns = calculate_returns(self._fast_panel)

        full_codes = codes[:min(FULL_UNIVERSE_ASSETS, len(codes))]
        self._full_panel = build_market_panel(full_codes, max_stocks=FULL_UNIVERSE_ASSETS)
        self._full_engine = ExpressionEngine(self._full_panel)
        self._full_fwd_returns = calculate_returns(self._full_panel)

    async def _call_llm(self, prompt: str) -> str:
        """Call Kimi 2.5 API (OpenAI-compatible)."""
        try:
            from openai import OpenAI
            client = OpenAI(api_key=LLM_API_KEY, base_url=LLM_BASE_URL)
            response = client.chat.completions.create(
                model=LLM_MODEL,
                max_tokens=4096,
                messages=[{"role": "user", "content": prompt}],
            )
            return response.choices[0].message.content
        except Exception as e:
            logger.error(f"LLM call failed: {e}")
            raise

    async def _generate_candidates(self) -> list[dict]:
        """Retrieve memory priors and generate candidate expressions via LLM."""
        memory_context = self.memory.retrieve(self.library.size)
        prompt = GENERATION_PROMPT.format(
            batch_size=self.batch_size,
            library_size=self.library.size,
            **memory_context,
        )

        await self._log(f"Calling LLM to generate {self.batch_size} candidates...")
        response_text = await self._call_llm(prompt)

        candidates = self._parse_candidates(response_text)
        await self._log(f"Parsed {len(candidates)} candidates from LLM response")
        return candidates

    def _parse_candidates(self, text: str) -> list[dict]:
        """Parse LLM response JSON into candidate list."""
        json_match = re.search(r'\[.*\]', text, re.DOTALL)
        if not json_match:
            logger.warning("No JSON array found in LLM response")
            return []

        try:
            raw = json.loads(json_match.group())
            candidates = []
            for item in raw:
                expr = item.get("expression", "").strip()
                logic = item.get("logic", "").strip()
                if expr:
                    valid, err = validate_expression(expr)
                    if valid:
                        candidates.append({"expression": expr, "logic": logic})
                    else:
                        logger.debug(f"Invalid expression '{expr}': {err}")
            return candidates
        except json.JSONDecodeError as e:
            logger.warning(f"Failed to parse LLM JSON: {e}")
            return []

    async def _evaluate_candidate(self, candidate: dict) -> dict:
        """Multi-stage evaluation pipeline for a single candidate."""
        expr = candidate["expression"]
        result = {
            "expression": expr,
            "logic": candidate.get("logic", ""),
            "admitted": False,
            "reason": "",
            "ic_mean": 0.0,
        }

        try:
            fast_signal = self._fast_engine.evaluate(expr)
        except Exception as e:
            result["reason"] = f"Evaluation error: {e}"
            return result

        nan_ratio = np.isnan(fast_signal).sum() / fast_signal.size
        if nan_ratio > 0.5:
            result["reason"] = f"Too many NaNs: {nan_ratio:.1%}"
            return result

        fast_ic = compute_ic(fast_signal, self._fast_fwd_returns)
        valid_ic = fast_ic[~np.isnan(fast_ic)]
        if len(valid_ic) < 10:
            result["reason"] = "Insufficient IC data from fast screen"
            return result

        ic_mean = float(np.mean(valid_ic))
        result["ic_mean"] = ic_mean

        if abs(ic_mean) < self.ic_threshold:
            result["reason"] = f"Fast screen IC too low: {abs(ic_mean):.4f}"
            return result

        await self._log(f"  [{expr[:50]}] passed fast screen IC={ic_mean:.4f}")

        try:
            full_signal = self._full_engine.evaluate(expr)
        except Exception as e:
            result["reason"] = f"Full eval error: {e}"
            return result

        full_ic = compute_ic(full_signal, self._full_fwd_returns)
        full_valid_ic = full_ic[~np.isnan(full_ic)]
        if len(full_valid_ic) < 10:
            result["reason"] = "Insufficient IC data from full validation"
            return result

        full_ic_mean = float(np.mean(full_valid_ic))
        full_ic_std = float(np.std(full_valid_ic))
        full_icir = compute_icir(full_ic)
        result["ic_mean"] = full_ic_mean

        self.library.clear_signal_cache()
        _ = self.library.get_all_signals(self._full_engine)

        admitted, reason, replace_idx = self.library.check_admission(
            full_signal, full_ic,
            ic_threshold=self.ic_threshold,
            corr_threshold=self.corr_threshold,
            engine=self._full_engine,
        )

        result["admitted"] = admitted
        result["reason"] = reason

        if admitted:
            positions = full_signal[:, 1:] - full_signal[:, :-1]
            turnover = float(np.nanmean(np.nanmean(np.abs(positions), axis=0)))

            max_corr = 0.0
            for existing in self.library.factors:
                if existing.expression in self.library._signal_cache:
                    from .factor_library import compute_correlation
                    corr = abs(compute_correlation(
                        full_signal,
                        self.library._signal_cache[existing.expression],
                    ))
                    max_corr = max(max_corr, corr)

            record = FactorRecord(
                expression=expr,
                ic_mean=full_ic_mean,
                ic_std=full_ic_std,
                icir=full_icir,
                max_correlation=max_corr,
                turnover=turnover,
                logic_description=candidate.get("logic", ""),
                mining_round=self._current_round,
            )
            self.library.admit(record, replace_idx)
            await self._log(
                f"  ADMITTED: {expr[:60]} | IC={full_ic_mean:.4f} ICIR={full_icir:.4f}",
                "info",
            )

        return result

    async def _distill(self, batch_results: list[dict]):
        """Update memory with batch results (Distill phase)."""
        self.memory.formation(batch_results)

        if self._current_round % 5 == 0 and self._current_round > 0:
            try:
                admitted = [r for r in batch_results if r["admitted"]]
                rejected = [r for r in batch_results if not r["admitted"]]

                admitted_text = "\n".join(
                    f"- {r['expression'][:60]}: IC={r['ic_mean']:.4f}, {r.get('logic', '')}"
                    for r in admitted
                ) or "无"

                rejected_text = "\n".join(
                    f"- {r['expression'][:60]}: {r['reason']}"
                    for r in rejected[:10]
                ) or "无"

                existing_text = "\n".join(
                    f"- {i['insight']}" for i in self.memory.strategic_insights
                ) or "无"

                prompt = INSIGHT_PROMPT.format(
                    admitted_count=len(admitted),
                    total_count=len(batch_results),
                    admission_rate=len(admitted) / max(len(batch_results), 1),
                    admitted_factors=admitted_text,
                    rejected_factors=rejected_text,
                    existing_insights=existing_text,
                )

                response = await self._call_llm(prompt)
                json_match = re.search(r'\[.*\]', response, re.DOTALL)
                if json_match:
                    insights = json.loads(json_match.group())
                    self.memory.evolution(insights)
                    await self._log(f"Distilled {len(insights)} new strategic insights")
            except Exception as e:
                logger.warning(f"Insight distillation failed: {e}")

    async def run(self):
        """Main mining loop implementing Algorithm 1."""
        if self._running:
            await self._log("Mining already running", "warning")
            return

        self._running = True
        self._status = "starting"

        try:
            await self._log("Preparing data panels...")
            loop = asyncio.get_event_loop()
            await loop.run_in_executor(None, self._prepare_data)
            await self._log(
                f"Data ready: fast={self._fast_panel['close'].shape}, "
                f"full={self._full_panel['close'].shape}"
            )

            self._status = "mining"

            for round_num in range(1, self.max_rounds + 1):
                if not self._running:
                    await self._log("Mining stopped by user")
                    break

                if self.library.size >= self.target_size:
                    await self._log(
                        f"Target library size {self.target_size} reached! "
                        f"Library has {self.library.size} factors."
                    )
                    break

                self._current_round = round_num
                self.memory.update_round(round_num)
                await self._log(f"=== Round {round_num}/{self.max_rounds} ===")

                candidates = await self._generate_candidates()
                if not candidates:
                    await self._log("No valid candidates generated, retrying...", "warning")
                    continue

                batch_results = []
                for i, candidate in enumerate(candidates):
                    if not self._running:
                        break
                    await self._log(
                        f"  Evaluating [{i+1}/{len(candidates)}]: "
                        f"{candidate['expression'][:60]}"
                    )
                    result = await self._evaluate_candidate(candidate)
                    batch_results.append(result)

                admitted_count = sum(1 for r in batch_results if r["admitted"])
                await self._log(
                    f"Round {round_num} summary: "
                    f"{admitted_count}/{len(batch_results)} admitted, "
                    f"library size = {self.library.size}"
                )

                await self._distill(batch_results)
                await asyncio.sleep(0.1)

            self._status = "completed"
            await self._log(
                f"Mining completed. Library: {self.library.size} factors, "
                f"Rounds: {self._current_round}"
            )

        except Exception as e:
            self._status = "error"
            await self._log(f"Mining error: {e}", "error")
            logger.exception("Mining loop error")

        finally:
            self._running = False

    def stop(self):
        """Signal the mining loop to stop."""
        self._running = False
        self._status = "stopping"

    def get_status(self) -> dict:
        return {
            "status": self._status,
            "current_round": self._current_round,
            "max_rounds": self.max_rounds,
            "library_size": self.library.size,
            "target_size": self.target_size,
            "is_running": self._running,
            "admission_rate": self.memory.mining_state.get("recent_admission_rate", 0),
        }
