"""Experience Memory: three-part structured memory for self-evolving factor mining.

Implements Section 3.3 of the FactorMiner paper:
- Mining State (S): Current library status and domain saturation
- Structural Experience (P): Recommended and forbidden directions
- Strategic Insights (I): High-level lessons accumulated across sessions
"""

import json
import logging
import time
from pathlib import Path
from typing import Optional

from config import EXPERIENCE_MEMORY_PATH

logger = logging.getLogger(__name__)

MAX_RECOMMENDED = 30
MAX_FORBIDDEN = 50
MAX_INSIGHTS = 20
MAX_REJECTIONS = 30


class ExperienceMemory:
    """Three-part structured memory for guiding factor mining."""

    def __init__(self, path: Path = EXPERIENCE_MEMORY_PATH):
        self.path = path

        self.mining_state: dict = {
            "library_size": 0,
            "total_candidates": 0,
            "total_admitted": 0,
            "recent_admission_rate": 0.0,
            "domain_saturation": {},
            "current_round": 0,
        }

        self.recommended_directions: list[dict] = []
        self.forbidden_directions: list[dict] = []
        self.strategic_insights: list[dict] = []
        self.recent_rejections: list[dict] = []

        self.load()

    def load(self):
        if self.path.exists():
            try:
                with open(self.path, "r") as f:
                    data = json.load(f)
                self.mining_state = data.get("mining_state", self.mining_state)
                self.recommended_directions = data.get("recommended_directions", [])
                self.forbidden_directions = data.get("forbidden_directions", [])
                self.strategic_insights = data.get("strategic_insights", [])
                self.recent_rejections = data.get("recent_rejections", [])
                logger.info("Loaded experience memory")
            except Exception as e:
                logger.error(f"Failed to load experience memory: {e}")

    def save(self):
        data = {
            "mining_state": self.mining_state,
            "recommended_directions": self.recommended_directions,
            "forbidden_directions": self.forbidden_directions,
            "strategic_insights": self.strategic_insights,
            "recent_rejections": self.recent_rejections,
        }
        with open(self.path, "w") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)

    def retrieve(self, library_size: int) -> dict:
        """Assemble context-dependent prompt material for LLM.

        Args:
            library_size: Current factor library size.

        Returns:
            Dict with keys used in the generation prompt template.
        """
        self.mining_state["library_size"] = library_size

        state_lines = [
            f"因子库大小: {library_size}",
            f"当前轮次: {self.mining_state.get('current_round', 0)}",
            f"近期录用率: {self.mining_state.get('recent_admission_rate', 0):.1%}",
        ]
        saturation = self.mining_state.get("domain_saturation", {})
        if saturation:
            state_lines.append("领域饱和度:")
            for domain, level in sorted(saturation.items(), key=lambda x: -x[1]):
                bar = "█" * int(level * 10) + "░" * (10 - int(level * 10))
                state_lines.append(f"  {domain}: {bar} {level:.0%}")

        rec_lines = []
        for r in self.recommended_directions[-10:]:
            rec_lines.append(f"- {r['direction']} (成功率: {r.get('success_rate', 'N/A')})")

        forb_lines = []
        for f_entry in self.forbidden_directions[-10:]:
            forb_lines.append(f"- {f_entry['direction']} (原因: {f_entry.get('reason', 'high correlation')})")

        rej_lines = []
        for rej in self.recent_rejections[-5:]:
            rej_lines.append(f"- {rej.get('expression', '?')}: {rej.get('reason', '?')}")

        return {
            "mining_state_summary": "\n".join(state_lines),
            "recommended_directions": "\n".join(rec_lines) if rec_lines else "暂无",
            "forbidden_directions": "\n".join(forb_lines) if forb_lines else "暂无",
            "recent_rejections": "\n".join(rej_lines) if rej_lines else "暂无",
        }

    def formation(self, batch_results: list[dict]):
        """Extract patterns from a mining batch and update memory.

        Args:
            batch_results: List of dicts with keys:
                - expression: str
                - admitted: bool
                - reason: str
                - ic_mean: float (if evaluated)
                - logic: str
        """
        admitted = [r for r in batch_results if r.get("admitted")]
        rejected = [r for r in batch_results if not r.get("admitted")]

        if batch_results:
            rate = len(admitted) / len(batch_results)
            self.mining_state["recent_admission_rate"] = rate
            self.mining_state["total_candidates"] = (
                self.mining_state.get("total_candidates", 0) + len(batch_results)
            )
            self.mining_state["total_admitted"] = (
                self.mining_state.get("total_admitted", 0) + len(admitted)
            )

        for r in admitted:
            logic = r.get("logic", "")
            domain = self._classify_domain(r.get("expression", ""), logic)

            existing = [d["direction"] for d in self.recommended_directions]
            direction_desc = f"{logic} [{r.get('expression', '')[:60]}]"
            if direction_desc not in existing:
                self.recommended_directions.append({
                    "direction": direction_desc,
                    "success_rate": f"{r.get('ic_mean', 0):.4f}",
                    "domain": domain,
                    "added_at": time.time(),
                })

            sat = self.mining_state.get("domain_saturation", {})
            sat[domain] = min(1.0, sat.get(domain, 0) + 0.05)
            self.mining_state["domain_saturation"] = sat

        for r in rejected:
            reason = r.get("reason", "")
            if "correlated" in reason.lower():
                direction_desc = f"{r.get('expression', '')[:80]} -- {reason}"
                self.forbidden_directions.append({
                    "direction": direction_desc,
                    "reason": reason,
                    "added_at": time.time(),
                })

            self.recent_rejections.append({
                "expression": r.get("expression", "")[:80],
                "reason": reason,
            })

        self._trim()
        self.save()

    def evolution(self, new_insights: Optional[list[str]] = None):
        """Integrate new strategic insights and consolidate memory.

        Called periodically (e.g. every 5 rounds) or when LLM distills insights.
        """
        if new_insights:
            for insight in new_insights:
                if not any(i["insight"] == insight for i in self.strategic_insights):
                    self.strategic_insights.append({
                        "insight": insight,
                        "added_at": time.time(),
                    })

        self._decay_old_entries()
        self._trim()
        self.save()

    def update_round(self, round_num: int):
        self.mining_state["current_round"] = round_num

    def get_state(self) -> dict:
        """Return full memory state for API/display."""
        return {
            "mining_state": self.mining_state,
            "recommended_directions": self.recommended_directions,
            "forbidden_directions": self.forbidden_directions,
            "strategic_insights": self.strategic_insights,
            "recent_rejections": self.recent_rejections[-10:],
        }

    def _classify_domain(self, expression: str, logic: str) -> str:
        """Classify a factor into a domain category based on expression/logic."""
        text = (expression + " " + logic).lower()
        if any(w in text for w in ["momentum", "动量", "ts_delta", "ts_lag"]):
            return "momentum"
        if any(w in text for w in ["revers", "反转", "mean_revert"]):
            return "reversal"
        if any(w in text for w in ["volatil", "波动", "ts_std", "ts_kurt", "ts_skew"]):
            return "volatility"
        if any(w in text for w in ["volume", "量", "amount", "turnover", "成交"]):
            return "volume_price"
        if any(w in text for w in ["corr", "相关", "cov"]):
            return "correlation"
        return "other"

    def _decay_old_entries(self):
        """Remove stale entries (older than ~20 rounds)."""
        cutoff = time.time() - 86400 * 7
        self.recommended_directions = [
            r for r in self.recommended_directions
            if r.get("added_at", time.time()) > cutoff
        ]
        self.forbidden_directions = [
            f_entry for f_entry in self.forbidden_directions
            if f_entry.get("added_at", time.time()) > cutoff
        ]

    def _trim(self):
        """Enforce maximum list sizes."""
        if len(self.recommended_directions) > MAX_RECOMMENDED:
            self.recommended_directions = self.recommended_directions[-MAX_RECOMMENDED:]
        if len(self.forbidden_directions) > MAX_FORBIDDEN:
            self.forbidden_directions = self.forbidden_directions[-MAX_FORBIDDEN:]
        if len(self.strategic_insights) > MAX_INSIGHTS:
            self.strategic_insights = self.strategic_insights[-MAX_INSIGHTS:]
        if len(self.recent_rejections) > MAX_REJECTIONS:
            self.recent_rejections = self.recent_rejections[-MAX_REJECTIONS:]
