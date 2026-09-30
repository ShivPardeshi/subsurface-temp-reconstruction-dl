"""Compute Budget and GPU-Hour Tracker.

Tracks:
1. Cumulative execution wall-clock time and GPU hours.
2. Step throughput (steps/sec) and average latency.
3. Estimated GCP spend against NVIDIA L4 rates (Spot ~$0.25/hr, On-Demand ~$0.70/hr).
"""

from typing import Dict, Any, Optional
from pathlib import Path
import time
import json
from src.utils.logging_config import get_logger

logger = get_logger("budget_tracker")

L4_SPOT_RATE_USD_HR = 0.25
L4_ON_DEMAND_RATE_USD_HR = 0.70


class BudgetTracker:
    """Tracks training duration, step performance, and monetary cost."""

    def __init__(
        self,
        run_name: str = "oceanembed_training",
        log_dir: str = "logs",
        hourly_rate_usd: float = L4_ON_DEMAND_RATE_USD_HR,
    ):
        self.run_name = run_name
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.hourly_rate_usd = hourly_rate_usd

        self.start_time = time.time()
        self.total_steps = 0
        self.step_durations = []
        self.report_file = self.log_dir / f"budget_report_{run_name}.json"

    def record_step(self, step_seconds: float) -> None:
        """Record the elapsed time for a single training step."""
        self.total_steps += 1
        self.step_durations.append(step_seconds)
        if len(self.step_durations) > 500:
            self.step_durations.pop(0)

    def get_summary(self) -> Dict[str, Any]:
        """Compute current budget statistics."""
        elapsed_sec = time.time() - self.start_time
        gpu_hours = elapsed_sec / 3600.0
        est_cost_usd = gpu_hours * self.hourly_rate_usd

        avg_step_sec = (
            sum(self.step_durations) / len(self.step_durations)
            if self.step_durations
            else 0.0
        )
        steps_per_sec = 1.0 / avg_step_sec if avg_step_sec > 0 else 0.0

        summary = {
            "run_name": self.run_name,
            "total_steps": self.total_steps,
            "elapsed_seconds": round(elapsed_sec, 1),
            "gpu_hours": round(gpu_hours, 4),
            "steps_per_second": round(steps_per_sec, 2),
            "avg_step_ms": round(avg_step_sec * 1000, 1),
            "est_cost_usd": round(est_cost_usd, 4),
            "hourly_rate_usd": self.hourly_rate_usd,
        }
        return summary

    def save_report(self) -> Path:
        """Dump budget statistics to JSON report."""
        summary = self.get_summary()
        with open(self.report_file, "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2)
        logger.info(
            f"Budget Update: {summary['total_steps']} steps | {summary['gpu_hours']:.4f} GPU-hrs | "
            f"${summary['est_cost_usd']:.4f} USD ({summary['steps_per_second']:.2f} steps/s)"
        )
        return self.report_file
