"""Evaluation and comparison reporting."""

from .metrics import ModelResult, build_comparison_table, evaluate_predictions

__all__ = ["ModelResult", "build_comparison_table", "evaluate_predictions"]
