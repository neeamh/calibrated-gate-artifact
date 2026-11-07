"""Typed evaluation contract; metric payloads retain their original meaning."""
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Literal

import pandas as pd

Status = Literal["Pass", "Conditional", "Fail", "NotEvaluated"]


@dataclass(frozen=True)
class DimensionResult:
    """A dimension verdict plus unmodified metrics and extension metadata.

    Frozen fields prevent reassignment; nested legacy dictionaries remain plain
    dictionaries. No normalization, rounding, missing-value replacement, or
    threshold remapping occurs here.
    """

    dimension: str
    status: Status
    metrics: dict[str, dict[str, Any]]
    weaknesses: list[str]
    warnings: tuple[str, ...] = ()
    details: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> DimensionResult:
        """Wrap a legacy result, keeping every metric and additional field."""
        core = {"dimension", "status", "metrics", "weaknesses", "warnings"}
        return cls(
            dimension=payload["dimension"], status=payload["status"],
            metrics=payload["metrics"], weaknesses=payload["weaknesses"],
            warnings=tuple(payload.get("warnings", ())),
            details={key: value for key, value in payload.items() if key not in core},
        )

    def to_dict(self, *, include_warnings: bool = False) -> dict[str, Any]:
        """Return the exact legacy schema by default; warnings are opt-in."""
        payload = {
            "dimension": self.dimension, "status": self.status,
            "metrics": self.metrics, "weaknesses": self.weaknesses,
            **self.details,
        }
        if include_warnings or self.warnings:
            payload["warnings"] = list(self.warnings)
        return payload


class DimensionEvaluator(ABC):
    """Shared interface for candidate-table evaluation."""

    @abstractmethod
    def evaluate(self, synthetic: pd.DataFrame) -> DimensionResult:
        """Return a dimension's evidence and verdict without aggregating others."""
        raise NotImplementedError
