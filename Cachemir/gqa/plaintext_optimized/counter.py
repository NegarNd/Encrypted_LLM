"""Operation accounting shared by all simulated HE primitives."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Set


@dataclass
class Counter:
    rotations: int = 0
    ct_pt_mult: int = 0
    ct_ct_mult: int = 0
    rotation_amounts: Set[int] = field(default_factory=set)
    rotation_trace: List[int] = field(default_factory=list)
    stages: Dict[str, Dict[str, object]] = field(default_factory=dict)
    total_runtime_seconds: float = 0.0

    def reset(self) -> None:
        self.rotations = 0
        self.ct_pt_mult = 0
        self.ct_ct_mult = 0
        self.rotation_amounts.clear()
        self.rotation_trace.clear()
        self.stages.clear()
        self.total_runtime_seconds = 0.0

    def record_rotation(self, amount: int, n_he: int) -> None:
        """Record one rotation and its normalized nonzero key amount."""
        normalized = amount % n_he
        if normalized == 0:
            return
        self.rotations += 1
        self.rotation_amounts.add(normalized)
        self.rotation_trace.append(normalized)

    @property
    def unique_rotations(self) -> int:
        """Number of distinct rotation amounts used by this execution."""
        return len(self.rotation_amounts)

    def snapshot(self) -> tuple[int, int, int, int]:
        return (
            self.rotations,
            self.ct_pt_mult,
            self.ct_ct_mult,
            len(self.rotation_trace),
        )

    def record_stage(
        self,
        name: str,
        before: tuple[int, int, int, int],
        runtime_seconds: float = 0.0,
        level_before: int | None = None,
        level_after: int | None = None,
    ) -> None:
        r0, cp0, cc0, trace_start = before
        amounts = sorted(set(self.rotation_trace[trace_start:]))
        self.stages[name] = {
            "rotations": self.rotations - r0,
            "unique_rotation_amounts": len(amounts),
            "rotation_amounts": amounts,
            "ct_pt_mult": self.ct_pt_mult - cp0,
            "ct_ct_mult": self.ct_ct_mult - cc0,
            "runtime_seconds": runtime_seconds,
            "level_before": level_before,
            "level_after": level_after,
        }

    def summary(self) -> str:
        # amounts = sorted(self.rotation_amounts)
        return (
            f"rotations={self.rotations}, "
            # f"unique_rotation_amounts={self.unique_rotations}, "
            # f"rotation_amounts={amounts}, "
            f"ct-pt={self.ct_pt_mult}, "
            f"ct-ct={self.ct_ct_mult}, "
            f"runtime={self.total_runtime_seconds:.6f}s"
        )


counter = Counter()