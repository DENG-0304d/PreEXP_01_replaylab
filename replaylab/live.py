"""Interface reserved for M2 independent closed-loop replay."""

from pathlib import Path
from typing import Protocol, TypedDict


class LiveRunSpec(TypedDict):
    """冻结研究和计划文件。计划指定 baseline_live 或 inflight_live 的独立回合。"""

    study_path: Path
    plan_path: Path
    output_dir: Path


class LiveReplay(Protocol):
    """分别用两组实时证据驱动同一个冻结动作。"""

    def run(self, spec: LiveRunSpec) -> Path:
        """在独立回合中记录真实触发、启用、派发、全池与 B 既有群证据，返回清单路径。"""
        ...
