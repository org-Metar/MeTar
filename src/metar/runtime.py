"""Small terminal reports and consistent CPU/accelerator selection.

Progress uses values already computed by the caller. It never loads a model,
evaluates a batch, copies a tensor or synchronizes an accelerator.
"""

from __future__ import annotations

import math
import numbers
import os
import sys
import time
from collections.abc import Mapping
from contextvars import ContextVar
from dataclasses import dataclass
from typing import TextIO


def resolve_device(requested: str = "auto") -> str:
    """Resolve a device once; an explicit accelerator never silently becomes CPU."""
    import torch

    value = str(requested).strip().lower()
    if value == "auto":
        if torch.cuda.is_available() and torch.cuda.device_count() > 0:
            return "cuda:0"
        return "cpu"
    try:
        device = torch.device(value)
    except (RuntimeError, ValueError, TypeError) as exc:
        raise ValueError(
            f"Invalid device {requested!r}; use auto, cpu, cuda or cuda:N."
        ) from exc
    if device.type == "cpu":
        if device.index not in (None, 0):
            raise ValueError("CPU has no selectable index; use --device cpu.")
        return "cpu"
    if device.type == "cuda":
        if not torch.cuda.is_available():
            build = (
                "a CPU-only Torch build"
                if torch.version.cuda is None
                else "no usable CUDA driver/device"
            )
            raise RuntimeError(
                f"CUDA was requested ({requested}), but this environment has {build}. "
                "Install CUDA-enabled PyTorch compatible with your NVIDIA driver, "
                "or select --device cpu / --device auto. "
                f"Python: {sys.executable}; Torch: {torch.__version__}."
            )
        count = torch.cuda.device_count()
        index = torch.cuda.current_device() if device.index is None else device.index
        if index < 0 or index >= count:
            raise ValueError(
                f"CUDA index {index} is unavailable; {count} device(s) are visible. "
                "Indices are relative to CUDA_VISIBLE_DEVICES; use --device cuda:0 "
                "for the first visible GPU."
            )
        return f"cuda:{index}"
    if device.type == "mps":
        mps = getattr(torch.backends, "mps", None)
        if mps is None or not mps.is_available():
            raise RuntimeError(
                "MPS was requested but is unavailable; use --device cpu or auto."
            )
        if device.index not in (None, 0):
            raise ValueError("Use --device mps without a device index.")
        return "mps"
    raise ValueError(f"Unsupported device {requested!r}; use auto, cpu, cuda:N or mps.")


def _value(value: object) -> str:
    if isinstance(value, numbers.Integral):
        return str(value)
    if isinstance(value, numbers.Real):
        scalar = float(value)
        return f"{scalar:.4g}" if math.isfinite(scalar) else str(scalar)
    return str(value).replace("\r", " ").replace("\n", " ")


@dataclass
class _Progress:
    current: int
    total: int
    printed_at: float


_ACTIVE: ContextVar[RunReporter | None] = ContextVar("metar_run_reporter", default=None)
_FALLBACK: RunReporter | None = None


class RunReporter:
    """Readable stderr reports, with throttled bars in terminals and plain logs."""

    def __init__(
        self,
        title: str,
        settings: Mapping[str, object] | None = None,
        *,
        quiet: bool = False,
        stream: TextIO | None = None,
    ) -> None:
        self.title = title
        self.settings = dict(settings or {})
        self.quiet = quiet
        self.stream = stream
        self.started = time.monotonic()
        self.steps = 0
        self.closed = False
        self.progress_states: dict[str, _Progress] = {}
        self._bar_open = False
        self._token = None

    @property
    def output(self) -> TextIO:
        return self.stream if self.stream is not None else sys.stderr

    @property
    def enabled(self) -> bool:
        return not self.quiet and os.environ.get("METAR_PROGRESS", "1").lower() not in {
            "0",
            "false",
            "off",
            "quiet",
        }

    def _line(self, text: str) -> None:
        if not self.enabled:
            return
        if self._bar_open:
            self.output.write("\n")
            self._bar_open = False
        print(text, file=self.output, flush=True)

    def __enter__(self) -> RunReporter:  # noqa: PYI034 -- Python 3.10 compatibility.
        self.started = time.monotonic()
        self._token = _ACTIVE.set(self)
        self._line(f"MeTar | {self.title}")
        for key, value in self.settings.items():
            if value is not None:
                self._line(f"  {key}: {_value(value)}")
        return self

    def __exit__(self, exc_type, exc, traceback) -> bool:
        try:
            if not self.closed:
                if exc is None or (
                    isinstance(exc, SystemExit) and exc.code in (0, None)
                ):
                    self.finish()
                else:
                    self.fail(str(exc) or exc_type.__name__)
        finally:
            if self._token is not None:
                _ACTIVE.reset(self._token)
        return False

    def step(self, message: str) -> None:
        self.steps += 1
        self._line(f"[Step {self.steps}] {message}")

    def metrics(self, metrics: Mapping[str, object], *, label: str = "Metrics") -> None:
        if not self.enabled:
            return
        text = " | ".join(f"{key}={_value(value)}" for key, value in metrics.items())
        self._line(f"[{label}] {text}")

    def progress(
        self,
        current: int,
        total: int,
        *,
        label: str,
        metrics: Mapping[str, object] | None = None,
    ) -> None:
        if not self.enabled:
            return
        if current < 0 or total < 0 or current > total:
            raise ValueError(f"Invalid progress for {label}: {current}/{total}")
        now = time.monotonic()
        terminal = bool(getattr(self.output, "isatty", lambda: False)())
        previous = self.progress_states.get(label)
        restarted = (
            previous is None or current < previous.current or total != previous.total
        )
        interval = 0.5 if terminal else 15.0
        if not restarted and current != total and now - previous.printed_at < interval:
            return
        if (
            previous is not None
            and current == total == previous.current
            and total == previous.total
        ):
            return
        self.progress_states[label] = _Progress(current, total, now)
        fraction = current / total if total else 0.0
        filled = int(fraction * 20)
        bar = "#" * filled + "-" * (20 - filled)
        text = f"[{bar}] {label}: {current}/{total} ({fraction:.0%})"
        if metrics:
            text += " | " + " | ".join(
                f"{key}={_value(value)}" for key, value in metrics.items()
            )
        if terminal:
            self.output.write("\r" + text + "\033[K")
            self.output.flush()
            self._bar_open = True
            if current == total:
                self.output.write("\n")
                self._bar_open = False
        else:
            self._line(text)

    def finish(self, **summary: object) -> None:
        if self.closed:
            return
        self.closed = True
        if summary:
            self.metrics(summary, label="Summary")
        self._line(f"[Done] {self.title} | {time.monotonic() - self.started:.1f}s")

    def fail(self, message: str) -> None:
        if self.closed:
            return
        self.closed = True
        self._line(f"[Failed] {self.title} | {message}")


def _reporter() -> RunReporter:
    global _FALLBACK
    active = _ACTIVE.get()
    if active is not None:
        return active
    if _FALLBACK is None:
        _FALLBACK = RunReporter("Runtime")
    return _FALLBACK


def report_step(message: str) -> None:
    _reporter().step(message)


def report_progress(
    current: int,
    total: int,
    *,
    label: str,
    metrics: Mapping[str, object] | None = None,
) -> None:
    _reporter().progress(current, total, label=label, metrics=metrics)


def report_metrics(metrics: Mapping[str, object], *, label: str = "Metrics") -> None:
    _reporter().metrics(metrics, label=label)
