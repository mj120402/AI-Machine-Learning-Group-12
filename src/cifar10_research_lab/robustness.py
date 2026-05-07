from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable


@dataclass(frozen=True)
class CorruptionSpec:
    name: str
    description: str
    kind: str
    severity: float


def clamp_pixel(value: float) -> float:
    return min(1.0, max(0.0, float(value)))


def build_corruption_plan(levels: Iterable[float] = (0.05, 0.15, 0.30)) -> list[CorruptionSpec]:
    """Return the corruptions used to test model robustness."""

    plan: list[CorruptionSpec] = []
    for level in levels:
        severity = float(level)
        label = f"{severity:.2f}"
        plan.extend(
            [
                CorruptionSpec(
                    name=f"gaussian_noise_{label}",
                    description=f"Gaussian pixel noise with severity {label}",
                    kind="gaussian_noise",
                    severity=severity,
                ),
                CorruptionSpec(
                    name=f"brightness_shift_{label}",
                    description=f"Brightness shift with severity {label}",
                    kind="brightness_shift",
                    severity=severity,
                ),
                CorruptionSpec(
                    name=f"center_occlusion_{label}",
                    description=f"Central square occlusion with severity {label}",
                    kind="center_occlusion",
                    severity=severity,
                ),
            ]
        )
    return plan


def apply_corruption(images, spec: CorruptionSpec, seed: int = 42):
    """Apply one corruption to a batch of 0-1 images."""

    import numpy as np

    corrupted = np.array(images, copy=True)
    if spec.kind == "gaussian_noise":
        rng = np.random.default_rng(seed)
        noise = rng.normal(loc=0.0, scale=spec.severity, size=corrupted.shape)
        return np.clip(corrupted + noise, 0.0, 1.0)

    if spec.kind == "brightness_shift":
        return np.clip(corrupted + spec.severity, 0.0, 1.0)

    if spec.kind == "center_occlusion":
        height, width = corrupted.shape[1], corrupted.shape[2]
        square_size = max(2, int(round(min(height, width) * (0.20 + spec.severity))))
        row_start = (height - square_size) // 2
        col_start = (width - square_size) // 2
        corrupted[
            :,
            row_start : row_start + square_size,
            col_start : col_start + square_size,
            :,
        ] = 0.5
        return corrupted

    raise ValueError(f"Unknown corruption kind: {spec.kind}")


def evaluate_robustness(model, x_test, y_test, *, sample_count: int, seed: int):
    """Measure how accuracy changes when the same images are corrupted."""

    x_subset = x_test[:sample_count]
    y_subset = y_test[:sample_count]
    results: list[dict[str, object]] = []

    for spec in build_corruption_plan():
        corrupted_images = apply_corruption(x_subset, spec, seed=seed)
        loss, accuracy = model.evaluate(corrupted_images, y_subset, verbose=0)
        results.append(
            {
                "corruption": spec.name,
                "description": spec.description,
                "loss": float(loss),
                "accuracy": float(accuracy),
            }
        )

    return results
