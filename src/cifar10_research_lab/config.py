from __future__ import annotations

import argparse
import os
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable, Mapping


TRUTHY_VALUES = {"1", "true", "t", "yes", "y", "on"}
FALSEY_VALUES = {"0", "false", "f", "no", "n", "off", ""}


def parse_bool(value: object) -> bool:
    """Parse common environment-variable style booleans."""

    if value is None:
        return False
    if isinstance(value, bool):
        return value

    normalised = str(value).strip().lower()
    if normalised in TRUTHY_VALUES:
        return True
    if normalised in FALSEY_VALUES:
        return False

    raise ValueError(f"Cannot parse boolean value: {value!r}")


@dataclass(frozen=True)
class ExperimentConfig:
    """All settings needed to reproduce a run."""

    architecture: str = "research_cnn"
    requested_epochs: int = 30
    epochs: int = 30
    batch_size: int = 64
    learning_rate: float = 1e-3
    validation_fraction: float = 0.1
    seed: int = 42
    output_dir: Path = Path("artifacts")
    smoke_test: bool = False
    no_plots: bool = False
    strong_augmentation: bool = False
    noise_augmentation: bool = False
    cutout_augmentation: bool = False
    label_smoothing: float = 0.0
    skip_robustness: bool = False
    skip_gradcam: bool = False
    robustness_samples: int = 1000
    smoke_train_size: int = 5000
    smoke_val_size: int = 1000
    smoke_test_size: int = 1000

    def to_json_dict(self) -> dict[str, object]:
        data = asdict(self)
        data["output_dir"] = str(self.output_dir)
        return data


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Train and analyse CIFAR-10 models with metrics, plots, robustness "
            "checks, and Grad-CAM examples."
        )
    )
    parser.add_argument(
        "--architecture",
        choices=("research_cnn", "wide_cnn", "resnet_small"),
        default="research_cnn",
        help="Model family to train.",
    )
    parser.add_argument("--epochs", type=int, default=30, help="Requested training epochs.")
    parser.add_argument("--batch-size", type=int, default=64, help="Training batch size.")
    parser.add_argument("--learning-rate", type=float, default=1e-3, help="Adam learning rate.")
    parser.add_argument(
        "--validation-fraction",
        type=float,
        default=0.1,
        help="Fraction of the training set held out for validation.",
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility.")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("artifacts"),
        help="Directory for checkpoints, metrics, plots, and report files.",
    )
    parser.add_argument("--smoke-test", action="store_true", help="Run a tiny one-epoch check.")
    parser.add_argument("--no-plots", action="store_true", help="Skip PNG visual artifact generation.")
    parser.add_argument(
        "--strong-augmentation",
        action="store_true",
        help="Use a stronger augmentation policy for the research CNN.",
    )
    parser.add_argument(
        "--noise-augmentation",
        action="store_true",
        help="Add GaussianNoise during training to improve corrupted-image robustness.",
    )
    parser.add_argument(
        "--cutout-augmentation",
        action="store_true",
        help="Randomly erase small image patches during training to improve generalisation.",
    )
    parser.add_argument(
        "--label-smoothing",
        type=float,
        default=0.0,
        help="Use categorical label smoothing, for example 0.05 or 0.1.",
    )
    parser.add_argument(
        "--skip-robustness",
        action="store_true",
        help="Skip corrupted-image robustness evaluation.",
    )
    parser.add_argument(
        "--skip-gradcam",
        action="store_true",
        help="Skip Grad-CAM interpretability examples.",
    )
    parser.add_argument(
        "--robustness-samples",
        type=int,
        default=1000,
        help="Number of test images used for robustness checks.",
    )
    return parser


def build_config(
    args: Iterable[str] | None = None,
    environ: Mapping[str, str] | None = None,
) -> ExperimentConfig:
    """Create a reproducible config from CLI arguments and environment variables."""

    environment = os.environ if environ is None else environ
    namespace = _build_parser().parse_args(list(args) if args is not None else None)

    smoke_test = namespace.smoke_test or parse_bool(environment.get("SMOKE_TEST"))
    no_plots = namespace.no_plots or parse_bool(environment.get("NO_PLOTS"))

    requested_epochs = int(namespace.epochs)
    effective_epochs = 1 if smoke_test else requested_epochs

    if not 0 < namespace.validation_fraction < 0.5:
        raise ValueError("--validation-fraction must be greater than 0 and less than 0.5")
    if namespace.batch_size <= 0:
        raise ValueError("--batch-size must be positive")
    if requested_epochs <= 0:
        raise ValueError("--epochs must be positive")
    if namespace.robustness_samples <= 0:
        raise ValueError("--robustness-samples must be positive")
    if not 0.0 <= namespace.label_smoothing <= 0.3:
        raise ValueError("--label-smoothing must be between 0.0 and 0.3")

    return ExperimentConfig(
        architecture=namespace.architecture,
        requested_epochs=requested_epochs,
        epochs=effective_epochs,
        batch_size=namespace.batch_size,
        learning_rate=namespace.learning_rate,
        validation_fraction=namespace.validation_fraction,
        seed=namespace.seed,
        output_dir=namespace.output_dir,
        smoke_test=smoke_test,
        no_plots=no_plots,
        strong_augmentation=namespace.strong_augmentation,
        noise_augmentation=namespace.noise_augmentation,
        cutout_augmentation=namespace.cutout_augmentation,
        label_smoothing=namespace.label_smoothing,
        skip_robustness=namespace.skip_robustness,
        skip_gradcam=namespace.skip_gradcam,
        robustness_samples=namespace.robustness_samples,
    )
