# CIFAR-10 Research Lab

This project trains and analyses convolutional neural networks on CIFAR-10.
It started as a single CNN training script, but is now organised like a small
research/coursework project with reproducible configuration, validation,
visualisations, robustness checks, and report-ready artifacts.

## Research Idea

The core question is:

> How well does a CNN generalise on CIFAR-10, and what can class-level,
> robustness, and interpretability analysis tell us about its mistakes?

That gives the coursework more substance than simply reporting one final test
accuracy. The project supports a clear discussion of model design, training
behaviour, failure modes, and practical reliability.

## Best Observed Result

The best run so far is the 45-epoch `research_cnn` experiment. The curated
output for that run is saved in `docs/results` so the coursework evidence is
included when the repository is zipped. It used the same architecture as the
30-epoch baseline, but gave the learning-rate scheduler more time to fine-tune
the model.

| Run | Best validation accuracy | Test accuracy | Test loss | Best epoch |
| --- | ---: | ---: | ---: | ---: |
| `artifacts_cnn` | `0.7976` | `0.7936` | `0.7579` | `27` |
| `artifacts_cnn_45` | `0.8256` | `0.8134` | `0.6764` | `45` |

The extra training improved test accuracy by about 1.98 percentage points. Since
the best validation score happened at the final epoch, more training could still
give a small improvement, but the later gains are already slowing down. For the
coursework report, it is reasonable to conclude that longer fine-tuning improves
this model, while much larger gains would likely need a stronger architecture or
additional augmentation methods rather than just many more epochs.

## What Is Included

- Three architectures:
  - `research_cnn`: a compact custom CNN with batch normalisation, dropout,
    global average pooling, and optional stronger augmentation.
  - `wide_cnn`: a higher-capacity VGG-style CNN for testing whether a wider
    feature extractor improves clean accuracy.
  - `resnet_small`: a small residual CNN for comparison.
- Reproducible CLI configuration.
- Stratified validation splitting.
- Best-checkpoint saving based on validation accuracy.
- Early stopping and learning-rate reduction.
- Report metrics saved to JSON and Markdown.
- Training-curve, confusion-matrix, prediction-grid, robustness, and Grad-CAM
  visualisations.
- Robustness checks using noise, brightness shift, and central occlusion.
- Optional Gaussian-noise training augmentation and label smoothing.
- Optional cutout-style random erasing augmentation.
- Standard-library unit tests for the project logic that does not require
  TensorFlow.

## Project Structure

```text
.
+-- cifar10_cnn.py                 # Backwards-compatible runner
+-- src/cifar10_research_lab/
|   +-- config.py                  # CLI/env configuration
|   +-- data.py                    # CIFAR-10 loading and splits
|   +-- models.py                  # CNN and residual architectures
|   +-- train.py                   # Main experiment workflow
|   +-- reporting.py               # Metrics and report writing
|   +-- visualization.py           # PNG artifact generation
|   +-- robustness.py              # Corrupted-image validation
|   +-- interpretability.py        # Grad-CAM examples
+-- tests/                         # Lightweight verification tests
+-- docs/coursework_research_guide.md
```

## Setup

TensorFlow support depends on Python version. Python 3.11 is recommended.

Windows PowerShell:

```powershell
py -3.11 -m venv .venv311
.\.venv311\Scripts\python -m pip install --upgrade pip
.\.venv311\Scripts\python -m pip install -e .
```

macOS/Linux:

```bash
python3.11 -m venv .venv311
./.venv311/bin/python -m pip install --upgrade pip
./.venv311/bin/python -m pip install -e .
```

## Smoke Test

Use this before a full run. It trains one epoch on smaller data and skips heavy
plot generation.

Windows:

```powershell
$env:SMOKE_TEST='1'
$env:NO_PLOTS='1'
.\.venv311\Scripts\python .\cifar10_cnn.py --skip-robustness --skip-gradcam
```

macOS/Linux:

```bash
SMOKE_TEST=1 NO_PLOTS=1 ./.venv311/bin/python ./cifar10_cnn.py --skip-robustness --skip-gradcam
```

## Full Research Runs

Train the upgraded custom CNN:

```powershell
.\.venv311\Scripts\python .\cifar10_cnn.py --architecture research_cnn --epochs 30 --strong-augmentation
```

Train the same model for longer with early stopping and learning-rate reduction:

```powershell
.\.venv311\Scripts\python .\cifar10_cnn.py --architecture research_cnn --epochs 45 --strong-augmentation --output-dir artifacts_cnn_45
```

Train the same model with robustness-oriented settings:

```powershell
.\.venv311\Scripts\python .\cifar10_cnn.py --architecture research_cnn --epochs 30 --strong-augmentation --noise-augmentation --label-smoothing 0.05 --output-dir artifacts_cnn_robust
```

Train the wider CNN comparison model:

```powershell
.\.venv311\Scripts\python .\cifar10_cnn.py --architecture wide_cnn --epochs 30 --strong-augmentation --output-dir artifacts_wide_cnn
```

Train the residual comparison model:

```powershell
.\.venv311\Scripts\python .\cifar10_cnn.py --architecture resnet_small --epochs 30 --strong-augmentation --output-dir artifacts_resnet
```

The same command also works through the package module:

```bash
python -m cifar10_research_lab --architecture research_cnn --epochs 30
```

## Saved Artifacts

Each run writes artifacts into the chosen output directory, usually `artifacts/`:

- `config.json`: exact run configuration.
- `metrics.json`: final train/validation/test metrics and best validation epoch.
- `per_class_accuracy.json`: class-by-class accuracy summary.
- `confusion_matrix.json`: raw confusion matrix values.
- `robustness.json`: accuracy/loss under corrupted inputs.
- `research_report.md`: report-ready summary.
- `training_log.csv`: epoch-by-epoch training log.
- `checkpoints/<architecture>_best.weights.h5`: best validation checkpoint.
- `plots/training_curves.png`
- `plots/confusion_matrix.png`
- `plots/per_class_accuracy.png`
- `plots/prediction_examples.png`
- `plots/robustness.png`
- `plots/gradcam_examples.png`

## Validation Strategy

The validation strategy for the coursework is:

1. Use validation accuracy for checkpoint selection.
2. Keep test accuracy for final unseen-data evaluation.
3. Use per-class accuracy and the confusion matrix to explain where the model
   succeeds or fails.
4. Use robustness checks to test whether performance survives common image
   corruptions.
5. Use Grad-CAM examples to connect predictions to image regions.
6. Compare `research_cnn`, `wide_cnn`, and `resnet_small` using the same seed,
   epochs, and output artifacts.

## Tests

The included tests check configuration, reporting, data helper logic, and
robustness-plan generation without needing TensorFlow installed:

```powershell
$env:PYTHONPATH='src'
python -m unittest discover -s tests -v
```

## Troubleshooting

- `ModuleNotFoundError: No module named 'tensorflow'`
  Use the virtual environment and install the project with `pip install -e .`.

- `No matching distribution found for tensorflow`
  Use Python 3.11. Some newer Python versions are not supported by TensorFlow.

- Plot windows do not appear
  This project saves plots as PNG artifacts instead of opening interactive
  windows. Check the `plots/` folder inside the output directory.
