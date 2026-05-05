# CIFAR-10 Deep CNN (TensorFlow/Keras)

Train a deeper CNN on CIFAR-10 with TensorFlow/Keras.

The script `cifar10_cnn.py` includes:
- data loading + preprocessing
- augmentation
- training + validation
- `ModelCheckpoint` (saves best validation model)
- reload best weights before test evaluation
- report metrics output
- optional plots/prediction examples

## Quick Start

From the folder containing `cifar10_cnn.py`:

### 1) Create environment and install dependencies

- Windows (PowerShell):
```powershell
py -3.11 -m venv .venv311
.\.venv311\Scripts\python -m pip install --upgrade pip
.\.venv311\Scripts\python -m pip install tensorflow numpy matplotlib
```

- macOS (Terminal):
```bash
python3.11 -m venv .venv311
./.venv311/bin/python -m pip install --upgrade pip
./.venv311/bin/python -m pip install tensorflow numpy matplotlib
```

### 2) Smoke test (fast check)

- Windows:
```powershell
$env:SMOKE_TEST='1'
$env:NO_PLOTS='1'
.\.venv311\Scripts\python .\cifar10_cnn.py
```

- macOS:
```bash
SMOKE_TEST=1 NO_PLOTS=1 ./.venv311/bin/python ./cifar10_cnn.py
```

### 3) Full training run

- Windows:
```powershell
$env:SMOKE_TEST='0'
$env:NO_PLOTS='0'
.\.venv311\Scripts\python .\cifar10_cnn.py
```

- macOS:
```bash
SMOKE_TEST=0 NO_PLOTS=0 ./.venv311/bin/python ./cifar10_cnn.py
```

## What to collect for your report

In console output, keep the `=== REPORT METRICS ===` block:
- final train accuracy/loss
- final validation accuracy/loss
- best validation epoch + accuracy
- final test accuracy/loss

Also keep:
- training/validation plots
- prediction example grid

Saved model checkpoint:
- `best_model.weights.h5` (best validation accuracy)

## Useful variants

Run without plots:
- Windows:
```powershell
$env:SMOKE_TEST='0'
$env:NO_PLOTS='1'
.\.venv311\Scripts\python .\cifar10_cnn.py
```
- macOS:
```bash
SMOKE_TEST=0 NO_PLOTS=1 ./.venv311/bin/python ./cifar10_cnn.py
```

Unset env vars:
- Windows:
```powershell
Remove-Item Env:SMOKE_TEST -ErrorAction SilentlyContinue
Remove-Item Env:NO_PLOTS -ErrorAction SilentlyContinue
```
- macOS:
```bash
unset SMOKE_TEST
unset NO_PLOTS
```

## Troubleshooting

- `ModuleNotFoundError: No module named 'tensorflow'`  
  Use the venv interpreter directly:
  - Windows: `.\.venv311\Scripts\python .\cifar10_cnn.py`
  - macOS: `./.venv311/bin/python ./cifar10_cnn.py`

- `No matching distribution found for tensorflow`  
  You are likely on an unsupported Python version. Use Python 3.11.

- PowerShell script execution policy error  
  Activation is optional; run with direct venv python path instead.

