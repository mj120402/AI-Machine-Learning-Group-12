# Coursework Research Guide

## Working Title

Robust and Interpretable CIFAR-10 Image Classification with Convolutional
Neural Networks

## Research Question

How well do compact convolutional neural networks generalise on CIFAR-10, and
what do robustness, per-class performance, and Grad-CAM visualisations reveal
about their strengths and weaknesses?

## Why This Is More Interesting Than A Basic CNN

A basic image-classification assignment usually reports training accuracy,
validation accuracy, and test accuracy. This project goes further by asking
whether the model is reliable and explainable:

- Does the model fail more often on visually similar classes, such as cats and
  dogs or automobiles and trucks?
- Does accuracy fall sharply when images become noisy, brighter, or partially
  occluded?
- Do Grad-CAM heatmaps focus on the object, or on irrelevant background areas?
- Does a residual architecture improve accuracy or stability compared with the
  custom CNN?

## Hypotheses

1. The residual model should improve validation and test accuracy because skip
   connections make deeper feature extraction easier to train.
2. Stronger augmentation should improve robustness, especially against mild
   noise and brightness shift.
3. The confusion matrix should show more mistakes between visually similar
   animal classes than between highly distinct classes such as ship and frog.
4. Grad-CAM examples should reveal whether correct predictions are based on
   object-like image regions rather than background texture.

## Methodology

Use the same train/validation/test split, random seed, batch size, and epoch
budget for each experiment. Save the best checkpoint by validation accuracy,
then evaluate that checkpoint on the test set.

The main experiment commands are:

```powershell
.\.venv311\Scripts\python .\cifar10_cnn.py --architecture research_cnn --epochs 30 --strong-augmentation --output-dir artifacts_cnn
.\.venv311\Scripts\python .\cifar10_cnn.py --architecture research_cnn --epochs 45 --strong-augmentation --output-dir artifacts_cnn_45
.\.venv311\Scripts\python .\cifar10_cnn.py --architecture research_cnn --epochs 30 --strong-augmentation --noise-augmentation --label-smoothing 0.05 --output-dir artifacts_cnn_robust
.\.venv311\Scripts\python .\cifar10_cnn.py --architecture wide_cnn --epochs 30 --strong-augmentation --output-dir artifacts_wide_cnn
.\.venv311\Scripts\python .\cifar10_cnn.py --architecture resnet_small --epochs 30 --strong-augmentation --output-dir artifacts_resnet
```

## Observed Results

The strongest clean-accuracy result came from training the original
`research_cnn` for 45 epochs instead of 30. The architecture did not change; the
extra epochs allowed the learning-rate schedule to reduce the step size several
times and fine-tune the weights.

| Run | Best validation accuracy | Test accuracy | Test loss | Best epoch |
| --- | ---: | ---: | ---: | ---: |
| `artifacts_cnn` | `0.7976` | `0.7936` | `0.7579` | `27` |
| `artifacts_cnn_45` | `0.8256` | `0.8134` | `0.6764` | `45` |
| `artifacts_cnn_robust` | `0.7930` | `0.7905` | `0.9256` | `30` |

This supports a useful coursework conclusion: the model was not fully saturated
at 30 epochs. Longer fine-tuning improved test accuracy by about 1.98 percentage
points while also reducing test loss. Because the best validation score occurred
at epoch 45, more training may still improve accuracy slightly, but the gains
are likely to be incremental. A larger jump would probably require a better
architecture, stronger augmentation such as MixUp/CutMix, or more compute.

## Report Narrative

### Model Design

The model-design section explains the architecture choices: convolutional layers
for spatial features, batch normalisation for stable training, dropout for
regularisation, global average pooling to reduce parameters, and residual
connections for the comparison model. The `wide_cnn` model tests whether extra
feature channels improve accuracy, while the longer `research_cnn` run tests
whether the same model simply needed more fine-tuning after the learning-rate
reductions.

### Training And Validation

The training discussion uses `training_curves.png` to explain convergence. If
validation accuracy stops improving while training accuracy continues to rise,
that points to overfitting. If both curves improve together, that supports
healthy generalisation.

### Final Metrics

The final metrics come from `metrics.json` or the `=== REPORT METRICS ===`
console block. The reported test accuracy and loss are taken from the best
validation checkpoint.

### Error Analysis

The error analysis uses `confusion_matrix.png` and `per_class_accuracy.json`.
The most important classes to discuss are the ones with the lowest accuracy,
because they show where the classifier struggles most.

### Robustness

The robustness section uses `robustness.png` and `robustness.json` to show which
corruptions cause the largest accuracy drops and what that suggests about the
model.

The `artifacts_cnn_robust` run is designed to test whether training with mild
Gaussian noise and label smoothing improves noise robustness without losing too
much clean test accuracy.

### Interpretability

The interpretability section uses `gradcam_examples.png`. A few correct and
incorrect predictions can be compared to explain whether the heatmap supports or
challenges the predicted class.

## Limitations

- CIFAR-10 images are small, so fine object detail is limited.
- A single train/test split does not prove the result is universal.
- Robustness corruptions are simulated and do not cover every real-world image
  problem.
- Grad-CAM is explanatory, but it is not a full proof of model reasoning.

## Possible Further Work

- Add MixUp or CutMix augmentation.
- Add calibration metrics such as expected calibration error.
- Compare model size, inference speed, and accuracy.
- Train on corrupted examples and test whether robustness improves.
- Add active learning by training first on a small labelled subset, then adding
  the most uncertain images.
- Test cutout augmentation by adding `--cutout-augmentation` to promising runs.
