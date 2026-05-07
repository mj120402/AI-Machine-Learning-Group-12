# CIFAR-10 Research Run

## Research Question

How well does the selected CNN architecture generalize on CIFAR-10, and how does its performance change under class-level and robustness analysis?

## Configuration

- Architecture: `research_cnn`
- Epochs used: `45` (requested `45`)
- Batch size: `64`
- Learning rate: `0.001`
- Validation fraction: `0.1`
- Seed: `42`
- Strong augmentation: `True`
- Noise augmentation: `False`
- Cutout augmentation: `False`
- Label smoothing: `0.0`

## Main Metrics

- Final Train Accuracy: `0.8020`
- Final Validation Accuracy: `0.8256`
- Final Train Loss: `0.6905`
- Final Validation Loss: `0.6541`
- Best Validation Accuracy: `0.8256`
- Best Validation Loss: `0.6541`
- Best Validation Epoch: `45`
- Final Test Accuracy: `0.8134`
- Final Test Loss: `0.6764`

## Per-Class Accuracy

- airplane: `0.8450` (845/1000)
- automobile: `0.9370` (937/1000)
- bird: `0.7230` (723/1000)
- cat: `0.5950` (595/1000)
- deer: `0.7440` (744/1000)
- dog: `0.6860` (686/1000)
- frog: `0.9450` (945/1000)
- horse: `0.8470` (847/1000)
- ship: `0.8950` (895/1000)
- truck: `0.9170` (917/1000)

## Robustness Checks

- gaussian_noise_0.05: accuracy `0.4040`, loss `2.4237`
- brightness_shift_0.05: accuracy `0.8170`, loss `0.6416`
- center_occlusion_0.05: accuracy `0.7360`, loss `0.9132`
- gaussian_noise_0.15: accuracy `0.1970`, loss `3.8813`
- brightness_shift_0.15: accuracy `0.8090`, loss `0.6728`
- center_occlusion_0.15: accuracy `0.5900`, loss `1.2948`
- gaussian_noise_0.30: accuracy `0.1360`, loss `4.5833`
- brightness_shift_0.30: accuracy `0.7820`, loss `0.7991`
- center_occlusion_0.30: accuracy `0.3290`, loss `2.5783`

## Saved Artifacts

- checkpoint: `artifacts_cnn_45\checkpoints\research_cnn_best.weights.h5`
- metrics_json: `artifacts_cnn_45\metrics.json`
- per_class_json: `artifacts_cnn_45\per_class_accuracy.json`
- confusion_matrix_json: `artifacts_cnn_45\confusion_matrix.json`
- robustness_json: `artifacts_cnn_45\robustness.json`
- report_markdown: `artifacts_cnn_45\research_report.md`
- history_plot: `artifacts_cnn_45\plots\training_curves.png`
- confusion_matrix_plot: `artifacts_cnn_45\plots\confusion_matrix.png`
- prediction_grid: `artifacts_cnn_45\plots\prediction_examples.png`
- robustness_plot: `artifacts_cnn_45\plots\robustness.png`
- gradcam_grid: `artifacts_cnn_45\plots\gradcam_examples.png`

## How To Explain This In Coursework

Use the training curves to discuss convergence and overfitting, the confusion matrix to discuss class-level failure modes, robustness results to test whether the model relies on fragile image cues, and Grad-CAM examples to connect model decisions back to image regions.
