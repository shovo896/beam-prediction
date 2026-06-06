# Beam Prediction Figure Gallery

All figures below use the final saved test metrics. The image-model figures use
the refined checkpoint result (Top-1: 88.32%), not the initial 87.62% result.
Run `python generate_ieee_comparison_figure.py` to regenerate every figure.

## Complete Findings and Reference Comparison

![Complete findings and direct reference comparison](figures/ieee_findings_comparison.png)

## All Model Test Accuracies

![Final Top-k test accuracy for every implemented modality](figures/all_model_test_accuracy.png)

## Top-1 Modality Comparison

![Top-1 test accuracy by sensing modality](figures/top1_modality_comparison.png)

## Image Model Refinement

![Initial and refined image model compared with the reference paper](figures/image_model_refinement.png)

Training curves and confusion matrices remain embedded in
`beam_predict_final_paper.ipynb` because they depend on per-epoch histories and
per-sample predictions. They are also exported below for convenient review.
Run `python export_notebook_figures.py` to refresh them from the notebook.
The initial image-model chart is a pre-refinement result.

## Notebook Diagnostic Images

### Initial Image Model, Before Refinement

![Initial image model test accuracy before refinement](figures/notebook_diagnostics/initial_image_model_test_accuracy.png)

### Position Model

![Position model training loss](figures/notebook_diagnostics/position_training_loss.png)

![Position model validation accuracy](figures/notebook_diagnostics/position_validation_accuracy.png)

![Position model test accuracy](figures/notebook_diagnostics/position_test_accuracy.png)

![Position model confusion matrix](figures/notebook_diagnostics/position_confusion_matrix.png)

### Position and Height Model

![Position and height training, validation, and test results](figures/notebook_diagnostics/position_height_training_validation_test.png)

![Position and height confusion matrix](figures/notebook_diagnostics/position_height_confusion_matrix.png)

### Position, Height, and Distance Model

![Position, height, and distance training, validation, and test results](figures/notebook_diagnostics/position_height_distance_training_validation_test.png)

![Position, height, and distance confusion matrix](figures/notebook_diagnostics/position_height_distance_confusion_matrix.png)

### Position-Family Comparison

![Position-family test accuracy comparison](figures/notebook_diagnostics/position_modality_test_accuracy_comparison.png)
