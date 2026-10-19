# Multi-Metric Explainability

A small MNIST feature-importance project combining sensitivity, robustness, and counterfactual metrics. Rankings are empirical, not formal verification guarantees.

## Setup (Windows PowerShell)

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
python models/train.py
python main.py
```

## Shared interface

Each metric defines `score(model, image)` where `image` is a float tensor of shape `(1, 1, 28, 28)` with values in `[0, 1]`. It returns a `torch.Tensor` of shape `(49,)`, scores between 0 and 1, higher meaning more important. Region index `row * 7 + col` corresponds to the 4x4 block at row, col of the 7x7 region grid.

Use the saved `models/mnist_model.pth` checkpoint for all comparisons. Keep the model in evaluation mode. Start with the same test images and record experimental settings.

## Responsibilities

- `metrics/sensitivity.py`: confidence change when a region is set to zero (working example).
- `metrics/robustness.py`: test changes of increasing magnitude; convert vulnerability into an importance score.
- `metrics/counterfactual.py`: quantify how changing a region pushes predictions toward an alternative class.

## Next steps

Implement the two remaining metrics, combine normalized outputs with adjustable weights, add heatmaps, and evaluate on held-out perturbations and a random-ranking baseline. Avoid claiming formal guarantees from sampled perturbations.
