# Beam Prediction

Experiments for beam prediction using camera images, positional/tabular
features, and deep-learning models.

## Contents

- `beam_predict_final_paper.ipynb`: main end-to-end experiment notebook
- `improve_tabular_models.py`: reusable tabular preprocessing, training, and
  evaluation utilities
- `ca_former*.ipynb`: CAFormer experiments
- `traditional_model.ipynb`: traditional model experiments
- `image*.ipynb`: image-based experiments

## Environment

The notebooks are designed for a Python/Jupyter environment with the project
dataset available locally or in Google Colab. Core dependencies include
NumPy, pandas, PyTorch, torchvision, scikit-learn, Matplotlib, Seaborn, and
Jupyter.

Dataset paths used by the notebooks may need to be updated for your
environment before execution.

## Usage

Open a notebook in Jupyter or Google Colab and run its cells in order. For the
main workflow, start with `beam_predict_final_paper.ipynb`.

Generated datasets, model checkpoints, notebook checkpoints, and Python cache
files are excluded from Git.
