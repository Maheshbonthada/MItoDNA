# MitoSeqGen

Research-grade project scaffold for mitochondrial codon-aware mRNA sequence generation.

## Project structure

- `config/` - experiment configuration and hyperparameter overrides
- `data/raw/` - raw downloads, never modified
- `data/processed/` - QCed and preprocessed data
- `data/splits/` - phylogenetic train/val/test split files
- `data/qc_reports/` - dataset audit JSONs for supplementary material
- `src/` - core codebase
- `src/data/` - dataset creation and preprocessing
- `src/models/` - model architectures
- `src/training/` - training scripts and loss functions
- `src/evaluation/` - metrics and reviewer-ready results
- `checkpoints/` - model weights and best checkpoints (not tracked in git due to file size; trained weights are published at https://huggingface.co/Sravankumarbonthada/mitoseqgen)
- `paper/` - figure and supplementary data output
- `webapp/` - dashboard for training/evaluation results plus a live CPU-inference demo (see `webapp/README.md`)

## Getting started

1. Install dependencies: `pip install -r requirements.txt`
2. Run dataset QC: `python src/data/qc.py`
3. Preprocess and tokenize: `python src/data/preprocess.py`
4. Train the model: `python src/training/train.py`
5. Evaluate outputs: `python src/evaluation/evaluate.py`
