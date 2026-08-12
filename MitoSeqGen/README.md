# MitoSeqGen

Research-grade project scaffold for mitochondrial codon-aware mRNA sequence generation.

## Project structure

- `config/` - experiment configuration and hyperparameter overrides
- `data/` - raw downloads, QCed/preprocessed data, splits, and QC audit JSONs (not tracked in git due to file size; the curated dataset is published at https://huggingface.co/datasets/Sravankumarbonthada/mitoseqgen-dataset)
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
5. Evaluate outputs on the held-out test split: `python src/evaluation/evaluate.py` (uses `data/splits/test.json`, downloadable from the [curated dataset on Hugging Face](https://huggingface.co/datasets/Sravankumarbonthada/mitoseqgen-dataset); reproduces Tables 1-2 and the significance tests reported in the paper)

## Example usage

Given a target protein sequence, `generate_cds()` (`src/models/generate.py`) autoregressively decodes a codon-optimized CDS, one codon per residue plus a final stop codon, guaranteed protein-exact and stop-codon-free by construction:

```python
import torch
from pathlib import Path
from src.models.generate import generate_cds, load_model_from_checkpoint

device = torch.device("cpu")
model = load_model_from_checkpoint(Path("experiments/20260722_140803/checkpoints/best.pt"), device)

protein = "MPQLDTIYILTVYLWAWLILHQMMQKTKTMLMTTPPQKHIITNKMMTPPTWL"  # ATP8, Crotaphopeltis hotamboeia (held-out test-set example, record XOB86100.1)
cds = generate_cds(model, protein, device, strategy="greedy")
print(cds)
```

Output (real output from the checkpoint used throughout the paper, on the held-out test-set example above):

```
ATGCCACAACTAGACACAATCTACATCTTAACTGTATACCTATGAGCCTGATTAATTCTACACCAAATAATACAAAAAACAAAAACAATACTAATAACAACACCACCACAAAAACACATCATCACCAATAAAATAATAACACCCCCAACATGACTATAA
```

53 codons (52 residues + stop), 159 nucleotides. Translating this output under the vertebrate mitochondrial genetic code reproduces the input protein exactly, with no internal stop codons, by construction.
