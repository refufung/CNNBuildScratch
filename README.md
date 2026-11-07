# CNN Build from Scratch

Train a CNN with Squeeze-and-Excitation (`CNNSE`) from scratch for 4-class dog breed classification:

- French Bulldog
- Cardigan
- Labrador Retriever
- Golden Retriever

## Layout

```
CNNBuildScratch/
  CNN_se.ipynb          # notebook workflow
  src/
    model.py            # CNNSE architecture
    dataset.py          # dataset + transforms
    prepare_split.py    # build train/val/test CSVs
    train.py            # training, TTA, ensemble eval
  outputs/              # metrics, plots, split CSVs (checkpoints excluded)
```

## Setup

```bash
cd CNNBuildScratch
python -m venv .venv
# Windows:
.venv\Scripts\activate
pip install -r ../requirements.txt
```

Place your cleaned images under `CNNBuildScratch/input/Cleaned Dataset/` (train/test folders), then prepare splits and train:

```bash
cd src
python prepare_split.py
python train.py --smoke                      # quick sanity check
python train.py --epochs 60 --seeds 42       # short run
python train.py --epochs 150 --seeds 42 1337 2025   # full multi-seed
python train.py --eval-only --seeds 42 1337 2025    # TTA + ensemble from checkpoints
```

## Notes

- Training checkpoints (`*.pt`) and the raw image dataset are **not** included in this repo (size limits).
- Metrics, confusion matrices, and training curves under `outputs/` are included when present.
