"""Dataset and transforms for the from-scratch track.

Split CSVs (outputs/{train,val,test}.csv + split_meta.json) are provisioned from
the pre-split `input/Cleaned Dataset` folders (train/validation/test, used as-is).
Missing CSVs are built from `input/`; CSVs whose paths no longer exist (e.g.
created on another machine) are backed up to outputs/backup_prev/ and rebuilt.

Train (strong aug): RandomResizedCrop(0.6-1.0) -> HFlip -> RandAugment(2,9)
  -> ToTensor -> Normalize -> RandomErasing(p=0.25, random fill)
Eval: Resize(224) -> ToTensor -> Normalize
Eval-TTA view: Resize(256) -> CenterCrop(224) -> ToTensor -> Normalize
"""

from __future__ import annotations

import json
import random
import shutil
from pathlib import Path

import pandas as pd
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms

TRACK_DIR = Path(__file__).resolve().parents[1]  # CNNBuildScratch/
OUTPUT_DIR = TRACK_DIR / "outputs"
INPUT_DIR = TRACK_DIR / "input" / "Cleaned Dataset"
IMAGE_SIZE = 224
SEED = 42

IMG_EXTS = {".jpg", ".jpeg", ".png"}
SPLIT_FOLDERS = {"train": "train", "val": "validation", "test": "test"}
# breed folder name (lower-cased) -> (class_name, label); label order matches train.py
BREED_FOLDERS = {
    "french_bulldog": ("French_bulldog", 0),
    "cardigan": ("Cardigan", 1),
    "labrador_retriever": ("Labrador_retriever", 2),
    "golden_retriever": ("golden_retriever", 3),
}


class DogBreedDataset(Dataset):
    def __init__(self, csv_path: Path | str, transform=None):
        self.df = pd.read_csv(csv_path)
        exists = self.df["path"].map(lambda p: Path(p).is_file())
        n_missing = int((~exists).sum())
        if n_missing:
            print(f"Warning: dropping {n_missing} missing paths from {csv_path}")
            self.df = self.df.loc[exists].reset_index(drop=True)
        if len(self.df) == 0:
            raise FileNotFoundError(f"No readable images in {csv_path}")
        self.transform = transform

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int):
        row = self.df.iloc[idx]
        with Image.open(row["path"]) as im:
            image = im.convert("RGB")
        if self.transform is not None:
            image = self.transform(image)
        return image, int(row["label"])


def load_norm_stats(meta_path: Path | None = None) -> tuple[list[float], list[float]]:
    meta_path = meta_path or (OUTPUT_DIR / "split_meta.json")
    with meta_path.open(encoding="utf-8") as f:
        meta = json.load(f)
    return meta["mean"], meta["std"]


def _collect_rows_from_input(seed: int = SEED) -> dict[str, list[dict]]:
    """Index INPUT_DIR -> {split: row dicts}, preserving the official folder splits."""
    rows: dict[str, list[dict]] = {"train": [], "val": [], "test": []}
    for split, folder in SPLIT_FOLDERS.items():
        split_dir = INPUT_DIR / folder
        if not split_dir.is_dir():
            raise FileNotFoundError(split_dir)
        for breed_dir in sorted(p for p in split_dir.iterdir() if p.is_dir()):
            key = breed_dir.name.lower()
            if key not in BREED_FOLDERS:
                print(f"Warning: unexpected folder {breed_dir.name} - skipped")
                continue
            class_name, label = BREED_FOLDERS[key]
            n_ok = n_bad = 0
            for p in sorted(breed_dir.iterdir()):
                if not p.is_file() or p.suffix.lower() not in IMG_EXTS:
                    continue
                try:
                    with Image.open(p) as im:
                        im.convert("RGB")
                except Exception as exc:
                    n_bad += 1
                    print(f"  skip unreadable {p.name}: {exc}")
                    continue
                rows[split].append({"path": str(p.resolve()), "label": label, "class_name": class_name})
                n_ok += 1
            msg = f"  {split}/{breed_dir.name}: {n_ok} usable"
            if n_bad:
                msg += f" ({n_bad} unreadable skipped)"
            print(msg)
    rng = random.Random(seed)
    for split in rows:
        rng.shuffle(rows[split])
    return rows


def _compute_train_mean_std(csv_path: Path, max_samples: int = 512) -> tuple[list[float], list[float]]:
    """Per-channel mean/std on a subsample of the train split (after resize, before normalize)."""
    tf = transforms.Compose([transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)), transforms.ToTensor()])
    ds = DogBreedDataset(csv_path, transform=tf)
    loader = DataLoader(ds, batch_size=32, shuffle=False, num_workers=0)
    sum_ = torch.zeros(3)
    sum_sq = torch.zeros(3)
    count = 0
    n = min(len(ds), max_samples)
    for images, _ in loader:
        b = images.size(0)
        flat = images.view(b, 3, -1)
        sum_ += flat.sum(dim=(0, 2))
        sum_sq += (flat**2).sum(dim=(0, 2))
        count += b * flat.size(2)
        if count // (IMAGE_SIZE * IMAGE_SIZE) >= n:
            break
    mean = (sum_ / count).tolist()
    var = sum_sq / count - torch.tensor(mean) ** 2
    std = torch.sqrt(var.clamp(min=1e-8)).tolist()
    return mean, std


def ensure_split_csvs(seed: int = SEED) -> None:
    """Ensure outputs/{train,val,test}.csv + split_meta.json exist and point at real files.

    The pre-split `input/Cleaned Dataset` folders (train/validation/test) are the
    source of truth. CSVs that are missing, or whose paths no longer exist (e.g.
    written on another machine), are backed up to outputs/backup_prev/ and rebuilt.
    """
    train_csv, val_csv, test_csv = (OUTPUT_DIR / f"{s}.csv" for s in ("train", "val", "test"))
    meta_path = OUTPUT_DIR / "split_meta.json"
    if all(p.is_file() for p in (train_csv, val_csv, test_csv, meta_path)):
        first = pd.read_csv(train_csv)["path"].iloc[0]
        if Path(first).is_file():
            return  # existing CSVs are valid
        print("Split CSVs point at missing files (created on another machine) - rebuilding from input/.")
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    stale = [p for p in (train_csv, val_csv, test_csv, meta_path) if p.is_file()]
    if stale:
        backup_dir = OUTPUT_DIR / "backup_prev"
        backup_dir.mkdir(parents=True, exist_ok=True)
        for p in stale:
            shutil.move(str(p), backup_dir / p.name)
        print(f"Moved {len(stale)} existing file(s) to {backup_dir}")

    rows = _collect_rows_from_input(seed)
    for split, r in rows.items():
        pd.DataFrame(r).to_csv(OUTPUT_DIR / f"{split}.csv", index=False)
        print(f"Wrote {split}.csv ({len(r)} rows)")
    mean, std = _compute_train_mean_std(train_csv)
    print(f"train mean={mean}\ntrain std ={std}")
    counts = {s: pd.DataFrame(r)["class_name"].value_counts().to_dict() for s, r in rows.items()}
    meta = {
        "seed": seed,
        "source": "input/Cleaned Dataset pre-split folders (train/validation/test), used as-is",
        "class_counts": counts,
        "mean": mean,
        "std": std,
        "note": (
            "Official pre-split folders used as-is; no re-splitting. "
            "Augmentation is applied only in the training DataLoader, never written to disk."
        ),
    }
    with meta_path.open("w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)
    print(f"Wrote {meta_path}")


def build_train_transform(mean: list[float], std: list[float], image_size: int = IMAGE_SIZE):
    return transforms.Compose(
        [
            transforms.RandomResizedCrop(image_size, scale=(0.6, 1.0)),
            transforms.RandomHorizontalFlip(),
            transforms.RandAugment(num_ops=2, magnitude=9),
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std),
            transforms.RandomErasing(p=0.25, scale=(0.02, 0.2), ratio=(0.3, 3.3), value="random"),
        ]
    )


def build_eval_transform(mean: list[float], std: list[float], image_size: int = IMAGE_SIZE):
    return transforms.Compose(
        [
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std),
        ]
    )


def build_eval_tta_transform(mean: list[float], std: list[float], image_size: int = IMAGE_SIZE):
    """Second TTA view: mild zoom-in via 256 resize + center crop."""
    return transforms.Compose(
        [
            transforms.Resize((int(image_size * 8 / 7),) * 2),
            transforms.CenterCrop(image_size),
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std),
        ]
    )


def make_loaders(
    batch_size: int = 32,
    num_workers: int = 0,
    seed: int = 42,
    image_size: int = IMAGE_SIZE,
) -> tuple[DataLoader, DataLoader, DataLoader, list[float], list[float]]:
    ensure_split_csvs(seed=seed)
    mean, std = load_norm_stats()
    train_ds = DogBreedDataset(OUTPUT_DIR / "train.csv", transform=build_train_transform(mean, std, image_size))
    val_ds = DogBreedDataset(OUTPUT_DIR / "val.csv", transform=build_eval_transform(mean, std, image_size))
    test_ds = DogBreedDataset(OUTPUT_DIR / "test.csv", transform=build_eval_transform(mean, std, image_size))

    generator = torch.Generator().manual_seed(seed)
    train_loader = DataLoader(
        train_ds, batch_size=batch_size, shuffle=True, num_workers=num_workers, generator=generator
    )
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    return train_loader, val_loader, test_loader, mean, std
