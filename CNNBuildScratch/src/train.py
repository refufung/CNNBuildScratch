"""Train CNNSE from scratch on the 1000/class split; TTA + ensemble one-shot test.

Recipe:
  AdamW (wd on conv/linear weights only) + per-step linear warmup -> cosine to 0
  label smoothing 0.1, MixUp(0.2)/CutMix(1.0) per batch with prob 0.5,
  grad clip 1.0, EMA(0.999), best checkpoint selected by EMA val macro-F1.

Usage:
  python train.py --smoke                      # 2-min end-to-end sanity check
  python train.py --epochs 60 --seeds 42       # sanity run
  python train.py --epochs 150 --seeds 42 1337 2025   # full run
  python train.py --eval-only --seeds 42 1337 2025    # TTA + ensemble test from checkpoints
"""

from __future__ import annotations

import argparse
import copy
import json
import math
import random
import re
import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch._utils  # force-load; avoids stale-kernel AttributeError via torch.optim.AdamW
import torch.nn.functional as F

# Disable torch._dynamo — corrupted Jupyter/CUDA sessions can break AdamW via lazy getattr.
try:
    import torch._dynamo as _dynamo

    _dynamo.config.suppress_errors = True
    _dynamo.disable()
except Exception:
    pass

from sklearn.metrics import (
    accuracy_score,
    auc,
    classification_report,
    confusion_matrix,
    f1_score,
    roc_curve,
)
from torch import nn
from torch.optim.adamw import AdamW
from torch.utils.data import DataLoader

from dataset import (
    OUTPUT_DIR,
    DogBreedDataset,
    build_eval_tta_transform,
    make_loaders,
)
from model import CNNSE, count_params

CLASS_NAMES = ["French_bulldog", "Cardigan", "Labrador_retriever", "golden_retriever"]


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.backends.mps.is_available():
        torch.mps.manual_seed(seed)


def get_device() -> torch.device:
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


# ---------------- MixUp / CutMix ----------------

def rand_bbox(size: torch.Size, lam: float) -> tuple[int, int, int, int]:
    h, w = size[2], size[3]
    cut_rat = math.sqrt(1.0 - lam)
    cut_w, cut_h = int(w * cut_rat), int(h * cut_rat)
    cx, cy = np.random.randint(w), np.random.randint(h)
    x1, x2 = np.clip(cx - cut_w // 2, 0, w), np.clip(cx + cut_w // 2, 0, w)
    y1, y2 = np.clip(cy - cut_h // 2, 0, h), np.clip(cy + cut_h // 2, 0, h)
    return x1, y1, x2, y2


def maybe_mix(
    images: torch.Tensor, targets: torch.Tensor, mixup_alpha: float, cutmix_alpha: float, prob: float
) -> tuple[torch.Tensor, torch.Tensor, tuple[torch.Tensor, float] | None]:
    if random.random() > prob:
        return images, targets, None
    perm = torch.randperm(images.size(0), device=images.device)
    if random.random() < 0.5:  # CutMix
        lam = float(np.random.beta(cutmix_alpha, cutmix_alpha))
        x1, y1, x2, y2 = rand_bbox(images.size(), lam)
        images[:, :, y1:y2, x1:x2] = images[perm][:, :, y1:y2, x1:x2]
        lam = 1.0 - ((x2 - x1) * (y2 - y1) / (images.size(-1) * images.size(-2)))
    else:  # MixUp
        lam = float(np.random.beta(mixup_alpha, mixup_alpha))
        images = lam * images + (1.0 - lam) * images[perm]
    return images, targets, (perm, lam)


def soft_ce(
    logits: torch.Tensor,
    targets: torch.Tensor,
    smoothing: float,
    mix: tuple[torch.Tensor, float] | None = None,
) -> torch.Tensor:
    """Cross-entropy with label smoothing; linearly combined for mixed batches."""
    logp = F.log_softmax(logits, dim=1)

    def ce(t: torch.Tensor) -> torch.Tensor:
        nll = -logp.gather(1, t.unsqueeze(1)).squeeze(1)
        return (1.0 - smoothing) * nll + smoothing * (-logp.mean(dim=1))

    loss = ce(targets)
    if mix is not None:
        perm, lam = mix
        loss = lam * loss + (1.0 - lam) * ce(targets[perm])
    return loss.mean()


# ---------------- EMA ----------------

class ModelEMA:
    def __init__(self, model: nn.Module, decay: float = 0.999):
        self.module = copy.deepcopy(model).eval()
        for p in self.module.parameters():
            p.requires_grad_(False)
        self.decay = decay

    @torch.no_grad()
    def update(self, model: nn.Module) -> None:
        msd = model.state_dict()
        for k, v in self.module.state_dict().items():
            if v.dtype.is_floating_point:
                v.mul_(self.decay).add_(msd[k].detach(), alpha=1.0 - self.decay)
            else:
                v.copy_(msd[k])


# ---------------- eval ----------------

@torch.no_grad()
def evaluate(
    model: nn.Module, loader: DataLoader, device: torch.device, smoothing: float = 0.0
) -> tuple[float, float, float, np.ndarray, list[int], list[int]]:
    model.eval()
    total_loss, n = 0.0, 0
    all_preds: list[int] = []
    all_labels: list[int] = []
    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)
        logits = model(images)
        loss = soft_ce(logits, labels, smoothing)
        total_loss += loss.item() * labels.size(0)
        n += labels.size(0)
        all_preds.extend(logits.argmax(1).cpu().tolist())
        all_labels.extend(labels.cpu().tolist())
    acc = accuracy_score(all_labels, all_preds)
    f1 = f1_score(all_labels, all_preds, average="macro", zero_division=0)
    cm = confusion_matrix(all_labels, all_preds, labels=list(range(len(CLASS_NAMES))))
    return total_loss / max(n, 1), acc, f1, cm, all_labels, all_preds


@torch.no_grad()
def predict_probs(
    model: nn.Module, loader: DataLoader, device: torch.device, tta_loader: DataLoader | None = None
) -> tuple[np.ndarray, np.ndarray]:
    """Mean softmax over views: [x, hflip(x)] for the plain loader (+ TTA zoom loader if given)."""
    model.eval()
    loaders = [loader] + ([tta_loader] if tta_loader is not None else [])
    view_probs: list[np.ndarray] = []
    labels_all: list[int] = []
    for view_idx, current in enumerate(loaders):
        batch_probs: list[np.ndarray] = []
        for images, labels in current:
            images = images.to(device)
            p = F.softmax(model(images), dim=1) + F.softmax(model(torch.flip(images, dims=[3])), dim=1)
            batch_probs.append((p / 2.0).cpu().numpy())
            if view_idx == 0:
                labels_all.extend(labels.tolist())
        view_probs.append(np.concatenate(batch_probs, axis=0))
    views = np.stack(view_probs, axis=0)  # (n_views, n, C)
    return views.mean(axis=0), np.array(labels_all)


# ---------------- train ----------------

def build_optimizer(model: nn.Module, lr: float, weight_decay: float) -> AdamW:
    decay, no_decay = [], []
    for p in model.parameters():
        (decay if p.ndim > 1 else no_decay).append(p)
    return AdamW(
        [
            {"params": decay, "weight_decay": weight_decay},
            {"params": no_decay, "weight_decay": 0.0},
        ],
        lr=lr,
    )


def lr_at(step: int, total_steps: int, warmup_steps: int, base_lr: float) -> float:
    if step < warmup_steps:
        return base_lr * (step + 1) / warmup_steps
    t = (step - warmup_steps) / max(1, total_steps - warmup_steps)
    return base_lr * 0.5 * (1.0 + math.cos(math.pi * t))


def train_one_seed(args, seed: int, device: torch.device) -> dict:
    set_seed(seed)
    train_loader, val_loader, _, mean, std = make_loaders(
        batch_size=args.batch_size, num_workers=args.num_workers, seed=seed
    )
    if args.smoke:
        args.max_batches = 6

    model = CNNSE(num_classes=len(CLASS_NAMES), dropout=args.dropout).to(device)
    if seed == args.seeds[0]:
        print(f"CNNSE params: {count_params(model) / 1e6:.2f}M")
    optimizer = build_optimizer(model, lr=args.lr, weight_decay=args.wd)
    ema = ModelEMA(model, decay=args.ema_decay)

    steps_per_epoch = len(train_loader)
    total_steps = steps_per_epoch * args.epochs
    warmup_steps = steps_per_epoch * args.warmup_epochs
    global_step = 0

    history = {"train_loss": [], "train_acc": [], "val_loss": [], "val_acc": [], "val_macro_f1": []}
    best_f1, best_epoch, best_state = -1.0, 0, None
    patience = int(getattr(args, "early_stop_patience", 0) or 0)  # 0 = disabled
    t0 = time.time()

    log_dir = OUTPUT_DIR / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    log_f = (log_dir / f"train_seed{seed}.log").open("w", encoding="utf-8")
    log_f.write(
        f"[seed {seed}] start epochs={args.epochs} bs={args.batch_size} lr={args.lr} wd={args.wd} "
        f"warmup={args.warmup_epochs} mix_prob={args.mix_prob} device={device}\n"
    )
    log_f.flush()

    for epoch in range(1, args.epochs + 1):
        model.train()
        running_loss, correct, total = 0.0, 0, 0
        for i, (images, labels) in enumerate(train_loader):
            if args.smoke and i >= args.max_batches:
                break
            images, labels = images.to(device), labels.to(device)
            images, labels, mix = maybe_mix(images, labels, args.mixup_alpha, args.cutmix_alpha, args.mix_prob)

            lr = lr_at(global_step, total_steps, warmup_steps, args.lr)
            for g in optimizer.param_groups:
                g["lr"] = lr

            optimizer.zero_grad(set_to_none=True)
            logits = model(images)
            loss = soft_ce(logits, labels, args.label_smoothing, mix)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            ema.update(model)
            global_step += 1

            running_loss += loss.item() * labels.size(0)
            correct += (logits.argmax(1) == labels).sum().item()
            total += labels.size(0)

        val_loss, val_acc, val_f1, _, _, _ = evaluate(ema.module, val_loader, device)
        history["train_loss"].append(running_loss / max(total, 1))
        history["train_acc"].append(correct / max(total, 1))
        history["val_loss"].append(val_loss)
        history["val_acc"].append(val_acc)
        history["val_macro_f1"].append(val_f1)

        marker = ""
        if val_f1 > best_f1:
            best_f1, best_epoch = val_f1, epoch
            best_state = {k: v.detach().cpu().clone() for k, v in ema.module.state_dict().items()}
            marker = " *"
        line = (
            f"[seed {seed}] epoch {epoch:03d}/{args.epochs} lr={lr:.2e} "
            f"train_loss={history['train_loss'][-1]:.4f} train_acc={history['train_acc'][-1]:.4f} "
            f"val_loss={val_loss:.4f} val_acc={val_acc:.4f} val_f1={val_f1:.4f} "
            f"({time.time() - t0:.0f}s){marker}"
        )
        print(line, flush=True)
        log_f.write(line + "\n")
        log_f.flush()
        if args.smoke and epoch >= 1:
            break
        if patience and epoch - best_epoch >= patience:
            stop_msg = (
                f"[seed {seed}] early stop at epoch {epoch}: no val macro-F1 improvement in "
                f"{patience} consecutive epochs (best {best_f1:.4f} @ epoch {best_epoch})"
            )
            print(stop_msg)
            log_f.write(stop_msg + "\n")
            log_f.flush()
            break

    assert best_state is not None
    ckpt = {
        "arch": "CNN_se",
        "state_dict": best_state,
        "seed": seed,
        "best_val_macro_f1": best_f1,
        "best_epoch": best_epoch,
        "class_names": CLASS_NAMES,
        "mean": mean,
        "std": std,
        "config": vars(args),
    }
    ckpt_path = OUTPUT_DIR / ("smoke_ckpt.pt" if args.smoke else f"best_CNN_se_seed{seed}.pt")
    torch.save(ckpt, ckpt_path)
    msg = f"[seed {seed}] saved {ckpt_path} (best val macro-F1={best_f1:.4f} @ epoch {best_epoch})"
    print(msg)
    log_f.write(msg + "\n")
    log_f.close()

    return {
        "seed": seed,
        "history": history,
        "best_val_macro_f1": best_f1,
        "best_epoch": best_epoch,
        "ckpt_path": str(ckpt_path),
        "train_seconds": time.time() - t0,
    }


# ---------------- eval helpers ----------------

def load_model(ckpt_path: Path, device: torch.device) -> nn.Module:
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    model = CNNSE(num_classes=len(CLASS_NAMES))
    model.load_state_dict(ckpt["state_dict"])
    return model.to(device).eval()


def make_tta_loader(split: str, mean, std, batch_size: int) -> DataLoader:
    ds = DogBreedDataset(OUTPUT_DIR / f"{split}.csv", transform=build_eval_tta_transform(mean, std))
    return DataLoader(ds, batch_size=batch_size, shuffle=False, num_workers=0)


def plot_curves(histories: list[dict], out_path: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    for h in histories:
        hist, seed = h["history"], h["seed"]
        epochs = range(1, len(hist["train_loss"]) + 1)
        axes[0].plot(epochs, hist["train_loss"], alpha=0.6, label=f"train s{seed}")
        axes[0].plot(epochs, hist["val_loss"], ls="--", alpha=0.8, label=f"val s{seed}")
        axes[1].plot(epochs, hist["val_acc"], alpha=0.8, label=f"val acc s{seed}")
        axes[1].plot(epochs, hist["val_macro_f1"], ls="--", alpha=0.8, label=f"val F1 s{seed}")
    axes[0].set_title("Loss")
    axes[1].set_title("Val accuracy / macro-F1")
    for ax in axes:
        ax.set_xlabel("Epoch")
        ax.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def plot_confusion(cm: np.ndarray, out_path: Path, title: str) -> None:
    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(cm, cmap="Blues")
    ax.set_xticks(range(len(CLASS_NAMES)))
    ax.set_yticks(range(len(CLASS_NAMES)))
    ax.set_xticklabels(CLASS_NAMES, rotation=30, ha="right")
    ax.set_yticklabels(CLASS_NAMES)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title(title)
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, str(cm[i, j]), ha="center", va="center", color="black")
    fig.colorbar(im, ax=ax, fraction=0.046)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


_EPOCH_LOG_RE = re.compile(
    r"\[seed (?P<seed>\d+)\] epoch (?P<epoch>\d+)/(?P<n_epochs>\d+) "
    r"lr=(?P<lr>[\d.e+-]+) "
    r"train_loss=(?P<train_loss>[\d.]+) train_acc=(?P<train_acc>[\d.]+) "
    r"val_loss=(?P<val_loss>[\d.]+) val_acc=(?P<val_acc>[\d.]+) val_f1=(?P<val_f1>[\d.]+) "
    r"\((?P<cum_s>\d+)s\)"
)


def parse_train_log(log_path: Path | str) -> list[dict]:
    """Parse a multi-seed training log into histories usable by plot helpers.

    Cumulative ``(Ns)`` timestamps are differenced into per-epoch seconds.
    """
    log_path = Path(log_path)
    by_seed: dict[int, dict] = {}
    for line in log_path.read_text(encoding="utf-8", errors="replace").splitlines():
        m = _EPOCH_LOG_RE.search(line)
        if not m:
            continue
        seed = int(m.group("seed"))
        bucket = by_seed.setdefault(
            seed,
            {
                "seed": seed,
                "history": {
                    "train_loss": [],
                    "train_acc": [],
                    "val_loss": [],
                    "val_acc": [],
                    "val_macro_f1": [],
                    "epoch_seconds": [],
                },
                "_cum": [],
            },
        )
        h = bucket["history"]
        h["train_loss"].append(float(m.group("train_loss")))
        h["train_acc"].append(float(m.group("train_acc")))
        h["val_loss"].append(float(m.group("val_loss")))
        h["val_acc"].append(float(m.group("val_acc")))
        h["val_macro_f1"].append(float(m.group("val_f1")))
        bucket["_cum"].append(int(m.group("cum_s")))

    runs = []
    for seed in sorted(by_seed):
        bucket = by_seed[seed]
        cum = bucket.pop("_cum")
        prev = 0
        secs = []
        for c in cum:
            secs.append(max(c - prev, 0))
            prev = c
        bucket["history"]["epoch_seconds"] = secs
        runs.append(bucket)
    return runs


def export_curves_csv(runs: list[dict], out_path: Path) -> Path:
    """Write a long-form CSV of per-seed epoch metrics (the step-9 curves dataset)."""
    import pandas as pd

    rows = []
    for run in runs:
        seed = run["seed"]
        hist = run["history"]
        n = len(hist["train_loss"])
        for i in range(n):
            rows.append(
                {
                    "seed": seed,
                    "epoch": i + 1,
                    "train_loss": hist["train_loss"][i],
                    "train_acc": hist["train_acc"][i],
                    "val_loss": hist["val_loss"][i],
                    "val_acc": hist["val_acc"][i],
                    "val_macro_f1": hist["val_macro_f1"][i],
                    "epoch_seconds": hist.get("epoch_seconds", [None] * n)[i],
                }
            )
    out_path = Path(out_path)
    pd.DataFrame(rows).to_csv(out_path, index=False)
    return out_path


def plot_curves_report(runs: list[dict], out_path: Path) -> Path:
    """3-panel report: train/val loss, train/val accuracy, seconds/epoch."""
    fig, axes = plt.subplots(1, 3, figsize=(14, 4))
    for run in runs:
        hist, seed = run["history"], run["seed"]
        epochs = range(1, len(hist["train_loss"]) + 1)
        axes[0].plot(epochs, hist["train_loss"], alpha=0.75, label=f"{seed}")
        axes[0].plot(epochs, hist["val_loss"], ls="--", alpha=0.85)
        axes[1].plot(epochs, hist["train_acc"], alpha=0.75, label=f"{seed}")
        axes[1].plot(epochs, hist["val_acc"], ls="--", alpha=0.85)
        if hist.get("epoch_seconds"):
            axes[2].plot(epochs, hist["epoch_seconds"], alpha=0.85, label=f"{seed}")
    axes[0].set_title("Train / val loss")
    axes[0].set_ylabel("Loss")
    axes[1].set_title("Train / val accuracy")
    axes[1].set_ylabel("Accuracy")
    axes[2].set_title("Seconds / epoch")
    axes[2].set_ylabel("Seconds")
    for ax in axes:
        ax.set_xlabel("Epoch")
        ax.legend(title="seed", fontsize=8)
    fig.tight_layout()
    out_path = Path(out_path)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    return out_path


def plot_test_tta_bars(metrics: dict, out_path: Path) -> Path:
    """Bar charts: per-seed + ensemble test accuracy and macro-F1 (TTA)."""
    seeds = metrics.get("seeds") or []
    # Support both flat metrics.json shapes (seeds at top-level or under runs+seeds).
    if not seeds and "runs" in metrics:
        # older backup shape embeds test metrics on runs
        seeds = [
            {
                "seed": r["seed"],
                "test_tta_acc": r.get("test_tta_acc"),
                "test_tta_macro_f1": r.get("test_tta_macro_f1"),
            }
            for r in metrics["runs"]
            if "test_tta_acc" in r
        ]
    ens = metrics.get("test_ensemble") or {}
    labels = [str(s["seed"]) for s in seeds] + ["Ensemble"]
    accs = [100.0 * float(s["test_tta_acc"]) for s in seeds] + [100.0 * float(ens["accuracy"])]
    f1s = [100.0 * float(s["test_tta_macro_f1"]) for s in seeds] + [100.0 * float(ens["macro_f1"])]

    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    x = np.arange(len(labels))
    for ax, vals, title in (
        (axes[0], accs, "Test accuracy with TTA"),
        (axes[1], f1s, "Test macro-F1 with TTA"),
    ):
        bars = ax.bar(x, vals, color=["#4c72b0", "#dd8452", "#55a868", "#c44e52"][: len(labels)])
        ax.set_xticks(x)
        ax.set_xticklabels(labels)
        ax.set_ylim(0, 100)
        ax.set_ylabel("%")
        ax.set_title(title)
        for bar, v in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width() / 2, v + 1.0, f"{v:.2f}%", ha="center", va="bottom", fontsize=8)
    fig.tight_layout()
    out_path = Path(out_path)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    return out_path


def build_run_report(
    seeds: list[int] | None = None,
    metrics_path: Path | None = None,
    out_dir: Path | None = None,
    prefix: str = "report_run",
) -> dict:
    """Build report curves/CSV + TTA figures for the seed(s) from this notebook/CLI run.

    Reads ``outputs/logs/train_seed{seed}.log`` and ``outputs/metrics.json`` (from steps 5–8).

    Writes (default prefix ``report_run``):
      - outputs/{prefix}_curves.csv
      - outputs/{prefix}_metrics.json
      - outputs/{prefix}_training_curves.png
      - outputs/{prefix}_confusion_matrix.png
      - outputs/{prefix}_test_tta_bars.png
    """
    out_dir = Path(out_dir or OUTPUT_DIR)
    metrics_path = Path(metrics_path or (out_dir / "metrics.json"))
    if not metrics_path.is_file():
        raise FileNotFoundError(metrics_path)
    metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
    if seeds is None:
        seeds = list(metrics.get("config", {}).get("seeds") or [])
        if not seeds:
            seeds = [int(s["seed"]) for s in (metrics.get("seeds") or [])]
    if not seeds:
        raise ValueError("No seeds provided and none found in metrics.json")

    log_paths = []
    runs: list[dict] = []
    for seed in seeds:
        log_path = out_dir / "logs" / f"train_seed{seed}.log"
        if not log_path.is_file():
            raise FileNotFoundError(log_path)
        parsed = parse_train_log(log_path)
        if not parsed:
            raise ValueError(f"No epoch rows parsed from {log_path}")
        # Keep only this seed's history (log files are per-seed).
        seed_runs = [r for r in parsed if r["seed"] == seed]
        runs.extend(seed_runs or parsed)
        log_paths.append(str(log_path))

    out_dir.mkdir(parents=True, exist_ok=True)
    curves_csv = export_curves_csv(runs, out_dir / f"{prefix}_curves.csv")
    metrics_out = out_dir / f"{prefix}_metrics.json"
    metrics_out.write_text(json.dumps(metrics, indent=2), encoding="utf-8")

    curves_png = plot_curves_report(runs, out_dir / f"{prefix}_training_curves.png")
    ens = metrics["test_ensemble"]
    cm = np.asarray(ens["confusion_matrix"], dtype=int)
    cm_png = out_dir / f"{prefix}_confusion_matrix.png"
    plot_confusion(cm, cm_png, "Ensemble + TTA test confusion matrix")
    bars_png = plot_test_tta_bars(metrics, out_dir / f"{prefix}_test_tta_bars.png")

    summary = {
        "seeds": seeds,
        "source_logs": log_paths,
        "source_metrics": str(metrics_path),
        "n_seeds": len(runs),
        "epochs_per_seed": {r["seed"]: len(r["history"]["train_loss"]) for r in runs},
        "artifacts": {
            "curves_csv": str(curves_csv),
            "metrics_json": str(metrics_out),
            "training_curves_png": str(curves_png),
            "confusion_matrix_png": str(cm_png),
            "test_tta_bars_png": str(bars_png),
        },
    }
    print(json.dumps(summary, indent=2))
    return summary


# Back-compat alias (prefer build_run_report).
def build_3seed_report(*args, **kwargs):
    return build_run_report(*args, **kwargs)


def _ensemble_tta_probs(args, device: torch.device, split: str) -> tuple[np.ndarray, np.ndarray]:
    """Return (probs [N,C], labels [N]) with mean TTA softmax over seeds."""
    _, val_loader, test_loader, mean, std = make_loaders(batch_size=args.batch_size, num_workers=0)
    eval_loader = {"val": val_loader, "test": test_loader}[split]
    tta_loader = make_tta_loader(split, mean, std, args.batch_size)
    split_probs: list[np.ndarray] = []
    labels = None
    for seed in args.seeds:
        ckpt_path = OUTPUT_DIR / f"best_CNN_se_seed{seed}.pt"
        if not ckpt_path.is_file():
            raise FileNotFoundError(ckpt_path)
        model = load_model(ckpt_path, device)
        probs, labels = predict_probs(model, eval_loader, device, tta_loader=tta_loader)
        split_probs.append(probs)
    return np.mean(split_probs, axis=0), np.asarray(labels)


def _youden_threshold(y_true: np.ndarray, y_score: np.ndarray) -> dict:
    """Select binary threshold maximizing Youden's J = TPR - FPR."""
    fpr, tpr, thresholds = roc_curve(y_true, y_score)
    j = tpr - fpr
    # sklearn appends thresholds with inf as first element when needed; clip to [0,1]
    idx = int(np.argmax(j))
    thr = float(thresholds[idx])
    if not np.isfinite(thr):
        thr = 1.0
    thr = float(np.clip(thr, 0.0, 1.0))
    return {
        "threshold": thr,
        "tpr": float(tpr[idx]),
        "fpr": float(fpr[idx]),
        "youden_j": float(j[idx]),
        "auc": float(auc(fpr, tpr)),
        "fpr_curve": fpr.tolist(),
        "tpr_curve": tpr.tolist(),
    }


def _predict_with_class_thresholds(probs: np.ndarray, thresholds: list[float]) -> np.ndarray:
    """Predict class with highest (p - thr); if none exceed thr, fall back to argmax."""
    adj = probs - np.asarray(thresholds, dtype=np.float64)[None, :]
    preds = adj.argmax(axis=1)
    any_pass = (probs >= np.asarray(thresholds)[None, :]).any(axis=1)
    fallback = probs.argmax(axis=1)
    preds = np.where(any_pass, preds, fallback)
    return preds.astype(int)


def plot_roc_curves(
    class_stats: list[dict],
    out_path: Path,
    title: str = "One-vs-rest ROC (validation TTA)",
) -> Path:
    fig, ax = plt.subplots(figsize=(7, 6))
    for s in class_stats:
        ax.plot(
            s["fpr_curve"],
            s["tpr_curve"],
            lw=2,
            label=f"{s['class_name']} (AUC={s['auc']:.3f}, thr={s['threshold']:.3f})",
        )
        ax.scatter([s["fpr"]], [s["tpr"]], s=40, zorder=5)
    ax.plot([0, 1], [0, 1], "k--", lw=1, alpha=0.6, label="chance")
    ax.set_xlim(0.0, 1.0)
    ax.set_ylim(0.0, 1.05)
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title(title)
    ax.legend(loc="lower right", fontsize=8)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    out_path = Path(out_path)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    return out_path


def run_roc_threshold_analysis(args, device: torch.device) -> dict:
    """OvR ROC on val TTA, select Youden thresholds, evaluate on val + test.

    Writes:
      - outputs/roc_val.png
      - outputs/roc_thresholds.json
    """
    print("[roc] computing validation TTA probabilities for threshold selection...")
    val_probs, val_labels = _ensemble_tta_probs(args, device, split="val")
    print("[roc] computing test TTA probabilities for hold-out evaluation...")
    test_probs, test_labels = _ensemble_tta_probs(args, device, split="test")

    class_stats = []
    thresholds = []
    for c, name in enumerate(CLASS_NAMES):
        y = (val_labels == c).astype(int)
        score = val_probs[:, c]
        stats = _youden_threshold(y, score)
        stats["class_index"] = c
        stats["class_name"] = name
        # Drop bulky curves from the printed/json summary after plotting
        class_stats.append(stats)
        thresholds.append(stats["threshold"])
        print(
            f"[roc] {name}: AUC={stats['auc']:.4f} Youden thr={stats['threshold']:.4f} "
            f"(TPR={stats['tpr']:.3f}, FPR={stats['fpr']:.3f})"
        )

    roc_png = plot_roc_curves(class_stats, OUTPUT_DIR / "roc_val.png")

    val_pred_argmax = val_probs.argmax(1)
    test_pred_argmax = test_probs.argmax(1)
    val_pred_thr = _predict_with_class_thresholds(val_probs, thresholds)
    test_pred_thr = _predict_with_class_thresholds(test_probs, thresholds)

    def _pack(y, pred_a, pred_t):
        return {
            "argmax_acc": float(accuracy_score(y, pred_a)),
            "argmax_macro_f1": float(f1_score(y, pred_a, average="macro", zero_division=0)),
            "thresholded_acc": float(accuracy_score(y, pred_t)),
            "thresholded_macro_f1": float(f1_score(y, pred_t, average="macro", zero_division=0)),
        }

    result = {
        "method": "one-vs-rest ROC; per-class Youden J = TPR-FPR on validation TTA",
        "decision_rule": "predict argmax(p - thr); if no class passes thr, fall back to argmax(p)",
        "seeds": list(args.seeds),
        "classes": [
            {
                "class_index": s["class_index"],
                "class_name": s["class_name"],
                "threshold": s["threshold"],
                "auc": s["auc"],
                "tpr_at_threshold": s["tpr"],
                "fpr_at_threshold": s["fpr"],
                "youden_j": s["youden_j"],
            }
            for s in class_stats
        ],
        "validation": _pack(val_labels, val_pred_argmax, val_pred_thr),
        "test": _pack(test_labels, test_pred_argmax, test_pred_thr),
        "artifacts": {
            "roc_png": str(roc_png),
            "thresholds_json": str(OUTPUT_DIR / "roc_thresholds.json"),
        },
    }
    with (OUTPUT_DIR / "roc_thresholds.json").open("w", encoding="utf-8") as f:
        json.dump(result, f, indent=2)
    print(
        f"[roc] val  argmax F1={result['validation']['argmax_macro_f1']:.4f} | "
        f"thresholded F1={result['validation']['thresholded_macro_f1']:.4f}"
    )
    print(
        f"[roc] test argmax F1={result['test']['argmax_macro_f1']:.4f} | "
        f"thresholded F1={result['test']['thresholded_macro_f1']:.4f}"
    )
    print(f"[roc] wrote {roc_png}")
    print(f"[roc] wrote {OUTPUT_DIR / 'roc_thresholds.json'}")
    return result


def _run_tta_eval(
    args, device: torch.device, split: str, cm_filename: str, tag: str
) -> tuple[dict, np.ndarray, np.ndarray]:
    """Per-seed TTA on `split` ("val" | "test") + seed-ensemble; saves the ensemble confusion matrix.

    Returns (results dict, ensemble labels, ensemble predictions).
    """
    _, val_loader, test_loader, mean, std = make_loaders(batch_size=args.batch_size, num_workers=0)
    eval_loader = {"val": val_loader, "test": test_loader}[split]
    tta_loader = make_tta_loader(split, mean, std, args.batch_size)

    results = {"seeds": [], f"{tag}_ensemble": None}
    split_probs: list[np.ndarray] = []
    split_labels = None
    for seed in args.seeds:
        ckpt_path = OUTPUT_DIR / f"best_CNN_se_seed{seed}.pt"
        if not ckpt_path.is_file():
            raise FileNotFoundError(ckpt_path)
        model = load_model(ckpt_path, device)
        probs, labels = predict_probs(model, eval_loader, device, tta_loader=tta_loader)
        split_probs.append(probs)
        split_labels = labels
        preds = probs.argmax(1)
        acc = accuracy_score(labels, preds)
        f1 = f1_score(labels, preds, average="macro", zero_division=0)
        results["seeds"].append({"seed": seed, f"{tag}_tta_acc": acc, f"{tag}_tta_macro_f1": f1})
        print(f"[eval] seed {seed} single+TTA {tag} acc={acc:.4f} f1={f1:.4f}")

    ens_probs = np.mean(split_probs, axis=0)
    ens_preds = ens_probs.argmax(1)
    acc = accuracy_score(split_labels, ens_preds)
    f1 = f1_score(split_labels, ens_preds, average="macro", zero_division=0)
    cm = confusion_matrix(split_labels, ens_preds, labels=list(range(len(CLASS_NAMES))))
    results[f"{tag}_ensemble"] = {
        "accuracy": acc,
        "macro_f1": f1,
        "confusion_matrix": cm.tolist(),
        "class_names": CLASS_NAMES,
        "n_seeds": len(args.seeds),
        "tta": "mean softmax over {224, hflip, 256-center-crop, hflip} x seeds",
    }
    print(f"[eval] ENSEMBLE+TTA {tag} acc={acc:.4f} f1={f1:.4f}")
    plot_confusion(cm, OUTPUT_DIR / cm_filename, f"Ensemble+TTA {tag} confusion matrix")
    return results, np.asarray(split_labels), ens_preds


def run_validation(args, device: torch.device) -> dict:
    """TTA per seed on the validation split (input/Cleaned Dataset/validation); ensemble over seeds.

    Note: the val split is used for checkpoint selection during training, so these
    numbers are mildly optimistic — the test split remains the one-shot final estimate.
    Result keys are val_* so the dict merges into metrics.json next to the test results.
    """
    results, labels, ens_preds = _run_tta_eval(
        args, device, split="val", cm_filename="confusion_matrix_val.png", tag="val"
    )
    print("[eval] val ensemble per-class report:")
    print(classification_report(labels, ens_preds, target_names=CLASS_NAMES, zero_division=0))
    return results


def run_final_eval(args, device: torch.device) -> dict:
    """TTA per seed on test; ensemble = mean over seeds. Test evaluated once."""
    results, _, _ = _run_tta_eval(
        args, device, split="test", cm_filename="confusion_matrix.png", tag="test"
    )
    return results


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=150)
    parser.add_argument("--seeds", type=int, nargs="+", default=[42])
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--wd", type=float, default=5e-2)
    parser.add_argument("--warmup-epochs", type=int, default=5)
    parser.add_argument("--dropout", type=float, default=0.3)
    parser.add_argument("--label-smoothing", type=float, default=0.1)
    parser.add_argument("--mixup-alpha", type=float, default=0.2)
    parser.add_argument("--cutmix-alpha", type=float, default=1.0)
    parser.add_argument("--mix-prob", type=float, default=0.5)
    parser.add_argument("--ema-decay", type=float, default=0.999)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--early-stop-patience", type=int, default=0,
                        help="stop when val macro-F1 has not improved for N consecutive epochs (0 = off)")
    parser.add_argument("--eval-only", action="store_true")
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()

    if args.smoke:
        args.epochs = 1
        args.warmup_epochs = 0

    device = get_device()
    print(f"device={device} epochs={args.epochs} seeds={args.seeds} bs={args.batch_size} "
          f"lr={args.lr} wd={args.wd} warmup={args.warmup_epochs} mix_prob={args.mix_prob}")

    runs = []
    if not args.eval_only:
        for seed in args.seeds:
            runs.append(train_one_seed(args, seed, device))
        if not args.smoke:
            plot_curves(runs, OUTPUT_DIR / "training_curves.png")

    if args.smoke:
        print("SMOKE OK")
        return

    eval_results = run_final_eval(args, device)
    metrics = {
        "config": {k: v for k, v in vars(args).items()},
        "device": str(device),
        "class_counts": json.loads((OUTPUT_DIR / "split_meta.json").read_text())["class_counts"],
        "runs": [
            {k: r[k] for k in ("seed", "best_val_macro_f1", "best_epoch", "train_seconds")}
            for r in runs
        ],
        **eval_results,
    }
    with (OUTPUT_DIR / "metrics.json").open("w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)
    print(f"Wrote {OUTPUT_DIR / 'metrics.json'}")


if __name__ == "__main__":
    main()
