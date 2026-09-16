# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Phase 4: train the latent-native caption CNN on control-labeled shards.

Trains :class:`~omnidreams.caption.model.CaptionBankClassifier` with a
class-balanced cross-entropy, holding out whole session(s) for validation.
Checkpoints the best-val model to ``--out`` for :mod:`omnidreams.caption.export`.
The model is tiny and data lives in RAM, so this is fast (minutes)::

    uv run --project integrations/omnidreams \
      python -m omnidreams.caption.train \
        --shards-dir DATA/raw --out DATA/checkpoints
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader

from omnidreams.caption.dataset import (
    InMemoryLatentDataset,
    class_weights,
    read_manifest,
    session_split,
    sessions_of,
)
from omnidreams.caption.model import CAPTION_CLASSES, CaptionBankClassifier


@torch.no_grad()
def evaluate(
    model: nn.Module, loader: DataLoader, device: str, classes: list[str]
) -> tuple[float, dict[str, float], np.ndarray]:
    """Return (top-1 accuracy, per-class recall, confusion matrix)."""
    model.eval()
    n_cls = len(classes)
    conf = np.zeros((n_cls, n_cls), dtype=np.int64)
    correct = total = 0
    for x, y in loader:
        pred = model(x.to(device)).argmax(1).cpu()
        correct += int((pred == y).sum())
        total += int(y.numel())
        for t, p in zip(y.tolist(), pred.tolist()):
            conf[t, p] += 1
    acc = correct / max(total, 1)
    recall = {
        classes[i]: float(conf[i, i] / max(conf[i].sum(), 1))
        for i in range(n_cls)
    }
    return acc, recall, conf


def main() -> None:
    ap = argparse.ArgumentParser(description="Train the latent caption CNN.")
    ap.add_argument("--shards-dir", required=True, help="Capture raw/ shard dir.")
    ap.add_argument("--out", required=True, help="Checkpoint output dir.")
    ap.add_argument(
        "--val-session",
        action="append",
        default=None,
        help="Session id(s) to hold out for validation (default: the last one).",
    )
    ap.add_argument(
        "--classes",
        default=",".join(CAPTION_CLASSES),
        help="Comma-separated subset of the 6 classes to train on "
        "(default: all). Windows of other classes are dropped.",
    )
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--weight-decay", type=float, default=1e-4)
    ap.add_argument("--label-smoothing", type=float, default=0.05)
    ap.add_argument("--hidden", type=int, default=256)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    classes = [c.strip() for c in args.classes.split(",") if c.strip()]
    unknown = [c for c in classes if c not in CAPTION_CLASSES]
    if unknown:
        raise SystemExit(
            f"unknown classes {unknown}; choose from {list(CAPTION_CLASSES)}"
        )
    items = read_manifest(args.shards_dir, classes)
    all_sessions = sessions_of(items)
    val_sessions = args.val_session or all_sessions[-1:]
    train_items, val_items = session_split(items, val_sessions)
    print(
        f"[train] {len(items)} windows across {len(all_sessions)} sessions; "
        f"classes={classes}; val={val_sessions} -> "
        f"train {len(train_items)} / val {len(val_items)}",
        flush=True,
    )
    if not train_items or not val_items:
        raise SystemExit("empty train or val split; check --val-session")

    train_ds = InMemoryLatentDataset(args.shards_dir, train_items, classes)
    val_ds = InMemoryLatentDataset(args.shards_dir, val_items, classes)
    train_ld = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True)
    val_ld = DataLoader(val_ds, batch_size=args.batch_size)

    device = args.device
    num_frames = int(train_ds.latents.shape[1])  # Twin (K * frames_per_chunk)
    # Per-channel input normalization stats from a sample of the training latents.
    sample = train_ds.latents[:512].float()
    in_mean = sample.mean(dim=(0, 1, 3, 4))
    in_std = sample.std(dim=(0, 1, 3, 4))
    del sample
    model = CaptionBankClassifier(
        num_frames=num_frames,
        num_classes=len(classes),
        hidden=args.hidden,
    )
    model.set_input_stats(in_mean, in_std)
    model = model.to(device)
    criterion = nn.CrossEntropyLoss(
        weight=class_weights(train_items, classes, device),
        label_smoothing=args.label_smoothing,
    )
    opt = torch.optim.AdamW(
        model.parameters(), lr=args.lr, weight_decay=args.weight_decay
    )
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs)

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    best_acc = 0.0
    best_recall: dict[str, float] = {}
    for epoch in range(args.epochs):
        model.train()
        running = 0.0
        nb = tr_correct = tr_total = 0
        for x, y in train_ld:
            x, y = x.to(device), y.to(device)
            opt.zero_grad()
            logits = model(x)
            loss = criterion(logits, y)
            loss.backward()
            opt.step()
            running += loss.item()
            nb += 1
            tr_correct += int((logits.argmax(1) == y).sum())
            tr_total += int(y.numel())
        sched.step()
        acc, recall, conf = evaluate(model, val_ld, device, classes)
        print(
            f"[train] epoch {epoch + 1}/{args.epochs} "
            f"loss={running / max(nb, 1):.4f} "
            f"train_acc={tr_correct / max(tr_total, 1):.3f} val_acc={acc:.3f}",
            flush=True,
        )
        if acc >= best_acc:
            best_acc, best_recall = acc, recall
            torch.save(
                {
                    "state_dict": model.state_dict(),
                    "classes": classes,
                    "num_frames": num_frames,
                    "hidden": args.hidden,
                    "val_acc": acc,
                },
                out / "caption_cnn_best.pt",
            )
            (out / "metrics.json").write_text(
                json.dumps(
                    {
                        "best_val_acc": best_acc,
                        "per_class_recall": recall,
                        "confusion": conf.tolist(),
                        "classes": classes,
                        "val_sessions": val_sessions,
                    },
                    indent=2,
                )
            )

    print(
        f"[train] best val_acc={best_acc:.3f}; per-class recall="
        f"{ {k: round(v, 3) for k, v in best_recall.items()} }",
        flush=True,
    )
    print(f"[train] best checkpoint -> {out / 'caption_cnn_best.pt'}", flush=True)


if __name__ == "__main__":
    main()
