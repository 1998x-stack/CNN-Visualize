"""Reproducible training snapshots for the time-travel visualization."""
from __future__ import annotations
import argparse
import json
import random
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
from .data import get_dataset
from .model import InspectableCNN


def accuracy(logits: torch.Tensor, labels: torch.Tensor) -> float:
    return float((logits.argmax(1) == labels).float().mean().item())


def run_training(*, dataset: str = "digits", epochs: int = 10,
                 out: str = "artifacts/demo", seed: int = 42,
                 train_limit: int = 0, download: bool = False) -> dict:
    torch.set_num_threads(min(4, torch.get_num_threads()))
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    x, y, train_ids, val_ids, labels, source = get_dataset(dataset, download=download, seed=seed)
    if train_limit:
        train_ids = train_ids[:train_limit]
    channels = int(x.shape[1])
    model = InspectableCNN(channels=channels)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.003)
    loader = DataLoader(TensorDataset(x[train_ids], y[train_ids]), batch_size=128, shuffle=True,
                        generator=torch.Generator().manual_seed(seed), num_workers=0)
    output = Path(out)
    output.mkdir(parents=True, exist_ok=True)
    samples = []
    # Stratified gallery, four items per class, from the validation split ONLY.
    for class_index, label in enumerate(labels):
        matching = [int(k) for k in val_ids if int(y[k]) == class_index]
        for i in matching[:4]:
            samples.append({"id": str(i), "label": label, "label_index": class_index})
    metrics = []

    def snapshot(epoch: int, train_loss: float | None, train_acc: float | None,
                 grad_norm: float | None, previous: dict | None) -> None:
        model.eval()
        total_loss = total_correct = count = 0
        with torch.no_grad():
            for part in torch.arange(len(val_ids)).split(512):
                xb, yb = x[val_ids[part]], y[val_ids[part]]
                logit = model(xb)
                total_loss += float(nn.functional.cross_entropy(logit, yb, reduction="sum"))
                total_correct += int((logit.argmax(1) == yb).sum())
                count += len(yb)
        current = {k: v.detach().clone().cpu() for k, v in model.state_dict().items()}
        torch.save(current, output / f"epoch_{epoch:03d}.pt")
        delta = 0.0 if previous is None else sum(float(torch.sum((v - previous[k]) ** 2)) for k, v in current.items()) ** .5
        metrics.append({"epoch": epoch, "train_loss": train_loss, "train_acc": train_acc,
                        "val_loss": round(total_loss / max(count, 1), 6),
                        "val_acc": round(total_correct / max(count, 1), 6),
                        "grad_norm": grad_norm, "weight_delta": round(delta, 6)})
        print(f"epoch {epoch:02d}/{epochs} | val accuracy {metrics[-1]['val_acc']:.1%} | loss {metrics[-1]['val_loss']:.4f}", flush=True)

    snapshot(0, None, None, None, None)
    for epoch in range(1, epochs + 1):
        previous = {k: v.detach().clone() for k, v in model.state_dict().items()}
        model.train()
        running_loss = running_correct = running_total = last_norm = 0
        for xb, yb in loader:
            optimizer.zero_grad(set_to_none=True)
            prediction = model(xb)
            loss = nn.functional.cross_entropy(prediction, yb)
            loss.backward()
            last_norm = float(nn.utils.clip_grad_norm_(model.parameters(), 5))
            optimizer.step()
            running_loss += float(loss.detach()) * len(xb)
            running_correct += int((prediction.argmax(1) == yb).sum())
            running_total += len(xb)
        snapshot(epoch, round(running_loss / running_total, 6),
                 round(running_correct / running_total, 6), round(last_norm, 6), previous)

    manifest = {"run_id": f"{dataset}-seed{seed}-e{epochs}", "dataset": dataset,
                "source": source, "classes": labels, "channels": channels,
                "model": "InspectableCNN", "input_size": [16, 16],
                "train_count": len(train_ids), "validation_count": len(val_ids),
                "seed": seed, "epochs": epochs, "metrics": metrics, "samples": samples,
                "notes": ["Every checkpoint is a real trained PyTorch state_dict, including epoch 0 (random initialization).",
                          "Gallery samples belong to the held-out validation/test split.",
                          "DeconvNet projects one peak activation and does not prove class attribution."]}
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description="Train an inspectable CNN and save every epoch")
    parser.add_argument("--dataset", choices=["digits", "cifar10"], default="digits")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--out", default="artifacts/demo")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--train-limit", type=int, default=0)
    parser.add_argument("--download", action="store_true", help="Allow torchvision download for CIFAR10")
    args = parser.parse_args()
    if args.epochs < 1 or args.epochs > 200:
        parser.error("epochs must be 1..200")
    run_training(**vars(args))


if __name__ == "__main__":
    main()