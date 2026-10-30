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
from .steps import gradient_statistics, parameter_statistics, state_copy


def accuracy(logits: torch.Tensor, labels: torch.Tensor) -> float:
    return float((logits.argmax(1) == labels).float().mean().item())


def run_training(*, dataset: str = "digits", epochs: int = 10,
                 out: str = "artifacts/demo", seed: int = 42,
                 train_limit: int = 0, download: bool = False,
                 step_interval: int = 4, record_steps: bool = True) -> dict:
    if not (1 <= step_interval <= 10000):
        raise ValueError("step_interval must be 1..10000")
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
    step_events: list[dict] = []
    snapshot_refs: list[dict] = []
    global_step = 0
    if record_steps:
        # Step zero is exactly epoch_000.pt; avoid duplicate initial state.
        snapshot_refs.append({"step": 0, "epoch": 0, "batch": 0,
                              "path": "epoch_000.pt", "reason": "initialization"})

    def snapshot(epoch: int, train_loss: float | None, train_acc: float | None,
                 grad_stats: dict | None, previous: dict | None) -> None:
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
                        "grad_norm": None if grad_stats is None else grad_stats["last"],
                        "grad_norm_mean": None if grad_stats is None else grad_stats["mean"],
                        "grad_norm_max": None if grad_stats is None else grad_stats["max"],
                        "grad_norm_last": None if grad_stats is None else grad_stats["last"],
                        "grad_norm_batches": None if grad_stats is None else grad_stats["batches"],
                        "weight_delta": round(delta, 6)})
        print(f"epoch {epoch:02d}/{epochs} | val accuracy {metrics[-1]['val_acc']:.1%} | loss {metrics[-1]['val_loss']:.4f}", flush=True)

    snapshot(0, None, None, None, None)
    for epoch in range(1, epochs + 1):
        previous = {k: v.detach().clone() for k, v in model.state_dict().items()}
        model.train()
        running_loss = running_correct = running_total = 0
        grad_norms = []
        for batch_index, (xb, yb) in enumerate(loader, start=1):
            before_step = {n: p.detach().clone() for n, p in model.named_parameters()} if record_steps else None
            optimizer.zero_grad(set_to_none=True)
            prediction = model(xb)
            loss = nn.functional.cross_entropy(prediction, yb)
            loss.backward()
            grad_norm = float(nn.utils.clip_grad_norm_(model.parameters(), 5))
            grad_norms.append(grad_norm)
            optimizer.step()
            global_step += 1
            if record_steps:
                update = parameter_statistics(model, before_step)
                # Gradients after clipping: use preclip * scaling factor for each layer.
                layer_grad = gradient_statistics(model)
                clip_factor = min(1.0, 5.0 / (grad_norm + 1e-6))
                if clip_factor > 0:
                    layer_grad = {k: v / clip_factor for k, v in layer_grad.items()}
                is_last_batch = batch_index == len(loader)
                save_checkpoint = (global_step % step_interval == 0 or is_last_batch)
                path = f"step_{global_step:06d}.pt" if save_checkpoint else None
                if save_checkpoint:
                    torch.save(state_copy(model), output / path)
                    snapshot_refs.append({"step": global_step, "epoch": epoch,
                                          "batch": batch_index, "path": path,
                                          "reason": "epoch_end" if is_last_batch else "interval"})
                step_events.append({"step": global_step, "epoch": epoch, "batch": batch_index,
                                    "samples_seen": min(batch_index * loader.batch_size, len(loader.dataset)),
                                    "loss_before_update": round(float(loss.detach()), 8),
                                    "batch_accuracy_before_update": round(accuracy(prediction.detach(), yb), 8),
                                    "gradient_norm_preclip": round(grad_norm, 8),
                                    "gradient_norm_layers_preclip": {k: round(v, 8) for k,v in layer_grad.items()},
                                    "gradient_clipped": grad_norm > 5.0,
                                    "weight_delta_post_step": round(update["global"], 10),
                                    "weight_delta_layers": {k: round(v, 10) for k,v in update["by_layer"].items()},
                                    "learning_rate": optimizer.param_groups[0]["lr"],
                                    "has_checkpoint": save_checkpoint})
            running_loss += float(loss.detach()) * len(xb)
            running_correct += int((prediction.argmax(1) == yb).sum())
            running_total += len(xb)
        snapshot(epoch, round(running_loss / running_total, 6),
                 round(running_correct / running_total, 6),
                 {"mean": round(float(np.mean(grad_norms)), 6),
                  "max": round(max(grad_norms), 6),
                  "last": round(grad_norms[-1], 6), "batches": len(grad_norms)}, previous)

    if record_steps:
        # Write atomically so the browser never sees a half-written step log.
        payload = {"schema": 1, "step_interval": step_interval,
                   "global_steps": global_step, "events": step_events,
                   "snapshots": snapshot_refs,
                   "semantics": {
                       "loss_before_update": "Cross entropy on the current training minibatch, before optimizer.step",
                       "batch_accuracy_before_update": "Accuracy on that minibatch, before optimizer.step",
                       "gradient_norm_preclip": "Global L2 norm before clip_grad_norm_(max_norm=5)",
                       "gradient_norm_layers_preclip": "Per-module L2 gradient norms before clipping",
                       "weight_delta_post_step": "Exact L2 norm of post-minus-pre optimizer parameters",
                       "samples_seen": "Samples processed within the epoch; not unique across epochs",
                       "snapshots": "State captured immediately after optimizer.step; epoch-end always retained",
                   }}
        temporary = output / "steps.json.tmp"
        temporary.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
        temporary.replace(output / "steps.json")
    else:
        (output / "steps.json").unlink(missing_ok=True)

    manifest = {"run_id": f"{dataset}-seed{seed}-e{epochs}", "dataset": dataset,
                "source": source, "classes": labels, "channels": channels,
                "model": "InspectableCNN", "input_size": [16, 16],
                "train_count": len(train_ids), "validation_count": len(val_ids),
                "seed": seed, "epochs": epochs, "metrics": metrics, "samples": samples,
                "step_observatory": {"available": record_steps, "schema": 1,
                    "global_steps": global_step if record_steps else 0,
                    "saved_snapshots": len(snapshot_refs), "step_interval": step_interval},
                "training_config": {"batch_size": 128, "optimizer": "Adam", "learning_rate": 0.003,
                    "clip_grad_norm": 5.0, "step_snapshot_policy": "interval + every epoch end"},
                "gradient_norm_definition": "Global L2 norm before clipping (clip threshold=5), mean/max/last over minibatches",
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
    parser.add_argument("--step-interval", type=int, default=4,
                        help="Persist post-optimizer state every N steps plus epoch ends")
    parser.add_argument("--no-step-records", action="store_true", help="Disable V0.3 step metrics and snapshots")
    args = parser.parse_args()
    if args.epochs < 1 or args.epochs > 200:
        parser.error("epochs must be 1..200")
    kwargs = vars(args)
    kwargs["record_steps"] = not kwargs.pop("no_step_records")
    run_training(**kwargs)


if __name__ == "__main__":
    main()