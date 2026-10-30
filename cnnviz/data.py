"""Built-in offline Digits and optional downloaded CIFAR-10 datasets."""
from __future__ import annotations
import numpy as np
import torch
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from torch.nn import functional as F

LABELS_DIGITS = [str(i) for i in range(10)]
LABELS_CIFAR = ["airplane", "automobile", "bird", "cat", "deer", "dog", "frog", "horse", "ship", "truck"]


def get_dataset(name: str, root: str = "data", download: bool = False, seed: int = 42):
    """Return CPU tensors, labels and fixed stratified train/validation indices."""
    if name == "digits":
        ds = load_digits()
        x = torch.from_numpy(np.ascontiguousarray(ds.images.astype("float32") / 16.0))[:, None]
        x = F.interpolate(x, size=(16, 16), mode="nearest")
        y = torch.tensor(ds.target, dtype=torch.long)
        labels = LABELS_DIGITS
        source = "sklearn.load_digits (UCI Optical Recognition; 1797 samples; native 8x8)"
        x_train, x_val = train_test_split(
            np.arange(len(y)), test_size=0.2, random_state=seed,
            stratify=ds.target,
        )
    elif name == "cifar10":
        from torchvision.datasets import CIFAR10
        from torchvision.transforms import functional as TF
        # Transform to 16x16 so traces share an identical layout with Digits.
        train = CIFAR10(root=root, train=True, download=download)
        test = CIFAR10(root=root, train=False, download=download)
        combined = list(zip(train.data, train.targets)) + list(zip(test.data, test.targets))
        # Keep full resolution in the source dataset; only the model input is reduced.
        x = torch.stack([F.interpolate(TF.to_tensor(image)[None], size=(16, 16), mode="bilinear", align_corners=False)[0] for image, _ in combined])
        y = torch.tensor([label for _, label in combined], dtype=torch.long)
        x_train = np.arange(len(train))
        x_val = np.arange(len(train), len(train) + len(test))
        labels = LABELS_CIFAR
        source = "torchvision.CIFAR10 (50k train / 10k official test; resized to 16x16)"
    else:
        raise ValueError("Supported datasets: digits, cifar10")
    return x, y, np.asarray(x_train), np.asarray(x_val), labels, source