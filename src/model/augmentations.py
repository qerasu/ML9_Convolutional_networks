import torch
import numpy as np


def _mix_targets(targets, perm, lam, num_classes, device):
    targets_onehot = torch.zeros(targets.size(0), num_classes, device=device)
    targets_onehot.scatter_(1, targets.long().unsqueeze(1), 1.0)

    targets_perm = targets_onehot[perm]
    
    return lam * targets_onehot + (1 - lam) * targets_perm


def mixup(images, targets, alpha=1.0, num_classes=None):
    if num_classes is None:
        raise ValueError("num_classes must be specified for MixUp")

    batch_size = images.size(0)
    lam = np.random.beta(alpha, alpha) if alpha > 0 else 1.0

    perm = torch.randperm(batch_size, device=images.device)

    mixed_images = lam * images + (1 - lam) * images[perm]
    mixed_targets = _mix_targets(targets, perm, lam, num_classes, images.device)

    return mixed_images, mixed_targets


def _rand_bbox(h, w, lam):
    cut_ratio = np.sqrt(1.0 - lam)
    cut_h = max(1, int(h * cut_ratio))
    cut_w = max(1, int(w * cut_ratio))

    cy = np.random.randint(h)
    cx = np.random.randint(w)

    y1 = np.clip(cy - cut_h // 2, 0, h)
    y2 = np.clip(cy + cut_h // 2, 0, h)
    x1 = np.clip(cx - cut_w // 2, 0, w)
    x2 = np.clip(cx + cut_w // 2, 0, w)

    if y2 - y1 < 1:
        y2 = min(y1 + 1, h)
    if x2 - x1 < 1:
        x2 = min(x1 + 1, w)

    return y1, y2, x1, x2


def cutmix(images, targets, alpha=1.0, num_classes=None):
    if num_classes is None:
        raise ValueError("num_classes must be specified for CutMix")

    batch_size = images.size(0)
    lam = np.random.beta(alpha, alpha) if alpha > 0 else 1.0

    perm = torch.randperm(batch_size, device=images.device)

    _, _, h, w = images.shape
    y1, y2, x1, x2 = _rand_bbox(h, w, lam)

    mixed_images = images.clone()
    mixed_images[:, :, y1:y2, x1:x2] = images[perm, :, y1:y2, x1:x2]

    lam_adjusted = 1 - ((y2 - y1) * (x2 - x1)) / (h * w)
    mixed_targets = _mix_targets(targets, perm, lam_adjusted, num_classes, images.device)

    return mixed_images, mixed_targets


def make_mixup_fn(num_classes, alpha=1.0, mode="mixup"):
    def _fn(images, targets):
        if mode == "mixup":
            return mixup(images, targets, alpha=alpha, num_classes=num_classes)
        elif mode == "cutmix":
            return cutmix(images, targets, alpha=alpha, num_classes=num_classes)
        elif mode == "random":
            if np.random.rand() < 0.5:
                return mixup(images, targets, alpha=alpha, num_classes=num_classes)
            else:
                return cutmix(images, targets, alpha=alpha, num_classes=num_classes)
        else:
            raise ValueError(f"Unknown mode: {mode!r}")

    return _fn