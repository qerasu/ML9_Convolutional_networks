import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from sklearn.metrics import roc_auc_score


def _compute_loss(logits, targets, criterion):
    if targets.ndim == 1:
        return criterion(logits, targets)

    log_probs = F.log_softmax(logits, dim=1)
    return -torch.sum(targets * log_probs) / logits.size(0)


def _get_device():
    if torch.cuda.is_available():
        return torch.device("cuda")

    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")

    return torch.device("cpu")


def train_one_epoch(model, loader, optimizer, criterion, device,
                    mixup_fn=None, scheduler=None):
    model.train()
    total_loss = 0.0
    n_samples = 0

    for images, targets in loader:
        images = images.to(device)
        targets = targets.to(device)
        bs = targets.size(0)

        if mixup_fn is not None:
            images, targets = mixup_fn(images, targets)

        optimizer.zero_grad()
        logits = model(images)
        loss = _compute_loss(logits, targets, criterion)
        loss.backward()
        optimizer.step()

        if scheduler is not None:
            scheduler.step()

        total_loss += loss.item() * bs
        n_samples += bs

    return total_loss / max(n_samples, 1)


@torch.no_grad()
def evaluate(model, loader, criterion, device):
    model.eval()
    total_loss = 0.0
    n_samples = 0

    all_probs = []
    all_targets = []

    for images, targets in loader:
        images = images.to(device)
        targets = targets.to(device)
        bs = targets.size(0)

        logits = model(images)
        loss = criterion(logits, targets)
        total_loss += loss.item() * bs
        n_samples += bs

        probs = torch.softmax(logits, dim=1).cpu().numpy()
        all_probs.append(probs)
        all_targets.append(targets.cpu().numpy())

    all_probs = np.concatenate(all_probs, axis=0)
    all_targets = np.concatenate(all_targets, axis=0)

    try:
        roc_auc = roc_auc_score(
            all_targets, all_probs, multi_class="ovr", average="macro"
        )
    except ValueError:
        roc_auc = float("nan")

    return {
        "loss": total_loss / max(n_samples, 1),
        "roc_auc": roc_auc,
    }


def train_model(
    model,
    train_loader,
    val_loader,
    epochs=10,
    lr=1e-3,
    device=None,
    optimizer=None,
    criterion=None,
    scheduler=None,
    scheduler_step_per_batch=False,
    mixup_fn=None,
):
    if device is None:
        device = _get_device()

    if optimizer is not None and lr != 1e-3:
        raise ValueError(
            "Both 'optimizer' and non-default 'lr' were provided. "
            "'lr' is ignored when 'optimizer' is supplied.",
        )

    print(f"Using device: {device}")
    model = model.to(device)

    if optimizer is None:
        optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    if criterion is None:
        criterion = nn.CrossEntropyLoss()

    batch_scheduler = scheduler if scheduler_step_per_batch else None
    epoch_scheduler = scheduler if not scheduler_step_per_batch else None

    history = []

    for epoch in range(1, epochs + 1):
        train_loss = train_one_epoch(
            model, train_loader, optimizer, criterion, device,
            mixup_fn=mixup_fn, scheduler=batch_scheduler,
        )
        val_metrics = evaluate(model, val_loader, criterion, device)

        if epoch_scheduler is not None:
            epoch_scheduler.step()

        record = {
            "epoch": epoch,
            "train_loss": train_loss,
            "val_loss": val_metrics["loss"],
            "val_roc_auc": val_metrics["roc_auc"],
        }
        history.append(record)

        print(
            f"Epoch {epoch:3d}/{epochs} | "
            f"train_loss={train_loss:.4f} | "
            f"val_loss={val_metrics['loss']:.4f} | "
            f"val_roc_auc={val_metrics['roc_auc']:.4f}"
        )

    return {"model": model, "history": history}