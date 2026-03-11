import torch
import numpy as np
import albumentations as A
from albumentations.pytorch import ToTensorV2


def default_tta_transforms(image_size=(224, 224)):
    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]
    base = A.Compose([
        A.Resize(*image_size),
        A.Normalize(mean=mean, std=std),
        ToTensorV2()
    ])

    hflip = A.Compose([
        A.Resize(*image_size),
        A.HorizontalFlip(p=1.0),
        A.Normalize(mean=mean, std=std),
        ToTensorV2()
    ])

    slight_rotate = A.Compose([
        A.Resize(*image_size),
        A.Rotate(limit=10, p=1.0),
        A.Normalize(mean=mean, std=std),
        ToTensorV2()
    ])

    return [base, hflip, slight_rotate]


@torch.no_grad()
def predict_with_tta(model, image_np, transforms_list, device):
    all_probs = []

    for tfm in transforms_list:
        augmented = tfm(image=image_np)["image"]
        batch = augmented.unsqueeze(0).to(device)
        logits = model(batch)
        probs = torch.softmax(logits, dim=1).cpu().numpy()[0]
        all_probs.append(probs)

    return np.mean(all_probs, axis=0)


@torch.no_grad()
def evaluate_with_tta(model, dataset, transforms_list, device):
    from sklearn.metrics import roc_auc_score

    model.eval()
    n = len(dataset)
    all_probs = []
    all_targets = np.empty(n, dtype=np.int64)

    for idx in range(n):
        image, target = dataset[idx]

        if isinstance(image, torch.Tensor):
            raise TypeError(
                "TTA expects raw uint8 numpy arrays from the dataset. "
                "Create the dataset with transform=None so that images "
                "are returned as uint8 numpy arrays."
            )

        if not isinstance(image, np.ndarray) or image.dtype != np.uint8:
            raise TypeError(
                f"TTA expects uint8 numpy arrays, got {type(image)} "
                f"with dtype={getattr(image, 'dtype', 'N/A')}"
            )

        probs = predict_with_tta(model, image, transforms_list, device)
        all_probs.append(probs)
        all_targets[idx] = target

    all_probs = np.array(all_probs)

    try:
        roc_auc = roc_auc_score(
            all_targets, all_probs, multi_class="ovr", average="macro"
        )
    except ValueError:
        roc_auc = float("nan")

    return {"roc_auc": roc_auc}