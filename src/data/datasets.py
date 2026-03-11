import os
import cv2
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader


def label_to_idx(labels):
    return {lbl: idx for idx, lbl in enumerate(sorted(set(labels)))}


class GestureDataset(Dataset):
    def __init__(self, dataframe, images_dir, label_to_idx, image_size=(64, 64), transform=None):
        df = dataframe.reset_index(drop=True)
        self.images_dir = images_dir
        self.image_size = image_size
        self.transform = transform

        self.label_to_idx = label_to_idx

        self.num_classes = len(self.label_to_idx)
        self.img_ids = df["img_IDS"].tolist()
        self.labels = df["Label"].tolist()


    def __len__(self):
        return len(self.img_ids)


    def __getitem__(self, idx):
        img_path = os.path.join(self.images_dir, f"{self.img_ids[idx]}.jpg")
        image = cv2.imread(img_path)

        if image is None:
            raise FileNotFoundError(f"Image not found: {img_path}")

        h, w = self.image_size
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = cv2.resize(image, (w, h))

        if self.transform is not None:
            image = self.transform(image=image)["image"]

        target = self.label_to_idx[self.labels[idx]]

        return image, target


def create_dataloaders(train_df, val_df, images_dir, image_size=(64, 64),
                       train_transform=None, val_transform=None,
                       batch_size=32, num_workers=0):
    all_labels = set(train_df["Label"].unique()) | set(val_df["Label"].unique())
    lti = label_to_idx(all_labels)

    train_ds = GestureDataset(
        train_df,
        images_dir=images_dir,
        label_to_idx=lti,
        image_size=image_size,
        transform=train_transform,
    )

    val_ds = GestureDataset(
        val_df,
        images_dir=images_dir,
        label_to_idx=lti,
        image_size=image_size,
        transform=val_transform,
    )

    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        drop_last=True,
    )

    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
    )

    return train_loader, val_loader, lti