import torch
import cv2
import os

import pandas as pd
import numpy as np

from config import CFG
from torch.utils.data import Dataset, DataLoader

import albumentations as A
from albumentations.core.composition import Compose, OneOf
from albumentations.pytorch import ToTensorV2


class SorghumDataset(Dataset):
    def __init__(self, df, transform=None):
        self.image_path = df['file_path'].values
        self.labels = df["cultivar_index"].values
        self.transform = transform

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        #         image_id = self.image_id[idx]
        label = torch.tensor(self.labels[idx], dtype=torch.float32)

        image_path = self.image_path[idx]
        image = cv2.imread(image_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        augmented = self.transform(image=image)
        image = augmented['image']
        return {'image': image, 'target': label}


def get_transform(phase: str):
    """
    Args:
        phase: Determines whether to get transformation set for train or test

    Returns: returns the transformation

    """
    if phase == "train":
        return Compose([
            A.RandomResizedCrop(height=CFG.img_size,
                                width=CFG.img_size),
            A.Flip(p=0.5),
            A.RandomRotate90(p=0.5),
            A.ShiftScaleRotate(p=0.5),
            A.HueSaturationValue(p=0.5),
            A.OneOf([
                A.RandomBrightnessContrast(p=0.5),
                A.RandomGamma(p=0.5),
            ], p=0.5),
            A.OneOf([
                A.Blur(p=0.1),
                A.GaussianBlur(p=0.1),
                A.MotionBlur(p=0.1),
            ], p=0.1),
            A.OneOf([
                A.GaussNoise(p=0.1),
                A.ISONoise(p=0.1),
                A.GridDropout(ratio=0.5, p=0.2),
                A.CoarseDropout(max_holes=16,
                                min_holes=8,
                                max_height=16,
                                max_width=16,
                                min_height=8,
                                min_width=8,
                                p=0.2)
            ], p=0.2),
            A.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225],
            ),
            ToTensorV2(),
        ])
    elif phase == "valid":
        return Compose([
            A.Resize(height=CFG.img_size,
                     width=CFG.img_size),
            A.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225],
            ),
            ToTensorV2(),
        ])


df_all = pd.read_csv(CFG.class_mapping_file).dropna(inplace=False)
print(len(df_all))

unique_cultivars = list(df_all["cultivar"].unique())
num_classes = len(unique_cultivars)

CFG.num_classes = num_classes
print(num_classes)


df_all["file_path"] = df_all["image"].apply(lambda image: os.path.join(CFG.train_dir, image))
df_all["cultivar_index"] = df_all["cultivar"].map(lambda item: unique_cultivars.index(item))
df_all["is_exist"] = df_all["file_path"].apply(lambda file_path: os.path.exists(file_path))
df_all = df_all[df_all.is_exist == True]
df_all.head()
