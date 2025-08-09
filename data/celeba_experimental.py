import os
import pandas as pd
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
from sklearn.model_selection import train_test_split
from datasets.utils import (
    TwoCropTransform,
    get_confusion_matrix,
    get_unsup_confusion_matrix,
)
import torch
import numpy as np


class CelebaDataset(Dataset):
    """Custom Dataset for loading CelebA face images"""

    def __init__(
        self,
        csv_df,
        data_dir,
        transform=None,
        get_path=False,
    ):
        self.data_dir = data_dir

        self.img_names = csv_df.index.values
        att1 = "Wearing_Lipstick"
        att2 = "Heavy_Makeup"
        csv_df.loc[csv_df["Male"] == -1, "Male"] = 0
        csv_df.loc[csv_df[att1] == -1, att1] = 0
        csv_df.loc[csv_df[att2] == -1, att2] = 0
        self.y = csv_df["Male"].values
        self.bias1 = csv_df[att1].values
        self.bias2 = csv_df[att2].values
        self.transform = transform
        self.get_path = get_path

        (
            self.confusion_matrix_org,
            self.confusion_matrix,
            self.confusion_matrix_by,
        ) = get_confusion_matrix(
            num_classes=2,
            targets=torch.from_numpy(self.y),
            biases=torch.from_numpy(self.bias1),
        )

    def __getitem__(self, index):
        img_path = os.path.join(self.data_dir, self.img_names[index])
        img = Image.open(img_path)
        img = self.transform(img) if self.transform else img

        label = self.y[index]

        return_values = [img, label, self.bias1[index], self.bias2[index]]
        if self.get_path:
            return_values.append(img_path)

        return tuple(return_values)

    def __len__(self):
        return self.y.shape[0]


def get_dataloaders(
    data_dir, csv_dir, precrop=256, crop=224, bs=64, nw=4, split=0.9, twocrop=False
):
    transform_train = transforms.Compose(
        [
            transforms.Resize((crop, crop)),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
        ]
    )
    if twocrop:
        transform_cont_train = TwoCropTransform(transform_train)

    transform_test = transforms.Compose(
        [
            transforms.Resize((crop, crop)),
            transforms.ToTensor(),
            transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
        ]
    )

    df = pd.read_csv(
        csv_dir,
        sep="\s+",
        skiprows=1,
    )
    df.loc[df["Male"] == -1, "Male"] = 0
    df.loc[df["Wearing_Lipstick"] == -1, "Wearing_Lipstick"] = 0
    df.loc[df["Heavy_Makeup"] == -1, "Heavy_Makeup"] = 0

    # Randomly split the dataset into train and test
    train_set, test_set = train_test_split(df, train_size=split, random_state=42)

    # Function to balance the test set only
    def balance_test_set(original_df, desired_test_size):
        # Get unique combinations of Male, Wearing_Lipstick, Heavy_Makeup
        groups = original_df.groupby(['Male', 'Wearing_Lipstick', 'Heavy_Makeup'])
        test_indices = []

        # Desired number of samples per group in test set
        test_size_per_group = desired_test_size // len(groups)

        for name, group in groups:
            # Sample test set
            test_sample = group.sample(n=min(test_size_per_group, len(group)), random_state=42)
            test_indices.extend(test_sample.index)

        # Create balanced test set
        balanced_test_set = original_df.loc[test_indices]

        # Verify co-occurrence in test set
        cooccurrence_lipstick = pd.crosstab(balanced_test_set["Male"], balanced_test_set["Wearing_Lipstick"], normalize="index")
        cooccurrence_makeup = pd.crosstab(balanced_test_set["Male"], balanced_test_set["Heavy_Makeup"], normalize="index")

        return balanced_test_set

    # Calculate desired test set size
    desired_test_size = int(len(df) * (1 - split))

    # Balance the test set using the original dataframe
    balanced_test_set = balance_test_set(df, desired_test_size)

    train_data = CelebaDataset(train_set, data_dir, transform_train)
    test_data = CelebaDataset(balanced_test_set, data_dir, transform_test)
    train_loader = DataLoader(
        dataset=train_data, batch_size=bs, shuffle=True, num_workers=nw
    )
    test_loader = DataLoader(
        dataset=test_data, batch_size=bs, shuffle=False, num_workers=nw
    )
    if twocrop:
        cont_train_data = CelebaDataset(train_set, data_dir, transform_cont_train)
        cont_train_loader = DataLoader(
            dataset=cont_train_data, batch_size=bs, shuffle=True, num_workers=nw
        )
        return train_loader, cont_train_loader, test_loader
    else:
        return train_loader, test_loader