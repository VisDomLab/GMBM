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
        # att2 = "Blond_Hair"
        csv_df.loc[csv_df["Male"] == -1, "Male"] = 0
        csv_df.loc[csv_df[att1] == -1, att1] = 0
        csv_df.loc[csv_df[att2] == -1, att2] = 0
        self.y = csv_df["Male"].values
        self.bias1 = csv_df[att1].values
        self.bias2 = csv_df[att2].values
        self.transform = transform
        self.get_path = get_path

        # Assuming labels, biases1, and biases2 are torch tensors
        # Convert them to numpy arrays
        # labels_np = np.array(self.y)
        # biases1_np = np.array(self.bias1)
        # biases2_np = np.array(self.bias2)

        # Example DataFrame with labels, biases1, and biases2
        data = {"labels": self.y, "biases1": self.bias1, "biases2": self.bias2}
        df = pd.DataFrame(data)

        # Compute co-occurrence between labels and biases1
        cooccurrence_1 = pd.crosstab(df["labels"], df["biases1"], normalize="index")

        # Compute co-occurrence between labels and biases2
        cooccurrence_2 = pd.crosstab(df["labels"], df["biases2"], normalize="index")

        # Compute co-occurrence between biases1 and biases2
        # Concatenate biases1 and biases2 into a single column
        df["biases_combined"] = df["biases1"] + df["biases2"]

        # Compute co-occurrence matrix between labels and combined biases
        cooccurrence_3 = pd.crosstab(
            df["labels"], df["biases_combined"], normalize="index"
        )

        # print("Co-occurrence between labels and biases1:")
        # print(cooccurrence_1)

        # print("\nCo-occurrence between labels and biases2:")
        # print(cooccurrence_2)

        # print("\nCo-occurrence between biases1 and biases2:")
        # print(cooccurrence_3)
        # print(cooccurrence_matrix_1, cooccurrence_matrix_2)

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
        # Load the image and corresponding label
        img_path = os.path.join(self.data_dir, self.img_names[index])
        img = Image.open(img_path)
        img = self.transform(img) if self.transform else img

        label = self.y[index]

        # Prepare the return values
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
            # transforms.RandomCrop((crop, crop)),
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
        # usecols=["Male", "Wearing_Lipstick", "Wearing_Earrings"],
    )
    # Load the mapping file

    # print(df.head)
    # print("Number of samples: ", df.shape[0])

    train_set, test_set = train_test_split(df, train_size=split, random_state=42)

    train_data = CelebaDataset(train_set, data_dir, transform_train)
    test_data = CelebaDataset(test_set, data_dir, transform_test)
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
