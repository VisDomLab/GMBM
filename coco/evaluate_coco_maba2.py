import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from models.resnet import ResNet18
from data.coco_dataloader2 import create_dataloader
from tqdm import tqdm
import numpy as np
import pandas as pd
from itertools import product
import warnings
warnings.filterwarnings('ignore', category=SyntaxWarning)
warnings.filterwarnings('ignore', category=UserWarning)
warnings.filterwarnings('ignore', category=FutureWarning)
# Device configuration
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

train_dataloader = create_dataloader(
    image_dir='/home/ankur/Desktop/badd_celeba/code/data/coco/train2017',
    captions_path='/home/ankur/Desktop/badd_celeba/code/data/coco/annotations/captions_train2017.json',
    instances_path='/home/ankur/Desktop/badd_celeba/code/data/coco/annotations/instances_train2017.json'
)

print("Loading validation data...")
test_dataloader = create_dataloader(
    image_dir='/home/ankur/Desktop/badd_celeba/code/data/coco/val2017',
    captions_path='/home/ankur/Desktop/badd_celeba/code/data/coco/annotations/captions_val2017.json',
    instances_path='/home/ankur/Desktop/badd_celeba/code/data/coco/annotations/instances_val2017.json',
    shuffle=False
)

# Load Model
model = ResNet18(num_classes=2).to(device)
model.load_state_dict(torch.load('/home/ankur/Desktop/badd_celeba/code/best_resnet18_with_bcc.pth'))
model.eval()

def compute_bias(labels, bias1, bias2):
    bias_counts = {}

    # Create all possible subsets of biases including individual biases
    for g in np.unique(labels):
        for a1, a2 in product([True, False], repeat=2):
            if not (a1 or a2):
                continue  # Skip empty set

            # Form subset based on selection
            if a1 and a2:
                subset = list(zip(bias1, bias2))
            elif a1:
                subset = list(zip(bias1, [None]*len(bias2)))
            else:
                subset = list(zip([None]*len(bias1), bias2))

            unique_attributes = list(set(subset))

            for a in unique_attributes:
                count = np.sum((labels == g) & (np.array(subset) == a).all(axis=1))
                bias_counts[(g, a)] = count

    return bias_counts

def compute_maba(model, train_loader, test_loader, device):
    model.eval()
    all_train_labels, all_train_bias1, all_train_bias2 = [], [], []
    all_test_preds, all_test_bias1, all_test_bias2 = [], [], []

    with torch.no_grad():
        for images, labels, biases in tqdm(train_loader, desc="Processing Train Data"):
            all_train_labels.extend(labels.cpu().numpy())
            all_train_bias1.extend(biases[:, 0].cpu().numpy())
            all_train_bias2.extend(biases[:, 1].cpu().numpy())

        for images, labels, biases in tqdm(test_loader, desc="Processing Test Data"):
            images, labels = images.to(device), labels.to(device)
            outputs, _ = model(images)
            _, predicted = torch.max(outputs, 1)

            all_test_preds.extend(predicted.cpu().numpy())
            all_test_bias1.extend(biases[:, 0].cpu().numpy())
            all_test_bias2.extend(biases[:, 1].cpu().numpy())

    # Compute biases
    bias_train = compute_bias(np.array(all_train_labels), np.array(all_train_bias1), np.array(all_train_bias2))
    bias_test = compute_bias(np.array(all_test_preds), np.array(all_test_bias1), np.array(all_test_bias2))

    # Calculate total co-occurrences for normalization
    train_totals = {}
    test_totals = {}
    for _, a in bias_train:
        train_totals[a] = sum(bias_train.get((g, a), 0) for g in np.unique(all_train_labels))
    for _, a in bias_test:
        test_totals[a] = sum(bias_test.get((g, a), 0) for g in np.unique(all_test_preds))

    # Calculate bias amplification (Delta_gm) and print train and test biases
    delta_gm = {}
    print("\n[Base MABA]")
    print("Group | Bias1 | Bias2 | Train Bias (Num/Denom) | Test Bias (Num/Denom) | Delta_gm")
    print("-" * 90)

    for key in bias_train:
        group_label, attr_set = key
        train_num = bias_train[key]
        train_denom = train_totals[attr_set] if train_totals[attr_set] != 0 else 1
        train_bias = (train_num / train_denom)*100

        test_num = bias_test.get(key, 0)
        test_denom = test_totals.get(attr_set, 1)
        test_bias = (test_num / test_denom) *100 if test_denom != 0 else 0

        # Enforce the condition for Delta_gm
        if train_bias > (1 / len(np.unique(all_train_labels))):
            delta_gm[key] = test_bias - train_bias

        # Print biases for each group with numerator and denominator
        print(f"{group_label:^6} | {str(attr_set[0]) if attr_set[0] is not None else 'N/A':^6} | {str(attr_set[1]) if attr_set[1] is not None else 'N/A':^6} | {train_bias:.4f} ({train_num}/{train_denom}) | {test_bias:.4f} ({test_num}/{test_denom}) | {delta_gm.get(key, 0):.4f}")

    # Calculate mean and variance of bias amplification
    avg_delta_gm = np.mean(np.abs(list(delta_gm.values())))
    var_delta_gm = np.var(list(delta_gm.values()))

    return avg_delta_gm, var_delta_gm

def compute_maba_with_threshold(model, train_loader, test_loader, device, min_support=5):
    model.eval()
    all_train_labels, all_train_bias1, all_train_bias2 = [], [], []
    all_test_preds, all_test_bias1, all_test_bias2 = [], [], []

    with torch.no_grad():
        for images, labels, biases in tqdm(train_loader, desc="Processing Train Data"):
            all_train_labels.extend(labels.cpu().numpy())
            all_train_bias1.extend(biases[:, 0].cpu().numpy())
            all_train_bias2.extend(biases[:, 1].cpu().numpy())

        for images, labels, biases in tqdm(test_loader, desc="Processing Test Data"):
            images, labels = images.to(device), labels.to(device)
            outputs, _ = model(images)
            _, predicted = torch.max(outputs, 1)

            all_test_preds.extend(predicted.cpu().numpy())
            all_test_bias1.extend(biases[:, 0].cpu().numpy())
            all_test_bias2.extend(biases[:, 1].cpu().numpy())

    # Compute biases
    bias_train = compute_bias(np.array(all_train_labels), np.array(all_train_bias1), np.array(all_train_bias2))
    bias_test = compute_bias(np.array(all_test_preds), np.array(all_test_bias1), np.array(all_test_bias2))

    # Calculate total co-occurrences for normalization
    train_totals = {}
    test_totals = {}
    for _, a in bias_train:
        train_totals[a] = sum(bias_train.get((g, a), 0) for g in np.unique(all_train_labels))
    for _, a in bias_test:
        test_totals[a] = sum(bias_test.get((g, a), 0) for g in np.unique(all_test_preds))

    # Calculate bias amplification (Delta_gm) and print train and test biases
    delta_gm = {}
    print("\n[MABA with Minimum Support Threshold (min_support={})]".format(min_support))
    print("Group | Bias1 | Bias2 | Train Bias (Num/Denom) | Test Bias (Num/Denom) | Delta_gm")
    print("-" * 90)

    for key in bias_train:
        group_label, attr_set = key
        train_num = bias_train[key]
        train_denom = train_totals[attr_set] if train_totals[attr_set] != 0 else 1
        train_bias = (train_num / train_denom)*100

        test_num = bias_test.get(key, 0)
        test_denom = test_totals.get(attr_set, 1)
        test_bias = (test_num / test_denom) *100 if test_denom != 0 else 0

        # Enforce the condition for Delta_gm and minimum support
        if train_denom >= min_support and train_bias > (1 / len(np.unique(all_train_labels))):
            delta_gm[key] = test_bias - train_bias
            print(f"{group_label:^6} | {str(attr_set[0]) if attr_set[0] is not None else 'N/A':^6} | {str(attr_set[1]) if attr_set[1] is not None else 'N/A':^6} | {train_bias:.4f} ({train_num}/{train_denom}) | {test_bias:.4f} ({test_num}/{test_denom}) | {delta_gm.get(key, 0):.4f}")
        elif train_denom < min_support and train_bias > (1 / len(np.unique(all_train_labels))):
            print(f"{group_label:^6} | {str(attr_set[0]) if attr_set[0] is not None else 'N/A':^6} | {str(attr_set[1]) if attr_set[1] is not None else 'N/A':^6} | {train_bias:.4f} ({train_num}/{train_denom}) | {test_bias:.4f} ({test_num}/{test_denom}) | Skipped (Low Support: {train_denom})")
        elif train_bias <= (1 / len(np.unique(all_train_labels))):
            print(f"{group_label:^6} | {str(attr_set[0]) if attr_set[0] is not None else 'N/A':^6} | {str(attr_set[1]) if attr_set[1] is not None else 'N/A':^6} | {train_bias:.4f} ({train_num}/{train_denom}) | {test_bias:.4f} ({test_num}/{test_denom}) | Skipped (Train Bias <= Uniform)")
        else:
            print(f"{group_label:^6} | {str(attr_set[0]) if attr_set[0] is not None else 'N/A':^6} | {str(attr_set[1]) if attr_set[1] is not None else 'N/A':^6} | {train_bias:.4f} ({train_num}/{train_denom}) | {test_bias:.4f} ({test_num}/{test_denom}) | N/A")


    # Calculate mean and variance of bias amplification, considering only valid values
    valid_delta_gm_values = [v for v in delta_gm.values() if v is not None]
    avg_delta_gm = np.mean(np.abs(valid_delta_gm_values)) if valid_delta_gm_values else 0
    var_delta_gm = np.var(valid_delta_gm_values) if len(valid_delta_gm_values) > 1 else 0

    return avg_delta_gm, var_delta_gm

def compute_maba_weighted(model, train_loader, test_loader, device):
    model.eval()
    all_train_labels, all_train_bias1, all_train_bias2 = [], [], []
    all_test_preds, all_test_bias1, all_test_bias2 = [], [], []

    with torch.no_grad():
        for images, labels, biases in tqdm(train_loader, desc="Processing Train Data"):
            all_train_labels.extend(labels.cpu().numpy())
            all_train_bias1.extend(biases[:, 0].cpu().numpy())
            all_train_bias2.extend(biases[:, 1].cpu().numpy())

        for images, labels, biases in tqdm(test_loader, desc="Processing Test Data"):
            images, labels = images.to(device), labels.to(device)
            outputs, _ = model(images)
            _, predicted = torch.max(outputs, 1)

            all_test_preds.extend(predicted.cpu().numpy())
            all_test_bias1.extend(biases[:, 0].cpu().numpy())
            all_test_bias2.extend(biases[:, 1].cpu().numpy())

    # Compute biases
    bias_train = compute_bias(np.array(all_train_labels), np.array(all_train_bias1), np.array(all_train_bias2))
    bias_test = compute_bias(np.array(all_test_preds), np.array(all_test_bias1), np.array(all_test_bias2))

    # Calculate total co-occurrences for normalization
    train_totals = {}
    test_totals = {}
    for _, a in bias_train:
        train_totals[a] = sum(bias_train.get((g, a), 0) for g in np.unique(all_train_labels))
    for _, a in bias_test:
        test_totals[a] = sum(bias_test.get((g, a), 0) for g in np.unique(all_test_preds))

    # Calculate bias amplification (Delta_gm)
    delta_gm = {}
    print("\n[MABA with Weighted Averaging]")
    print("Group | Bias1 | Bias2 | Train Bias (Num/Denom) | Test Bias (Num/Denom) | Delta_gm")
    print("-" * 90)

    for key in bias_train:
        group_label, attr_set = key
        train_num = bias_train[key]
        train_denom = train_totals[attr_set] if train_totals[attr_set] != 0 else 1
        train_bias = (train_num / train_denom)*100

        test_num = bias_test.get(key, 0)
        test_denom = test_totals.get(attr_set, 1)
        test_bias = (test_num / test_denom) *100 if test_denom != 0 else 0

        # Enforce the condition for Delta_gm
        if train_bias > (1 / len(np.unique(all_train_labels))):
            delta_gm[key] = test_bias - train_bias
            print(f"{group_label:^6} | {str(attr_set[0]) if attr_set[0] is not None else 'N/A':^6} | {str(attr_set[1]) if attr_set[1] is not None else 'N/A':^6} | {train_bias:.4f} ({train_num}/{train_denom}) | {test_bias:.4f} ({test_num}/{test_denom}) | {delta_gm.get(key, 0):.4f}")
        else:
            print(f"{group_label:^6} | {str(attr_set[0]) if attr_set[0] is not None else 'N/A':^6} | {str(attr_set[1]) if attr_set[1] is not None else 'N/A':^6} | {train_bias:.4f} ({train_num}/{train_denom}) | {test_bias:.4f} ({test_num}/{test_denom}) | Skipped (Train Bias <= Uniform)")

    # Calculate weighted average of bias amplification
    weighted_sum_delta_gm = 0
    total_support = 0
    for key, delta in delta_gm.items():
        attr_set = key[1]
        train_denom = train_totals[attr_set] if train_totals[attr_set] != 0 else 1
        weighted_sum_delta_gm += np.abs(delta) * train_denom
        total_support += train_denom

    avg_delta_gm = weighted_sum_delta_gm / total_support if total_support > 0 else 0
    var_delta_gm = np.nan # Variance is not straightforward to calculate with weighted average of absolute values

    return avg_delta_gm, var_delta_gm

# Evaluate MABA
print("----------------------")
maba_base_avg, maba_base_var = compute_maba(model, train_dataloader, test_dataloader, device)
print(f"Base MABA Score: ({maba_base_avg:.4f}, {maba_base_var:.4f})")
print("----------------------")

min_support_threshold = 5
maba_threshold_avg, maba_threshold_var = compute_maba_with_threshold(model, train_dataloader, test_dataloader, device, min_support=min_support_threshold)
print(f"MABA Score with Minimum Support Threshold ({min_support_threshold}): ({maba_threshold_avg:.4f}, {maba_threshold_var:.4f})")
print("----------------------")

maba_weighted_avg, maba_weighted_var = compute_maba_weighted(model, train_dataloader, test_dataloader, device)
print(f"MABA Score with Weighted Averaging: ({maba_weighted_avg:.4f}, {maba_weighted_var})")
print("----------------------")
