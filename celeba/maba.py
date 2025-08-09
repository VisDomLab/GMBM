import argparse
import datetime
import logging
import os
import time
from pathlib import Path
import pandas as pd
import numpy as np
from itertools import product
import torch
from torch import nn, optim
import warnings
from datasets.biased_mnist import get_color_mnist
from models.simple_conv import SimpleConvNet
from models.resnet import ResNet18
from utils.logging import set_logging
from utils.utils import (
    AverageMeter,
    MultiDimAverageMeter,
    accuracy,
    load_model,
    pretty_dict,
    save_model,
    set_seed)
from tqdm import tqdm

warnings.filterwarnings('ignore', category=SyntaxWarning)
warnings.filterwarnings('ignore', category=UserWarning)
warnings.filterwarnings('ignore', category=FutureWarning)
def parse_option():
    parser = argparse.ArgumentParser()
    parser.add_argument("--gpu", type=int, default=0)
    opt = parser.parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = str(opt.gpu)
    return opt


def set_model(opt):
    model = SimpleConvNet()
    model.cuda()
    model.load_state_dict(load_model("/home/ankur/Desktop/BAdd_Bias_Mitigation/code/results/badd-color_mnist_corrA0.99-corrB0.99-test-lr0.001-bs128-seed1--alpha0.1--beta0.1/checkpoints/fine_tuned_model_grad_sup.pth"))
    return model

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

    # Extract data from loaders
    with torch.no_grad():
        for images, labels, bias1, bias2, _ in train_loader:
            all_train_labels.extend(labels.cpu().numpy())
            all_train_bias1.extend(bias1.cpu().numpy())
            all_train_bias2.extend(bias2.cpu().numpy())

        for images, _, bias1, bias2, _ in test_loader:
            images = images.to(device)
            output, _ = model(images)
            preds = output.data.max(1, keepdim=True)[1].squeeze(1).cpu().numpy()
            all_test_preds.extend(preds)
            all_test_bias1.extend(bias1.cpu().numpy())
            all_test_bias2.extend(bias2.cpu().numpy())

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
    # print("\n[Base MABA]")
    # print("Group | Bias1 | Bias2 | Train Bias (Num/Denom) | Test Bias (Num/Denom) | Delta_gm")
    # print("-" * 90)

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
        #print(f"{group_label:^6} | {str(attr_set[0]) if attr_set[0] is not None else 'N/A':^6} | {str(attr_set[1]) if attr_set[1] is not None else 'N/A':^6} | {train_bias:.4f} ({train_num}/{train_denom}) | {test_bias:.4f} ({test_num}/{test_denom}) | {delta_gm.get(key, 0):.4f}")

    # Calculate mean and variance of bias amplification
    avg_delta_gm = np.mean(np.abs(list(delta_gm.values())))
    var_delta_gm = np.var(list(delta_gm.values()))

    return avg_delta_gm, var_delta_gm

def compute_maba_with_threshold(model, train_loader, test_loader, device, min_support=5):
    model.eval()
    all_train_labels, all_train_bias1, all_train_bias2 = [], [], []
    all_test_preds, all_test_bias1, all_test_bias2 = [], [], []

    # Extract data from loaders
    with torch.no_grad():
        for images, labels, bias1, bias2,_ in train_loader:
            all_train_labels.extend(labels.cpu().numpy())
            all_train_bias1.extend(bias1.cpu().numpy())
            all_train_bias2.extend(bias2.cpu().numpy())

        for images, _, bias1, bias2, _ in test_loader:
            images = images.to(device)
            output, _ = model(images)
            preds = output.data.max(1, keepdim=True)[1].squeeze(1).cpu().numpy()
            all_test_preds.extend(preds)
            all_test_bias1.extend(bias1.cpu().numpy())
            all_test_bias2.extend(bias2.cpu().numpy())

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
    # print("\n[MABA with Minimum Support Threshold (min_support={})]".format(min_support))
    # print("Group | Bias1 | Bias2 | Train Bias (Num/Denom) | Test Bias (Num/Denom) | Delta_gm")
    # print("-" * 90)

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
            #print(f"{group_label:^6} | {str(attr_set[0]) if attr_set[0] is not None else 'N/A':^6} | {str(attr_set[1]) if attr_set[1] is not None else 'N/A':^6} | {train_bias:.4f} ({train_num}/{train_denom}) | {test_bias:.4f} ({test_num}/{test_denom}) | {delta_gm.get(key, 0):.4f}")
        elif train_denom < min_support and train_bias > (1 / len(np.unique(all_train_labels))):
            #print(f"{group_label:^6} | {str(attr_set[0]) if attr_set[0] is not None else 'N/A':^6} | {str(attr_set[1]) if attr_set[1] is not None else 'N/A':^6} | {train_bias:.4f} ({train_num}/{train_denom}) | {test_bias:.4f} ({test_num}/{test_denom}) | Skipped (Low Support: {train_denom})")
            pass
        elif train_bias <= (1 / len(np.unique(all_train_labels))):
            #print(f"{group_label:^6} | {str(attr_set[0]) if attr_set[0] is not None else 'N/A':^6} | {str(attr_set[1]) if attr_set[1] is not None else 'N/A':^6} | {train_bias:.4f} ({train_num}/{train_denom}) | {test_bias:.4f} ({test_num}/{test_denom}) | Skipped (Train Bias <= Uniform)")
            pass            
        else:
            #print(f"{group_label:^6} | {str(attr_set[0]) if attr_set[0] is not None else 'N/A':^6} | {str(attr_set[1]) if attr_set[1] is not None else 'N/A':^6} | {train_bias:.4f} ({train_num}/{train_denom}) | {test_bias:.4f} ({test_num}/{test_denom}) | N/A")
            pass

    # Calculate mean and variance of bias amplification, considering only valid values
    valid_delta_gm_values = [v for v in delta_gm.values() if v is not None]
    avg_delta_gm = np.mean(np.abs(valid_delta_gm_values)) if valid_delta_gm_values else 0
    var_delta_gm = np.var(valid_delta_gm_values) if len(valid_delta_gm_values) > 1 else 0

    return avg_delta_gm, var_delta_gm

def compute_maba_weighted(model, train_loader, test_loader, device):
    model.eval()
    all_train_labels, all_train_bias1, all_train_bias2 = [], [], []
    all_test_preds, all_test_bias1, all_test_bias2 = [], [], []

    # Extract data from loaders
    with torch.no_grad():
        for images, labels, bias1, bias2, _ in train_loader:
            all_train_labels.extend(labels.cpu().numpy())
            all_train_bias1.extend(bias1.cpu().numpy())
            all_train_bias2.extend(bias2.cpu().numpy())

        for images, _, bias1, bias2, _ in test_loader:
            images = images.to(device)
            output, _ = model(images)
            preds = output.data.max(1, keepdim=True)[1].squeeze(1).cpu().numpy()
            all_test_preds.extend(preds)
            all_test_bias1.extend(bias1.cpu().numpy())
            all_test_bias2.extend(bias2.cpu().numpy())

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
    # print("\n[MABA with Weighted Averaging]")
    # print("Group | Bias1 | Bias2 | Train Bias (Num/Denom) | Test Bias (Num/Denom) | Delta_gm")
    # print("-" * 90)

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
            #print(f"{group_label:^6} | {str(attr_set[0]) if attr_set[0] is not None else 'N/A':^6} | {str(attr_set[1]) if attr_set[1] is not None else 'N/A':^6} | {train_bias:.4f} ({train_num}/{train_denom}) | {test_bias:.4f} ({test_num}/{test_denom}) | {delta_gm.get(key, 0):.4f}")
        else:
            #print(f"{group_label:^6} | {str(attr_set[0]) if attr_set[0] is not None else 'N/A':^6} | {str(attr_set[1]) if attr_set[1] is not None else 'N/A':^6} | {train_bias:.4f} ({train_num}/{train_denom}) | {test_bias:.4f} ({test_num}/{test_denom}) | Skipped (Train Bias <= Uniform)")
            pass
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

def main():
    opt = parse_option()
    np.set_printoptions(precision=3)
    torch.set_printoptions(precision=3)
    root = "../data/biased_mnist"
    val_loaders = {}
    train_loader = get_color_mnist(
        root,
        batch_size=256,
        data_label_correlation1=0.99,
        data_label_correlation2=0.99,
        n_confusing_labels=9,
        split="train",
        seed=1,
        aug=True,
        ratio=10,
    )
    
    val_loaders["valid"] = get_color_mnist(
        root,
        batch_size=256,
        data_label_correlation1=0.1,
        data_label_correlation2=0.1,
        n_confusing_labels=9,
        split="train_val",
        seed=1,
        aug=False,
    )
    val_loaders["test"] = get_color_mnist(
        root,
        batch_size=256,
        data_label_correlation1=0.1,
        data_label_correlation2=0.1,
        n_confusing_labels=9,
        split="valid",
        seed=1,
        aug=False,
    )
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = set_model(opt)
    model.to(device)

    print("----------------------")
    maba_base_avg, maba_base_var = compute_maba(model, train_loader, val_loaders["test"], device)
    print(f"Base MABA Score: ({maba_base_avg:.4f}, {maba_base_var:.4f})")
    print("----------------------")

    min_support_threshold = 5
    maba_threshold_avg, maba_threshold_var = compute_maba_with_threshold(model, train_loader, val_loaders["test"], device, min_support=min_support_threshold)
    print(f"MABA Score with Minimum Support Threshold ({min_support_threshold}): ({maba_threshold_avg:.4f}, {maba_threshold_var:.4f})")
    print("----------------------")

    maba_weighted_avg, maba_weighted_var = compute_maba_weighted(model, train_loader, val_loaders["test"], device)
    print(f"MABA Score with Weighted Averaging: ({maba_weighted_avg:.4f}, {maba_weighted_var})")
    print("----------------------")

if __name__ == "__main__":
    main()
