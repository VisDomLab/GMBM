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
from datasets.celeba_org import get_dataloaders
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
    model = ResNet18()
    model.cuda()
    model.load_state_dict(load_model("/home/ankur/Desktop/badd_celeba/code/results-celeba2/badd-celeba_org-test-lr0.001-bs128-bcc_bg1-bcc_fg1-seed1/checkpoints/best_test1.pth"))
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

def compute_sba(model, test_loader, device):
    model.eval()
    all_test_preds, all_test_labels, all_test_bias1, all_test_bias2 = [], [], [], []

    # Extract test data
    with torch.no_grad():
        for images, labels, bias1, bias2 in test_loader:
            images = images.to(device)
            output, _ = model(images)
            preds = output.data.max(1, keepdim=True)[1].squeeze(1).cpu().numpy()
            all_test_preds.extend(preds)
            all_test_labels.extend(labels.cpu().numpy())
            all_test_bias1.extend(bias1.cpu().numpy())
            all_test_bias2.extend(bias2.cpu().numpy())

    # Compute biases
    bias_actual = compute_bias(np.array(all_test_labels), np.array(all_test_bias1), np.array(all_test_bias2))
    bias_model = compute_bias(np.array(all_test_preds), np.array(all_test_bias1), np.array(all_test_bias2))

    # Co-occurrence totals for normalization
    totals_actual = {}
    for _, a in bias_actual:
        totals_actual[a] = sum(bias_actual.get((g, a), 0) for g in np.unique(all_test_labels))

    # Scaled bias amplification
    delta_sba = {}
    for key in bias_actual:
        group_label, attr_set = key
        actual_num = bias_actual[key]
        model_num = bias_model.get(key, 0)
        denom = totals_actual[attr_set] if totals_actual[attr_set] != 0 else 1

        actual_bias = (actual_num / denom) * 100
        model_bias = (model_num / denom) * 100
        delta = model_bias - actual_bias

        scale_factor = 1 / (np.sqrt(denom) + 1e-6)
        delta_scaled = delta * scale_factor

        delta_sba[key] = delta_scaled

    avg_sba = np.mean(np.abs(list(delta_sba.values())))
    return avg_sba, delta_sba


def main():
    opt = parse_option()
    np.set_printoptions(precision=3)
    torch.set_printoptions(precision=3)
    val_loaders = {}
    train_loader, val_loaders["test"] = get_dataloaders(
        "/home/ankur/Desktop/BAdd_Bias_Mitigation/code/data/celeba/img_align_celeba",
        "/home/ankur/Desktop/BAdd_Bias_Mitigation/code/data/celeba/list_attr_celeba.txt",
        precrop=256,
        crop=224,
        bs=64,
        nw=4,
        split=0.7,
    )
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    model = set_model(opt)
    model.to(device)
    print("----------------------")
    maba_base_avg, maba_base_var = compute_sba(model, val_loaders["test"], device)
    print(maba_base_avg)

if __name__ == "__main__":
    main()
