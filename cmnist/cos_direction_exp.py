import argparse
import datetime
import logging
import os
import time
from pathlib import Path
from tqdm import tqdm
import numpy as np
import torch
from torch import nn, optim
import torch.nn.functional as F

from datasets.biased_mnist import get_color_mnist
from models.simple_conv import SimpleConvNet
from utils.logging import set_logging
from utils.utils import (
    AverageMeter,
    MultiDimAverageMeter,
    accuracy,
    load_model,
    pretty_dict,
    save_model,
    set_seed,
)

def parse_option():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--exp_name",
        type=str,
        default="test",
    )
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument(
        "--color_classifier",
        type=str,
        default="./bias_capturing_classifiers/bcc_multibiased_mnist_1.pth",
    )
    parser.add_argument(
        "--color_classifier2",
        type=str,
        default="./bias_capturing_classifiers/bcc_multibiased_mnist_2.pth",
    )
    parser.add_argument("--print_freq", type=int, default=300, help="print frequency")
    parser.add_argument("--save_freq", type=int, default=200, help="save frequency")
    parser.add_argument(
        "--epochs", type=int, default=80, help="number of training epochs"
    )
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--corr1", type=float, default=0.99)
    parser.add_argument("--corr2", type=float, default=0.99)

    parser.add_argument("--bs", type=int, default=128, help="batch_size")
    parser.add_argument("--lr", type=float, default=1e-3)
    opt = parser.parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = str(opt.gpu)

    return opt

def set_model(opt):
    model = SimpleConvNet().cuda()
    model.load_state_dict(load_model("/home/ankur/Desktop/BAdd_Bias_Mitigation/code/RESULTS-ablation-grad-supp/badd-color_mnist_corrA0.9-corrB0.9-test-lr0.001-bs128-seed1/checkpoints/fine_tuned_alpha100_beta100.pth"))
    criterion1 = nn.CrossEntropyLoss()
    protected_net = SimpleConvNet()
    protected_net.load_state_dict(load_model(opt.color_classifier))
    protected_net.cuda()
    protected_net2 = SimpleConvNet()
    protected_net2.load_state_dict(load_model(opt.color_classifier2))
    protected_net2.cuda()

    # for param in model.parameters():
    #     param.requires_grad = False

    # #Unfreeze only the last layer
    # for name, param in model.named_parameters():
    #     if "fc" in name or "extracter.10" in name:  
    #         param.requires_grad = True
    # print("\nModel Layers and Freeze Status:")
    # for name, param in model.named_parameters():
    #     status = "Trainable" if param.requires_grad else "Frozen"
    #     print(f"{name}: {status}")
    return model, criterion1, protected_net, protected_net2

def fine_tune(train_loader, val_loader, model, criterion, optimizer, protected_net, protected_net2, opt, epochs=80):
    model.train()
    protected_net.eval()
    protected_net2.eval()

    # MultiStepLR Scheduler
    decay_epochs = [epochs // 3, epochs * 2 // 3]
    scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=decay_epochs, gamma=0.1)

    best_acc = 0.0  # Track best validation accuracy
    best_model_path = f"results/{opt.exp_name}/checkpoints/best_model.pth"
    os.makedirs(os.path.dirname(best_model_path), exist_ok=True)

    for epoch in range(epochs):
        avg_loss = AverageMeter()
        avg_ce_loss = AverageMeter()
        avg_cosine_loss = AverageMeter()

        train_iter = iter(train_loader)

        with tqdm(total=len(train_loader), desc=f"Epoch {epoch+1}/{epochs}") as pbar:
            for idx, (images, labels, biases, biases2, _) in enumerate(train_iter):
                optimizer.zero_grad()
                bsz = labels.shape[0]

                images, labels = images.cuda(), labels.cuda()
                loss = 0 
                with torch.no_grad():
                    pr_l, pr_feat = protected_net(images)
                    pr_pred = pr_l.argmax(dim=1, keepdim=True)
                    pr_pred = pr_pred.T.detach().cpu()
            
                unique_biases = torch.unique(biases)

                shuffled_pr_feat = pr_feat[-1].clone()  

                for bias_label in unique_biases:
                    # Get indices of rows with the current bias label
                    indices = (biases == bias_label).nonzero().squeeze()

                    if indices.dim() > 0:
                        # Shuffle the rows with the current bias label
                        shuffled_indices = indices[torch.randperm(len(indices))]

                        # Update the shuffled_pr_feat tensor with the shuffled rows
                        shuffled_pr_feat[indices] = shuffled_pr_feat[shuffled_indices]

                with torch.no_grad():
                    pr_l2, pr_feat2 = protected_net2(images)
                    pr_pred2 = pr_l2.argmax(dim=1, keepdim=True)
                    pr_pred2 = pr_pred2.T.detach().cpu()

                unique_biases2 = torch.unique(biases2)
                shuffled_pr_feat2 = pr_feat2[-1].clone()  # [None] * 4

                for bias_label in unique_biases2:
                    # Get indices of rows with the current bias label
                    indices = (biases2 == bias_label).nonzero().squeeze()

                    if indices.dim() > 0:
                        # Shuffle the rows with the current bias label
                        shuffled_indices = indices[torch.randperm(len(indices))]

                        # Update the shuffled_pr_feat tensor with the shuffled rows
                        shuffled_pr_feat2[indices] = shuffled_pr_feat2[shuffled_indices]

                logits, features = model.concat_forward(
                    images, shuffled_pr_feat, shuffled_pr_feat2
                )
                ce_loss = criterion(logits, labels)
                _, pr_feat = protected_net(images)
                _, pr_feat2 = protected_net2(images)
                _, model_feat = model(images)

                model_feat = model_feat[-1].clone().cuda()
                pr_feat = pr_feat[-1].clone().cuda()
                pr_feat2 = pr_feat2[-1].clone().cuda()

                model_feat = torch.flatten(model_feat, start_dim=1)
                pr_feat = torch.flatten(pr_feat, start_dim=1)
                pr_feat2 = torch.flatten(pr_feat2, start_dim=1)

                # Cosine similarity loss
                cosine_loss1 = F.cosine_similarity(model_feat, pr_feat, dim=1).mean()
                cosine_loss2 = F.cosine_similarity(model_feat, pr_feat2, dim=1).mean()
                alpha = 0
                beta = 0
                loss = ce_loss + alpha * cosine_loss1 + beta * cosine_loss2

                avg_loss.update(loss.item(), bsz)
                avg_ce_loss.update(ce_loss.item(), bsz)
                avg_cosine_loss.update((cosine_loss1.item() + cosine_loss2.item()) / 2, bsz)

                loss.backward()
                optimizer.step()

                pbar.update(1)
                pbar.set_postfix(loss=avg_loss.avg, ce_loss=avg_ce_loss.avg, cosine_loss=avg_cosine_loss.avg)

        scheduler.step()

        # Validate on validation set
        val_acc, _, _ = validate(val_loader, model)
        print(f"Validation Accuracy after Epoch {epoch+1}: {val_acc:.4f}")

        # Save model if it's the best so far
        if val_acc > best_acc:
            best_acc = val_acc
            save_model(model, optimizer, opt, epoch+1, best_model_path)
            print(f"New best model saved with accuracy: {best_acc:.4f}")

    print(f"Best Validation Accuracy: {best_acc:.4f}")
    return best_model_path

def validate(val_loader, model):
    model.eval()

    top1 = AverageMeter()
    attrwise_acc_meter = MultiDimAverageMeter(dims=(10, 10))
    attrwise_acc_meter2 = MultiDimAverageMeter(dims=(10, 10))

    with torch.no_grad():
        for idx, (images, labels, biases, biases2, _) in enumerate(val_loader):
            images, labels, biases = images.cuda(), labels.cuda(), biases.cuda()
            bsz = labels.shape[0]

            output, _ = model(images)
            preds = output.data.max(1, keepdim=True)[1].squeeze(1)

            (acc1,) = accuracy(output, labels, topk=(1,))
            top1.update(acc1[0], bsz)
  
            corrects = (preds == labels).long()
            attrwise_acc_meter.add(
                corrects.cpu(), torch.stack([labels.cpu(), biases.cpu()], dim=1)
            )
            attrwise_acc_meter2.add(
                corrects.cpu(), torch.stack([labels.cpu(), biases2], dim=1)
            )

    return (
        top1.avg,
        attrwise_acc_meter.get_unbiased_acc(),
        attrwise_acc_meter2.get_unbiased_acc(),
    )

def validate_with_similarity(val_loader, model,protected_net, protected_net2):
    model.eval()
    
    # Store feature vectors for each biased class separately
    bias_feat_dict = {}
    bias2_feat_dict = {}
    
    with torch.no_grad():
        for idx, (images, labels, biases, biases2, _) in enumerate(val_loader):
            images, labels, biases, biases2 = images.cuda(), labels.cuda(), biases.cuda(), biases2.cuda()
            unique_biases = torch.unique(biases)
            pr_l, pr_feat = protected_net(images)
            pr_pred = pr_l.argmax(dim=1, keepdim=True)
            pr_pred = pr_pred.T.detach().cpu()
            shuffled_pr_feat = pr_feat[-1].clone()  

            for bias_label in unique_biases:
                # Get indices of rows with the current bias label
                indices = (biases == bias_label).nonzero().squeeze()

                if indices.dim() > 0:
                    # Shuffle the rows with the current bias label
                    shuffled_indices = indices[torch.randperm(len(indices))]

                    # Update the shuffled_pr_feat tensor with the shuffled rows
                    shuffled_pr_feat[indices] = shuffled_pr_feat[shuffled_indices]

            with torch.no_grad():
                pr_l2, pr_feat2 = protected_net2(images)
                pr_pred2 = pr_l2.argmax(dim=1, keepdim=True)
                pr_pred2 = pr_pred2.T.detach().cpu()

            unique_biases2 = torch.unique(biases2)
            shuffled_pr_feat2 = pr_feat2[-1].clone()  # [None] * 4

            for bias_label in unique_biases2:
                # Get indices of rows with the current bias label
                indices = (biases2 == bias_label).nonzero().squeeze()

                if indices.dim() > 0:
                    # Shuffle the rows with the current bias label
                    shuffled_indices = indices[torch.randperm(len(indices))]

                    # Update the shuffled_pr_feat tensor with the shuffled rows
                    shuffled_pr_feat2[indices] = shuffled_pr_feat2[shuffled_indices]
            # Extract model features
            _, model_feat = model.concat_forward(images,shuffled_pr_feat, shuffled_pr_feat2)
            #model_feat = model_feat[-1].clone().cuda()
            model_feat = torch.flatten(model_feat, start_dim=1)
            
            # Store feature vectors based on bias categories
            for i in range(len(biases)):
                bias_label = biases[i].item()
                if bias_label not in bias_feat_dict:
                    bias_feat_dict[bias_label] = []
                bias_feat_dict[bias_label].append(model_feat[i].cpu())
                
                bias_label2 = biases2[i].item()
                if bias_label2 not in bias2_feat_dict:
                    bias2_feat_dict[bias_label2] = []
                bias2_feat_dict[bias_label2].append(model_feat[i].cpu())
    
    # Convert lists to tensors
    for key in bias_feat_dict:
        bias_feat_dict[key] = torch.stack(bias_feat_dict[key])
    for key in bias2_feat_dict:
        bias2_feat_dict[key] = torch.stack(bias2_feat_dict[key])
    
    num_groups = len(bias_feat_dict) + len(bias2_feat_dict)
    print(f"Total number of groups: {num_groups}")
    
    # Compute cosine similarities within each bias group
    # intra_similarities = []
    # intra_dot_products = []
    # intra_magnitudes = []
    # for key in bias_feat_dict:
    #     if bias_feat_dict[key].shape[0] > 1:
    #         sim_matrix = F.cosine_similarity(
    #         bias_feat_dict[key].unsqueeze(1), bias_feat_dict[key].unsqueeze(0), dim=-1
    #         )
    #         intra_similarities.append(sim_matrix.mean().item())
    #         dot_product_matrix = torch.matmul(bias_feat_dict[key], bias_feat_dict[key].T)
    #         intra_dot_products.append(dot_product_matrix.mean().item())
    #         intra_magnitudes.append(bias_feat_dict[key].norm(dim=1).mean().item())
    
    # for key in bias2_feat_dict:
    #     if bias2_feat_dict[key].shape[0] > 1:
    #         sim_matrix = F.cosine_similarity(
    #         bias2_feat_dict[key].unsqueeze(1), bias2_feat_dict[key].unsqueeze(0), dim=-1
    #         )
    #         intra_similarities.append(sim_matrix.mean().item())
    #         dot_product_matrix = torch.matmul(bias2_feat_dict[key], bias2_feat_dict[key].T)
    #         intra_dot_products.append(dot_product_matrix.mean().item())
    #         intra_magnitudes.append(bias2_feat_dict[key].norm(dim=1).mean().item())
    
    # Compute cosine similarities between different bias groups
    inter_similarities = []
    inter_dot_products = []
    inter_magnitudes = []
    bias_keys = list(bias_feat_dict.keys())
    for i in range(len(bias_keys)):
        for j in range(i + 1, len(bias_keys)):
            sim_matrix = F.cosine_similarity(
                bias_feat_dict[bias_keys[i]].unsqueeze(1), bias_feat_dict[bias_keys[j]].unsqueeze(0), dim=-1
            )
            inter_similarities.append(sim_matrix.mean().item())
            dot_product_matrix = torch.matmul(bias_feat_dict[bias_keys[i]], bias_feat_dict[bias_keys[j]].T)
            inter_dot_products.append(dot_product_matrix.mean().item())
            inter_magnitudes.append(bias_feat_dict[bias_keys[i]].norm(dim=1).mean().item())
    
    # Compute averages
    #avg_intra_sim = sum(intra_similarities) / len(intra_similarities) if intra_similarities else 0
    avg_inter_sim = sum(inter_similarities) / len(inter_similarities) if inter_similarities else 0
    #avg_intra_dot = sum(intra_dot_products) / len(intra_dot_products) if intra_dot_products else 0
    avg_inter_dot = sum(inter_dot_products) / len(inter_dot_products) if inter_dot_products else 0
    #avg_intra_mag = sum(intra_magnitudes) / len(intra_magnitudes) if intra_magnitudes else 0
    avg_inter_mag = sum(inter_magnitudes) / len(inter_magnitudes) if inter_magnitudes else 0
    
    #print(f"Average Intra-group Cosine Similarity: {avg_intra_sim:.4f}")
    print(f"Average Inter-group Cosine Similarity: {avg_inter_sim:.4f}")
    #print(f"Average Intra-group Dot Product: {avg_intra_dot:.4f}")
    print(f"Average Inter-group Dot Product: {avg_inter_dot:.4f}")
    #print(f"Average Intra-group Magnitude: {avg_intra_mag:.4f}")
    print(f"Average Inter-group Magnitude: {avg_inter_mag:.4f}")
    
    #return avg_intra_sim, avg_inter_sim
    return 0,0

def validate2(val_loader, model):
    model.eval()
    
    # Store feature vectors for each class label separately
    class_feat_dict = {}
    
    with torch.no_grad():
        for idx, (images, labels, _, _, _) in enumerate(val_loader):
            images, labels = images.cuda(), labels.cuda()
            
            # Extract model features
            _, model_feat = model(images)
            model_feat = model_feat[-1].clone().cuda()
            model_feat = torch.flatten(model_feat, start_dim=1)
            
            # Store feature vectors based on actual class labels
            for i in range(len(labels)):
                class_label = labels[i].item()
                if class_label not in class_feat_dict:
                    class_feat_dict[class_label] = []
                class_feat_dict[class_label].append(model_feat[i].cpu())
    
    # Convert lists to tensors
    for key in class_feat_dict:
        class_feat_dict[key] = torch.stack(class_feat_dict[key])
    
    num_groups = len(class_feat_dict)
    print(f"Total number of class groups: {num_groups}")
    
    # Compute cosine similarities within each class group (intra-group)
    intra_similarities = []
    for key in class_feat_dict:
        if class_feat_dict[key].shape[0] > 1:
            sim_matrix = F.cosine_similarity(
                class_feat_dict[key].unsqueeze(1), class_feat_dict[key].unsqueeze(0), dim=-1
            )
            intra_similarities.append(sim_matrix.mean().item())
    
    # Compute average intra-class cosine similarity
    avg_intra_class_sim = sum(intra_similarities) / len(intra_similarities) if intra_similarities else 0
    
    # Compute cosine similarities between different class groups (inter-group)
    inter_similarities = []
    class_keys = list(class_feat_dict.keys())
    for i in range(len(class_keys)):
        for j in range(i + 1, len(class_keys)):
            sim_matrix = F.cosine_similarity(
                class_feat_dict[class_keys[i]].unsqueeze(1),
                class_feat_dict[class_keys[j]].unsqueeze(0),
                dim=-1
            )
            inter_similarities.append(sim_matrix.mean().item())
    
    # Compute average inter-class cosine similarity
    avg_inter_class_sim = sum(inter_similarities) / len(inter_similarities) if inter_similarities else 0
    
    print(f"Average Intra-class Cosine Similarity: {avg_intra_class_sim:.4f}")
    print(f"Average Inter-class Cosine Similarity: {avg_inter_class_sim:.4f}")
    
    return avg_intra_class_sim, avg_inter_class_sim

def main():
    opt = parse_option()
    exp_name = f"badd-color_mnist_corrA{opt.corr1}-corrB{opt.corr2}-{opt.exp_name}-lr{opt.lr}-bs{opt.bs}-seed{opt.seed}"
    opt.exp_name = exp_name

    output_dir = f"results/{exp_name}"
    save_path = Path(output_dir)
    save_path.mkdir(parents=True, exist_ok=True)

    np.set_printoptions(precision=3)
    torch.set_printoptions(precision=3)
    root = "../data/biased_mnist"

    train_loader = get_color_mnist(
        root, batch_size=opt.bs, data_label_correlation1=opt.corr1,
        data_label_correlation2=opt.corr2, n_confusing_labels=9, split="train",
        seed=opt.seed, aug=True, ratio=10,
    )

    val_loaders = {
        "valid": get_color_mnist(root, batch_size=256, data_label_correlation1=0.1,
                                 data_label_correlation2=0.1, n_confusing_labels=9,
                                 split="train_val", seed=opt.seed, aug=False),
        "test": get_color_mnist(root, batch_size=256, data_label_correlation1=0.1,
                                data_label_correlation2=0.1, n_confusing_labels=9,
                                split="valid", seed=opt.seed, aug=False)
    }

    model, criterion, protected_net, protected_net2 = set_model(opt)
    # optimizer = optim.Adam(model.parameters(), lr=opt.lr)

    # print("Starting fine-tuning with cosine similarity loss...")
    # best_model_path = fine_tune(train_loader, val_loaders["valid"], model, criterion, optimizer, protected_net, protected_net2, opt)

    # # Load the best model for final evaluation
    # print("\nLoading best saved model for final evaluation on test set...")
    # model.load_state_dict(load_model(best_model_path))
    # model.cuda()
    # model.eval()

    # Evaluate on test set
    _,_ = validate_with_similarity(val_loaders['test'], model,protected_net, protected_net2)
    #_,_ = validate2(val_loaders['test'], model)
    #test_acc, attrwise_acc_1, attrwise_acc_2 = validate(val_loaders["test"], model)
    # print(f"Final Test Accuracy: {test_acc:.4f}")
    # print("Attribute-wise Accuracy Meter 1:")
    # print(attrwise_acc_1 * 100)
    # print("Attribute-wise Accuracy Meter 2:")
    # print(attrwise_acc_2 * 100)

if __name__ == "__main__":
    main()
