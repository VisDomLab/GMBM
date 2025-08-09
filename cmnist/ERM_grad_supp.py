import argparse
import datetime
import logging
import os
from tqdm import tqdm
import time
from pathlib import Path
import torch.nn.functional as F
import numpy as np
import torch
from torch import nn, optim
from datasets.biased_mnist import get_color_mnist
from models.simple_conv import SimpleConvNet
from utils.logging import set_logging
from utils.utils import (AverageMeter,MultiDimAverageMeter,accuracy,load_model,pretty_dict,save_model,set_seed)

def parse_option():
    parser = argparse.ArgumentParser()
    parser.add_argument("--exp_name", type=str, default="test")
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--color_classifier", type=str, default="./bias_capturing_classifiers/bcc_multibiased_mnist_1.pth")
    parser.add_argument("--color_classifier2", type=str, default="./bias_capturing_classifiers/bcc_multibiased_mnist_2.pth")
    parser.add_argument("--print_freq", type=int, default=300, help="print frequency")
    parser.add_argument("--save_freq", type=int, default=200, help="save frequency")
    parser.add_argument("--epochs", type=int, default=20, help="number of training epochs")
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--corr1", type=float, default=0.9)
    parser.add_argument("--corr2", type=float, default=0.9)
    parser.add_argument("--bs", type=int, default=128, help="batch_size")
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--alpha", type=float, default=0.01, help="alpha value")
    parser.add_argument("--beta", type=float, default=0.01, help="beta value")
    opt = parser.parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = str(opt.gpu)
    return opt

def set_model(opt):
    model = SimpleConvNet().cuda()
    criterion1 = nn.CrossEntropyLoss()
    protected_net = SimpleConvNet()
    protected_net.load_state_dict(load_model(opt.color_classifier))
    protected_net.cuda()
    protected_net2 = SimpleConvNet()
    protected_net2.load_state_dict(load_model(opt.color_classifier2))
    protected_net2.cuda()
    return model, criterion1, protected_net, protected_net2

def train(train_loader, model, criterion, optimizer, protected_net, protected_net2, opt):

    model.train()
    protected_net.eval()
    protected_net2.eval()
    avg_loss = AverageMeter()
    avg_clloss = AverageMeter()
    avg_miloss = AverageMeter()

    train_iter = iter(train_loader)
    for idx, (images, labels, biases, biases2, _) in enumerate(train_iter):
        optimizer.zero_grad()
        bsz = labels.shape[0]
        labels = labels.cuda()
        images = images.cuda()

        loss = 0
        logits, features = model(images)
        loss_cl = criterion(logits, labels)
        loss += loss_cl 

        avg_loss.update(loss.item(), bsz)
        avg_clloss.update(loss_cl.item(), bsz)
        avg_miloss.update(0, bsz)

        loss.backward()
        optimizer.step()

    return avg_loss.avg, avg_clloss.avg, avg_miloss.avg

def validate(val_loader, model):
    model.eval()
    top1 = AverageMeter()
    attrwise_acc_meter = MultiDimAverageMeter(dims=(10, 10))
    attrwise_acc_meter2 = MultiDimAverageMeter(dims=(10, 10))

    with torch.no_grad():
        for idx, (images, labels, biases, biases2, _) in enumerate(val_loader):
            images, labels, biases, biases2 = images.cuda(), labels.cuda(), biases.cuda(), biases2.cuda()
            bsz = labels.shape[0]

            output, feats = model(images)  
            preds = output.data.max(1, keepdim=True)[1].squeeze(1)

            (acc1,) = accuracy(output, labels, topk=(1,))
            top1.update(acc1[0], bsz)

            corrects = (preds == labels).long()
            attrwise_acc_meter.add(corrects.cpu(), torch.stack([labels.cpu(), biases.cpu()], dim=1))
            attrwise_acc_meter2.add(corrects.cpu(), torch.stack([labels.cpu(), biases2.cpu()], dim=1))

    return (top1.avg,attrwise_acc_meter.get_unbiased_acc(),attrwise_acc_meter2.get_unbiased_acc())

def fine_tune(train_loader, val_loader, model, criterion, optimizer, protected_net, protected_net2, opt, epochs=15):
    model.train()
    protected_net.eval()
    protected_net2.eval()
    decay_epochs = []
    scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=decay_epochs, gamma=0.1)

    best_acc = 0.0
    best_model_path = f"RESULTS-ablation-grad-supp-only/{opt.exp_name}/checkpoints/fine_tuned_model_grad_sup.pth"
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
                    pr_l2, pr_feat2 = protected_net2(images)

                logits, model_feat = model(images)
                ce_loss = criterion(logits, labels)
                model_feat = model_feat[-1]

                pr_feat = pr_feat[-1].clone().cuda()
                pr_feat2 = pr_feat2[-1].clone().cuda()

                pr_feat = torch.flatten(pr_feat, start_dim=1)
                pr_feat2 = torch.flatten(pr_feat2, start_dim=1)
                model_feat = torch.flatten(model_feat, start_dim=1)

                # Compute projections
                def project(u, v):
                    return (torch.sum(u * v, dim=1, keepdim=True) / (torch.sum(v * v, dim=1, keepdim=True) + 1e-8)) * v
                
                pr1_proj = project(pr_feat, model_feat)
                pr2_proj = project(pr_feat2, model_feat)

                # Compute residuals
                pr1_residual = pr_feat - pr1_proj
                pr2_residual = pr_feat2 - pr2_proj

                loss = ce_loss 

                # Compute gradient of loss w.r.t model_feat
                grad_model_feat = torch.autograd.grad(loss, model_feat, retain_graph=True)[0]

                # Compute dot product with residuals
                grad_dot_res1 = torch.sum(grad_model_feat * pr1_residual, dim=1).mean()
                grad_dot_res2 = torch.sum(grad_model_feat * pr2_residual, dim=1).mean()
                alpha = opt.alpha
                beta = opt.beta
                # Add regularization loss
                loss += alpha * (grad_dot_res1) ** 2 + beta * (grad_dot_res2) ** 2

                avg_loss.update(loss.item(), bsz)

                loss.backward()
                optimizer.step()

                pbar.update(1)
                pbar.set_postfix(loss=avg_loss.avg, ce_loss=avg_ce_loss.avg, cosine_loss=avg_cosine_loss.avg)

        scheduler.step()

        # Validate on validation set
        val_acc, _, _ = validate(val_loader, model)
        print(f"Validation Accuracy after Epoch {epoch+1}: {val_acc:.4f}")

        # Save best model
        if val_acc > best_acc:
            best_acc = val_acc
            save_model(model, optimizer, opt, epoch+1, best_model_path)
            print(f"New best model saved with accuracy: {best_acc:.4f}")

    print(f"Best Validation Accuracy: {best_acc:.4f}")
    return best_model_path

def main():
    opt = parse_option()

    exp_name = f"badd-color_mnist_corrA{opt.corr1}-corrB{opt.corr2}-{opt.exp_name}-lr{opt.lr}-bs{opt.bs}-seed{opt.seed}"
    opt.exp_name = exp_name

    output_dir = f"RESULTS-ablation-grad-supp-only/{exp_name}"
    save_path = Path(output_dir)
    save_path.mkdir(parents=True, exist_ok=True)

    set_logging(exp_name, "INFO", str(save_path))
    set_seed(opt.seed)
    logging.info(f"save_path: {save_path}")

    np.set_printoptions(precision=3)
    torch.set_printoptions(precision=3)

    root = "../data/biased_mnist"
    train_loader = get_color_mnist(
        root,
        batch_size=opt.bs,
        data_label_correlation1=opt.corr1,
        data_label_correlation2=opt.corr2,
        n_confusing_labels=9,
        split="train",
        seed=opt.seed,
        aug=True,
        ratio=10)
    
    val_loader = get_color_mnist(
        root,
        batch_size=256,
        data_label_correlation1=0.1,
        data_label_correlation2=0.1,
        n_confusing_labels=9,
        split="train_val",
        seed=opt.seed,
        aug=False)
    
    test_loader = get_color_mnist(
        root,
        batch_size=256,
        data_label_correlation1=0.1,
        data_label_correlation2=0.1,
        n_confusing_labels=9,
        split="valid",
        seed=opt.seed,
        aug=False,
    )
    
    model, criterion, protected_net, protected_net2 = set_model(opt)
    optimizer = optim.Adam(model.parameters(), lr=opt.lr)
    scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=[], gamma=0.1)

    # Check if a pretrained model exists
    checkpoints_dir = save_path / "checkpoints"
    pre_finetune_path = checkpoints_dir / "pre_finetune_model.pth"

    if pre_finetune_path.exists():
        logging.info(f"Loading pretrained model from {pre_finetune_path}")
        #load_model(model, optimizer, pre_finetune_path) 
    else:
        # Train the model if no pretrained model is found
        logging.info("No pretrained model found. Starting training...")
        start_time = time.time()
        for epoch in range(1, opt.epochs + 1):
            logging.info(f"[{epoch} / {opt.epochs}] Learning rate: {scheduler.get_last_lr()[0]}")
            loss, cl_loss, mi_loss = train(train_loader, model, criterion, optimizer, protected_net, protected_net2, opt)
            logging.info(f"[{epoch} / {opt.epochs}] Loss: {loss}, CE Loss: {cl_loss}, MI Loss: {mi_loss}")
            scheduler.step()

        checkpoints_dir.mkdir(parents=True, exist_ok=True)
        save_model(model, optimizer, opt, opt.epochs, pre_finetune_path)
        logging.info(f"Model trained and saved at {pre_finetune_path}")

    # Validate the model (whether loaded or trained)
    val_acc_before, attrwise_acc_before, attrwise_acc2_before = validate(test_loader, model)
    logging.info(f"Test Accuracy: {val_acc_before:.4f}")
    print("--------------------------------------------------------------")
    print()
    # Freeze model layers except for selected ones
    for param in model.parameters():
        param.requires_grad = False
    for name, param in model.named_parameters():
        if "fc" in name or "extracter.10" in name:  
            param.requires_grad = True

    logging.info("\nFine-tuning model with different alpha and beta values...")

    alpha_values = [0, 0.01, 0.1,1,10,100] 
    beta_values = [0,0.01, 0.1,1,10,100]  
    for alpha in alpha_values:
        for beta in beta_values:
            logging.info(f"Fine-tuning with alpha={alpha}, beta={beta}")
            model.load_state_dict(load_model(pre_finetune_path))
            model.cuda()
            model.train()
            optimizer = optim.Adam(model.parameters(), lr=opt.lr * 0.1)
            opt.alpha = alpha
            opt.beta = beta

            # Fine-tune the model
            fine_tuned_model_path = save_path / f"checkpoints/fine_tuned_alpha{alpha}_beta{beta}.pth"
            fine_tuned_model_path = fine_tune(train_loader, val_loader, model, criterion, optimizer, protected_net, protected_net2, opt)

            # Load fine-tuned model for final validation
            model.load_state_dict(load_model(fine_tuned_model_path))
            model.cuda()
            model.eval()

            # Final validation on test set
            val_acc_after, attrwise_acc_after, attrwise_acc2_after = validate(test_loader, model)
            logging.info(f"After fine-tuning (alpha={alpha}, beta={beta}) - Test Accuracy: {val_acc_after:.4f}")

    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    logging.info(f"Total training and fine-tuning time: {total_time_str}")

if __name__ == "__main__":
    main()
