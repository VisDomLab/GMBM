import argparse
import datetime
import logging
import os
import time
from pathlib import Path
import torch.nn.functional as F
import numpy as np
import torch
from torch import nn, optim

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
    parser.add_argument("--corr1", type=float, default=0.9)
    parser.add_argument("--corr2", type=float, default=0.9)

    parser.add_argument("--bs", type=int, default=128, help="batch_size")
    parser.add_argument("--alpha", type=float, default=1, help="alpha value")
    parser.add_argument("--beta", type=float, default=1, help="beta value")
    parser.add_argument("--lr", type=float, default=1e-3)
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


def train(
    train_loader, model, criterion, optimizer, protected_net, protected_net2, opt
):
    model.train()
    protected_net.eval()
    protected_net2.eval()
    avg_loss = AverageMeter()
    avg_clloss = AverageMeter()
    avg_miloss = AverageMeter()

    train_iter = iter(train_loader)
    total = 0
    corr1 = 0
    corr2 = 0
    for idx, (images, labels, biases, biases2, _) in enumerate(train_iter):
        optimizer.zero_grad()
        bsz = labels.shape[0]
        labels = labels.cuda()
        images = images.cuda()

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

        logits, model_feat = model.concat_forward(
            images, shuffled_pr_feat, shuffled_pr_feat2
        )
        loss_cl = criterion(logits, labels)

        _, pr_feat = protected_net(images)
        _, pr_feat2 = protected_net2(images)
        #_, model_feat = model(images)

        #model_feat = model_feat[-1]

        pr_feat = pr_feat[-1].clone().cuda()
        pr_feat2 = pr_feat2[-1].clone().cuda()

        pr_feat = torch.flatten(pr_feat, start_dim=1)
        pr_feat2 = torch.flatten(pr_feat2, start_dim=1)
        model_feat = torch.flatten(model_feat, start_dim=1)

        def project(u, v):
            return (torch.sum(u * v, dim=1, keepdim=True) / (torch.sum(v * v, dim=1, keepdim=True) + 1e-8)) * v
                
        pr1_proj = project(pr_feat, model_feat)
        pr2_proj = project(pr_feat2, model_feat)

                # Compute residuals
        pr1_residual = pr_feat - pr1_proj
        pr2_residual = pr_feat2 - pr2_proj

        loss = loss_cl 

        # Compute gradient of loss w.r.t model_feat
        grad_model_feat = torch.autograd.grad(loss, model_feat, retain_graph=True)[0]

        # Compute dot product with residuals
        grad_dot_res1 = torch.sum(grad_model_feat * pr1_residual, dim=1).mean()
        grad_dot_res2 = torch.sum(grad_model_feat * pr2_residual, dim=1).mean()
        alpha = opt.alpha
        beta = opt.beta
                # Add regularization loss
        loss += (alpha * (grad_dot_res1) ** 2 + beta * (grad_dot_res2) ** 2)

        #loss += loss_cl #+ alpha * cosine_loss1 + beta * cosine_loss2

        avg_loss.update(loss.item(), bsz)
        avg_clloss.update(loss_cl.item(), bsz)
        avg_miloss.update(0, bsz)

        loss.backward()

        optimizer.step()

        # corr1 += pr_pred.eq(biases).sum().item()
        # corr2 += pr_pred2.eq(biases2).sum().item()

    return avg_loss.avg, avg_clloss.avg, avg_miloss.avg


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


def main():
    opt = parse_option()

    exp_name = f"badd-color_mnist_corrA{opt.corr1}-corrB{opt.corr2}-{opt.exp_name}-lr{opt.lr}-bs{opt.bs}-seed{opt.seed}"
    opt.exp_name = exp_name

    output_dir = f"Results-gdsb/{exp_name}"
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
        ratio=10,
    )
    logging.info(
        f"confusion_matrix - \n original: {train_loader.dataset.confusion_matrix_org}, \n normalized: {train_loader.dataset.confusion_matrix}"
    )

    val_loaders = {}
    val_loaders["valid"] = get_color_mnist(
        root,
        batch_size=256,
        data_label_correlation1=0.1,
        data_label_correlation2=0.1,
        n_confusing_labels=9,
        split="train_val",
        seed=opt.seed,
        aug=False,
    )
    val_loaders["test"] = get_color_mnist(
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

    #decay_epochs = [opt.epochs // 3, opt.epochs * 2 // 3]
    decay_epochs = []
    optimizer = optim.Adam(model.parameters(), lr=opt.lr)
    scheduler = optim.lr_scheduler.MultiStepLR(
        optimizer, milestones=decay_epochs, gamma=0.1
    )
    logging.info(f"decay_epochs: {decay_epochs}")

    (save_path / "checkpoints").mkdir(parents=True, exist_ok=True)

    best_accs = {"valid1": 0, "test1": 0, "valid2": 0, "test2": 0}
    best_epochs = {"valid1": 0, "test1": 0, "valid2": 0, "test2": 0}
    best_stats = {}
    start_time = time.time()
    for epoch in range(1, opt.epochs + 1):
        # if epoch > 30:
        #     continue
        logging.info(
            f"[{epoch} / {opt.epochs}] Learning rate: {scheduler.get_last_lr()[0]}"
        )
        loss, cllossp, milossp = train(
            train_loader,
            model,
            criterion,
            optimizer,
            protected_net,
            protected_net2,
            opt,
        )
        logging.info(
            f"[{epoch} / {opt.epochs}] Loss: {loss}  Loss CE: {cllossp}  Loss MI: {milossp}"
        )

        scheduler.step()

        stats = pretty_dict(epoch=epoch)
        _, acc_unbiased_train1, acc_unbiased_train2 = validate(train_loader, model)
        logging.info(
            f"/acc_unbiased_train1 {acc_unbiased_train1.item() * 100}, /acc_unbiased_train2 {acc_unbiased_train2.item() * 100}"
        )
        for key, val_loader in val_loaders.items():
            _, acc_unbiased1, acc_unbiased2 = validate(val_loader, model)

            stats[f"{key}1/acc_unbiased"] = acc_unbiased1.item() * 100
            stats[f"{key}2/acc_unbiased"] = acc_unbiased2.item() * 100

        logging.info(f"[{epoch} / {opt.epochs}] {stats}")
        for tag in best_accs.keys():
            if stats[f"{tag}/acc_unbiased"] > best_accs[tag]:
                best_accs[tag] = stats[f"{tag}/acc_unbiased"]
                best_epochs[tag] = epoch
                best_stats[tag] = pretty_dict(
                    **{f"best_{tag}_{k}": v for k, v in stats.items()}
                )

                save_file = save_path / "checkpoints" / f"best_{tag}.pth"
                save_model(model, optimizer, opt, epoch, save_file)
            logging.info(
                f"[{epoch} / {opt.epochs}] best {tag} accuracy: {best_accs[tag]:.3f} at epoch {best_epochs[tag]} \n best_stats: {best_stats[tag]}"
            )

    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    logging.info(f"Total training time: {total_time_str}")

    save_file = save_path / "checkpoints" / f"last.pth"
    save_model(model, optimizer, opt, opt.epochs, save_file)
    model.plot_magnitude()

if __name__ == "__main__":
    main()
