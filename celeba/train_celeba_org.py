import argparse
import datetime
import logging
import os
import time
from pathlib import Path

import numpy as np
import torch
from torch import nn, optim

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
    set_seed,
)

from tqdm import tqdm


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
        default="./bias_capturing_classifiers/bcc_lipstick.pth",
    )
    parser.add_argument(
        "--color_classifier2",
        type=str,
        default="./bias_capturing_classifiers/bcc_makeup.pth",
    )
    parser.add_argument("--print_freq", type=int, default=300, help="print frequency")
    parser.add_argument("--save_freq", type=int, default=200, help="save frequency")
    parser.add_argument(
        "--epochs", type=int, default=20, help="number of training epochs"
    )
    parser.add_argument("--seed", type=int, default=1)

    parser.add_argument("--bs", type=int, default=128, help="batch_size")
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--bcc_bg", type=int, default=0)
    parser.add_argument("--bcc_fg", type=int, default=0)
    parser.add_argument("--alpha", type=float, default=20, help="alpha value")
    parser.add_argument("--beta", type=float, default=20, help="beta value")
    opt = parser.parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = str(opt.gpu)

    return opt


def set_model(opt):
    model = ResNet18()
    model.cuda()
    criterion1 = nn.CrossEntropyLoss()
    protected_net = ResNet18()
    protected_net.load_state_dict(load_model(opt.color_classifier))
    protected_net.cuda()
    protected_net2 = ResNet18()
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

    for idx, (images, labels, biases, biases2) in enumerate(tqdm(train_iter)):
        optimizer.zero_grad()
        bsz = labels.shape[0]
        labels = labels.cuda()
        images = images.cuda()

        loss = 0
        # if opt.bcc_fg:
        #     with torch.no_grad():
        #         pr_l, pr_feat = protected_net(images)
        #         pr_pred = pr_l.argmax(dim=1, keepdim=True)
        #         pr_pred = pr_pred.T.detach().cpu()


        # if opt.bcc_bg:
        #     with torch.no_grad():
        #         pr_l2, pr_feat2 = protected_net2(images)
        #         pr_pred2 = pr_l2.argmax(dim=1, keepdim=True)
        #         pr_pred2 = pr_pred2.T.detach().cpu()

        
        if opt.bcc_fg and opt.bcc_bg:
            logits, _ = model.concat_forward3(images, pr_feat, pr_feat2)
        elif opt.bcc_fg:
            logits, _ = model.concat_forward(images, pr_feat)
        elif opt.bcc_bg:
            logits, _ = model.concat_forward(images, pr_feat2)
        else:
            logits, _ = model(images)
        loss_cl = criterion(logits, labels)
        loss += loss_cl  
        avg_loss.update(loss.item(), bsz)
        avg_clloss.update(loss_cl.item(), bsz)
        avg_miloss.update(0, bsz)

        loss.backward()

        optimizer.step()

    return avg_loss.avg, avg_clloss.avg, avg_miloss.avg


def fine_tune(train_loader,val_loaders, model, criterion, optimizer, protected_net, protected_net2, opt, epochs=20):
    model.train()
    protected_net.eval()
    protected_net2.eval()
    decay_epochs = []

    scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=decay_epochs, gamma=0.1)

    output_dir = f"results-celeba2/{opt.exp_name}"
    save_path = Path(output_dir)
    best_accs = {"test1": 0, "test2": 0}
    best_epochs = {"test1": 0, "test2": 0}
    best_stats = {}
    stats = pretty_dict(epoch=epochs)
    best_model_path = f"results-celeba2/{opt.exp_name}/checkpoints/fine_tuned_model_grad_sup.pth"
    os.makedirs(os.path.dirname(best_model_path), exist_ok=True)

    for epoch in range(epochs):
        avg_loss = AverageMeter()
        avg_ce_loss = AverageMeter()
        avg_cosine_loss = AverageMeter()

        train_iter = iter(train_loader)

        with tqdm(total=len(train_loader), desc=f"Epoch {epoch+1}/{epochs}") as pbar:
            for idx, (images, labels, biases, biases2) in enumerate(train_iter):
                optimizer.zero_grad()
                bsz = labels.shape[0]

                images, labels = images.cuda(), labels.cuda()
                loss = 0 
                
                with torch.no_grad():
                    pr_l, pr_feat = protected_net(images)
                    pr_l2, pr_feat2 = protected_net2(images)

                logits, model_feat = model(images)
                
                ce_loss = criterion(logits, labels)
                #model_feat = model_feat[-1]

                # pr_feat = pr_feat[-1].clone().cuda()
                # pr_feat2 = pr_feat2[-1].clone().cuda()

                pr_feat = torch.flatten(pr_feat, start_dim=1)
                pr_feat2 = torch.flatten(pr_feat2, start_dim=1)
                model_feat = torch.flatten(model_feat, start_dim=1)
                # Compute projections
                def project(u, v):
                    return (torch.sum(u * v, dim=1, keepdim=True) / (torch.sum(v * v, dim=1, keepdim=True) + 1e-8)) * v
                
                pr1_proj = project(pr_feat,model_feat)
                pr2_proj = project(pr_feat2,model_feat)

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
                loss += (alpha * (grad_dot_res1)  + beta * (grad_dot_res2) )

                avg_loss.update(loss.item(), bsz)

                loss.backward()
                optimizer.step()

                pbar.update(1)
                pbar.set_postfix(loss=avg_loss.avg, ce_loss=avg_ce_loss.avg, cosine_loss=avg_cosine_loss.avg)

        scheduler.step()
        for key, val_loader in val_loaders.items():
            _, acc_unbiased1, acc_unbiased2 = validate(val_loader, model)

            stats[f"{key}1/acc_unbiased"] = torch.mean(acc_unbiased1).item() * 100
            stats[f"{key}2/acc_unbiased"] = torch.mean(acc_unbiased2).item() * 100

            eye_tsr = torch.eye(2)
            stats[f"{key}1/acc_skew"] = acc_unbiased1[eye_tsr > 0.0].mean().item() * 100
            stats[f"{key}1/acc_align"] = (
                acc_unbiased1[eye_tsr == 0.0].mean().item() * 100
            )

            stats[f"{key}2/acc_skew"] = acc_unbiased2[eye_tsr > 0.0].mean().item() * 100
            stats[f"{key}2/acc_align"] = (
                acc_unbiased2[eye_tsr == 0.0].mean().item() * 100
            )

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


        
    return best_model_path


def validate(val_loader, model):
    model.eval()

    top1 = AverageMeter()
    attrwise_acc_meter = MultiDimAverageMeter(dims=(2, 2))
    attrwise_acc_meter2 = MultiDimAverageMeter(dims=(2, 2))

    with torch.no_grad():
        for idx, (images, labels, biases, biases2) in enumerate(tqdm(val_loader)):
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
        attrwise_acc_meter.get_mean(),
        attrwise_acc_meter2.get_mean(),
    )


def main():
    opt = parse_option()

    exp_name = f"badd-celeba_org-{opt.exp_name}-lr{opt.lr}-bs{opt.bs}-bcc_bg{opt.bcc_bg}-bcc_fg{opt.bcc_fg}-seed{opt.seed}"
    opt.exp_name = exp_name

    output_dir = f"results-erm/{exp_name}"
    save_path = Path(output_dir)
    save_path.mkdir(parents=True, exist_ok=True)

    set_logging(exp_name, "INFO", str(save_path))
    set_seed(opt.seed)
    logging.info(f"save_path: {save_path}")

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

    model, criterion, protected_net, protected_net2 = set_model(opt)

    decay_epochs = [opt.epochs // 3, opt.epochs * 2 // 3]

    optimizer = optim.Adam(model.parameters(), lr=opt.lr)
    scheduler = optim.lr_scheduler.MultiStepLR(
        optimizer, milestones=decay_epochs, gamma=0.1
    )
    logging.info(f"decay_epochs: {decay_epochs}")

    (save_path / "checkpoints").mkdir(parents=True, exist_ok=True)

    best_accs = {"test1": 0, "test2": 0}
    best_epochs = {"test1": 0, "test2": 0}
    best_stats = {"test1": 0, "test2": 0}

    start_time = time.time()
    for epoch in range(1, opt.epochs + 1):
        logging.info(f"[{epoch} / {opt.epochs}] Learning rate: {scheduler.get_last_lr()[0]}")
        loss, cllossp, milossp = train(train_loader,model,criterion,optimizer,protected_net,protected_net2,opt,)
        logging.info(f"[{epoch} / {opt.epochs}] Loss: {loss}  Loss CE: {cllossp}  Loss MI: {milossp}")

        scheduler.step()

        stats = pretty_dict(epoch=epoch)

        best_top1_acc = 0.0
        best_top1_epoch = 0

        for key, val_loader in val_loaders.items():
            top1_acc, acc_unbiased1, acc_unbiased2 = validate(val_loader, model)

            stats[f"{key}/top1_acc"] = top1_acc.item() 
            stats[f"{key}1/acc_unbiased"] = torch.mean(acc_unbiased1).item() * 100
            stats[f"{key}2/acc_unbiased"] = torch.mean(acc_unbiased2).item() * 100

            eye_tsr = torch.eye(2)
            stats[f"{key}1/acc_skew"] = acc_unbiased1[eye_tsr > 0.0].mean().item() * 100
            stats[f"{key}1/acc_align"] = (
                acc_unbiased1[eye_tsr == 0.0].mean().item() * 100
            )

            stats[f"{key}2/acc_skew"] = acc_unbiased2[eye_tsr > 0.0].mean().item() * 100
            stats[f"{key}2/acc_align"] = (
                acc_unbiased2[eye_tsr == 0.0].mean().item() * 100
            )

            # Save best top-1 accuracy model
            if stats[f"{key}/top1_acc"] > best_top1_acc:
                best_top1_acc = stats[f"{key}/top1_acc"]
                best_top1_epoch = epoch
                save_file = save_path / "checkpoints" / f"best_top1.pth"
                save_model(model, optimizer, opt, epoch, save_file)
                logging.info(
                    f"New best top-1 accuracy: {best_top1_acc:.3f} at epoch {best_top1_epoch}"
                )

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
    '''
    pre_finetune_path = "/home/ankur/Desktop/badd_celeba/code/results/badd-celeba_org-test-lr0.001-bs128-bcc_bg1-bcc_fg1-seed1/checkpoints/best_test1.pth"
    model.load_state_dict(load_model(pre_finetune_path))
    for param in model.parameters():
        param.requires_grad = False
    for name, param in model.named_parameters():
        if "fc" in name or "extracter.7.1" in name or "extractor.6.1" in name:  
            param.requires_grad = True
    optimizer = optim.Adam(model.parameters(), lr=opt.lr)
    fine_tune(train_loader,val_loaders, model, criterion, optimizer, protected_net, protected_net2, opt, epochs=20)
         
    '''

if __name__ == "__main__":
    main()
