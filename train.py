import argparse
import glob
import json
import os
import pickle
import sys
from progress.bar import IncrementalBar as ProgressBar
import warnings
import sklearn.metrics as skl


import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision
from torchvision import datasets, models, transforms
import copy

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=100)


def evaluate(
    model,
    loader,
    loss_fn=nn.CrossEntropyLoss(),
    use_cuda=True,
    save_type="probs",

):

    loss_meter, acc_meter = [AverageMeter() for i in range(3)]
    auroc = -1
    all_targets, all_probs, all_logits = [], [], []
    all_ids = []
    model.eval()
    for batch_idx, batch in enumerate(loader):
        with torch.no_grad():
            inputs, targets = batch
            targets = targets["target"]
            images = inputs["image"]
            if use_cuda:
                targets_data = targets_data.cuda(non_blocking=True)
                images = images.cuda(non_blocking=True)
            inputs_dict = {"images": images}
            output = model.image_model(inputs_dict["images"])

            if isinstance(output, tuple):
                logits, attn = output
            else:
                logits, attn = output, None
            c_loss = loss_fn(logits, targets)
            loss = c_loss
            acc, (_, total) = accuracy(targets, logits, return_stats=True)

        acc_meter.update(acc, total)
        loss_meter.update(loss, total)
        probs = F.softmax(logits.data, dim=1)
        all_targets.append(targets.cpu())
        all_logits.append(logits.data.cpu())
        all_probs.append(probs.cpu())
        all_ids.extend(batch[0]["id"])
        targets_cat = torch.cat(all_targets).numpy()
        probs_cat = torch.cat(all_probs).numpy()
        auroc = compute_roc_auc(targets_cat, probs_cat)

    logits_cat = torch.cat(all_logits).numpy()
    saved = probs_cat if save_type == "probs" else logits_cat
    return loss_meter.avg, acc_meter.avg, auroc, all_ids, saved

class AverageMeter(object):
    """Computes and stores the average and current value
    Imported from https://github.com/pytorch/examples/blob/master/imagenet/main.py#L247-L262
    """

    def __init__(self):
        self.reset()

    def reset(self):
        self.val, self.avg, self.sum, self.count = [0] * 4

    def update(self, val, n=1):
        if isinstance(val, torch.Tensor):
            val = val.data.cpu().item()
        if isinstance(n, torch.Tensor):
            n = n.data.cpu().item()
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / max(self.count, 1)

def accuracy(targets, probs, return_stats=False, as_percent=True):
    """Return the number of probs that match the associated ground truth label,
    along with the total number of probs/labels."""
    probs, targets = torch.as_tensor(probs), torch.as_tensor(targets)
    if len(probs.shape) == 1 or probs.shape[-1] == 1:
        predicted = (probs.data > 0.5).float()
    else:
        predicted = torch.argmax(probs.data, 1)
    total = targets.shape[0]
    correct = (predicted == targets).sum().item()
    acc = correct / total
    if as_percent:
        acc *= 100
    if return_stats:
        return acc, (correct, total)
    return acc

def compute_roc_auc(targets, probs):
    try:
        num_classes = len(set(np.array(targets)))
        if (
            num_classes < 2
            or len(probs.shape) < 1
            or len(probs.shape) > 2
            or (len(probs.shape) == 2 and probs.shape[1] != num_classes)
        ):
            raise ValueError
        elif num_classes == 2:
            if len(probs.shape) == 2:
                probs = probs[:, 1]
        else:
            if len(probs.shape) < 2:
                raise ValueError
        auroc = skl.roc_auc_score(targets, probs, multi_class="ovo")
    except ValueError:
        auroc = -1
    return auroc


def train_epoch(model, loader, optimizer, loss_fn=nn.CrossEntropyLoss(), use_cuda=True):


    loss_meter, acc_meter = [AverageMeter()]*2
    all_targets, all_probs = [], []
    auroc = -1
    model.train()
    for batch_idx, batch in enumerate(loader):

        inputs, targets = batch
        targets_data = targets["target"]
        images = inputs["image"]
        if use_cuda:
            targets_data = targets_data.cuda(non_blocking=True)
            images = images.cuda(non_blocking=True)
        targets_dict = {"target": targets_data}
        inputs_dict = {"images": images}

        targets = targets_dict["target"]

        output = model.image_model(inputs_dict["images"])

        if isinstance(output, tuple):
            logits, attn = output
        else:
            logits, attn = output, None

        c_loss = loss_fn(logits, targets)
        loss = c_loss
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        acc, (_, total) = accuracy(targets, logits, return_stats=True)
        acc_meter.update(acc, total)
        loss_meter.update(c_loss, total)

        probs = F.softmax(logits.data, dim=1)
        all_targets.append(targets.cpu())
        all_probs.append(probs.cpu())
        targets_cat = torch.cat(all_targets).numpy()
        probs_cat = torch.cat(all_probs).numpy()
        auroc = compute_roc_auc(targets_cat, probs_cat)

    return loss_meter.avg, acc_meter.avg, auroc


def get_lrs(optimizer):
    lrs = []
    for param_group in optimizer.param_groups:
        if "lr" in param_group:
            lrs.append(param_group["lr"])
    return lrs


def train(model, optimizer, scheduler, loaders, state, args, use_cuda=True):
    loss_fn = nn.CrossEntropyLoss()

    ### implement metric choice

    best_model_wts = copy.deepcopy(model.state_dict())
    best_auroc = 0.0
    metrics = {
        "train_loss": [],
        "val_loss": [],
        "train_acc": [],
        "val_acc": [],
        "train_auroc": [],
        "val_auroc": [],
    }

    for epoch in range(args.epochs):

        cur_lrs = get_lrs(optimizer)


        print(
            f'\nEpoch: [{epoch+1} | {args.epochs}]; LRs: {", ".join([f"{lr:.2E}" for lr in cur_lrs])}'
        )

        train_loss, train_acc, train_auroc = train_epoch(
            model,
            loaders["train"],
            optimizer,
            loss_fn=loss_fn,
            use_cuda=use_cuda,
        )

        val_loss, val_acc, val_auroc, _, _= evaluate(
            model,
            loaders["val"],
            loss_fn=loss_fn,
            use_cuda=use_cuda,
        )

        metrics['train_loss'].append(train_loss)
        metrics['train_acc'].append(train_acc)
        metrics['train_auroc'].append(train_auroc)
        metrics['val_loss'].append(val_loss)
        metrics['val_acc'].append(val_acc)
        metrics['val_auroc'].append(val_auroc)


        if val_auroc >= best_auroc:
            best_auroc = val_auroc
            best_model_wts = copy.deepcopy(model.state_dict())

    model.load_state_dict(best_model_wts)
    print(f"\nBest Auroc: {best_auroc:.3f}")
    return model, best_auroc, metrics



def main():
    args = parse_args()
    use_cuda = torch.cuda.is_available()
    if use_cuda:
        print("Using CUDA...")
        torch.backends.cudnn.deterministic = args.deterministic
        torch.backends.cudnn.benchmark = not args.deterministic


    #load in data loader

    loaders = fetch_dataloaders()

    num_classes = loaders["train"].dataset.num_classes


    model = models.resnet50(pretrained=True)
    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, num_classes)


    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    params_to_update = model.parameters()
    optimizer = optim.SGD(params_to_update, lr=0.001, momentum=0.9)

    ##TODO: default scheduler

    best_model, val_auroc, metrics = train(model, optimizer, loaders, args, use_cuda=True)
    print(f"Best Val Auroc {val_auroc}")
    test_loss, test_acc, test_auroc, _, _ = evaluate(model, loaders['test'], loss_fn=nn.CrossEntropyLoss())
    print(f"Besst Test Auroc {test_auroc}")

if __name__ == "__main__":
    main()