import argparse
import glob
import json
import os
import pickle
import sys
import warnings
import sklearn.metrics as skl
import json


import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision
from torchvision import datasets, models, transforms
import copy
from dataloader import fetch_dataloaders

from utils import AverageMeter, accuracy, compute_roc_auc, build_scheduler, get_lrs

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=25, help="Number of training epochs")
    parser.add_argument("--min_lr", type=float, default=0, help="Minimum learning rate")
    parser.add_argument("--lr", type=float, default=0.0001, help="Learning rate")
    parser.add_argument("--wd", type=float, default=0.0001, help="L2 Weight")
    parser.add_argument("--save_model", action='store_true', help="Whether to save the best model found or not") ##will be false unles flag is specified
    parser.add_argument("--checkpoint_dir", type=str, default=None, help="Whether to use a given saved model")
    parser.add_argument("--seed", type=int, default=0, help="Seed to use in dataloader")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size")

    parser.add_argument("--train_set", type=str, choices=['cxr_a','cxr_p'], required=True, help="Set to train on")
    parser.add_argument("--test_set", type=str, choices=['cxr_a','cxr_p', 'mimic_cxr', 'chexpert', 'chestxray8'], required=True, help="Test set to evaluate on")
    parser.add_argument("--ood_shift", type=str, choices=['hospital','hospital_age', 'age', None], default=None, help="Distribution shift to experiment with")

    parser.add_argument("--save_dir", type=str, default="/mnt/gaze_robustness_results/resnet_only", help="Save Dir")

    args = parser.parse_args()
    return args

def evaluate(
    model,
    loader,
    loss_fn=nn.CrossEntropyLoss(),
    use_cuda=True,
    save_type="probs",

):

    loss_meter, acc_meter = [AverageMeter() for i in range(2)]
    auroc = -1
    all_targets, all_probs, all_logits = [], [], []
    all_ids = []
    model.eval()
    for batch_idx, batch in enumerate(loader):
        with torch.no_grad():
            inputs, targets = batch
            #print(inputs.size())
            #print(targets.size())
            #targets_data = targets["target"]
            #images = inputs["image"]
            if use_cuda:
                targets = targets.cuda(non_blocking=True)
                inputs = inputs.cuda(non_blocking=True)
            #targets_dict = {"target": targets_data}
            #inputs_dict = {"images": images}

            #targets = targets_dict["tacfrget"]

            #output = model.image_model(inputs_dict["images"])
            output = model(inputs)

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
        #all_ids.extend(batch[0]["id"])
        targets_cat = torch.cat(all_targets).numpy()
        probs_cat = torch.cat(all_probs).numpy()
        auroc = compute_roc_auc(targets_cat, probs_cat)

    logits_cat = torch.cat(all_logits).numpy()
    saved = probs_cat if save_type == "probs" else logits_cat
    return loss_meter.avg, acc_meter.avg, auroc, all_ids, saved

def train_epoch(model, loader, optimizer, loss_fn=nn.CrossEntropyLoss(), use_cuda=True):


    loss_meter, acc_meter = [AverageMeter()]*2
    all_targets, all_probs = [], []
    auroc = -1
    model.train()
    for batch_idx, batch in enumerate(loader):

        inputs, targets = batch
        #print(inputs.size())
        #print(targets.size())
        #targets_data = targets["target"]
        #images = inputs["image"]
        if use_cuda:
            targets = targets.cuda(non_blocking=True)
            inputs = inputs.cuda(non_blocking=True)
        #targets_dict = {"target": targets_data}
        #inputs_dict = {"images": images}

        #targets = targets_dict["tacfrget"]

        #output = model.image_model(inputs_dict["images"])
        output = model(inputs)

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

def train(model, optimizer, scheduler, loaders, args, use_cuda=True):
    loss_fn = nn.CrossEntropyLoss()

    ### implement metric choice

    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0
    metrics = {
        "train_loss": [],
        "val_loss": [],
        "train_acc": [],
        "val_acc": [],
        "train_auroc": [],
        "val_auroc": [],
    }

    for epoch in range(args.epochs):

        scheduler.last_epoch = epoch - 1
        scheduler_args = []
        with warnings.catch_warnings():
            if epoch == 0:
                warnings.simplefilter("ignore")
            scheduler.step(*scheduler_args)

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


        if val_acc >= best_acc:
            best_acc = val_acc
            best_model_wts = copy.deepcopy(model.state_dict())

    model.load_state_dict(best_model_wts)
    print(f"\nBest Val Acc: {best_acc:.3f}")
    return model, best_acc, metrics



def main():
    args = parse_args()
    use_cuda = torch.cuda.is_available()
    if use_cuda:
        print("Using CUDA...")
        #torch.backends.cudnn.deterministic = args.deterministic
        #torch.backends.cudnn.benchmark = not args.deterministic


    #load in data loader

    if args.ood_shift is not None:
        loaders = fetch_dataloaders(args.train_set,"/media",0.2,args.seed,args.batch_size,4, ood_set= args.test_set, ood_shift = args.ood_shift)
    else:
        loaders = fetch_dataloaders(args.train_set,"/media",0.2,args.seed,args.batch_size,4)
    #dls = fetch_dataloaders("cxr_p","/media",0.2,0,32,4, ood_set='mimic_cxr', ood_shift='hospital')

    num_classes = 2 #loaders["train"].dataset.num_classes

    if args.checkpoint_dir is None:


        model = models.resnet50(pretrained=True)
        num_ftrs = model.fc.in_features
        model.fc = nn.Linear(num_ftrs, num_classes)
        model.cuda()


        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        model = model.to(device)
        params_to_update = model.parameters()

        ##weight decay is L2
        #optimizer = optim.SGD(params_to_update, lr=0.0001, weight_decay=0.0001, momentum=0.9, nesterov=True)
        print(f"lr: {args.lr} and l2: {args.wd} and seed {args.seed}")
        optimizer = optim.Adam(params_to_update, lr=args.lr, weight_decay=args.wd)
        
        scheduler = build_scheduler(args, optimizer)

        best_model, val_acc, metrics = train(model, optimizer, scheduler,loaders, args, use_cuda=True)


        #save model 

        if args.save_model:

            save_path = f"{args.save_dir}/train_set_{args.train_set}/seed_{args.seed}"

            os.makedirs(save_path, exist_ok=True)

            torch.save(model.state_dict(), save_path + "/model.pt")
            print(f"Saved Best Model to {save_path}")
        
        test_loss, test_acc, test_auroc, _, _ = evaluate(model, loaders['test'], loss_fn=nn.CrossEntropyLoss())
        print(f"Best Test Auroc {test_auroc}")

        save_dict = {"test_loss": test_loss, "test_acc": test_acc, "test_auroc": test_auroc}

        #save results 

        save_res = f"{args.save_dir}/train_set_{args.train_set}/test_set_{args.test_set}/seed_{args.seed}"
        os.makedirs(save_res, exist_ok=True)
        save_res = save_res + "/results.json"

        with open(save_res, 'w') as fp:
            json.dump(save_dict, fp)


    else:
        num_classes = 2
        print("Using checkpointed model...")
        model = models.resnet50(pretrained=True)
        num_ftrs = model.fc.in_features
        model.fc = nn.Linear(num_ftrs, num_classes)
        model.load_state_dict(torch.load(args.checkpoint_dir))
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        model = model.to(device)
        print(f"evaluating on {args.test_set}:")
        test_loss, test_acc, test_auroc, _, _ = evaluate(model, loaders['test'], loss_fn=nn.CrossEntropyLoss())
        print(f"Best Test Auroc {test_auroc}")
        save_dict = {"test_loss": test_loss, "test_acc": test_acc, "test_auroc": test_auroc}
        
        #save results 

        
        save_res = f"{args.save_dir}/train_set_{args.train_set}/test_set_{args.test_set}/seed_{args.seed}"
        os.makedirs(save_res, exist_ok=True)
        save_res = save_res + "/results.json"

        with open(save_res, 'w') as fp:
            json.dump(save_dict, fp)

if __name__ == "__main__":
    main()