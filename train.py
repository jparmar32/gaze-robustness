import argparse
import glob
import json
import os
import pickle
import sys
import warnings
import sklearn.metrics as skl
import json
import math


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
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter

from utils import AverageMeter, accuracy, compute_roc_auc, build_scheduler, get_lrs, calculate_actdiff_loss
from models.extract_CAM import get_CAM_from_img
from models.extract_activations import get_model_activations

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

    parser.add_argument("--train_set", type=str, choices=['cxr_a','cxr_p', 'synth'], required=True, help="Set to train on")
    parser.add_argument("--test_set", type=str, choices=['cxr_a','cxr_p', 'mimic_cxr', 'chexpert', 'chestxray8', 'synth'], required=True, help="Test set to evaluate on")
    parser.add_argument("--ood_shift", type=str, choices=['hospital','hospital_age', 'age', None], default=None, help="Distribution shift to experiment with")

    parser.add_argument("--save_dir", type=str, default="/mnt/gaze_robustness_results/resnet_only", help="Save Dir")
    parser.add_argument("--subclass_eval", action='store_true', help="Whether to report subclass performance metrics on the test set")
    parser.add_argument("--num_classes", type=int, default=2, help="Number of classes in the training set")

    parser.add_argument("--gaze_task", type=str, choices=['data_augment','cam_reg', 'cam_reg_convex', 'segmentation_reg', 'actdiff', 'actdiff_gaze', 'actdiff_lungmask', None], default=None, help="Type of gaze enhanced or baseline expeeriment to try out")
    parser.add_argument("--cam_weight", type=float, default=0, help="Weight to apply to the CAM Regularization with Gaze Heatmap Approach")
    parser.add_argument("--cam_convex_alpha", type=float, default=0, help="Weight in the convex combination of average gaze heatmap and image specific")

    parser.add_argument("--gan_positive_model", type=str, default=None, help="Location of saved GAN model for the positive class")
    parser.add_argument("--gan_negative_model", type=str, default=None, help="Location of saved GAN model for the negative class")
    parser.add_argument("--gan_type", type=str, choices=['gan','wgan', 'acgan', 'cgan', None], default=None, help="GAN type to load in")

    parser.add_argument("--actdiff_lambda", type=float, default=0, help="Weight associated with the actdiff loss")
    parser.add_argument("--actdiff_gaze_threshold", type=float, default=0, help="Value to threshold Gaze Heatmaps at")
    parser.add_argument("--actdiff_segmask_size", type=int, default=224, help="Segmentation Mask Resolution to Use")
    parser.add_argument("--actdiff_gazemap_size", type=int, default=7, help="Gaze Heatmap Resolution to Use")
    parser.add_argument("--actdiff_lungmask_size", type=int, default=224, help="Lungmask Resolution to Use")
    parser.add_argument("--actdiff_similarity_type", type=str, default="l2", help="Type of similarity metric to use between embeddings of masked and regular images")
    parser.add_argument("--actdiff_augmentation_type", type=str, choices=['normal', 'gaussian_noise', 'gaussian_blur', 'color_jitter', 'color_gamma', 'sobel_horizontal', 'sobel_magnitude' ], default='normal', help="Type of augmentation to use on the masked images")
    parser.add_argument("--actdiff_segmentation_classes", type=str, choices=['all', 'positive'], default='positive', help="Which classes to use for the segmentations in ActDiff")


    args = parser.parse_args()
    return args

def evaluate(
    model,
    loader,
    args,
    loss_fn=nn.CrossEntropyLoss(),
    use_cuda=True,
    save_type="probs",

):


    if args.subclass_eval:

        loss_meter_dict = {} ## class label is first
        acc_meter_dict = {}
        auroc_dict = {}
        all_targets_dict = {}
        all_probs_dict = {}
        all_logits_dict = {}

        for i in range(args.num_classes):
            for j in range(2):

                loss_meter_dict[(i,j)] = AverageMeter()
                acc_meter_dict[(i,j)] = AverageMeter()
                all_targets_dict[(i,j)] = []
                all_probs_dict[(i,j)] = []
                all_logits_dict[(i,j)] = []

        auroc_dict["robust_auroc"] = 0
        auroc_dict["majority_auroc"] = 0

        all_ids = []
        model.eval()
        for batch_idx, batch in enumerate(loader):
            with torch.no_grad():
                inputs, targets, subclass_labels = batch
    
                if use_cuda:
                    targets = targets.cuda(non_blocking=True)
                    inputs = inputs.cuda(non_blocking=True)

                output = model(inputs)
                sub_groups = {}
                for i in range(args.num_classes):
                    for j in range(2):
                        sub_groups[(i,j)] = [[], []]

                output_np = output.cpu().numpy()
                targets_np = targets.cpu().numpy()
                subclass_labels_np = subclass_labels.cpu().numpy()
                for out, target, subclass in zip(output_np, targets_np, subclass_labels_np):
                    ret_list = sub_groups[(target, subclass)]
                    ret_list[0].append(out)
                    ret_list[1].append(target)
                    sub_groups[(target, subclass)] = ret_list

                sub_acc_group = {}
                sub_total_group = {}
                sub_loss_group = {}
                sub_logits_group = {}
                sub_target_group = {}

                for key, val in sub_groups.items():
                    if len(val[0]) != 0:

                        output_torch = torch.Tensor(val[0])
                        target_torch = torch.LongTensor(val[1])

                        dev = "cpu" 
                        if use_cuda:  
                            dev = "cuda" 
                        device = torch.device(dev)  
                        output_torch = output_torch.to(device)
                        target_torch = target_torch.to(device)
        
                        
                        if isinstance(output_torch, tuple):
                            logits, attn = output_torch
                        else:
                            logits, attn = output_torch, None

                        c_loss = loss_fn(logits, target_torch)
                        loss = c_loss
                        acc, (_, total) = accuracy(target_torch, logits, return_stats=True)

                        sub_acc_group[key] = acc
                        sub_total_group[key] = total
                        sub_loss_group[key] = loss
                        sub_logits_group[key] = logits
                        sub_target_group[key] = target_torch


            for key, val in sub_acc_group.items():

                acc_meter_dict[key].update(sub_acc_group[key], sub_total_group[key])
                loss_meter_dict[key].update(sub_loss_group[key], sub_total_group[key])
                probs = F.softmax(sub_logits_group[key].data, dim=1)
                all_targets_dict[key].append(sub_target_group[key].cpu())
                all_logits_dict[key].append(sub_logits_group[key].data.cpu())
                all_probs_dict[key].append(probs.cpu())
                targets_cat = torch.cat(all_targets_dict[key]).numpy()
                probs_cat = torch.cat(all_probs_dict[key]).numpy()
                
        ## class label is 1 and subclass is 0, class label is 0 and subclass is 1
        robust_targets = all_targets_dict[(1,0)] + all_targets_dict[(0,1)]
        robust_probs = all_probs_dict[(1,0)] + all_probs_dict[(0,1)]

        majority_targets = all_targets_dict[(1,1)] + all_targets_dict[(0,0)]
        majority_probs = all_probs_dict[(1,1)] + all_probs_dict[(0,0)]

        robust_targets_cat = torch.cat(robust_targets).numpy()
        robust_probs_cat = torch.cat(robust_probs).numpy()

        majority_targets_cat = torch.cat(majority_targets).numpy()
        majority_probs_cat = torch.cat(majority_probs).numpy()

        auroc_dict["robust_auroc"] = compute_roc_auc(robust_targets_cat, robust_probs_cat)
        auroc_dict["majority_auroc"] = compute_roc_auc(majority_targets_cat, majority_probs_cat)
         

        saved_dict = {}
        for i in range(args.num_classes):
            for j in range(2):
                logits_cat = torch.cat(all_logits_dict[(i,j)]).numpy() #cangee if needed
                all_logits_dict[(i,j)] = logits_cat
                probs_cat = torch.cat(all_probs_dict[(i,j)]).numpy() #change if needed
                saved_dict[(i,j)] = probs_cat if save_type == "probs" else logits_cat
                loss_meter_dict[(i,j)] = loss_meter_dict[(i,j)].avg
                acc_meter_dict[(i,j)] = acc_meter_dict[(i,j)].avg

        return loss_meter_dict, acc_meter_dict, auroc_dict, all_ids, saved_dict
        
    else: 

        loss_meter, acc_meter = [AverageMeter() for i in range(2)]
        auroc = -1
        all_targets, all_probs, all_logits = [], [], []

        all_ids = []
        model.eval()
        for batch_idx, batch in enumerate(loader):
            with torch.no_grad():
                inputs, targets, subclass_labels = batch
    
                if use_cuda:
                    targets = targets.cuda(non_blocking=True)
                    inputs = inputs.cuda(non_blocking=True)

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
            targets_cat = torch.cat(all_targets).numpy()
            probs_cat = torch.cat(all_probs).numpy()
            auroc = compute_roc_auc(targets_cat, probs_cat)

        logits_cat = torch.cat(all_logits).numpy()
        saved = probs_cat if save_type == "probs" else logits_cat
        return loss_meter.avg, acc_meter.avg, auroc, all_ids, saved

def train_epoch(model, loader, optimizer, loss_fn=nn.CrossEntropyLoss(), use_cuda=True, cam_weight=0, cam_convex_alpha=0, gaze_task=None, args = None, writer = None, global_step = None):


    loss_meter, acc_meter = [AverageMeter()]*2
    all_targets, all_probs = [], []
    auroc = -1
    model.train()
    for batch_idx, batch in tqdm(enumerate(loader), total = len(loader)):

        #gaze_attribute here is tthe heatmap
        inputs, targets, gaze_attributes = batch

        if use_cuda:
            targets = targets.cuda(non_blocking=True)
            inputs = inputs.cuda(non_blocking=True)
            if gaze_task in ["actdiff", 'actdiff_gaze', 'actdiff_lungmask']:
                gaze_attributes = gaze_attributes.cuda(non_blocking=True)
      
        output = model(inputs)

        if isinstance(output, tuple):
            logits, attn = output
        else:
            logits, attn = output, None

        c_loss = loss_fn(logits, targets)


        a_loss = 0
        if cam_weight:
            n = len(targets)
            cum_losses = logits.new_zeros(n)
            for i in range(n):
                image = inputs[i,...]
                cam = get_CAM_from_img(image, model, targets[i].cpu())

                if gaze_task == "cam_reg_convex":
                    eye_hm = gaze_attributes['gaze_attribute'][i,...]
                    eye_hm = cam_convex_alpha*eye_hm + (1 - cam_convex_alpha)*gaze_attributes['average_hm'][i,...]
                else:
                    eye_hm = gaze_attributes[i,...]

                if len(eye_hm.shape) != 2:
                    eye_hm = eye_hm.reshape(int(math.sqrt(eye_hm.shape[0])),int(math.sqrt(eye_hm.shape[0])))

                if eye_hm.shape != cam.shape:
                    pool_dim = int(eye_hm.shape[0] / cam.shape[0])
                    eye_hm = nn.functional.avg_pool2d(eye_hm.unsqueeze(0).unsqueeze(0), pool_dim).squeeze()
                    
                eye_hm_norm = eye_hm / eye_hm.sum()
                cam_normalized = cam / cam.sum()
                eye_hm_norm = eye_hm_norm.to(device="cuda:0")
                cam_normalized = cam_normalized.to(device="cuda:0")

                if gaze_task == "segmentation_reg":
                    if not torch.equal(eye_hm,-1*torch.ones((7,7))):
                        if not (torch.isnan(cam_normalized).any() or torch.isnan(eye_hm_norm).any()):
                            cum_losses[i] += cam_weight * torch.nn.functional.mse_loss(eye_hm_norm,cam_normalized,reduction='sum')
                else:
                    if not (torch.isnan(cam_normalized).any() or torch.isnan(eye_hm_norm).any()):
                        cum_losses[i] += cam_weight * torch.nn.functional.mse_loss(eye_hm_norm,cam_normalized,reduction='sum')
            a_loss = cum_losses.sum()

        
        
        actdiff_loss = 0
        if args.actdiff_lambda:
            batch_size = len(targets)
            cum_activation_losses = logits.new_zeros(batch_size)
            for i in range(batch_size):
                image = inputs[i,...].unsqueeze(0)
                masked_image = gaze_attributes[i,...].unsqueeze(0)
                regular_activations = get_model_activations(image, model)
                masked_activations = get_model_activations(masked_image, model)
                cum_activation_losses[i] = args.actdiff_lambda * calculate_actdiff_loss(regular_activations, masked_activations, args.actdiff_similarity_type)
            actdiff_loss = cum_activation_losses.sum()

        ## writer log losses
        if global_step % 2 == 0:
            writer.add_scalar(f"train/ce_loss", c_loss, global_step)
            writer.add_scalar(f"train/actdiff_loss", actdiff_loss, global_step)

        
        if args.actdiff_similarity_type == "l2":
            loss = c_loss + a_loss + actdiff_loss
        elif args.actdiff_similarity_type == "cosine":
            loss = c_loss + a_loss - actdiff_loss

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

        ## writer log acc, auroc
        if global_step % 2 == 0:
            writer.add_scalar(f"train/acc", acc, global_step)
            writer.add_scalar(f"train/auroc", auroc, global_step)

        global_step += 1

    return loss_meter.avg, acc_meter.avg, auroc, global_step

def train(model, optimizer, scheduler, loaders, args, use_cuda=True, writer=None):
    loss_fn = nn.CrossEntropyLoss()

    ### implement metric choice

    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0
    global_step = 0
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

        ## writer log train metrics in here
        train_loss, train_acc, train_auroc, global_step = train_epoch(
            model,
            loaders["train"],
            optimizer,
            loss_fn=loss_fn,
            use_cuda=use_cuda,
            cam_weight=args.cam_weight,
            cam_convex_alpha=args.cam_convex_alpha,
            gaze_task=args.gaze_task,
            args = args,
            writer = writer,
            global_step = global_step
        )

        val_loss, val_acc, val_auroc, _, _= evaluate(
            model,
            loaders["val"],
            args,
            loss_fn=loss_fn,
            use_cuda=use_cuda,
        )

        metrics['train_loss'].append(train_loss)
        metrics['train_acc'].append(train_acc)
        metrics['train_auroc'].append(train_auroc)
        metrics['val_loss'].append(val_loss)
        metrics['val_acc'].append(val_acc)
        metrics['val_auroc'].append(val_auroc)

        ##writer log val metrics

        writer.add_scalar(f"val/loss", val_loss, epoch)
        writer.add_scalar(f"val/acc", val_acc, epoch)
        writer.add_scalar(f"val/auroc", val_auroc, epoch)


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
        loaders = fetch_dataloaders(args.train_set,"/media",0.2,args.seed,args.batch_size,4, gaze_task=args.gaze_task, ood_set= args.test_set, ood_shift = args.ood_shift, gan_positive = args.gan_positive_model, gan_negative = args.gan_negative_model, gan_type = args.gan_type, args = args)
    else:
        loaders = fetch_dataloaders(args.train_set,"/media",0.2,args.seed,args.batch_size,4, gaze_task=args.gaze_task, subclass=args.subclass_eval, gan_positive = args.gan_positive_model , gan_negative = args.gan_negative_model, gan_type = args.gan_type, args = args)
    #dls = fetch_dataloaders("cxr_p","/media",0.2,0,32,4, ood_set='mimic_cxr', ood_shift='hospital')
    if args.gaze_task is not None:
        print(f"Running a gaze experiment: {args.gaze_task}")


    num_classes = args.num_classes #loaders["train"].dataset.num_classes

    if args.checkpoint_dir is None:

        writer = SummaryWriter(os.path.join(args.save_dir, f'train_set_{args.train_set}') )


        model = models.resnet50(pretrained=True) ### may have to make own resnet class if this doesn't work
        num_ftrs = model.fc.in_features
        model.fc = nn.Linear(num_ftrs, num_classes)
        model.cuda()


        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = model.to(device)
        params_to_update = model.parameters()

        ##weight decay is L2
        #optimizer = optim.SGD(params_to_update, lr=0.0001, weight_decay=0.0001, momentum=0.9, nesterov=True)
        print(f"lr: {args.lr} and l2: {args.wd} and seed: {args.seed} and actdiff lungmask size: {args.actdiff_lungmask_size} and actdiff similarity: {args.actdiff_similarity_type} and actdiff lambda: {args.actdiff_lambda} and augmentation_type: {args.actdiff_augmentation_type} and segmentation classes: {args.actdiff_segmentation_classes} ")
        optimizer = optim.Adam(params_to_update, lr=args.lr, weight_decay=args.wd)
        
        scheduler = build_scheduler(args, optimizer)

        best_model, val_acc, metrics = train(model, optimizer, scheduler,loaders, args, use_cuda=True, writer=writer)
        writer.close()

        #save model 

        if args.save_model:

            save_path = f"{args.save_dir}/train_set_{args.train_set}/seed_{args.seed}"

            os.makedirs(save_path, exist_ok=True)

            torch.save(best_model.state_dict(), save_path + "/model.pt")
            print(f"Saved Best Model to {save_path}")
        
        test_loss, test_acc, test_auroc, _, _ = evaluate(model, loaders['test'], args, loss_fn=nn.CrossEntropyLoss())
        if not args.subclass_eval:
            val_loss, val_acc, val_auroc, _, _ = evaluate(model, loaders['val'], args, loss_fn=nn.CrossEntropyLoss())
        if args.subclass_eval:
            print(f"Best Test Acc {test_acc}")
        else:
            print(f"Best Test Auroc {test_auroc}")

        save_dict = {"test_loss": test_loss, "test_acc": test_acc, "test_auroc": test_auroc}
        if not args.subclass_eval:
            val_save_dict = {"val_loss": val_loss, "val_acc": val_acc, "val_auroc": val_auroc}

        #save results 

        if args.subclass_eval:
            save_res = f"{args.save_dir}/train_set_{args.train_set}/test_set_{args.test_set}_subclass_evaluation/seed_{args.seed}"
            max_loss = max(save_dict['test_loss'].values())
            min_acc = min(save_dict['test_acc'].values())
            save_dict = {"test_loss": max_loss, "test_acc": min_acc, "robust_auroc": save_dict['test_auroc']["robust_auroc"]}
        elif args.ood_shift is not None:
            save_res = f"{args.save_dir}/train_set_{args.train_set}/test_set_{args.test_set}/ood_shift_{args.ood_shift}/seed_{args.seed}"
            val_save_res = f"{args.save_dir}/train_set_{args.train_set}/val_set/ood_shift_{args.ood_shift}/seed_{args.seed}"
        else:
            save_res = f"{args.save_dir}/train_set_{args.train_set}/test_set_{args.test_set}/seed_{args.seed}"
            val_save_res = f"{args.save_dir}/train_set_{args.train_set}/val_set_{args.train_set}/seed_{args.seed}"
        os.makedirs(save_res, exist_ok=True)
        save_res = save_res + "/results.json"

        if not args.subclass_eval:
            os.makedirs(val_save_res, exist_ok=True)
            val_save_res = val_save_res + "/results.json"

        with open(save_res, 'w') as fp:
            json.dump(save_dict, fp)

        if not args.subclass_eval:
            with open(val_save_res, 'w') as fp:
                json.dump(val_save_dict, fp)


    else:
        num_classes = args.num_classes
        print("Using checkpointed model...")
        model = models.resnet50(pretrained=True)
        num_ftrs = model.fc.in_features
        model.fc = nn.Linear(num_ftrs, num_classes)
        model.load_state_dict(torch.load(args.checkpoint_dir))
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = model.to(device)
        print(f"evaluating on {args.test_set}:")
        test_loss, test_acc, test_auroc, _, _ = evaluate(model, loaders['test'], args, loss_fn=nn.CrossEntropyLoss())
        if not args.subclass_eval:
            val_loss, val_acc, val_auroc, _, _ = evaluate(model, loaders['val'], args, loss_fn=nn.CrossEntropyLoss())

        if args.subclass_eval:
            print(f"Best Test Acc {test_acc}")
        else:
            print(f"Best Test Auroc {test_auroc}")
        save_dict = {"test_loss": test_loss, "test_acc": test_acc, "test_auroc": test_auroc}

        if not args.subclass_eval:
            val_save_dict = {"val_loss": val_loss, "val_acc": val_acc, "val_auroc": val_auroc}

        #save results 
        if args.subclass_eval:
            save_res = f"{args.save_dir}/train_set_{args.train_set}/test_set_{args.test_set}_subclass_evaluation/seed_{args.seed}"
            max_loss = max(save_dict['test_loss'].values())
            min_acc = min(save_dict['test_acc'].values())
            save_dict = {"test_loss": max_loss, "test_acc": min_acc, "robust_auroc": save_dict['test_auroc']["robust_auroc"]}

        elif args.ood_shift is not None:
            save_res = f"{args.save_dir}/train_set_{args.train_set}/test_set_{args.test_set}/ood_shift_{args.ood_shift}/seed_{args.seed}"
            val_save_res = f"{args.save_dir}/train_set_{args.train_set}/val_set_{args.train_set}/seed_{args.seed}"
        else:
            save_res = f"{args.save_dir}/train_set_{args.train_set}/test_set_{args.test_set}/seed_{args.seed}"
            val_save_res = f"{args.save_dir}/train_set_{args.train_set}/val_set_{args.train_set}/seed_{args.seed}"
        os.makedirs(save_res, exist_ok=True)
        save_res = save_res + "/results.json"

        if not args.subclass_eval:
            os.makedirs(val_save_res, exist_ok=True)
            val_save_res = val_save_res + "/results.json"

        with open(save_res, 'w') as fp:
            json.dump(save_dict, fp)

        if not args.subclass_eval:
            with open(val_save_res, 'w') as fp:
                json.dump(val_save_dict, fp)

if __name__ == "__main__":
    main()