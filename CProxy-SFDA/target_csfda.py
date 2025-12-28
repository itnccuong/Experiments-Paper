from copy import deepcopy
import logging
import os
import time
from weakref import proxy

from tqdm import tqdm
import torch
import torch.distributed as dist
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
import wandb
import math
import torch.nn as nn 

from classifier import Classifier
from image_list import ImageList
from Model import CSFDA_Model, NCropsTransform
import numpy as np
from Contrastive_loss import *
import matplotlib.pyplot as plt
import torchvision.utils as utils
from sklearn.manifold import TSNE

from utils import (
    adjust_learning_rate,
    concat_all_gather,
    get_augmentation,
    get_distances,
    is_master,
    per_class_accuracy,
    log_per_class_thresholds,
    remove_wrap_arounds,
    save_checkpoint,
    use_wandb,
    AverageMeter,
    CustomDistributedDataParallel,
    ProgressMeter,
    plot_training_stats,
    linear_rampup,
    interleave,
)


@torch.no_grad()
def visualize_tsne(dataloader, model, args, save_path="tsne_visualization.png", max_samples=2000):
    """Generate t-SNE visualization of feature representations."""
    model.eval()
    
    features_list = []
    labels_list = []
    paths_list = []
    
    logging.info("Extracting features for t-SNE...")
    num_samples = 0
    
    for _, imgs, labels, idxs in dataloader:
        if num_samples >= max_samples:
            break
            
        imgs = imgs.to("cuda")
        
        # Extract features (not logits)
        feats, _ = model(imgs, cls_only=True)
        
        features_list.append(feats.cpu())
        labels_list.append(labels)
        
        num_samples += imgs.size(0)
    
    # Concatenate all features and labels
    features = torch.cat(features_list, dim=0)[:max_samples]
    labels = torch.cat(labels_list, dim=0)[:max_samples]
    
    logging.info(f"Running t-SNE on {features.shape[0]} samples...")
    
    # Apply t-SNE
    tsne = TSNE(n_components=2, random_state=42, perplexity=30, n_iter=1000)
    features_2d = tsne.fit_transform(features.numpy())
    
    # Plot
    plt.figure(figsize=(10, 8))
    
    # Determine number of classes
    num_classes = labels.max().item() + 1
    
    # Use a colormap
    colors = plt.cm.tab20(np.linspace(0, 1, num_classes))
    
    for class_id in range(num_classes):
        mask = labels == class_id
        plt.scatter(
            features_2d[mask, 0],
            features_2d[mask, 1],
            c=[colors[class_id]],
            label=f"Class {class_id}",
            alpha=0.6,
            s=20
        )
    
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=8)
    plt.title("t-SNE Visualization of Feature Space")
    plt.xlabel("t-SNE Dimension 1")
    plt.ylabel("t-SNE Dimension 2")
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    logging.info(f"t-SNE visualization saved to {save_path}")


@torch.no_grad()
def eval_and_label_dataset(dataloader, model, args, per_class_conf_th=None, per_class_uncer_th=None):
    wandb_dict = dict()

    # Make sure to switch to eval mode
    model.eval()

    # Run inference
    logits, gt_labels, indices = [], [], []
    logging.info("Eval and labeling...")
    iterator = tqdm(dataloader) if is_master(args) else dataloader
    for _ , imgs, labels, idxs in iterator:
        imgs = imgs.to("cuda")

        # (B, D) x (D, K) -> (B, K)
        _, logits_cls = model(imgs, cls_only=True)

        logits.append(logits_cls)
        gt_labels.append(labels)
        indices.append(idxs)

    logits    = torch.cat(logits)
    gt_labels = torch.cat(gt_labels).to("cuda")
    indices   = torch.cat(indices).to("cuda")

    if args.distributed:

        ## Gather results from all ranks
        logits = concat_all_gather(logits)
        gt_labels = concat_all_gather(gt_labels)
        indices = concat_all_gather(indices)

        ## Remove extra wrap-arounds from DDP
        ranks = len(dataloader.dataset) % dist.get_world_size()
        logits = remove_wrap_arounds(logits, ranks)
        gt_labels = remove_wrap_arounds(gt_labels, ranks)
        indices = remove_wrap_arounds(indices, ranks)

    assert len(logits) == len(dataloader.dataset)

    pred_labels = logits.argmax(dim=1)
    accuracy = (pred_labels == gt_labels).float().mean() * 100
    logging.info(f"Accuracy of direct prediction: {accuracy:.2f}")
    wandb_dict["Test Acc"] = accuracy
    if args.data.dataset == "VISDA-C":
        acc_per_class = per_class_accuracy(
            y_true=gt_labels.cpu().numpy(),
            y_pred=pred_labels.cpu().numpy(),
        )
        wandb_dict["Test Avg"] = acc_per_class.mean()
        wandb_dict["Test Per-class"] = acc_per_class
        
        # Log thresholds right after accuracy
        if per_class_conf_th is not None and per_class_uncer_th is not None:
            conf_mean, uncer_mean = log_per_class_thresholds(
                per_class_conf_th, 
                per_class_uncer_th, 
                name="Current Thresholds"
            )
            return acc_per_class, conf_mean, uncer_mean

    if use_wandb(args):
        wandb.log(wandb_dict)

    return acc_per_class



def get_augmentation_versions(args):
    """
    Get a list of augmentations. "w" stands for weak, "s" stands for strong.
    E.g., "wss" stands for one weak, two strong.
    """
    transform_list = []
    for version in args.learn.aug_versions:    ## Change the value of augmented versions 
        if version == "s":
            transform_list.append(get_augmentation(args.data.aug_type))
        elif version == "w":
            transform_list.append(get_augmentation("plain"))
        elif version == "t":
            transform_list.append(get_augmentation("test"))
        else:
            raise NotImplementedError(f"{version} version not implemented.")
    
    transform = NCropsTransform(transform_list)

    return transform


def get_target_optimizer(model, args):
    if args.distributed:
        model = model.module
    backbone_params, extra_params = (
        model.src_model.get_params()
        if hasattr(model, "src_model")
        else model.get_params()
    )

    if args.optim.name == "sgd":
        optimizer = torch.optim.SGD(
            [
                {
                    "params": backbone_params,
                    "lr": args.optim.lr,
                    "momentum": args.optim.momentum,
                    "weight_decay": args.optim.weight_decay,
                    "nesterov": args.optim.nesterov,
                },
                {
                    "params": extra_params,
                    "lr": 30*args.optim.lr,                  ## For Fully test-time domain adaptation, don't use a high learning rate for VISDA-C.
                                                             ## And, For DomainNet-126, best online result comes when Learning Rate is 2e-4, and use LRM of 20
                    "momentum": args.optim.momentum,
                    "weight_decay": args.optim.weight_decay,
                    "nesterov": args.optim.nesterov,
                },
            ]
        )
    else:
        raise NotImplementedError(f"{args.optim.name} not implemented.")

    for param_group in optimizer.param_groups:
        param_group["lr0"] = param_group["lr"]  # snapshot of the initial lr

    return optimizer



def train_target_domain(args):
    logging.info(
        f"Start target training on {args.data.src_domain}-{args.data.tgt_domain}..."
    )

    # If not specified, use the full length of dataset.
    if args.learn.queue_size == -1:
        label_file = os.path.join(
            args.data.image_root, f"{args.data.tgt_domain}_list.txt"
        )
        dummy_dataset = ImageList(os.path.join(args.data.image_root, args.data.tgt_domain), label_file)
        data_length   = len(dummy_dataset)
        args.learn.queue_size = data_length
        del dummy_dataset

    checkpoint_path = os.path.join(
        args.model_tta.src_log_dir,
        f"best_{args.data.src_domain}_{args.seed}.pth.tar",
    )

    # filename = f"checkpoint_0001_{args.data.src_domain}-{args.data.tgt_domain}-{args.sub_memo}_{args.seed}.pth.tar"
    # checkpoint_path = os.path.join(args.log_dir, filename)    

    ## Model Initization
    src_model = Classifier(args.model_src, checkpoint_path)
    ema_model = Classifier(args.model_src, checkpoint_path)       ## For contrastive loss
    
    # TODO: Check this model method
    model = CSFDA_Model(
        src_model,
        ema_model,
        m=args.model_tta.m, # m is the ema coefficient
    ).cuda()

    # TODO: Check the distributed mode later
    if args.distributed:
        model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)
        model = CustomDistributedDataParallel(model, device_ids=[args.gpu])
    logging.info(f"1 - Created target model")

    ## Validation Data
    val_transform   = get_augmentation("test") # test is the name of type aug
    label_file  = os.path.join(args.data.image_root, f"{args.data.tgt_domain}_list.txt")
    val_dataset = ImageList( # Imagelist is dataset class that basically also perform transformations
        image_root=os.path.join(args.data.image_root, args.data.tgt_domain),
        label_file=label_file,
        transform =val_transform,
    )
    
    # Basically a distributed component for dataloader
    val_sampler = (
        DistributedSampler(val_dataset, shuffle=False) if args.distributed else None
    )
    
    val_loader = DataLoader(
        val_dataset, batch_size=256, sampler=val_sampler, num_workers=2
    )

    acc = eval_and_label_dataset(
        val_loader, model, args=args
    )
    logging.info("2 - Computed initial pseudo labels")

    ### Training data  (Is it domain by domain, look for online settings!)
    train_transform = get_augmentation_versions(args)
    train_dataset = ImageList(
        image_root=os.path.join(args.data.image_root, args.data.tgt_domain),
        label_file=label_file,                # Uses pseudo labels
        transform=train_transform,
        pseudo_item_list=None,
    )
    train_sampler = DistributedSampler(train_dataset) if args.distributed else None
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.data.batch_size,
        shuffle=(train_sampler is None),
        num_workers=args.data.workers,
        pin_memory=False,            ## make it False, otherwise 
        sampler=train_sampler,
        drop_last=False,
    )

    args.learn.full_progress = args.learn.epochs * len(train_loader)
    logging.info("3 - Created train/val loader")

    ### Define loss function (criterion) and optimizer   
    # The SGD optimizer with different learning rates for backbone (feature extractor) and extra layers (classifier)
    optimizer = get_target_optimizer(model, args)
    logging.info("4 -- Created Optimizer")

    # Resume from checkpoint if provided
    if args.resume:
        if os.path.isfile(args.resume):
            logging.info(f"=> loading checkpoint '{args.resume}'")
            checkpoint = torch.load(args.resume, map_location='cuda')
            args.learn.start_epoch = checkpoint['epoch'] + 1
            model.load_state_dict(checkpoint['state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer'])
            logging.info(f"=> loaded checkpoint '{args.resume}' (epoch {checkpoint['epoch']})")
        else:
            logging.info(f"=> no checkpoint found at '{args.resume}'")

    logging.info("Start Training ...")

    ### Main Training Part    
    if args.distributed:
        train_sampler.set_epoch(1)

    ### Our Proposed Training Algorithm
    train_csfda(train_loader, val_loader, model, optimizer,  args)

    filename = f"checkpoint_1_{1:04d}_{args.data.src_domain}-{args.data.tgt_domain}-{args.sub_memo}_{args.seed}.pth.tar"
    save_path = os.path.join('./checkpoint/', filename)
    save_checkpoint(model, optimizer, 1, save_path=save_path)
    logging.info(f"Saved checkpoint {save_path}")


            ### Our Proposed Training Algorithm ######
            ##########################################
def train_csfda(train_loader, val_loader, model, optimizer, args):
    
    epoch = 1
    batch_time = AverageMeter("Time", ":6.3f")
    loss_meter = AverageMeter("Loss", ":.4f")
    top1_ins   = AverageMeter("SSL-Acc@1", ":6.2f")
    top1_psd   = AverageMeter("CLS-Acc@1", ":6.2f")
    
    progress = ProgressMeter(
        len(train_loader),
        [batch_time, loss_meter, top1_ins, top1_psd],
        prefix=f"Epoch: [{epoch}]",
    )

    ## Make sure to switch to train mode
    model.train()

    ## Number of Augmentations 
    if args.data.dataset == "VISDA-C":
        num_class = 12
    else:
        num_class = 126

    N = 8
    accuracy_tot = 0
    accuracy_r = 0
    total_acc = 1
    end = time.time()
    zero_tensor = torch.tensor([0.0]).to("cuda")
    loss_coef   = 1
    con_coeff   = 0.5
    contrastive_criterion = SupConLoss()
    L2loss                = torch.nn.MSELoss()
    
    mem_size       = 2
    class_features = torch.zeros((mem_size, num_class, 256))
    probs_class    = torch.zeros((mem_size, num_class, num_class))

    con_coeffs   = [] 
    loss_classes = [] 
    loss_coefs   = [] 
    con_losses   = [] 
    mix_cls_losses = []
    remix_reg_losses = []
    unsupervised_losses    = [] 
    total_losses = [] # New: Track total loss
    uncertainty_thresholds = [] 
    conf_thress = [] 
    acc_classes = []
    accuracies  = []
    batch_accuracies = [] # New: Track overall batch accuracy
    per_class_accs = []
    
    # Track thresholds per epoch
    epoch_conf_th_means = []
    epoch_uncer_th_means = []
    epoch_conf_th_per_class = []
    epoch_uncer_th_per_class = []
    
    sel_Samples = []
    
    # Accumulators for running averages
    accuracy_tot = 0
    total_acc = 0
    batch_acc_tot = 0
    batch_total_acc = 0
    unsel_samples = []
    ind = 0
    
    

    for epoch in range(args.learn.start_epoch, args.learn.epochs):
        progress.prefix = f"Epoch: [{epoch}]"
        
        # init Proxy Domain
        proxy_domain = []
        # Track counts to enforce a global cap per epoch for the proxy domain
        proxy_class_counts = {k: 0 for k in range(num_class)}
        PROXY_PER_CLASS_CAP = 500 # Limit total samples per class in the proxy file
        
        for i, data in enumerate(train_loader):
            
            ## Unpack and move data
            img_path, images, labels_check , idxs = data
            labels_check = labels_check.to("cuda")
            idxs = idxs.to("cuda")

            ## Images for updating the model
            images_w, images_q, images_k = (
                images[0].to("cuda"),
                images[1].to("cuda"),
                images[2].to("cuda"),
            )

            ## (N-3) number of Images for Calculating uncertainty 
            images_un    = torch.stack(images[3:N]).to("cuda")
            outputs_emas = []
            feats = []

            with torch.no_grad():
                for jj in range(N-3):
                    outputs_emas.append(model.ema_model(images_un[jj], return_feats=True)[1])                      

                ## Average the predictions for pseudo-labels
                outputs_ema = torch.stack(outputs_emas).mean(0)
                probs_w, pseudo_labels_w = torch.nn.functional.softmax(outputs_ema, dim=1).max(1)

            ## Per-step scheduler (Learning Rate Decay)
            step = i + epoch * len(train_loader)
            adjust_learning_rate(optimizer, step, args)

            ## Get the logits 
            feats_con, logits_q = model(images_q , images_k)

                                ############################################
                                ########## PL Selection Module ##############
                                #############################################
            pred_start     = torch.nn.functional.softmax(torch.squeeze(torch.stack(outputs_emas)), dim=2).max(2)[0] 

            ## Per-class threshold computation
            batch_size = pseudo_labels_w.size(0)
            pred_con = pred_start  # (N-3, batch_size) confidence scores
            pred_std = pred_start.std(0)  # (batch_size,) uncertainty scores
            
            # Initialize per-class thresholds
            per_class_conf_th = torch.zeros(num_class).to("cuda")
            per_class_uncer_th = torch.zeros(num_class).to("cuda")
            per_class_counts = torch.zeros(num_class).to("cuda")
            
            # Compute per-class thresholds based on EMA predictions
            for c in range(num_class):
                class_mask = (pseudo_labels_w == c)
                class_count = class_mask.sum()
                per_class_counts[c] = class_count
                
                if class_count > 0:
                    # Per-class confidence threshold: mean confidence for this class
                    per_class_conf_th[c] = pred_con[:, class_mask].mean()
                    # Per-class uncertainty threshold: mean uncertainty for this class
                    per_class_uncer_th[c] = pred_std[class_mask].mean()
                else:
                    # If no samples for this class, use global mean as fallback
                    per_class_conf_th[c] = pred_con.mean()
                    per_class_uncer_th[c] = pred_std.mean()
            
            # Compute weighted global thresholds: sum(class_size * class_threshold) / batch_size
            conf_th = (per_class_counts * per_class_conf_th).sum() / batch_size
            uncer_th = (per_class_counts * per_class_uncer_th).sum() / batch_size
            
            ## Confidence Based Selection (per-class threshold for each sample)
            # Each sample is compared against its predicted class's threshold
            sample_conf_th = per_class_conf_th[pseudo_labels_w]  # (batch_size,) threshold for each sample
            confidence_sel = pred_con.mean(0) > sample_conf_th
            
            ## Uncertainty Based Selection (per-class threshold for each sample)
            sample_uncer_th = per_class_uncer_th[pseudo_labels_w]  # (batch_size,) threshold for each sample
            uncertainty_sel = pred_std < sample_uncer_th

            ## Confidence and Uncertainty Based Selection
            truth_array = torch.logical_and(uncertainty_sel, confidence_sel)
            ind_keep   = truth_array.nonzero()
            ind_remove = (~truth_array).nonzero()
            
            try:
                ind_total = torch.cat((torch.squeeze(ind_keep), torch.squeeze(ind_remove)), dim=0)
            except:
                ind_total = ind_remove

            ## Confidence Score Difference (DoC) Based Selection
            if ind_remove.numel():
                threshold = torch.zeros(len(ind_remove))
                num = 0
                for kk in ind_remove:
                    out     = torch.squeeze(outputs_ema[kk])
                    out , _ = out.sort(descending=True)
                    threshold[num] = out[0] - out[1]
                    num    += 1

                pre_threshold = threshold.mean(0) 
                truth_array1  = threshold>pre_threshold
                truth_array2  = pred_std[ind_remove] < pred_std[ind_remove].mean(0)                   ## Add Underconfident Clean Samples 
                truth_array   = torch.logical_and(truth_array1.cuda(), truth_array2.cuda())
                ind_add       = truth_array.nonzero()
                
                try:
                    ind_keep   = torch.cat((torch.squeeze(ind_keep), torch.squeeze(ind_remove[ind_add])), dim=0)
                    ind_remove = torch.stack([kk for kk in ind_total if kk not in ind_keep])
                except:
                    pass 
                            
            try:
                                ### Apply Class-Balancing (Only the selected Samples) ###
                unique_labels, counts =  pseudo_labels_w[ind_keep].unique(return_counts = True)
                min_count             =  torch.min(counts)

                ## For Missing Classes
                if len(counts) < num_class:
                    counts_new = torch.ones(num_class)
                    missing_classes = [ii for ii in range(num_class) if ii not in unique_labels]
                    
                    for kk in missing_classes:
                        indices = (pseudo_labels_w == kk).nonzero(as_tuple=True)[0]

                        if indices.numel()>0 and ind_keep.numel()>0:
                            probs  = probs_w[indices]
                            _ , index_miss = probs.sort(descending=True)                                    
                
                            try:
                                ind_keep  = torch.cat((ind_keep, indices[index_miss[0:min_count]]))         ## Taking all missing classes samples deteriorates the performance
                            except:
                                pass
                            counts_new[kk] = 1 
                        else:
                            counts_new[kk] = 1
                    
                    ## Other Classes
                    num = 0
                    for nn in unique_labels:
                        counts_new[nn] = counts[num]
                        num += 1
                else:
                    counts_new = counts 

                loss_cls , accuracy_psd = classification_loss(
                    torch.squeeze(outputs_ema[ind_keep]), torch.squeeze(logits_q[ind_keep]), torch.squeeze(pseudo_labels_w[ind_keep]), torch.squeeze(outputs_ema[ind_keep]),  args, 1/counts_new.cuda()
                )

            except:
                loss_cls , accuracy_psd = propagation_loss(
                    torch.squeeze(outputs_ema), torch.squeeze(logits_q), torch.squeeze(pseudo_labels_w), torch.squeeze(outputs_ema), args
                )      

            # --- Cap Dominant Classes in Proxy Domain ---
            # To prevent overfitting to easy classes (bias), we limit the max samples per class.
            if ind_keep.numel() > 0:
                MAX_SAMPLES_PER_CLASS = 5
                current_labels_keep = pseudo_labels_w[ind_keep]
                unique_classes, counts = current_labels_keep.unique(return_counts=True)
                
                mask_keep = torch.ones(ind_keep.size(0), dtype=torch.bool).to(ind_keep.device)
                
                for cls, count in zip(unique_classes, counts):
                    if count > MAX_SAMPLES_PER_CLASS:
                        # Find positions of this class in ind_keep
                        cls_positions = (current_labels_keep == cls).nonzero(as_tuple=True)[0]
                        
                        # Randomly select excess to drop
                        perm = torch.randperm(cls_positions.size(0)).to(ind_keep.device)
                        drop_indices = cls_positions[perm[MAX_SAMPLES_PER_CLASS:]]
                        
                        mask_keep[drop_indices] = False
                
                ind_keep = ind_keep[mask_keep]

            # Store reliable samples in Proxy Domain
            # if ind_keep.numel():
            #     # Move indices to cpu
            #     indices_np = idxs[ind_keep].cpu().numpy().flatten()
                
            #     for idx_k in indices_np:
            #         path = img_path[idx_k]
            #         label = pseudo_labels_w[idx_k].item()
                    
            #         # Only add if we haven't reached the cap for this class
            #         if proxy_class_counts[label] < PROXY_PER_CLASS_CAP:
            #             proxy_domain.append(f"{path} {label}")
            #             proxy_class_counts[label] += 1
            

            ### Calculate Pseudo-Label Accuracy Accuracy ###
            # Accuracy of SELECTED samples
            accuracy = (pseudo_labels_w[ind_keep] == labels_check[ind_keep]).float().mean() * 100
            if not math.isnan(accuracy):
                accuracy_tot += accuracy
                total_acc += 1
            
            if total_acc > 0:
                accuracies.append((accuracy_tot/total_acc).cpu().item())
            else:
                accuracies.append(0.0)

            # Accuracy of ALL samples in batch
            acc_all = (pseudo_labels_w == labels_check).float().mean() * 100
            if not math.isnan(acc_all):
                batch_acc_tot += acc_all
                batch_total_acc += 1
            
            if batch_total_acc > 0:
                batch_accuracies.append((batch_acc_tot/batch_total_acc).cpu().item())
            else:
                batch_accuracies.append(0.0)

                ### Contrastive Learning ###
            if ind_remove.numel() > 0:
                feats_k = model(images_k, cls_only=True)[0]
                f1 = F.normalize(torch.squeeze(feats_con[ind_remove]), dim=1)
                f2 = F.normalize(torch.squeeze(feats_k[ind_remove]), dim=1)
                features = torch.cat([f1.unsqueeze(1), f2.unsqueeze(1)], dim=1)
                loss_contrast = contrastive_criterion(features)
            else:
                loss_contrast = torch.tensor(0.0).cuda()

            ### Inter-domain Mixup ###
            mix_cls_loss = torch.tensor(0.).cuda()
            if ind_keep.numel() > 0:
                # Assuming args.learn.alpha and args.learn.mix_ratio will be available in the config.
                # Using 0.75 as default for alpha if not present.
                alpha = getattr(args.learn, 'alpha', 0.75)
                rho = np.random.beta(alpha, alpha)
                
                # Proxy samples are the reliable samples from the current batch
                inputs_proxy = images_w[ind_keep.squeeze()]
                labels_proxy = pseudo_labels_w[ind_keep.squeeze()]
                
                # Target samples are all samples from the current batch
                inputs_target = images_w
                
                # Resize proxy to match target batch size for mixup
                num_proxy = inputs_proxy.size(0)
                num_target = inputs_target.size(0)
                
                if num_proxy < num_target:
                    ratio = math.ceil(num_target / num_proxy)
                    inputs_proxy = inputs_proxy.repeat(ratio, 1, 1, 1)[:num_target]
                    labels_proxy = labels_proxy.repeat(ratio)[:num_target]
                elif num_proxy > num_target:
                    indices = torch.randperm(num_proxy)[:num_target]
                    inputs_proxy = inputs_proxy[indices]
                    labels_proxy = labels_proxy[indices]

                mix_img = inputs_target * rho + inputs_proxy * (1 - rho)
                
                # It is assumed that the model can handle the mixed image and return logits.
                _, mix_out = model(mix_img, cls_only=True)
                
                sft_label = torch.nn.functional.softmax(outputs_ema, dim=1)
                targets_s = torch.zeros(num_target, num_class).cuda().scatter_(1, labels_proxy.view(-1, 1), 1)
                
                mix_target = sft_label * rho + targets_s * (1 - rho)

                step = i + epoch * len(train_loader)
                eff = step / args.learn.full_progress

                # Assuming args.learn.mix_ratio will be available
                mix_ratio = getattr(args.learn, 'mix_ratio', 1.0)
                mix_cls_loss = eff * KLD(F.softmax(mix_out, dim=1), mix_target) * mix_ratio

            ### ReMixMatch Regression Loss ###
            remix_reg_loss = torch.tensor(0.).cuda()
            reg_ratio = getattr(args.learn, 'reg_ratio', 1.0)
            
            # Only calculate if reg_ratio is > 0 to save compute
            if reg_ratio > 0:
                alpha = getattr(args.learn, 'alpha', 0.75)
                lambda_u = getattr(args.learn, 'lambda_u', 75)
                
                # Simple single-pass mixup (no interleaving, no micro-batching)
                rho = np.random.beta(alpha, alpha)
                targets_u = torch.softmax(outputs_ema, dim=1)
                
                ind_perm = torch.randperm(images_q.size(0)).cuda()
                mixed_input_remix = rho * images_q + (1 - rho) * images_q[ind_perm]
                mixed_target_remix = rho * targets_u + (1 - rho) * targets_u[ind_perm]
                
                # Single forward pass
                with torch.no_grad():
                    _, logits_u = model(mixed_input_remix, cls_only=True)
                
                probs_u = torch.softmax(logits_u, dim=1)
                
                step = i + epoch * len(train_loader)
                rampup_val = linear_rampup(step, args.learn.full_progress)
                
                remix_reg_loss = (torch.mean((probs_u - mixed_target_remix) ** 2) * lambda_u * rampup_val) * reg_ratio


            ### Loss Coefficients & Total Loss ###
            difficulty_score = uncer_th / conf_th
            loss_coef *= (1 - 0.005 * torch.exp(-1 / difficulty_score))
            con_coeff *= np.exp(-0.0001)

            # Define total loss, incorporating the new mix_cls_loss and remix_reg_loss
            loss = con_coeff * loss_contrast + mix_cls_loss + remix_reg_loss + loss_coef * loss_cls

            ## Update the Parameters
            loss_meter.update(loss.item())
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

                    ### Training Statistics ####
            con_coeffs.append(con_coeff) 
            loss_classes.append(loss_cls.item())
            loss_coefs.append(loss_coef.item() if torch.is_tensor(loss_coef) else loss_coef)
            con_losses.append(loss_contrast.item())
            mix_cls_losses.append(mix_cls_loss.item())
            remix_reg_losses.append(remix_reg_loss.item())
            # unsupervised_losses.append(loss_cls_rem.item() if torch.is_tensor(loss_cls_rem) else loss_cls_rem)
            total_losses.append(loss.item()) # New: Track total loss
            uncertainty_thresholds.append(uncer_th.item() if torch.is_tensor(uncer_th) else uncer_th)
            conf_thress.append(conf_th.item() if torch.is_tensor(conf_th) else conf_th)
            sel_Samples.append(len(ind_keep))
            unsel_samples.append(len(ind_remove))

            batch_time.update(time.time() - end)
            end = time.time()

            if i % args.learn.print_freq == 0:
                progress.display(i)
                if is_master(args):
                    logging.info(
                        "Iter %d | loss=%.4f | contrast=%.4f | mix=%.4f | remix=%.4f",
                        i,
                        float(loss.detach()),
                        float(loss_contrast.detach()),
                        float(mix_cls_loss.detach()),
                        float(remix_reg_loss.detach()),
                    )

            if is_master(args) and (ind % 500 == 0):
                # Plot training stats
                current_stats = {
                    'pseudo_label_acc': accuracies,
                    'batch_acc': batch_accuracies,
                    'ce_loss': loss_classes,
                    'con_loss': con_losses,
                    'mix_cls_loss': mix_cls_losses,
                    'remix_reg_loss': remix_reg_losses,
                    'prop_loss': unsupervised_losses,
                    'total_loss': total_losses, # New: Pass total loss
                    'val_acc_mean': acc_classes,
                    'val_acc_per_class': per_class_accs,
                    'conf_th_mean': epoch_conf_th_means,
                    'uncer_th_mean': epoch_uncer_th_means,
                    'conf_th_per_class': epoch_conf_th_per_class,
                    'uncer_th_per_class': epoch_uncer_th_per_class,
                }
                plot_training_stats(current_stats, save_path="training_process.png")

                # Save stats only every 50 iterations to save I/O
                np.savez("training_stats.npz", pseudo_label_acc = accuracies, acc_class= acc_classes, conf = conf_thress, unc = uncertainty_thresholds, labeled_loss_coeff = loss_coefs, con_coeff = con_coeffs, ce_loss = loss_classes, con_loss = con_losses, prop_loss = unsupervised_losses, sel_Samples = sel_Samples , unsel_samples = unsel_samples)

            ind += 1 

            ## Evaluate the model ##
        result = eval_and_label_dataset(val_loader, model, args, per_class_conf_th, per_class_uncer_th)
        
        # Handle return values
        if isinstance(result, tuple):
            acc_per_class, conf_mean, uncer_mean = result
            if is_master(args):
                epoch_conf_th_means.append(conf_mean)
                epoch_uncer_th_means.append(uncer_mean)
                epoch_conf_th_per_class.append(per_class_conf_th.cpu().numpy())
                epoch_uncer_th_per_class.append(per_class_uncer_th.cpu().numpy())
        else:
            acc_per_class = result
        
        # Generate t-SNE visualization every epoch
        # if is_master(args):
        #     tsne_path = os.path.join(args.model_tta.src_log_dir, f"tsne_epoch_{epoch}.png")
        #     visualize_tsne(val_loader, model, args, save_path=tsne_path, max_samples=1500)
        
        model.train()
        acc_classes.append(acc_per_class.mean().item() if torch.is_tensor(acc_per_class.mean()) else acc_per_class.mean())
        per_class_accs.append(acc_per_class)
        
        # Log proxy domain class statistics
        if len(proxy_domain) > 0:
            try:
                # Extract labels from strings "path label"
                proxy_labels = [int(line.split(' ')[-1]) for line in proxy_domain]
                unique_classes, counts = np.unique(proxy_labels, return_counts=True)
                class_counts_str = ", ".join([f"Class {cls}: {count}" for cls, count in zip(unique_classes, counts)])
                logging.info(f"Proxy Domain Class Distribution (Epoch {epoch}): [{class_counts_str}]")
                logging.info(f"Total Proxy Samples: {len(proxy_domain)}")
            except Exception as e:
                logging.warning(f"Could not calculate proxy stats: {e}")

        # Save proxy domain after each epoch
        proxy_file_path = os.path.join(args.model_tta.src_log_dir, f"proxy_domain_epoch_{epoch}.txt")
        with open(proxy_file_path, 'w') as f:
            for line in proxy_domain:
                f.write(f"{line}\n")
        logging.info(f"Saved proxy domain file at {proxy_file_path}")
        proxy_domain.clear() # Clear for next epoch
        
        if is_master(args):
            # Save checkpoint for each epoch (replace existing)
            filename_latest = f"checkpoint_1_latest_{args.data.src_domain}-{args.data.tgt_domain}.pth.tar"
            save_path_latest = os.path.join('./checkpoint/', filename_latest)
            save_checkpoint(model, optimizer, epoch, save_path=save_path_latest)





def KLD(sfm, sft):
    return -torch.mean(torch.sum(sfm.log() * sft, dim=1))


@torch.jit.script
def softmax_entropy(x, x_ema):  # -> torch.Tensor:
    """Entropy of softmax distribution from logits."""
    return -(x_ema.softmax(1) * x.log_softmax(1)).sum(1)

@torch.no_grad()
def calculate_acc(logits, labels):
    preds = logits.argmax(dim=1)
    accuracy = (preds == labels).float().mean() * 100
    return accuracy


def classification_loss(outputs_ema, logits_s, target_labels, targets_preds, args, class_weights= None):
    if args.learn.ce_sup_type == "weak_weak":
        loss_cls = cross_entropy_loss(outputs_ema, target_labels, args, class_weights)
        accuracy = calculate_acc(outputs_ema, target_labels)
    elif args.learn.ce_sup_type == "weak_strong":
        # loss_cls = torch.mean((logits_s - targets_preds)**2)
        loss_cls = cross_entropy_loss(logits_s, target_labels, args, class_weights)
        accuracy = calculate_acc(logits_s, target_labels)
    else:
        raise NotImplementedError(
            f"{args.learn.ce_sup_type} CE supervision type not implemented."
        )
    return loss_cls, accuracy

def propagation_loss(outputs_ema, logits_s, target_labels, targets_preds, args):
    if args.learn.ce_sup_type == "weak_weak":
        loss_cls = cross_entropy_loss(outputs_ema, target_labels, args)
        accuracy = calculate_acc(outputs_ema, target_labels)
    elif args.learn.ce_sup_type == "weak_strong":
        loss_cls = torch.mean((targets_preds-logits_s)**2)
        # loss_cls = cross_entropy_loss(logits_s, target_labels, args)
        accuracy = calculate_acc(logits_s, target_labels)
    else:
        raise NotImplementedError(
            f"{args.learn.ce_sup_type} CE supervision type not implemented."
        )
    return loss_cls, accuracy



def smoothed_cross_entropy(logits, labels, num_classes, epsilon=0):
    log_probs = F.log_softmax(logits, dim=1)
    with torch.no_grad():
        targets = torch.zeros_like(log_probs).scatter_(1, labels.unsqueeze(1), 1)
        targets = (1 - epsilon) * targets + epsilon / num_classes
    loss = (-targets * log_probs).sum(dim=1).mean()

    return loss


def cross_entropy_loss(logits, labels, args, class_weights=None):
    if args.learn.ce_type == "standard":
        return F.cross_entropy(logits, labels, weight = class_weights)
    raise NotImplementedError(f"{args.learn.ce_type} CE loss is not implemented.")


def entropy_minimization(logits):
    if len(logits) == 0:
        return torch.tensor([0.0]).cuda()
    probs = F.softmax(logits, dim=1)
    ents = -(probs * probs.log()).sum(dim=1)

    loss = ents.mean()

    return loss
