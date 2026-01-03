"""
t-SNE Visualization Script for SFDA Models

Usage:
    python visualize_tsne.py \
        --image_root /path/to/images \
        --label_file /path/to/label_file.txt \
        --checkpoint /path/to/checkpoint.pth.tar \
        --num_classes 12 \
        --output tsne_result.png

The label file format should be: image_path label (space-separated, one per line)
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import argparse
import torch
import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from torch.utils.data import DataLoader

from classifier import Classifier
from image_list import ImageList
from utils import get_augmentation


# Class names for common datasets
CLASS_NAMES = {
    12: [  # VISDA-C
        "plane", "bcycl", "bus", "car", "horse",
        "knife", "mcycl", "person", "plant", "sktbd", "train", "truck"
    ],
    126: None,  # DomainNet - too many classes
}


def load_model(checkpoint_path, num_classes, bottleneck_dim=256, device="cuda"):
    """Load model from checkpoint."""
    model = Classifier(num_classes=num_classes, bottleneck_dim=bottleneck_dim)

    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    state_dict = checkpoint.get("state_dict", checkpoint)

    # Handle DDP prefix
    new_state_dict = {}
    for k, v in state_dict.items():
        name = k.replace("module.", "") if k.startswith("module.") else k
        # Handle CSFDA_Model wrapping (src_model prefix)
        name = name.replace("src_model.", "") if name.startswith("src_model.") else name
        new_state_dict[name] = v

    model.load_state_dict(new_state_dict, strict=False)
    model = model.to(device)
    model.eval()

    print(f"Loaded checkpoint from: {checkpoint_path}")
    return model


@torch.no_grad()
def extract_features(model, dataloader, device="cuda", max_samples=None):
    """Extract features and labels from the dataset."""
    features_list = []
    labels_list = []
    predictions_list = []

    total = 0
    for _, imgs, labels, _ in dataloader:
        imgs = imgs.to(device)
        feats, logits = model(imgs, return_feats=True)
        preds = logits.argmax(dim=1)

        features_list.append(feats.cpu())
        labels_list.append(labels)
        predictions_list.append(preds.cpu())

        total += imgs.size(0)
        if max_samples and total >= max_samples:
            break

    features = torch.cat(features_list)
    labels = torch.cat(labels_list)
    predictions = torch.cat(predictions_list)

    if max_samples:
        features = features[:max_samples]
        labels = labels[:max_samples]
        predictions = predictions[:max_samples]

    return features.numpy(), labels.numpy(), predictions.numpy()


def plot_tsne(features_2d, labels, predictions, num_classes, output_path, class_names=None):
    """Create t-SNE visualization with confusion analysis."""

    # Create figure with subplots
    fig, axes = plt.subplots(1, 2, figsize=(20, 8))

    # Color palette
    if num_classes <= 12:
        colors = plt.cm.tab10(np.linspace(0, 1, num_classes))
    else:
        colors = plt.cm.tab20(np.linspace(0, 1, min(20, num_classes)))
        if num_classes > 20:
            colors = plt.cm.nipy_spectral(np.linspace(0, 1, num_classes))

    # --- Left plot: colored by TRUE labels ---
    ax1 = axes[0]
    for c in range(num_classes):
        mask = labels == c
        if mask.sum() > 0:
            label = class_names[c] if class_names else f"Class {c}"
            ax1.scatter(
                features_2d[mask, 0],
                features_2d[mask, 1],
                c=[colors[c % len(colors)]],
                label=label,
                alpha=0.6,
                s=15
            )

    ax1.set_title("t-SNE colored by TRUE labels", fontsize=14)
    ax1.set_xlabel("t-SNE dim 1")
    ax1.set_ylabel("t-SNE dim 2")
    if num_classes <= 20:
        ax1.legend(loc='center left', bbox_to_anchor=(1, 0.5), fontsize=8)

    # --- Right plot: show misclassifications ---
    ax2 = axes[1]
    correct_mask = labels == predictions
    wrong_mask = ~correct_mask

    # Plot correct predictions (gray, smaller)
    ax2.scatter(
        features_2d[correct_mask, 0],
        features_2d[correct_mask, 1],
        c='lightgray',
        alpha=0.3,
        s=10,
        label=f'Correct ({correct_mask.sum()})'
    )

    # Plot wrong predictions colored by TRUE label
    for c in range(num_classes):
        mask = (labels == c) & wrong_mask
        if mask.sum() > 0:
            label = class_names[c] if class_names else f"Class {c}"
            ax2.scatter(
                features_2d[mask, 0],
                features_2d[mask, 1],
                c=[colors[c % len(colors)]],
                label=f"{label} misclassified ({mask.sum()})",
                alpha=0.8,
                s=25,
                edgecolors='black',
                linewidths=0.5
            )

    ax2.set_title("Misclassified samples (colored by TRUE label)", fontsize=14)
    ax2.set_xlabel("t-SNE dim 1")
    ax2.set_ylabel("t-SNE dim 2")
    if num_classes <= 20:
        ax2.legend(loc='center left', bbox_to_anchor=(1, 0.5), fontsize=7)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved t-SNE plot to: {output_path}")


def print_confusion_summary(labels, predictions, num_classes, class_names=None):
    """Print summary of what classes are confused with what."""
    print("\n" + "="*60)
    print("CONFUSION SUMMARY")
    print("="*60)

    accuracy = (labels == predictions).mean() * 100
    print(f"\nOverall Accuracy: {accuracy:.2f}%\n")

    # Per-class accuracy and confusion
    print(f"{'Class':<15} {'Acc%':<8} {'Most confused with'}")
    print("-"*60)

    for c in range(num_classes):
        mask = labels == c
        if mask.sum() == 0:
            continue

        class_name = class_names[c] if class_names else f"Class {c}"
        class_acc = (predictions[mask] == c).mean() * 100

        # Find what this class is confused with
        wrong_preds = predictions[mask & (predictions != c)]
        if len(wrong_preds) > 0:
            unique, counts = np.unique(wrong_preds, return_counts=True)
            sorted_idx = np.argsort(-counts)[:3]  # Top 3 confusions

            confused_with = []
            for idx in sorted_idx:
                conf_class = unique[idx]
                conf_name = class_names[conf_class] if class_names else f"Class {conf_class}"
                confused_with.append(f"{conf_name}({counts[idx]})")

            conf_str = ", ".join(confused_with)
        else:
            conf_str = "-"

        print(f"{class_name:<15} {class_acc:<8.1f} {conf_str}")

    print("="*60 + "\n")


def main():
    parser = argparse.ArgumentParser(description="t-SNE Visualization for SFDA Models")
    parser.add_argument("--image_root", type=str, required=True,
                        help="Root directory containing images")
    parser.add_argument("--label_file", type=str, required=True,
                        help="Path to label file (format: image_path label)")
    parser.add_argument("--checkpoint", type=str, required=True,
                        help="Path to model checkpoint")
    parser.add_argument("--num_classes", type=int, default=12,
                        help="Number of classes (default: 12 for VISDA-C)")
    parser.add_argument("--bottleneck_dim", type=int, default=256,
                        help="Bottleneck dimension (default: 256)")
    parser.add_argument("--output", type=str, default="tsne_visualization.png",
                        help="Output path for the plot")
    parser.add_argument("--max_samples", type=int, default=3000,
                        help="Max samples for t-SNE (default: 3000)")
    parser.add_argument("--batch_size", type=int, default=128,
                        help="Batch size for inference (default: 128)")
    parser.add_argument("--perplexity", type=int, default=30,
                        help="t-SNE perplexity (default: 30)")
    parser.add_argument("--device", type=str, default="cuda",
                        help="Device to use (default: cuda)")

    args = parser.parse_args()

    # Check device
    if args.device == "cuda" and not torch.cuda.is_available():
        print("CUDA not available, using CPU")
        args.device = "cpu"

    # Load model
    print("Loading model...")
    model = load_model(
        args.checkpoint,
        args.num_classes,
        args.bottleneck_dim,
        args.device
    )

    # Create dataset and dataloader
    print("Loading dataset...")
    transform = get_augmentation("test")
    dataset = ImageList(
        image_root=args.image_root,
        label_file=args.label_file,
        transform=transform
    )
    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=4
    )
    print(f"Dataset size: {len(dataset)}")

    # Extract features
    print(f"Extracting features (max {args.max_samples} samples)...")
    features, labels, predictions = extract_features(
        model, dataloader, args.device, args.max_samples
    )
    print(f"Extracted features shape: {features.shape}")

    # Run t-SNE
    print(f"Running t-SNE (perplexity={args.perplexity})...")
    tsne = TSNE(
        n_components=2,
        random_state=42,
        perplexity=args.perplexity,
        n_iter=1000,
        verbose=1
    )
    features_2d = tsne.fit_transform(features)

    # Get class names
    class_names = CLASS_NAMES.get(args.num_classes)

    # Print confusion summary
    print_confusion_summary(labels, predictions, args.num_classes, class_names)

    # Plot and save
    print("Creating visualization...")
    plot_tsne(features_2d, labels, predictions, args.num_classes, args.output, class_names)

    print("Done!")


if __name__ == "__main__":
    main()
