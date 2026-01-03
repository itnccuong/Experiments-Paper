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
def extract_features(model, dataloader, device="cuda"):
    """Extract features and labels from the dataset."""
    features_list = []
    labels_list = []

    for _, imgs, labels, _ in dataloader:
        imgs = imgs.to(device)
        feats, _ = model(imgs, return_feats=True)

        features_list.append(feats.cpu())
        labels_list.append(labels)

    features = torch.cat(features_list)
    labels = torch.cat(labels_list)

    return features.numpy(), labels.numpy()


def sample_per_class(features, labels, max_per_class=100):
    """Sample up to max_per_class samples from each class."""
    unique_classes = np.unique(labels)
    selected_indices = []

    for c in unique_classes:
        class_indices = np.where(labels == c)[0]
        if len(class_indices) > max_per_class:
            # Randomly sample
            np.random.seed(42)
            class_indices = np.random.choice(class_indices, max_per_class, replace=False)
        selected_indices.extend(class_indices)

    selected_indices = np.array(selected_indices)
    return features[selected_indices], labels[selected_indices]


def plot_tsne(features_2d, labels, num_classes, output_path, class_names=None):
    """Create simple t-SNE visualization colored by true class."""

    fig, ax = plt.subplots(figsize=(12, 10))

    # Color palette
    if num_classes <= 12:
        colors = plt.cm.tab10(np.linspace(0, 1, num_classes))
    else:
        colors = plt.cm.tab20(np.linspace(0, 1, min(20, num_classes)))
        if num_classes > 20:
            colors = plt.cm.nipy_spectral(np.linspace(0, 1, num_classes))

    for c in range(num_classes):
        mask = labels == c
        if mask.sum() > 0:
            label = class_names[c] if class_names else f"Class {c}"
            ax.scatter(
                features_2d[mask, 0],
                features_2d[mask, 1],
                c=[colors[c % len(colors)]],
                label=f"{label} ({mask.sum()})",
                alpha=0.7,
                s=20
            )

    ax.set_title("t-SNE Visualization", fontsize=14)
    ax.set_xlabel("t-SNE dim 1")
    ax.set_ylabel("t-SNE dim 2")
    if num_classes <= 20:
        ax.legend(loc='center left', bbox_to_anchor=(1, 0.5), fontsize=9)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved t-SNE plot to: {output_path}")


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
    parser.add_argument("--samples_per_class", type=int, default=100,
                        help="Max samples per class for t-SNE (default: 100)")
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
    print("Extracting features...")
    features, labels = extract_features(model, dataloader, args.device)
    print(f"Extracted features shape: {features.shape}")

    # Sample per class
    print(f"Sampling up to {args.samples_per_class} samples per class...")
    features, labels = sample_per_class(features, labels, args.samples_per_class)
    print(f"After sampling: {features.shape[0]} samples")

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

    # Plot and save
    print("Creating visualization...")
    plot_tsne(features_2d, labels, args.num_classes, args.output, class_names)

    print("Done!")


if __name__ == "__main__":
    main()
