#!/bin/bash

# Example usage for VISDA-C validation set
# Modify these paths according to your setup

# VISDA-C example
python visualize_tsne.py \
    --image_root /path/to/VISDA-C/validation \
    --label_file /path/to/VISDA-C/validation_list.txt \
    --checkpoint ../checkpoint/best_train_1.pth.tar \
    --num_classes 12 \
    --output tsne_visda.png \
    --max_samples 3000 \
    --perplexity 30

# DomainNet example (uncomment to use)
# python visualize_tsne.py \
#     --image_root /path/to/domainnet/clipart \
#     --label_file /path/to/domainnet/clipart_list.txt \
#     --checkpoint ../checkpoint/best_real_1.pth.tar \
#     --num_classes 126 \
#     --output tsne_domainnet.png \
#     --max_samples 2000 \
#     --perplexity 50
