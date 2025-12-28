# Experiments-Paper

target_csfda2.py: celoss + mix_cls_loss + contrast learning
target_csfda.py: celoss + mix_cls_loss + contrast learning + $\tau$ for each class.
target_csfda3.py: above + (squeeze -> flatten) + remix_reg_loss. (batch to 32)
target_csfda4.py: add warmup: 5 epochs.