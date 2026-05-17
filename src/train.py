"""
train.py  –  Training, validation, and evaluation script.

Usage
-----
python src/train.py --config config.yaml
python src/train.py --config config.yaml --eval_only --checkpoint checkpoints/best.pt
"""

import os
import argparse
import yaml
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import CosineAnnealingLR
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from model import MultimodalStoryModel, MultimodalLoss
from utils import (set_seed, get_device, load_config,
                   save_checkpoint, load_checkpoint,
                   compute_bleu, compute_image_mse,
                   plot_story_sequence, plot_attention_heatmap)


# ──────────────────────────────────────────────────────────────
# Dataset  (StoryReasoning via HuggingFace)
# ──────────────────────────────────────────────────────────────

def build_dataloaders(cfg: dict):
    """
    Load the StoryReasoning dataset from HuggingFace and return
    train / val DataLoaders.

    NOTE: We use a lightweight wrapper class defined below.
          Adjust field names if the HF dataset schema changes.
    """
    from datasets import load_dataset
    from torchvision import transforms
    from PIL import Image
    import numpy as np

    ds_cfg = cfg["dataset"]
    image_size = ds_cfg["image_size"]
    max_text_len = ds_cfg["max_text_len"]
    K = ds_cfg["sequence_length"]

    train_tf = transforms.Compose([
        transforms.Resize((image_size + 32, image_size + 32)),
        transforms.RandomCrop(image_size),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])
    val_tf = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])

    print(f"[Data] Loading StoryReasoning dataset from HuggingFace …")
    raw = load_dataset(ds_cfg["hf_repo"])

    class StoryDataset(torch.utils.data.Dataset):
        def __init__(self, hf_split, transform, K, max_text_len, vocab):
            self.data      = hf_split
            self.transform = transform
            self.K         = K
            self.max_len   = max_text_len
            self.vocab     = vocab

        def tokenize(self, text: str) -> torch.Tensor:
            """Simple whitespace tokeniser mapped to integer ids."""
            tokens = text.lower().split()[:self.max_len]
            ids = [self.vocab.get(w, 1) for w in tokens]  # 1 = <UNK>
            pad = [0] * (self.max_len - len(ids))         # 0 = <PAD>
            return torch.tensor(ids + pad, dtype=torch.long)

        def __len__(self):
            # Each sample has K+1 frames; we need at least K+1 frames per story
            return len(self.data)

        def __getitem__(self, idx):
            sample = self.data[idx]

            # ── Adapt these field names to the actual HF schema ──
            # Expected fields: "images" (list of PIL/path), "captions" (list of str)
            images_raw  = sample.get("images",   sample.get("frames", []))
            captions_raw = sample.get("captions", sample.get("texts",  []))

            # Ensure we have at least K+1 items
            n = min(len(images_raw), len(captions_raw))
            if n < self.K + 1:
                # Repeat last item to pad (rare edge case)
                images_raw   = (images_raw * (self.K + 2))[:self.K + 1]
                captions_raw = (captions_raw * (self.K + 2))[:self.K + 1]

            context_imgs  = []
            context_texts = []
            for k in range(self.K):
                img = images_raw[k]
                if not isinstance(img, Image.Image):
                    img = Image.open(img).convert("RGB")
                context_imgs.append(self.transform(img))
                context_texts.append(self.tokenize(captions_raw[k]))

            # Target (K+1)
            tgt_img = images_raw[self.K]
            if not isinstance(tgt_img, Image.Image):
                tgt_img = Image.open(tgt_img).convert("RGB")
            tgt_img = transforms.Compose([
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406],
                                     [0.229, 0.224, 0.225]),
            ])(tgt_img)
            tgt_text = self.tokenize(captions_raw[self.K])

            return {
                "images":        torch.stack(context_imgs),   # (K, C, H, W)
                "tokens":        torch.stack(context_texts),  # (K, T)
                "target_image":  tgt_img,                     # (C, H, W)
                "target_tokens": tgt_text,                    # (T,)
                "raw_caption":   captions_raw[self.K],
            }

    # Build vocabulary from training captions
    print("[Data] Building vocabulary …")
    vocab = {"<PAD>": 0, "<UNK>": 1, "<SOS>": 2, "<EOS>": 3}
    vocab_size = cfg["text_encoder"]["vocab_size"]
    from collections import Counter
    counter = Counter()
    for sample in raw[ds_cfg["train_split"]]:
        caps = sample.get("captions", sample.get("texts", []))
        for cap in caps:
            counter.update(cap.lower().split())
    for word, _ in counter.most_common(vocab_size - len(vocab)):
        vocab[word] = len(vocab)

    train_ds = StoryDataset(raw[ds_cfg["train_split"]],   train_tf, K, max_text_len, vocab)
    val_ds   = StoryDataset(raw[ds_cfg["val_split"]],     val_tf,   K, max_text_len, vocab)

    train_loader = DataLoader(train_ds, batch_size=cfg["training"]["batch_size"],
                              shuffle=True,  num_workers=4, pin_memory=True)
    val_loader   = DataLoader(val_ds,   batch_size=cfg["training"]["batch_size"],
                              shuffle=False, num_workers=4, pin_memory=True)

    print(f"[Data] Train: {len(train_ds)} | Val: {len(val_ds)}")
    return train_loader, val_loader, vocab


# ──────────────────────────────────────────────────────────────
# Training loop
# ──────────────────────────────────────────────────────────────

def train_one_epoch(model, loader, optimizer, criterion, device, epoch, cfg):
    model.train()
    total_loss = img_loss_sum = txt_loss_sum = 0.0
    for step, batch in enumerate(loader):
        images        = batch["images"].to(device)
        tokens        = batch["tokens"].to(device)
        target_image  = batch["target_image"].to(device)
        target_tokens = batch["target_tokens"].to(device)

        optimizer.zero_grad()
        out = model(images, tokens, target_tokens, teacher_forcing_ratio=0.5)
        losses = criterion(out["pred_image"], target_image,
                           out["pred_text_logits"], target_tokens)
        losses["total"].backward()
        nn.utils.clip_grad_norm_(model.parameters(),
                                  cfg["training"]["grad_clip"])
        optimizer.step()

        total_loss   += losses["total"].item()
        img_loss_sum += losses["image"].item()
        txt_loss_sum += losses["text"].item()

        if step % 50 == 0:
            print(f"  Epoch {epoch} | Step {step}/{len(loader)} "
                  f"| Loss {losses['total'].item():.4f} "
                  f"(img {losses['image'].item():.4f}, "
                  f"txt {losses['text'].item():.4f})")

    n = len(loader)
    return total_loss / n, img_loss_sum / n, txt_loss_sum / n


@torch.no_grad()
def validate(model, loader, criterion, device):
    model.eval()
    total_loss = img_mse_sum = 0.0
    references, hypotheses = [], []
    id2word = None  # filled lazily if vocab is passed

    for batch in loader:
        images        = batch["images"].to(device)
        tokens        = batch["tokens"].to(device)
        target_image  = batch["target_image"].to(device)
        target_tokens = batch["target_tokens"].to(device)

        out = model(images, tokens, target_tokens, teacher_forcing_ratio=0.0)
        losses = criterion(out["pred_image"], target_image,
                           out["pred_text_logits"], target_tokens)
        total_loss   += losses["total"].item()
        img_mse_sum  += losses["image"].item()

        # Collect captions for BLEU (raw strings)
        references.extend(batch["raw_caption"])

    avg_loss = total_loss / len(loader)
    avg_mse  = img_mse_sum / len(loader)
    return avg_loss, avg_mse


# ──────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config",     default="config.yaml")
    parser.add_argument("--eval_only",  action="store_true")
    parser.add_argument("--checkpoint", default=None)
    args = parser.parse_args()

    cfg    = load_config(args.config)
    device = get_device()
    set_seed(cfg["training"]["seed"])

    print(f"[Setup] Device: {device}")

    # Data
    train_loader, val_loader, vocab = build_dataloaders(cfg)

    # Model
    model = MultimodalStoryModel(cfg).to(device)
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"[Model] Trainable parameters: {total_params:,}")

    criterion = MultimodalLoss(
        image_weight=cfg["training"]["image_loss_weight"],
        text_weight=cfg["training"]["text_loss_weight"],
    )
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=cfg["training"]["learning_rate"],
        weight_decay=cfg["training"]["weight_decay"],
    )
    scheduler = CosineAnnealingLR(
        optimizer, T_max=cfg["training"]["epochs"], eta_min=1e-6)

    start_epoch, best_val_loss = 0, float("inf")
    if args.checkpoint:
        start_epoch, best_val_loss = load_checkpoint(
            args.checkpoint, model, optimizer, device)

    if args.eval_only:
        val_loss, val_mse = validate(model, val_loader, criterion, device)
        print(f"[Eval] Val loss: {val_loss:.4f} | Image MSE: {val_mse:.6f}")
        return

    # ── Training loop ──
    os.makedirs(cfg["paths"]["checkpoint_dir"], exist_ok=True)
    os.makedirs(cfg["paths"]["results_dir"],    exist_ok=True)

    train_losses, val_losses = [], []

    for epoch in range(start_epoch + 1, cfg["training"]["epochs"] + 1):
        tr_loss, tr_img, tr_txt = train_one_epoch(
            model, train_loader, optimizer, criterion, device, epoch, cfg)
        val_loss, val_mse = validate(model, val_loader, criterion, device)
        scheduler.step()

        train_losses.append(tr_loss)
        val_losses.append(val_loss)

        print(f"Epoch {epoch:03d} | Train {tr_loss:.4f} | Val {val_loss:.4f} "
              f"| Val MSE {val_mse:.6f}")

        # Save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            save_checkpoint({
                "epoch": epoch,
                "model_state":     model.state_dict(),
                "optimizer_state": optimizer.state_dict(),
                "best_val_loss":   best_val_loss,
                "cfg":             cfg,
            }, os.path.join(cfg["paths"]["checkpoint_dir"], "best.pt"))

        # Periodic checkpoint
        if epoch % 5 == 0:
            save_checkpoint({
                "epoch": epoch,
                "model_state":     model.state_dict(),
                "optimizer_state": optimizer.state_dict(),
                "best_val_loss":   best_val_loss,
            }, os.path.join(cfg["paths"]["checkpoint_dir"], f"epoch_{epoch:03d}.pt"))

    # ── Save loss curve ──
    plt.figure(figsize=(8, 4))
    plt.plot(train_losses, label="Train")
    plt.plot(val_losses,   label="Val")
    plt.xlabel("Epoch"); plt.ylabel("Loss")
    plt.title("Training / Validation Loss")
    plt.legend(); plt.tight_layout()
    plt.savefig(os.path.join(cfg["paths"]["results_dir"], "loss_curve.png"), dpi=150)
    print("[Done] Loss curve saved.")


if __name__ == "__main__":
    main()
