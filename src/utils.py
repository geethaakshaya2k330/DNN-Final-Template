"""
utils.py  –  Shared utilities for the multimodal story-reasoning project.
"""

import os
import random
import yaml
import numpy as np
import torch
import matplotlib.pyplot as plt
from torchvision import transforms


# ──────────────────────────────────────────────────────────────
# Config loader
# ──────────────────────────────────────────────────────────────

def load_config(path: str = "config.yaml") -> dict:
    with open(path, "r") as f:
        return yaml.safe_load(f)


# ──────────────────────────────────────────────────────────────
# Reproducibility
# ──────────────────────────────────────────────────────────────

def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# ──────────────────────────────────────────────────────────────
# Device helper
# ──────────────────────────────────────────────────────────────

def get_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


# ──────────────────────────────────────────────────────────────
# Standard image transforms
# ──────────────────────────────────────────────────────────────

def get_image_transforms(image_size: int = 224, train: bool = True):
    if train:
        return transforms.Compose([
            transforms.Resize((image_size + 32, image_size + 32)),
            transforms.RandomCrop(image_size),
            transforms.RandomHorizontalFlip(),
            transforms.ColorJitter(0.2, 0.2, 0.2, 0.1),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225]),
        ])
    return transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
    ])


# ──────────────────────────────────────────────────────────────
# Checkpoint helpers
# ──────────────────────────────────────────────────────────────

def save_checkpoint(state: dict, path: str):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.save(state, path)
    print(f"[Checkpoint] Saved → {path}")


def load_checkpoint(path: str, model, optimizer=None, device=None):
    device = device or get_device()
    ckpt = torch.load(path, map_location=device)
    model.load_state_dict(ckpt["model_state"])
    if optimizer and "optimizer_state" in ckpt:
        optimizer.load_state_dict(ckpt["optimizer_state"])
    print(f"[Checkpoint] Loaded ← {path}  (epoch {ckpt.get('epoch', '?')})")
    return ckpt.get("epoch", 0), ckpt.get("best_val_loss", float("inf"))


# ──────────────────────────────────────────────────────────────
# Visualisation helpers
# ──────────────────────────────────────────────────────────────

def denormalise(tensor: torch.Tensor) -> np.ndarray:
    """Convert a normalised image tensor to a displayable numpy array."""
    mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
    std  = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
    img = tensor.cpu() * std + mean
    img = img.clamp(0, 1).permute(1, 2, 0).numpy()
    return img


def plot_story_sequence(images, captions, pred_image=None, pred_caption="",
                        save_path=None):
    """
    Plot a story sequence (K context frames + optional prediction).
    Args:
        images   : list of (C,H,W) tensors (normalised)
        captions : list of strings
        pred_image   : optional (C,H,W) tensor for predicted frame
        pred_caption : predicted caption string
        save_path    : if given, save figure to this path
    """
    n = len(images) + (1 if pred_image is not None else 0)
    fig, axes = plt.subplots(1, n, figsize=(4 * n, 5))
    if n == 1:
        axes = [axes]

    for i, (img, cap) in enumerate(zip(images, captions)):
        axes[i].imshow(denormalise(img))
        axes[i].set_title(f"Frame {i+1}", fontsize=10, fontweight="bold")
        axes[i].set_xlabel(cap, fontsize=7, wrap=True)
        axes[i].axis("off")

    if pred_image is not None:
        axes[-1].imshow(denormalise(pred_image))
        axes[-1].set_title("Prediction (K+1)", fontsize=10,
                           fontweight="bold", color="green")
        axes[-1].set_xlabel(pred_caption, fontsize=7, wrap=True)
        axes[-1].axis("off")

    plt.tight_layout()
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.show()


def plot_attention_heatmap(attention_weights: torch.Tensor,
                           labels=None, title="Attention Weights",
                           save_path=None):
    """
    Visualise a (seq_len, seq_len) or (1, seq_len) attention matrix.
    """
    weights = attention_weights.detach().cpu().numpy()
    if weights.ndim == 1:
        weights = weights[np.newaxis, :]

    fig, ax = plt.subplots(figsize=(8, max(2, weights.shape[0])))
    im = ax.imshow(weights, cmap="viridis", aspect="auto")
    plt.colorbar(im, ax=ax)

    if labels:
        ax.set_xticks(range(len(labels)))
        ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=8)

    ax.set_title(title, fontsize=12)
    plt.tight_layout()
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.show()


# ──────────────────────────────────────────────────────────────
# Metric helpers
# ──────────────────────────────────────────────────────────────

def compute_bleu(references: list, hypotheses: list) -> float:
    """Compute corpus-level BLEU-4 using sacrebleu."""
    try:
        from sacrebleu.metrics import BLEU
        bleu = BLEU()
        result = bleu.corpus_score(hypotheses, [references])
        return result.score
    except ImportError:
        print("[Warning] sacrebleu not installed – returning 0.0 for BLEU.")
        return 0.0


def compute_image_mse(pred: torch.Tensor, target: torch.Tensor) -> float:
    return torch.nn.functional.mse_loss(pred, target).item()
