"""
explainability.py  –  XAI techniques for the Multimodal Story Model.

Techniques implemented
──────────────────────
1. Attention Rollout  – aggregates temporal attention weights across
   layers to show which context frames the model focused on.
2. Grad-CAM           – highlights image regions most relevant to
   the predicted visual continuation.
3. Token Saliency     – gradient w.r.t. text embedding to identify
   which words most influenced the prediction.
"""

import os
import numpy as np
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from torchvision.transforms.functional import to_pil_image


# ══════════════════════════════════════════════════════════════
# 1. Attention Rollout  (primary XAI, from cfg["explainability"])
# ══════════════════════════════════════════════════════════════

def attention_rollout(attn_weights: torch.Tensor) -> np.ndarray:
    """
    Compute attention rollout from a single temporal attention matrix.

    Args:
        attn_weights : (B, num_heads, K, K)  or  (B, K, K)
    Returns:
        rollout : (B, K) – importance score per context frame
    """
    if attn_weights.dim() == 4:
        # Average over heads
        attn = attn_weights.mean(dim=1)          # (B, K, K)
    else:
        attn = attn_weights                      # (B, K, K)

    # Add identity residual (standard rollout recipe)
    B, K, _ = attn.shape
    eye = torch.eye(K, device=attn.device).unsqueeze(0).expand(B, -1, -1)
    attn = 0.5 * attn + 0.5 * eye
    attn = attn / attn.sum(dim=-1, keepdim=True)

    # Rollout = product through "layers" (here just one layer, so = attn itself)
    rollout = attn.mean(dim=1)                   # (B, K)
    rollout = (rollout - rollout.min()) / (rollout.max() - rollout.min() + 1e-8)
    return rollout.detach().cpu().numpy()


def plot_attention_rollout(rollout: np.ndarray, save_path: str = None):
    """
    Bar plot of per-frame attention importance.

    Args:
        rollout   : (B, K) numpy array
        save_path : optional file path
    """
    B, K = rollout.shape
    fig, axes = plt.subplots(1, min(B, 4), figsize=(4 * min(B, 4), 3),
                              sharey=True)
    if B == 1:
        axes = [axes]
    for i, ax in enumerate(axes[:B]):
        ax.bar(range(K), rollout[i], color="steelblue")
        ax.set_xticks(range(K))
        ax.set_xticklabels([f"Frame {k+1}" for k in range(K)], fontsize=8)
        ax.set_title(f"Sample {i+1}", fontsize=9)
        ax.set_ylabel("Attention score")
    fig.suptitle("Attention Rollout – Frame Importance", fontsize=11,
                 fontweight="bold")
    plt.tight_layout()
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"[XAI] Attention rollout saved → {save_path}")
    plt.show()


# ══════════════════════════════════════════════════════════════
# 2. Grad-CAM on the last ResNet conv layer
# ══════════════════════════════════════════════════════════════

class GradCAM:
    """
    Grad-CAM applied to the VisualEncoder's last convolutional layer.

    Usage
    -----
    cam = GradCAM(model.visual_encoder)
    heatmap = cam(image_tensor)          # (H, W) numpy array
    cam.remove_hooks()
    """

    def __init__(self, visual_encoder):
        self.encoder = visual_encoder
        self.gradients = None
        self.activations = None

        # Hook into the last conv layer of ResNet backbone
        # backbone[-2] is layer4 (the last residual block group)
        target_layer = list(visual_encoder.backbone.children())[-2]
        self._fwd_hook = target_layer.register_forward_hook(self._save_activation)
        self._bwd_hook = target_layer.register_full_backward_hook(self._save_gradient)

    def _save_activation(self, module, inp, out):
        self.activations = out.detach()

    def _save_gradient(self, module, grad_in, grad_out):
        self.gradients = grad_out[0].detach()

    def __call__(self, image: torch.Tensor) -> np.ndarray:
        """
        Args:
            image : (1, C, H, W) – single image tensor with grad enabled
        Returns:
            heatmap : (H, W) numpy array in [0, 1]
        """
        image = image.requires_grad_(True)
        feat = self.encoder(image)               # triggers hooks
        score = feat.mean()                      # scalar proxy
        self.encoder.zero_grad()
        score.backward()

        # Global average pool of gradients
        weights = self.gradients.mean(dim=[2, 3], keepdim=True)  # (1, C, 1, 1)
        cam = (weights * self.activations).sum(dim=1, keepdim=True)
        cam = F.relu(cam)

        # Resize to input spatial size
        H, W = image.shape[2], image.shape[3]
        cam = F.interpolate(cam, size=(H, W), mode="bilinear",
                            align_corners=False)
        cam = cam.squeeze().cpu().numpy()
        cam = (cam - cam.min()) / (cam.max() - cam.min() + 1e-8)
        return cam

    def remove_hooks(self):
        self._fwd_hook.remove()
        self._bwd_hook.remove()


def plot_grad_cam(image: torch.Tensor, heatmap: np.ndarray,
                  title: str = "Grad-CAM", save_path: str = None):
    """
    Overlay Grad-CAM heatmap on the original image.

    Args:
        image    : (C, H, W) normalised tensor
        heatmap  : (H, W) numpy array
        save_path: optional save location
    """
    mean = np.array([0.485, 0.456, 0.406])
    std  = np.array([0.229, 0.224, 0.225])
    img_np = image.cpu().permute(1, 2, 0).numpy()
    img_np = (img_np * std + mean).clip(0, 1)

    heatmap_colored = cm.jet(heatmap)[..., :3]
    overlay = 0.5 * img_np + 0.5 * heatmap_colored

    fig, axes = plt.subplots(1, 3, figsize=(12, 4))
    axes[0].imshow(img_np);           axes[0].set_title("Original")
    axes[1].imshow(heatmap, cmap="jet"); axes[1].set_title("Grad-CAM Heatmap")
    axes[2].imshow(overlay);          axes[2].set_title("Overlay")
    for ax in axes:
        ax.axis("off")
    fig.suptitle(title, fontsize=12, fontweight="bold")
    plt.tight_layout()
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"[XAI] Grad-CAM saved → {save_path}")
    plt.show()


# ══════════════════════════════════════════════════════════════
# 3. Token Saliency (text)
# ══════════════════════════════════════════════════════════════

def token_saliency(model, images: torch.Tensor, tokens: torch.Tensor,
                   frame_idx: int = 0, vocab_inv: dict = None):
    """
    Compute L2 gradient saliency of text tokens for a given frame.
    Model must be in train() mode for LSTM backward to work on CUDA.
    """
    # MUST be train() mode — cudnn RNN backward fails in eval() on CUDA
    model.train()

    emb_module = model.text_encoder.embedding
    tok = tokens[:, frame_idx]                   # (1, T)

    # Register hook to capture embedding gradients
    captured_grad = {}
    def save_grad(grad):
        captured_grad['grad'] = grad

    embeds = emb_module(tok)                     # (1, T, E)
    embeds.retain_grad()
    handle = embeds.register_hook(save_grad)

    # Full forward pass
    out = model(images, tokens)
    score = out["pred_text_logits"].mean()
    model.zero_grad()
    score.backward()
    handle.remove()

    # Back to eval
    model.eval()

    grad = captured_grad.get('grad', embeds.grad)  # (1, T, E)
    if grad is None:
        # Fallback: uniform saliency
        saliency = np.ones(tokens.shape[-1])
    else:
        saliency = grad.norm(dim=-1).squeeze(0).detach().cpu().numpy()
        saliency = (saliency - saliency.min()) / (saliency.max() - saliency.min() + 1e-8)

    if vocab_inv:
        words = [vocab_inv.get(int(t), "<UNK>") for t in tok[0]]
    else:
        words = [str(int(t)) for t in tok[0]]

    return saliency, words


def plot_token_saliency(saliency: np.ndarray, words: list,
                        title: str = "Token Saliency",
                        save_path: str = None):
    fig, ax = plt.subplots(figsize=(max(8, len(words) * 0.5), 3))
    colors = cm.Reds(saliency)
    bars = ax.bar(range(len(words)), saliency, color=colors)
    ax.set_xticks(range(len(words)))
    ax.set_xticklabels(words, rotation=45, ha="right", fontsize=8)
    ax.set_ylabel("Saliency")
    ax.set_title(title, fontsize=11, fontweight="bold")
    plt.tight_layout()
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"[XAI] Token saliency saved → {save_path}")
    plt.show()


# ══════════════════════════════════════════════════════════════
# 4. Run all XAI methods on a batch sample
# ══════════════════════════════════════════════════════════════

def run_explainability(model, batch: dict, device: torch.device,
                       results_dir: str = "results/explainability",
                       vocab_inv: dict = None):
    """
    Convenience wrapper: run all three XAI methods on the first
    sample of a batch and save figures.
    """
    model.eval()
    images        = batch["images"][:1].to(device)   # (1, K, C, H, W)
    tokens        = batch["tokens"][:1].to(device)   # (1, K, T)
    target_tokens = batch["target_tokens"][:1].to(device)

    with torch.no_grad():
        out = model(images, tokens, target_tokens, teacher_forcing_ratio=0.0)
        attn = out["attn_weights"]                   # (1, heads, K, K)

    # 1. Attention Rollout
    rollout = attention_rollout(attn)
    plot_attention_rollout(
        rollout,
        save_path=os.path.join(results_dir, "attention_rollout.png"))

    # 2. Grad-CAM on first context frame
    gcam = GradCAM(model.visual_encoder)
    single_img = images[0, 0].unsqueeze(0)           # (1, C, H, W)
    heatmap = gcam(single_img)
    gcam.remove_hooks()
    plot_grad_cam(
        images[0, 0], heatmap,
        title="Grad-CAM – Context Frame 1",
        save_path=os.path.join(results_dir, "grad_cam_frame1.png"))

    # 3. Token Saliency
    saliency, words = token_saliency(
        model, images, tokens, frame_idx=0, vocab_inv=vocab_inv)
    plot_token_saliency(
        saliency, words,
        title="Token Saliency – Context Frame 1",
        save_path=os.path.join(results_dir, "token_saliency.png"))

    print("[XAI] All explainability figures saved.")
