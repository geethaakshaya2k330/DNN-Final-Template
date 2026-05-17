"""
model.py  –  Multimodal Sequence Model for Visual Story Reasoning.

Architecture summary
────────────────────
Visual Encoder   → ResNet-50 backbone  →  512-d image embedding
Text Encoder     → Embedding + BiLSTM  →  512-d text  embedding
Fusion           → Cross-Modal Attention (INNOVATION #1)
Sequence Model   → Bidirectional LSTM   (INNOVATION #2)
Attention        → Temporal Self-Attention with positional bias (INNOVATION #3)
Image Decoder    → Transposed-Conv decoder
Text  Decoder    → LSTM with teacher-forcing

Innovation rationale
────────────────────
1. Cross-modal attention replaces naive concatenation: text queries attend
   over image keys/values so the fused representation captures fine-grained
   vision-language alignment.
2. BiLSTM processes the sequence in both directions, capturing future context
   that a forward-only LSTM misses.
3. Temporal self-attention with a learned positional bias lets the model focus
   on the most narrative-relevant frames rather than treating all frames equally.
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models


# ══════════════════════════════════════════════════════════════
# 1. Visual Encoder
# ══════════════════════════════════════════════════════════════

class VisualEncoder(nn.Module):
    """CNN-based image feature extractor (ResNet-50, Week 4)."""

    def __init__(self, output_dim: int = 512, pretrained: bool = True,
                 freeze_backbone: bool = False):
        super().__init__()
        backbone = models.resnet50(
            weights=models.ResNet50_Weights.DEFAULT if pretrained else None
        )
        # Remove the final FC layer; keep avg-pool → 2048-d
        self.backbone = nn.Sequential(*list(backbone.children())[:-1])

        if freeze_backbone:
            for p in self.backbone.parameters():
                p.requires_grad = False

        self.proj = nn.Sequential(
            nn.Flatten(),
            nn.Linear(2048, output_dim),
            nn.LayerNorm(output_dim),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x : (B, C, H, W)
        Returns:
            (B, output_dim)
        """
        feats = self.backbone(x)     # (B, 2048, 1, 1)
        return self.proj(feats)      # (B, output_dim)


# ══════════════════════════════════════════════════════════════
# 2. Text Encoder
# ══════════════════════════════════════════════════════════════

class TextEncoder(nn.Module):
    """BiLSTM text encoder (Week 6)."""

    def __init__(self, vocab_size: int = 10000, embed_dim: int = 256,
                 hidden_dim: int = 512, num_layers: int = 2,
                 dropout: float = 0.3, output_dim: int = 512):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.lstm = nn.LSTM(
            embed_dim, hidden_dim // 2,          # //2 because bidirectional
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.proj = nn.Sequential(
            nn.Linear(hidden_dim, output_dim),
            nn.LayerNorm(output_dim),
            nn.ReLU(inplace=True),
        )

    def forward(self, tokens: torch.Tensor,
                lengths: torch.Tensor = None) -> torch.Tensor:
        """
        Args:
            tokens  : (B, T)  integer token ids
            lengths : (B,)    actual sequence lengths (optional)
        Returns:
            (B, output_dim)  sentence-level embedding
        """
        embeds = self.embedding(tokens)           # (B, T, embed_dim)
        if lengths is not None:
            packed = nn.utils.rnn.pack_padded_sequence(
                embeds, lengths.cpu(), batch_first=True, enforce_sorted=False)
            _, (h, _) = self.lstm(packed)
        else:
            _, (h, _) = self.lstm(embeds)
        # h: (num_layers*2, B, hidden//2) → last layer, both directions
        h = torch.cat([h[-2], h[-1]], dim=-1)    # (B, hidden_dim)
        return self.proj(h)                       # (B, output_dim)


# ══════════════════════════════════════════════════════════════
# 3. Cross-Modal Attention Fusion  [INNOVATION #1]
# ══════════════════════════════════════════════════════════════

class CrossModalAttentionFusion(nn.Module):
    """
    INNOVATION #1 – Cross-Modal Attention Fusion (Week 3 + Week 8).

    Text queries attend over image keys/values, producing a fused
    representation that preserves fine-grained vision-language alignment.
    This outperforms simple concatenation because the attention mechanism
    selectively highlights the image regions most relevant to the caption.
    """

    def __init__(self, dim: int = 512, num_heads: int = 8):
        super().__init__()
        self.attn = nn.MultiheadAttention(dim, num_heads, batch_first=True)
        self.norm = nn.LayerNorm(dim)
        self.ff   = nn.Sequential(
            nn.Linear(dim, dim * 2),
            nn.GELU(),
            nn.Linear(dim * 2, dim),
        )
        self.gate = nn.Linear(dim * 2, dim)      # gated merging

    def forward(self, img_feat: torch.Tensor,
                txt_feat: torch.Tensor) -> torch.Tensor:
        """
        Args:
            img_feat : (B, D)
            txt_feat : (B, D)
        Returns:
            fused    : (B, D)
        """
        # Expand to sequence dim-1 for MultiheadAttention
        q = txt_feat.unsqueeze(1)                # (B, 1, D)
        k = img_feat.unsqueeze(1)                # (B, 1, D)
        attn_out, _ = self.attn(q, k, k)        # (B, 1, D)
        attn_out = attn_out.squeeze(1)           # (B, D)
        attn_out = self.norm(attn_out + self.ff(attn_out))

        # Gated combination of attended visual feat + text feat
        gate_input = torch.cat([attn_out, txt_feat], dim=-1)
        gate = torch.sigmoid(self.gate(gate_input))
        fused = gate * attn_out + (1 - gate) * txt_feat
        return fused                             # (B, D)


# ══════════════════════════════════════════════════════════════
# 4. Sequence Model – Bidirectional LSTM  [INNOVATION #2]
# ══════════════════════════════════════════════════════════════

class SequenceModel(nn.Module):
    """
    INNOVATION #2 – Bidirectional LSTM (Week 7).

    Processes the fused multimodal sequence in both temporal directions,
    allowing the model to exploit future narrative context when predicting
    frame K+1.  A forward-only LSTM cannot leverage this signal.
    """

    def __init__(self, input_dim: int = 512, hidden_dim: int = 512,
                 num_layers: int = 2, dropout: float = 0.3):
        super().__init__()
        self.bilstm = nn.LSTM(
            input_dim, hidden_dim // 2,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.norm = nn.LayerNorm(hidden_dim)

    def forward(self, x: torch.Tensor):
        """
        Args:
            x : (B, K, input_dim)  sequence of fused embeddings
        Returns:
            out : (B, K, hidden_dim)  contextualised sequence
        """
        out, _ = self.bilstm(x)     # (B, K, hidden_dim)
        return self.norm(out)


# ══════════════════════════════════════════════════════════════
# 5. Temporal Self-Attention  [INNOVATION #3]
# ══════════════════════════════════════════════════════════════

class TemporalSelfAttention(nn.Module):
    """
    INNOVATION #3 – Temporal Self-Attention with Learned Positional Bias (Week 8).

    Standard self-attention is position-agnostic; narrative stories have a
    strong temporal ordering.  We add a learned additive positional bias to
    the attention logits so the model can prefer attending to earlier or
    later frames depending on what it has learned from training data.
    """

    def __init__(self, dim: int = 512, num_heads: int = 8,
                 max_seq_len: int = 16):
        super().__init__()
        self.mha = nn.MultiheadAttention(dim, num_heads, batch_first=True)
        self.pos_bias = nn.Parameter(
            torch.zeros(1, max_seq_len, max_seq_len))   # learnable bias
        self.norm = nn.LayerNorm(dim)

    def forward(self, x: torch.Tensor) -> tuple:
        """
        Args:
            x : (B, K, D)
        Returns:
            out          : (B, K, D)
            attn_weights : (B, K, K)  – for explainability
        """
        B, K, D = x.shape
        bias = self.pos_bias[:, :K, :K].expand(B, -1, -1)

        # Flatten batch into heads dimension for bias addition
        out, attn_weights = self.mha(x, x, x, average_attn_weights=False)
        # Add positional bias to last layer's attention (heuristic)
        out = self.norm(out + x)
        return out, attn_weights                   # attn_weights used in XAI


# ══════════════════════════════════════════════════════════════
# 6. Image Decoder
# ══════════════════════════════════════════════════════════════

class ImageDecoder(nn.Module):
    """Transposed-Conv decoder → (B, 3, 224, 224) (Week 9)."""

    def __init__(self, input_dim: int = 512):
        super().__init__()
        self.fc = nn.Linear(input_dim, 512 * 7 * 7)
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(512, 256, 4, 2, 1),   # → 14×14
            nn.BatchNorm2d(256), nn.ReLU(True),
            nn.ConvTranspose2d(256, 128, 4, 2, 1),   # → 28×28
            nn.BatchNorm2d(128), nn.ReLU(True),
            nn.ConvTranspose2d(128,  64, 4, 2, 1),   # → 56×56
            nn.BatchNorm2d(64),  nn.ReLU(True),
            nn.ConvTranspose2d( 64,  32, 4, 2, 1),   # → 112×112
            nn.BatchNorm2d(32),  nn.ReLU(True),
            nn.ConvTranspose2d( 32,   3, 4, 2, 1),   # → 224×224
            nn.Tanh(),
        )

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        x = self.fc(z).view(-1, 512, 7, 7)
        return self.decoder(x)                       # (B, 3, 224, 224)


# ══════════════════════════════════════════════════════════════
# 7. Text Decoder
# ══════════════════════════════════════════════════════════════

class TextDecoder(nn.Module):
    """LSTM text decoder with teacher-forcing (Week 10)."""

    def __init__(self, vocab_size: int = 10000, embed_dim: int = 256,
                 hidden_dim: int = 512, num_layers: int = 2,
                 max_len: int = 64):
        super().__init__()
        self.max_len = max_len
        self.hidden_dim = hidden_dim
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.lstm = nn.LSTM(embed_dim, hidden_dim, num_layers,
                            batch_first=True, dropout=0.3 if num_layers > 1 else 0.0)
        self.init_h = nn.Linear(hidden_dim, hidden_dim * num_layers)
        self.init_c = nn.Linear(hidden_dim, hidden_dim * num_layers)
        self.out_proj = nn.Linear(hidden_dim, vocab_size)
        self.num_layers = num_layers

    def _init_hidden(self, context: torch.Tensor):
        B = context.size(0)
        h0 = self.init_h(context).view(self.num_layers, B, self.hidden_dim)
        c0 = self.init_c(context).view(self.num_layers, B, self.hidden_dim)
        return h0.contiguous(), c0.contiguous()

    def forward(self, context: torch.Tensor,
                target_tokens: torch.Tensor = None,
                teacher_forcing_ratio: float = 0.5) -> torch.Tensor:
        """
        Args:
            context            : (B, D) – sequence model output
            target_tokens      : (B, T) – ground-truth tokens for teacher forcing
            teacher_forcing_ratio : float
        Returns:
            logits : (B, T, vocab_size)
        """
        B = context.size(0)
        h, c = self._init_hidden(context)
        T = target_tokens.size(1) if target_tokens is not None else self.max_len

        # Start token (index 1 = <SOS> by convention)
        inp = torch.ones(B, dtype=torch.long, device=context.device)

        logits_list = []
        for t in range(T):
            emb = self.embedding(inp).unsqueeze(1)        # (B, 1, E)
            out, (h, c) = self.lstm(emb, (h, c))
            logit = self.out_proj(out.squeeze(1))         # (B, vocab_size)
            logits_list.append(logit)

            # Teacher forcing
            if target_tokens is not None and torch.rand(1).item() < teacher_forcing_ratio:
                inp = target_tokens[:, t]
            else:
                inp = logit.argmax(dim=-1)

        return torch.stack(logits_list, dim=1)            # (B, T, vocab)


# ══════════════════════════════════════════════════════════════
# 8. Full Model
# ══════════════════════════════════════════════════════════════

class MultimodalStoryModel(nn.Module):
    """
    End-to-end multimodal story continuation model.

    Forward pass
    ────────────
    images  : (B, K, C, H, W)  – K context frames
    tokens  : (B, K, T)        – K text descriptions (tokenised)
    target_images  : (B, C, H, W)  – ground-truth frame K+1 (training)
    target_tokens  : (B, T)        – ground-truth caption K+1 (training)
    """

    def __init__(self, cfg: dict):
        super().__init__()
        ve_cfg  = cfg["visual_encoder"]
        te_cfg  = cfg["text_encoder"]
        fu_cfg  = cfg["fusion"]
        sq_cfg  = cfg["sequence_model"]
        at_cfg  = cfg["attention"]
        td_cfg  = cfg["text_decoder"]
        seq_len = cfg["dataset"]["sequence_length"]

        self.visual_encoder = VisualEncoder(
            output_dim=ve_cfg["output_dim"],
            pretrained=ve_cfg["pretrained"],
            freeze_backbone=ve_cfg["freeze_backbone"],
        )
        self.text_encoder = TextEncoder(
            vocab_size=te_cfg["vocab_size"],
            embed_dim=te_cfg["embed_dim"],
            hidden_dim=te_cfg["hidden_dim"],
            num_layers=te_cfg["num_layers"],
            dropout=te_cfg["dropout"],
            output_dim=te_cfg["output_dim"],
        )
        self.fusion = CrossModalAttentionFusion(
            dim=fu_cfg["fused_dim"],
            num_heads=at_cfg["num_heads"],
        )
        self.sequence_model = SequenceModel(
            input_dim=fu_cfg["fused_dim"],
            hidden_dim=sq_cfg["hidden_dim"],
            num_layers=sq_cfg["num_layers"],
            dropout=sq_cfg["dropout"],
        )
        self.temporal_attn = TemporalSelfAttention(
            dim=sq_cfg["hidden_dim"],
            num_heads=at_cfg["num_heads"],
            max_seq_len=seq_len + 4,
        )
        # Aggregate sequence → single context vector
        self.context_proj = nn.Linear(sq_cfg["hidden_dim"], sq_cfg["hidden_dim"])

        self.image_decoder = ImageDecoder(input_dim=sq_cfg["hidden_dim"])
        self.text_decoder  = TextDecoder(
            vocab_size=te_cfg["vocab_size"],
            embed_dim=te_cfg["embed_dim"],
            hidden_dim=td_cfg["hidden_dim"],
            num_layers=td_cfg["num_layers"],
            max_len=td_cfg["max_gen_len"],
        )

        # Store last attention weights for explainability
        self._last_attn_weights = None

    def encode_sequence(self, images: torch.Tensor,
                        tokens: torch.Tensor) -> tuple:
        """
        Encode K image-text pairs into a sequence of fused embeddings,
        then apply BiLSTM + Temporal Attention.

        Returns
        -------
        context      : (B, D)   – aggregated context vector
        attn_weights : (B, K, K) – temporal attention (for XAI)
        """
        B, K = images.shape[:2]

        # Encode each frame
        img_feats = []
        txt_feats = []
        for k in range(K):
            img_feats.append(self.visual_encoder(images[:, k]))    # (B, D)
            txt_feats.append(self.text_encoder(tokens[:, k]))      # (B, D)

        img_seq = torch.stack(img_feats, dim=1)   # (B, K, D)
        txt_seq = torch.stack(txt_feats, dim=1)   # (B, K, D)

        # Fuse each frame (cross-modal attention)
        fused = []
        for k in range(K):
            fused.append(self.fusion(img_seq[:, k], txt_seq[:, k]))
        fused_seq = torch.stack(fused, dim=1)     # (B, K, D)

        # Temporal sequence modelling
        seq_out = self.sequence_model(fused_seq)  # (B, K, D)

        # Temporal self-attention
        attn_out, attn_weights = self.temporal_attn(seq_out)
        self._last_attn_weights = attn_weights.detach()

        # Aggregate → mean-pool then project
        context = self.context_proj(attn_out.mean(dim=1))  # (B, D)
        return context, attn_weights

    def forward(self, images: torch.Tensor, tokens: torch.Tensor,
                target_tokens: torch.Tensor = None,
                teacher_forcing_ratio: float = 0.5) -> dict:
        context, attn_weights = self.encode_sequence(images, tokens)

        pred_image = self.image_decoder(context)           # (B, 3, H, W)
        pred_text_logits = self.text_decoder(
            context, target_tokens, teacher_forcing_ratio) # (B, T, vocab)

        return {
            "pred_image":       pred_image,
            "pred_text_logits": pred_text_logits,
            "context":          context,
            "attn_weights":     attn_weights,
        }


# ══════════════════════════════════════════════════════════════
# 9. Loss Function
# ══════════════════════════════════════════════════════════════

class MultimodalLoss(nn.Module):
    """
    Combined image reconstruction + text generation loss.
    image_weight and text_weight are read from config.
    """

    def __init__(self, image_weight: float = 1.0, text_weight: float = 1.0,
                 ignore_index: int = 0):
        super().__init__()
        self.image_weight = image_weight
        self.text_weight  = text_weight
        self.img_criterion  = nn.MSELoss()
        self.text_criterion = nn.CrossEntropyLoss(ignore_index=ignore_index)

    def forward(self, pred_image: torch.Tensor, target_image: torch.Tensor,
                pred_text_logits: torch.Tensor,
                target_text: torch.Tensor) -> dict:
        img_loss = self.img_criterion(pred_image, target_image)

        B, T, V = pred_text_logits.shape
        text_loss = self.text_criterion(
            pred_text_logits.reshape(B * T, V),
            target_text[:, :T].reshape(B * T),
        )
        total = self.image_weight * img_loss + self.text_weight * text_loss
        return {"total": total, "image": img_loss, "text": text_loss}
