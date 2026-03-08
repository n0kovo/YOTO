"""
Integrated Charm + DINOv3 + YOTO model.

Architecture:
  DINOv3-ConvNeXt-Base backbone (hierarchical 4-stage features)
  -> YOTO's CTR / MSSA / CSA multi-scale fusion
  -> Segment embeddings for FR/NR mode switching
  -> Weighted score regression

ConvNeXt-Base produces feature maps at the same spatial resolutions as
Swin-T (56x56, 28x28, 14x14, 7x7 for 224 input), so YOTO's fusion
modules can be reused with only channel projection changes.

The model internally re-normalizes from YOTO's [-1,1] range to ImageNet
normalization, so no data pipeline changes are needed.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from transformers import AutoModel


# ---------------------------------------------------------------------------
# Attention modules (identical to YOTO's original model.py)
# ---------------------------------------------------------------------------

class CTR(nn.Module):
    """Channel-wise attention response."""

    def __init__(self, dim, drop=0.1):
        super().__init__()
        self.c_q = nn.Linear(dim, dim)
        self.c_k = nn.Linear(dim, dim)
        self.c_v = nn.Linear(dim, dim)
        self.norm_fact = dim ** -0.5
        self.softmax = nn.Softmax(dim=-1)
        self.proj_drop = nn.Dropout(drop)
        self.scale = nn.Sequential(nn.Linear(dim, 1), nn.GELU())

    def forward(self, x):
        _x = x
        B, C, N = x.shape
        q = self.c_q(x)
        k = self.c_k(x)
        v = self.c_v(x)
        scale = self.scale(x)
        attn = q @ k.transpose(-2, -1) * self.norm_fact
        attn = self.softmax(attn)
        x = (attn @ v).transpose(1, 2).reshape(B, C, N)
        x = self.proj_drop(x)
        x = x * scale + _x
        x = x + _x
        return x


class CSA(nn.Module):
    """Cross-scale attention."""

    def __init__(self, dim, drop=0.1):
        super().__init__()
        self.c_q = nn.Linear(dim, dim)
        self.c_k = nn.Linear(dim, dim)
        self.c_v = nn.Linear(dim, dim)
        self.norm_fact = dim ** -0.5
        self.softmax = nn.Softmax(dim=-1)
        self.proj_drop = nn.Dropout(drop)
        self.scale = nn.Sequential(nn.Linear(dim, 1), nn.GELU())

    def patchify(self, lo, hi, lohw, hihw, minhw=7):
        B, Nl, Cl = lo.shape
        _, Nh, Ch = hi.shape
        base = hihw // 7
        rate = lohw // hihw * base

        nphi = rearrange(hi, "b (h w) c -> b c h w", h=hihw, w=hihw)
        nphi = F.unfold(nphi, kernel_size=base, stride=base, padding=0)
        nphi = rearrange(nphi, "b (c h w) p -> b p (h w) c", c=Ch, h=base, w=base)

        nplo = rearrange(lo, "b (h w) c -> b c h w", h=lohw, w=lohw)
        nplo = F.unfold(nplo, kernel_size=rate, stride=rate, padding=0)
        nplo = rearrange(nplo, "b (c h w) p -> b p (h w) c", c=Cl, h=rate, w=rate)

        return nplo, nphi

    def forward(self, lo, hi, lohw, hihw, patchify=True):
        if patchify:
            lo, hi = self.patchify(lo, hi, lohw, hihw)
            B, p, N, C = lo.shape
            base = hihw // 7
        else:
            B, N, C = lo.shape
            base = 1

        _lo = lo
        q = self.c_q(lo)
        k = self.c_k(hi)
        v = self.c_v(hi)
        scale = self.scale(lo)

        attn = q @ k.transpose(-2, -1) * self.norm_fact
        attn = self.softmax(attn)
        if patchify:
            lo = (attn @ v).reshape(B, p, N, C)
        else:
            lo = (attn @ v).reshape(B, N, C)

        lo = self.proj_drop(lo)
        lo = lo * scale + _lo

        if patchify:
            lo = rearrange(
                lo, "b p (h w) c -> b (c h w) p",
                c=C, h=lohw // hihw * base, w=lohw // hihw * base,
            )
            lo = F.fold(
                lo, output_size=lohw,
                kernel_size=lohw // hihw * base,
                stride=lohw // hihw * base, padding=0,
            )
            lo = rearrange(lo, "b c h w -> b (h w) c", h=lohw, w=lohw)

        return lo


class MSSA(nn.Module):
    """Multi-scale spatial attention with optional patch-wise local attention."""

    def __init__(self, dim, hw_dim, drop=0.1, patchify=False):
        super().__init__()
        self.c_q1 = nn.Linear(dim, dim)
        self.c_k1 = nn.Linear(dim, dim)
        self.c_v1 = nn.Linear(dim, dim)
        self.norm_fact = dim ** -0.5
        self.proj_drop1 = nn.Dropout(drop)

        if patchify:
            self.c_q2 = nn.Linear(dim, dim)
            self.c_k2 = nn.Linear(dim, dim)
            self.c_v2 = nn.Linear(dim, dim)
            self.proj_drop2 = nn.Dropout(drop)

        self.softmax = nn.Softmax(dim=-1)
        self.patch_wise = patchify
        self.hw = hw_dim

        self.scale1 = nn.Sequential(nn.Linear(dim, 1), nn.GELU())
        self.scale2 = nn.Sequential(nn.Linear(dim, 1), nn.GELU())

    def patchify(self, feat, channel):
        np_ = rearrange(feat, "b (h w) c -> b c h w", h=self.hw, w=self.hw)
        np_ = F.unfold(np_, kernel_size=self.hw // 2, stride=self.hw // 2, padding=0)
        np_ = rearrange(
            np_, "b (c h w) p -> b p (h w) c",
            c=channel, h=self.hw // 2, w=self.hw // 2,
        )
        return np_

    def forward(self, x, y):
        B, N, C = x.shape

        _x = x
        q1 = self.c_q1(y)
        k1 = self.c_k1(x)
        v1 = self.c_v1(x)
        attn_glb = q1 @ k1.transpose(-2, -1) * self.norm_fact
        attn_glb = self.softmax(attn_glb)
        xg = (attn_glb @ v1).reshape(B, N, C)
        xg = self.proj_drop1(xg)
        scale1 = self.scale1(x)
        x = scale1 * xg + _x

        if self.patch_wise:
            _x = x
            np_ = self.patchify(x, channel=C)
            npy = self.patchify(y, channel=C)

            q2 = self.c_q2(npy)
            k2 = self.c_k2(np_)
            v2 = self.c_v2(np_)
            attn_lcl = q2 @ k2.transpose(-2, -1) * self.norm_fact
            attn_lcl = self.softmax(attn_lcl)
            xl = (attn_lcl @ v2).reshape(B, 4, (self.hw // 2) ** 2, C)

            xl = rearrange(
                xl, "b p (h w) c -> b (c h w) p",
                c=C, h=self.hw // 2, w=self.hw // 2,
            )
            xl = F.fold(
                xl, output_size=self.hw,
                kernel_size=self.hw // 2, stride=self.hw // 2, padding=0,
            )
            xl = rearrange(xl, "b c h w -> b (h w) c", h=self.hw, w=self.hw)

            xl = self.proj_drop2(xl)
            scale2 = self.scale2(x)
            x = scale2 * xl + _x

        return x


# ---------------------------------------------------------------------------
# Backbone helpers
# ---------------------------------------------------------------------------

def _detect_dims(backbone):
    """Return per-stage channel dims from the backbone config or model name."""
    cfg = getattr(backbone, "config", None)
    if cfg is not None:
        if hasattr(cfg, "hidden_sizes"):
            return list(cfg.hidden_sizes)
        if hasattr(cfg, "dims"):
            return list(cfg.dims)
    # Fallback: infer from model name string
    name = str(getattr(cfg, "_name_or_path", "")).lower() if cfg else ""
    if "large" in name:
        return [192, 384, 768, 1536]
    if "base" in name:
        return [128, 256, 512, 1024]
    return [96, 192, 384, 768]


# ---------------------------------------------------------------------------
# Main model
# ---------------------------------------------------------------------------

class Net(nn.Module):
    """
    Charm + DINOv3-ConvNeXt + YOTO unified FR/NR IQA model.

    Drop-in replacement for the original YOTO Net — same forward(x, y, mode)
    signature, same training / inference scripts.
    """

    def __init__(self, cfg, device):
        super().__init__()
        self.device = device
        self.cfg = cfg

        backbone_name = getattr(
            cfg, "backbone",
            "facebook/dinov3-convnext-base-pretrain-lvd1689m",
        )
        self.encoder = AutoModel.from_pretrained(
            backbone_name, output_hidden_states=True
        )
        dims = _detect_dims(self.encoder)
        self._stage_dims = dims

        # ImageNet normalisation buffers
        self.register_buffer(
            "img_mean", torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
        )
        self.register_buffer(
            "img_std", torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
        )

        # ---- channel projections (stage dims -> 768) ----
        self.emb1 = nn.Conv2d(dims[0], 768, kernel_size=1, bias=True)
        self.emb2 = nn.Conv2d(dims[1], 768, kernel_size=1, bias=True)
        self.emb3 = nn.Conv2d(dims[2], 768, kernel_size=1, bias=True)
        self.emb4 = nn.Conv2d(dims[3], 768, kernel_size=1, bias=True)

        # ---- channel-wise attention (B, C, N) ----
        self.CA1 = CTR(3136)   # 56*56
        self.CA2 = CTR(784)    # 28*28
        self.CA3 = CTR(196)    # 14*14
        self.CA4 = CTR(49)     #  7*7

        # ---- cross-scale / multi-scale spatial attention (B, N, C) ----
        self.CSA11 = MSSA(dim=768, hw_dim=56, patchify=True)
        self.CSA12 = CSA(768)
        self.CSA13 = CSA(768)
        self.CSA14 = CSA(768)

        self.CSA22 = MSSA(dim=768, hw_dim=28, patchify=True)
        self.CSA23 = CSA(768)
        self.CSA24 = CSA(768)

        self.CSA33 = MSSA(dim=768, hw_dim=14, patchify=True)
        self.CSA34 = CSA(768)

        self.CSA44 = MSSA(dim=768, hw_dim=7, patchify=False)

        # ---- layer norms ----
        self.ln11 = nn.LayerNorm(768)
        self.ln22 = nn.LayerNorm(768)
        self.ln33 = nn.LayerNorm(768)
        self.ln44 = nn.LayerNorm(768)
        self.ln12 = nn.LayerNorm(768)
        self.ln13 = nn.LayerNorm(768)
        self.ln14 = nn.LayerNorm(768)
        self.ln23 = nn.LayerNorm(768)
        self.ln24 = nn.LayerNorm(768)
        self.ln34 = nn.LayerNorm(768)

        # ---- score generation ----
        self.fc_score = nn.Sequential(
            nn.Linear(768, 384), nn.ReLU(), nn.Dropout(0.1),
            nn.Linear(384, 1), nn.ReLU(),
        )
        self.fc_weight = nn.Sequential(
            nn.Linear(768, 384), nn.ReLU(), nn.Dropout(0.1),
            nn.Linear(384, 1), nn.Sigmoid(),
        )

        # ---- segment embeddings: 0 = NR / self-ref, 1 = FR / true ref ----
        self.seg_embd1 = nn.Embedding(2, 768)
        self.seg_embd2 = nn.Embedding(2, 768)
        self.seg_embd3 = nn.Embedding(2, 768)
        self.seg_embd4 = nn.Embedding(2, 768)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _renormalize(self, x):
        """Convert YOTO normalisation ([-1, 1]) to ImageNet normalisation."""
        x = x * 0.5 + 0.5                          # -> [0, 1]
        x = (x - self.img_mean) / self.img_std      # -> ImageNet
        return x

    def _extract_features(self, x):
        """Run the ConvNeXt backbone and return 4-level (B, C, H, W) features.

        Groups hidden states by spatial resolution, keeps only those whose
        channel count matches one of the expected stage dims, and returns
        the most-processed version at each level.
        """
        outputs = self.encoder(pixel_values=x, output_hidden_states=True)
        hs = outputs.hidden_states

        expected_channels = set(self._stage_dims)

        # De-duplicate by spatial size, keep last (most processed) at each res.
        # Skip tensors whose channel dim doesn't match any expected stage dim
        # (e.g. the raw 3-channel input or stem embedding).
        by_size = {}
        for h in hs:
            if h.dim() == 4 and h.shape[1] in expected_channels:
                by_size[(h.shape[2], h.shape[3])] = h

        # Sort largest-first (56 > 28 > 14 > 7)
        sorted_feats = [
            by_size[k]
            for k in sorted(by_size, key=lambda k: k[0] * k[1], reverse=True)
        ]
        assert len(sorted_feats) >= 4, (
            f"Expected >= 4 feature levels, got {len(sorted_feats)}: "
            f"{[f.shape for f in sorted_feats]}"
        )
        return sorted_feats[0], sorted_feats[1], sorted_feats[2], sorted_feats[3]

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------

    def common_forward(self, x):
        B = x.shape[0]
        x = self._renormalize(x)
        layer1, layer2, layer3, layer4 = self._extract_features(x)

        # project to 768 channels
        layer1 = self.emb1(layer1)
        layer2 = self.emb2(layer2)
        layer3 = self.emb3(layer3)
        layer4 = self.emb4(layer4)

        # reshape to B, C, N for channel-wise attention
        layer1 = rearrange(layer1, "b c h w -> b c (h w)")
        layer2 = rearrange(layer2, "b c h w -> b c (h w)")
        layer3 = rearrange(layer3, "b c h w -> b c (h w)")
        layer4 = rearrange(layer4, "b c h w -> b c (h w)")

        layer1 = self.CA1(layer1)
        layer2 = self.CA2(layer2)
        layer3 = self.CA3(layer3)
        layer4 = self.CA4(layer4)

        # reshape to B, N, C for spatial attention
        layer1 = rearrange(layer1, "b c n -> b n c")
        layer2 = rearrange(layer2, "b c n -> b n c")
        layer3 = rearrange(layer3, "b c n -> b n c")
        layer4 = rearrange(layer4, "b c n -> b n c")

        return layer1, layer2, layer3, layer4

    def forward(self, x, y, mode="NR"):
        B = x.shape[0]

        l1, l2, l3, l4 = self.common_forward(x)
        l1y, l2y, l3y, l4y = self.common_forward(y)

        # segment embeddings
        embd1 = self.seg_embd1(torch.zeros(B, 3136).long().to(x.device))
        embd2 = self.seg_embd2(torch.zeros(B, 784).long().to(x.device))
        embd3 = self.seg_embd3(torch.zeros(B, 196).long().to(x.device))
        embd4 = self.seg_embd4(torch.zeros(B, 49).long().to(x.device))

        if mode == "FR":
            embd1y = self.seg_embd1(torch.ones(B, 3136).long().to(x.device))
            embd2y = self.seg_embd2(torch.ones(B, 784).long().to(x.device))
            embd3y = self.seg_embd3(torch.ones(B, 196).long().to(x.device))
            embd4y = self.seg_embd4(torch.ones(B, 49).long().to(x.device))
        else:
            embd1y = self.seg_embd1(torch.zeros(B, 3136).long().to(x.device))
            embd2y = self.seg_embd2(torch.zeros(B, 784).long().to(x.device))
            embd3y = self.seg_embd3(torch.zeros(B, 196).long().to(x.device))
            embd4y = self.seg_embd4(torch.zeros(B, 49).long().to(x.device))

        # same-scale cross-attention (distorted vs reference)
        out11 = self.CSA11(l1 + embd1, l1y + embd1y)
        out22 = self.CSA22(l2 + embd2, l2y + embd2y)
        out33 = self.CSA33(l3 + embd3, l3y + embd3y)
        out44 = self.CSA44(l4 + embd4, l4y + embd4y)

        # cross-scale attention (lo queries hi)
        out12 = self.CSA12(l1, l2, lohw=56, hihw=28, patchify=True)
        out13 = self.CSA13(l1, l3, lohw=56, hihw=14, patchify=True)
        out14 = self.CSA14(l1, l4, lohw=56, hihw=7, patchify=True)

        out23 = self.CSA23(l2, l3, lohw=28, hihw=14, patchify=True)
        out24 = self.CSA24(l2, l4, lohw=28, hihw=7, patchify=True)

        out34 = self.CSA34(l3, l4, lohw=14, hihw=7, patchify=True)

        # normalise
        out11 = self.ln11(out11)
        out22 = self.ln22(out22)
        out33 = self.ln33(out33)
        out44 = self.ln44(out44)
        out12 = self.ln12(out12)
        out13 = self.ln13(out13)
        out14 = self.ln14(out14)
        out23 = self.ln23(out23)
        out24 = self.ln24(out24)
        out34 = self.ln34(out34)

        # aggregate per scale
        out1 = (out11 + out12 + out13 + out14) / 4
        out2 = (out22 + out23 + out24) / 3
        out3 = (out33 + out34) / 2

        out = torch.cat([out1, out2, out3, out44], dim=1)  # B, N_total, 768

        # weighted score
        s = self.fc_score(out)
        w = self.fc_weight(out)
        score = torch.sum(s * w, dim=1) / torch.sum(w, dim=1)

        return score
