"""
SHAP (SHapley Additive exPlanations) for image classification using GradientExplainer.

Reference: Lundberg & Lee, 2017 — "A Unified Approach to Interpreting Model Predictions."
"""

import numpy as np
import torch
import shap
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from PIL import Image
from utils.preprocessing import denormalize


def run_shap(
    model: torch.nn.Module,
    img_tensor: torch.Tensor,
    n_background: int = 20,
) -> tuple[plt.Figure, float]:
    """
    Compute SHAP values for an image using GradientExplainer with random background.

    Args:
        model:        Trained FusionModel.
        img_tensor:   Preprocessed input tensor [1, 3, H, W].
        n_background: Number of random Gaussian background images.

    Returns:
        fig:        matplotlib Figure with SHAP overlay
        shap_score: Mean absolute SHAP value (0–1 normalized) as proxy score
    """
    model.eval()
    device = next(model.parameters()).device

    # Generate random background samples (Gaussian noise)
    background = torch.randn(n_background, *img_tensor.shape[1:]).to(device)

    # GradientExplainer (faster than DeepExplainer for ResNet)
    explainer = shap.GradientExplainer(model, background)

    shap_values = explainer.shap_values(img_tensor)   # list of [1, 3, H, W] per class

    # Use the predicted class's SHAP values
    with torch.no_grad():
        logits = model(img_tensor)
        pred_class = torch.argmax(logits, dim=1).item()

    # shap_values is list[num_classes] each [1, 3, H, W]
    sv_class = shap_values[pred_class][0]  # [3, H, W]

    # Normalize SHAP map to [0, 1] for display
    sv_abs = np.abs(sv_class).sum(axis=0)  # [H, W]
    if sv_abs.max() > 0:
        sv_norm = sv_abs / sv_abs.max()
    else:
        sv_norm = sv_abs

    shap_score = float(np.clip(sv_norm.mean() * 5, 0, 1))  # rescale mean to 0–1

    # ── Denormalize original image for display ──────────────────────────
    orig_display = denormalize(img_tensor.squeeze(0).cpu())  # [3, H, W]
    orig_np = orig_display.permute(1, 2, 0).numpy()         # [H, W, 3]

    # ── SHAP overlay as red–blue diverging ──────────────────────────────
    sv_signed = sv_class.sum(axis=0)  # [H, W]
    vmax = np.abs(sv_signed).max() + 1e-8
    norm = mcolors.TwoSlopeNorm(vmin=-vmax, vcenter=0, vmax=vmax)

    fig, axes = plt.subplots(1, 3, figsize=(13, 4), facecolor="#0d0d0d")

    axes[0].imshow(orig_np, cmap="gray")
    axes[0].set_title("Original X-ray", color="white", fontsize=11)
    axes[0].axis("off")

    im1 = axes[1].imshow(sv_norm, cmap="hot")
    axes[1].set_title("SHAP Magnitude Map", color="white", fontsize=11)
    axes[1].axis("off")
    fig.colorbar(im1, ax=axes[1], fraction=0.03, pad=0.02).ax.yaxis.set_tick_params(color="white")

    im2 = axes[2].imshow(sv_signed, cmap="RdBu_r", norm=norm)
    axes[2].set_title(f"SHAP Signed (score={shap_score:.3f})", color="white", fontsize=11)
    axes[2].axis("off")
    fig.colorbar(im2, ax=axes[2], fraction=0.03, pad=0.02).ax.yaxis.set_tick_params(color="white")

    plt.tight_layout(pad=0.5)
    return fig, shap_score
