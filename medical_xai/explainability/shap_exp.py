"""
SHAP Explainability using Integrated Gradients.
Clean white background, focused on lung regions.
"""

import numpy as np
import torch
import torch.nn.functional as F
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import cv2
from PIL import Image
from utils.preprocessing import denormalize


def _compute_integrated_gradients(model, img_tensor, steps=30):
    model.eval()
    device = next(model.parameters()).device
    img    = img_tensor.to(device)

    with torch.no_grad():
        logits     = model(img)
        pred_class = torch.argmax(logits, dim=1).item()
        confidence = torch.softmax(logits, dim=1)[0, pred_class].item()

    baseline = torch.zeros_like(img)
    alphas   = torch.linspace(0, 1, steps).to(device)
    integrated_grads = torch.zeros_like(img)

    for a in alphas:
        interp = (baseline + a * (img - baseline)).requires_grad_(True)
        score  = model(interp)[0, pred_class]
        model.zero_grad()
        score.backward()
        integrated_grads += interp.grad.detach()

    attributions = (integrated_grads / steps) * (img - baseline)
    return attributions.squeeze(0).cpu().numpy(), pred_class, confidence


def run_shap(model, img_tensor, n_background: int = 20):
    model.eval()
    device = next(model.parameters()).device

    # Always use Integrated Gradients — more reliable than GradientExplainer
    attributions, pred_class, confidence = _compute_integrated_gradients(
        model, img_tensor
    )

    # ── Process maps ─────────────────────────────────────────────────────
    sv_abs    = np.abs(attributions).sum(axis=0)    # [H, W] magnitude
    sv_signed = attributions.sum(axis=0)             # [H, W] signed

    # Normalize magnitude
    sv_norm    = sv_abs / (sv_abs.max() + 1e-8)
    shap_score = float(np.clip(sv_norm.mean() * 10, 0, 1))

    # ── Original image ───────────────────────────────────────────────────
    orig_np   = denormalize(img_tensor.squeeze(0).cpu()).permute(1, 2, 0).numpy()
    orig_np   = np.clip(orig_np, 0, 1)
    orig_gray = np.mean(orig_np, axis=2)
    H, W      = orig_gray.shape
    xray_rgb  = np.stack([orig_gray] * 3, axis=2)

    # Resize & smooth
    sv_norm_r   = cv2.resize(sv_norm,   (W, H), interpolation=cv2.INTER_LINEAR)
    sv_signed_r = cv2.resize(sv_signed, (W, H), interpolation=cv2.INTER_LINEAR)
    sv_norm_r   = cv2.GaussianBlur(sv_norm_r,   (15, 15), 0)
    sv_signed_r = cv2.GaussianBlur(sv_signed_r, (15, 15), 0)

    # Re-normalise after blur
    sv_norm_r = sv_norm_r / (sv_norm_r.max() + 1e-8)

    # ── Magnitude overlay — keep X-ray visible, subtle heatmap ───────────
    sv_gamma  = np.power(sv_norm_r, 0.6)           # less aggressive gamma
    heat_rgb  = plt.colormaps.get_cmap("hot")(sv_gamma)[:, :, :3]
    alpha_mag = sv_gamma[:, :, np.newaxis] * 0.55  # reduced opacity
    mag_blend = np.clip((1 - alpha_mag) * xray_rgb + alpha_mag * heat_rgb, 0, 1)

    # ── Signed overlay ───────────────────────────────────────────────────
    vmax      = np.abs(sv_signed_r).max() + 1e-8
    sv01      = (sv_signed_r + vmax) / (2 * vmax)
    rdbu_rgb  = plt.colormaps.get_cmap("RdBu_r")(sv01)[:, :, :3]
    alpha_sgn = np.power(np.abs(sv_signed_r) / vmax, 0.6)[:, :, np.newaxis] * 0.55
    sgn_blend = np.clip((1 - alpha_sgn) * xray_rgb + alpha_sgn * rdbu_rgb, 0, 1)

    # ── Plot — white background ──────────────────────────────────────────
    fig, axes = plt.subplots(1, 3, figsize=(15, 5), facecolor="white")
    fig.patch.set_facecolor("white")

    pred_label = "Pneumonia" if pred_class == 1 else "Normal"

    axes[0].imshow(orig_gray, cmap="gray", vmin=0, vmax=1)
    axes[0].set_title("Original X-ray", fontsize=12,
                       fontweight="bold", color="#01579b", pad=10)
    axes[0].axis("off")

    axes[1].imshow(mag_blend)
    axes[1].set_title("Attribution Magnitude\nWhite/Yellow = most important regions",
                       fontsize=10, fontweight="bold", color="#01579b", pad=10)
    axes[1].axis("off")
    sm1 = plt.cm.ScalarMappable(cmap="hot", norm=plt.Normalize(0, 1))
    cb1 = fig.colorbar(sm1, ax=axes[1], fraction=0.04, pad=0.02)
    cb1.set_label("Importance", fontsize=9, color="#01579b")
    cb1.ax.tick_params(colors="#01579b", labelsize=8)

    axes[2].imshow(sgn_blend)
    axes[2].set_title(f"Signed Attribution  (score={shap_score:.4f})\nRed = Pneumonia evidence  |  Blue = Normal evidence",
                       fontsize=10, fontweight="bold", color="#01579b", pad=10)
    axes[2].axis("off")
    norm2 = mcolors.TwoSlopeNorm(vmin=-1, vcenter=0, vmax=1)
    sm2   = plt.cm.ScalarMappable(cmap="RdBu_r", norm=norm2)
    cb2   = fig.colorbar(sm2, ax=axes[2], fraction=0.04, pad=0.02)
    cb2.set_label("Attribution", fontsize=9, color="#01579b")
    cb2.ax.tick_params(colors="#01579b", labelsize=8)

    fig.suptitle(
        f"SHAP / Gradient Attribution  |  Predicted: {pred_label}  |  Confidence: {confidence:.1%}",
        fontsize=12, fontweight="bold", color="#0288d1", y=1.01
    )
    plt.tight_layout(pad=1.0)

    return fig, shap_score