"""
Grad-CAM (Gradient-weighted Class Activation Mapping) for ResNet18.

Reference: Selvaraju et al., 2017 — "Grad-CAM: Visual Explanations from
Deep Networks via Gradient-based Localization."
"""

import numpy as np
import cv2
import torch
import torch.nn.functional as F
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.cm as cm


# ── Hook helpers ─────────────────────────────────────────────────────────────

class GradCAMHook:
    """Registers forward/backward hooks on a target layer."""

    def __init__(self, layer: torch.nn.Module):
        self.activations = None
        self.gradients   = None
        self._fwd_hook = layer.register_forward_hook(self._save_activation)
        self._bwd_hook = layer.register_full_backward_hook(self._save_gradient)

    def _save_activation(self, module, inp, out):
        self.activations = out.detach()

    def _save_gradient(self, module, grad_in, grad_out):
        self.gradients = grad_out[0].detach()

    def remove(self):
        self._fwd_hook.remove()
        self._bwd_hook.remove()


# ── Core function ─────────────────────────────────────────────────────────────

def generate_gradcam(
    model: torch.nn.Module,
    img_tensor: torch.Tensor,
    original_image: Image.Image,
    target_class: int = None,
    alpha: float = 0.45,
) -> tuple[Image.Image, float]:
    """
    Generate Grad-CAM heatmap overlaid on original_image.

    Args:
        model:          The FusionModel instance.
        img_tensor:     Preprocessed input tensor [1, 3, H, W] with grad enabled.
        original_image: PIL image for overlay.
        target_class:   Class index to explain. None = argmax prediction.
        alpha:          Heatmap transparency (0 = no overlay, 1 = heatmap only).

    Returns:
        overlay_pil: PIL.Image with Grad-CAM overlay
        score:       Grad-CAM confidence proxy (mean max activation, 0–1)
    """
    model.eval()

    # Target layer: last residual block of ResNet18
    target_layer = model.cnn.layer4[-1]
    hook = GradCAMHook(target_layer)

    # Forward pass
    img_tensor = img_tensor.clone().requires_grad_(True)
    logits = model(img_tensor)
    probs  = torch.softmax(logits, dim=1)

    if target_class is None:
        target_class = torch.argmax(probs, dim=1).item()

    score_val = probs[0, target_class].item()

    # Backward pass on target class score
    model.zero_grad()
    class_score = logits[0, target_class]
    class_score.backward()

    # ── Compute Grad-CAM ───────────────────────────────────────────────────
    grads   = hook.gradients   # [1, C, H, W]
    acts    = hook.activations # [1, C, H, W]
    hook.remove()

    # Global average pooling of gradients → weights
    weights = grads.mean(dim=(2, 3), keepdim=True)  # [1, C, 1, 1]
    cam     = (weights * acts).sum(dim=1, keepdim=True)  # [1, 1, H, W]
    cam     = F.relu(cam)

    # Normalize 0–1
    cam_np = cam.squeeze().cpu().numpy()
    if cam_np.max() > 0:
        cam_np = cam_np / cam_np.max()

    # Resize to original image size
    orig_w, orig_h = original_image.size
    cam_resized = cv2.resize(cam_np, (orig_w, orig_h), interpolation=cv2.INTER_LINEAR)

    # Apply colormap
    colormap   = plt.colormaps.get_cmap("jet")
    heatmap_np = colormap(cam_resized)[:, :, :3]  # Drop alpha
    heatmap_np = (heatmap_np * 255).astype(np.uint8)

    # Overlay
    orig_np  = np.array(original_image.convert("RGB")).astype(np.float32)
    heat_np  = heatmap_np.astype(np.float32)
    overlay  = (1 - alpha) * orig_np + alpha * heat_np
    overlay  = np.clip(overlay, 0, 255).astype(np.uint8)

    # Add colorbar via matplotlib
    fig, axes = plt.subplots(1, 2, figsize=(10, 4), facecolor="#0d0d0d")
    axes[0].imshow(original_image, cmap="gray")
    axes[0].set_title("Original X-ray", color="white", fontsize=11)
    axes[0].axis("off")

    im = axes[1].imshow(overlay)
    axes[1].set_title(f"Grad-CAM Heatmap  (score={score_val:.3f})", color="white", fontsize=11)
    axes[1].axis("off")

    fig.colorbar(
        plt.cm.ScalarMappable(cmap="jet", norm=plt.Normalize(0, 1)),
        ax=axes[1], fraction=0.03, pad=0.02, label="Activation"
    )
    plt.tight_layout(pad=0.5)

    # Render figure to PIL image (compatible with all matplotlib versions)
    fig.canvas.draw()
    w_fig, h_fig = fig.canvas.get_width_height()
    try:
        buf = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
        buf = buf.reshape(h_fig, w_fig, 3)
    except AttributeError:
        buf = np.frombuffer(fig.canvas.buffer_rgba(), dtype=np.uint8)
        buf = buf.reshape(h_fig, w_fig, 4)[:, :, :3]
    plt.close(fig)

    return Image.fromarray(buf), score_val
