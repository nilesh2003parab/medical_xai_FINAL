"""
Grad-CAM for ResNet18 — with chest region masking.
Forces heatmap to focus on the central lung area only.
"""

import numpy as np
import cv2
import torch
import torch.nn.functional as F
from PIL import Image
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


class GradCAMHook:
    def __init__(self, layer):
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


def _chest_mask(h, w):
    """
    Create a soft mask that keeps only the central chest/lung region.
    Removes corners and borders where non-lung artifacts appear.
    Top 10%, bottom 15%, left 8%, right 8% are suppressed.
    """
    mask = np.ones((h, w), dtype=np.float32)

    # Suppress top border (shoulders/neck area above lungs)
    top_cut    = int(h * 0.10)
    # Suppress bottom border (below diaphragm)
    bottom_cut = int(h * 0.15)
    # Suppress left/right borders
    side_cut   = int(w * 0.08)

    # Zero out borders
    mask[:top_cut, :]    = 0.0
    mask[h-bottom_cut:, :] = 0.0
    mask[:, :side_cut]   = 0.0
    mask[:, w-side_cut:] = 0.0

    # Soft gradient at the edges using distance transform
    mask_uint8 = (mask * 255).astype(np.uint8)
    dist = cv2.distanceTransform(mask_uint8, cv2.DIST_L2, 5)
    if dist.max() > 0:
        dist = dist / dist.max()
    # Soften with power — gentle falloff from centre
    soft_mask = np.power(dist, 0.3)

    return soft_mask


def generate_gradcam(model, img_tensor, original_image, target_class=None, alpha=0.55):
    model.eval()

    # Use layer3 only — larger spatial resolution (14x14) = better lung coverage
    hook = GradCAMHook(model.cnn.layer3[-1])

    img_tensor = img_tensor.clone().requires_grad_(True)
    logits = model(img_tensor)
    probs  = torch.softmax(logits, dim=1)

    if target_class is None:
        target_class = torch.argmax(probs, dim=1).item()

    score_val = probs[0, target_class].item()

    model.zero_grad()
    logits[0, target_class].backward()

    grads = hook.gradients
    acts  = hook.activations
    hook.remove()

    # Grad-CAM formula
    weights = grads.mean(dim=(2, 3), keepdim=True)
    cam     = F.relu((weights * acts).sum(dim=1, keepdim=True))
    cam_np  = cam.squeeze().cpu().numpy()

    orig_w, orig_h = original_image.size

    # Resize CAM to image size
    cam_r = cv2.resize(cam_np, (orig_w, orig_h), interpolation=cv2.INTER_CUBIC)

    # ── Apply chest mask — suppress corners and borders ──────────────────
    chest_mask = _chest_mask(orig_h, orig_w)
    cam_masked = cam_r * chest_mask

    # Normalise after masking
    if cam_masked.max() > 0:
        cam_masked = cam_masked / cam_masked.max()

    # Smooth
    cam_smooth = cv2.GaussianBlur(cam_masked, (25, 25), 0)
    if cam_smooth.max() > 0:
        cam_smooth = cam_smooth / cam_smooth.max()

    # Sharpen contrast
    cam_final = np.power(cam_smooth, 0.6)

    # Build overlay
    import matplotlib.pyplot as plt
    colormap   = plt.colormaps.get_cmap("jet")
    heatmap_np = (colormap(cam_final)[:, :, :3] * 255).astype(np.uint8)

    orig_np = np.array(original_image.convert("RGB")).astype(np.float32)
    heat_np = heatmap_np.astype(np.float32)
    overlay = np.clip((1 - alpha) * orig_np + alpha * heat_np, 0, 255).astype(np.uint8)

    # ── Plot ──────────────────────────────────────────────────────────────
    fig, axes = plt.subplots(1, 2, figsize=(11, 5), facecolor="white")
    fig.patch.set_facecolor("white")

    axes[0].imshow(np.array(original_image.convert("RGB")), cmap="gray")
    axes[0].set_title("Original X-ray", color="#01579b",
                       fontsize=12, fontweight="bold", pad=10)
    axes[0].axis("off")

    axes[1].imshow(overlay)
    axes[1].set_title(
        f"Grad-CAM Heatmap  (score={score_val:.3f})\n"
        "Red = High activation  |  Blue = Low activation",
        color="#01579b", fontsize=11, fontweight="bold", pad=10
    )
    axes[1].axis("off")

    sm = plt.cm.ScalarMappable(cmap="jet", norm=plt.Normalize(0, 1))
    cb = fig.colorbar(sm, ax=axes[1], fraction=0.04, pad=0.02)
    cb.set_label("Activation", fontsize=9, color="#01579b")
    cb.ax.tick_params(colors="#01579b", labelsize=8)

    plt.tight_layout(pad=1.0)

    fig.canvas.draw()
    w_fig, h_fig = fig.canvas.get_width_height()
    try:
        buf = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8).reshape(h_fig, w_fig, 3)
    except AttributeError:
        buf = np.frombuffer(fig.canvas.buffer_rgba(), dtype=np.uint8).reshape(h_fig, w_fig, 4)[:, :, :3]
    plt.close(fig)

    return Image.fromarray(buf), score_val