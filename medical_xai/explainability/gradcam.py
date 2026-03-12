"""
Grad-CAM (Gradient-weighted Class Activation Mapping) for ResNet18.
Reference: Selvaraju et al., 2017
Uses multiple layers for better coverage of lung regions.
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


def _compute_cam(hook, orig_w, orig_h):
    """Compute normalised CAM from hook activations and gradients."""
    grads = hook.gradients   # [1, C, H, W]
    acts  = hook.activations # [1, C, H, W]

    # Grad-CAM weights = global average pooled gradients
    weights = grads.mean(dim=(2, 3), keepdim=True)
    cam     = F.relu((weights * acts).sum(dim=1, keepdim=True))
    cam_np  = cam.squeeze().cpu().numpy()

    if cam_np.max() > 0:
        cam_np = cam_np / cam_np.max()

    # Resize to image dimensions
    cam_r = cv2.resize(cam_np, (orig_w, orig_h), interpolation=cv2.INTER_CUBIC)
    return cam_r


def generate_gradcam(model, img_tensor, original_image, target_class=None, alpha=0.55):
    model.eval()

    # Hook BOTH layer3 and layer4 — combine for better lung coverage
    hook3 = GradCAMHook(model.cnn.layer3[-1])
    hook4 = GradCAMHook(model.cnn.layer4[-1])

    img_tensor = img_tensor.clone().requires_grad_(True)
    logits = model(img_tensor)
    probs  = torch.softmax(logits, dim=1)

    if target_class is None:
        target_class = torch.argmax(probs, dim=1).item()

    score_val = probs[0, target_class].item()

    model.zero_grad()
    logits[0, target_class].backward()

    orig_w, orig_h = original_image.size

    cam3 = _compute_cam(hook3, orig_w, orig_h)
    cam4 = _compute_cam(hook4, orig_w, orig_h)
    hook3.remove()
    hook4.remove()

    # Combine: layer4 = fine details, layer3 = broader regions
    cam_combined = 0.4 * cam3 + 0.6 * cam4

    # Normalise combined map
    if cam_combined.max() > 0:
        cam_combined = cam_combined / cam_combined.max()

    # Smooth — larger kernel for cleaner lung-region heatmap
    cam_smooth = cv2.GaussianBlur(cam_combined, (21, 21), 0)
    if cam_smooth.max() > 0:
        cam_smooth = cam_smooth / cam_smooth.max()

    # Apply power to sharpen contrast (0.7 = mild boost to bright areas)
    cam_final = np.power(cam_smooth, 0.7)

    # Build heatmap overlay
    import matplotlib.pyplot as plt
    colormap   = plt.colormaps.get_cmap("jet")
    heatmap_np = (colormap(cam_final)[:, :, :3] * 255).astype(np.uint8)

    orig_np  = np.array(original_image.convert("RGB")).astype(np.float32)
    heat_np  = heatmap_np.astype(np.float32)
    overlay  = np.clip((1 - alpha) * orig_np + alpha * heat_np, 0, 255).astype(np.uint8)

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