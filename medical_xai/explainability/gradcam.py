"""
Grad-CAM (Gradient-weighted Class Activation Mapping) for ResNet18.
Reference: Selvaraju et al., 2017
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


def generate_gradcam(model, img_tensor, original_image, target_class=None, alpha=0.50):
    model.eval()
    target_layer = model.cnn.layer4[-1]
    hook = GradCAMHook(target_layer)

    img_tensor = img_tensor.clone().requires_grad_(True)
    logits = model(img_tensor)
    probs  = torch.softmax(logits, dim=1)

    if target_class is None:
        target_class = torch.argmax(probs, dim=1).item()

    score_val = probs[0, target_class].item()

    model.zero_grad()
    logits[0, target_class].backward()

    grads   = hook.gradients
    acts    = hook.activations
    hook.remove()

    weights = grads.mean(dim=(2, 3), keepdim=True)
    cam     = F.relu((weights * acts).sum(dim=1, keepdim=True))

    cam_np = cam.squeeze().cpu().numpy()
    if cam_np.max() > 0:
        cam_np = cam_np / cam_np.max()

    orig_w, orig_h = original_image.size
    cam_resized = cv2.resize(cam_np, (orig_w, orig_h), interpolation=cv2.INTER_LINEAR)

    # Smooth for cleaner look
    cam_resized = cv2.GaussianBlur(cam_resized, (11, 11), 0)
    if cam_resized.max() > 0:
        cam_resized = cam_resized / cam_resized.max()

    import matplotlib.pyplot as plt
    colormap   = plt.colormaps.get_cmap("jet")
    heatmap_np = (colormap(cam_resized)[:, :, :3] * 255).astype(np.uint8)

    orig_np = np.array(original_image.convert("RGB")).astype(np.float32)
    heat_np = heatmap_np.astype(np.float32)
    overlay = np.clip((1 - alpha) * orig_np + alpha * heat_np, 0, 255).astype(np.uint8)

    # ── Plot — clean white background ────────────────────────────────────
    fig, axes = plt.subplots(1, 2, figsize=(11, 5), facecolor="white")
    fig.patch.set_facecolor("white")

    axes[0].imshow(np.array(original_image.convert("RGB")), cmap="gray")
    axes[0].set_title("Original X-ray", color="#01579b", fontsize=12, fontweight="bold", pad=10)
    axes[0].axis("off")

    axes[1].imshow(overlay)
    axes[1].set_title(f"Grad-CAM Heatmap  (score={score_val:.3f})\nRed = High activation  |  Blue = Low activation",
                      color="#01579b", fontsize=11, fontweight="bold", pad=10)
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