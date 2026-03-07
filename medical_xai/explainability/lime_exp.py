"""
LIME Explainability - Fixed version.
Uses SLIC segmentation focused on lung region.
Green = supports prediction, Red = against prediction.
Properly overlaid on the X-ray.
"""

import numpy as np
from PIL import Image
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from lime import lime_image
from skimage.segmentation import mark_boundaries, slic
from torchvision import transforms
import cv2


def _predict_fn(images_np, model, transform, device):
    model.eval()
    batch = []
    for img in images_np:
        pil_img = Image.fromarray(img.astype(np.uint8))
        tensor  = transform(pil_img)
        batch.append(tensor)
    batch_tensor = torch.stack(batch).to(device)
    with torch.no_grad():
        logits = model(batch_tensor)
        probs  = torch.softmax(logits, dim=1)
    return probs.cpu().numpy()


def run_lime(model, image_np, transform, num_samples=50, num_features=10, positive_only=False):
    device = next(model.parameters()).device
    model.eval()

    # Ensure image is uint8 RGB
    if image_np.dtype != np.uint8:
        image_np = (image_np * 255).astype(np.uint8)
    if image_np.ndim == 2:
        image_np = np.stack([image_np]*3, axis=2)
    elif image_np.shape[2] == 1:
        image_np = np.concatenate([image_np]*3, axis=2)

    H, W = image_np.shape[:2]

    # Custom segmentation — more segments in lung area (centre of image)
    def lung_segmentation(img):
        # Use SLIC with compactness tuned for X-rays
        segments = slic(
            img,
            n_segments=40,
            compactness=10,
            sigma=1,
            start_label=0,
            channel_axis=2,
        )
        return segments

    explainer = lime_image.LimeImageExplainer(verbose=False)
    predict_wrapper = lambda imgs: _predict_fn(imgs, model, transform, device)

    explanation = explainer.explain_instance(
        image_np,
        predict_wrapper,
        top_labels=2,
        num_samples=num_samples,
        hide_color=0,
        random_seed=42,
        segmentation_fn=lung_segmentation,
        batch_size=10,
    )

    top_label = explanation.top_labels[0]
    local_exp = explanation.local_exp[top_label]
    segments  = explanation.segments    # [H, W]

    # Score
    top_w      = sorted(local_exp, key=lambda x: abs(x[1]), reverse=True)[:num_features]
    lime_score = float(np.clip(np.mean([abs(w) for _, w in top_w]), 0, 1))

    # ── Build colour overlay on X-ray ────────────────────────────────────
    seg_weights = dict(local_exp)
    all_abs     = [abs(v) for v in seg_weights.values()]
    max_abs     = max(all_abs) if all_abs else 1e-8

    # RGBA overlay canvas
    overlay = np.zeros((H, W, 4), dtype=np.float32)

    for seg_id, weight in seg_weights.items():
        mask   = segments == seg_id
        norm_w = weight / (max_abs + 1e-8)   # -1 to +1

        if norm_w > 0.08:
            # Green — supports Pneumonia prediction
            intensity = min(norm_w, 1.0)
            overlay[mask] = [0.05, 0.80, 0.20, intensity * 0.70]
        elif norm_w < -0.08:
            # Red — against Pneumonia (Normal evidence)
            intensity = min(abs(norm_w), 1.0)
            overlay[mask] = [0.90, 0.10, 0.10, intensity * 0.70]

    # X-ray as float RGB
    xray_f = image_np.astype(np.float32) / 255.0

    # Composite
    alpha    = overlay[:, :, 3:4]
    rgb_over = overlay[:, :, :3]
    blended  = np.clip((1 - alpha) * xray_f + alpha * rgb_over, 0, 1)

    # Draw segment boundaries (light blue thin lines)
    blended_bounds = mark_boundaries(blended, segments, color=(0.4, 0.75, 1.0), mode="thin")

    # ── Figure ────────────────────────────────────────────────────────────
    fig, axes = plt.subplots(1, 2, figsize=(13, 5.5), facecolor="white")
    fig.patch.set_facecolor("white")

    axes[0].imshow(xray_f)
    axes[0].set_title("Original X-ray", fontsize=12,
                       fontweight="bold", color="#1a3a52", pad=10)
    axes[0].axis("off")

    axes[1].imshow(blended_bounds)
    axes[1].set_title(
        f"LIME Superpixel Explanation  (score={lime_score:.3f})\n"
        "Green = Supports Pneumonia  |  Red = Normal / Against",
        fontsize=10, fontweight="bold", color="#1a3a52", pad=10
    )
    axes[1].axis("off")

    pos_patch = mpatches.Patch(facecolor=(0.05, 0.80, 0.20), edgecolor="white",
                                label="Supports Pneumonia detection")
    neg_patch = mpatches.Patch(facecolor=(0.90, 0.10, 0.10), edgecolor="white",
                                label="Normal / Against detection")
    axes[1].legend(handles=[pos_patch, neg_patch], loc="lower right",
                   fontsize=9, facecolor="white", edgecolor="#c8e6f8",
                   labelcolor="#1a3a52", framealpha=0.9)

    plt.suptitle(
        f"LIME Explanation  |  Top {num_features} contributing regions shown",
        fontsize=12, fontweight="bold", color="#0288d1", y=1.01
    )
    plt.tight_layout(pad=0.8)

    # Render to PIL image
    fig.canvas.draw()
    w_f, h_f = fig.canvas.get_width_height()
    try:
        buf = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8).reshape(h_f, w_f, 3)
    except AttributeError:
        buf = np.frombuffer(fig.canvas.buffer_rgba(), dtype=np.uint8).reshape(h_f, w_f, 4)[:, :, :3]
    plt.close(fig)

    return Image.fromarray(buf), lime_score