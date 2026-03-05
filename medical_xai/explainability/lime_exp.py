"""
LIME (Local Interpretable Model-agnostic Explanations) for image classification.

Reference: Ribeiro et al., 2016 — "Why Should I Trust You?"
"""

import numpy as np
from PIL import Image
import torch
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from lime import lime_image
from skimage.segmentation import mark_boundaries
from torchvision import transforms


def _predict_fn(images_np, model, transform, device):
    """
    Prediction function for LIME.
    Takes numpy array of shape [N, H, W, C], returns softmax probabilities [N, K].
    """
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


def run_lime(
    model: torch.nn.Module,
    image_np: np.ndarray,
    transform,
    num_samples: int = 500,
    num_features: int = 10,
    positive_only: bool = False,
) -> tuple[Image.Image, float]:
    """
    Run LIME explanation on a single image.

    Args:
        model:         Trained FusionModel
        image_np:      Original image as numpy array [H, W, 3], uint8
        transform:     Torchvision preprocessing transform
        num_samples:   Number of LIME perturbation samples
        num_features:  Number of superpixels to highlight
        positive_only: Show only positively contributing regions

    Returns:
        result_pil: PIL.Image with LIME boundaries
        lime_score: Confidence score from LIME explanation (0–1)
    """
    device = next(model.parameters()).device
    model.eval()

    explainer = lime_image.LimeImageExplainer(verbose=False)

    predict_wrapper = lambda imgs: _predict_fn(imgs, model, transform, device)

    explanation = explainer.explain_instance(
        image_np,
        predict_wrapper,
        top_labels=2,
        num_samples=num_samples,
        hide_color=0,
        random_seed=42,
    )

    # Get the top predicted label
    top_label = explanation.top_labels[0]

    # Get image with highlighted superpixels
    temp_img, mask = explanation.get_image_and_mask(
        top_label,
        positive_only=positive_only,
        num_features=num_features,
        hide_rest=False,
    )

    # Get local explanation weights → use as score
    local_exp   = explanation.local_exp[top_label]
    top_weights = [w for _, w in sorted(local_exp, key=lambda x: abs(x[1]), reverse=True)[:num_features]]
    lime_score  = float(np.clip(np.mean([abs(w) for w in top_weights]), 0, 1))

    # ── Visualization ────────────────────────────────────────────────────
    fig, axes = plt.subplots(1, 2, figsize=(10, 4), facecolor="#0d0d0d")

    axes[0].imshow(image_np, cmap="gray")
    axes[0].set_title("Original X-ray", color="white", fontsize=11)
    axes[0].axis("off")

    lime_display = mark_boundaries(temp_img / 255.0, mask, color=(1, 0.8, 0), mode="thick")
    axes[1].imshow(lime_display)
    axes[1].set_title(f"LIME Superpixels  (score={lime_score:.3f})", color="white", fontsize=11)
    axes[1].axis("off")

    # Legend
    pos_patch = mpatches.Patch(color=(1, 0.8, 0), label="Contributing regions")
    axes[1].legend(handles=[pos_patch], loc="lower right", fontsize=9,
                   facecolor="#1a1a1a", labelcolor="white")

    plt.tight_layout(pad=0.5)
    fig.canvas.draw()
    w, h = fig.canvas.get_width_height()
    try:
        buf = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8).reshape(h, w, 3)
    except AttributeError:
        buf = np.frombuffer(fig.canvas.buffer_rgba(), dtype=np.uint8).reshape(h, w, 4)[:, :, :3]
    plt.close(fig)

    return Image.fromarray(buf), lime_score
