"""
E-Score: A unified explainability quality score combining:
  1. Prediction Confidence  — model certainty
  2. Grad-CAM Activation    — how focused the heatmap is
  3. Feature Consistency    — gradient magnitude proxy

Formula:
    E = 0.4 * confidence + 0.3 * focus_score + 0.3 * gradient_score

Range: [0, 1] — Higher is better.

Reference: Custom metric inspired by Samek et al., 2016 and
           "Evaluating the Faithfulness of Saliency Maps" (Kindermans et al.)
"""

import torch
import torch.nn.functional as F
import numpy as np


def _gradient_score(model: torch.nn.Module, img_tensor: torch.Tensor) -> float:
    """
    Compute gradient-based score: mean L2 norm of input gradients.
    Higher = more informative gradients.
    """
    model.eval()
    inp = img_tensor.clone().requires_grad_(True)
    logits = model(inp)
    pred = torch.argmax(logits, dim=1)
    score = logits[0, pred].sum()
    model.zero_grad()
    score.backward()
    grad = inp.grad.data
    grad_norm = grad.norm(2).item()
    return float(np.clip(grad_norm / (grad_norm + 1.0), 0, 1))


def _focus_score(model: torch.nn.Module, img_tensor: torch.Tensor) -> float:
    """
    Measure how focused the last-layer activations are.
    A lower entropy = more focused = higher focus score.
    """
    model.eval()
    activations = []

    def hook_fn(m, i, o):
        activations.append(o.detach())

    target_layer = model.cnn.layer4[-1]
    handle = target_layer.register_forward_hook(hook_fn)

    with torch.no_grad():
        model(img_tensor)

    handle.remove()

    act = activations[0].squeeze()  # [C, H, W]
    act_map = act.mean(dim=0)       # [H, W]
    act_flat = act_map.flatten()
    act_flat = F.softmax(act_flat, dim=0)

    # Normalized entropy: 0 (most focused) → 1 (uniform)
    entropy = -(act_flat * (act_flat + 1e-8).log()).sum().item()
    max_entropy = float(np.log(act_flat.numel()))
    norm_entropy = entropy / (max_entropy + 1e-8)

    return float(1.0 - norm_entropy)  # Invert: higher = more focused


def e_score(
    model: torch.nn.Module,
    img_tensor: torch.Tensor,
    label: str,
) -> float:
    """
    Compute the E-Score for a given prediction.

    Args:
        model:       FusionModel (eval mode)
        img_tensor:  Preprocessed input tensor [1, 3, H, W]
        label:       Predicted label string ("Pneumonia" or "Normal")

    Returns:
        score: float in [0, 1]
    """
    model.eval()

    # 1. Confidence
    with torch.no_grad():
        logits = model(img_tensor)
        probs  = torch.softmax(logits, dim=1)
        confidence = probs.max().item()

    # 2. Focus score (Grad-CAM activation entropy)
    focus = _focus_score(model, img_tensor)

    # 3. Gradient magnitude score
    grad = _gradient_score(model, img_tensor)

    # Weighted combination
    score = 0.40 * confidence + 0.30 * focus + 0.30 * grad
    return float(np.clip(score, 0.0, 1.0))
