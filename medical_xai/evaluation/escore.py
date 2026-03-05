"""
E-Score: A unified explainability quality score combining:
  1. Prediction Confidence  — model certainty
  2. Grad-CAM Focus Score   — how concentrated the activations are
  3. Gradient Score         — gradient magnitude proxy

Formula:
    E = 0.40 * confidence + 0.30 * focus_score + 0.30 * gradient_score

Range: [0, 1] — Higher is better.

Reference: Custom metric inspired by Samek et al., 2016 and
           "Evaluating the Faithfulness of Saliency Maps" (Kindermans et al.)
"""

import torch
import torch.nn.functional as F
import numpy as np


def _gradient_score(model: torch.nn.Module, img_tensor: torch.Tensor) -> float:
    """Compute gradient-based score: normalized L2 norm of input gradients."""
    model.eval()
    inp = img_tensor.clone().detach().requires_grad_(True)
    logits = model(inp)
    pred   = torch.argmax(logits, dim=1)
    score  = logits[0, pred].sum()
    model.zero_grad()
    score.backward()
    grad_norm = inp.grad.data.norm(2).item()
    return float(np.clip(grad_norm / (grad_norm + 1.0), 0, 1))


def _focus_score(model: torch.nn.Module, img_tensor: torch.Tensor) -> float:
    """
    Measure how focused the last-layer activations are.
    Lower entropy = more focused = higher score.
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

    if not activations:
        return 0.5

    act     = activations[0].squeeze()           # [C, H, W]
    act_map = act.mean(dim=0).flatten()           # [H*W]
    act_map = F.softmax(act_map, dim=0)

    entropy     = -(act_map * (act_map + 1e-8).log()).sum().item()
    max_entropy = float(np.log(act_map.numel()))
    norm_entropy = entropy / (max_entropy + 1e-8)

    return float(1.0 - norm_entropy)


def e_score(
    model: torch.nn.Module,
    img_tensor: torch.Tensor,
    label: str,
) -> float:
    """
    Compute the E-Score for a given prediction.

    Args:
        model:       FusionModel in eval mode.
        img_tensor:  Preprocessed input [1, 3, H, W].
        label:       Predicted label ("Pneumonia" or "Normal").

    Returns:
        score: float in [0, 1]
    """
    model.eval()

    # 1 — Prediction confidence
    with torch.no_grad():
        logits     = model(img_tensor)
        probs      = torch.softmax(logits, dim=1)
        confidence = probs.max().item()

    # 2 — Activation focus score
    focus = _focus_score(model, img_tensor)

    # 3 — Input gradient score
    grad = _gradient_score(model, img_tensor)

    score = 0.40 * confidence + 0.30 * focus + 0.30 * grad
    return float(np.clip(score, 0.0, 1.0))
