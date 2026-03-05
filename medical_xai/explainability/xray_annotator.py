"""
xray_annotator.py
=================
Detects abnormal regions in a chest X-ray using Grad-CAM activation maps,
then draws annotated bounding boxes with labels and severity indicators.

Returns:
  - Annotated PIL image with colored boxes + labels
  - List of finding dictionaries (zone, severity, confidence)
"""

import numpy as np
import cv2
from PIL import Image, ImageDraw, ImageFont
import torch
import torch.nn.functional as F


# ── Lung zone mapping ──────────────────────────────────────────────────────
# Divides image into 6 anatomical zones (Left/Right × Upper/Mid/Lower)
LUNG_ZONES = {
    "Right Upper": (0.0, 0.0, 0.5, 0.33),
    "Right Mid":   (0.0, 0.33, 0.5, 0.66),
    "Right Lower": (0.0, 0.66, 0.5, 1.0),
    "Left Upper":  (0.5, 0.0, 1.0, 0.33),
    "Left Mid":    (0.5, 0.33, 1.0, 0.66),
    "Left Lower":  (0.5, 0.66, 1.0, 1.0),
}

SEVERITY_COLORS = {
    "Severe":   (220, 53,  69,  200),   # Red
    "Moderate": (255, 140,  0,  200),   # Orange
    "Mild":     (255, 193,  7,  200),   # Yellow
    "Normal":   ( 40, 167, 69,  200),   # Green
}

SEVERITY_THRESHOLDS = [
    (0.70, "Severe"),
    (0.45, "Moderate"),
    (0.25, "Mild"),
    (0.00, "Normal"),
]


def _severity_from_activation(val: float) -> str:
    for thresh, label in SEVERITY_THRESHOLDS:
        if val >= thresh:
            return label
    return "Normal"


def _get_cam_map(model, img_tensor):
    """Extract Grad-CAM activation map as numpy array [H, W], values 0–1."""
    model.eval()
    activations, gradients = [], []

    def fwd_hook(m, i, o): activations.append(o.detach())
    def bwd_hook(m, gi, go): gradients.append(go[0].detach())

    layer = model.cnn.layer4[-1]
    h1 = layer.register_forward_hook(fwd_hook)
    h2 = layer.register_full_backward_hook(bwd_hook)

    inp = img_tensor.clone().requires_grad_(True)
    logits = model(inp)
    pred = torch.argmax(logits, dim=1).item()
    model.zero_grad()
    logits[0, pred].backward()

    h1.remove(); h2.remove()

    if not activations or not gradients:
        return np.zeros((7, 7)), pred, torch.softmax(logits, dim=1)[0, pred].item()

    acts = activations[0]   # [1, C, H, W]
    grads = gradients[0]    # [1, C, H, W]
    weights = grads.mean(dim=(2, 3), keepdim=True)
    cam = F.relu((weights * acts).sum(dim=1, keepdim=True))
    cam_np = cam.squeeze().cpu().numpy()
    if cam_np.max() > 0:
        cam_np = cam_np / cam_np.max()

    conf = torch.softmax(logits, dim=1)[0, pred].item()
    return cam_np, pred, conf


def annotate_xray(
    model,
    img_tensor: torch.Tensor,
    original_image: Image.Image,
    label: str,
    confidence: float,
    activation_threshold: float = 0.30,
) -> tuple[Image.Image, list[dict]]:
    """
    Annotate chest X-ray with detected abnormal regions.

    Args:
        model:                Trained FusionModel
        img_tensor:           Preprocessed tensor [1, 3, H, W]
        original_image:       Original PIL image
        label:                Predicted label ("Pneumonia" / "Normal")
        confidence:           Prediction confidence
        activation_threshold: Minimum Grad-CAM activation to flag as abnormal

    Returns:
        annotated_img: PIL image with bounding boxes + labels
        findings:      List of finding dicts with zone, severity, activation
    """
    W, H = original_image.size

    # Get Grad-CAM map
    cam_np, pred_class, _ = _get_cam_map(model, img_tensor)

    # Resize cam to original image size
    cam_full = cv2.resize(cam_np, (W, H), interpolation=cv2.INTER_LINEAR)

    # ── Create RGBA overlay canvas ────────────────────────────────────────
    base = original_image.convert("RGBA")
    overlay = Image.new("RGBA", (W, H), (0, 0, 0, 0))
    draw = ImageDraw.Draw(overlay)

    findings = []

    # Load fonts
    try:
        font_bold  = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", max(12, H // 30))
        font_small = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",      max(10, H // 36))
    except Exception:
        font_bold  = ImageFont.load_default()
        font_small = ImageFont.load_default()

    if label == "Normal":
        # Just draw a clean green border — no abnormalities
        draw.rectangle([4, 4, W - 4, H - 4], outline=(40, 167, 69, 220), width=3)
        annotated = Image.alpha_composite(base, overlay).convert("RGB")
        return annotated, []

    # ── Detect abnormal zones ─────────────────────────────────────────────
    for zone_name, (x1r, y1r, x2r, y2r) in LUNG_ZONES.items():
        x1 = int(x1r * W); y1 = int(y1r * H)
        x2 = int(x2r * W); y2 = int(y2r * H)

        # Mean activation in this zone
        region_act = cam_full[y1:y2, x1:x2]
        mean_act = float(region_act.mean())
        max_act  = float(region_act.max())

        if mean_act < activation_threshold * 0.5:
            continue  # Skip low-activity zones

        severity = _severity_from_activation(max_act)
        if severity == "Normal":
            continue

        color = SEVERITY_COLORS[severity]

        # Semi-transparent fill
        fill_color = (*color[:3], 35)
        draw.rectangle([x1, y1, x2, y2], fill=fill_color, outline=color[:3] + (220,), width=2)

        # Label background + text
        label_text = f"{zone_name}: {severity}"
        conf_text  = f"Act: {max_act:.2f}"

        # Small label box
        pad = 4
        tx, ty = x1 + 6, y1 + 6
        draw.rectangle([tx - pad, ty - pad, tx + 160, ty + 32], fill=(0, 0, 0, 160))
        draw.text((tx, ty),      label_text, fill=(*color[:3], 255), font=font_bold)
        draw.text((tx, ty + 17), conf_text,  fill=(200, 200, 200, 220), font=font_small)

        findings.append({
            "zone":       zone_name,
            "severity":   severity,
            "activation": round(max_act, 4),
            "mean_act":   round(mean_act, 4),
            "bbox":       (x1, y1, x2, y2),
        })

    # ── Draw CAM heatmap as faint overlay ────────────────────────────────
    import matplotlib.cm as cm
    colormap  = plt.colormaps.get_cmap("Reds")
    heat_rgba = (colormap(cam_full) * 255).astype(np.uint8)
    heat_rgba[:, :, 3] = (cam_full * 90).astype(np.uint8)  # Alpha from activation
    heat_pil  = Image.fromarray(heat_rgba, "RGBA")
    base      = Image.alpha_composite(base, heat_pil)

    # ── Composite annotation overlay ─────────────────────────────────────
    annotated = Image.alpha_composite(base, overlay).convert("RGB")

    # ── Add header bar ────────────────────────────────────────────────────
    draw_final = ImageDraw.Draw(annotated)
    header_h   = max(28, H // 18)
    bar_color  = (180, 30, 30) if label == "Pneumonia" else (30, 120, 60)
    draw_final.rectangle([0, 0, W, header_h], fill=bar_color)

    try:
        hfont = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", max(11, header_h - 8))
    except Exception:
        hfont = ImageFont.load_default()

    header_text = f"AI DETECTION: {label.upper()}  |  Confidence: {confidence:.1%}  |  Abnormal Zones: {len(findings)}"
    draw_final.text((8, 6), header_text, fill=(255, 255, 255), font=hfont)

    return annotated, findings
