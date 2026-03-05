"""
Doctor feedback module.
Returns a PIL image (placeholder / icon) and a clinical text recommendation
based on model prediction label.
"""

from PIL import Image, ImageDraw, ImageFont
import numpy as np


# ── Clinical feedback text ──────────────────────────────────────────────────

FEEDBACK = {
    "Pneumonia": {
        "color": (220, 53, 69),   # Bootstrap red
        "title": "⚠️  Pneumonia Detected",
        "lines": [
            "AI model suggests pneumonia pattern in X-ray.",
            "Recommend: Confirm with physical examination.",
            "Consider: CBC, CRP, Sputum culture.",
            "Treatment: Antibiotics if bacterial (e.g. Amoxicillin).",
            "Follow-up: Repeat X-ray in 4–6 weeks post treatment.",
            "Note: Clinical judgement overrides AI prediction.",
        ],
    },
    "Normal": {
        "color": (40, 167, 69),   # Bootstrap green
        "title": "✅  No Pneumonia Detected",
        "lines": [
            "AI model suggests normal lung appearance.",
            "No immediate radiological concern identified.",
            "Recommend: Clinical correlation with symptoms.",
            "If symptomatic, consider further evaluation.",
            "Routine follow-up as per clinical protocol.",
            "Note: AI is a decision-support tool only.",
        ],
    },
}


def _make_feedback_image(label: str) -> Image.Image:
    """Generate a styled PIL image card for the given label."""
    info = FEEDBACK.get(label, FEEDBACK["Normal"])
    color = info["color"]

    width, height = 560, 220
    img = Image.new("RGB", (width, height), color=(248, 249, 250))
    draw = ImageDraw.Draw(img)

    # Header bar
    draw.rectangle([0, 0, width, 48], fill=color)

    # Try to load a font; fall back to default
    try:
        font_title = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 16)
        font_body  = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 13)
    except Exception:
        font_title = ImageFont.load_default()
        font_body  = ImageFont.load_default()

    draw.text((14, 14), info["title"], fill=(255, 255, 255), font=font_title)

    # Body lines
    y = 58
    for line in info["lines"]:
        draw.text((14, y), line, fill=(33, 37, 41), font=font_body)
        y += 24

    # Border
    draw.rectangle([0, 0, width - 1, height - 1], outline=color, width=2)

    return img


def doctor_feedback(label: str):
    """
    Returns:
        img  (PIL.Image): Visual feedback card
        text (str):       Plain text feedback summary
    """
    info = FEEDBACK.get(label, FEEDBACK["Normal"])
    img  = _make_feedback_image(label)
    text = info["title"] + "\n" + "\n".join(info["lines"])
    return img, text
