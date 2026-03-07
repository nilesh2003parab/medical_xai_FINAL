"""
image_quality.py
================
Detects image quality issues and visual outliers in chest X-ray images.

Checks performed:
  1. Blur Detection        — Laplacian variance (low = blurry)
  2. Brightness/Exposure   — Mean pixel intensity (too dark or overexposed)
  3. Contrast Check        — Standard deviation of pixel values
  4. Noise Detection       — High-frequency noise via FFT
  5. Spot/Artifact Detection — Isolated bright/dark blobs (foreign objects, dust, film artifacts)
  6. Dent/Scratch Detection — Edge anomalies via Canny + Hough lines
  7. Obstruction Check     — Large uniform dark/bright regions (finger, clothing, lead apron)

Each issue returns:
  - issue_type  : str   (name)
  - severity    : str   ("Critical", "Warning", "Info")
  - description : str   (plain English explanation)
  - region      : tuple (x, y, w, h) or None  (where on the image)
  - confidence  : float (0–1)

Also produces an annotated PIL image showing all detected issues.
"""

import numpy as np
import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from PIL import Image, ImageDraw, ImageFont
from dataclasses import dataclass, field
from typing import List, Optional, Tuple


# ─────────────────────────────────────────────────────────────────────────────
# DATA STRUCTURES
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class ImageIssue:
    issue_type:  str
    severity:    str          # "Critical", "Warning", "Info"
    description: str
    region:      Optional[Tuple[int, int, int, int]] = None  # (x, y, w, h)
    confidence:  float = 1.0
    value:       Optional[float] = None   # measured value


@dataclass
class QualityReport:
    overall_quality: str          # "Good", "Acceptable", "Poor", "Unusable"
    quality_score:   float        # 0–1 (1 = perfect)
    is_reliable:     bool         # True if model result can be trusted
    issues:          List[ImageIssue] = field(default_factory=list)
    summary:         str = ""


# ─────────────────────────────────────────────────────────────────────────────
# THRESHOLDS (tuned for 224×224 chest X-rays)
# ─────────────────────────────────────────────────────────────────────────────

BLUR_CRITICAL  = 30.0    # Laplacian variance — below this = very blurry
BLUR_WARNING   = 80.0    # below this = slightly blurry
BRIGHT_LOW     = 30      # mean pixel < this = too dark
BRIGHT_HIGH    = 220     # mean pixel > this = overexposed
CONTRAST_LOW   = 20      # std dev < this = flat/no contrast
NOISE_HIGH     = 0.35    # FFT high-freq ratio > this = noisy
SPOT_MIN_AREA  = 80      # min blob area (pixels²) to flag
SPOT_MAX_AREA  = 3000    # max (above this = obstruction, not spot)
OBSTRUCT_RATIO = 0.12    # fraction of image that is uniformly bright/dark


# ─────────────────────────────────────────────────────────────────────────────
# INDIVIDUAL CHECKS
# ─────────────────────────────────────────────────────────────────────────────

def _check_blur(gray: np.ndarray) -> Optional[ImageIssue]:
    """Laplacian variance blur detection."""
    lap_var = cv2.Laplacian(gray, cv2.CV_64F).var()
    if lap_var < BLUR_CRITICAL:
        return ImageIssue(
            issue_type="Severe Blur",
            severity="Critical",
            description=f"Image is severely blurred (sharpness={lap_var:.1f}). "
                        "Model results are unreliable. Please retake the X-ray.",
            confidence=1.0,
            value=lap_var,
        )
    elif lap_var < BLUR_WARNING:
        return ImageIssue(
            issue_type="Mild Blur",
            severity="Warning",
            description=f"Image appears slightly blurry (sharpness={lap_var:.1f}). "
                        "Consider retaking for better accuracy.",
            confidence=0.8,
            value=lap_var,
        )
    return None


def _check_brightness(gray: np.ndarray) -> Optional[ImageIssue]:
    """Exposure / brightness check."""
    mean_val = float(gray.mean())
    if mean_val < BRIGHT_LOW:
        return ImageIssue(
            issue_type="Underexposed",
            severity="Critical",
            description=f"Image is too dark (mean brightness={mean_val:.1f}/255). "
                        "Lung fields are not visible. X-ray is underexposed.",
            confidence=0.95,
            value=mean_val,
        )
    elif mean_val > BRIGHT_HIGH:
        return ImageIssue(
            issue_type="Overexposed",
            severity="Critical",
            description=f"Image is overexposed (mean brightness={mean_val:.1f}/255). "
                        "Detail is washed out. Pathology may be hidden.",
            confidence=0.95,
            value=mean_val,
        )
    return None


def _check_contrast(gray: np.ndarray) -> Optional[ImageIssue]:
    """Contrast check via standard deviation."""
    std_val = float(gray.std())
    if std_val < CONTRAST_LOW:
        return ImageIssue(
            issue_type="Low Contrast",
            severity="Warning",
            description=f"Image has very low contrast (std={std_val:.1f}). "
                        "Subtle pathologies may not be detectable.",
            confidence=0.85,
            value=std_val,
        )
    return None


def _check_noise(gray: np.ndarray) -> Optional[ImageIssue]:
    """High-frequency noise via FFT magnitude spectrum."""
    f = np.fft.fft2(gray.astype(np.float32))
    fshift = np.fft.fftshift(f)
    magnitude = np.abs(fshift)

    h, w = gray.shape
    total_energy = magnitude.sum() + 1e-8

    # High-frequency region: outer 30% of spectrum
    cy, cx = h // 2, w // 2
    radius  = int(min(h, w) * 0.35)
    mask    = np.zeros_like(gray, dtype=np.uint8)
    cv2.circle(mask, (cx, cy), radius, 255, -1)
    hf_energy = magnitude[mask == 0].sum()
    hf_ratio  = float(hf_energy / total_energy)

    if hf_ratio > NOISE_HIGH:
        return ImageIssue(
            issue_type="High Noise",
            severity="Warning",
            description=f"Image contains significant noise (HF ratio={hf_ratio:.3f}). "
                        "This may introduce false activations in the AI model.",
            confidence=0.75,
            value=hf_ratio,
        )
    return None


def _check_spots_and_artifacts(gray: np.ndarray, original_size: Tuple) -> List[ImageIssue]:
    """
    Detect small isolated bright/dark blobs — film artifacts, dust, foreign bodies.
    Returns a list of issues (one per detected region).
    """
    issues = []
    H, W   = gray.shape
    scale_x = original_size[0] / W
    scale_y = original_size[1] / H

    # ── Bright spots (foreign objects, film marks) ────────────────────────
    _, bright_thresh = cv2.threshold(gray, 245, 255, cv2.THRESH_BINARY)
    bright_thresh    = cv2.morphologyEx(bright_thresh, cv2.MORPH_OPEN,
                                        np.ones((3, 3), np.uint8))
    bright_contours, _ = cv2.findContours(bright_thresh, cv2.RETR_EXTERNAL,
                                           cv2.CHAIN_APPROX_SIMPLE)
    for cnt in bright_contours:
        area = cv2.contourArea(cnt)
        if SPOT_MIN_AREA < area < SPOT_MAX_AREA:
            x, y, bw, bh = cv2.boundingRect(cnt)
            issues.append(ImageIssue(
                issue_type="Bright Artifact / Spot",
                severity="Warning",
                description=f"Bright spot detected — possible film artifact, foreign body, "
                            f"or jewellery artifact (area={area:.0f}px²). "
                            "May cause false AI activation in this region.",
                region=(int(x * scale_x), int(y * scale_y),
                        int(bw * scale_x), int(bh * scale_y)),
                confidence=0.80,
                value=area,
            ))

    # ── Dark spots (film damage, grid artifacts, dents) ──────────────────
    dark_thresh = cv2.adaptiveThreshold(
        gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 21, 8
    )
    # Remove large regions (these are normal lung/bone structures)
    kernel     = np.ones((5, 5), np.uint8)
    dark_clean = cv2.morphologyEx(dark_thresh, cv2.MORPH_OPEN, kernel)
    dark_clean = cv2.morphologyEx(dark_clean, cv2.MORPH_CLOSE, kernel)

    dark_contours, _ = cv2.findContours(dark_clean, cv2.RETR_EXTERNAL,
                                         cv2.CHAIN_APPROX_SIMPLE)
    # Filter by circularity + area (actual spots are roughly circular)
    for cnt in dark_contours:
        area = cv2.contourArea(cnt)
        if area < SPOT_MIN_AREA or area > SPOT_MAX_AREA:
            continue
        perimeter = cv2.arcLength(cnt, True)
        if perimeter == 0:
            continue
        circularity = 4 * np.pi * area / (perimeter ** 2)
        if circularity > 0.35:  # Fairly circular = spot/dent
            x, y, bw, bh = cv2.boundingRect(cnt)
            issues.append(ImageIssue(
                issue_type="Dark Spot / Dent",
                severity="Warning",
                description=f"Dark circular artifact detected (area={area:.0f}px², "
                            f"circularity={circularity:.2f}). Could be a film dent, "
                            "processing artifact, or dust on sensor. "
                            "Verify it is not a real pathology.",
                region=(int(x * scale_x), int(y * scale_y),
                        int(bw * scale_x), int(bh * scale_y)),
                confidence=float(circularity),
                value=area,
            ))

    return issues


def _check_obstruction(gray: np.ndarray) -> List[ImageIssue]:
    """
    Detect large uniform obstructions — fingers at edge, clothing, lead aprons,
    or large portions of the image being cut off.
    """
    issues = []
    H, W   = gray.shape

    # Check borders for large uniform dark/bright bands
    band_h = int(H * 0.12)
    bands  = {
        "Top edge":    gray[:band_h, :],
        "Bottom edge": gray[H - band_h:, :],
        "Left edge":   gray[:, :band_h],
        "Right edge":  gray[:, W - band_h:],
    }
    for name, region in bands.items():
        mean_v = float(region.mean())
        std_v  = float(region.std())
        if std_v < 8 and (mean_v < 25 or mean_v > 235):
            color = "dark" if mean_v < 25 else "bright"
            issues.append(ImageIssue(
                issue_type=f"Edge Obstruction ({name})",
                severity="Warning",
                description=f"{name} is uniformly {color} (mean={mean_v:.0f}, std={std_v:.1f}). "
                            "Possible collimation cut-off, finger, or clothing obstruction.",
                confidence=0.85,
                value=mean_v,
            ))

    # Check for large uniform block in centre (rare but possible)
    center = gray[H // 4: 3 * H // 4, W // 4: 3 * W // 4]
    if center.std() < 5:
        issues.append(ImageIssue(
            issue_type="Central Obstruction",
            severity="Critical",
            description="Central lung region has almost no variation — "
                        "image may be completely obstructed or corrupted.",
            confidence=0.9,
            value=float(center.std()),
        ))

    return issues


def _check_scratch_lines(gray: np.ndarray) -> List[ImageIssue]:
    """Detect straight-line artifacts (scratches, grid lines, cassette damage)."""
    issues = []
    edges  = cv2.Canny(gray, 50, 150)

    # Probabilistic Hough lines
    lines  = cv2.HoughLinesP(edges, 1, np.pi / 180, threshold=80,
                              minLineLength=int(gray.shape[1] * 0.5),
                              maxLineGap=10)
    if lines is not None and len(lines) > 3:
        issues.append(ImageIssue(
            issue_type="Linear Artifacts / Scratches",
            severity="Warning",
            description=f"Detected {len(lines)} strong linear artifacts. "
                        "These may be cassette scratches, grid lines, "
                        "or compression artefacts from lossy encoding. "
                        "Verify image is a proper diagnostic X-ray.",
            confidence=0.70,
            value=float(len(lines)),
        ))
    return issues


# ─────────────────────────────────────────────────────────────────────────────
# MAIN QUALITY CHECK FUNCTION
# ─────────────────────────────────────────────────────────────────────────────

def check_image_quality(pil_image: Image.Image) -> QualityReport:
    """
    Run all quality checks on a PIL image.

    Args:
        pil_image: Original uploaded PIL image (any size, RGB)

    Returns:
        QualityReport with all detected issues and overall quality score
    """
    # Work at a standard resolution
    img_rgb  = pil_image.convert("RGB")
    img_small= img_rgb.resize((512, 512), Image.LANCZOS)
    gray     = np.array(img_small.convert("L"))
    orig_w, orig_h = pil_image.size

    issues: List[ImageIssue] = []

    # Run all checks
    for check_fn in [_check_blur, _check_brightness, _check_contrast, _check_noise]:
        result = check_fn(gray)
        if result:
            issues.append(result)

    issues += _check_spots_and_artifacts(gray, (orig_w, orig_h))
    issues += _check_obstruction(gray)
    issues += _check_scratch_lines(gray)

    # ── Compute quality score ─────────────────────────────────────────────
    critical_count = sum(1 for i in issues if i.severity == "Critical")
    warning_count  = sum(1 for i in issues if i.severity == "Warning")
    penalty = critical_count * 0.30 + warning_count * 0.10
    quality_score  = float(max(0.0, 1.0 - penalty))

    if critical_count >= 1:
        overall = "Unusable" if critical_count >= 2 else "Poor"
    elif warning_count >= 3:
        overall = "Poor"
    elif warning_count >= 1:
        overall = "Acceptable"
    else:
        overall = "Good"

    is_reliable = critical_count == 0 and quality_score >= 0.60

    # Summary text
    if not issues:
        summary = "No quality issues detected. Image is suitable for AI analysis."
    else:
        parts = []
        if critical_count:
            parts.append(f"{critical_count} critical issue(s) found")
        if warning_count:
            parts.append(f"{warning_count} warning(s)")
        summary = (f"Image quality: {overall}. " + ", ".join(parts) + ". " +
                   ("AI results may be unreliable." if not is_reliable else
                    "Proceed with caution."))

    return QualityReport(
        overall_quality=overall,
        quality_score=quality_score,
        is_reliable=is_reliable,
        issues=issues,
        summary=summary,
    )


# ─────────────────────────────────────────────────────────────────────────────
# ANNOTATED IMAGE GENERATOR
# ─────────────────────────────────────────────────────────────────────────────

SEVERITY_COLORS_DRAW = {
    "Critical": (220, 53,  69),   # Red
    "Warning":  (255, 140,  0),   # Orange
    "Info":     ( 23, 162, 184),  # Cyan
}

def annotate_quality_issues(
    pil_image: Image.Image,
    report: QualityReport,
) -> Image.Image:
    """
    Draw quality issue annotations on the image.

    Args:
        pil_image: Original PIL image
        report:    QualityReport from check_image_quality()

    Returns:
        Annotated PIL image with colored boxes + labels + quality banner
    """
    W, H    = pil_image.size
    base    = pil_image.convert("RGBA")
    overlay = Image.new("RGBA", (W, H), (0, 0, 0, 0))
    draw    = ImageDraw.Draw(overlay)

    try:
        font_bold  = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", max(11, H // 32))
        font_small = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",      max(9,  H // 40))
    except Exception:
        font_bold  = ImageFont.load_default()
        font_small = ImageFont.load_default()

    # ── Draw bounding boxes for issues with known regions ─────────────────
    for issue in report.issues:
        if issue.region is None:
            continue
        x, y, bw, bh = issue.region
        color   = SEVERITY_COLORS_DRAW.get(issue.severity, (100, 100, 100))
        alpha_c = (*color, 200)
        fill_c  = (*color, 25)

        # Dashed-style border (draw 4 rectangles slightly offset)
        draw.rectangle([x, y, x + bw, y + bh], outline=alpha_c, width=2)
        draw.rectangle([x + 2, y + 2, x + bw - 2, y + bh - 2],
                       fill=fill_c, outline=None)

        # Corner markers
        corner_len = max(8, min(bw, bh) // 4)
        for cx, cy, dx, dy in [
            (x, y, 1, 1), (x + bw, y, -1, 1),
            (x, y + bh, 1, -1), (x + bw, y + bh, -1, -1),
        ]:
            draw.line([(cx, cy), (cx + dx * corner_len, cy)], fill=alpha_c, width=3)
            draw.line([(cx, cy), (cx, cy + dy * corner_len)], fill=alpha_c, width=3)

        # Label pill
        label_txt = f"⚠ {issue.issue_type}"
        tx = max(0, x)
        ty = max(0, y - max(18, H // 28) - 2)
        draw.rectangle([tx - 2, ty - 2, tx + 200, ty + max(16, H // 30)],
                       fill=(*color, 200))
        draw.text((tx + 2, ty), label_txt, fill=(255, 255, 255, 255), font=font_small)

    # ── Blur overlay — semi-transparent red vignette if blurry ────────────
    has_blur = any("Blur" in i.issue_type for i in report.issues)
    if has_blur:
        blur_overlay = Image.new("RGBA", (W, H), (220, 53, 69, 0))
        bd = ImageDraw.Draw(blur_overlay)
        # Radial vignette effect (concentric ellipses)
        for r in range(10, 0, -1):
            alpha = int((11 - r) * 3.5)
            bd.ellipse([W * (1 - r / 10) / 2, H * (1 - r / 10) / 2,
                        W - W * (1 - r / 10) / 2, H - H * (1 - r / 10) / 2],
                       outline=(220, 53, 69, alpha), width=max(1, W // 60))
        base = Image.alpha_composite(base, blur_overlay)

    # Composite annotation overlay
    annotated = Image.alpha_composite(base, overlay).convert("RGB")
    draw_final = ImageDraw.Draw(annotated)

    # ── Quality banner at top ─────────────────────────────────────────────
    banner_h = max(30, H // 16)
    quality_colors = {
        "Good":       (27,  94,  32),
        "Acceptable": (230, 119,   0),
        "Poor":       (183,  28,  28),
        "Unusable":   (74,   20,  140),
    }
    bc = quality_colors.get(report.overall_quality, (100, 100, 100))
    draw_final.rectangle([0, 0, W, banner_h], fill=bc)

    try:
        bf = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
                                max(10, banner_h - 10))
    except Exception:
        bf = ImageFont.load_default()

    q_icon   = {"Good": "✓", "Acceptable": "⚠", "Poor": "✗", "Unusable": "✗✗"}.get(
        report.overall_quality, "?")
    q_text   = (f"{q_icon}  Image Quality: {report.overall_quality}  |  "
                f"Score: {report.quality_score:.0%}  |  "
                f"Issues: {len(report.issues)}  |  "
                f"Reliable: {'YES' if report.is_reliable else 'NO – VERIFY RESULT'}")
    draw_final.text((8, (banner_h - max(10, banner_h - 10)) // 2), q_text,
                    fill=(255, 255, 255), font=bf)

    return annotated


# ─────────────────────────────────────────────────────────────────────────────
# QUALITY GAUGE — returns matplotlib figure
# ─────────────────────────────────────────────────────────────────────────────

def quality_gauge_figure(score: float):
    """Return a matplotlib gauge-style chart for the quality score."""
    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches
    from matplotlib.patches import FancyArrowPatch

    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(5, 2.8), facecolor="#0d1b2a")
    ax.set_facecolor("#0d1b2a")
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 5)
    ax.axis("off")

    # Segments
    segs = [
        (0.5, 2.5,  "#c0392b", "Poor"),
        (2.5, 5.0,  "#e67e22", "Acceptable"),
        (5.0, 7.5,  "#f1c40f", "Good"),
        (7.5, 9.5,  "#27ae60", "Excellent"),
    ]
    for x0, x1, color, lbl in segs:
        ax.barh(1, x1 - x0, left=x0, height=1.1, color=color, alpha=0.85)
        ax.text((x0 + x1) / 2, 0.35, lbl, color="white", fontsize=7,
                ha="center", va="center", fontweight="bold")

    # Needle
    needle_x = 0.5 + score * 9.0
    ax.annotate("", xy=(needle_x, 1.55), xytext=(needle_x, 2.8),
                arrowprops=dict(arrowstyle="-|>", color="white", lw=2))
    ax.text(needle_x, 3.0, f"{score:.0%}", color="white", fontsize=12,
            ha="center", fontweight="bold")
    ax.text(5, 4.3, "Image Quality Score", color="#8cb8d8", fontsize=10,
            ha="center", fontweight="bold")

    plt.tight_layout(pad=0.3)
    return fig