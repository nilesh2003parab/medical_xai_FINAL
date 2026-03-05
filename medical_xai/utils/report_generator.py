"""
report_generator.py
===================
Generates a downloadable PDF medical report including:
  - Patient demographics
  - Annotated X-ray image
  - Findings summary (zone + severity)
  - Explainability scores (Grad-CAM, LIME, SHAP, E-Score)
  - Full treatment protocol
  - Disclaimer

Uses: reportlab (pip install reportlab)
"""

import io
from datetime import datetime
from PIL import Image as PILImage

from reportlab.lib.pagesizes import A4
from reportlab.lib import colors
from reportlab.lib.units import mm, cm
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.enums import TA_CENTER, TA_LEFT, TA_RIGHT
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle,
    HRFlowable, Image as RLImage, KeepTogether,
)
from reportlab.lib.colors import HexColor


# ── Brand colors ─────────────────────────────────────────────────────────────
C_DARK    = HexColor("#0d1b2a")
C_BLUE    = HexColor("#1565c0")
C_LIGHT   = HexColor("#e8f4fd")
C_RED     = HexColor("#c0392b")
C_GREEN   = HexColor("#1a7f4b")
C_ORANGE  = HexColor("#e67e22")
C_YELLOW  = HexColor("#f39c12")
C_GREY    = HexColor("#546e7a")
C_LGREY   = HexColor("#eceff1")

SEVERITY_COLOR = {
    "Severe":   C_RED,
    "Moderate": C_ORANGE,
    "Mild":     C_YELLOW,
    "Normal":   C_GREEN,
}


def _pil_to_rl_image(pil_img: PILImage.Image, max_width: float, max_height: float) -> RLImage:
    """Convert PIL image to ReportLab Image object, fit within bounds."""
    buf = io.BytesIO()
    pil_img.save(buf, format="PNG")
    buf.seek(0)

    w, h = pil_img.size
    aspect = h / w
    if w > max_width:
        w = max_width
        h = w * aspect
    if h > max_height:
        h = max_height
        w = h / aspect

    return RLImage(buf, width=w, height=h)


def generate_pdf_report(
    patient_info: dict,
    label: str,
    confidence: float,
    escore: float,
    findings: list,
    treatment_plan,
    annotated_img: PILImage.Image,
    xai_scores: dict,
) -> bytes:
    """
    Generate a complete medical PDF report.

    Args:
        patient_info:   dict with keys: id, name, age, sex, ward, comorbidities
        label:          "Pneumonia" or "Normal"
        confidence:     Model confidence (0–1)
        escore:         E-Score value
        findings:       List of finding dicts from annotate_xray()
        treatment_plan: TreatmentPlan dataclass
        annotated_img:  PIL Image — annotated X-ray
        xai_scores:     dict with gradcam, lime, shap scores

    Returns:
        bytes: PDF file content
    """
    buf = io.BytesIO()
    doc = SimpleDocTemplate(
        buf,
        pagesize=A4,
        leftMargin=18*mm, rightMargin=18*mm,
        topMargin=15*mm,  bottomMargin=15*mm,
        title="MedXAI Radiology Report",
        author="MedXAI System",
    )

    W, H = A4
    content_w = W - 36*mm
    styles    = getSampleStyleSheet()

    # ── Custom styles ────────────────────────────────────────────────────────
    def style(name, parent="Normal", **kwargs):
        s = ParagraphStyle(name, parent=styles[parent], **kwargs)
        return s

    s_title     = style("S_Title",    fontSize=20, textColor=C_DARK,  fontName="Helvetica-Bold", spaceAfter=2)
    s_subtitle  = style("S_Sub",      fontSize=10, textColor=C_GREY,  spaceAfter=6)
    s_h1        = style("S_H1",       fontSize=13, textColor=C_BLUE,  fontName="Helvetica-Bold", spaceBefore=8, spaceAfter=4)
    s_h2        = style("S_H2",       fontSize=11, textColor=C_DARK,  fontName="Helvetica-Bold", spaceBefore=4, spaceAfter=3)
    s_body      = style("S_Body",     fontSize=9,  textColor=C_DARK,  leading=14)
    s_small     = style("S_Small",    fontSize=8,  textColor=C_GREY,  leading=12)
    s_center    = style("S_Center",   fontSize=9,  alignment=TA_CENTER)
    s_warn      = style("S_Warn",     fontSize=8,  textColor=C_RED,   fontName="Helvetica-Bold")
    s_bullet    = style("S_Bullet",   fontSize=9,  leftIndent=12,     leading=14, bulletIndent=4, textColor=C_DARK)

    story = []

    # ══════════════════════════════════════════════════════════════════════════
    # HEADER
    # ══════════════════════════════════════════════════════════════════════════
    header_data = [[
        Paragraph("<b>🫁 MedXAI</b>", style("hd", fontSize=18, textColor=colors.white, fontName="Helvetica-Bold")),
        Paragraph(
            f"<b>AI Radiology Report</b><br/>"
            f"<font size='8'>Generated: {datetime.now().strftime('%d %b %Y, %H:%M')}</font>",
            style("hdr", fontSize=10, textColor=colors.white, alignment=TA_RIGHT)
        ),
    ]]
    header_tbl = Table(header_data, colWidths=[content_w * 0.5, content_w * 0.5])
    header_tbl.setStyle(TableStyle([
        ("BACKGROUND",  (0, 0), (-1, -1), C_DARK),
        ("VALIGN",      (0, 0), (-1, -1), "MIDDLE"),
        ("LEFTPADDING", (0, 0), (-1, -1), 12),
        ("RIGHTPADDING",(0, 0), (-1, -1), 12),
        ("TOPPADDING",  (0, 0), (-1, -1), 10),
        ("BOTTOMPADDING",(0,0), (-1, -1), 10),
        ("ROUNDEDCORNERS", (0, 0), (-1, -1), 6),
    ]))
    story.append(header_tbl)
    story.append(Spacer(1, 6*mm))

    # ══════════════════════════════════════════════════════════════════════════
    # PATIENT INFORMATION + PREDICTION (side by side)
    # ══════════════════════════════════════════════════════════════════════════
    pred_color = C_RED if label == "Pneumonia" else C_GREEN
    pred_icon  = "⚠" if label == "Pneumonia" else "✔"

    patient_rows = [
        ["Patient ID",   patient_info.get("id",   "—")],
        ["Full Name",    patient_info.get("name",  "—")],
        ["Age / Sex",    f"{patient_info.get('age', '—')} yrs / {patient_info.get('sex', '—')}"],
        ["Ward",         patient_info.get("ward",  "—")],
        ["Comorbidities",patient_info.get("comorbidities", "None")],
        ["Referring Clinician", "—"],
    ]
    pat_tbl = Table([[Paragraph(k, s_h2), Paragraph(str(v), s_body)] for k, v in patient_rows],
                    colWidths=[content_w * 0.38 * 0.4, content_w * 0.38 * 0.6])
    pat_tbl.setStyle(TableStyle([
        ("GRID",        (0, 0), (-1, -1), 0.3, C_LGREY),
        ("BACKGROUND",  (0, 0), (0, -1), C_LGREY),
        ("TOPPADDING",  (0, 0), (-1, -1), 4),
        ("BOTTOMPADDING",(0,0), (-1, -1), 4),
        ("LEFTPADDING", (0, 0), (-1, -1), 6),
    ]))

    pred_data = [
        [Paragraph(f"<b>{pred_icon} {label.upper()}</b>",
                   style("prd", fontSize=16, textColor=colors.white, fontName="Helvetica-Bold", alignment=TA_CENTER))],
        [Paragraph(f"Confidence: <b>{confidence:.1%}</b>",
                   style("prc", fontSize=10, textColor=colors.white, alignment=TA_CENTER))],
        [Paragraph(f"E-Score: <b>{escore:.4f}</b>",
                   style("pre", fontSize=10, textColor=colors.white, alignment=TA_CENTER))],
        [Paragraph(f"Abnormal zones: <b>{len(findings)}</b>",
                   style("prz", fontSize=9, textColor=C_LIGHT, alignment=TA_CENTER))],
    ]
    pred_tbl = Table(pred_data, colWidths=[content_w * 0.28])
    pred_tbl.setStyle(TableStyle([
        ("BACKGROUND",  (0, 0), (-1, -1), pred_color),
        ("TOPPADDING",  (0, 0), (-1, -1), 5),
        ("BOTTOMPADDING",(0,0), (-1, -1), 5),
        ("ROUNDEDCORNERS", (0,0), (-1,-1), 6),
    ]))

    xai_rows = [
        ["Method", "Score"],
        ["Grad-CAM", f"{xai_scores.get('gradcam', 0):.4f}"],
        ["LIME",     f"{xai_scores.get('lime', 0):.4f}"],
        ["SHAP",     f"{xai_scores.get('shap', 0):.4f}"],
        ["E-Score",  f"{escore:.4f}"],
    ]
    xai_tbl = Table(xai_rows, colWidths=[content_w * 0.16, content_w * 0.16])
    xai_tbl.setStyle(TableStyle([
        ("BACKGROUND",  (0, 0), (-1, 0), C_BLUE),
        ("TEXTCOLOR",   (0, 0), (-1, 0), colors.white),
        ("FONTNAME",    (0, 0), (-1, 0), "Helvetica-Bold"),
        ("FONTSIZE",    (0, 0), (-1, -1), 8),
        ("GRID",        (0, 0), (-1, -1), 0.3, C_LGREY),
        ("ALIGN",       (1, 0), (1, -1), "CENTER"),
        ("TOPPADDING",  (0, 0), (-1, -1), 3),
        ("BOTTOMPADDING",(0,0), (-1, -1), 3),
    ]))

    top_section = Table(
        [[pat_tbl, pred_tbl, xai_tbl]],
        colWidths=[content_w * 0.40, content_w * 0.30, content_w * 0.30],
    )
    top_section.setStyle(TableStyle([
        ("VALIGN",  (0, 0), (-1, -1), "TOP"),
        ("LEFTPADDING",  (1, 0), (-1, -1), 6),
    ]))
    story.append(top_section)
    story.append(Spacer(1, 5*mm))
    story.append(HRFlowable(width="100%", thickness=1, color=C_LGREY))
    story.append(Spacer(1, 3*mm))

    # ══════════════════════════════════════════════════════════════════════════
    # ANNOTATED X-RAY IMAGE
    # ══════════════════════════════════════════════════════════════════════════
    story.append(Paragraph("Annotated Chest X-Ray", s_h1))
    rl_img = _pil_to_rl_image(annotated_img, max_width=content_w * 0.65, max_height=9*cm)
    story.append(rl_img)
    story.append(Paragraph(
        "AI-generated annotation. Coloured zones indicate model-detected abnormal regions. "
        "Red = Severe, Orange = Moderate, Yellow = Mild. Always interpret alongside clinical findings.",
        s_small
    ))
    story.append(Spacer(1, 4*mm))

    # ══════════════════════════════════════════════════════════════════════════
    # FINDINGS TABLE
    # ══════════════════════════════════════════════════════════════════════════
    story.append(Paragraph("Detected Findings by Lung Zone", s_h1))

    if findings:
        f_header = ["Zone", "Severity", "Max Activation", "Mean Activation"]
        f_rows   = [f_header] + [
            [
                f["zone"],
                f["severity"],
                f"{f['activation']:.4f}",
                f"{f['mean_act']:.4f}",
            ]
            for f in sorted(findings, key=lambda x: x["activation"], reverse=True)
        ]
        f_tbl = Table(f_rows, colWidths=[
            content_w * 0.3, content_w * 0.2, content_w * 0.25, content_w * 0.25
        ])
        f_style = [
            ("BACKGROUND",  (0, 0), (-1, 0), C_DARK),
            ("TEXTCOLOR",   (0, 0), (-1, 0), colors.white),
            ("FONTNAME",    (0, 0), (-1, 0), "Helvetica-Bold"),
            ("FONTSIZE",    (0, 0), (-1, -1), 9),
            ("GRID",        (0, 0), (-1, -1), 0.3, C_LGREY),
            ("TOPPADDING",  (0, 0), (-1, -1), 4),
            ("BOTTOMPADDING",(0,0), (-1, -1), 4),
            ("LEFTPADDING", (0, 0), (-1, -1), 6),
            ("ALIGN",       (2, 0), (-1, -1), "CENTER"),
        ]
        # Color severity cells
        for i, f in enumerate(findings, start=1):
            sc = SEVERITY_COLOR.get(f["severity"], C_GREY)
            f_style.append(("BACKGROUND", (1, i), (1, i), sc))
            f_style.append(("TEXTCOLOR",  (1, i), (1, i), colors.white))
            f_style.append(("FONTNAME",   (1, i), (1, i), "Helvetica-Bold"))
        f_tbl.setStyle(TableStyle(f_style))
        story.append(f_tbl)
    else:
        story.append(Paragraph("No significant abnormalities detected in lung zones.", s_body))

    story.append(Spacer(1, 4*mm))
    story.append(HRFlowable(width="100%", thickness=1, color=C_LGREY))
    story.append(Spacer(1, 3*mm))

    # ══════════════════════════════════════════════════════════════════════════
    # TREATMENT PROTOCOL
    # ══════════════════════════════════════════════════════════════════════════
    story.append(Paragraph(f"Treatment Protocol — {treatment_plan.condition} ({treatment_plan.severity})", s_h1))
    story.append(Paragraph(f"<b>Prognosis:</b> {treatment_plan.prognosis}", s_body))
    story.append(Spacer(1, 3*mm))

    def section(title: str, items: list, color=C_BLUE):
        section_items = [
            [Paragraph(f"<b>{title}</b>",
                       style(f"sec_{title}", fontSize=10, textColor=colors.white,
                             fontName="Helvetica-Bold", leftPadding=6))]
        ]
        hdr = Table(section_items, colWidths=[content_w])
        hdr.setStyle(TableStyle([
            ("BACKGROUND",  (0, 0), (-1, -1), color),
            ("TOPPADDING",  (0, 0), (-1, -1), 4),
            ("BOTTOMPADDING",(0,0), (-1, -1), 4),
        ]))
        return hdr

    # Immediate actions
    story.append(KeepTogether([
        section("⚡ Immediate Actions", treatment_plan.immediate_actions, C_RED if label == "Pneumonia" else C_GREEN),
        *[Paragraph(f"• {a}", s_bullet) for a in treatment_plan.immediate_actions],
        Spacer(1, 2*mm),
    ]))

    # Medications
    story.append(section("💊 Medications", [], C_BLUE))
    med_rows = [["Medication", "Dose", "Duration"]]
    for m in treatment_plan.medications:
        med_rows.append([m["name"], m["dose"], m["duration"]])
    med_tbl = Table(med_rows, colWidths=[content_w * 0.38, content_w * 0.35, content_w * 0.27])
    med_tbl.setStyle(TableStyle([
        ("BACKGROUND",  (0, 0), (-1, 0), C_BLUE),
        ("TEXTCOLOR",   (0, 0), (-1, 0), colors.white),
        ("FONTNAME",    (0, 0), (-1, 0), "Helvetica-Bold"),
        ("FONTSIZE",    (0, 0), (-1, -1), 8),
        ("GRID",        (0, 0), (-1, -1), 0.3, C_LGREY),
        ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, C_LGREY]),
        ("TOPPADDING",  (0, 0), (-1, -1), 3),
        ("BOTTOMPADDING",(0,0), (-1, -1), 3),
        ("LEFTPADDING", (0, 0), (-1, -1), 5),
    ]))
    story.append(med_tbl)
    story.append(Spacer(1, 3*mm))

    # Investigations
    story.append(KeepTogether([
        section("🔬 Investigations to Order", treatment_plan.investigations, C_GREY),
        *[Paragraph(f"• {i}", s_bullet) for i in treatment_plan.investigations],
        Spacer(1, 2*mm),
    ]))

    # Lifestyle
    story.append(KeepTogether([
        section("🌿 Lifestyle & Supportive Care", treatment_plan.lifestyle, HexColor("#2e7d32")),
        *[Paragraph(f"• {l}", s_bullet) for l in treatment_plan.lifestyle],
        Spacer(1, 2*mm),
    ]))

    # Follow-up
    story.append(KeepTogether([
        section("📅 Follow-Up Plan", treatment_plan.follow_up, C_BLUE),
        *[Paragraph(f"• {f}", s_bullet) for f in treatment_plan.follow_up],
        Spacer(1, 2*mm),
    ]))

    # Red flags
    story.append(KeepTogether([
        section("🚨 Red Flags — When to Escalate", treatment_plan.red_flags, C_RED),
        *[Paragraph(f"• {r}", s_bullet) for r in treatment_plan.red_flags],
        Spacer(1, 4*mm),
    ]))

    # ══════════════════════════════════════════════════════════════════════════
    # DISCLAIMER
    # ══════════════════════════════════════════════════════════════════════════
    story.append(HRFlowable(width="100%", thickness=1, color=C_LGREY))
    story.append(Spacer(1, 3*mm))
    disclaimer = Table([[Paragraph(
        "<b>⚠️ DISCLAIMER:</b> This report is generated by an AI-assisted research system (MedXAI) "
        "for educational and research purposes only. It does NOT constitute a medical diagnosis or "
        "treatment recommendation. All findings must be reviewed and confirmed by a qualified "
        "clinician before any clinical action is taken. The AI model may produce errors. "
        "Clinical judgement supersedes AI output at all times.",
        style("disc", fontSize=7.5, textColor=C_RED)
    )]])
    disclaimer.setStyle(TableStyle([
        ("BACKGROUND",  (0, 0), (-1, -1), HexColor("#fff3f3")),
        ("BOX",         (0, 0), (-1, -1), 0.5, C_RED),
        ("TOPPADDING",  (0, 0), (-1, -1), 6),
        ("BOTTOMPADDING",(0,0), (-1, -1), 6),
        ("LEFTPADDING", (0, 0), (-1, -1), 8),
    ]))
    story.append(disclaimer)
    story.append(Spacer(1, 3*mm))
    story.append(Paragraph(
        f"MedXAI Research System  |  Report ID: {datetime.now().strftime('RPT-%Y%m%d-%H%M%S')}  |  "
        f"Model: ResNet18 FusionModel  |  For Research Use Only",
        style("footer", fontSize=7, textColor=C_GREY, alignment=TA_CENTER)
    ))

    doc.build(story)
    return buf.getvalue()
