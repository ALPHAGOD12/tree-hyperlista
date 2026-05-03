"""Generate presentation slides for Tree-HyperLISTA paper."""
from pptx import Presentation
from pptx.util import Inches, Pt, Emu
from pptx.dml.color import RGBColor
from pptx.enum.text import PP_ALIGN, MSO_ANCHOR
from pptx.enum.shapes import MSO_SHAPE
import os

BLUE = RGBColor(0x1A, 0x47, 0x8A)
DARK = RGBColor(0x22, 0x22, 0x22)
GRAY = RGBColor(0x66, 0x66, 0x66)
WHITE = RGBColor(0xFF, 0xFF, 0xFF)
ACCENT = RGBColor(0x2E, 0x86, 0xC1)
LIGHT_BG = RGBColor(0xF2, 0xF6, 0xFA)
GREEN = RGBColor(0x1D, 0x8D, 0x48)
RED = RGBColor(0xC0, 0x39, 0x2B)

prs = Presentation()
prs.slide_width = Inches(13.333)
prs.slide_height = Inches(7.5)

FIG_DIR = "tree_figures"


def add_bg(slide, color=WHITE):
    bg = slide.background
    fill = bg.fill
    fill.solid()
    fill.fore_color.rgb = color


def add_textbox(slide, left, top, width, height, text, font_size=18,
                bold=False, color=DARK, alignment=PP_ALIGN.LEFT, font_name="Calibri"):
    txBox = slide.shapes.add_textbox(Inches(left), Inches(top), Inches(width), Inches(height))
    tf = txBox.text_frame
    tf.word_wrap = True
    p = tf.paragraphs[0]
    p.text = text
    p.font.size = Pt(font_size)
    p.font.bold = bold
    p.font.color.rgb = color
    p.font.name = font_name
    p.alignment = alignment
    return tf


def add_bullet_list(slide, left, top, width, height, items, font_size=18,
                    color=DARK, spacing=Pt(6)):
    txBox = slide.shapes.add_textbox(Inches(left), Inches(top), Inches(width), Inches(height))
    tf = txBox.text_frame
    tf.word_wrap = True
    for i, item in enumerate(items):
        if i == 0:
            p = tf.paragraphs[0]
        else:
            p = tf.add_paragraph()
        p.text = item
        p.font.size = Pt(font_size)
        p.font.color.rgb = color
        p.font.name = "Calibri"
        p.space_after = spacing
        p.level = 0
    return tf


def add_accent_bar(slide, left, top, width=0.06, height=1.0):
    shape = slide.shapes.add_shape(
        MSO_SHAPE.RECTANGLE, Inches(left), Inches(top), Inches(width), Inches(height))
    shape.fill.solid()
    shape.fill.fore_color.rgb = ACCENT
    shape.line.fill.background()


def add_image(slide, path, left, top, width=None, height=None):
    full_path = os.path.join(FIG_DIR, path) if not os.path.isabs(path) else path
    if not os.path.exists(full_path):
        add_textbox(slide, left, top, 4, 1, f"[Missing: {path}]", 14, color=RED)
        return
    kwargs = {}
    if width:
        kwargs["width"] = Inches(width)
    if height:
        kwargs["height"] = Inches(height)
    slide.shapes.add_picture(full_path, Inches(left), Inches(top), **kwargs)


def section_header(slide, title):
    add_bg(slide, BLUE)
    add_textbox(slide, 1.0, 2.5, 11, 2, title, 44, bold=True, color=WHITE,
                alignment=PP_ALIGN.CENTER, font_name="Calibri Light")


# ──────────────── SLIDE 1: Title ────────────────
slide = prs.slides.add_slide(prs.slide_layouts[6])
add_bg(slide, LIGHT_BG)
shape = slide.shapes.add_shape(
    MSO_SHAPE.RECTANGLE, 0, 0, prs.slide_width, Inches(0.12))
shape.fill.solid()
shape.fill.fore_color.rgb = ACCENT
shape.line.fill.background()

add_textbox(slide, 1.0, 1.8, 11.3, 1.5,
            "Tree-HyperLISTA", 52, bold=True, color=BLUE,
            alignment=PP_ALIGN.CENTER, font_name="Calibri Light")
add_textbox(slide, 1.0, 3.0, 11.3, 1.0,
            "Ultra-Lightweight Deep Unfolding for Tree-Sparse Recovery",
            28, color=DARK, alignment=PP_ALIGN.CENTER)
add_textbox(slide, 1.0, 4.3, 11.3, 0.7,
            "Shresth Verma    Aditya Agrawal",
            22, bold=True, color=GRAY, alignment=PP_ALIGN.CENTER)
add_textbox(slide, 1.0, 4.9, 11.3, 0.7,
            "Department of CSE, IIT Bombay",
            18, color=GRAY, alignment=PP_ALIGN.CENTER)

# ──────────────── SLIDE 2: Motivation ────────────────
slide = prs.slides.add_slide(prs.slide_layouts[6])
add_bg(slide)
add_textbox(slide, 0.8, 0.4, 11, 0.8, "Why Tree-Sparse Recovery?", 36, bold=True, color=BLUE)
add_accent_bar(slide, 0.8, 1.1, height=0.06, width=3.0)

add_bullet_list(slide, 1.0, 1.6, 5.8, 5, [
    "Sparse recovery: reconstruct x from y = Ax + ε",
    "Tree-structured sparsity arises naturally in:",
    "   • Wavelet decompositions of images",
    "   • Hierarchical signal representations",
    "   • Multi-resolution analysis",
    "",
    "Key constraint: if a child coefficient is active,",
    "its parent must also be active (tree consistency)",
], 20)

add_bullet_list(slide, 7.0, 1.6, 5.5, 5, [
    "Gap in existing methods:",
    "",
    "Classical (ISTA, FISTA): no learning → slow",
    "LISTA (1.56M params): ignores tree structure",
    "HyperLISTA (3 params): elementwise only",
    "",
    "→ No method combines tree structure",
    "   with lightweight deep unfolding",
], 20)


# ──────────────── SLIDE 3: Method Overview ────────────────
slide = prs.slides.add_slide(prs.slide_layouts[6])
add_bg(slide)
add_textbox(slide, 0.8, 0.4, 11, 0.8, "Tree-HyperLISTA: Method", 36, bold=True, color=BLUE)
add_accent_bar(slide, 0.8, 1.1, height=0.06, width=3.0)

add_bullet_list(slide, 1.0, 1.5, 6.0, 5.5, [
    "Only 3 learned hyperparameters (c₁, c₂, c₃)",
    "",
    "Each unfolded layer performs:",
    "  1. Momentum step:  z = x + β(x − x_prev)",
    "  2. Gradient step:  u = z + Wᵀ(y − Az)",
    "  3. Tree scoring:   sᵢ = |uᵢ| + ρ Σ s_child  (bottom-up)",
    "  4. Support select: S = TopKTree(s, K)",
    "  5. Tree threshold: soft-threshold within S, zero outside",
    "",
    "Per-layer β, θ, K derived from (c₁, c₂, c₃)",
    "via adaptive formulas — NOT learned per-layer",
], 19)

add_bullet_list(slide, 7.5, 1.5, 5.0, 5, [
    "Three support selection modes:",
    "",
    "M1 — Hard tree projection:",
    "    greedy top-K with ancestor closure",
    "",
    "M2 — Threshold + closure:",
    "    threshold then enforce tree consistency",
    "",
    "M3 — Hybrid (recommended):",
    "    top-K tree projection + soft threshold",
    "    → best NMSE with lowest variance",
], 19)


# ──────────────── SLIDE 4: Key Innovation ────────────────
slide = prs.slides.add_slide(prs.slide_layouts[6])
add_bg(slide)
add_textbox(slide, 0.8, 0.4, 11, 0.8, "Key Innovation: Tree-Aware Proximal Operator", 36, bold=True, color=BLUE)
add_accent_bar(slide, 0.8, 1.1, height=0.06, width=3.0)

add_bullet_list(slide, 1.0, 1.6, 5.8, 5.5, [
    "Standard (LISTA/HyperLISTA):",
    "   xᵢ = sign(uᵢ) max(|uᵢ| − θ, 0)",
    "   → treats each coefficient independently",
    "   → ignores tree structure completely",
    "",
    "Tree-HyperLISTA:",
    "   1. Score each node using subtree info",
    "   2. Select tree-consistent support S",
    "   3. Threshold only within S",
    "   → exploits parent-child dependencies",
    "   → O(n) via bottom-up aggregation",
], 20)

add_bullet_list(slide, 7.2, 1.6, 5.5, 5.5, [
    "What makes it \"ultra-lightweight\":",
    "",
    "• Same 3 global hyperparameters as HyperLISTA",
    "• W matrix is precomputed (not learned)",
    "• Tree topology is fixed (not learned)",
    "• ρ decay is fixed at 0.5 (not learned)",
    "",
    "The ONLY change vs HyperLISTA is",
    "replacing soft-threshold with tree-aware operator",
    "",
    "→ Structure in the operator, not parameters",
], 20)


# ──────────────── SLIDE 5: Section header ────────────────
slide = prs.slides.add_slide(prs.slide_layouts[6])
section_header(slide, "Experimental Results")


# ──────────────── SLIDE 6: Core Results Table ────────────────
slide = prs.slides.add_slide(prs.slide_layouts[6])
add_bg(slide)
add_textbox(slide, 0.8, 0.4, 11, 0.8, "Core Results: Synthetic Tree-Sparse Recovery", 36, bold=True, color=BLUE)
add_accent_bar(slide, 0.8, 1.1, height=0.06, width=3.0)

add_textbox(slide, 1.0, 1.4, 11, 0.5,
            "n=255, m=127, K=16 layers, 30 dB SNR, 5 seeds", 16, color=GRAY)

header = ["Model", "NMSE (dB)", "Precision", "Recall", "#Params"]
rows = [
    ["Tree-ISTA",       "−6.00 ± 0.02",  "0.759", "0.734", "0"],
    ["Tree-FISTA",      "−8.34 ± 0.02",  "0.840", "0.816", "0"],
    ["LISTA",           "−13.18 ± 0.08", "0.255", "0.997", "1.56M"],
    ["HyperLISTA",      "−15.91 ± 6.91", "0.791", "0.967", "3"],
    ["TH-Hard (M1)",    "−28.80 ± 0.11", "0.740", "1.000", "3"],
    ["TH-Hybrid (M3)",  "−28.66 ± 0.10", "0.663", "1.000", "3"],
]

table_shape = slide.shapes.add_table(len(rows) + 1, 5,
    Inches(1.0), Inches(2.0), Inches(11.0), Inches(3.8))
table = table_shape.table

col_widths = [2.8, 2.5, 1.8, 1.8, 2.1]
for i, w in enumerate(col_widths):
    table.columns[i].width = Inches(w)

for j, h in enumerate(header):
    cell = table.cell(0, j)
    cell.text = h
    for p in cell.text_frame.paragraphs:
        p.font.size = Pt(16)
        p.font.bold = True
        p.font.color.rgb = WHITE
        p.alignment = PP_ALIGN.CENTER
    cell.fill.solid()
    cell.fill.fore_color.rgb = BLUE

for i, row in enumerate(rows):
    is_ours = "TH-" in row[0]
    for j, val in enumerate(row):
        cell = table.cell(i + 1, j)
        cell.text = val
        for p in cell.text_frame.paragraphs:
            p.font.size = Pt(15)
            p.font.bold = is_ours
            p.font.color.rgb = GREEN if is_ours else DARK
            p.alignment = PP_ALIGN.CENTER
        cell.fill.solid()
        cell.fill.fore_color.rgb = RGBColor(0xE8, 0xF8, 0xE8) if is_ours else (
            RGBColor(0xF8, 0xF8, 0xF8) if i % 2 == 0 else WHITE)

add_bullet_list(slide, 1.0, 6.0, 11, 1.2, [
    "TH-Hybrid: +12.8 dB over HyperLISTA (same 3 params)  |  +15.5 dB over LISTA (1.56M params)  |  Perfect recall (1.000)",
], 18, color=BLUE)


# ──────────────── SLIDE 7: Convergence ────────────────
slide = prs.slides.add_slide(prs.slide_layouts[6])
add_bg(slide)
add_textbox(slide, 0.8, 0.4, 11, 0.8, "Convergence & Support Recovery", 36, bold=True, color=BLUE)
add_accent_bar(slide, 0.8, 1.1, height=0.06, width=3.0)
add_image(slide, "tree_fig1_convergence.png", 0.5, 1.5, width=6.0)
add_image(slide, "tree_fig2_support.png", 6.8, 1.5, width=6.0)
add_textbox(slide, 0.5, 6.2, 12, 0.8,
            "Tree-HyperLISTA converges to −28.66 dB in 16 layers with perfect support recall",
            18, color=GRAY, alignment=PP_ALIGN.CENTER)


# ──────────────── SLIDE 8: Mismatch ────────────────
slide = prs.slides.add_slide(prs.slide_layouts[6])
add_bg(slide)
add_textbox(slide, 0.8, 0.4, 11, 0.8, "Robustness: Mismatch Experiments", 36, bold=True, color=BLUE)
add_accent_bar(slide, 0.8, 1.1, height=0.06, width=3.0)
add_image(slide, "tree_fig3_snr_db.png", 0.3, 1.4, width=6.2)
add_image(slide, "tree_fig4_target_sparsity.png", 6.8, 1.4, width=6.2)
add_textbox(slide, 0.5, 6.0, 12, 1.0,
            "Tree-HyperLISTA degrades gracefully under SNR variation and sparsity mismatch",
            18, color=GRAY, alignment=PP_ALIGN.CENTER)


# ──────────────── SLIDE 9: Ablation ────────────────
slide = prs.slides.add_slide(prs.slide_layouts[6])
add_bg(slide)
add_textbox(slide, 0.8, 0.4, 11, 0.8, "Ablation: Sensitivity & Mechanisms", 36, bold=True, color=BLUE)
add_accent_bar(slide, 0.8, 1.1, height=0.06, width=3.0)
add_image(slide, "tree_fig6_sensitivity.png", 0.3, 1.4, width=6.2)
add_image(slide, "tree_fig7_mechanisms.png", 6.8, 1.4, width=6.2)
add_bullet_list(slide, 0.5, 5.8, 12, 1.5, [
    "Sensitivity: broad optimal basin for c₂ (momentum); c₁ (threshold) most critical",
    "Mechanisms: M1 (hard) and M3 (hybrid) perform comparably; M2 (threshold) unstable",
], 17, color=GRAY)


# ──────────────── SLIDE 10: Image CS ────────────────
slide = prs.slides.add_slide(prs.slide_layouts[6])
add_bg(slide)
add_textbox(slide, 0.8, 0.4, 11, 0.8, "Wavelet-Domain Image Compressed Sensing", 36, bold=True, color=BLUE)
add_accent_bar(slide, 0.8, 1.1, height=0.06, width=3.0)

add_image(slide, "tree_fig9_image_cs.png", 0.3, 1.4, width=7.5)

add_bullet_list(slide, 8.2, 1.6, 4.5, 5, [
    "Real wavelet coefficients from",
    "8 natural images (Haar DWT)",
    "",
    "TH-hybrid achieves:",
    "  • Best SSIM at all CS ratios",
    "  • 31.66 dB at CS 50%",
    "  • Beats HyperLISTA (3p) by 10+ dB at 50%",
    "",
    "Tree structure in wavelet domain",
    "→ tree-aware proximal outperforms",
    "   elementwise proximal",
], 18)


# ──────────────── SLIDE 11: Backbone Ablation ────────────────
slide = prs.slides.add_slide(prs.slide_layouts[6])
add_bg(slide)
add_textbox(slide, 0.8, 0.4, 11, 0.8, "Backbone Ablation", 36, bold=True, color=BLUE)
add_accent_bar(slide, 0.8, 1.1, height=0.06, width=3.0)
add_image(slide, "tree_fig_backbone.png", 0.3, 1.4, width=6.5)

add_bullet_list(slide, 7.3, 1.6, 5.5, 5.5, [
    "Compared 6 backbone variants:",
    "",
    "Tree-ALISTA (32p):       −15.00 dB",
    "Tree-ALISTA-MM (32p):    −17.02 dB",
    "TH-Elem (3p, no tree):  −15.75 dB",
    "Tree-HyperLISTA (3p):   −28.66 dB",
    "",
    "Key insight: tree-aware operator is",
    "the critical ingredient, not backbone",
    "complexity or parameter count",
], 19)


# ──────────────── SLIDE 12: Extended Mismatch ────────────────
slide = prs.slides.add_slide(prs.slide_layouts[6])
add_bg(slide)
add_textbox(slide, 0.8, 0.4, 11, 0.8, "Extended Robustness Analysis", 36, bold=True, color=BLUE)
add_accent_bar(slide, 0.8, 1.1, height=0.06, width=3.0)
add_image(slide, "tree_fig_mm_magnitude.png", 0.2, 1.3, width=6.3)
add_image(slide, "tree_fig_mm_consistency.png", 6.8, 1.3, width=6.0)
add_textbox(slide, 0.5, 6.0, 12, 1.0,
            "Robust to magnitude scaling (−28.7→−28.1 dB at 3×)  |  Graceful degradation as tree consistency decreases",
            17, color=GRAY, alignment=PP_ALIGN.CENTER)


# ──────────────── SLIDE 13: Key Takeaways ────────────────
slide = prs.slides.add_slide(prs.slide_layouts[6])
add_bg(slide, LIGHT_BG)
add_textbox(slide, 0.8, 0.4, 11, 0.8, "Key Takeaways", 40, bold=True, color=BLUE,
            alignment=PP_ALIGN.CENTER)

takeaways = [
    ("1", "Structure > Parameters",
     "3 params with tree-aware operator beats 1.56M params (LISTA) by 15.5 dB"),
    ("2", "Minimal Parameterization Generalizes",
     "The 3-parameter principle extends from elementwise to tree-structured sparsity"),
    ("3", "Hard Projection Works",
     "Simple greedy top-K tree projection nearly matches the hybrid mode"),
    ("4", "Real-World Validation",
     "Best SSIM on wavelet image CS at all compression ratios"),
]

for i, (num, title, desc) in enumerate(takeaways):
    y = 1.5 + i * 1.4
    shape = slide.shapes.add_shape(
        MSO_SHAPE.OVAL, Inches(1.2), Inches(y + 0.05), Inches(0.6), Inches(0.6))
    shape.fill.solid()
    shape.fill.fore_color.rgb = ACCENT
    shape.line.fill.background()
    tf = shape.text_frame
    tf.paragraphs[0].text = num
    tf.paragraphs[0].font.size = Pt(22)
    tf.paragraphs[0].font.bold = True
    tf.paragraphs[0].font.color.rgb = WHITE
    tf.paragraphs[0].alignment = PP_ALIGN.CENTER
    tf.vertical_anchor = MSO_ANCHOR.MIDDLE

    add_textbox(slide, 2.2, y, 9, 0.5, title, 24, bold=True, color=DARK)
    add_textbox(slide, 2.2, y + 0.5, 9, 0.5, desc, 18, color=GRAY)


# ──────────────── SLIDE 14: Conclusion ────────────────
slide = prs.slides.add_slide(prs.slide_layouts[6])
add_bg(slide, BLUE)
add_textbox(slide, 1.0, 1.5, 11.3, 1.5,
            "Thank You", 52, bold=True, color=WHITE,
            alignment=PP_ALIGN.CENTER, font_name="Calibri Light")
add_textbox(slide, 1.0, 3.2, 11.3, 0.8,
            "Tree-HyperLISTA: 3 parameters, 12.8 dB better than HyperLISTA, perfect recall",
            24, color=RGBColor(0xBB, 0xDD, 0xFF), alignment=PP_ALIGN.CENTER)
add_textbox(slide, 1.0, 4.5, 11.3, 0.8,
            "Code: github.com/ALPHAGOD12/tree-hyperlista",
            20, color=RGBColor(0x99, 0xCC, 0xFF), alignment=PP_ALIGN.CENTER)
add_textbox(slide, 1.0, 5.3, 11.3, 0.8,
            "Shresth Verma & Aditya Agrawal  •  IIT Bombay",
            20, color=RGBColor(0x99, 0xCC, 0xFF), alignment=PP_ALIGN.CENTER)

out_path = "tree_hyperlista_slides.pptx"
prs.save(out_path)
print(f"Saved {out_path} ({len(prs.slides)} slides)")
