"""Generate PDF slides for Tree-HyperLISTA paper using reportlab."""
from reportlab.lib.pagesizes import landscape
from reportlab.lib.units import inch
from reportlab.lib.colors import HexColor, white, black
from reportlab.platypus import SimpleDocTemplate, Spacer, Image, Table, TableStyle, PageBreak
from reportlab.lib.enums import TA_LEFT, TA_CENTER
from reportlab.lib.styles import ParagraphStyle
from reportlab.platypus import Paragraph
import os

W, H = landscape((13.333*inch, 7.5*inch))
FIG = "tree_figures"

BLUE = HexColor("#1A478A")
ACCENT = HexColor("#2E86C1")
DARK = HexColor("#222222")
GRAY = HexColor("#666666")
GREEN = HexColor("#1D8D48")
LIGHT_BG = HexColor("#F2F6FA")

title_style = ParagraphStyle("title", fontSize=36, textColor=BLUE, fontName="Helvetica-Bold",
                             alignment=TA_CENTER, spaceAfter=20)
subtitle_style = ParagraphStyle("sub", fontSize=22, textColor=DARK, alignment=TA_CENTER, spaceAfter=10)
heading_style = ParagraphStyle("head", fontSize=28, textColor=BLUE, fontName="Helvetica-Bold",
                               spaceAfter=12)
body_style = ParagraphStyle("body", fontSize=16, textColor=DARK, leading=22, spaceAfter=6)
small_style = ParagraphStyle("small", fontSize=14, textColor=GRAY, leading=20, spaceAfter=4)
bullet_style = ParagraphStyle("bullet", fontSize=16, textColor=DARK, leading=22, leftIndent=20,
                              spaceAfter=4, bulletIndent=8)
sec_title = ParagraphStyle("sec", fontSize=36, textColor=white, fontName="Helvetica-Bold",
                           alignment=TA_CENTER, spaceAfter=0)
highlight_style = ParagraphStyle("hl", fontSize=16, textColor=ACCENT, fontName="Helvetica-Bold",
                                 alignment=TA_CENTER, spaceAfter=6)

def fig(name, w=5*inch, h=3.5*inch):
    fp = os.path.join(FIG, name)
    if os.path.exists(fp):
        return Image(fp, width=w, height=h)
    return Paragraph(f"[Missing: {name}]", small_style)

elements = []

def add_slide(elems):
    elements.extend(elems)
    elements.append(PageBreak())

# S1: Title
add_slide([
    Spacer(1, 1.5*inch),
    Paragraph("Tree-HyperLISTA", title_style),
    Paragraph("Ultra-Lightweight Deep Unfolding for Tree-Sparse Recovery", subtitle_style),
    Spacer(1, 0.5*inch),
    Paragraph("Shresth Verma &nbsp;&nbsp; Aditya Agrawal", ParagraphStyle("a", fontSize=20, textColor=GRAY, alignment=TA_CENTER, fontName="Helvetica-Bold")),
    Paragraph("Department of CSE, IIT Bombay", ParagraphStyle("a2", fontSize=16, textColor=GRAY, alignment=TA_CENTER)),
])

# S2: Motivation
add_slide([
    Paragraph("Why Tree-Sparse Recovery?", heading_style),
    Paragraph("\u2022 Sparse recovery: reconstruct x from y = Ax + \u03b5", bullet_style),
    Paragraph("\u2022 Tree-structured sparsity: wavelet decompositions, hierarchical representations", bullet_style),
    Paragraph("\u2022 Key constraint: child active \u21d2 parent must be active (tree consistency)", bullet_style),
    Spacer(1, 0.3*inch),
    Paragraph("<b>Gap:</b> Classical methods (ISTA/FISTA) have no learning. LISTA (1.56M params) ignores tree structure. HyperLISTA (3 params) is elementwise only. No method combines tree structure with lightweight deep unfolding.", body_style),
])

# S3: Method
add_slide([
    Paragraph("Tree-HyperLISTA: Method Overview", heading_style),
    Paragraph("Only <b>3 learned hyperparameters</b> (c<sub>1</sub>, c<sub>2</sub>, c<sub>3</sub>)", body_style),
    Spacer(1, 0.15*inch),
    Paragraph("Each unfolded layer performs:", body_style),
    Paragraph("1. <b>Momentum:</b> z = x + \u03b2(x \u2212 x<sub>prev</sub>)", bullet_style),
    Paragraph("2. <b>Gradient:</b> u = z + W<super>T</super>(y \u2212 Az)", bullet_style),
    Paragraph("3. <b>Tree scoring:</b> s<sub>i</sub> = |u<sub>i</sub>| + \u03c1 \u03a3 s<sub>child</sub> (bottom-up, O(n))", bullet_style),
    Paragraph("4. <b>Support select:</b> S = TopKTree(s, K)", bullet_style),
    Paragraph("5. <b>Tree threshold:</b> soft-threshold within S, zero outside", bullet_style),
    Spacer(1, 0.2*inch),
    Paragraph("<b>Three modes:</b> M1 (hard top-K) | M2 (threshold+closure) | M3 (hybrid, recommended)", body_style),
    Paragraph("Per-layer \u03b2, \u03b8, K derived from (c<sub>1</sub>,c<sub>2</sub>,c<sub>3</sub>) via adaptive formulas \u2014 NOT learned per-layer", small_style),
])

# S4: Key Innovation
add_slide([
    Paragraph("Key Innovation: Tree-Aware Proximal Operator", heading_style),
    Paragraph("<b>Standard (LISTA/HyperLISTA):</b> x<sub>i</sub> = sign(u<sub>i</sub>) max(|u<sub>i</sub>| \u2212 \u03b8, 0) \u2192 treats each coefficient independently", body_style),
    Spacer(1, 0.15*inch),
    Paragraph("<b>Tree-HyperLISTA:</b> Score nodes via subtree info \u2192 select tree-consistent support S \u2192 threshold within S", body_style),
    Spacer(1, 0.15*inch),
    Paragraph("\u2022 Same 3 global hyperparameters as HyperLISTA", bullet_style),
    Paragraph("\u2022 W matrix precomputed (not learned); tree topology fixed; \u03c1=0.5 fixed", bullet_style),
    Paragraph("\u2022 The ONLY change: replacing soft-threshold with tree-aware operator", bullet_style),
    Spacer(1, 0.2*inch),
    Paragraph("\u2192 Structure in the operator, not in parameters", highlight_style),
])

# S5: Core Results
core_data = [
    ["Model", "NMSE (dB)", "Precision", "Recall", "#Params"],
    ["Tree-ISTA", "\u22126.00 \u00b1 0.02", "0.759", "0.734", "0"],
    ["Tree-FISTA", "\u22128.34 \u00b1 0.02", "0.840", "0.816", "0"],
    ["LISTA", "\u221213.18 \u00b1 0.08", "0.255", "0.997", "1.56M"],
    ["HyperLISTA", "\u221215.91 \u00b1 6.91", "0.791", "0.967", "3"],
    ["TH-Hard (M1)", "\u221228.80 \u00b1 0.11", "0.740", "1.000", "3"],
    ["TH-Hybrid (M3)", "\u221228.66 \u00b1 0.10", "0.663", "1.000", "3"],
]
ts = TableStyle([
    ('BACKGROUND', (0,0), (-1,0), BLUE),
    ('TEXTCOLOR', (0,0), (-1,0), white),
    ('FONTNAME', (0,0), (-1,0), 'Helvetica-Bold'),
    ('FONTSIZE', (0,0), (-1,-1), 13),
    ('ALIGN', (0,0), (-1,-1), 'CENTER'),
    ('GRID', (0,0), (-1,-1), 0.5, HexColor("#CCCCCC")),
    ('BACKGROUND', (0,5), (-1,5), HexColor("#E8F8E8")),
    ('BACKGROUND', (0,6), (-1,6), HexColor("#E8F8E8")),
    ('TEXTCOLOR', (0,5), (-1,6), GREEN),
    ('FONTNAME', (0,5), (-1,6), 'Helvetica-Bold'),
    ('ROWBACKGROUNDS', (0,1), (-1,4), [HexColor("#F8F8F8"), white]),
])
tbl = Table(core_data, colWidths=[2.2*inch, 2*inch, 1.3*inch, 1.3*inch, 1.2*inch])
tbl.setStyle(ts)
add_slide([
    Paragraph("Core Results: Synthetic Tree-Sparse Recovery", heading_style),
    Paragraph("n=255, m=127, K=16 layers, 30 dB SNR, 5 seeds", small_style),
    Spacer(1, 0.15*inch),
    tbl,
    Spacer(1, 0.2*inch),
    Paragraph("TH-Hybrid: <b>+12.8 dB</b> over HyperLISTA (same 3 params) | <b>+15.5 dB</b> over LISTA (1.56M params) | <b>Perfect recall</b>", highlight_style),
])

# S6: Convergence
add_slide([
    Paragraph("Convergence &amp; Support Recovery", heading_style),
    fig("tree_fig1_convergence.png", 9*inch, 4.2*inch),
    Paragraph("Tree-HyperLISTA converges to \u221228.66 dB in 16 layers with perfect support recall", small_style),
])

# S7: Mismatch
add_slide([
    Paragraph("Robustness: Mismatch Experiments", heading_style),
    fig("tree_fig3_snr_db.png", 9*inch, 4.2*inch),
    Paragraph("Tree-HyperLISTA degrades gracefully under SNR variation and sparsity mismatch", small_style),
])

# S8: Ablation
add_slide([
    Paragraph("Ablation: Sensitivity &amp; Mechanisms", heading_style),
    fig("tree_fig6_sensitivity.png", 9*inch, 4.2*inch),
    Paragraph("Broad optimal basin for c2 (momentum); c1 (threshold) most critical. M1 and M3 comparable.", small_style),
])

# S9: Image CS
add_slide([
    Paragraph("Wavelet-Domain Image Compressed Sensing", heading_style),
    fig("tree_fig9_image_cs.png", 9*inch, 4*inch),
    Paragraph("\u2022 TH-hybrid: <b>best SSIM at all CS ratios</b> (0.707, 0.834, 0.952)", body_style),
    Paragraph("\u2022 31.66 dB at 50% CS, beating HyperLISTA by 10+ dB with same 3 params", body_style),
])

# S10: Backbone Ablation
add_slide([
    Paragraph("Appendix B: Backbone Ablation", heading_style),
    fig("tree_fig_backbone.png", 8*inch, 3.8*inch),
    Paragraph("Tree-ALISTA (32p): \u221215.00 dB | TH-Elem (3p, no tree): \u221215.75 dB | <b>Tree-HyperLISTA (3p): \u221228.66 dB</b>", body_style),
    Paragraph("Tree-aware operator is the critical ingredient, not backbone complexity", small_style),
])

# S11: Extended Mechanisms
add_slide([
    Paragraph("Appendix C: Extended Mechanism Ablation", heading_style),
    fig("tree_fig_mechanisms_ext.png", 7*inch, 3.5*inch),
    Paragraph("Differentiable relaxations (Diff. Ancestor, Gumbel Top-K) underperform non-differentiable hard projection", body_style),
])

# S12: Rho + Sparsity
add_slide([
    Paragraph("Appendix D\u2013E: Rho Sensitivity &amp; Sparsity Mismatch", heading_style),
    fig("tree_fig_sparsity.png", 8*inch, 3.8*inch),
    Paragraph("\u03c1=0.1 and \u03c1=0.5 give same result \u2014 robust to decay choice. Graceful at sparser signals; all degrade when denser.", small_style),
])

# S13: Extended Mismatch
add_slide([
    Paragraph("Appendix F: Extended Mismatch Robustness", heading_style),
    fig("tree_fig_mm_magnitude.png", 8*inch, 3.5*inch),
    Paragraph("Magnitude: TH robust (\u221228.7\u2192\u221228.1 dB at 3\u00d7) while HyperLISTA diverges. Consistency: degrades as tree structure breaks.", body_style),
])

# S14: Low-SNR + Superlinear
add_slide([
    Paragraph("Appendix G\u2013H: Low-SNR &amp; Superlinear Convergence", heading_style),
    fig("tree_fig_lowsnr.png", 8*inch, 3.5*inch),
    Paragraph("TH maintains advantage at all SNR levels. Per-instance tuning \u2192 \u22124.0 dB at only 6 layers (future work direction).", body_style),
])

# S15: Cross-Structure
add_slide([
    Paragraph("Appendix I: Cross-Structure Experiments", heading_style),
    fig("tree_fig_cross.png", 7*inch, 3.5*inch),
    Paragraph("Tree-aware method does NOT hurt performance on non-tree signals (\u22123.45 dB vs \u22122.73 dB). Safe to use when structure is uncertain.", body_style),
])

# S16: Key Takeaways
add_slide([
    Paragraph("Key Takeaways", heading_style),
    Spacer(1, 0.15*inch),
    Paragraph("<b>1. Structure &gt; Parameters:</b> 3 params + tree operator beats 1.56M params by 15.5 dB", body_style),
    Spacer(1, 0.1*inch),
    Paragraph("<b>2. Minimal Parameterization Generalizes:</b> 3-parameter principle extends to tree-structured sparsity", body_style),
    Spacer(1, 0.1*inch),
    Paragraph("<b>3. Hard Projection Works:</b> Simple greedy top-K tree projection nearly matches hybrid mode", body_style),
    Spacer(1, 0.1*inch),
    Paragraph("<b>4. Real-World Validation:</b> Best SSIM on wavelet image CS at all compression ratios", body_style),
    Spacer(1, 0.1*inch),
    Paragraph("<b>5. Robust &amp; Safe:</b> Graceful degradation under mismatch; doesn't hurt non-tree signals", body_style),
])

# S17: Thank You
add_slide([
    Spacer(1, 1.5*inch),
    Paragraph("Thank You", ParagraphStyle("ty", fontSize=44, textColor=BLUE, fontName="Helvetica-Bold", alignment=TA_CENTER)),
    Spacer(1, 0.3*inch),
    Paragraph("Tree-HyperLISTA: 3 parameters, 12.8 dB better than HyperLISTA, perfect recall", ParagraphStyle("ty2", fontSize=20, textColor=ACCENT, alignment=TA_CENTER)),
    Spacer(1, 0.3*inch),
    Paragraph("Code: github.com/ALPHAGOD12/tree-hyperlista", ParagraphStyle("ty3", fontSize=16, textColor=GRAY, alignment=TA_CENTER)),
    Paragraph("Shresth Verma &amp; Aditya Agrawal \u2022 IIT Bombay", ParagraphStyle("ty4", fontSize=16, textColor=GRAY, alignment=TA_CENTER)),
])

doc = SimpleDocTemplate("tree_hyperlista_slides.pdf", pagesize=(W, H),
                        leftMargin=0.8*inch, rightMargin=0.8*inch,
                        topMargin=0.6*inch, bottomMargin=0.4*inch)
doc.build(elements)
print(f"Saved tree_hyperlista_slides.pdf ({len(elements)} elements)")
