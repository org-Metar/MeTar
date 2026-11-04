from pathlib import Path
import os
import hashlib
import json
import pymupdf
from PIL import Image

ROOT = Path(__file__).resolve().parent
WORK = ROOT
BASE = ROOT / "external_inputs/Figure4_baseline.pdf"
PANEL = ROOT / "external_inputs/Figure4a_editable.pdf"
OUT = ROOT / "Figure4.pdf"
MM = 72 / 25.4

def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()

def add_legend(page):
    font_path = Path(os.environ["METAR_PLOT_ARIAL"])
    page.insert_font(fontname="FigureArial", fontfile=str(font_path))
    font = pymupdf.Font(fontfile=str(font_path))
    fs = 8.0
    gap = 3.5
    labels = ["MeTar", "PertKGE", "Random"]
    colors = [(45/255, 111/255, 159/255), (111/255, 144/255, 127/255), (148/255, 156/255, 166/255)]
    widths = [5 + font.text_length(label, fontsize=fs) / MM for label in labels]
    total = sum(widths) + 2 * gap
    start = (178 - total) / 2
    positions = [start, start + widths[0] + gap, start + widths[0] + widths[1] + 2 * gap]
    old_text_y = 55.10
    text_y = old_text_y - 0.55
    icon_y = old_text_y - 1.586
    for i, (label, x, color) in enumerate(zip(labels, positions, colors)):
        # Existing icons and line segments remain in the formal baseline PDF.
        # Only the text glyphs are covered and redrawn at the adjusted baseline.
        text_rect = pymupdf.Rect((x + 4.7) * MM, (old_text_y - 2.6) * MM, (x + 5 + font.text_length(label, fontsize=fs) / MM + 0.8) * MM, (old_text_y + 0.9) * MM)
        page.draw_rect(text_rect, color=None, fill=(1, 1, 1), overlay=True)
        page.insert_text(((x + 5) * MM, text_y * MM), label, fontname="FigureArial", fontsize=fs, color=(38/255, 50/255, 56/255))
    return {"labels": labels, "font_size_pt": fs, "icon_y_mm": icon_y, "old_text_y_mm": old_text_y, "new_text_y_mm": text_y, "text_baseline_shift_mm": -0.55, "icons_unchanged": True}

def main():
    baseline = pymupdf.open(BASE)
    panel = pymupdf.open(PANEL)
    doc = pymupdf.open()
    page = doc.new_page(width=baseline[0].rect.width, height=baseline[0].rect.height)
    page.show_pdf_page(page.rect, baseline, 0)
    # Remove only the old editable panel-a and legend text layers before placing
    # their replacements. Existing plot data and legend icons remain untouched.
    page.add_redact_annot(pymupdf.Rect(4 * MM, 3.7877879 * MM, 174 * MM, 48.5 * MM), fill=(1, 1, 1))
    font = pymupdf.Font(fontfile=str(Path(os.environ["METAR_PLOT_ARIAL"])))
    old_text_y = 55.10
    fs = 8.0
    gap = 3.5
    labels = ["MeTar", "PertKGE", "Random"]
    widths = [5 + font.text_length(label, fontsize=fs) / MM for label in labels]
    total = sum(widths) + 2 * gap
    start = (178 - total) / 2
    positions = [start, start + widths[0] + gap, start + widths[0] + widths[1] + 2 * gap]
    for label, x in zip(labels, positions):
        page.add_redact_annot(pymupdf.Rect((x + 4.7) * MM, (old_text_y - 2.6) * MM, (x + 5 + font.text_length(label, fontsize=fs) / MM + 0.8) * MM, (old_text_y + 0.9) * MM), fill=(1, 1, 1))
    page.apply_redactions()
    target = pymupdf.Rect(4 * MM, 3.7877879 * MM, 174 * MM, 48.5 * MM)
    page.show_pdf_page(target, panel, 0, keep_proportion=True)
    label_rect = pymupdf.Rect(2.5 * MM, 0, 5.5 * MM, 4.6 * MM)
    page.show_pdf_page(label_rect, baseline, 0, clip=label_rect)
    legend = add_legend(page)
    doc.save(OUT, garbage=4, deflate=True)
    outputs = {}
    for dpi, name in [(300, "Figure4_178mm.png"), (600, "Figure4_600dpi.png")]:
        pix = page.get_pixmap(matrix=pymupdf.Matrix(dpi / 72, dpi / 72), alpha=False)
        out = WORK / name
        Image.frombytes("RGB", [pix.width, pix.height], pix.samples).save(out, dpi=(dpi, dpi))
        outputs[name] = sha(out)


if __name__ == "__main__":
    main()
