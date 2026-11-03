"""Compose the editable Figure 2 workflow with the selected statistical panels."""
from pathlib import Path
import os

import pymupdf
from PIL import Image

output = Path(os.environ["METAR_PLOT_OUTPUT"])
mm = 72 / 25.4
panel = pymupdf.open(output / "panel/Figure2a_editable.pdf")
normalized = pymupdf.open()
page = normalized.new_page(width=178 * mm, height=51.81493051317 * mm)
page.show_pdf_page(page.rect, panel, 0)
normalized.save(output / "panel/Figure2a.pdf")

figure = pymupdf.open(output / "baseline/Figure2.pdf")
page = figure[0]
region = pymupdf.Rect(0, 0, 178 * mm, 45.3 * mm)
page.add_redact_annot(region, fill=(1, 1, 1))
page.apply_redactions(images=2, graphics=2, text=0)
page.show_pdf_page(region, normalized, 0, clip=region)
figure.save(output / "full/Figure2.pdf", garbage=4, deflate=True)
for dpi, name in [(300, "Figure2_178mm.png"), (600, "Figure2_600dpi.png")]:
    pixels = page.get_pixmap(dpi=dpi, alpha=False)
    Image.frombytes("RGB", (pixels.width, pixels.height), pixels.samples).save(
        output / "full" / name, dpi=(dpi, dpi)
    )
