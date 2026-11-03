from pathlib import Path
import hashlib
import json
import re
import pymupdf as fitz
import numpy as np

ROOT = Path(__file__).resolve().parent
BASE = ROOT / "external_inputs/Figure5_baseline.pdf"
OUT = ROOT / "Figure5.pdf"
PPTX = ROOT / "Figure5a_editable.pptx"
STREAM_XREF = 51

def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()

def u16(text: str) -> bytes:
    return text.encode("utf-16-be").hex().upper().encode("ascii")

def spans(page):
    return [s for block in page.get_text("dict")["blocks"]
            for line in block.get("lines", [])
            for s in line.get("spans", []) if s["text"].strip()]

def main() -> None:
    assert sha(BASE) == "47f3d8a76f2067667e6d687a999043fc2e5304bc930d09562c44521d285a7763"
    doc = fitz.open(BASE)
    original_stream = doc.xref_stream(STREAM_XREF)
    assert original_stream is not None
    old_b = u16("Best withheld-target percentile in the 16,583-gene universe (%)")
    new_b = u16("Best withheld-target percentile in the 16,583-gene target space (%)")
    old_e1 = u16("log2(input cells per drug-cell")
    new_e = u16("log2(input cells per drug-cell unit)")
    old_e2 = u16("unit)")
    block_b = b"q 1 0 0 1 186.34111 214.26358 cm .14901962 .19607844 rg .14901962 .19607844 RG 1 w BT/F1 8.5 Tf[<" + old_b + b">]TJ ET Q"
    block_e1 = b"q 1 0 0 1 96.65133 513.4798 cm .14901962 .19607844 rg .14901962 .19607844 RG 1 w BT/F1 8.5 Tf[<" + old_e1 + b">]TJ ET Q"
    block_e2 = b"q 1 0 0 1 275.54548 9.05529 cm .14901962 .19607844 rg .14901962 .19607844 RG 1 w BT/F1 8.5 Tf[<" + old_e2 + b">]TJ ET Q"
    assert original_stream.count(old_b) == 1
    assert original_stream.count(old_e1) == 1
    assert original_stream.count(old_e2) == 1
    patched = original_stream.replace(old_b, new_b)
    patched = patched.replace(old_e1, new_e)
    # Remove only the second split-line text operator, retaining its surrounding
    # graphics and all other panel text.
    patched, removed = re.subn(rb"q 1 0 0 1 275\.54548 9\.05529 cm .*?BT/F1 8\.5 Tf\[<" + old_e2 + rb">\]TJ ET Q", b"", patched, count=1)
    assert removed == 1
    doc.update_stream(STREAM_XREF, patched)
    OUT.unlink(missing_ok=True)
    doc.save(OUT, garbage=4, deflate=True)
    doc.close()
    out = fitz.open(OUT)
    out[0].get_pixmap(dpi=180, alpha=False).save(ROOT / "Figure5_178mm.png")
    out[0].get_pixmap(dpi=600, alpha=False).save(ROOT / "Figure5_600dpi.png")
    out[0].get_pixmap(dpi=600, alpha=False, clip=fitz.Rect(160, 450, 360, 691.6536)).save(ROOT / "Figure5_5b_5e_detail_600dpi.png")
    out.close()

if __name__ == "__main__":
    main()
