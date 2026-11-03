from pathlib import Path
import re, hashlib, json, os
import pymupdf as fitz

ROOT=Path(os.environ["METAR_PLOT_OUTPUT"])
WORK=ROOT
BASE=Path(os.environ["METAR_PLOT_ASSETS"])/"Figure6_baseline.pdf"
PPTPDF=Path(os.environ["METAR_PLOT_ASSETS"])/"Figure6a_editable.pdf"
MM=72/25.4
MCIDS=[10,15,19,24,68,69,70,86,87,88,89,92,95,101,104,105]
def sha(p):
    h=hashlib.sha256(); h.update(p.read_bytes()); return h.hexdigest()

d=fitz.open(BASE); page=d[0]; xref=page.get_contents()[0]
stream=d.xref_stream(xref).decode('latin1')
do='q\n1 0 0 1 0 0 cm\n/fzFrm0 Do\nQ\n'
assert stream.count(do)==1, f'expected one fzFrm0 Do, found {stream.count(do)}'
stream=stream.replace(do,'',1)
for mcid in MCIDS:
    pat=rf'/P <</MCID {mcid}>>BDC.*?EMC\n'
    stream,n=re.subn(pat,'',stream,flags=re.S)
    assert n==1, (mcid,n)
d.update_stream(xref,stream.encode('latin1'))
panel=fitz.open(PPTPDF); a_box=fitz.Rect(3*MM,12.073*MM,93*MM,37.6*MM)
page.show_pdf_page(a_box,panel,0)
label_box=fitz.Rect(116*MM,86.2*MM,171*MM,94*MM)
page.add_redact_annot(label_box,fill=(1,1,1)); page.apply_redactions(images=fitz.PDF_REDACT_IMAGE_NONE)
font=os.environ["METAR_PLOT_ARIAL"]; text='Profile-drug pairing gain (percentage points)'; fs=8.5
tw=fitz.Font(fontfile=font).text_length(text,fontsize=fs)
page.insert_text((((116+171)*MM/2)-tw/2,89.8*MM),text,fontname='Arial',fontfile=font,fontsize=fs,color=(38/255,50/255,56/255),overlay=True)
out=WORK/'Figure6.pdf'; d.save(out,garbage=4,deflate=True)
od=fitz.open(out)
od[0].get_pixmap(dpi=180,alpha=False).save(WORK/'Figure6_178mm.png')
od[0].get_pixmap(dpi=600,alpha=False).save(WORK/'Figure6_178mm_600dpi.png')
