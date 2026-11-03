"""Expanded Figure 3; frozen a, reflowed statistics and recovery evidence."""
from pathlib import Path
import os
import hashlib
import json
import inspect
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pymupdf as pdf
import statistical_panels as old
import recovery as rec

OUT=Path(os.environ["METAR_PLOT_OUTPUT"])
BASE=Path(os.environ["METAR_PLOT_BASELINE"])
PT=72/25.4
old.DATA=OUT/'source_data'
old.PANELS=OUT/'panels'
old.BOTTOM_WIDTHS_MM={'c':68.,'d':84.,'e':84.}
old.BOTTOM_H_MM=51.5
old.configure_matplotlib()
plt.rcParams['svg.fonttype']='none'
# Reflow b's axes at original font sizes, with a shared method legend above.
code=inspect.getsource(old.build_panel_b)
code=code.replace('mm(PAGE_W_MM), mm(PANEL_B_H_MM)','mm(100), mm(55)')
code=code.replace('[0.125, 0.155, 0.855, 0.755]','[0.235, 0.155, 0.735, 0.755]')
code=code.replace('ax.legend(handles=method_handles(), ncol=3, frameon=False, fontsize=8, handlelength=1.35, handletextpad=0.45, columnspacing=1.25, loc="upper right", bbox_to_anchor=(0.99, 1.045), borderaxespad=0)','')
code=code.replace('add_panel_label(fig, "b", PAGE_W_MM, PANEL_B_H_MM)','add_panel_label(fig, "b", 100, 55)')
code=code.replace('save_panel(fig, "b", PAGE_W_MM, PANEL_B_H_MM)','save_panel(fig, "b", 100, 55)')
exec(code,old.__dict__)
# Keep every statistical point and existing axis-label roles, updating labels to latest baseline.
for name in ['build_panel_d','build_panel_e']:
    code=inspect.getsource(getattr(old,name))
    code=code.replace('Support groups per cell','Available reference groups' if name.endswith('_d') else 'Reference-set setting\\n(Groups)')
    exec(code,old.__dict__)
original_save=old.save_panel
def save_panel(fig,label,w,h):
    fig.savefig(OUT/'panels'/f'Figure3{label}.svg')
    return original_save(fig,label,w,h)
old.save_panel=save_panel
old.build_panel_b(old.read_tsv('Figure3_direct_comparison_fair_per_cell_hit10_source.tsv'))
old.build_panel_c(old.read_tsv('Figure3_direct_comparison_endpoint_estimates_95ci_source.tsv'))
old.build_panel_d(old.read_tsv('Figure3_direct_comparison_support_continuous_hit10_source.tsv'),old.read_tsv('Figure3_direct_comparison_support_strata_summary_source.tsv'))
old.build_panel_e(old.read_tsv('Figure3b_paired_delta_long.tsv'),old.read_tsv('Figure3b_paired_delta_summary.tsv'))
# Draw the supplied recovery panel with its final letter; no summary recomputation.
rec.OUT=OUT/'panels'
rec.setup_style()
code=inspect.getsource(rec.draw).replace('"d", fontsize=10.5','"f", fontsize=10.5')
code=code.replace('21 / HEIGHT_MM', '18 / HEIGHT_MM').replace('42 / HEIGHT_MM', '27 / HEIGHT_MM').replace('69 / HEIGHT_MM', '50.5 / HEIGHT_MM')
rec.HEIGHT_MM=55
exec(code,rec.__dict__)
rec.draw(pd.read_csv(OUT/'source_data/grouped_recovery_summary.tsv',sep='\t'))
# Shared legend has fixed 8 pt text; no panel-size or font rescaling.
fig=plt.figure(figsize=(178/25.4,7/25.4))
handles=old.method_handles()
handles[0].set_label('MeTar_direct')
fig.legend(handles=handles,loc='center',ncol=3,frameon=False,fontsize=8,handlelength=1.35,columnspacing=2)
fig.savefig(OUT/'panels/shared_legend.pdf');plt.close(fig)
placements={'a':[0,0,178,60], 'legend':[0,60,178,7], 'b':[0,66,100,55],'c':[107,66,68,51.5], 'd':[1.3,122,84,51.5],'e':[91,122,84,51.5],'f':[0,174,178,55]}
# c remains at native 68x56: center within its 65 mm row; never stretch.
placements['c']=[107,66,68,51.5]
paths={k:OUT/'panels'/f'Figure3{k}.pdf' for k in 'bcde'}
paths['legend']=OUT/'panels/shared_legend.pdf'
paths['f']=OUT/'panels/Adapted_cross_cell_recovery_panel.pdf'
with pdf.open(BASE) as baseline,pdf.open() as doc:
    page=doc.new_page(width=178*PT,height=229.5*PT)
    for key,xywh in placements.items():
        x,y,w,h=xywh;r=pdf.Rect(x*PT,y*PT,(x+w)*PT,(y+h)*PT)
        if key=='a':page.show_pdf_page(r,baseline,0,clip=pdf.Rect(0,0,178*PT,60*PT))
        else:
            with pdf.open(paths[key]) as p:page.show_pdf_page(r,p,0)
    doc.save(OUT/'Figure3.pdf',garbage=4,deflate=True)
    for dpi in [180,600]:page.get_pixmap(dpi=dpi,alpha=False).save(OUT/f'Figure3_{dpi}dpi.png')
    (OUT/'Figure3.svg').write_text(page.get_svg_image(text_as_path=False),encoding='utf-8')
    fonts=sorted({(s['font'],round(s['size'],2)) for b in page.get_text('dict')['blocks'] for l in b.get('lines',[]) for s in l['spans']})
