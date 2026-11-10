from docx import Document
from docx.shared import Pt,Cm
from docx.enum.text import WD_LINE_SPACING
from docx.oxml.ns import qn
p='C:/Users/BAI/Desktop/中期答辩/中期报告-白云飞-表达优化版.docx'
d=Document(p)
for i in [20,21]:
 f=d.paragraphs[i].paragraph_format;f.line_spacing_rule=WD_LINE_SPACING.EXACTLY;f.line_spacing=Pt(1);f.space_before=Pt(0);f.space_after=Pt(0)
f=d.paragraphs[22].paragraph_format
f.page_break_before=False;f.space_before=Pt(0);f.space_after=Pt(0);f.line_spacing_rule=WD_LINE_SPACING.EXACTLY;f.line_spacing=Pt(20)
for t in d.tables[:3]:
 for row in t.rows:
  for c in row.cells:
   for b in list(c._tc.xpath('./w:tcPr/w:tcBorders')):b.getparent().remove(b)
   for x in c.paragraphs:
    if x.text=='两个损失按以下方式组合：':x.paragraph_format.keep_with_next=True
for shape in d.inline_shapes:
 old=shape.width;shape.width=Cm(14);shape.height=int(shape.height*shape.width/old)
d.save(p)
