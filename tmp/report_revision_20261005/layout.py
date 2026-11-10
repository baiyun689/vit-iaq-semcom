from copy import deepcopy
from docx import Document
from docx.oxml import OxmlElement
from docx.oxml.ns import qn
from docx.shared import Pt
p='C:/Users/BAI/Desktop/中期答辩/中期报告-白云飞-表达优化版.docx'
d=Document(p)
d.paragraphs[22].paragraph_format.keep_with_next=False
d.paragraphs[23].paragraph_format.page_break_before=True
for table in d.tables[:3]:
    content=[]
    template=deepcopy(table.rows[-1].cells[0]._tc.tcPr)
    for row in table.rows:
        for x in row.cells[0].paragraphs: content.append(deepcopy(x._p))
    for tr in list(table._tbl.tr_lst): table._tbl.remove(tr)
    borders=table._tbl.tblPr.find(qn('w:tblBorders'))
    inner=borders.find(qn('w:insideH'))
    inner.set(qn('w:val'),'nil')
    for para in content:
        row=OxmlElement('w:tr');cell=OxmlElement('w:tc');cell.append(deepcopy(template));cell.append(para);row.append(cell);table._tbl.append(row)
    # Start official report sections on their own page, keeping the form boundaries.
    prev=table._tbl.getprevious()
    if prev is not None and prev.tag==qn('w:p'):
        from docx.text.paragraph import Paragraph
        Paragraph(prev,d._body).paragraph_format.page_break_before=True
        Paragraph(prev,d._body).paragraph_format.keep_with_next=True
# Keep the complete teacher and examination signature form on its own page.
from docx.text.paragraph import Paragraph
prev=d.tables[3]._tbl.getprevious()
Paragraph(prev,d._body).paragraph_format.page_break_before=True
Paragraph(prev,d._body).paragraph_format.keep_with_next=True
d.save(p)
