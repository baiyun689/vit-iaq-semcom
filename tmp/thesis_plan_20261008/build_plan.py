from pathlib import Path
import re
from docx import Document
from docx.shared import Cm, Pt, RGBColor
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.enum.table import WD_TABLE_ALIGNMENT, WD_CELL_VERTICAL_ALIGNMENT
from docx.oxml import OxmlElement
from docx.oxml.ns import qn
from docx.opc.constants import RELATIONSHIP_TYPE as RT

BASE = Path(__file__).parent
OUT = BASE.parents[1] / 'docs' / '硕士毕业论文研究与写作计划书_20261008.docx'
lines = (BASE / 'plan_source.txt').read_text(encoding='utf-8').splitlines()
doc = Document()
sec = doc.sections[0]
sec.page_width, sec.page_height = Cm(21), Cm(29.7)
sec.top_margin, sec.bottom_margin = Cm(2.2), Cm(2.1)
sec.left_margin, sec.right_margin = Cm(2.3), Cm(2.3)
sec.header_distance, sec.footer_distance = Cm(1.0), Cm(1.0)

def set_font(style, size, east='宋体', bold=False):
    style.font.name = 'Times New Roman'
    style.font.size = Pt(size)
    style.font.bold = bold
    style.font.italic = False
    style.font.color.rgb = RGBColor(0, 0, 0)
    rpr = style.element.get_or_add_rPr()
    rf = rpr.find(qn('w:rFonts'))
    if rf is None:
        rf = OxmlElement('w:rFonts'); rpr.append(rf)
    rf.set(qn('w:eastAsia'), east)
    for attr in ['asciiTheme','hAnsiTheme','eastAsiaTheme','cstheme']:
        rf.attrib.pop(qn('w:'+attr),None)
    lang = OxmlElement('w:lang'); lang.set(qn('w:val'), 'zh-CN'); lang.set(qn('w:eastAsia'), 'zh-CN'); rpr.append(lang)

for name, size, east, bold in [('Normal',12,'宋体',False), ('Title',24,'黑体',True), ('Subtitle',13,'宋体',False), ('Heading 1',16,'黑体',True), ('Heading 2',13,'黑体',True), ('Heading 3',12,'黑体',True), ('List Bullet',12,'宋体',False), ('Footer',9,'宋体',False)]:
    set_font(doc.styles[name],size,east,bold)
normal = doc.styles['Normal'].paragraph_format
normal.line_spacing = Pt(19)
normal.space_after = Pt(6)
normal.widow_control = True
for name in ['Heading 1', 'Heading 2', 'Heading 3']:
    fmt = doc.styles[name].paragraph_format
    fmt.space_before, fmt.space_after = Pt(14), Pt(7)
    fmt.keep_with_next = True
    fmt.keep_together = True
    fmt.line_spacing = 1.15

for name in ['TOC 1','TOC Heading']:
    if name not in doc.styles:
        doc.styles.add_style(name,1)
    set_font(doc.styles[name], 11 if name=='TOC 1' else 14, '宋体' if name=='TOC 1' else '黑体', name!='TOC 1')
    doc.styles[name].paragraph_format.line_spacing = Pt(18)
    doc.styles[name].paragraph_format.space_after = Pt(1)

def field(p, code, default=''):
    r = p.add_run(); a=OxmlElement('w:fldChar'); a.set(qn('w:fldCharType'),'begin'); r._r.append(a)
    r=p.add_run(); a=OxmlElement('w:instrText'); a.set(qn('xml:space'),'preserve'); a.text=' '+code+' ';r._r.append(a)
    r=p.add_run(); a=OxmlElement('w:fldChar'); a.set(qn('w:fldCharType'),'separate');r._r.append(a)
    if default:p.add_run(default)
    r=p.add_run(); a=OxmlElement('w:fldChar');a.set(qn('w:fldCharType'),'end');r._r.append(a)

foot = sec.footer.paragraphs[0]
foot.alignment = WD_ALIGN_PARAGRAPH.CENTER
foot.add_run('第 ');field(foot,'PAGE','1');foot.add_run(' 页')
doc.core_properties.title='硕士毕业论文研究与写作计划书'
doc.core_properties.subject='重要性感知图像传输中的定位可靠性与分层量化修补'
doc.core_properties.author=''
doc.core_properties.keywords='硕士论文 研究计划 分层量化 可靠性 原生分辨率'

p=doc.add_paragraph(lines[0][2:],'Title');p.paragraph_format.space_after=Pt(10)
p=doc.add_paragraph(lines[1],'Subtitle');p.paragraph_format.space_after=Pt(12)
for line in lines[2:4]:
    p=doc.add_paragraph(line);p.paragraph_format.line_spacing=Pt(17);p.paragraph_format.space_after=Pt(2)
    for r in p.runs:r.font.size=Pt(10.5)
p=doc.add_paragraph('研究主线是先控制增强数据的位置与数值风险，再利用可信量化信息修正残余损伤。前四周先用原生分辨率强基线检验研究空间，随后分阶段完成两项方法、实验与论文。')
p.paragraph_format.space_before=Pt(10)
p.paragraph_format.space_after=Pt(12)
p=doc.add_paragraph('内容导航','TOC Heading')
field(doc.add_paragraph(), 'TOC \\o "1-1" \\h \\z \\u', '更新目录后显示页码')
for style in doc.styles:
    for el in list(style.element.iter(qn('w:pBdr'))):
        el.getparent().remove(el)

def add_link(text):
    p=doc.add_paragraph()
    p.paragraph_format.space_after=Pt(6)
    p.paragraph_format.line_spacing=Pt(13)
    link=OxmlElement('w:hyperlink')
    rid=doc.part.relate_to(text,RT.HYPERLINK,is_external=True)
    link.set(qn('r:id'),rid)
    r=OxmlElement('w:r'); props=OxmlElement('w:rPr')
    sz=OxmlElement('w:sz');sz.set(qn('w:val'),'19');props.append(sz)
    c=OxmlElement('w:color');c.set(qn('w:val'),'000000');props.append(c)
    r.append(props); t=OxmlElement('w:t');t.text=text;r.append(t);link.append(r);p._p.append(link)

def add_table(rows):
    cols=len(rows[0]);table=doc.add_table(rows=0,cols=cols)
    table.alignment=WD_TABLE_ALIGNMENT.CENTER;table.autofit=False
    if rows[0][0].startswith('周次'): widths=[4.1,6.2,6.1]
    elif cols==4: widths=[3.1,4.4,4.4,4.5]
    elif rows[0][0]=='模型':widths=[1.3,7.1,8.0]
    elif rows[0][0]=='项目':widths=[2.6,6.7,7.1]
    else: widths=[3.4,6.1,6.9]
    for col,w in zip(table.columns,widths):col.width=Cm(w)
    pr=table._tbl.tblPr
    borders=OxmlElement('w:tblBorders')
    for edge in ['top','left','bottom','right','insideH','insideV']:
        el=OxmlElement('w:'+edge);el.set(qn('w:val'),'single');el.set(qn('w:sz'),'4');el.set(qn('w:color'),'D9D9D9');borders.append(el)
    pr.append(borders)
    margin=OxmlElement('w:tblCellMar')
    for side,val in [('top',80),('left',105),('bottom',80),('right',105)]:
        e=OxmlElement('w:'+side);e.set(qn('w:w'),str(val));e.set(qn('w:type'),'dxa');margin.append(e)
    pr.append(margin)
    for j,vals in enumerate(rows):
        row=table.add_row()
        trpr=row._tr.get_or_add_trPr();trpr.append(OxmlElement('w:cantSplit'))
        if j==0:trpr.append(OxmlElement('w:tblHeader'))
        for k,(cell,val) in enumerate(zip(row.cells,vals)):
            cell.width=Cm(widths[k]);cell.vertical_alignment=WD_CELL_VERTICAL_ALIGNMENT.CENTER
            p=cell.paragraphs[0];p.paragraph_format.line_spacing=Pt(16)
            p.paragraph_format.space_before=Pt(0);p.paragraph_format.space_after=Pt(0)
            p.paragraph_format.keep_with_next = j==0
            p.alignment=WD_ALIGN_PARAGRAPH.CENTER if j==0 or (cols==4 and k>0) or (rows[0][0]=='模型' and k==0) else WD_ALIGN_PARAGRAPH.LEFT
            r=p.add_run(val);r.font.size=Pt(10.5);r.bold=j==0
            if j==0:
                sh=OxmlElement('w:shd');sh.set(qn('w:fill'),'E9EDF1');cell._tc.get_or_add_tcPr().append(sh)
    p=doc.add_paragraph();p.paragraph_format.space_after=Pt(2);p.paragraph_format.line_spacing=Pt(4)

i=4
while i<len(lines):
    line=lines[i].strip()
    if not line:i+=1;continue
    if line.startswith('|'):
        rows=[]
        while i<len(lines) and lines[i].strip().startswith('|'):
            vals=[x.strip() for x in lines[i].strip().strip('|').split('|')]
            if not all(re.fullmatch(r'[-:]+',x) for x in vals):rows.append(vals)
            i+=1
        add_table(rows);continue
    if line.startswith('### '):doc.add_paragraph(line[4:],'Heading 2')
    elif line.startswith('## '):
        p=doc.add_paragraph(line[3:],'Heading 1')
        if line.startswith('## 1 '):p.paragraph_format.page_break_before=True
    elif line.startswith('- '):
        p=doc.add_paragraph(line[2:],'List Bullet');p.paragraph_format.space_after=Pt(4)
    elif line.startswith('https://'):add_link(line)
    else:
        p=doc.add_paragraph(line)
        p.alignment=WD_ALIGN_PARAGRAPH.JUSTIFY
        if re.match(r'E\d\d ',line):
            for r in p.runs:r.text=''
            a,sep,b=line.partition('。');p.add_run(a+sep).bold=True;p.add_run(b)
        if re.match(r'\[\d+\]',line):
            p.paragraph_format.keep_with_next=True
            p.paragraph_format.line_spacing=Pt(16)
            for r in p.runs:r.font.size=Pt(10.5)
        if line.startswith('内部依据'):
            p.paragraph_format.keep_with_next=True
            for r in p.runs:r.bold=True
        if line.startswith('E:/'):
            p.paragraph_format.line_spacing=Pt(13);p.paragraph_format.space_after=Pt(6)
            for r in p.runs:r.font.size=Pt(9)
    i+=1

doc.save(OUT)
print(str(OUT))
print('paragraphs',len(doc.paragraphs),'tables',len(doc.tables),'source_characters',len('\n'.join(lines)))
