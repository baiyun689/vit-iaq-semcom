from docx import Document
from docx.shared import Pt
from docx.enum.text import WD_LINE_SPACING
p='C:/Users/BAI/Desktop/中期答辩/中期报告-白云飞-表达优化版.docx'
d=Document(p)
for i,x in enumerate(d.paragraphs):
    print(i,repr(x.text[:20]),x.paragraph_format.page_break_before,[(e.tag.split('}')[-1],dict(e.attrib)) for e in x._p.xpath('.//w:br')])
    if i>=30 and not x.text.strip():
        for child in list(x._p):
            if not child.tag.endswith('}pPr'):x._p.remove(child)
        f=x.paragraph_format;f.page_break_before=False;f.keep_with_next=False
        f.line_spacing_rule=WD_LINE_SPACING.EXACTLY;f.line_spacing=Pt(1)
        f.space_before=Pt(0);f.space_after=Pt(0)
for t in d.tables[:3]:
    for row in t.rows:
        for c in row.cells:
            for x in c.paragraphs:
                x.paragraph_format.page_break_before=False
                for e in list(x._p.xpath('.//w:br')):
                    if e.get('{http://schemas.openxmlformats.org/wordprocessingml/2006/main}type')=='page': e.getparent().remove(e)
                if x.text.startswith('本文将图像块'):pass
                if x.text.startswith('目前已搭建'):
                    for r in x.runs:
                        r.text=r.text.replace('本文将图像块统一称为“图像块”，仅在网络结构说明中保留 patch 这一英文术语。','下文中的 patch 即图像块。')
                if x.text.startswith('信道采用二进制'):
                    x.runs[0].text=x.text+'图中 Eb/N0 表示每比特能量与噪声功率谱密度之比，是这些实验所用的信噪比口径。'
                if x.text.startswith('图 3 使用原实验'):
                    x.runs[0].text=x.text.replace('原实验中的 256 张图像','原实验中 3 dB 条件下的 256 张图像')
                if x.text.startswith('网络结构见图 5'):
                    x.runs[0].text=x.text+'图中带波浪线的 x 表示修补前图像，带帽的 x 表示修补后图像；跳跃连接用于保留输入并与修正量相加。'
                if x.text.startswith('六组对照分别为'):
                    x.runs[0].text=x.text+'图 7 中 B = 392 表示各块比特深度之和；ρ = 0.25 表示载荷预算相对于每个像素通道值使用 8 bit 的比例。该比例不包含元信息和纠错冗余，不能代替总信道预算核算。'
d.save(p)
