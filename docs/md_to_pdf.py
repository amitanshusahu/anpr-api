#!/usr/bin/env python3
"""Markdown -> Beautiful PDF generator (ReportLab).

Usage:
    python md_to_pdf.py INPUT.md OUTPUT.pdf [--accent "#0EA5E9"] [--subtitle "..."]
    python md_to_pdf.py --batch docs/  # converts all *.md in dir

Features: cover page, styled headings, tables, quotes, code, lists,
header/footer with page numbers, TOC-ish scene cards.
"""
import argparse
import html
import os
import re
import sys
from datetime import date

from reportlab.lib import colors
from reportlab.lib.enums import TA_LEFT, TA_CENTER
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import mm
from reportlab.platypus import (BaseDocTemplate, Frame, HRFlowable, KeepTogether,
                                ListFlowable, ListItem, PageBreak, PageTemplate,
                                Paragraph, Spacer, Table, TableStyle)

PAGE_W, PAGE_H = A4

DEFAULT_ACCENT = "#0EA5E9"
DARK = colors.HexColor("#0F172A")
MUTED = colors.HexColor("#64748B")
LIGHT_BG = colors.HexColor("#F1F5F9")


def hexc(s):
    return colors.HexColor(s)


def inline_md(text):
    t = html.escape(text.strip())
    t = re.sub(r'`([^`]+?)`', r'<font face="Courier" backColor="#F1F5F9" color="#0F172A">&nbsp;\1&nbsp;</font>', t)
    t = re.sub(r'\*\*([^*]+?)\*\*', r'<b>\1</b>', t)
    t = re.sub(r'(?<!\w)__([^_]+?)__(?!\w)', r'<b>\1</b>', t)
    t = re.sub(r'(?<!\w)\*([^*\n]+?)\*(?!\w)', r'<i>\1</i>', t)
    t = re.sub(r'\[([^\]]+?)\]\(([^)]+?)\)', r'<a href="\2" color="#0284C2">\1</a>', t)
    return t


def make_styles(accent):
    ACC = hexc(accent)
    ACC_DARK = hexc("#0C4A6E") if accent == DEFAULT_ACCENT else DARK
    ss = getSampleStyleSheet()
    s = {}
    s['h1'] = ParagraphStyle('h1', parent=ss['Heading1'], fontName='Helvetica-Bold',
                             fontSize=19, leading=24, textColor=DARK, spaceBefore=14, spaceAfter=6)
    s['h2'] = ParagraphStyle('h2', parent=ss['Heading2'], fontName='Helvetica-Bold',
                             fontSize=14, leading=18, textColor=ACC_DARK, spaceBefore=12, spaceAfter=5,
                             borderPadding=(0, 0, 4, 0))
    s['h3'] = ParagraphStyle('h3', parent=ss['Heading3'], fontName='Helvetica-Bold',
                             fontSize=11.5, leading=15, textColor=DARK, spaceBefore=9, spaceAfter=4)
    s['body'] = ParagraphStyle('body', parent=ss['BodyText'], fontName='Helvetica',
                               fontSize=9.6, leading=14.5, textColor=hexc("#1E293B"), spaceAfter=5)
    s['bullet'] = ParagraphStyle('bullet', parent=s['body'], leftIndent=16, bulletIndent=4, spaceAfter=3)
    s['quote'] = ParagraphStyle('quote', parent=s['body'], leftIndent=12, textColor=hexc("#334155"),
                                borderPadding=(6, 8, 6, 8), fontName='Helvetica-Oblique', fontSize=9.4)
    s['code'] = ParagraphStyle('code', parent=ss['Code'], fontName='Courier', fontSize=8.3,
                               leading=12, textColor=hexc("#1E293B"), backColor=LIGHT_BG,
                               borderPadding=(6, 6, 6, 6), spaceAfter=6)
    s['codeblock_title'] = ParagraphStyle('cbt', parent=s['body'], fontName='Helvetica-Bold',
                                          fontSize=8, textColor=MUTED, spaceAfter=1)
    s['table_cell'] = ParagraphStyle('tc', parent=ss['BodyText'], fontName='Helvetica',
                                     fontSize=8.6, leading=12, textColor=hexc("#1E293B"))
    s['table_head'] = ParagraphStyle('th', parent=s['table_cell'], fontName='Helvetica-Bold',
                                     textColor=colors.white)
    s['caption'] = ParagraphStyle('cap', parent=ss['BodyText'], fontName='Helvetica-Bold',
                                  fontSize=8, textColor=MUTED, alignment=TA_CENTER, spaceAfter=8)
    s['cover_title'] = ParagraphStyle('ct', fontName='Helvetica-Bold', fontSize=30, leading=34,
                                      textColor=colors.white, alignment=TA_LEFT)
    s['cover_sub'] = ParagraphStyle('cs', fontName='Helvetica', fontSize=11, leading=16,
                                    textColor=hexc("#E0F2FE"), alignment=TA_LEFT)
    s['cover_meta'] = ParagraphStyle('cm', fontName='Helvetica', fontSize=8.5, leading=12,
                                     textColor=hexc("#BAE6FD"), alignment=TA_LEFT)
    s['kicker'] = ParagraphStyle('k', fontName='Helvetica-Bold', fontSize=8.5, leading=11,
                                 textColor=ACC, alignment=TA_LEFT, spaceAfter=6)
    s['scene_tag'] = ParagraphStyle('stag', fontName='Helvetica-Bold', fontSize=8,
                                    textColor=colors.white, alignment=TA_CENTER)
    return s, ACC


def header_footer(canvas, doc):
    canvas.saveState()
    if doc.page > 1:
        canvas.setStrokeColor(LIGHT_BG)
        canvas.setLineWidth(0.6)
        canvas.line(18 * mm, PAGE_H - 14 * mm, PAGE_W - 15 * mm, PAGE_H - 14 * mm)
        canvas.setFont("Helvetica", 7)
        canvas.setFillColor(MUTED)
        canvas.drawString(18 * mm, PAGE_H - 11.5 * mm, doc._doctitle[:90])
        canvas.setFont("Helvetica", 7)
        canvas.setFillColor(MUTED)
        canvas.drawRightString(PAGE_W - 15 * mm, 12 * mm, f"{doc.page - 1}")
        canvas.setFont("Helvetica-Bold", 6.5)
        canvas.setFillColor(hexc(doc._accent))
        canvas.drawString(15 * mm, 12 * mm, "SIH 26127  •  CITY-WIDE AI ENGINE")
    canvas.restoreState()


def cover_header(canvas, doc):
    canvas.saveState()
    canvas.setFillColor(DARK)
    canvas.rect(0, 0, PAGE_W, PAGE_H, fill=1, stroke=0)
    ACC = hexc(getattr(doc, '_accent', DEFAULT_ACCENT))
    canvas.setFillColor(ACC)
    canvas.rect(0, PAGE_H - 9 * mm, PAGE_W, 9 * mm, fill=1, stroke=0)
    canvas.setFillColor(hexc("#16294A"))
    canvas.circle(PAGE_W - 20 * mm, 52 * mm, 55 * mm, fill=1, stroke=0)
    canvas.setFillColor(hexc("#13253F"))
    canvas.circle(18 * mm, 88 * mm, 30 * mm, fill=1, stroke=0)
    canvas.setFillColor(ACC)
    canvas.setFillAlpha(0.9)
    canvas.rect(15 * mm, 46 * mm, 14 * mm, 1.4 * mm, fill=1, stroke=0)
    canvas.restoreState()


def all_pages(canvas, doc):
    if doc.page == 1:
        cover_header(canvas, doc)
    else:
        header_footer(canvas, doc)


def split_table_row(line):
    return [c.strip() for c in line.strip().strip('|').split('|')]


def md_table_to_flowable(rows, styles, ACC):
    hdr = split_table_row(rows[0])
    body_rows = [split_table_row(r) for r in rows[2:]]
    ncols = len(hdr)
    avail = PAGE_W - 30 * mm
    col_w = [avail / ncols] * ncols
    if ncols >= 3:
        col_w = [avail * 0.24, avail * 0.52, avail * 0.24][:ncols]
    data = [[Paragraph(f"<b>{inline_md(c)}</b>", styles['table_head']) for c in hdr]]
    for r in body_rows:
        r += [''] * (ncols - len(r))
        data.append([Paragraph(inline_md(c), styles['table_cell']) for c in r[:ncols]])
    t = Table(data, colWidths=col_w, repeatRows=1)
    t.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), DARK),
        ('TEXTCOLOR', (0, 0), (-1, 0), colors.white),
        ('BACKGROUND', (0, 1), (-1, -1), colors.white),
        ('ROWBACKGROUNDS', (0, 1), (-1, -1), [colors.white, hexc("#F8FAFC")]),
        ('GRID', (0, 0), (-1, -1), 0.5, hexc("#CBD5E1")),
        ('VALIGN', (0, 0), (-1, -1), 'TOP'),
        ('TOPPADDING', (0, 0), (-1, -1), 5),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 5),
        ('LEFTPADDING', (0, 0), (-1, -1), 6),
        ('RIGHTPADDING', (0, 0), (-1, -1), 6),
        ('LINEBELOW', (0, 0), (-1, 0), 2, ACC),
    ]))
    return t


def quote_block(lines, styles, ACC):
    txt = ' '.join(l.lstrip('> ').strip() for l in lines)
    inner = Paragraph(inline_md(txt), styles['quote'])
    t = Table([[inner]], colWidths=[PAGE_W - 30 * mm - 4])
    t.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, -1), hexc("#F0F9FF")),
        ('BOX', (0, 0), (-1, -1), 0.6, hexc("#BAE6FD")),
        ('LINEBELOW', (0, 0), (-1, 0), 0, colors.white),
        ('LEFTPADDING', (0, 0), (-1, -1), 8),
        ('RIGHTPADDING', (0, 0), (-1, -1), 8),
        ('TOPPADDING', (0, 0), (-1, -1), 6),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 6),
        ('LINEBEFORE', (0, 0), (0, -1), 3.5, ACC),
    ]))
    return t


def scene_card(title, styles, ACC):
    tag = Table([[Paragraph("SCENE", styles['scene_tag'])]], colWidths=[22 * mm])
    tag.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, -1), ACC),
        ('ROUNDEDCORNERS', [3, 3, 3, 3]),
        ('TOPPADDING', (0, 0), (-1, -1), 3),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 3),
    ]))
    head = Table([[tag, Paragraph(inline_md(title), styles['h2'])]], colWidths=[24 * mm, PAGE_W - 30 * mm - 24 * mm])
    head.setStyle(TableStyle([('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
                              ('LEFTPADDING', (0, 0), (-1, -1), 0)]))
    return head


def field_row(label, text, styles, ACC):
    lab = Paragraph(f'<font color="{ACC}"><b>{label}</b></font>', styles['body'])
    val = Paragraph(text, styles['body'])
    t = Table([[lab, val]], colWidths=[24 * mm, PAGE_W - 30 * mm - 24 * mm])
    t.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, -1), hexc("#F8FAFC")),
        ('BOX', (0, 0), (-1, -1), 0.5, hexc("#E2E8F0")),
        ('VALIGN', (0, 0), (-1, -1), 'TOP'),
        ('TOPPADDING', (0, 0), (-1, -1), 5),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 5),
        ('LEFTPADDING', (0, 0), (-1, -1), 6),
        ('RIGHTPADDING', (0, 0), (-1, -1), 6),
    ]))
    return t


def parse_markdown(md_text, styles, ACC):
    lines = md_text.splitlines()
    story = []
    i = 0
    in_code = False
    code_buf = []
    title_seen = False

    def flush_code():
        nonlocal code_buf
        if code_buf:
            story.append(Paragraph(html.escape('\n'.join(code_buf)).replace('\n', '<br/>'), styles['code']))
            story.append(Spacer(1, 2 * mm))
            code_buf = []

    while i < len(lines):
        ln = lines[i]
        s = ln.strip()

        if s.startswith('```'):
            if in_code:
                flush_code()
                in_code = False
            else:
                in_code = True
            i += 1
            continue
        if in_code:
            code_buf.append(ln)
            i += 1
            continue
        if not s:
            i += 1
            continue
        if re.match(r'^---+\s*$', s) or re.match(r'^\*\*\*+\s*$', s):
            story.append(Spacer(1, 2 * mm))
            story.append(HRFlowable(width="100%", thickness=0.7, color=hexc("#CBD5E1")))
            story.append(Spacer(1, 3 * mm))
            i += 1
            continue
        if s.startswith('|') and i + 1 < len(lines) and re.match(r'^\|?[\s:\-|]+\|?\s*$', lines[i + 1].strip()):
            tbl = [s, lines[i + 1]]
            i += 2
            while i < len(lines) and lines[i].strip().startswith('|'):
                tbl.append(lines[i].strip())
                i += 1
            story.append(md_table_to_flowable(tbl, styles, ACC))
            story.append(Spacer(1, 3 * mm))
            continue
        m = re.match(r'^(#{1,4})\s+(.*)', s)
        if m:
            lvl, txt = len(m.group(1)), m.group(2).strip()
            if lvl == 1 and not title_seen:
                title_seen = True
                i += 1
                continue
            if lvl == 2 and txt.upper().startswith('SCENE'):
                story.append(Spacer(1, 2 * mm))
                story.append(scene_card(txt, styles, ACC))
                story.append(HRFlowable(width="100%", thickness=1.2, color=ACC,
                                        spaceBefore=2, spaceAfter=4))
            elif lvl == 2:
                story.append(HRFlowable(width="18%", thickness=2.2, color=ACC,
                                        spaceBefore=6, spaceAfter=1, hAlign='LEFT'))
                story.append(Paragraph(inline_md(txt), styles['h2']))
            elif lvl == 3:
                story.append(Paragraph(inline_md(txt), styles['h3']))
            else:
                story.append(Paragraph(inline_md(txt), styles['h2']))
            i += 1
            continue
        if s.startswith('>'):
            q = [ln]
            i += 1
            while i < len(lines) and lines[i].strip().startswith('>'):
                q.append(lines[i])
                i += 1
            story.append(quote_block(q, styles, ACC))
            story.append(Spacer(1, 2 * mm))
            continue
        mf = re.match(r'^\*\*(ON SCREEN|DO|CHECK|SAY|5[A-C].*|[A-Z /]{3,}):\*\*\s*(.*)', s)
        if mf:
            label, rest = mf.group(1), mf.group(2)
            nxt = []
            if rest:
                nxt.append(rest)
            i += 1
            while i < len(lines) and lines[i].strip() and not re.match(
                    r'^(\#{1,4}\s|>\s*"?\|?-{3}|\*\*(ON SCREEN|DO|CHECK|SAY))', lines[i].strip()) \
                    and not lines[i].strip().startswith(('|', '-', '*', '1.', '2.', '3.', '4.', '5.', '6.')):
                nxt.append(lines[i].strip())
                i += 1
            body = ' '.join(nxt)
            if label == 'SAY':
                story.append(field_row('◉ SAY', f'<i>"{inline_md(body)}"</i>' if body else '', styles, ACC))
            else:
                story.append(field_row(label, inline_md(body), styles, ACC))
            story.append(Spacer(1, 1.5 * mm))
            continue
        if re.match(r'^(\d+)\.\s+', s):
            items = []
            while i < len(lines) and re.match(r'^(\d+)\.\s+', lines[i].strip()):
                items.append(Paragraph(inline_md(re.sub(r'^\d+\.\s+', '', lines[i].strip())), styles['body']))
                i += 1
            lf = ListFlowable([ListItem(p, leftIndent=16, bulletColor=ACC) for p in items],
                              bulletType='1', start='1', leftIndent=16)
            story.append(lf)
            story.append(Spacer(1, 2 * mm))
            continue
        if re.match(r'^[-*]\s+', s):
            items = []
            while i < len(lines) and re.match(r'^[-*]\s+', lines[i].strip()):
                items.append(Paragraph(inline_md(re.sub(r'^[-*]\s+', '', lines[i].strip())), styles['body']))
                i += 1
            lf = ListFlowable([ListItem(p, leftIndent=16, bulletColor=ACC,
                                        value='●') for p in items], bulletType='bullet', leftIndent=16)
            story.append(lf)
            story.append(Spacer(1, 2 * mm))
            continue
        para = [s]
        i += 1
        while i < len(lines) and lines[i].strip() and not re.match(
                r'^(#{1,4}\s|>\s*"?\||\|?-{3,}\s*$|\*\*|```|\d+\.\s+|[-*]\s+|\|)', lines[i].strip()):
            para.append(lines[i].strip())
            i += 1
        story.append(Paragraph(inline_md(' '.join(para)), styles['body']))
    return story


def extract_title(md_text, fallback):
    for ln in md_text.splitlines():
        m = re.match(r'^#\s+(.*)', ln.strip())
        if m:
            return m.group(1).strip()
    return fallback


def build_pdf(md_path, pdf_path, accent=DEFAULT_ACCENT, subtitle=None, kicker=None):
    with open(md_path, encoding='utf-8') as f:
        md_text = f.read()
    styles, ACC = make_styles(accent)
    title = extract_title(md_text, os.path.splitext(os.path.basename(md_path))[0])
    body = parse_markdown(md_text, styles, ACC)

    doc = BaseDocTemplate(pdf_path, pagesize=A4,
                          leftMargin=15 * mm, rightMargin=15 * mm,
                          topMargin=18 * mm, bottomMargin=15 * mm,
                          title=title, author="SIH 26127")
    doc._doctitle = title
    doc._accent = accent

    frame = Frame(doc.leftMargin, doc.bottomMargin, doc.width, doc.height, id='main')
    tpl = PageTemplate(id='all', frames=[frame], onPage=all_pages)
    doc.addPageTemplates([tpl])

    story = []
    # ---- Cover ----
    story.append(Spacer(1, 6 * mm))
    story.append(Paragraph((kicker or "SIH 2026  •  PROBLEM STATEMENT 26127").upper(), styles['kicker']))
    story.append(Paragraph(html.escape(title), styles['cover_title']))
    # accent bar
    bar = Table([['']], colWidths=[32 * mm], rowHeights=[1.6 * mm])
    bar.setStyle(TableStyle([('BACKGROUND', (0, 0), (-1, -1), ACC),
                             ('ROUNDEDCORNERS', [2, 2, 2, 2])]))
    story.append(Spacer(1, 4 * mm))
    story.append(bar)
    story.append(Spacer(1, 4 * mm))
    if subtitle:
        story.append(Paragraph(html.escape(subtitle), styles['cover_sub']))
        story.append(Spacer(1, 3 * mm))
    meta = (f"{date.today().strftime('%d %B %Y')} &nbsp;&nbsp;•&nbsp;&nbsp; "
            f"{html.escape(os.path.basename(md_path))} &nbsp;&nbsp;•&nbsp;&nbsp; City-Wide AI Engine")
    story.append(Paragraph(meta, styles['cover_meta']))
    story.append(Spacer(1, 8 * mm))
    info = [
        [Paragraph("<b><font color='#0F172A'>Stack</font></b><br/><font color='#475569'>PREFINAL.pptx + main.py + api.py</font>", styles['body']),
         Paragraph("<b><font color='#0F172A'>Source</font></b><br/><font color='#475569'>docs/ • Markdown → PDF</font>", styles['body']),
         Paragraph("<b><font color='#0F172A'>Team</font></b><br/><font color='#475569'>SIH 26127 • Prefinal</font>", styles['body'])],
    ]
    it = Table(info, colWidths=[(PAGE_W - 30 * mm) / 3] * 3)
    it.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, -1), colors.white),
        ('BOX', (0, 0), (-1, -1), 0.6, hexc("#E2E8F0")),
        ('INNERGRID', (0, 0), (-1, -1), 0.4, hexc("#E2E8F0")),
        ('TOPPADDING', (0, 0), (-1, -1), 8),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 8),
        ('LEFTPADDING', (0, 0), (-1, -1), 8),
    ]))
    story.append(it)
    story.append(PageBreak())
    story.extend(body)
    # closing strip
    story.append(Spacer(1, 6 * mm))
    strip = Table([[Paragraph("<b><font color='white'>End of document • Generated from Markdown • SIH 26127</font></b>",
                              ParagraphStyle('e', parent=styles['body'], textColor=colors.white,
                                             alignment=TA_CENTER, fontSize=8))]],
                  colWidths=[PAGE_W - 30 * mm])
    strip.setStyle(TableStyle([('BACKGROUND', (0, 0), (-1, -1), DARK),
                               ('ROUNDEDCORNERS', [4, 4, 4, 4]),
                               ('TOPPADDING', (0, 0), (-1, -1), 8),
                               ('BOTTOMPADDING', (0, 0), (-1, -1), 8)]))
    story.append(strip)
    doc.build(story)
    return pdf_path


def main():
    ap = argparse.ArgumentParser(description="Markdown -> Beautiful PDF")
    ap.add_argument("input", nargs="?", help="input .md file")
    ap.add_argument("output", nargs="?", help="output .pdf file")
    ap.add_argument("--accent", default=DEFAULT_ACCENT)
    ap.add_argument("--subtitle", default=None)
    ap.add_argument("--kicker", default=None)
    ap.add_argument("--batch", default=None, help="convert all *.md in directory")
    a = ap.parse_args()
    if a.batch:
        for fn in sorted(os.listdir(a.batch)):
            if fn.lower().endswith('.md'):
                src = os.path.join(a.batch, fn)
                dst = os.path.join(a.batch, os.path.splitext(fn)[0] + '.pdf')
                build_pdf(src, dst, accent=a.accent)
                print(f"OK {dst}")
        return
    if not a.input or not a.output:
        ap.error("provide INPUT.md OUTPUT.pdf or use --batch DIR")
    build_pdf(a.input, a.output, accent=a.accent, subtitle=a.subtitle, kicker=a.kicker)
    print(f"OK {a.output}")


if __name__ == '__main__':
    main()
