"""Build the Word report from the same LaTeX sources used for the PDF.

Content lives in one place (report/latex/) and is converted with pandoc, so the
two documents cannot disagree. Three problems have to be handled because pandoc
does not implement the LaTeX page model:

* ``tabularx`` with an ``X`` column produces unusable column widths, so every
  tabularx environment is rewritten with explicit ``p{}`` widths before the
  conversion;
* ``\\chapter`` does not start a new page, so the Heading 1 style carries
  ``page_break_before``;
* the table of contents is emitted at the top of the document, so it is moved
  after the abstract where the departmental format expects it.

Usage:  python scripts/build_docx.py
"""
from __future__ import annotations

import re
import subprocess
from pathlib import Path

import pypandoc
from docx import Document
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.oxml import OxmlElement
from docx.oxml.ns import qn
from docx.shared import Cm, Pt, RGBColor

ROOT = Path(__file__).resolve().parents[1]
LATEX = ROOT / "report" / "latex"
FIGDIR = ROOT / "report" / "figures"
OUTDIR = ROOT / "report" / "docx"
BUILD = ROOT / "report" / "build"
OUT = OUTDIR / "Dry_Bean_Case_Study_Report.docx"
REFERENCE = BUILD / "reference.docx"

FONT = "Times New Roman"
TEXT_WIDTH_CM = 16.4          # A4 minus 2.54 cm left and 2.03 cm right margins


# --------------------------------------------------------------------------- #
# 1. flatten \input and give every table explicit column widths
# --------------------------------------------------------------------------- #
def flatten(tex: Path, seen: set[Path] | None = None) -> str:
    seen = seen or set()
    text = tex.read_text(encoding="utf-8")
    pattern = re.compile(r"\\input\{([^}]+)\}")

    def repl(match: re.Match) -> str:
        target = LATEX / match.group(1)
        if not target.suffix:
            target = target.with_suffix(".tex")
        if not target.exists():
            raise FileNotFoundError(f"missing \\input target: {target}")
        if target in seen:
            return ""
        seen.add(target)
        return flatten(target, seen)

    return pattern.sub(repl, text)


def _widths(spec: str) -> list[str]:
    """Turn a column spec into explicit centimetre widths that fill the page."""
    fixed, flexible = [], []
    for token in re.findall(r">\{[^}]*\}|p\{[^}]*\}|[lcrX]|@\{[^}]*\}", spec):
        if token.startswith("p{"):
            fixed.append(float(re.search(r"([\d.]+)", token).group(1)))
            flexible.append(False)
        elif token in "lcrX":
            fixed.append(1.9)
            flexible.append(token == "X")
    n = len(fixed)
    if n == 0:
        return [f"{TEXT_WIDTH_CM:.2f}cm"]
    used = sum(w for w, flex in zip(fixed, flexible) if not flex)
    n_flex = sum(flexible)
    if n_flex:
        share = max((TEXT_WIDTH_CM - used) / n_flex, 1.4)
        widths = [share if flex else w for w, flex in zip(fixed, flexible)]
    else:
        scale = TEXT_WIDTH_CM / used if used else 1
        widths = [w * scale for w in fixed]
    return [f"{w:.2f}cm" for w in widths]


def fix_tabularx(text: str) -> str:
    pattern = re.compile(
        r"\\begin\{tabularx\}\{[^}]*\}\{([^}]*)\}(.*?)\\end\{tabularx\}",
        re.S)

    def repl(match: re.Match) -> str:
        spec, body = match.group(1), match.group(2)
        widths = _widths(spec)
        new_spec = "".join(">{\\raggedright\\arraybackslash}p{" + w + "}"
                           for w in widths)
        return (f"\\begin{{tabular}}{{{new_spec}}}{body}"
                f"\\end{{tabular}}")

    return pattern.sub(repl, text)


# --------------------------------------------------------------------------- #
# 2. reference document
# --------------------------------------------------------------------------- #
def build_reference() -> None:
    BUILD.mkdir(parents=True, exist_ok=True)
    with open(REFERENCE, "wb") as fh:
        subprocess.run([pypandoc.get_pandoc_path(),
                        "--print-default-data-file", "reference.docx"],
                       stdout=fh, check=True)

    doc = Document(REFERENCE)
    for section in doc.sections:
        section.page_width, section.page_height = Cm(21.0), Cm(29.7)
        section.left_margin = Cm(2.54)
        section.right_margin = Cm(2.03)
        section.top_margin = Cm(2.03)
        section.bottom_margin = Cm(2.03)

    def style(name, size, bold=False, italic=False, spacing=1.5,
              align=WD_ALIGN_PARAGRAPH.JUSTIFY, before=0, after=8,
              page_break=False):
        try:
            st = doc.styles[name]
        except KeyError:
            return
        st.font.name = FONT
        st.font.size = Pt(size)
        st.font.bold = bold
        st.font.italic = italic
        st.font.color.rgb = RGBColor(0, 0, 0)
        rpr = st.element.get_or_add_rPr()
        rfonts = rpr.find(qn("w:rFonts"))
        if rfonts is None:
            rfonts = OxmlElement("w:rFonts")
            rpr.append(rfonts)
        for attr in ("w:ascii", "w:hAnsi", "w:cs", "w:eastAsia"):
            rfonts.set(qn(attr), FONT)
        pf = st.paragraph_format
        pf.line_spacing = spacing
        pf.alignment = align
        pf.space_before = Pt(before)
        pf.space_after = Pt(after)
        pf.page_break_before = page_break
        pf.keep_with_next = name.startswith("Heading")

    style("Normal", 12)
    style("Body Text", 12)
    style("First Paragraph", 12)
    style("Compact", 11)
    style("Block Text", 11, spacing=1.0)
    style("Source Code", 9, spacing=1.0, align=WD_ALIGN_PARAGRAPH.LEFT)
    style("Heading 1", 16, bold=True, spacing=1.5,
          align=WD_ALIGN_PARAGRAPH.CENTER, before=0, after=14,
          page_break=True)
    style("Heading 2", 14, bold=True, align=WD_ALIGN_PARAGRAPH.LEFT,
          before=14, after=6)
    style("Heading 3", 12, bold=True, align=WD_ALIGN_PARAGRAPH.LEFT,
          before=10, after=4)
    style("Heading 4", 12, bold=True, italic=True,
          align=WD_ALIGN_PARAGRAPH.LEFT, before=8, after=4)
    style("Image Caption", 10, italic=True, spacing=1.0,
          align=WD_ALIGN_PARAGRAPH.CENTER, after=12)
    style("Table Caption", 10, italic=True, spacing=1.0,
          align=WD_ALIGN_PARAGRAPH.CENTER, before=6, after=4)
    style("Title", 18, bold=True, spacing=1.5, align=WD_ALIGN_PARAGRAPH.CENTER)
    doc.save(REFERENCE)


# --------------------------------------------------------------------------- #
# 3. conversion
# --------------------------------------------------------------------------- #
def convert() -> None:
    source = fix_tabularx(flatten(LATEX / "main.tex"))
    flat = BUILD / "report_flat.tex"
    flat.write_text(source, encoding="utf-8")
    pypandoc.convert_file(
        str(flat), "docx", outputfile=str(OUT),
        extra_args=[
            f"--reference-doc={REFERENCE}",
            f"--resource-path={FIGDIR}:{LATEX}",
            "--toc", "--toc-depth=2", "--wrap=preserve",
        ],
    )


# --------------------------------------------------------------------------- #
# 4. post-processing
# --------------------------------------------------------------------------- #
def _is_heading(paragraph, level: int) -> bool:
    return paragraph.style.name == f"Heading {level}"


def move_toc(doc: Document) -> None:
    """Move the generated TOC block after the abstract.

    pandoc emits the TOC as a ``w:sdt`` content control at the very start of the
    body, before the title page. The departmental format wants it after the
    abstract, so the whole control is relocated.
    """
    body = doc.element.body
    toc = None
    for child in list(body):
        if child.tag.endswith("}sdt"):
            text = "".join(t.text or "" for t in child.iter(qn("w:t")))
            if "Table of Contents" in text:
                toc = child
                break
    if toc is None:
        return

    body.remove(toc)

    target = None
    for child in body:
        if not child.tag.endswith("}p"):
            continue
        ppr = child.find(qn("w:pPr"))
        if ppr is None:
            continue
        style = ppr.find(qn("w:pStyle"))
        if style is None or style.get(qn("w:val")) != "Heading1":
            continue
        text = "".join(t.text or "" for t in child.iter(qn("w:t")))
        if "INTRODUCTION" in text.upper():
            target = child
            break

    if target is not None:
        target.addprevious(toc)
    else:
        body.append(toc)


def add_caption_lists(doc: Document) -> None:
    """Clone the TOC control into a List of Figures and a List of Tables.

    Both are TOC fields restricted to caption paragraphs (``\\c "Figure"``),
    which is how Word builds these lists natively.
    """
    body = doc.element.body
    toc = None
    for child in body:
        if child.tag.endswith("}sdt"):
            text = "".join(t.text or "" for t in child.iter(qn("w:t")))
            if "Table of Contents" in text:
                toc = child
                break
    if toc is None:
        return

    import copy

    anchor = toc
    for heading, seq in (("LIST OF FIGURES", "Figure"),
                         ("LIST OF TABLES", "Table")):
        clone = copy.deepcopy(toc)
        for node in clone.iter(qn("w:instrText")):
            node.text = f'TOC \\h \\z \\c "{seq}"'
        for node in clone.iter(qn("w:t")):
            if node.text and "Table of Contents" in node.text:
                node.text = heading
        anchor.addnext(clone)
        anchor = clone


def enable_field_updates(doc: Document) -> None:
    settings = doc.settings.element
    node = settings.find(qn("w:updateFields"))
    if node is None:
        node = OxmlElement("w:updateFields")
        settings.append(node)
    node.set(qn("w:val"), "true")


def post_process() -> None:
    doc = Document(OUT)
    for section in doc.sections:
        section.page_width, section.page_height = Cm(21.0), Cm(29.7)
        section.left_margin = Cm(2.54)
        section.right_margin = Cm(2.03)
        section.top_margin = Cm(2.03)
        section.bottom_margin = Cm(2.03)

        footer = section.footer
        para = footer.paragraphs[0] if footer.paragraphs else footer.add_paragraph()
        para.alignment = WD_ALIGN_PARAGRAPH.CENTER
        if not para.text.strip():
            run = para.add_run()
            fld = OxmlElement("w:fldSimple")
            fld.set(qn("w:instr"), "PAGE")
            run._r.addnext(fld)

        header = section.header
        hpara = header.paragraphs[0] if header.paragraphs else header.add_paragraph()
        hpara.alignment = WD_ALIGN_PARAGRAPH.RIGHT
        if not hpara.text.strip():
            run = hpara.add_run("Dry Bean Variety Classification")
            run.font.name = FONT
            run.font.size = Pt(9)
            run.italic = True

    for table in doc.tables:
        tbl = table._tbl
        borders = OxmlElement("w:tblBorders")
        for edge in ("top", "left", "bottom", "right", "insideH", "insideV"):
            el = OxmlElement(f"w:{edge}")
            el.set(qn("w:val"), "single")
            el.set(qn("w:sz"), "4")
            el.set(qn("w:space"), "0")
            el.set(qn("w:color"), "808080")
            borders.append(el)
        tbl.tblPr.append(borders)
        table.autofit = False
        table.alignment = 1
        for row in table.rows:
            for cell in row.cells:
                for paragraph in cell.paragraphs:
                    paragraph.paragraph_format.line_spacing = 1.0
                    paragraph.paragraph_format.space_after = Pt(0)
                    for run in paragraph.runs:
                        run.font.size = Pt(9)

    move_toc(doc)
    add_caption_lists(doc)
    enable_field_updates(doc)
    doc.save(OUT)


def main() -> None:
    OUTDIR.mkdir(parents=True, exist_ok=True)
    build_reference()
    convert()
    post_process()
    print(f"wrote {OUT.relative_to(ROOT)} "
          f"({OUT.stat().st_size / 1024:.0f} KB)")


if __name__ == "__main__":
    main()
