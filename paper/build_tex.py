#!/usr/bin/env python3
"""Generate paper/main.tex (JOAS LaTeX form) from paper/gentle-igloo-joas-draft.md.

The Markdown file is the editable source; main.tex is derived from it with pandoc and this
script. Run from the repository root or from paper/:

    python3 paper/build_tex.py            # writes paper/main.tex
    python3 paper/build_tex.py --preview  # also writes paper/preview/ and compiles a PDF preview

Requirements: pandoc >= 3. The official JOAS class (extra/joas.cls) is not part of this
repository because the template's GitHub repository declares no licence file; fetch it from
https://github.com/open-aviation/joas-template (or the Overleaf gallery) and place the
``extra/`` folder next to main.tex before compiling with ``latexmk -pdf main.tex``.

``--preview`` copies main.tex into paper/preview/, replaces the JOAS class by the standard
``article`` class, drops the JOAS-only front-matter macros, switches biblatex to the bibtex
backend (biber is not required) and runs ``latexmk -pdf`` if it is installed. The result,
paper/preview/gentle-igloo-joas-preview.pdf, is a reading copy only: it is NOT the JOAS layout.
"""
from __future__ import annotations

import re
import shutil
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
MD = HERE / "gentle-igloo-joas-draft.md"
OUT = HERE / "main.tex"
PREVIEW_DIR = HERE / "preview"
PREVIEW_NAME = "gentle-igloo-joas-preview"

PREAMBLE = r"""%% main.tex -- GENERATED from gentle-igloo-joas-draft.md by build_tex.py; edit the Markdown.
%% Journal of Open Aviation Science (JOAS) article form. Requires the official class file
%% extra/joas.cls from https://github.com/open-aviation/joas-template (not bundled here).
\documentclass[
  manuscript=article,  %% article (default), rescience, data, software, proceedings, poster
  layout=preprint,     %% preprint (for submission) or publish (for publisher only)
  year=2026,
  volume=4,
]{extra/joas}

\doi{xx.xxxxx/joas.xxxx.xxxx}

\received {1 January 2026}
\revised  {1 January 2026}
\accepted {1 January 2026}
\published{1 January 2026}

\editor{Editor Name}
\reviewers{First Reviewer, Second Reviewer}

\addbibresource{reference.bib}

%% pandoc table/figure support (harmless if the class already loads these)
\usepackage{longtable}
\usepackage{booktabs}
\usepackage{array}
\usepackage{calc}
\usepackage{xcolor}
\usepackage{amsmath}
\providecommand{\tightlist}{\setlength{\itemsep}{0pt}\setlength{\parskip}{0pt}}
\providecommand{\real}[1]{#1}
\providecommand{\passthrough}[1]{#1}
\providecommand{\pandocbounded}[1]{#1}
\newcommand{\placeholder}[1]{\textcolor{red}{[[RESULT:#1]]}}
\graphicspath{{figures/}}

% --- below is the area for authors ---

% Title within 12 words
\title{A Recording-Regime Mixture for Taxi-Out Time Reconstruction at Ten European Airports}

\author{Denys Kolomiiets \orcid{0000-0000-0000-0000}}  % [[RESULT: ORCID]]
\affiliation{Independent researcher, team gentle-igloo}
\email{correspondence@email.domain}  % [[RESULT: contact e-mail]]

% maximum five keywords
\keywords{taxi-out time; recording regimes; ADS-B surface tracks; gradient boosting; open aviation data}

\abbreviations{
    NM: Network Manager,
    RMSE: root-mean-square error,
    PRC: Performance Review Commission,
    LIRF: Rome Fiumicino airport (ICAO code),
    ADS-B: Automatic Dependent Surveillance--Broadcast
}

\begin{document}

"""

POSTAMBLE = r"""
\printbibliography

\end{document}
"""


def run_pandoc() -> str:
    cmd = [
        "pandoc", str(MD), "-f", "markdown", "-t", "latex", "--biblatex", "--wrap=none",
    ]
    return subprocess.run(cmd, check=True, capture_output=True, text=True).stdout


def strip_numbers(tex: str) -> str:
    # "\section{1 Introduction}" -> "\section{Introduction}"; "\subsection{2.1 Challenge data}" -> ...
    return re.sub(r"\\(sub)*section\{(\d+(\.\d+)*)\s+", lambda m: "\\" + (m.group(1) or "") + "section{", tex)


def match_brace(s: str, i: int) -> int:
    """Index just after the brace group that opens at s[i] == '{' (escaped braces are skipped)."""
    assert s[i] == "{", s[i:i + 20]
    depth = 0
    j = i
    while j < len(s):
        c = s[j]
        if c == "\\":
            j += 2
            continue
        if c == "{":
            depth += 1
        elif c == "}":
            depth -= 1
            if depth == 0:
                return j + 1
        j += 1
    raise ValueError("unbalanced braces from %d" % i)


def number_figures_and_tables(body: str) -> str:
    r"""Turn the Markdown's literal "Figure N." / "Table N." captions into LaTeX captions with labels.

    The Markdown numbers figures and tables by hand, in document order. LaTeX numbers floats itself,
    so the literal prefixes are stripped, every figure/table gets \label{fig:N}/\label{tab:N}, the
    text-only stage diagram becomes a figure float (so the numbering stays aligned), and the
    references in the text become "Figure~\ref{fig:N}" / "Table~\ref{tab:N}" as JOAS asks.
    """
    figs, tabs = set(), set()

    # Image figures: \caption{Figure N. text} -> \caption{text}\label{fig:N}
    out, pos = [], 0
    for m in re.finditer(r"\\caption\{Figure (\d+)\. ", body):
        if m.start() < pos:
            continue
        end = match_brace(body, m.start() + len("\\caption"))
        text = body[m.end():end - 1]
        out.append(body[pos:m.start()] + "\\caption{%s}\\label{fig:%s}" % (text, m.group(1)))
        figs.add(m.group(1))
        pos = end
    body = "".join(out) + body[pos:]

    # Text figure (the stage diagram): verbatim block followed by \emph{Figure N. text}.
    m = re.search(r"(\{\\scriptsize\\begin\{verbatim\}.*?\\end\{verbatim\}\})\s*\n\s*\n\\emph\{Figure (\d+)\. ", body, re.S)
    if m:
        cap_open = m.end() - len("Figure %s. " % m.group(2)) - 1
        end = match_brace(body, cap_open)
        text = body[m.end():end - 1]
        float_ = ("\\begin{figure}[htbp]\n\\centering\n" + m.group(1) + "\n\\caption{%s}\\label{fig:%s}\n\\end{figure}"
                  % (text, m.group(2)))
        body = body[:m.start()] + float_ + body[end:]
        figs.add(m.group(2))

    # Tables: "\textbf{Table N. text}" paragraph followed by a longtable -> caption inside the longtable.
    while True:
        m = re.search(r"\\textbf\{Table (\d+)\. ", body)
        if not m:
            break
        n = m.group(1)
        cap_end = match_brace(body, m.start() + len("\\textbf"))
        text = body[m.end():cap_end - 1]
        lt = body.index("\\begin{longtable}[]", cap_end)
        between = body[cap_end:lt]
        if between.strip() not in ("", "{"):
            raise ValueError("Table %s: unexpected text between caption and table: %r" % (n, between[:80]))
        spec_start = lt + len("\\begin{longtable}[]")
        spec_end = match_brace(body, spec_start)
        head_end = body.index("\\endhead", spec_end)
        header = body[spec_end:head_end]
        new = (body[spec_start - len("\\begin{longtable}[]"):spec_end]
               + "\n\\caption{%s}\\label{tab:%s}\\tabularnewline" % (text, n)
               + header + "\\endfirsthead\n\\caption[]{(continued)}\\tabularnewline"
               + header + "\\endhead")
        body = body[:m.start()] + between.lstrip() + new + body[head_end + len("\\endhead"):]
        tabs.add(n)

    # References in the text.
    def fig_ref(m: re.Match) -> str:
        if m.group(1) not in figs:
            print("warning: reference to unknown Figure %s" % m.group(1), file=sys.stderr)
        return "Figure~\\ref{fig:%s}%s" % (m.group(1), m.group(2) or "")

    def tab_ref(m: re.Match) -> str:
        if m.group(1) not in tabs:
            print("warning: reference to unknown Table %s" % m.group(1), file=sys.stderr)
        return "Table~\\ref{tab:%s}" % m.group(1)

    body = re.sub(r"\bFigure (\d+)([ab])?\b", fig_ref, body)
    body = re.sub(r"\bTable (\d+)\b", tab_ref, body)
    print("figures labelled:", len(figs), "tables labelled:", len(tabs))
    return body


PREVIEW_PREAMBLE = r"""%% PREVIEW ONLY -- generated by build_tex.py --preview from main.tex. NOT the JOAS layout:
%% the standard article class replaces extra/joas.cls, JOAS front-matter macros are dropped,
%% and biblatex uses the bibtex backend (numeric, unsorted, as in the JOAS class).
\documentclass[11pt,a4paper]{article}
\usepackage[T1]{fontenc}
\usepackage[utf8]{inputenc}
\usepackage{lmodern}
\usepackage[margin=2.3cm]{geometry}
\usepackage{graphicx}
\usepackage{parskip}
\usepackage[font=small,labelfont=bf,labelsep=period]{caption}
\usepackage{csquotes}
\usepackage[backend=bibtex,style=numeric,maxbibnames=10,sorting=none]{biblatex}
\usepackage[hidelinks]{hyperref}
\newcommand{\orcid}[1]{}
\newcommand{\previewaffiliation}{}
\newcommand{\affiliation}[1]{\renewcommand{\previewaffiliation}{#1}}
\newcommand{\previewemail}{}
\newcommand{\email}[1]{\renewcommand{\previewemail}{#1}}
\newcommand{\previewkeywords}{}
\newcommand{\keywords}[1]{\renewcommand{\previewkeywords}{#1}}
\newcommand{\abbreviations}[1]{}
\date{Reading preview generated from \texttt{main.tex}; not the JOAS layout}
"""


def make_preview(tex: str) -> int:
    """Write paper/preview/<PREVIEW_NAME>.tex (article class) and compile it if latexmk exists."""
    i = tex.index("\\documentclass")
    j = tex.index("{extra/joas}") + len("{extra/joas}")
    tex = tex[:i] + PREVIEW_PREAMBLE + tex[j:]
    for macro in ["doi", "received", "revised", "accepted", "published", "editor", "reviewers"]:
        tex = re.sub(r"^\\%s\s*\{.*?\}\s*$" % macro, "", tex, flags=re.M)
    tex = tex.replace("\\graphicspath{{figures/}}", "\\graphicspath{{../figures/}}")
    tex = tex.replace("\\begin{document}\n", "\\begin{document}\n\\maketitle\n"
                      "\\noindent\\textit{\\previewaffiliation}\\par\n", 1)
    tex = tex.replace("\\end{abstract}\n", "\\end{abstract}\n\n\\noindent\\textbf{Keywords:} \\previewkeywords\\par\n", 1)
    PREVIEW_DIR.mkdir(exist_ok=True)
    src = PREVIEW_DIR / (PREVIEW_NAME + ".tex")
    src.write_text(tex)
    shutil.copyfile(HERE / "reference.bib", PREVIEW_DIR / "reference.bib")
    print("wrote", src)
    latexmk = shutil.which("latexmk")
    if not latexmk or not shutil.which("pdflatex"):
        print("latexmk/pdflatex not installed: preview .tex written, PDF not compiled", file=sys.stderr)
        return 0
    cmd = [latexmk, "-pdf", "-interaction=nonstopmode", "-halt-on-error", src.name]
    r = subprocess.run(cmd, cwd=PREVIEW_DIR, capture_output=True, text=True)
    (PREVIEW_DIR / "latexmk.out").write_text(r.stdout + r.stderr)
    pdf = PREVIEW_DIR / (PREVIEW_NAME + ".pdf")
    if r.returncode != 0 or not pdf.exists():
        print("latexmk failed (see paper/preview/latexmk.out)", file=sys.stderr)
        return 1
    # Keep only the preview source and the PDF; everything else is regenerated on the next run.
    for f in PREVIEW_DIR.iterdir():
        if f.name not in (PREVIEW_NAME + ".tex", PREVIEW_NAME + ".pdf"):
            f.unlink()
    print("wrote", pdf)
    return 0


def main() -> int:
    tex = run_pandoc()

    # Draft notice (blockquote before the abstract) and abstract.
    m_abs = re.search(r"\\section\{Abstract\}.*?\n", tex)
    m_intro = re.search(r"\\section\{1 Introduction\}", tex)
    if not (m_abs and m_intro):
        print("could not locate Abstract / Introduction headings", file=sys.stderr)
        return 1
    notice = tex[: m_abs.start()].strip()
    abstract = tex[m_abs.end(): m_intro.start()].strip()
    body = tex[m_intro.start():]

    # Back matter: unnumbered sections and the bibliography.
    for name in [
        "Acknowledgements", "Funding statement", "Open data statement",
        "Reproducibility statement", "AI-assistance statement",
    ]:
        body = body.replace("\\section{%s}" % name, "\\section*{%s}" % name)
    body = re.sub(r"\\section\{References\}.*\Z", "", body, flags=re.S)

    body = strip_numbers(body)

    # Longtables: pandoc wraps caption-less tables in "{\def\LTcaptype{none} ... }", which the
    # JOAS class rejects ("No counter 'none' defined"); keep the grouping braces only.
    body = body.replace("{\\def\\LTcaptype{none} % do not increment counter", "{")
    # The ASCII stage diagram is wider than the text block at normal size.
    body = body.replace("\\begin{verbatim}", "{\\scriptsize\\begin{verbatim}").replace("\\end{verbatim}", "\\end{verbatim}}")

    # Figures: keep only the file name; all figures live in paper/figures/ (\graphicspath).
    body = re.sub(r"\\includegraphics(\[[^\]]*\])?\{figures/", r"\\includegraphics[width=\\textwidth]{", body)
    body = number_figures_and_tables(body)

    # Long file paths in \texttt{} overflow the margin: allow line breaks after "/" and "_".
    body = re.sub(r"\\texttt\{[^{}]*\}",
                  lambda m: m.group(0).replace("/", "/\\allowbreak{}").replace("\\_", "\\_\\allowbreak{}"), body)

    # Display maths -> equation environment (JOAS asks for numbered equations).
    body = body.replace("\\[", "\\begin{equation}").replace("\\]", "\\end{equation}")

    # [[RESULT: ...]] placeholders -> red text.
    ph = re.compile(r"\{\[\}\{\[\}RESULT:(.*?)\{\]\}\{\]\}", re.S)
    body = ph.sub(lambda m: "\\placeholder{%s}" % m.group(1), body)
    abstract = ph.sub(lambda m: "\\placeholder{%s}" % m.group(1), abstract)
    notice = ph.sub(lambda m: "\\placeholder{%s}" % m.group(1), notice)

    # pdflatex (the engine of the JOAS Makefile) has no glyphs for these characters.
    unicode_map = {
        "−": "$-$", "≤": "$\\le$", "≥": "$\\ge$", "→": "$\\rightarrow$",
        "×": "$\\times$", "±": "$\\pm$", "…": "\\ldots{}", "·": "$\\cdot$",
        "ρ": "$\\rho$", "σ": "$\\sigma$", "τ": "$\\tau$", "²": "\\textsuperscript{2}",
        "⁻⁵": "\\textsuperscript{$-$5}", "©": "\\textcopyright{}", "≈": "$\\approx$",
        "⁻¹³": "\\textsuperscript{$-$13}", "°": "\\textdegree{}", "Δ": "$\\Delta$",
    }
    for k, v in unicode_map.items():
        body = body.replace(k, v)
        abstract = abstract.replace(k, v)
        notice = notice.replace(k, v)
    # "$-$" inside maths (e.g. "$\le$ 7,200") is harmless; "$-$0.17" in tables renders as a minus sign.
    bad = sorted({ch for ch in body + abstract + notice if ord(ch) > 127 and ch not in "éèüöä–—’“”"})
    if bad:
        print("warning: non-ASCII characters without a mapping:", bad, file=sys.stderr)

    # The draft notice is kept in main.tex as a comment only, so the file is ready for submission.
    notice = re.sub(r"^\\begin\{quote\}|\\end\{quote\}$", "", notice, flags=re.M).strip()
    notice_comment = "\n".join("%% " + line for line in notice.splitlines())

    out = (
        PREAMBLE
        + "%% Draft notice (from the Markdown source; not typeset):\n" + notice_comment + "\n\n"
        + "\\begin{abstract}\n" + abstract + "\n\\end{abstract}\n\n"
        + body.strip() + "\n" + POSTAMBLE
    )
    OUT.write_text(out)
    print("wrote", OUT, len(out.splitlines()), "lines")
    if "--preview" in sys.argv[1:]:
        return make_preview(out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
