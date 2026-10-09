#!/usr/bin/env python3
"""Generate paper/main.tex (JOAS LaTeX form) from paper/gentle-igloo-joas-draft.md.

The Markdown file is the editable source; main.tex is derived from it with pandoc and this
script. Run from the repository root or from paper/:

    python3 paper/build_tex.py

Requirements: pandoc >= 3. The official JOAS class (extra/joas.cls) is not part of this
repository because the template's GitHub repository declares no licence file; fetch it from
https://github.com/open-aviation/joas-template (or the Overleaf gallery) and place the
``extra/`` folder next to main.tex before compiling with ``latexmk -pdf main.tex``.
"""
from __future__ import annotations

import re
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
MD = HERE / "gentle-igloo-joas-draft.md"
OUT = HERE / "main.tex"

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
\graphicspath{{../docs/figures/}{figures/}}

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

    # Figures: keep only the file name; the class expects figures/ and we add ../docs/figures/ to the path.
    body = re.sub(r"\\includegraphics(\[[^\]]*\])?\{\.\./docs/figures/", r"\\includegraphics[width=\\textwidth]{", body)
    body = body.replace("\\includegraphics[width=\\textwidth][width=\\textwidth]", "\\includegraphics[width=\\textwidth]")

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

    # Longtables: pandoc wraps caption-less tables in "{\def\LTcaptype{none} ... }", which the
    # JOAS class rejects ("No counter 'none' defined"); keep the grouping braces only.
    body = body.replace("{\\def\\LTcaptype{none} % do not increment counter", "{")
    # The ASCII stage diagram is wider than the text block at normal size.
    body = body.replace("\\begin{verbatim}", "{\\scriptsize\\begin{verbatim}").replace("\\end{verbatim}", "\\end{verbatim}}")

    out = (
        PREAMBLE
        + "\\begin{abstract}\n" + abstract + "\n\\end{abstract}\n\n"
        + "\\noindent\\textcolor{red}{\\textbf{Draft status.}} " + re.sub(r"^\\begin\{quote\}|\\end\{quote\}$", "", notice, flags=re.M).strip()
        + "\n\n" + body.strip() + "\n" + POSTAMBLE
    )
    OUT.write_text(out)
    print("wrote", OUT, len(out.splitlines()), "lines")
    return 0


if __name__ == "__main__":
    sys.exit(main())
