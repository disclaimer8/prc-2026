# JOAS paper draft

**Status: draft, not submitted.** `gentle-igloo-joas-draft.md` is a full research-article draft for the
Journal of Open Aviation Science (JOAS) describing team gentle-igloo's solution to the EUROCONTROL PRC
Data Challenge 2026; `main.tex` is the same text in the JOAS LaTeX form (`manuscript=article`), and
`reference.bib` holds the references; `OUTLINE.md` is the skeleton (thesis, contributions, the
evidence behind each section, figure and table lists, what is still missing). Every number in the
draft is copied from the repository documentation as of 9 October 2026 (`README.md`, `docs/*.md`,
`docs/tables/*.csv`, the dated reports in `analysis/`); figures are referenced from `docs/figures/`.
Numbers that did not exist when the draft was written are marked `[[RESULT: …]]`: the scores of the
two pending measurement uploads (v12079, v12080), the organiser's blind four-month ranking after
14 October 2026, and the author's front-matter fields (ORCID, e-mail, funding, AI-assistance
wording). The draft is then re-read against the docs and the placeholders removed.
JOAS accepts LaTeX only, built with its own template (`extra/joas.cls`, `main.tex`, `reference.bib`,
`figures/`; Overleaf gallery and <https://github.com/open-aviation/joas-template>); the class file is
not copied here because the GitHub repository declares no licence file. `main.tex` is generated from
the Markdown by `python3 paper/build_tex.py` (pandoc ≥ 3); to compile, place the official `extra/`
folder next to it and run `latexmk -pdf main.tex` (pdflatex + biber). On 8 October 2026 the generated
file compiled without errors against the official class (23 pages, preprint layout); biber was not
installed on the build machine, so the bibliography rendering is still unverified. Abstract ≤ 300
words, title ≤ 12 words, ≤ 5 keywords, mandatory *Open data statement* and *Reproducibility
statement*, CC BY 4.0, no fees, open peer review.
