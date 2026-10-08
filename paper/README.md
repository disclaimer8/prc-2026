# JOAS paper draft

**Status: draft, not submitted.** `gentle-igloo-joas-draft.md` is a full research-article draft for the
Journal of Open Aviation Science (JOAS) describing team gentle-igloo's solution to the EUROCONTROL PRC
Data Challenge 2026; `main.tex` is the same text in the JOAS LaTeX form (`manuscript=article`), and
`reference.bib` holds the references. Every number in the draft is copied from the repository
documentation as of 8 October 2026 (`README.md`, `docs/model-design.md`, `docs/development-history.md`,
`docs/data-insights.md`, `docs/validation.md`, `docs/negative-results.md`, `docs/reproduction.md`,
`SOURCES.md`, `docs/tables/*.csv`); figures are referenced from `docs/figures/`. Numbers that did not
exist when the draft was written are marked `[[RESULT: …]]` and will be filled from
`docs/generalisation.md`, `docs/uncertainty.md`, `docs/data-quality-regimes.md`, `docs/pru-view.md` and
the organiser's final four-month ranking after 14 October 2026; the draft is then re-read against the
docs, the placeholders removed, and the author confirms the AI-assistance, funding and ORCID fields.
JOAS accepts LaTeX only, built with its own template (`extra/joas.cls`, `main.tex`, `reference.bib`,
`figures/`; Overleaf gallery and <https://github.com/open-aviation/joas-template>); the class file is
not copied here because the GitHub repository declares no licence file. `main.tex` is generated from
the Markdown by `python3 paper/build_tex.py` (pandoc ≥ 3); to compile, place the official `extra/`
folder next to it and run `latexmk -pdf main.tex` (pdflatex + biber). On 8 October 2026 the generated
file compiled without errors against the official class (23 pages, preprint layout); biber was not
installed on the build machine, so the bibliography rendering is still unverified. Abstract ≤ 300
words, title ≤ 12 words, ≤ 5 keywords, mandatory *Open data statement* and *Reproducibility
statement*, CC BY 4.0, no fees, open peer review.
