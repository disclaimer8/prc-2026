# JOAS paper draft

**Status: draft, not submitted (updated through 13 October 2026; package checked 14 October).** `gentle-igloo-joas-draft.md` is a full
research-article draft for the Journal of Open Aviation Science (JOAS) describing team gentle-igloo's
solution to the EUROCONTROL PRC Data Challenge 2026. The organiser said on 9 October 2026 that a JOAS
paper is not required but welcome.

| File | What it is |
|---|---|
| `gentle-igloo-joas-draft.md` | The editable source of the paper (Markdown, pandoc citations) |
| `main.tex` | The same text in the JOAS LaTeX form (`manuscript=article`), **generated** by `build_tex.py`; do not edit by hand |
| `reference.bib` | References |
| `figures/` | Every figure the paper uses: ten copied unchanged from `docs/figures/`, two drawn for the paper (`nn_weight_by_month.png`, `final_recipe_price.png`) |
| `make_paper_figures.py` | Rebuilds `figures/` (copies the reused figures, draws the two new ones from `analysis/e212_nn_weight_10seed_20261008/per_month.json`, `analysis/v12080_control_explanation_20261009/results/*.json` and the score and forecast files of `analysis/v1208{1,2,3}_*_20261011/`) |
| `build_tex.py` | Markdown → `main.tex`; with `--preview` also the PDF preview below |
| `preview/gentle-igloo-joas-preview.pdf` | **Reading preview only — not the JOAS layout** (see below) |
| `OUTLINE.md` | Skeleton: thesis, contributions, the evidence behind each section, figure and table lists, what is still missing |

Every number in the draft is copied from the repository documentation (`README.md`, `docs/*.md`,
`docs/tables/*.csv`, the dated reports in `analysis/`, `HYPOTHESES_LOG.md`, including protocols 463–464);
`OUTLINE.md` names the source of each claim. Numbers that do not exist yet are marked `[[RESULT: …]]`
and typeset in red. The current placeholders cover the organiser's blind final ranking after 14 October 2026 (six
places, all marked `[[RESULT: final ranking — …]]`) and the DOI of a public archive if one is created.

The 12 October wording review distinguishes agreement with the regime-labelling rule from
independently verified recording errors, and indicator sensitivity from a causal correction to
physical taxi-out. The registered airport-day analysis is now integrated in §§4.5 and 5.9: it
measures residual association in existing 2025 regime probabilities, with conditional inference
and no conversion into RMSE uncertainty. Planning-match controls, tolerance sensitivity and
ADS-B comparisons are integrated in §5.10, including the inconclusive strict raw-episode check.
The independent arithmetic audits and the separately labelled post hoc raw-admission diagnostic
are archived with the results (experiment-register entries 463–466). The generated LaTeX and
reading preview were rebuilt on 13 October 2026, including the blind E218 scenarios,
dither-stability scenarios and concurrent-work discussion (register entries 467–471).
`OUTLINE.md` records the evidence and insertion locations.

## Build

```bash
/path/to/prc-2026/.venv/bin/python paper/make_paper_figures.py   # figures/ (matplotlib; never `uv run`)
python3 paper/build_tex.py                                        # main.tex (needs pandoc >= 3)
python3 paper/build_tex.py --preview                              # main.tex + preview/ PDF (needs latexmk, pdflatex, bibtex)
```

`build_tex.py` strips the hand-written "Figure N." / "Table N." caption prefixes of the Markdown,
gives every figure and table a LaTeX caption and label (`fig:N`, `tab:N`; the text stage diagram
becomes a figure float so the numbering stays aligned), turns the references in the text into
`Figure~\ref{…}` / `Table~\ref{…}` as the JOAS template asks, and keeps the draft-status notice
only as a comment in `main.tex`.

## Two different PDFs

- **JOAS layout.** JOAS accepts LaTeX only, built with its own template (`extra/joas.cls`, `main.tex`,
  `reference.bib`, `figures/`; Overleaf gallery and <https://github.com/open-aviation/joas-template>).
  The class file is not copied here because the template's GitHub repository declares no licence
  file. To compile, place the official `extra/` folder next to `main.tex` and run `make pdf` of the
  template (pdflatex + biber) or `latexmk -pdf main.tex`. On 10 October 2026 the generated `main.tex`
  compiled without errors against the official class (40 pages, preprint layout); biber is not
  installed on the build machine, so in that build the bibliography is empty and its JOAS rendering is
  still unverified.
- **Preview (`preview/gentle-igloo-joas-preview.pdf`).** A reading copy of the same `main.tex` with
  the standard `article` class instead of the JOAS class, the JOAS-only front-matter macros
  (`\doi`, `\received`, `\editor`, `\reviewers`, `\orcid`, `\affiliation`, `\email`, `\keywords`,
  `\abbreviations`) dropped or reduced to plain text, and biblatex on the bibtex backend (numeric,
  unsorted, as in the JOAS class) so that the references resolve without biber. It is **not the JOAS
  layout**: fonts, margins, page count, abstract box and header differ. On 13 October 2026 it has
  52 pages (11 pt, A4).

## JOAS limits (venue rules checked 11 October; manuscript counts 12 October 2026)

Title 11 words (≤ 12); five keywords (≤ 5); abstract 270 whitespace-delimited words counting every unit symbol and the
placeholder as a word, 264 without the placeholder (≤ 300); mandatory *Open data
statement* and *Reproducibility statement* present (the latter names the scripts behind every figure
and the sources of every table, as the template asks); *Funding statement* and *AI-assistance
statement* written; single author, so no CRediT section. CC BY 4.0, no fees, open peer review.
