"""Build review/brief.html from REVIEW_BRIEF.md so the page and the markdown never diverge."""
import re
import shutil
import subprocess
from pathlib import Path

try:
    import markdown
except ModuleNotFoundError:
    markdown = None


def render(text):
    if markdown is not None:
        return markdown.markdown(text, extensions=["tables", "fenced_code", "sane_lists"])
    # The paper build already uses Pandoc; reuse it without modifying .venv.
    pandoc = shutil.which("pandoc")
    if not pandoc:
        raise RuntimeError("Building the brief requires Python markdown or existing Pandoc")
    return subprocess.run(
        [pandoc, "--from=markdown+pipe_tables+fenced_code_blocks-auto_identifiers-smart",
         "--to=html", "--wrap=none"],
        input=text, text=True, capture_output=True, check=True,
    ).stdout.strip()

root = Path(__file__).resolve().parent.parent
md = (root / "REVIEW_BRIEF.md").read_text()
page = (root / "review" / "brief.html").read_text()

body_md = md[md.index("\n## ") + 1:]  # from the first numbered section (0. already checked)
html = render(body_md)

# 01.10: the header of the page comes from the markdown too (it had been frozen at 16.09): the "state as of" lede, the status
# strip from the status table row, and the "read first" block (### before the first ##) as its own section
head_md = md[:md.index("\n## ")]
lede_line = next(l for l in head_md.splitlines() if "Состояние на" in l).lstrip("> ").strip()
lede_html = render(lede_line)[3:-4]  # strip <p></p>
row = next(l for l in head_md.splitlines() if l.startswith("| **") and "|" in l[3:])
cells = [c.strip() for c in row.strip().strip("|").split("|")]
bold = lambda c: (re.search(r"\*\*(.+?)\*\*", c) or re.search(r"(.+)", c)).group(1)
plain = lambda c, n: re.sub(r"\*\*|`", "", re.sub(r"\s*\(.*", "", c))[:n]
deadline = re.search(r"\*\*(\d{2}\.\d{2}) (\d{2}:\d{2})Z\*\*", head_md)
if not deadline:
    raise ValueError("Missing explicit UTC deadline in the brief header")
strip_html = (
    '<div class="strip" role="group" aria-label="Статус">\n'
    f'      <div class="hl"><div class="k">Лидерборд</div><div class="v">{bold(cells[0])}</div><div class="s">{(re.search(r"\(v\d+", cells[0]) or re.search(r"(.)", "?")).group(0).lstrip("(")} — лучший файл</div></div>\n'
    f'      <div><div class="k">Место</div><div class="v">{bold(cells[1])}</div><div class="s">{re.sub(r"[*]", "", cells[1].split("(", 1)[1].rstrip(")"))[:90] if "(" in cells[1] else ""}</div></div>\n'
    f'      <div><div class="k">Финальная подача</div><div class="v">{plain(cells[2].split(",")[0], 30)}</div><div class="s">{plain(cells[3], 90)}</div></div>\n'
    f'      <div><div class="k">Дедлайн</div><div class="v">{deadline.group(1)}</div><div class="s">{deadline.group(2)} UTC, репозиторий и документация</div></div>\n'
    '    </div>')
first_md = head_md[head_md.index("### Читать первым"):] if "### Читать первым" in head_md else ""
first_html = render(first_md.replace("### Читать первым", "## Ч. Читать первым", 1)) if first_md else ""
html = first_html + html
# Markdown links are rooted at the repository; this HTML lives one directory below.
def local_link(m):
    target = m.group(2)
    if target.startswith(("#", "/")) or re.match(r"[A-Za-z][A-Za-z0-9+.-]*:", target):
        return m.group(0)
    return f'{m.group(1)}../{target}{m.group(3)}'

html = re.sub(r'(\b(?:href|src)=")([^"]+)(")', local_link, html)

# section anchors and table containers
toc = []
def h2(m):
    title = re.sub(r"<[^>]+>", "", m.group(1))
    num, _, name = title.partition(". ")
    anchor = f"s{num}"
    toc.append((anchor, num, name))
    return f'</section><section id="{anchor}"><h2><span class="n">{num}</span>{name}</h2>'
html = re.sub(r"<h2>(.*?)</h2>", h2, html)
html = html.replace("</section>", "", 1) + "</section>"
html = html.replace("<table>", '<div class="table-wrap"><table>').replace("</table>", "</table></div>")
for word, cls in [("исправлено, v8", "ok"), ("обход в v8", "ok"), ("исправлено (гигиена)", "ok"), ("в работе", "warn"), ("запланировано", "info")]:
    html = html.replace(f"<td>{word}</td>", f'<td><span class="chip {cls}">{word}</span></td>')
    html = html.replace(f"<td><strong>{word}</strong></td>", f'<td><span class="chip {cls}">{word}</span></td>')

toc_html = "\n".join(f'      <li><a href="#{a}">{n}</a></li>' for a, _, n in toc)
head, rest = page.split("<ol>", 1)
_, rest = rest.split("</ol>", 1)
page = head + "<ol>\n" + toc_html + "\n    </ol>" + rest

start = page.index('<section id="')  # first content section, whatever its id
end = page.index('<footer class="doc">')
page = page[:start] + html + "\n\n    " + page[end:]
page = re.sub(r'<p class="lede">.*?</p>', lambda m: f'<p class="lede">{lede_html}</p>', page, count=1, flags=re.S)
page = re.sub(r'<div class="strip" role="group" aria-label="Статус">.*?\n    </div>', lambda m: strip_html, page, count=1, flags=re.S)
(root / "review" / "brief.html").write_text(page)
print(f"Built review/brief.html: {len(toc)} sections")
