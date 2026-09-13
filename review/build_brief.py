"""Build review/brief.html from REVIEW_BRIEF.md so the page and the markdown never diverge."""
import re
from pathlib import Path

import markdown

root = Path(__file__).resolve().parent.parent
md = (root / "REVIEW_BRIEF.md").read_text()
page = (root / "review" / "brief.html").read_text()

body_md = md[md.index("## 1. Задача"):]
html = markdown.markdown(body_md, extensions=["tables", "fenced_code", "sane_lists"])

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
(root / "review" / "brief.html").write_text(page)
print("sections:", [n for _, _, n in toc])
