"""docs/labelling.md as HTML for the page's guide panel.

A small Markdown subset, enough for that one file: headings, paragraphs,
bullet and numbered lists (one level of nesting), tables, block quotes,
fenced code, and inline bold, italic, code and links. Everything is escaped
first, so the doc cannot inject markup into the page.
"""

from __future__ import annotations

import html
import re

_INLINE = [
    (re.compile(r"\*\*([^*]+)\*\*"), r"<strong>\1</strong>"),
    (re.compile(r"(?<![\w*])\*([^*\s][^*]*)\*(?![\w*])"), r"<em>\1</em>"),
]
LINK = re.compile(r"\[([^\]]+)\]\(([^)\s]+)\)")
# Sibling docs are not served by the page; link them on GitHub instead.
DOCS_URL = "https://github.com/chostudio/snow/blob/main/docs/"


def _link(m: re.Match) -> str:
    text, href = m.group(1), m.group(2)
    if href.startswith("#"):
        return f'<a href="{href}" data-anchor="{href[1:]}">{text}</a>'
    if not re.match(r"^https?://", href):
        href = DOCS_URL + href
    return f'<a href="{href}" target="_blank" rel="noopener">{text}</a>'


def slug(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", "-", text.lower()).strip("-")


def inline(text: str) -> str:
    # Code spans first, kept out of the other rules by a placeholder.
    codes: list[str] = []

    def keep(m: re.Match) -> str:
        codes.append(m.group(1))
        return f"\x00{len(codes) - 1}\x00"

    out = re.sub(r"`([^`]+)`", keep, html.escape(text, quote=False))
    for pattern, repl in _INLINE:
        out = pattern.sub(repl, out)
    out = LINK.sub(_link, out)
    return re.sub(r"\x00(\d+)\x00", lambda m: f"<code>{codes[int(m.group(1))]}</code>", out)


def _table(lines: list[str]) -> str:
    def cells(line: str) -> list[str]:
        return [c.strip() for c in line.strip().strip("|").split("|")]

    head, body = cells(lines[0]), [cells(x) for x in lines[2:]]
    out = ["<table><thead><tr>", *(f"<th>{inline(c)}</th>" for c in head), "</tr></thead><tbody>"]
    for row in body:
        out += ["<tr>", *(f"<td>{inline(c)}</td>" for c in row), "</tr>"]
    return "".join(out + ["</tbody></table>"])


def to_html(text: str) -> str:
    lines = text.splitlines()
    out: list[str] = []
    i = 0
    while i < len(lines):
        line = lines[i]
        stripped = line.strip()
        if not stripped:
            i += 1
            continue
        if stripped.startswith("```"):
            j = i + 1
            while j < len(lines) and not lines[j].strip().startswith("```"):
                j += 1
            out.append("<pre><code>" + html.escape("\n".join(lines[i + 1:j])) + "</code></pre>")
            i = j + 1
            continue
        m = re.match(r"^(#{1,4})\s+(.*)$", stripped)
        if m:
            level, title = len(m.group(1)), m.group(2)
            out.append(f'<h{level} id="{slug(title)}">{inline(title)}</h{level}>')
            i += 1
            continue
        if stripped.startswith("|") and i + 1 < len(lines) and re.match(r"^\|?[\s:|-]+\|?$",
                                                                       lines[i + 1].strip()):
            j = i
            while j < len(lines) and lines[j].strip().startswith("|"):
                j += 1
            out.append(_table(lines[i:j]))
            i = j
            continue
        if stripped.startswith(">"):
            j = i
            quote = []
            while j < len(lines) and lines[j].strip().startswith(">"):
                quote.append(lines[j].strip()[1:].strip())
                j += 1
            out.append(f"<blockquote>{to_html(chr(10).join(quote))}</blockquote>")
            i = j
            continue
        if re.match(r"^([-*]|\d+\.)\s", stripped):
            ordered = bool(re.match(r"^\d+\.", stripped))
            items: list[list[str]] = []
            j = i
            while j < len(lines) and lines[j].strip():
                cur = lines[j]
                if re.match(r"^([-*]|\d+\.)\s", cur):
                    items.append([re.sub(r"^([-*]|\d+\.)\s+", "", cur)])
                elif items:
                    items[-1].append(cur)
                j += 1
            parts = []
            for item in items:
                first, rest = item[0], item[1:]
                nested = [r for r in rest if re.match(r"^\s+([-*]|\d+\.)\s", r)]
                text_lines = [first] + [r.strip() for r in rest if r not in nested]
                body = inline(" ".join(text_lines))
                if nested:
                    body += to_html("\n".join(r.strip() for r in nested))
                parts.append(f"<li>{body}</li>")
            tag = "ol" if ordered else "ul"
            out.append(f"<{tag}>{''.join(parts)}</{tag}>")
            i = j
            continue
        j = i
        para = []
        while j < len(lines) and lines[j].strip() and not re.match(
                r"^(#{1,4}\s|```|\||>|[-*]\s|\d+\.\s)", lines[j].strip()):
            para.append(lines[j].strip())
            j += 1
        out.append(f"<p>{inline(' '.join(para))}</p>")
        i = j
    return "\n".join(out)
