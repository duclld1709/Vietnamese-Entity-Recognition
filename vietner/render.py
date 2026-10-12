"""HTML highlighting for the demo UI."""

from __future__ import annotations

import html

from vietner.schemas import Entity

# Light backgrounds keep dark text readable; identity is also given by the label badge.
COLORS = {
    "PER": "#fde2e1",
    "ORG": "#dbeafe",
    "LOC": "#dcfce7",
    "MISC": "#f3e8ff",
}
RULE_COLOR = "#fef3c7"


def render_html(text: str, entities: list[Entity]) -> str:
    """Highlight non-overlapping ``entities`` in ``text``. All user text is escaped."""

    parts, cursor = [], 0
    for e in sorted(entities, key=lambda e: e.start):
        if e.start < cursor:
            continue
        color = COLORS.get(e.label, RULE_COLOR)
        parts.append(html.escape(text[cursor : e.start]))
        parts.append(
            f'<mark style="background:{color};color:#0b0b0b;padding:1px 4px;border-radius:4px" '
            f'title="{html.escape(e.label)} · {e.source}">{html.escape(text[e.start : e.end])}'
            f'<sup style="font-size:0.65em;font-weight:600;margin-left:3px">{html.escape(e.label)}</sup></mark>'
        )
        cursor = e.end
    parts.append(html.escape(text[cursor:]))
    body = "".join(parts).replace("\n", "<br>")
    return f'<div style="line-height:2;font-size:1.05rem">{body}</div>'
