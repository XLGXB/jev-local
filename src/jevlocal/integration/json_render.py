"""Strict catalog v1 and inert, deterministic HTML preview for local Choice UI."""

from __future__ import annotations

import hashlib
from html import escape
import json
from pathlib import Path
import re
import unicodedata


CATALOG_VERSION = 1
_ID = re.compile(r"[A-Za-z][A-Za-z0-9_-]{0,63}\Z", re.ASCII)
_COMPONENT_FIELDS = {
    "heading": ("text", "level"),
    "text": ("text", "tone"),
    "metric": ("label", "value", "unit"),
    "list": ("title", "items", "ordered"),
    "notice": ("text", "tone"),
    "button": ("label", "action_id"),
}
_CSP = (
    "default-src 'none'; script-src 'none'; style-src 'unsafe-inline'; "
    "img-src 'none'; font-src 'none'; connect-src 'none'; media-src 'none'; "
    "object-src 'none'; frame-src 'none'; child-src 'none'; "
    "form-action 'none'; base-uri 'none'"
)
_CSS = """
:root{color-scheme:light;font-family:ui-sans-serif,system-ui,-apple-system,Segoe UI,sans-serif;
  background:#f3f5f8;color:#172337}
*{box-sizing:border-box}body{margin:0}main{max-width:1060px;margin:0 auto;padding:clamp(20px,4vw,48px)}
.eyebrow{font-size:.75rem;letter-spacing:.1em;text-transform:uppercase;color:#405773;font-weight:750}
header{margin-bottom:26px}h1{font-size:clamp(1.7rem,4vw,2.6rem);line-height:1.15;margin:.45rem 0}
.disclaimer{max-width:68ch;color:#526175;font-size:.91rem;line-height:1.55}
.layout{display:grid;gap:var(--gap)}.layout.stack{grid-template-columns:1fr}
.layout.grid.cols-1{grid-template-columns:repeat(1,minmax(0,1fr))}
.layout.grid.cols-2{grid-template-columns:repeat(2,minmax(0,1fr))}
.layout.grid.cols-3{grid-template-columns:repeat(3,minmax(0,1fr))}
.gap-compact{--gap:10px}.gap-normal{--gap:18px}.gap-wide{--gap:28px}
.card{min-width:0;border:1px solid #dbe2eb;border-radius:16px;background:#fff;padding:22px;
  box-shadow:0 3px 12px #2334480b;overflow-wrap:anywhere}
.card h2,.card h3,.card h4{margin:0;font-size:1.2rem;line-height:1.35}
.card p{margin:0;line-height:1.6}.muted{color:#637187}
.metric-label{display:block;color:#53637a;font-size:.83rem;font-weight:650}
.metric-value{font-size:2rem;line-height:1.2;font-weight:750;letter-spacing:-.03em}
.metric-unit{margin-left:.4em;color:#526175;font-size:.88rem}.list-title{margin:0 0 12px;font-weight:700}
ol,ul{margin:0;padding-left:1.4rem;line-height:1.65}li+li{margin-top:5px}
.notice{border-left:4px solid #407cb4}.notice.warning{border-left-color:#b27320;background:#fffaf1}
.notice.success{border-left-color:#26815c;background:#f5fcf8}
button{font:inherit;font-weight:700;border:1px solid #a8b9cd;border-radius:9px;padding:10px 16px;
  background:#e8edf3;color:#455366;cursor:not-allowed;opacity:1}
.action-id{display:block;margin-top:8px;color:#6c7888;font-size:.76rem}
@media(max-width:720px){.layout.grid{grid-template-columns:1fr!important}.card{padding:18px}}
"""


def _object(value: object, fields: set[str], location: str) -> dict:
    if type(value) is not dict or set(value) != fields:
        raise ValueError(f"{location}: exact fields required")
    return value


def _string(value: object, limit: int, location: str) -> str:
    if type(value) is not str or not value or len(value) > limit:
        raise ValueError(f"{location}: invalid string")
    if any(unicodedata.category(char) in {"Cc", "Cs"} for char in value):
        raise ValueError(f"{location}: control or surrogate character")
    return value


def _id(value: object, location: str) -> str:
    if type(value) is not str or _ID.fullmatch(value) is None:
        raise ValueError(f"{location}: invalid ASCII id")
    return value


def _integer(value: object, lower: int, upper: int, location: str) -> int:
    if type(value) is not int or not lower <= value <= upper:
        raise ValueError(f"{location}: invalid integer")
    return value


def _choice(value: object, allowed: tuple[str, ...], location: str) -> str:
    if type(value) is not str or value not in allowed:
        raise ValueError(f"{location}: invalid value")
    return value


def _props(kind: str, value: object) -> dict:
    raw = _object(value, set(_COMPONENT_FIELDS[kind]), f"{kind}.props")
    if kind == "heading":
        return {"text": _string(raw["text"], 120, "heading.text"),
                "level": _integer(raw["level"], 1, 3, "heading.level")}
    if kind == "text":
        return {"text": _string(raw["text"], 2000, "text.text"),
                "tone": _choice(raw["tone"], ("default", "muted"), "text.tone")}
    if kind == "metric":
        return {name: _string(raw[name], 120, f"metric.{name}")
                for name in ("label", "value", "unit")}
    if kind == "list":
        items = raw["items"]
        if type(items) is not list or not 1 <= len(items) <= 12:
            raise ValueError("list.items: expected 1..12 items")
        if type(raw["ordered"]) is not bool:
            raise ValueError("list.ordered: expected bool")
        return {"title": _string(raw["title"], 120, "list.title"),
                "items": [_string(item, 2000, "list.item") for item in items],
                "ordered": raw["ordered"]}
    if kind == "notice":
        return {"text": _string(raw["text"], 2000, "notice.text"),
                "tone": _choice(raw["tone"], ("info", "warning", "success"), "notice.tone")}
    return {"label": _string(raw["label"], 120, "button.label"),
            "action_id": _id(raw["action_id"], "button.action_id")}


def validate_ui(value: object) -> dict:
    """Return a strict, deep-copied JSON UI; reject any ambiguity with ValueError."""
    raw = _object(value, {"schema_version", "title", "layout", "components"}, "ui")
    if type(raw["schema_version"]) is not int or raw["schema_version"] != CATALOG_VERSION:
        raise ValueError("ui.schema_version: unsupported")
    layout = _object(raw["layout"], {"kind", "columns", "gap"}, "ui.layout")
    kind = _choice(layout["kind"], ("stack", "grid"), "layout.kind")
    columns = _integer(layout["columns"], 1, 3, "layout.columns")
    if kind == "stack" and columns != 1:
        raise ValueError("layout.stack requires one column")
    components = raw["components"]
    if type(components) is not list or not 1 <= len(components) <= 24:
        raise ValueError("ui.components: expected 1..24 components")
    normalized = {"schema_version": CATALOG_VERSION,
                  "title": _string(raw["title"], 120, "ui.title"),
                  "layout": {"kind": kind, "columns": columns,
                             "gap": _choice(layout["gap"], ("compact", "normal", "wide"), "layout.gap")},
                  "components": []}
    seen = set()
    for index, entry in enumerate(components):
        item = _object(entry, {"id", "type", "props"}, f"component[{index}]")
        component_id = _id(item["id"], f"component[{index}].id")
        if component_id in seen:
            raise ValueError("ui.components: duplicate id")
        seen.add(component_id)
        component_type = _choice(item["type"], tuple(_COMPONENT_FIELDS), f"component[{index}].type")
        normalized["components"].append({"id": component_id, "type": component_type,
                                         "props": _props(component_type, item["props"])})
    try:
        encoded = json.dumps(normalized, ensure_ascii=False, separators=(",", ":"),
                             allow_nan=False).encode("utf-8")
    except (TypeError, ValueError, UnicodeError) as exc:
        raise ValueError("ui: not valid UTF-8 JSON") from exc
    if len(encoded) > 32768:
        raise ValueError("ui: exceeds 32 KiB UTF-8 JSON")
    return normalized


def catalog_fingerprint() -> str:
    """SHA-256 of the actual catalog and renderer source, including trusted CSS."""
    return hashlib.sha256(Path(__file__).read_bytes()).hexdigest()


def render_html(ui: object) -> str:
    """Render validated UI as a self-contained, scriptless, inert HTML document."""
    value = validate_ui(ui)
    layout = value["layout"]
    layout_class = f"{layout['kind']} cols-{layout['columns']} gap-{layout['gap']}"
    parts = [
        "<!doctype html><html lang=\"zh-CN\"><head><meta charset=\"utf-8\">",
        '<meta name="viewport" content="width=device-width,initial-scale=1">',
        f'<meta http-equiv="Content-Security-Policy" content="{_CSP}">',
        f"<title>{escape(value['title'], quote=True)}</title>",
        f"<style>{_CSS}</style></head><body><main>",
        '<header><div class="eyebrow">Local Choice UI · Preview</div>',
        f"<h1>{escape(value['title'], quote=True)}</h1>",
        '<p class="disclaimer">本地模型建议 · 非授权。内容仅供人工审阅；按钮不可操作，不会执行任何动作。</p></header>',
        f'<div class="layout {layout_class}">',
    ]
    for component in value["components"]:
        kind = component["type"]
        props = component["props"]
        tone_class = f" {props['tone']}" if kind == "notice" else ""
        parts.append(f'<section class="card {kind}{tone_class}" id="{component["id"]}">')
        if kind == "heading":
            tag = f"h{props['level'] + 1}"
            parts.append(f"<{tag}>{escape(props['text'], quote=True)}</{tag}>")
        elif kind == "text":
            parts.append(f'<p class="{props["tone"]}">{escape(props["text"], quote=True)}</p>')
        elif kind == "metric":
            parts.append(f'<span class="metric-label">{escape(props["label"], quote=True)}</span>')
            parts.append(f'<span class="metric-value">{escape(props["value"], quote=True)}</span>')
            parts.append(f'<span class="metric-unit">{escape(props["unit"], quote=True)}</span>')
        elif kind == "list":
            tag = "ol" if props["ordered"] else "ul"
            parts.append(f'<p class="list-title">{escape(props["title"], quote=True)}</p><{tag}>')
            parts.extend(f"<li>{escape(item, quote=True)}</li>" for item in props["items"])
            parts.append(f"</{tag}>")
        elif kind == "notice":
            parts.append(f'<p>{escape(props["text"], quote=True)}</p>')
        else:
            parts.append(f'<button type="button" disabled aria-disabled="true">{escape(props["label"], quote=True)}</button>')
            parts.append(f'<span class="action-id">Action ID: {escape(props["action_id"], quote=True)}</span>')
        parts.append("</section>")
    parts.append("</div></main></body></html>")
    return "".join(parts)
