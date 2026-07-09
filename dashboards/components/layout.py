"""App shell — topbar, left rail, main viewport.

CSS lives in dashboards/assets/styles.css (auto-loaded by Dash).
"""

from __future__ import annotations

import subprocess
from pathlib import Path

import dash
from dash import Input, Output, callback, clientside_callback, dcc, html
from flask import current_app


_NAV_META: dict[str, tuple[str, str]] = {
    "Home":              ("operations", "◐"),
    "Pipeline":          ("operations", "≡"),
    "Recordings":        ("operations", "▦"),
    "Run":               ("operations", "▸"),
    "Burst Diagnostic":  ("analysis",   "∿"),
    "Burst Inspector":   ("analysis",   "⌇"),
    "Raster Gallery":    ("analysis",   "⊞"),
    "Settings":          ("system",     "{}"),
}

_SECTION_ORDER = ("operations", "analysis", "system")


def _git_rev() -> str:
    try:
        out = subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"],
            stderr=subprocess.DEVNULL,
            cwd=Path(__file__).resolve().parent,
            timeout=1,
        )
        return out.decode().strip()
    except (subprocess.SubprocessError, FileNotFoundError, OSError):
        return ""


_GIT_REV = _git_rev()


def no_config_banner() -> html.Div:
    return html.Div(
        [
            html.Strong("No config loaded. "),
            html.Span("Go to "),
            dcc.Link("Settings", href="/settings"),
            html.Span(" to set paths and save mea_config.json."),
        ],
        className="banner info",
    )


def build_layout() -> html.Div:
    return html.Div(
        [
            dcc.Location(id="dashboard-url"),
            # Shared store: recordings page writes, run page reads to pre-fill data dir.
            dcc.Store(id="mea-rerun-dir", storage_type="session"),
            _topbar(),
            _rail(),
            html.Main(
                html.Div(dash.page_container, className="viewport"),
                className="main",
            ),
        ],
        className="app",
    )


def _topbar() -> html.Div:
    chip_label = f"dev · {_GIT_REV}" if _GIT_REV else "dev"
    return html.Div(
        [
            html.Div(
                [
                    html.Span(className="brand-mark"),
                    html.Span("MEA Analysis"),
                    html.Span("/", className="brand-sep"),
                    html.Span("pipeline ops", style={"color": "var(--ink-3)"}),
                ],
                className="brand",
            ),
            html.Span(chip_label, className="branch-chip"),
            html.Div(className="topbar-spacer"),
            html.Div(id="topbar-status", className="topbar-meta"),
        ],
        className="topbar",
    )


def _rail() -> html.Aside:
    pages = sorted(
        dash.page_registry.values(),
        key=lambda p: (p.get("order", 100), p["name"]),
    )
    by_section: dict[str, list] = {s: [] for s in _SECTION_ORDER}
    for page in pages:
        section, _glyph = _NAV_META.get(page["name"], ("operations", "·"))
        by_section.setdefault(section, []).append(page)

    children: list = []
    for section in _SECTION_ORDER:
        items = by_section.get(section, [])
        if not items:
            continue
        children.append(html.Div(section, className="rail-section"))
        for page in items:
            children.append(_rail_item(page))

    children.append(html.Div(id="rail-footer-slot"))
    return html.Aside(children, className="rail")


def _rail_item(page: dict) -> dcc.Link:
    name = page["name"]
    _section, glyph = _NAV_META.get(name, ("operations", "·"))
    return dcc.Link(
        [
            html.Span(glyph, className="glyph"),
            html.Span(name),
        ],
        href=page["relative_path"],
        className="rail-item",
        id={"rail-link": page["relative_path"]},
        refresh=False,
    )


@callback(
    Output("rail-footer-slot", "children"),
    Output("topbar-status", "children"),
    Input("dashboard-url", "pathname"),
)
def _render_rail_footer(_pathname: str):
    try:
        ctx = current_app.config.get("MEA", {})
        checkpoint_dir = ctx.get("checkpoint_dir")
        config_path = ctx.get("config_path")
    except RuntimeError:
        checkpoint_dir = None
        config_path = None

    cp_str = str(checkpoint_dir) if checkpoint_dir else "(not set)"
    cfg_str = str(config_path) if config_path else "(not set)"

    footer = html.Div(
        [
            html.Div("checkpoint_dir", style={"color": "var(--ink-3)", "marginBottom": "4px"}),
            html.Code(cp_str),
            html.Div("config", style={"color": "var(--ink-3)", "marginTop": "8px"}),
            html.Code(cfg_str),
        ],
        className="rail-footer",
    )

    status_dot = html.Span(
        [html.Span(className="dot"), "pipeline · ready"],
    )

    return footer, status_dot


clientside_callback(
    """
    function(pathname, ids) {
        if (!ids) { return []; }
        return ids.map(function(id) {
            var href = id["rail-link"];
            var isActive = (pathname === href) ||
                           (href !== "/" && pathname && pathname.indexOf(href) === 0);
            return isActive ? "rail-item active" : "rail-item";
        });
    }
    """,
    Output({"rail-link": dash.ALL}, "className"),
    Input("dashboard-url", "pathname"),
    Input({"rail-link": dash.ALL}, "id"),
)
