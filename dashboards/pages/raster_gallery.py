"""Raster Gallery — raster plots collected across all completed wells.

Images are served via the /raster-img Flask route (not base64-embedded),
so the browser fetches and caches each file independently. PNG is preferred
over SVG for fast rendering; SVG is used as a fallback for older runs.
"""

from __future__ import annotations

from pathlib import Path
from urllib.parse import quote

import dash
from dash import Input, Output, callback, dcc, html
from flask import current_app

from dashboards.components import no_config_banner
from dashboards.data import load_checkpoints, load_network_results_from_checkpoints

dash.register_page(__name__, path="/rasters", name="Raster Gallery", order=7)

_WINDOW_OPTIONS = [
    {"label": "60 s", "value": "60s"},
    {"label": "30 s", "value": "30s"},
    {"label": "full", "value": "full"},
]

# Stem (no extension) for each window choice
_STEM_MAP = {
    "60s":  "raster_burst_plot_60s",
    "30s":  "raster_burst_plot_30s",
    "full": "raster_burst_plot",
}

layout = html.Div(
    [
        html.Div(
            [
                html.Div(
                    [
                        html.Div("analysis", className="breadcrumb"),
                        html.H1("Raster Gallery"),
                        html.Div(
                            "Raster plots collected across all completed wells",
                            className="subtitle",
                        ),
                    ]
                ),
                html.Div(
                    [
                        dcc.Dropdown(
                            id="rg-project-filter",
                            placeholder="All projects",
                            clearable=True,
                            style={
                                "width": "200px",
                                "fontFamily": "var(--font-mono)",
                                "fontSize": "12px",
                            },
                        ),
                        dcc.RadioItems(
                            id="rg-window",
                            options=_WINDOW_OPTIONS,
                            value="60s",
                            inline=True,
                            style={
                                "fontFamily": "var(--font-mono)",
                                "fontSize": "12px",
                                "display": "flex",
                                "gap": "12px",
                                "alignItems": "center",
                            },
                        ),
                        html.Button(
                            "↺ Refresh",
                            id="rg-refresh-btn",
                            n_clicks=0,
                            className="btn",
                        ),
                    ],
                    className="view-actions",
                ),
            ],
            className="view-head",
        ),
        html.Div(id="rg-no-config"),
        html.Div(id="rg-count-bar"),
        html.Div(id="rg-gallery"),
    ],
    className="page",
)


@callback(
    Output("rg-no-config", "children"),
    Output("rg-project-filter", "options"),
    Output("rg-count-bar", "children"),
    Output("rg-gallery", "children"),
    Input("rg-refresh-btn", "n_clicks"),
    Input("rg-project-filter", "value"),
    Input("rg-window", "value"),
    Input("dashboard-url", "pathname"),
)
def _render_gallery(_refresh, project_filter, window, _path):
    try:
        ctx = current_app.config.get("MEA", {})
    except RuntimeError:
        return no_config_banner(), [], None, []

    if not ctx.get("config_exists"):
        return no_config_banner(), [], None, []

    checkpoint_dir = ctx.get("checkpoint_dir")
    if not checkpoint_dir:
        return html.Div("checkpoint_dir not set.", className="banner warn"), [], None, []

    df = load_checkpoints(checkpoint_dir)
    rows = load_network_results_from_checkpoints(df)

    if not rows:
        return None, [], None, _empty("No completed wells found.")

    projects = sorted({r["project"] or "unknown" for r in rows})
    proj_opts = [{"label": p, "value": p} for p in projects]

    if project_filter:
        rows = [r for r in rows if (r["project"] or "unknown") == project_filter]

    stem = _STEM_MAP.get(window or "60s", "raster_burst_plot_60s")

    by_project: dict[str, list] = {}
    for r in rows:
        by_project.setdefault(r["project"] or "unknown", []).append(r)

    n_found = n_missing = 0
    sections = []

    for proj in sorted(by_project):
        cards = []
        for r in by_project[proj]:
            img_path = _find_raster(Path(r["path"]), stem)
            if img_path is None:
                n_missing += 1
                continue
            n_found += 1

            label = " / ".join(str(r.get(k) or "?") for k in ("chip", "run", "well"))
            meta_parts = []
            if r.get("n_units") is not None:
                meta_parts.append(f"{r['n_units']} units")
            if r.get("network_bursts_count"):
                meta_parts.append(f"{r['network_bursts_count']} NB")
            if r.get("burst_rate_hz"):
                meta_parts.append(f"{r['burst_rate_hz']:.3g} Hz")

            cards.append(
                html.Div(
                    [
                        html.Div(
                            [
                                html.Span(
                                    label,
                                    style={
                                        "fontWeight": 600,
                                        "fontSize": "11px",
                                        "fontFamily": "var(--font-mono)",
                                    },
                                ),
                                html.Span(
                                    " · ".join(meta_parts),
                                    style={
                                        "color": "var(--ink-3)",
                                        "fontSize": "10px",
                                        "fontFamily": "var(--font-mono)",
                                    },
                                ),
                            ],
                            style={
                                "display": "flex",
                                "justifyContent": "space-between",
                                "alignItems": "center",
                                "padding": "6px 10px 4px",
                                "borderBottom": "1px solid var(--line-soft)",
                            },
                        ),
                        html.Img(
                            src=f"/raster-img?p={quote(str(img_path))}",
                            style={
                                "width": "100%",
                                "height": "200px",
                                "objectFit": "contain",
                                "display": "block",
                                "background": "var(--bg-elev)",
                            },
                        ),
                    ],
                    className="card",
                    style={"padding": 0, "overflow": "hidden"},
                )
            )

        if not cards:
            continue

        sections.append(
            html.Div(
                [
                    html.Div(proj, className="rec-group-header"),
                    html.Div(
                        cards,
                        style={
                            "display": "grid",
                            "gridTemplateColumns": "repeat(auto-fill, minmax(360px, 1fr))",
                            "gap": "12px",
                            "paddingBottom": "24px",
                        },
                    ),
                ]
            )
        )

    count_bar = None
    if n_found or n_missing:
        parts: list = [html.Span(f"{n_found} plots")]
        if n_missing:
            parts += [
                html.Span(" · "),
                html.Span(f"{n_missing} missing", style={"color": "var(--ink-3)"}),
            ]
        count_bar = html.Div(
            parts,
            style={
                "fontFamily": "var(--font-mono)",
                "fontSize": "11px",
                "padding": "4px 0 12px",
                "color": "var(--ink-2)",
            },
        )

    if not sections:
        return None, proj_opts, count_bar, _empty(f"No {window} raster plots found.")

    return None, proj_opts, count_bar, html.Div(sections)


def _find_raster(output_dir: Path, stem: str) -> Path | None:
    """Return first existing file: PNG preferred over SVG, plain over fixed_y."""
    for prefix in ("", "fixed_y_"):
        for ext in (".png", ".svg"):
            f = output_dir / f"{prefix}{stem}{ext}"
            if f.exists():
                return f
    return None


def _empty(msg: str) -> html.Div:
    return html.Div(
        msg,
        style={
            "padding": "24px",
            "color": "var(--ink-3)",
            "fontFamily": "var(--font-mono)",
            "fontSize": "12px",
        },
    )
