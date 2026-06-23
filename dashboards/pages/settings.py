"""Settings page — 3-column config editor for mea_config.json.

Left:   section tree (io / sorting / merging / filtering / plotting / curation)
Middle: JSON editor for selected section
Right:  read-only full-config preview
"""

from __future__ import annotations

import json
from pathlib import Path

import dash
from dash import Input, Output, State, callback, dcc, html
from flask import current_app

dash.register_page(__name__, path="/settings", name="Settings", order=10)

_SECTIONS = ["io", "sorting", "merging", "filtering", "plotting", "curation"]

_SECTION_DESCS = {
    "io": "Output directories, checkpoint path, PHY export, cleanup.",
    "sorting": "Spike sorter selection and Docker image.",
    "merging": "UnitMatch merge parameters.",
    "filtering": "Reference file and assay-type filters for the driver.",
    "plotting": "Raster plot mode, sorting, debug overlays.",
    "curation": "Quality metric thresholds for unit curation.",
}

_SECTION_GLYPHS = {
    "io":       "⇄",
    "sorting":  "∿",
    "merging":  "⋈",
    "filtering": "▿",
    "plotting": "▤",
    "curation": "✓",
}


def _tree_node(section: str) -> html.Div:
    return html.Div(
        [
            html.Span(_SECTION_GLYPHS.get(section, "·"), className="leaf"),
            html.Span(section, className="lbl"),
        ],
        id={"settings-node": section},
        className="tree-node",
        n_clicks=0,
    )


layout = html.Div(
    [
        dcc.Store(id="settings-section", data="io"),
        dcc.Store(id="settings-save-ts", data=0),
        html.Div(
            [
                html.Div(
                    [
                        html.Div("settings", className="breadcrumb"),
                        html.H1("Settings"),
                        html.Div("mea_config.json editor", className="subtitle"),
                    ]
                ),
                html.Div(
                    [
                        html.Button(
                            "Save section",
                            id="settings-save-btn",
                            n_clicks=0,
                            className="btn primary",
                        ),
                        html.Button(
                            "Save all",
                            id="settings-save-all-btn",
                            n_clicks=0,
                            className="btn",
                        ),
                        html.Span(id="settings-status",
                                  style={"fontFamily": "var(--font-mono)", "fontSize": "11px",
                                         "color": "var(--ink-3)", "marginLeft": "8px"}),
                    ],
                    className="view-actions",
                ),
            ],
            className="view-head",
        ),
        html.Div(id="settings-banner"),
        html.Div(
            [
                # Left: section tree
                html.Div(
                    [
                        html.Div("sections", className="section-label"),
                        html.Div(
                            [_tree_node(s) for s in _SECTIONS],
                            className="tree",
                            id="settings-tree",
                        ),
                    ],
                    style={"width": "200px", "flexShrink": 0},
                ),
                # Middle: section JSON editor
                html.Div(
                    [
                        html.Div(id="settings-section-head", className="section-label"),
                        html.Div(id="settings-section-desc",
                                 style={"color": "var(--ink-3)", "fontFamily": "var(--font-mono)",
                                        "fontSize": "11px", "marginBottom": "10px"}),
                        dcc.Textarea(
                            id="settings-editor",
                            style={
                                "width": "100%",
                                "minHeight": "420px",
                                "fontFamily": "var(--font-mono)",
                                "fontSize": "12px",
                                "background": "var(--bg)",
                                "color": "var(--ink)",
                                "border": "1px solid var(--line)",
                                "borderRadius": "4px",
                                "padding": "12px 14px",
                                "resize": "vertical",
                            },
                        ),
                        html.Div(id="settings-parse-error",
                                 style={"color": "oklch(0.43 0.16 28)",
                                        "fontFamily": "var(--font-mono)",
                                        "fontSize": "11px",
                                        "marginTop": "6px",
                                        "minHeight": "16px"}),
                    ],
                    className="grow",
                    style={"display": "flex", "flexDirection": "column"},
                ),
                # Right: full JSON preview
                html.Div(
                    [
                        html.Div("full config", className="section-label"),
                        html.Pre(id="settings-full-preview", className="code",
                                 style={"maxHeight": "520px", "overflow": "auto"}),
                    ],
                    style={"width": "340px", "flexShrink": 0},
                ),
            ],
            className="row",
            style={"alignItems": "flex-start", "gap": "24px"},
        ),
    ],
    className="page",
)


@callback(
    Output("settings-section", "data"),
    Output({"settings-node": dash.ALL}, "className"),
    Input({"settings-node": dash.ALL}, "n_clicks"),
    State("settings-section", "data"),
    prevent_initial_call=True,
)
def _select_section(clicks, current):
    from dash import ctx as dash_ctx
    if not dash_ctx.triggered_id or not isinstance(dash_ctx.triggered_id, dict):
        return current, ["tree-node"] * len(_SECTIONS)
    selected = dash_ctx.triggered_id.get("settings-node", current)
    classes = [
        "tree-node active" if s == selected else "tree-node"
        for s in _SECTIONS
    ]
    return selected, classes


@callback(
    Output("settings-editor", "value"),
    Output("settings-section-head", "children"),
    Output("settings-section-desc", "children"),
    Output("settings-banner", "children"),
    Output("settings-full-preview", "children"),
    Input("settings-section", "data"),
    Input("settings-save-ts", "data"),
    Input("dashboard-url", "pathname"),
)
def _load_section(section, _ts, _path):
    try:
        ctx = current_app.config.get("MEA", {})
    except RuntimeError:
        return "{}", section, "", _no_config_banner(), "{}"

    config_path: Path | None = ctx.get("config_path")
    config_exists = ctx.get("config_exists", False)

    banner = None
    if not config_exists:
        banner = html.Div(
            [
                html.Strong("Config file not found. "),
                html.Span(f"Will create: {config_path}"),
            ],
            className="banner warn",
        )

    from config_loader import DEFAULTS, load_config  # lazy
    try:
        config = load_config(config_path) if config_exists else {}
    except Exception as e:
        config = {}
        banner = html.Div(f"Error loading config: {e}", className="banner warn")

    section_data = config.get(section, DEFAULTS.get(section, {}))
    editor_text = json.dumps(section_data, indent=2)
    desc = _SECTION_DESCS.get(section, "")

    full_merged = {}
    for s in _SECTIONS:
        full_merged[s] = config.get(s, DEFAULTS.get(s, {}))
    full_preview = json.dumps(full_merged, indent=2)

    return editor_text, section, desc, banner, full_preview


@callback(
    Output("settings-status", "children"),
    Output("settings-parse-error", "children"),
    Output("settings-save-ts", "data"),
    Input("settings-save-btn", "n_clicks"),
    State("settings-editor", "value"),
    State("settings-section", "data"),
    State("settings-save-ts", "data"),
    prevent_initial_call=True,
)
def _save_section(n_clicks, editor_text, section, ts):
    if not n_clicks:
        return "", "", ts
    try:
        section_data = json.loads(editor_text or "{}")
    except json.JSONDecodeError as e:
        return "", f"JSON parse error: {e}", ts

    try:
        ctx = current_app.config.get("MEA", {})
        config_path: Path = ctx.get("config_path")
    except RuntimeError:
        return "Error: no app context.", "", ts

    from config_loader import DEFAULTS, load_config
    try:
        config = load_config(config_path) if config_path.exists() else {}
    except Exception:
        config = {}

    config[section] = section_data
    config_path.parent.mkdir(parents=True, exist_ok=True)
    config_path.write_text(json.dumps(config, indent=2))

    # Update server-side stash
    ctx["config"] = config
    ctx["config_exists"] = True

    return f"Saved [{section}]", "", ts + 1


@callback(
    Output("settings-status", "children", allow_duplicate=True),
    Output("settings-parse-error", "children", allow_duplicate=True),
    Output("settings-save-ts", "data", allow_duplicate=True),
    Input("settings-save-all-btn", "n_clicks"),
    State("settings-editor", "value"),
    State("settings-section", "data"),
    State("settings-save-ts", "data"),
    prevent_initial_call=True,
)
def _save_all(n_clicks, editor_text, section, ts):
    if not n_clicks:
        return "", "", ts
    try:
        section_data = json.loads(editor_text or "{}")
    except json.JSONDecodeError as e:
        return "", f"JSON parse error: {e}", ts

    try:
        ctx = current_app.config.get("MEA", {})
        config_path: Path = ctx.get("config_path")
    except RuntimeError:
        return "Error: no app context.", "", ts

    from config_loader import DEFAULTS, load_config
    try:
        config = load_config(config_path) if config_path.exists() else {}
    except Exception:
        config = {}

    config[section] = section_data

    # Fill missing sections with defaults
    for s in _SECTIONS:
        if s not in config:
            config[s] = DEFAULTS.get(s, {})

    config_path.parent.mkdir(parents=True, exist_ok=True)
    config_path.write_text(json.dumps(config, indent=2))
    ctx["config"] = config
    ctx["config_exists"] = True

    return "Saved all sections", "", ts + 1


def _no_config_banner():
    return html.Div(
        [
            html.Strong("No config loaded. "),
            html.Span("Fill in the fields below and click Save."),
        ],
        className="banner info",
    )
