"""Run page — CLI command builder for run_pipeline_driver.py.

Builds the invocation string; does NOT spawn the process.
"""

from __future__ import annotations

import dash
from dash import Input, Output, State, callback, dcc, html
from flask import current_app

from dashboards.components import no_config_banner
from dashboards.data import TERMINAL_STAGE, load_checkpoints

dash.register_page(__name__, path="/run", name="Run", order=3)


def _field(label: str, child) -> html.Div:
    return html.Div(
        [html.Label(label, className="section-label"), child],
        style={"flex": "1 1 220px"},
    )


layout = html.Div(
    [
        html.Div(
            [
                html.Div(
                    [
                        html.Div("run", className="breadcrumb"),
                        html.H1("Run Pipeline"),
                        html.Div(
                            "Build a run_pipeline_driver.py invocation",
                            className="subtitle",
                        ),
                    ]
                ),
            ],
            className="view-head",
        ),
        html.Div(id="run-no-config"),
        # ── Inputs ──────────────────────────────────────────────────────────
        html.Div(
            [
                html.Div(
                    [html.Span("options", className="h-title")],
                    className="card-head",
                ),
                html.Div(
                    [
                        html.Div(
                            [
                                _field("Data directory",
                                       dcc.Input(id="run-data-dir", type="text",
                                                 placeholder="/path/to/experiment",
                                                 style={"width": "100%"})),
                                _field("Config file",
                                       dcc.Input(id="run-config-path", type="text",
                                                 placeholder="mea_config.json",
                                                 style={"width": "100%"})),
                                _field("Sorter",
                                       dcc.Dropdown(
                                           id="run-sorter",
                                           options=[
                                               {"label": "kilosort4", "value": "kilosort4"},
                                               {"label": "kilosort3", "value": "kilosort3"},
                                               {"label": "spykingcircus2", "value": "spykingcircus2"},
                                           ],
                                           value="kilosort4",
                                           clearable=False,
                                       )),
                                _field("Resume from stage",
                                       dcc.Dropdown(
                                           id="run-resume-from",
                                           options=[
                                               {"label": "— (no resume)", "value": ""},
                                               {"label": "preprocessing", "value": "preprocessing"},
                                               {"label": "sorting", "value": "sorting"},
                                               {"label": "merge", "value": "merge"},
                                               {"label": "analyzer", "value": "analyzer"},
                                               {"label": "reports", "value": "reports"},
                                           ],
                                           value="",
                                           clearable=False,
                                       )),
                            ],
                            style={"display": "flex", "flexWrap": "wrap", "gap": "16px"},
                        ),
                        html.Div(
                            [
                                dcc.Checklist(
                                    id="run-flags",
                                    options=[
                                        {"label": "  --skip-spikesorting", "value": "skip_spikesorting"},
                                        {"label": "  --reanalyze-bursts",  "value": "reanalyze_bursts"},
                                        {"label": "  --force-restart",     "value": "force_restart"},
                                        {"label": "  --export-to-phy",     "value": "export_to_phy"},
                                        {"label": "  --clean-up",          "value": "clean_up"},
                                        {"label": "  --dry",               "value": "dry"},
                                    ],
                                    value=[],
                                    style={
                                        "fontFamily": "var(--font-mono)",
                                        "fontSize": "12px",
                                        "display": "grid",
                                        "gridTemplateColumns": "repeat(3, 1fr)",
                                        "gap": "6px",
                                        "marginTop": "12px",
                                    },
                                ),
                            ]
                        ),
                    ],
                    className="card-body",
                ),
            ],
            className="card",
        ),
        # ── Generated command ────────────────────────────────────────────────
        html.Div(
            [
                html.Div(
                    [
                        html.Span("command", className="h-title"),
                        html.Button(
                            "Copy",
                            id="run-copy-btn",
                            n_clicks=0,
                            className="btn ghost",
                        ),
                    ],
                    className="card-head h-actions",
                ),
                html.Div(
                    html.Pre(id="run-command-pre", className="code terminal"),
                    className="card-body",
                ),
            ],
            className="card",
        ),
        # ── Preview ──────────────────────────────────────────────────────────
        dcc.Loading(
            html.Div(
                [
                    html.Div(
                        [html.Span("pending wells", className="h-title"),
                         html.Div(id="run-preview-count", className="h-actions")],
                        className="card-head",
                    ),
                    html.Div(id="run-preview-body", className="card-body flush"),
                ],
                className="card",
            ),
            type="circle",
            color="var(--accent)",
            style={"minHeight": "80px"},
        ),
        # Copy-to-clipboard client callback
        dcc.Clipboard(id="run-clipboard", style={"display": "none"}),
    ],
    className="page",
)


@callback(
    Output("run-no-config", "children"),
    Output("run-command-pre", "children"),
    Output("run-preview-body", "children"),
    Output("run-preview-count", "children"),
    Output("run-config-path", "value"),
    Output("run-data-dir", "value"),
    Input("run-data-dir", "value"),
    Input("run-config-path", "value"),
    Input("run-sorter", "value"),
    Input("run-resume-from", "value"),
    Input("run-flags", "value"),
    Input("dashboard-url", "pathname"),
    State("mea-rerun-dir", "data"),
)
def _build_command(data_dir, config_path, sorter, resume_from, flags, _path, prefill_dir):
    try:
        ctx = current_app.config.get("MEA", {})
    except RuntimeError:
        return no_config_banner(), "", [], "", "mea_config.json", dash.no_update

    # Pre-fill data dir from store when arriving via Rerun button
    if not data_dir and prefill_dir:
        data_dir = prefill_dir

    # Pre-fill config path from loaded config
    if not config_path:
        cp = ctx.get("config_path")
        config_path = str(cp) if cp else "mea_config.json"

    no_config = None if ctx.get("config_exists") else no_config_banner()
    flags = flags or []

    if not data_dir:
        return no_config, "# Enter a data directory above.", [], "", config_path, dash.no_update

    parts = ["python run_pipeline_driver.py", f'"{data_dir}"']
    parts.append(f"--config '{config_path}'")
    if sorter and sorter != "kilosort4":
        parts.append(f"--sorter {sorter}")
    if resume_from:
        parts.append(f"--resume-from {resume_from}")
    for f in flags:
        parts.append(f"--{f.replace('_', '-')}")

    command = " \\\n  ".join(parts)

    # Preview: load checkpoints and show pending wells
    checkpoint_dir = ctx.get("checkpoint_dir")
    preview_body = []
    count_pill = ""
    if checkpoint_dir:
        df = load_checkpoints(checkpoint_dir)
        if not df.empty:
            pending = df[df["stage"] != TERMINAL_STAGE]
            if "reanalyze_bursts" not in flags:
                pending = pending[~pending["failed"]]
            count_pill = html.Span(str(len(pending)), className="count")
            if not pending.empty:
                header = html.Tr([html.Th(c) for c in ["project", "chip", "run", "well", "stage"]])
                rows = [
                    html.Tr([
                        html.Td(str(r.project or "—"), className="mono"),
                        html.Td(str(r.chip or "—"), className="mono"),
                        html.Td(str(r.run or "—"), className="mono"),
                        html.Td(str(r.well or "—"), className="mono"),
                        html.Td(str(r.stage or "—"), className="mono"),
                    ])
                    for r in pending.head(30).itertuples()
                ]
                table = html.Table([html.Thead(header), html.Tbody(rows)], className="tbl")
                preview_body = html.Div(table, className="tbl-wrap")
            else:
                preview_body = html.Div(
                    "All wells complete.",
                    style={"padding": "14px", "color": "var(--ink-3)",
                           "fontFamily": "var(--font-mono)", "fontSize": "12px"},
                )

    return no_config, command, preview_body, count_pill, config_path, data_dir


# Wire copy button → clipboard
clientside_callback = """
function(n_clicks, text) {
    if (n_clicks > 0 && text) {
        navigator.clipboard.writeText(text);
    }
    return "";
}
"""
