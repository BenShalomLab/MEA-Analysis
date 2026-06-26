"""Pipeline page — stage matrix + inspector with reset + bulk reset."""

from __future__ import annotations

import dash
from dash import ALL, Input, Output, State, callback, clientside_callback, dcc, html
from flask import current_app

from dashboards.components import no_config_banner
from dashboards.data import (
    STAGE_COLS, STAGE_MAP,
    bulk_delete_checkpoints, bulk_reset_checkpoints,
    delete_checkpoint, load_checkpoints, reset_checkpoint,
)

dash.register_page(__name__, path="/pipeline", name="Pipeline", order=1)

_STAGE_LABELS = ["Preproc", "Sorting", "Merge", "Analyzer", "Reports"]

# Reset-to options: name → stage_num to write
_RESET_TO_OPTIONS = [
    {"label": "— pick stage —",          "value": ""},
    {"label": "Before preprocessing (0)", "value": "0"},
    {"label": "Before sorting (2)",       "value": "2"},
    {"label": "Before merge (4)",         "value": "4"},
    {"label": "Before analyzer (6)",      "value": "6"},
    {"label": "Before reports (8)",       "value": "8"},
]

layout = html.Div(
    [
        dcc.Interval(id="pipe-interval", interval=60_000, n_intervals=0),
        dcc.Store(id="pipe-selected-idx", data=None),

        # ── view-head ────────────────────────────────────────────────────
        html.Div(
            [
                html.Div(
                    [
                        html.Div("pipeline", className="breadcrumb"),
                        html.H1("Pipeline Status"),
                        html.Div("Per-well stage matrix", className="subtitle"),
                    ]
                ),
                html.Div(
                    [
                        html.Div(
                            [
                                html.Button(label, id={"pipe-filter": f}, n_clicks=0,
                                            className="active" if f == "all" else "")
                                for f, label in [
                                    ("all",         "All"),
                                    ("complete",    "Complete"),
                                    ("running",     "Running"),
                                    ("failed",      "Failed"),
                                    ("not_started", "Pending"),
                                ]
                            ],
                            className="toggle-group",
                        ),
                        html.Button("↺ Refresh", id="pipe-refresh-btn", n_clicks=0,
                                    className="btn", style={"marginLeft": "8px"}),
                    ],
                    className="view-actions",
                ),
            ],
            className="view-head",
        ),

        html.Div(id="pipe-no-config"),

        # ── matrix + inspector ───────────────────────────────────────────
        dcc.Loading(
            html.Div(
                [
                    html.Div(
                        [
                            html.Div(
                                [html.Span("wells", className="h-title"),
                                 html.Div(id="pipe-count", className="h-actions")],
                                className="card-head",
                            ),
                            html.Div(id="pipe-matrix-body", className="card-body flush"),
                        ],
                        className="card grow",
                    ),
                    html.Div(id="pipe-inspector", className="inspector"),
                ],
                className="row",
                style={"alignItems": "flex-start"},
            ),
            type="circle",
            color="var(--accent)",
            style={"minHeight": "120px"},
        ),

        # ── bulk reset ───────────────────────────────────────────────────
        html.Div(
            [
                html.Div(
                    [
                        html.Span("bulk reset", className="h-title"),
                        html.Span(
                            "Cascades later stages automatically (linear pipeline).",
                            className="h-actions",
                            style={"color": "var(--ink-3)", "fontFamily": "var(--font-mono)",
                                   "fontSize": "11px", "textTransform": "none",
                                   "letterSpacing": "0"},
                        ),
                    ],
                    className="card-head",
                ),
                html.Div(
                    [
                        html.Div(
                            [
                                html.Div(
                                    [
                                        html.Label("Filter — current stage", className="section-label"),
                                        dcc.Dropdown(
                                            id="bulk-filter-stage",
                                            options=[{"label": "all stages", "value": "all"}]
                                                    + [{"label": v, "value": v}
                                                       for v in STAGE_MAP.values()],
                                            value="all", clearable=False,
                                        ),
                                    ],
                                    style={"flex": "1 1 220px"},
                                ),
                                html.Div(
                                    [
                                        html.Label("Reset to", className="section-label"),
                                        dcc.Dropdown(
                                            id="bulk-reset-to",
                                            options=_RESET_TO_OPTIONS,
                                            value="", clearable=False,
                                        ),
                                    ],
                                    style={"flex": "1 1 220px"},
                                ),
                                html.Div(
                                    [
                                        html.Label("Scope", className="section-label"),
                                        dcc.Checklist(
                                            id="bulk-failed-only",
                                            options=[{"label": "  failed wells only", "value": "failed"}],
                                            value=[],
                                            style={"fontFamily": "var(--font-mono)", "fontSize": "12px"},
                                        ),
                                    ],
                                    style={"flex": "0 0 180px"},
                                ),
                            ],
                            style={"display": "flex", "gap": "16px",
                                   "flexWrap": "wrap", "alignItems": "flex-start"},
                        ),
                        html.Div(
                            [
                                html.Button(
                                    [html.Span("…", className="glyph"), "Preview"],
                                    id="bulk-preview-btn", n_clicks=0, className="btn",
                                ),
                                html.Button(
                                    [html.Span("⟲", className="glyph"), "Reset"],
                                    id="bulk-execute-btn", n_clicks=0, className="btn primary",
                                ),
                                html.Span(id="bulk-status",
                                          style={"fontFamily": "var(--font-mono)",
                                                 "fontSize": "11px", "color": "var(--ink-3)"}),
                            ],
                            style={"display": "flex", "alignItems": "center",
                                   "gap": "8px", "marginTop": "14px"},
                        ),
                    ],
                    className="card-body",
                ),
            ],
            className="card",
            style={"marginTop": "16px"},
        ),
        # ── bulk delete ──────────────────────────────────────────────────
        html.Div(
            [
                html.Div(
                    [
                        html.Span("bulk delete", className="h-title"),
                        html.Span(
                            "Permanently removes checkpoint JSON files.",
                            className="h-actions",
                            style={"color": "var(--fail)", "fontFamily": "var(--font-mono)",
                                   "fontSize": "11px", "textTransform": "none",
                                   "letterSpacing": "0"},
                        ),
                    ],
                    className="card-head",
                ),
                html.Div(
                    [
                        html.Div(
                            [
                                html.Div(
                                    [
                                        html.Label("Filter — current stage", className="section-label"),
                                        dcc.Dropdown(
                                            id="bulk-del-filter-stage",
                                            options=[{"label": "all stages", "value": "all"}]
                                                    + [{"label": v, "value": v}
                                                       for v in STAGE_MAP.values()],
                                            value="all", clearable=False,
                                        ),
                                    ],
                                    style={"flex": "1 1 220px"},
                                ),
                                html.Div(
                                    [
                                        html.Label("Scope", className="section-label"),
                                        dcc.Checklist(
                                            id="bulk-del-failed-only",
                                            options=[{"label": "  failed wells only", "value": "failed"}],
                                            value=[],
                                            style={"fontFamily": "var(--font-mono)", "fontSize": "12px"},
                                        ),
                                    ],
                                    style={"flex": "0 0 180px"},
                                ),
                            ],
                            style={"display": "flex", "gap": "16px",
                                   "flexWrap": "wrap", "alignItems": "flex-start"},
                        ),
                        html.Div(
                            dcc.Checklist(
                                id="bulk-del-confirm",
                                options=[{"label": "  I understand this permanently deletes the selected checkpoint files",
                                          "value": "confirmed"}],
                                value=[],
                                style={"fontFamily": "var(--font-mono)", "fontSize": "11px",
                                       "color": "var(--ink-3)"},
                            ),
                            style={"marginTop": "12px"},
                        ),
                        html.Div(
                            [
                                html.Button(
                                    [html.Span("…", className="glyph"), "Preview"],
                                    id="bulk-del-preview-btn", n_clicks=0, className="btn",
                                ),
                                html.Button(
                                    [html.Span("✕", className="glyph"), "Delete files"],
                                    id="bulk-del-execute-btn", n_clicks=0, className="btn",
                                    style={"borderColor": "var(--fail)", "color": "var(--fail)"},
                                ),
                                html.Span(id="bulk-del-status",
                                          style={"fontFamily": "var(--font-mono)",
                                                 "fontSize": "11px", "color": "var(--ink-3)"}),
                            ],
                            style={"display": "flex", "alignItems": "center",
                                   "gap": "8px", "marginTop": "12px"},
                        ),
                    ],
                    className="card-body",
                ),
            ],
            className="card",
            style={"marginTop": "12px", "borderColor": "rgba(224,49,49,0.25)"},
        ),
    ],
    className="page",
)


# ── Matrix ────────────────────────────────────────────────────────────────────

@callback(
    Output("pipe-no-config", "children"),
    Output("pipe-matrix-body", "children"),
    Output("pipe-count", "children"),
    Input("pipe-interval", "n_intervals"),
    Input("pipe-refresh-btn", "n_clicks"),
    Input({"pipe-filter": ALL}, "n_clicks"),
    Input("dashboard-url", "pathname"),
)
def _refresh_matrix(_n, _refresh, _filter_clicks, _path):
    try:
        ctx = current_app.config.get("MEA", {})
    except RuntimeError:
        return no_config_banner(), _empty_matrix(), ""

    if not ctx.get("config_exists"):
        return no_config_banner(), _empty_matrix(), ""

    checkpoint_dir = ctx.get("checkpoint_dir")
    if not checkpoint_dir:
        return html.Div("checkpoint_dir not set.", className="banner warn"), _empty_matrix(), ""

    from dash import ctx as dash_ctx
    active_filter = "all"
    if dash_ctx.triggered_id and isinstance(dash_ctx.triggered_id, dict):
        active_filter = dash_ctx.triggered_id.get("pipe-filter", "all")

    df = load_checkpoints(checkpoint_dir)
    if df.empty:
        return None, _empty_matrix(), ""

    if active_filter == "complete":
        df = df[df["stage"] == "REPORTS_COMPLETE"]
    elif active_filter == "running":
        df = df[df["stage"].isin({"PREPROCESSING", "SORTING", "MERGE", "ANALYZER", "REPORTS"})]
    elif active_filter == "failed":
        df = df[df["failed"] == True]  # noqa: E712
    elif active_filter == "not_started":
        df = df[df["stage"] == "NOT_STARTED"]

    count_pill = html.Span(str(len(df)), className="count")

    header = html.Tr(
        [html.Th("Well", className="well-col")]
        + [html.Th(lbl) for lbl in _STAGE_LABELS]
        + [html.Th("Units")]
    )

    body_rows = []
    for i, r in enumerate(df.itertuples()):
        well_label = " / ".join(str(x) for x in [r.project, r.chip, r.run, r.well] if x)
        cells = []
        for _col, complete_thresh, running_val in STAGE_COLS:
            if r.failed and r.stage_num is not None and r.stage_num == running_val:
                css = "failed"
            elif r.stage_num is None:
                css = "not_run"
            elif r.stage_num >= complete_thresh:
                css = "complete"
            elif r.stage_num == running_val:
                css = "running"
            else:
                css = "not_run"
            cells.append(html.Td(
                html.Button(html.Span(className="dot"),
                            className=f"cell-btn {css}",
                            id={"pipe-cell": i}, n_clicks=0),
            ))

        body_rows.append(html.Tr(
            [html.Td(well_label, className="well-cell")]
            + cells
            + [html.Td(str(r.num_units) if r.num_units is not None else "—", className="num")],
            id={"pipe-row": i}, style={"cursor": "pointer"},
        ))

    table = html.Table([html.Thead(header), html.Tbody(body_rows)], className="matrix")
    return None, html.Div(table, className="tbl-wrap"), count_pill


# ── Inspector ─────────────────────────────────────────────────────────────────

@callback(
    Output("pipe-inspector", "children"),
    Output("pipe-selected-idx", "data"),
    Input({"pipe-row": ALL}, "n_clicks"),
    State("pipe-selected-idx", "data"),
    Input("dashboard-url", "pathname"),
    prevent_initial_call=True,
)
def _show_inspector(row_clicks, selected, _path):
    from dash import ctx as dash_ctx
    if not dash_ctx.triggered_id or not isinstance(dash_ctx.triggered_id, dict):
        return _empty_inspector(), selected

    row_idx = dash_ctx.triggered_id.get("pipe-row")
    if row_idx is None:
        return _empty_inspector(), selected

    try:
        ctx = current_app.config.get("MEA", {})
        checkpoint_dir = ctx.get("checkpoint_dir")
    except RuntimeError:
        return _empty_inspector(), selected

    df = load_checkpoints(checkpoint_dir)
    if df.empty or row_idx >= len(df):
        return _empty_inspector(), selected

    r = df.iloc[int(row_idx)]
    items = [
        ("project", r.get("project")), ("date", r.get("date")),
        ("chip", r.get("chip")), ("run", r.get("run")),
        ("well", r.get("well")), ("rec", r.get("rec")),
        ("stage", r.get("stage")), ("num_units", r.get("num_units")),
        ("last_updated", r.get("last_updated")), ("file", r.get("file")),
    ]
    kv = html.Dl(
        [child for k, v in items for child in [
            html.Dt(k), html.Dd(str(v) if v is not None else "—", className="path")
        ]],
        className="kv",
    )

    error_block = None
    err = r.get("error")
    if err:
        error_block = html.Div([
            html.Div("error", className="section-label", style={"marginTop": "12px"}),
            html.Pre(str(err), className="code",
                     style={"background": "var(--fail-soft)",
                            "borderColor": "rgba(224,49,49,0.35)",
                            "color": "#b02525", "maxHeight": "100px"}),
        ])

    reset_block = html.Div(
        [
            html.Div("reset stage", className="section-label",
                     style={"marginTop": "14px"}),
            html.Div(
                [
                    dcc.Dropdown(
                        id="insp-reset-to",
                        options=_RESET_TO_OPTIONS,
                        value="", clearable=False,
                        style={"flex": "1", "fontFamily": "var(--font-mono)", "fontSize": "11px"},
                    ),
                    html.Button(
                        [html.Span("⟲", className="glyph"), "Reset"],
                        id="insp-reset-btn", n_clicks=0, className="btn primary",
                        style={"height": "32px"},
                    ),
                ],
                style={"display": "flex", "gap": "8px", "alignItems": "center",
                       "marginTop": "6px"},
            ),
            html.Div(id="insp-reset-status",
                     style={"fontFamily": "var(--font-mono)", "fontSize": "11px",
                            "color": "var(--ink-3)", "marginTop": "6px",
                            "minHeight": "16px"}),
        ]
    )

    delete_block = html.Div(
        [
            html.Div(
                style={"borderTop": "1px solid var(--line-soft)", "margin": "14px 0 0"}
            ),
            html.Div("delete checkpoint", className="section-label",
                     style={"marginTop": "10px", "color": "var(--fail)"}),
            dcc.Checklist(
                id="insp-delete-confirm",
                options=[{"label": "  Permanently delete this file", "value": "confirmed"}],
                value=[],
                style={"fontFamily": "var(--font-mono)", "fontSize": "11px",
                       "color": "var(--ink-3)", "marginBottom": "6px"},
            ),
            html.Button(
                [html.Span("✕", className="glyph"), "Delete file"],
                id="insp-delete-btn", n_clicks=0, className="btn",
                style={"height": "32px", "borderColor": "var(--fail)", "color": "var(--fail)"},
            ),
            html.Div(id="insp-delete-status",
                     style={"fontFamily": "var(--font-mono)", "fontSize": "11px",
                            "color": "var(--ink-3)", "marginTop": "6px", "minHeight": "16px"}),
        ]
    )

    body = [kv]
    if error_block:
        body.append(error_block)
    body.append(reset_block)
    body.append(delete_block)

    return html.Div([
        html.Div(
            [html.Div(str(r.get("well") or "Well"), style={"fontWeight": 600}),
             html.Div(str(r.get("stage") or ""), style={"color": "var(--ink-3)",
                      "fontFamily": "var(--font-mono)", "fontSize": "11px"})],
            className="inspector-head",
        ),
        html.Div(body, className="inspector-body"),
    ]), row_idx


@callback(
    Output("insp-reset-status", "children"),
    Input("insp-reset-btn", "n_clicks"),
    State("insp-reset-to", "value"),
    State("pipe-selected-idx", "data"),
    prevent_initial_call=True,
)
def _inspector_reset(_n, to_stage, row_idx):
    if not to_stage:
        return "Pick a stage first."
    if row_idx is None:
        return "No well selected."
    try:
        ctx = current_app.config.get("MEA", {})
        checkpoint_dir = ctx.get("checkpoint_dir")
    except RuntimeError:
        return "Error: no app context."
    df = load_checkpoints(checkpoint_dir)
    if df.empty or row_idx >= len(df):
        return "Well not found."
    path = df.iloc[int(row_idx)]["path"]
    ok, err = reset_checkpoint(path, int(to_stage))
    if ok:
        return f"Reset to stage {to_stage}. Refresh to update."
    return f"Failed: {err}"


# ── Bulk reset ────────────────────────────────────────────────────────────────

@callback(
    Output("bulk-status", "children"),
    Input("bulk-preview-btn", "n_clicks"),
    State("bulk-filter-stage", "value"),
    State("bulk-reset-to", "value"),
    State("bulk-failed-only", "value"),
    prevent_initial_call=True,
)
def _bulk_preview(_n, filter_stage, reset_to, failed_only):
    if not reset_to:
        return "Pick a reset-to stage."
    try:
        ctx = current_app.config.get("MEA", {})
        checkpoint_dir = ctx.get("checkpoint_dir")
    except RuntimeError:
        return "No app context."
    df = load_checkpoints(checkpoint_dir)
    subset = df
    if filter_stage and filter_stage != "all":
        subset = df[df["stage"] == filter_stage]
    if "failed" in (failed_only or []):
        subset = subset[subset["failed"] == True]  # noqa: E712
    return f"Preview: {len(subset)} checkpoint(s) would reset to stage {reset_to}."


@callback(
    Output("bulk-status", "children", allow_duplicate=True),
    Input("bulk-execute-btn", "n_clicks"),
    State("bulk-filter-stage", "value"),
    State("bulk-reset-to", "value"),
    State("bulk-failed-only", "value"),
    prevent_initial_call=True,
)
def _bulk_execute(_n, filter_stage, reset_to, failed_only):
    if not reset_to:
        return "Pick a reset-to stage."
    try:
        ctx = current_app.config.get("MEA", {})
        checkpoint_dir = ctx.get("checkpoint_dir")
    except RuntimeError:
        return "No app context."
    df = load_checkpoints(checkpoint_dir)
    ok, fail = bulk_reset_checkpoints(
        df,
        to_stage_num=int(reset_to),
        filter_stage=filter_stage if filter_stage != "all" else None,
        filter_failed_only="failed" in (failed_only or []),
    )
    return f"Reset {ok} checkpoint(s){f', {fail} failed' if fail else ''}. Refresh to update."


@callback(
    Output("insp-delete-status", "children"),
    Input("insp-delete-btn", "n_clicks"),
    State("insp-delete-confirm", "value"),
    State("pipe-selected-idx", "data"),
    prevent_initial_call=True,
)
def _inspector_delete(_n, confirmed, row_idx):
    if "confirmed" not in (confirmed or []):
        return "Check the box to confirm deletion."
    if row_idx is None:
        return "No well selected."
    try:
        ctx = current_app.config.get("MEA", {})
        checkpoint_dir = ctx.get("checkpoint_dir")
    except RuntimeError:
        return "Error: no app context."
    df = load_checkpoints(checkpoint_dir)
    if df.empty or row_idx >= len(df):
        return "Well not found."
    path = df.iloc[int(row_idx)]["path"]
    ok, err = delete_checkpoint(path)
    if ok:
        return "Deleted. Refresh to update."
    return f"Failed: {err}"


@callback(
    Output("bulk-del-status", "children"),
    Input("bulk-del-preview-btn", "n_clicks"),
    State("bulk-del-filter-stage", "value"),
    State("bulk-del-failed-only", "value"),
    prevent_initial_call=True,
)
def _bulk_del_preview(_n, filter_stage, failed_only):
    try:
        ctx = current_app.config.get("MEA", {})
        checkpoint_dir = ctx.get("checkpoint_dir")
    except RuntimeError:
        return "No app context."
    df = load_checkpoints(checkpoint_dir)
    subset = df
    if filter_stage and filter_stage != "all":
        subset = df[df["stage"] == filter_stage]
    if "failed" in (failed_only or []):
        subset = subset[subset["failed"] == True]  # noqa: E712
    return f"Preview: {len(subset)} checkpoint file(s) would be deleted."


@callback(
    Output("bulk-del-status", "children", allow_duplicate=True),
    Input("bulk-del-execute-btn", "n_clicks"),
    State("bulk-del-filter-stage", "value"),
    State("bulk-del-failed-only", "value"),
    State("bulk-del-confirm", "value"),
    prevent_initial_call=True,
)
def _bulk_del_execute(_n, filter_stage, failed_only, confirmed):
    if "confirmed" not in (confirmed or []):
        return "Check the confirmation box first."
    try:
        ctx = current_app.config.get("MEA", {})
        checkpoint_dir = ctx.get("checkpoint_dir")
    except RuntimeError:
        return "No app context."
    df = load_checkpoints(checkpoint_dir)
    ok, fail = bulk_delete_checkpoints(
        df,
        filter_stage=filter_stage if filter_stage != "all" else None,
        filter_failed_only="failed" in (failed_only or []),
    )
    return f"Deleted {ok} file(s){f', {fail} failed' if fail else ''}. Refresh to update."


def _empty_matrix():
    return html.Div(
        "No checkpoint data.",
        style={"padding": "24px 14px", "color": "var(--ink-3)",
               "fontFamily": "var(--font-mono)", "fontSize": "12px"},
    )


def _empty_inspector():
    return html.Div([
        html.Div(html.Div("Inspector", style={"fontWeight": 600}), className="inspector-head"),
        html.Div(
            html.Div("Click a row to inspect.",
                     style={"color": "var(--ink-3)", "fontFamily": "var(--font-mono)",
                            "fontSize": "12px"}),
            className="inspector-body",
        ),
    ])
