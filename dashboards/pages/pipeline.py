"""Pipeline page — per-stage status matrix + well inspector.

Rows = wells (from checkpoints). Columns = 5 pipeline stages.
Cell dots: complete / running / failed / not_run.
Click a row → inspector shows error, units, path.
"""

from __future__ import annotations

import dash
from dash import ALL, Input, Output, State, callback, clientside_callback, dcc, html
from flask import current_app

from dashboards.components import no_config_banner
from dashboards.data import STAGE_COLS, load_checkpoints

dash.register_page(__name__, path="/pipeline", name="Pipeline", order=1)

_STAGE_LABELS = ["Preproc", "Sorting", "Merge", "Analyzer", "Reports"]
_FILTER_OPTS = ["all", "complete", "running", "failed", "not_started"]


layout = html.Div(
    [
        dcc.Interval(id="pipe-interval", interval=30_000, n_intervals=0),
        dcc.Store(id="pipe-selected-row", data=None),
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
                                for f, label in zip(_FILTER_OPTS, ["All", "Complete", "Running", "Failed", "Pending"])
                            ],
                            className="toggle-group",
                            id="pipe-filter-group",
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
    ],
    className="page",
)


@callback(
    Output("pipe-no-config", "children"),
    Output("pipe-matrix-body", "children"),
    Output("pipe-count", "children"),
    Input("pipe-interval", "n_intervals"),
    Input("pipe-refresh-btn", "n_clicks"),
    Input({"pipe-filter": ALL}, "n_clicks"),
    Input("dashboard-url", "pathname"),
)
def _refresh_matrix(_n, _refresh, filter_clicks, _path):
    try:
        ctx = current_app.config.get("MEA", {})
    except RuntimeError:
        return no_config_banner(), _empty_matrix(), ""

    if not ctx.get("config_exists"):
        return no_config_banner(), _empty_matrix(), ""

    checkpoint_dir = ctx.get("checkpoint_dir")
    if not checkpoint_dir:
        return (
            html.Div("checkpoint_dir not set.", className="banner warn"),
            _empty_matrix(), "",
        )

    # Determine active filter from which button was clicked last.
    from dash import ctx as dash_ctx
    active_filter = "all"
    if dash_ctx.triggered_id and isinstance(dash_ctx.triggered_id, dict):
        active_filter = dash_ctx.triggered_id.get("pipe-filter", "all")

    df = load_checkpoints(checkpoint_dir)
    if df.empty:
        return None, _empty_matrix(), ""

    # Apply filter
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
        well_label = " / ".join(
            str(x) for x in [r.project, r.chip, r.run, r.well] if x
        )

        # Derive per-stage CSS class
        failed_stage_num: int | None = None
        if r.failed:
            # Try to extract failed stage number from stage string
            stage_str = r.stage or ""
            if stage_str.startswith("FAILED_AT_"):
                from dashboards.data import STAGE_MAP, LEGACY_STAGE_MAP
                name_after = stage_str[len("FAILED_AT_"):]
                rev = {v: k for k, v in {**STAGE_MAP, **LEGACY_STAGE_MAP}.items()}
                failed_stage_num = rev.get(name_after)

        cells = []
        for col_name, complete_thresh, running_val in STAGE_COLS:
            if r.failed and failed_stage_num is not None:
                if r.stage_num is not None and r.stage_num == running_val:
                    css = "failed"
                elif r.stage_num is not None and r.stage_num > running_val:
                    css = "complete"
                else:
                    css = "not_run"
            elif r.stage_num is None:
                css = "not_run"
            elif r.stage_num >= complete_thresh:
                css = "complete"
            elif r.stage_num == running_val:
                css = "running"
            else:
                css = "not_run"

            cells.append(html.Td(
                html.Button(
                    html.Span(className="dot"),
                    className=f"cell-btn {css}",
                    id={"pipe-cell": f"{i}_{col_name}"},
                    n_clicks=0,
                ),
            ))

        body_rows.append(html.Tr(
            [html.Td(well_label, className="well-cell")]
            + cells
            + [html.Td(str(r.num_units) if r.num_units is not None else "—", className="num")],
            id={"pipe-row": i},
            style={"cursor": "pointer"},
        ))

    table = html.Table(
        [html.Thead(header), html.Tbody(body_rows)],
        className="matrix",
    )
    return None, html.Div(table, className="tbl-wrap"), count_pill


@callback(
    Output("pipe-inspector", "children"),
    Output("pipe-selected-row", "data"),
    Input({"pipe-row": ALL}, "n_clicks"),
    State("pipe-selected-row", "data"),
    Input("dashboard-url", "pathname"),
    Input("pipe-interval", "n_intervals"),
    prevent_initial_call=True,
)
def _show_inspector(row_clicks, selected, _path, _n):
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

    if not checkpoint_dir:
        return _empty_inspector(), selected

    df = load_checkpoints(checkpoint_dir)
    if df.empty or row_idx >= len(df):
        return _empty_inspector(), selected

    r = df.iloc[int(row_idx)]

    items = [
        ("project", r.get("project", "—")),
        ("date", r.get("date", "—")),
        ("chip", r.get("chip", "—")),
        ("run", r.get("run", "—")),
        ("well", r.get("well", "—")),
        ("rec", r.get("rec", "—")),
        ("stage", r.get("stage", "—")),
        ("num_units", r.get("num_units", "—")),
        ("last_updated", r.get("last_updated", "—")),
        ("file", r.get("file", "—")),
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
        error_block = html.Div(
            [
                html.Div("error", className="section-label"),
                html.Pre(str(err), className="code"),
            ],
            style={"marginTop": "12px"},
        )

    return html.Div(
        [
            html.Div(
                [
                    html.Div("Well detail", style={"fontWeight": 600, "fontSize": "13px"}),
                    html.Div(
                        str(r.get("well", "")),
                        style={"color": "var(--ink-3)", "fontFamily": "var(--font-mono)", "fontSize": "11px"},
                    ),
                ],
                className="inspector-head",
            ),
            html.Div(
                [kv, error_block] if error_block else [kv],
                className="inspector-body",
            ),
        ]
    ), row_idx


def _empty_matrix() -> html.Div:
    return html.Div(
        "No checkpoint data.",
        style={"padding": "24px 14px", "color": "var(--ink-3)",
               "fontFamily": "var(--font-mono)", "fontSize": "12px"},
    )


def _empty_inspector() -> html.Div:
    return html.Div(
        [
            html.Div(
                html.Div("Inspector", style={"fontWeight": 600, "fontSize": "13px"}),
                className="inspector-head",
            ),
            html.Div(
                html.Div(
                    "Click a row to inspect.",
                    style={"color": "var(--ink-3)", "fontFamily": "var(--font-mono)", "fontSize": "12px"},
                ),
                className="inspector-body",
            ),
        ]
    )
