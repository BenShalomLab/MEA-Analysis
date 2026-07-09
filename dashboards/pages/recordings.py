"""Recordings page — browse checkpoints grouped by recording.

Groups checkpoint rows by (project, date, chip, run) to form "recordings".
Left rail: recording cards with progress bar.
Right: wells table for selected recording.
"""

from __future__ import annotations

import dash
from dash import ALL, Input, Output, State, callback, dcc, html
from flask import current_app

from dashboards.components import no_config_banner
from dashboards.data import TERMINAL_STAGE, load_checkpoints

dash.register_page(__name__, path="/recordings", name="Recordings", order=2)


layout = html.Div(
    [
        dcc.Store(id="rec-selected-key", data=""),
        dcc.Interval(id="rec-interval", interval=60_000, n_intervals=0),
        html.Div(
            [
                html.Div(
                    [
                        html.Div("recordings", className="breadcrumb"),
                        html.H1("Recordings"),
                        html.Div("Grouped by project / chip / run", className="subtitle"),
                    ]
                ),
                html.Div(
                    html.Button("↺ Refresh", id="rec-refresh-btn", n_clicks=0, className="btn"),
                    className="view-actions",
                ),
            ],
            className="view-head",
        ),
        html.Div(id="rec-no-config"),
        dcc.Loading(
            html.Div(
                [
                    html.Div(id="rec-list", className="rec-list"),
                    html.Div(id="rec-detail", className="grow"),
                ],
                className="split-h",
            ),
            type="circle",
            color="var(--accent)",
            style={"minHeight": "120px"},
        ),
    ],
    className="page",
)


@callback(
    Output("rec-no-config", "children"),
    Output("rec-list", "children"),
    Input("rec-interval", "n_intervals"),
    Input("rec-refresh-btn", "n_clicks"),
    Input("dashboard-url", "pathname"),
)
def _render_list(_n, _refresh, _path):
    try:
        ctx = current_app.config.get("MEA", {})
    except RuntimeError:
        return no_config_banner(), []

    if not ctx.get("config_exists"):
        return no_config_banner(), []

    checkpoint_dir = ctx.get("checkpoint_dir")
    if not checkpoint_dir:
        return html.Div("checkpoint_dir not set.", className="banner warn"), []

    df = load_checkpoints(checkpoint_dir)
    if df.empty:
        return None, html.Div(
            "No checkpoints found.",
            style={"padding": "16px", "color": "var(--ink-3)",
                   "fontFamily": "var(--font-mono)", "fontSize": "12px"},
        )

    # Group by recording key
    df["rec_key"] = df.apply(
        lambda r: "_".join(str(r[c] or "") for c in ["project", "date", "chip", "run"]),
        axis=1,
    )
    groups = df.groupby("rec_key", sort=False)

    prev_project = None
    cards = []
    for key, grp in sorted(groups, key=lambda x: x[0]):
        project = grp["project"].iloc[0] or ""
        if project != prev_project:
            cards.append(html.Div(project or "(unknown)", className="rec-group-header"))
            prev_project = project

        n_wells = len(grp)
        n_complete = (grp["stage"] == TERMINAL_STAGE).sum()
        n_failed = grp["failed"].sum()
        pct = n_complete / n_wells if n_wells else 0

        date = grp["date"].iloc[0] or "?"
        chip = grp["chip"].iloc[0] or "?"
        run = grp["run"].iloc[0] or "?"

        progress = html.Div(
            [
                html.Span(style={"width": f"{pct * 100:.0f}%", "background": "var(--ok)"}),
                html.Span(style={"width": f"{n_failed / n_wells * 100:.0f}%",
                                  "background": "var(--fail)"}),
            ],
            className="rec-progress",
        )

        cards.append(html.Button(
            [
                html.Div(
                    [
                        html.Span(f"{chip} / {run}", style={"fontWeight": 600}),
                        html.Span(f"{n_complete}/{n_wells}", style={"color": "var(--ink-3)"}),
                    ],
                    className="rec-card-title",
                ),
                html.Div(
                    [
                        html.Span(date, style={"color": "var(--ink-3)"}),
                        html.Span(f"{n_failed} failed", style={"color": "var(--fail)"}) if n_failed else None,
                    ],
                    className="rec-card-meta",
                ),
                progress,
            ],
            id={"rec-card": key},
            n_clicks=0,
            className="rec-card",
        ))

    return None, cards


@callback(
    Output("rec-selected-key", "data"),
    Input({"rec-card": ALL}, "n_clicks"),
    prevent_initial_call=True,
)
def _select_recording(clicks):
    from dash import ctx as dash_ctx
    if not dash_ctx.triggered_id or not isinstance(dash_ctx.triggered_id, dict):
        return ""
    return dash_ctx.triggered_id.get("rec-card", "")


@callback(
    Output("rec-detail", "children"),
    Output("mea-rerun-dir", "data"),
    Input("rec-selected-key", "data"),
    Input("rec-interval", "n_intervals"),
    Input({"rec-rerun": ALL}, "n_clicks"),
    State("mea-rerun-dir", "data"),
    prevent_initial_call=False,
)
def _render_detail(key, _n, rerun_clicks, _store):
    from dash import ctx as dash_ctx

    # If rerun button clicked, just update the store (detail re-render happens on next input)
    if dash_ctx.triggered_id and isinstance(dash_ctx.triggered_id, dict) and \
            "rec-rerun" in dash_ctx.triggered_id:
        data_dir = dash_ctx.triggered_id["rec-rerun"]
        return dash.no_update, data_dir

    if not key:
        return html.Div(
            "Select a recording to view well details.",
            style={"padding": "24px", "color": "var(--ink-3)",
                   "fontFamily": "var(--font-mono)", "fontSize": "12px"},
        ), dash.no_update

    try:
        ctx = current_app.config.get("MEA", {})
        checkpoint_dir = ctx.get("checkpoint_dir")
    except RuntimeError:
        return [], dash.no_update

    if not checkpoint_dir:
        return [], dash.no_update

    df = load_checkpoints(checkpoint_dir)
    df["rec_key"] = df.apply(
        lambda r: "_".join(str(r[c] or "") for c in ["project", "date", "chip", "run"]),
        axis=1,
    )
    grp = df[df["rec_key"] == key]
    if grp.empty:
        return html.Div("Recording not found.", className="banner warn"), dash.no_update

    r0 = grp.iloc[0]
    data_dir = str(r0.get("data_dir") or "")

    meta_items = [
        ("project", r0.get("project", "—")),
        ("date",    r0.get("date", "—")),
        ("chip",    r0.get("chip", "—")),
        ("run",     r0.get("run", "—")),
        ("wells",   str(len(grp))),
    ]
    meta = html.Dl(
        [child for k, v in meta_items for child in [html.Dt(k), html.Dd(str(v or "—"))]],
        className="kv",
    )

    data_dir_row = html.Div(
        [
            html.Div("data dir", className="section-label"),
            html.Div(
                [
                    html.Code(
                        data_dir or "—",
                        style={"fontSize": "11px", "wordBreak": "break-all",
                               "flex": "1", "userSelect": "all"},
                    ),
                    dcc.Link(
                        html.Button("Rerun →", className="btn ghost",
                                    style={"fontSize": "11px", "whiteSpace": "nowrap"}),
                        href="/run",
                        id={"rec-rerun": data_dir},
                        style={"textDecoration": "none"},
                    ) if data_dir else None,
                ],
                style={"display": "flex", "alignItems": "flex-start",
                       "gap": "10px", "padding": "6px 0"},
            ),
        ],
        style={"padding": "0 0 8px"},
    )

    header = html.Tr([
        html.Th(c) for c in ["well", "stage", "units", "inspect", "updated"]
    ])
    rows = [
        html.Tr([
            html.Td(str(r.well or "—"), className="mono"),
            html.Td(_stage_pill(r.stage, r.failed)),
            html.Td(str(r.num_units) if r.num_units is not None else "—", className="num"),
            html.Td(
                dcc.Link("→", href="/burst-inspector",
                         style={"fontFamily": "var(--font-mono)", "fontSize": "12px"})
                if r.stage == TERMINAL_STAGE else html.Span("—", className="muted")
            ),
            html.Td(str(r.last_updated or "—"), className="muted"),
        ])
        for r in grp.itertuples()
    ]
    table = html.Table([html.Thead(header), html.Tbody(rows)], className="tbl")

    detail = html.Div(
        [
            html.Div(
                [html.Span("recording detail", className="h-title")],
                className="card-head",
            ),
            html.Div(meta, className="card-body"),
            html.Div(data_dir_row, className="card-body",
                     style={"borderTop": "1px solid var(--line-soft)"}),
            html.Div(
                [html.Div("wells", className="section-label"),
                 html.Div(table, className="tbl-wrap")],
                className="card-body",
            ),
        ],
        className="card",
        style={"margin": "0 0 0 1px"},
    )
    return detail, dash.no_update


def _stage_pill(stage: str, failed: bool) -> html.Span:
    if failed:
        cls = "pill fail"
    elif stage == TERMINAL_STAGE:
        cls = "pill ok"
    elif stage in {"PREPROCESSING", "SORTING", "MERGE", "ANALYZER", "REPORTS"}:
        cls = "pill run"
    else:
        cls = "pill idle"
    return html.Span([html.Span(className="swatch"), stage], className=cls)
