"""Home page — KPI overview + stage throughput + pending queue."""

from __future__ import annotations

import dash
from dash import Input, Output, callback, dcc, html
from flask import current_app

from dashboards.components import no_config_banner
from dashboards.data import STAGE_COLS, checkpoint_kpis, load_checkpoints, stage_throughput

dash.register_page(__name__, path="/", name="Home", order=0)

_STAGE_LABELS = {
    "preproc":  "Preprocessing",
    "sorting":  "Sorting",
    "merge":    "Merge",
    "analyzer": "Analyzer",
    "reports":  "Reports",
}


def _kpi(label: str, value: object, sub: str = "", cls: str = "") -> html.Div:
    return html.Div(
        [
            html.Div(label, className="label"),
            html.Div(str(value), className="value"),
            html.Div(sub, className="sub") if sub else None,
        ],
        className=f"kpi{' ' + cls if cls else ''}",
    )


def _throughput_row(row: dict) -> html.Tr:
    total = max(1, row["total"])
    ok_pct  = f"{100 * row['complete'] / total:.1f}%"
    run_pct = f"{100 * row['running'] / total:.1f}%"
    return html.Tr([
        html.Td(
            _STAGE_LABELS.get(row["name"], row["name"]),
            style={"fontFamily": "var(--font-mono)", "fontSize": "11px",
                   "color": "var(--ink-2)", "padding": "5px 10px 5px 0",
                   "width": "120px", "whiteSpace": "nowrap"},
        ),
        html.Td(
            html.Div(
                [
                    html.Span(style={"background": "var(--ok)",   "width": ok_pct}),
                    html.Span(style={"background": "var(--run)",  "width": run_pct}),
                ],
                style={"display": "flex", "gap": "1px", "height": "6px",
                       "background": "var(--bg-deep)", "borderRadius": "3px",
                       "overflow": "hidden"},
            ),
            style={"padding": "5px 0"},
        ),
        html.Td(
            f"{row['complete']}/{row['total']}",
            style={"fontFamily": "var(--font-mono)", "fontSize": "11px",
                   "color": "var(--ink-3)", "textAlign": "right",
                   "width": "64px", "padding": "5px 0 5px 10px",
                   "fontFeatureSettings": "'tnum'", "whiteSpace": "nowrap"},
        ),
    ])


layout = html.Div(
    [
        dcc.Interval(id="home-interval", interval=30_000, n_intervals=0),
        html.Div(
            [
                html.Div(
                    [
                        html.Div("pipeline ops", className="breadcrumb"),
                        html.H1("Overview"),
                        html.Div("Checkpoint summary", className="subtitle"),
                    ]
                ),
                html.Div(
                    html.Button(
                        [html.Span("↻", className="glyph"), "Refresh"],
                        id="home-refresh-btn", n_clicks=0, className="btn",
                    ),
                    className="view-actions",
                ),
            ],
            className="view-head",
        ),
        html.Div(id="home-no-config"),
        html.Div(id="home-kpi-grid", className="kpi-grid"),
        html.Div(
            [
                # Stage throughput
                html.Div(
                    [
                        html.Div(
                            [html.Span("stage throughput", className="h-title"),
                             html.Div(
                                 [html.Span([html.Span(className="swatch",
                                             style={"background": "var(--ok)"}), " complete"],
                                           style={"display": "inline-flex",
                                                  "alignItems": "center", "gap": "4px"}),
                                  html.Span([html.Span(className="swatch",
                                             style={"background": "var(--run)"}), " running"],
                                           style={"display": "inline-flex",
                                                  "alignItems": "center", "gap": "4px"})],
                                 className="legend h-actions",
                             )],
                            className="card-head",
                        ),
                        html.Div(
                            html.Table(
                                html.Tbody(id="home-throughput-rows"),
                                style={"width": "100%", "borderCollapse": "collapse"},
                            ),
                            className="card-body",
                        ),
                    ],
                    className="card",
                ),
                # Pending queue
                html.Div(
                    [
                        html.Div(
                            [html.Span("pending · next to run", className="h-title"),
                             html.Div(id="home-pending-count", className="h-actions")],
                            className="card-head",
                        ),
                        html.Div(id="home-pending-body", className="card-body flush"),
                    ],
                    className="card",
                ),
            ],
            className="grid-2",
            style={"marginTop": "20px"},
        ),
        html.Div(
            [
                html.Div(
                    [
                        html.Div(
                            [html.Span("recent activity", className="h-title")],
                            className="card-head",
                        ),
                        html.Div(id="home-recent-table", className="card-body flush"),
                    ],
                    className="card grow",
                ),
            ],
            className="row",
            style={"marginTop": "20px"},
        ),
    ],
    className="page",
)


@callback(
    Output("home-no-config", "children"),
    Output("home-kpi-grid", "children"),
    Output("home-throughput-rows", "children"),
    Output("home-pending-body", "children"),
    Output("home-pending-count", "children"),
    Output("home-recent-table", "children"),
    Input("home-interval", "n_intervals"),
    Input("home-refresh-btn", "n_clicks"),
    Input("dashboard-url", "pathname"),
)
def _refresh(_n, _refresh, _path):
    try:
        ctx = current_app.config.get("MEA", {})
    except RuntimeError:
        return no_config_banner(), [], [], _empty(""), "", _empty("")

    if not ctx.get("config_exists"):
        return no_config_banner(), [], [], _empty(""), "", _empty("")

    checkpoint_dir = ctx.get("checkpoint_dir")
    if not checkpoint_dir:
        banner = html.Div("checkpoint_dir not set.", className="banner warn")
        return banner, [], [], _empty(""), "", _empty("")

    df = load_checkpoints(checkpoint_dir)
    kpis = checkpoint_kpis(df)

    complete_pct = (
        f"{kpis['complete'] / kpis['total'] * 100:.0f}% done"
        if kpis["total"] else ""
    )
    tiles = [
        _kpi("Total wells",  kpis["total"],   "checkpoints found"),
        _kpi("Complete",     kpis["complete"], complete_pct, "ok" if kpis["complete"] else ""),
        _kpi("In progress",  kpis["running"],  "active stages",  "run" if kpis["running"] else ""),
        _kpi("Failed",       kpis["failed"],   "need attention", "fail" if kpis["failed"] else ""),
    ]

    if df.empty:
        return None, tiles, [], _empty("No checkpoints."), "", _empty("No data.")

    # Throughput rows
    tp = stage_throughput(df)
    throughput_rows = [_throughput_row(r) for r in tp]

    # Pending queue — wells not yet REPORTS_COMPLETE, grouped by stage
    from dashboards.data import TERMINAL_STAGE
    pending = df[df["stage"] != TERMINAL_STAGE].sort_values("stage_num", ascending=False)
    count_pill = html.Span(str(len(pending)), className="count")

    if pending.empty:
        pending_body = html.Div(
            "All wells complete.",
            style={"padding": "14px", "color": "var(--ink-3)",
                   "fontFamily": "var(--font-mono)", "fontSize": "12px"},
        )
    else:
        header = html.Tr([html.Th(c) for c in ["project", "chip", "run", "well", "stage"]])
        rows = [
            html.Tr([
                html.Td(str(r.project or "—"), className="mono"),
                html.Td(str(r.chip or "—"), className="mono"),
                html.Td(str(r.run or "—"), className="mono"),
                html.Td(str(r.well or "—"), className="mono"),
                html.Td(_stage_pill(r.stage, r.failed)),
            ])
            for r in pending.head(10).itertuples()
        ]
        pending_body = html.Div(
            html.Table([html.Thead(header), html.Tbody(rows)], className="tbl"),
            className="tbl-wrap",
        )

    # Recent activity
    recent = (
        df.sort_values("last_updated", ascending=False, na_position="last")
        .head(20)[["project", "date", "chip", "run", "well", "stage", "num_units", "last_updated"]]
    )
    header = html.Tr([
        html.Th(c) for c in ["project", "date", "chip", "run", "well", "stage", "units", "updated"]
    ])
    recent_rows = [
        html.Tr([
            html.Td(str(r.project or "—"), className="mono"),
            html.Td(str(r.date or "—"), className="mono"),
            html.Td(str(r.chip or "—"), className="mono"),
            html.Td(str(r.run or "—"), className="mono"),
            html.Td(str(r.well or "—"), className="mono"),
            html.Td(_stage_pill(r.stage, r.failed)),
            html.Td(str(r.num_units) if r.num_units is not None else "—", className="num"),
            html.Td(str(r.last_updated or "—"), className="muted"),
        ])
        for r in recent.itertuples()
    ]
    recent_table = html.Div(
        html.Table([html.Thead(header), html.Tbody(recent_rows)], className="tbl"),
        className="tbl-wrap",
    )

    return None, tiles, throughput_rows, pending_body, count_pill, recent_table


def _stage_pill(stage: str, failed: bool) -> html.Span:
    if failed:
        cls = "pill fail"
    elif stage == "REPORTS_COMPLETE":
        cls = "pill ok"
    elif stage in {"PREPROCESSING", "SORTING", "MERGE", "ANALYZER", "REPORTS"}:
        cls = "pill run"
    else:
        cls = "pill idle"
    return html.Span([html.Span(className="swatch"), stage], className=cls)


def _empty(msg: str) -> html.Div:
    return html.Div(
        msg or "No data.",
        style={"padding": "24px 14px", "color": "var(--ink-3)",
               "fontFamily": "var(--font-mono)", "fontSize": "12px"},
    )
