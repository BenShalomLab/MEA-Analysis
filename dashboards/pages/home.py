"""Home page — KPI overview of pipeline state."""

from __future__ import annotations

import dash
from dash import Input, Output, callback, dcc, html
from flask import current_app

from dashboards.components import no_config_banner
from dashboards.data import checkpoint_kpis, load_checkpoints

dash.register_page(__name__, path="/", name="Home", order=0)


def _kpi(label: str, value: object, sub: str = "", cls: str = "") -> html.Div:
    return html.Div(
        [
            html.Div(label, className="label"),
            html.Div(str(value), className="value"),
            html.Div(sub, className="sub") if sub else None,
        ],
        className=f"kpi{' ' + cls if cls else ''}",
    )


layout = html.Div(
    [
        dcc.Interval(id="home-interval", interval=30_000, n_intervals=0),
        html.Div(
            [
                html.Div(
                    [
                        html.Div(
                            [html.Span("operations", className="breadcrumb"), html.Span("home")],
                            className="breadcrumb",
                        ),
                        html.H1("Overview"),
                        html.Div("Pipeline checkpoint summary", className="subtitle"),
                    ]
                ),
            ],
            className="view-head",
        ),
        html.Div(id="home-no-config"),
        html.Div(id="home-kpi-grid", className="kpi-grid"),
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
    Output("home-recent-table", "children"),
    Input("home-interval", "n_intervals"),
    Input("dashboard-url", "pathname"),
)
def _refresh(_n, _path):
    try:
        ctx = current_app.config.get("MEA", {})
    except RuntimeError:
        return no_config_banner(), [], _empty_table()

    if not ctx.get("config_exists"):
        return no_config_banner(), [], _empty_table()

    checkpoint_dir = ctx.get("checkpoint_dir")
    if not checkpoint_dir:
        banner = html.Div(
            "checkpoint_dir not set — configure it in Settings.",
            className="banner warn",
        )
        return banner, [], _empty_table()

    df = load_checkpoints(checkpoint_dir)
    kpis = checkpoint_kpis(df)

    complete_pct = (
        f"{kpis['complete'] / kpis['total'] * 100:.0f}% done"
        if kpis["total"] else ""
    )

    tiles = [
        _kpi("Total wells", kpis["total"], "checkpoints found"),
        _kpi("Complete", kpis["complete"], complete_pct, "ok" if kpis["complete"] else ""),
        _kpi("In progress", kpis["running"], "active stages", "run" if kpis["running"] else ""),
        _kpi("Failed", kpis["failed"], "need attention", "fail" if kpis["failed"] else ""),
    ]

    if df.empty:
        return None, tiles, _empty_table()

    recent = (
        df.sort_values("last_updated", ascending=False, na_position="last")
        .head(20)[["project", "date", "chip", "run", "well", "stage", "num_units", "last_updated"]]
    )

    header = html.Tr([
        html.Th(c) for c in ["project", "date", "chip", "run", "well", "stage", "units", "updated"]
    ])
    rows = [
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
    table = html.Table([html.Thead(header), html.Tbody(rows)], className="tbl")
    return None, tiles, html.Div(table, className="tbl-wrap")


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


def _empty_table() -> html.Div:
    return html.Div(
        "No checkpoint data loaded.",
        style={"padding": "24px 14px", "color": "var(--ink-3)",
               "fontFamily": "var(--font-mono)", "fontSize": "12px"},
    )
