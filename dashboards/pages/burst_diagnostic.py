"""Burst diagnostic — cross-well network burst summary."""

from __future__ import annotations

import io

import dash
import plotly.graph_objects as go
from dash import Input, Output, State, callback, dcc, html
from flask import current_app

from collect_network_jsons import collect_from_checkpoints, collect, to_dataframes
from dashboards.components import no_config_banner
from dashboards.data import load_checkpoints, load_network_results, load_network_results_from_checkpoints
from dashboards.theme import apply_default_theme as _adt; _adt()

dash.register_page(__name__, path="/burst-diagnostic", name="Burst Diagnostic", order=5)

layout = html.Div(
    [
        dcc.Interval(id="burst-diag-interval", interval=60_000, n_intervals=0),
        dcc.Download(id="burst-diag-download"),
        html.Div(
            [
                html.Div(
                    [
                        html.Div("analysis", className="breadcrumb"),
                        html.H1("Burst Diagnostic"),
                        html.Div("Network burst summary across wells", className="subtitle"),
                    ]
                ),
                html.Div(
                    [
                        dcc.Dropdown(
                            id="burst-diag-project-filter",
                            placeholder="All projects",
                            clearable=True,
                            style={"width": "200px", "fontFamily": "var(--font-mono)",
                                   "fontSize": "12px"},
                        ),
                        html.Button(
                            "↓ Export CSV",
                            id="burst-diag-export-btn", n_clicks=0,
                            className="btn ghost",
                        ),
                        html.Button(
                            [html.Span("↺", className="glyph"), "Refresh"],
                            id="burst-diag-refresh", n_clicks=0, className="btn",
                        ),
                    ],
                    className="view-actions",
                ),
            ],
            className="view-head",
        ),
        html.Div(id="burst-diag-no-config"),
        html.Div(id="burst-diag-kpi-grid", className="kpi-grid"),
        html.Div(
            [
                html.Div(
                    [
                        html.Div(
                            [html.Span("network bursts per well", className="h-title")],
                            className="card-head",
                        ),
                        html.Div(
                            dcc.Graph(id="burst-diag-bar", config={"displayModeBar": False},
                                      style={"height": "320px"}),
                            className="card-body",
                            style={"padding": "8px 0 0"},
                        ),
                    ],
                    className="card grow",
                ),
            ],
            className="row",
            style={"marginTop": "20px"},
        ),
        html.Div(
            [
                html.Div(
                    [
                        html.Div(
                            [html.Span("per-well summary", className="h-title")],
                            className="card-head",
                        ),
                        html.Div(id="burst-diag-table", className="card-body flush"),
                    ],
                    className="card grow",
                ),
            ],
            className="row",
            style={"marginTop": "16px"},
        ),
    ],
    className="page",
)


@callback(
    Output("burst-diag-no-config", "children"),
    Output("burst-diag-kpi-grid", "children"),
    Output("burst-diag-bar", "figure"),
    Output("burst-diag-table", "children"),
    Output("burst-diag-project-filter", "options"),
    Input("burst-diag-interval", "n_intervals"),
    Input("burst-diag-refresh", "n_clicks"),
    Input("burst-diag-project-filter", "value"),
    Input("dashboard-url", "pathname"),
)
def _refresh(_n, _refresh, project_filter, _path):
    try:
        ctx = current_app.config.get("MEA", {})
    except RuntimeError:
        return no_config_banner(), [], _empty_fig(), _empty_msg(), []

    if not ctx.get("config_exists"):
        return no_config_banner(), [], _empty_fig(), _empty_msg(), []

    checkpoint_dir = ctx.get("checkpoint_dir")
    if checkpoint_dir:
        rows = load_network_results_from_checkpoints(load_checkpoints(checkpoint_dir))
    else:
        output_root = ((ctx.get("config") or {}).get("io") or {}).get("output_dir") or ""
        if not output_root:
            warn = html.Div("io.output_dir not set in config.", className="banner warn")
            return warn, [], _empty_fig(), _empty_msg(), []
        rows = load_network_results(output_root)

    if not rows:
        return None, [], _empty_fig(), html.Div(
            "No network_results.json found.",
            style={"padding": "24px 14px", "color": "var(--ink-3)",
                   "fontFamily": "var(--font-mono)", "fontSize": "12px"},
        ), []

    projects = sorted({r["project"] or "unknown" for r in rows})
    proj_opts = [{"label": p, "value": p} for p in projects]

    if project_filter:
        rows = [r for r in rows if (r["project"] or "unknown") == project_filter]

    total_wells       = len(rows)
    total_nb          = sum(r["network_bursts_count"] for r in rows)
    mean_rate         = sum(r["burst_rate_hz"] for r in rows) / total_wells if total_wells else 0
    total_superbursts = sum(r["superbursts_count"] for r in rows)

    kpis = [
        _kpi("Wells",           str(total_wells)),
        _kpi("Network bursts",  str(total_nb)),
        _kpi("Mean burst rate", f"{mean_rate:.3f} Hz"),
        _kpi("Superbursts",     str(total_superbursts)),
    ]

    labels = [f"{r['run']}/{r['well']}" for r in rows]
    fig = go.Figure(
        [
            go.Bar(name="network bursts", x=labels,
                   y=[r["network_bursts_count"] for r in rows],
                   marker_color="#0984e3"),
            go.Bar(name="burst fragments", x=labels,
                   y=[r["burstlets_count"] for r in rows],
                   marker_color="#00b894", opacity=0.7),
        ],
        layout=go.Layout(
            template="mea_paper",
            barmode="group",
            margin=dict(l=40, r=10, t=10, b=60),
            legend=dict(orientation="h", x=0, y=-0.25),
            xaxis=dict(tickangle=-40, tickfont=dict(size=10)),
        ),
    )

    header = html.Tr([
        html.Th(c) for c in [
            "project", "chip", "run", "well",
            "n_units", "burst fragments", "network bursts", "superbursts",
            "burst rate (Hz)", "mean dur (s)",
        ]
    ])
    body_rows = [
        html.Tr([
            html.Td(str(r["project"]),                        className="mono"),
            html.Td(str(r["chip"]),                           className="mono"),
            html.Td(str(r["run"]),                            className="mono"),
            html.Td(str(r["well"]),                           className="mono"),
            html.Td(str(r["n_units"] or "—"),                 className="num"),
            html.Td(str(r["burstlets_count"]),                className="num"),
            html.Td(str(r["network_bursts_count"]),           className="num"),
            html.Td(str(r["superbursts_count"]),              className="num"),
            html.Td(f"{r['burst_rate_hz']:.4f}",              className="num"),
            html.Td(f"{r['mean_burst_dur_s']:.3f}",           className="num"),
        ])
        for r in rows
    ]
    table = html.Div(
        html.Table([html.Thead(header), html.Tbody(body_rows)], className="tbl"),
        className="tbl-wrap",
    )

    return None, kpis, fig, table, proj_opts


@callback(
    Output("burst-diag-download", "data"),
    Input("burst-diag-export-btn", "n_clicks"),
    State("burst-diag-project-filter", "value"),
    prevent_initial_call=True,
)
def _export(n_clicks, project_filter):
    try:
        ctx = current_app.config.get("MEA", {})
        checkpoint_dir = ctx.get("checkpoint_dir")
    except RuntimeError:
        return dash.no_update

    if checkpoint_dir:
        from pathlib import Path
        rows = collect_from_checkpoints(Path(checkpoint_dir))
    else:
        output_root = ((ctx.get("config") or {}).get("io") or {}).get("output_dir") or ""
        if not output_root:
            return dash.no_update
        from pathlib import Path
        rows = collect(Path(output_root))

    if not rows:
        return dash.no_update

    dfs = to_dataframes(rows)

    if project_filter and project_filter in dfs:
        df = dfs[project_filter]
        filename = f"network_metrics_{project_filter}.csv"
    else:
        df = dfs["ALL"]
        filename = "network_metrics_all.csv"

    buf = io.StringIO()
    df.to_csv(buf, index=False)
    return dcc.send_string(buf.getvalue(), filename)


def _kpi(label: str, value: str) -> html.Div:
    return html.Div(
        [html.Div(label, className="label"), html.Div(value, className="value")],
        className="kpi",
    )


def _empty_fig() -> go.Figure:
    return go.Figure(layout=go.Layout(template="mea_paper",
                                      margin=dict(l=10, r=10, t=10, b=10)))


def _empty_msg() -> html.Div:
    return html.Div("No data.", style={"padding": "24px 14px", "color": "var(--ink-3)",
                                       "fontFamily": "var(--font-mono)", "fontSize": "12px"})
