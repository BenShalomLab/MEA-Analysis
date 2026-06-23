"""Burst diagnostic — cross-well network burst summary."""

from __future__ import annotations

import dash
import plotly.graph_objects as go
from dash import Input, Output, callback, dcc, html
from flask import current_app

from dashboards.components import no_config_banner
from dashboards.data import load_network_results
from dashboards.theme import apply_default_theme as _adt; _adt()

dash.register_page(__name__, path="/burst-diagnostic", name="Burst Diagnostic", order=5)

layout = html.Div(
    [
        dcc.Interval(id="burst-diag-interval", interval=60_000, n_intervals=0),
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
                    html.Button(
                        [html.Span("↺", className="glyph"), "Refresh"],
                        id="burst-diag-refresh", n_clicks=0, className="btn",
                    ),
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
    Input("burst-diag-interval", "n_intervals"),
    Input("burst-diag-refresh", "n_clicks"),
    Input("dashboard-url", "pathname"),
)
def _refresh(_n, _refresh, _path):
    try:
        ctx = current_app.config.get("MEA", {})
    except RuntimeError:
        return no_config_banner(), [], _empty_fig(), _empty_msg()

    if not ctx.get("config_exists"):
        return no_config_banner(), [], _empty_fig(), _empty_msg()

    config = ctx.get("config") or {}
    output_root = (config.get("io") or {}).get("output_dir") or ""

    if not output_root:
        warn = html.Div("io.output_dir not set in config.", className="banner warn")
        return warn, [], _empty_fig(), _empty_msg()

    rows = load_network_results(output_root)
    if not rows:
        return None, [], _empty_fig(), html.Div(
            "No network_results.json found under output_dir.",
            style={"padding": "24px 14px", "color": "var(--ink-3)",
                   "fontFamily": "var(--font-mono)", "fontSize": "12px"},
        )

    total_wells      = len(rows)
    total_nb         = sum(r["network_bursts_count"] for r in rows)
    mean_rate        = sum(r["burst_rate_hz"] for r in rows) / total_wells
    total_superbursts = sum(r["superbursts_count"] for r in rows)

    kpis = [
        _kpi("Wells",          str(total_wells)),
        _kpi("Network bursts", str(total_nb)),
        _kpi("Mean burst rate", f"{mean_rate:.3f} Hz"),
        _kpi("Superbursts",    str(total_superbursts)),
    ]

    labels = [f"{r['run']}/{r['well']}" for r in rows]
    fig = go.Figure(
        [
            go.Bar(name="network bursts", x=labels,
                   y=[r["network_bursts_count"] for r in rows],
                   marker_color="var(--ok-hex, #4caf7d)"),
            go.Bar(name="burstlets",       x=labels,
                   y=[r["burstlets_count"] for r in rows],
                   marker_color="var(--run-hex, #d4893a)", opacity=0.7),
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
            "n_units", "burstlets", "network bursts", "superbursts",
            "burst rate (Hz)", "mean dur (s)",
        ]
    ])
    body_rows = [
        html.Tr([
            html.Td(str(r["project"]), className="mono"),
            html.Td(str(r["chip"]),    className="mono"),
            html.Td(str(r["run"]),     className="mono"),
            html.Td(str(r["well"]),    className="mono"),
            html.Td(str(r["n_units"] or "—"),            className="num"),
            html.Td(str(r["burstlets_count"]),            className="num"),
            html.Td(str(r["network_bursts_count"]),       className="num"),
            html.Td(str(r["superbursts_count"]),          className="num"),
            html.Td(f"{r['burst_rate_hz']:.4f}",          className="num"),
            html.Td(f"{r['mean_burst_dur_s']:.3f}",       className="num"),
        ])
        for r in rows
    ]
    table = html.Div(
        html.Table([html.Thead(header), html.Tbody(body_rows)], className="tbl"),
        className="tbl-wrap",
    )

    return None, kpis, fig, table


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
