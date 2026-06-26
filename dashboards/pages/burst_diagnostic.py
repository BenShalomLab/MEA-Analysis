"""Burst diagnostic — cross-well network burst summary."""

from __future__ import annotations

import io

import dash
from dash import Input, Output, State, callback, dcc, html
from flask import current_app

from dashboards.components import no_config_banner
from dashboards.data import invalidate_network_cache, load_network_rows

dash.register_page(__name__, path="/burst-diagnostic", name="Burst Diagnostic", order=5)

layout = html.Div(
    [
        dcc.Download(id="burst-diag-download"),
        html.Div(
            [
                html.Div(
                    [
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
                            "Export CSV",
                            id="burst-diag-export-btn", n_clicks=0,
                            className="btn ghost",
                        ),
                        html.Button(
                            "Refresh",
                            id="burst-diag-refresh", n_clicks=0, className="btn",
                        ),
                    ],
                    className="view-actions",
                ),
            ],
            className="view-head",
        ),
        html.Div(id="burst-diag-no-config"),
        html.Div(id="burst-diag-summary", style={"marginBottom": "8px",
                                                  "fontFamily": "var(--font-mono)",
                                                  "fontSize": "12px", "color": "var(--ink-2)"}),
        html.Div(id="burst-diag-table"),
    ],
    className="page",
)


@callback(
    Output("burst-diag-no-config", "children"),
    Output("burst-diag-summary", "children"),
    Output("burst-diag-table", "children"),
    Output("burst-diag-project-filter", "options"),
    Input("burst-diag-refresh", "n_clicks"),
    Input("burst-diag-project-filter", "value"),
    Input("dashboard-url", "pathname"),
)
def _refresh(_refresh_clicks, project_filter, _path):
    from dash import ctx as dash_ctx

    try:
        ctx = current_app.config.get("MEA", {})
    except RuntimeError:
        return no_config_banner(), "", _empty_msg(), []

    if not ctx.get("config_exists"):
        return no_config_banner(), "", _empty_msg(), []

    force = (dash_ctx.triggered_id == "burst-diag-refresh")
    if force:
        invalidate_network_cache()

    checkpoint_dir = ctx.get("checkpoint_dir")
    if checkpoint_dir:
        rows = load_network_rows(checkpoint_dir, from_checkpoints=True, force=force)
    else:
        output_root = ((ctx.get("config") or {}).get("io") or {}).get("output_dir") or ""
        if not output_root:
            warn = html.Div("io.output_dir not set in config.", className="banner warn")
            return warn, "", _empty_msg(), []
        rows = load_network_rows(output_root, from_checkpoints=False, force=force)

    if not rows:
        return None, "", html.Div("No network_results.json found.",
                                  style={"padding": "16px 0", "color": "var(--ink-3)",
                                         "fontFamily": "var(--font-mono)", "fontSize": "12px"}), []

    projects = sorted({r["project"] or "unknown" for r in rows})
    proj_opts = [{"label": p, "value": p} for p in projects]

    if project_filter:
        rows = [r for r in rows if (r["project"] or "unknown") == project_filter]

    total_wells       = len(rows)
    total_nb          = sum(r["network_bursts_count"] for r in rows)
    mean_rate         = sum(r["burst_rate_hz"] for r in rows) / total_wells if total_wells else 0
    total_superbursts = sum(r["superbursts_count"] for r in rows)

    summary = (
        f"{total_wells} wells | {total_nb} network bursts | "
        f"mean rate {mean_rate:.4f} Hz | {total_superbursts} superbursts"
    )

    header = html.Tr([
        html.Th(c) for c in [
            "project", "chip", "run", "well",
            "n_units", "fragments", "network bursts", "superbursts",
            "burst rate (Hz)", "mean dur (s)",
        ]
    ])
    body_rows = [
        html.Tr([
            html.Td(str(r["project"]),              className="mono"),
            html.Td(str(r["chip"]),                 className="mono"),
            html.Td(str(r["run"]),                  className="mono"),
            html.Td(str(r["well"]),                 className="mono"),
            html.Td(str(r["n_units"] or "—"),       className="num"),
            html.Td(str(r["burstlets_count"]),       className="num"),
            html.Td(str(r["network_bursts_count"]), className="num"),
            html.Td(str(r["superbursts_count"]),    className="num"),
            html.Td(f"{r['burst_rate_hz']:.4f}",   className="num"),
            html.Td(f"{r['mean_burst_dur_s']:.3f}", className="num"),
        ])
        for r in rows
    ]
    table = html.Div(
        html.Table([html.Thead(header), html.Tbody(body_rows)], className="tbl"),
        className="tbl-wrap",
    )

    return None, summary, table, proj_opts


@callback(
    Output("burst-diag-download", "data"),
    Input("burst-diag-export-btn", "n_clicks"),
    State("burst-diag-project-filter", "value"),
    prevent_initial_call=True,
)
def _export(n_clicks, project_filter):
    from collect_network_jsons import collect_from_checkpoints, collect, to_dataframes
    from pathlib import Path

    try:
        ctx = current_app.config.get("MEA", {})
        checkpoint_dir = ctx.get("checkpoint_dir")
    except RuntimeError:
        return dash.no_update

    if checkpoint_dir:
        rows = collect_from_checkpoints(Path(checkpoint_dir))
    else:
        output_root = ((ctx.get("config") or {}).get("io") or {}).get("output_dir") or ""
        if not output_root:
            return dash.no_update
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


def _empty_msg() -> html.Div:
    return html.Div("No data.", style={"padding": "16px 0", "color": "var(--ink-3)",
                                       "fontFamily": "var(--font-mono)", "fontSize": "12px"})
