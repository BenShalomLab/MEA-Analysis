"""Burst inspector — single-well view with participation signal + SVG raster."""

from __future__ import annotations

import json
from pathlib import Path

import dash
import plotly.graph_objects as go
from dash import Input, Output, State, callback, dcc, html
from flask import current_app

from dashboards.components import no_config_banner
from dashboards.data import load_network_results
from dashboards.theme import apply_default_theme as _adt; _adt()

dash.register_page(__name__, path="/burst-inspector", name="Burst Inspector", order=6)

layout = html.Div(
    [
        dcc.Store(id="binsp-well-data", data=None),
        html.Div(
            [
                html.Div(
                    [
                        html.Div("analysis", className="breadcrumb"),
                        html.H1("Burst Inspector"),
                        html.Div("Single-well network burst detail", className="subtitle"),
                    ]
                ),
            ],
            className="view-head",
        ),
        html.Div(id="binsp-no-config"),

        # Well selector
        html.Div(
            [
                html.Div(
                    [
                        html.Div(
                            [html.Span("select well", className="h-title")],
                            className="card-head",
                        ),
                        html.Div(
                            [
                                dcc.Dropdown(
                                    id="binsp-well-select",
                                    placeholder="Choose a well…",
                                    clearable=False,
                                    style={"fontFamily": "var(--font-mono)", "fontSize": "12px"},
                                ),
                            ],
                            className="card-body",
                            style={"paddingBottom": "12px"},
                        ),
                    ],
                    className="card",
                ),
            ],
            id="binsp-selector-row",
        ),

        html.Div(id="binsp-content"),
    ],
    className="page",
)


@callback(
    Output("binsp-no-config", "children"),
    Output("binsp-well-select", "options"),
    Output("binsp-well-select", "value"),
    Input("dashboard-url", "pathname"),
)
def _populate_selector(_path):
    try:
        ctx = current_app.config.get("MEA", {})
    except RuntimeError:
        return no_config_banner(), [], None

    if not ctx.get("config_exists"):
        return no_config_banner(), [], None

    config = ctx.get("config") or {}
    output_root = (config.get("io") or {}).get("output_dir") or ""
    if not output_root:
        return html.Div("io.output_dir not set.", className="banner warn"), [], None

    rows = load_network_results(output_root)
    options = [
        {"label": f"{r['run']} / {r['well']} ({r['chip']})",
         "value": r["path"]}
        for r in rows
    ]
    return None, options, (options[0]["value"] if options else None)


@callback(
    Output("binsp-content", "children"),
    Input("binsp-well-select", "value"),
    prevent_initial_call=True,
)
def _load_well(well_path):
    if not well_path:
        return _empty_msg("Select a well above.")

    p = Path(well_path)
    if not p.exists():
        return _empty_msg(f"Directory not found: {well_path}")

    # --- network_results.json ---
    nr_file = p / "network_results.json"
    nr = {}
    if nr_file.exists():
        try:
            nr = json.loads(nr_file.read_text())
        except Exception:
            pass

    # KV summary
    nb_m  = (nr.get("network_bursts") or {}).get("metrics") or {}
    bl_m  = (nr.get("burstlets")      or {}).get("metrics") or {}
    sb_m  = (nr.get("superbursts")    or {}).get("metrics") or {}
    diag  = nr.get("diagnostics") or {}

    kv_items = [
        ("n_units",            nr.get("n_units")),
        ("n_bursty_units",     diag.get("n_bursty_units")),
        ("burstlets",          bl_m.get("count")),
        ("network_bursts",     nb_m.get("count")),
        ("superbursts",        sb_m.get("count")),
        ("burst_rate_hz",      nb_m.get("rate")),
        ("mean_burst_dur_s",   (nb_m.get("duration") or {}).get("mean")),
        ("adaptive_bin_ms",    diag.get("adaptive_bin_ms")),
    ]
    kv = html.Dl(
        [child for k, v in kv_items for child in [
            html.Dt(k),
            html.Dd("—" if v is None else (f"{v:.4g}" if isinstance(v, float) else str(v))),
        ]],
        className="kv",
        style={"columns": "2", "columnGap": "32px"},
    )

    kv_card = html.Div(
        [
            html.Div([html.Span("burst summary", className="h-title")], className="card-head"),
            html.Div(kv, className="card-body"),
        ],
        className="card",
    )

    # --- participation signal from npz ---
    sig_card = _participation_card(p)

    # --- SVG raster ---
    raster_card = _raster_card(p)

    return html.Div(
        [kv_card, sig_card, raster_card],
        style={"display": "flex", "flexDirection": "column", "gap": "16px",
               "marginTop": "16px"},
    )


def _participation_card(p: Path) -> html.Div:
    npz_file = p / "network_plot_data.npz"
    if not npz_file.exists():
        return _info_card("participation signal", "network_plot_data.npz not found.")

    try:
        import numpy as np
        d = np.load(str(npz_file))
    except Exception as exc:
        return _info_card("participation signal", f"Load error: {exc}")

    time_bins = d.get("time_bins") if "time_bins" in d else None
    sig       = d.get("participation_signal") if "participation_signal" in d else None
    rate      = d.get("rate_signal") if "rate_signal" in d else None
    baseline  = float(d["participation_baseline"]) if "participation_baseline" in d else None
    threshold = float(d["participation_threshold"]) if "participation_threshold" in d else None

    if sig is None:
        return _info_card("participation signal", "participation_signal key missing in npz.")

    x = time_bins.tolist() if time_bins is not None else list(range(len(sig)))

    traces = [
        go.Scatter(x=x, y=sig.tolist(), mode="lines", name="participation",
                   line=dict(color="oklch(0.52 0.14 165)", width=1.5)),
    ]
    if rate is not None:
        traces.append(go.Scatter(x=x, y=rate.tolist(), mode="lines", name="rate signal",
                                 line=dict(color="oklch(0.62 0.16 60)", width=1, dash="dot"),
                                 opacity=0.7, yaxis="y2"))
    shapes = []
    if baseline is not None:
        shapes.append(dict(type="line", x0=x[0], x1=x[-1], y0=baseline, y1=baseline,
                           line=dict(color="oklch(0.55 0.1 165 / 0.5)", width=1, dash="dash")))
    if threshold is not None:
        shapes.append(dict(type="line", x0=x[0], x1=x[-1], y0=threshold, y1=threshold,
                           line=dict(color="oklch(0.55 0.16 28 / 0.6)", width=1, dash="dash")))

    fig = go.Figure(
        traces,
        layout=go.Layout(
            template="mea_paper",
            margin=dict(l=40, r=10, t=10, b=40),
            shapes=shapes,
            legend=dict(orientation="h", x=0, y=-0.2),
            xaxis=dict(title="time (s)"),
            yaxis=dict(title="participation"),
            yaxis2=dict(title="rate", overlaying="y", side="right", showgrid=False)
            if rate is not None else {},
            height=260,
        ),
    )
    return html.Div(
        [
            html.Div([html.Span("participation signal", className="h-title")], className="card-head"),
            html.Div(dcc.Graph(figure=fig, config={"displayModeBar": False}), className="card-body",
                     style={"padding": "8px 0 0"}),
        ],
        className="card",
    )


def _raster_card(p: Path) -> html.Div:
    svgs = sorted(p.glob("*_raster_burst_plot.svg"))
    if not svgs:
        return _info_card("raster plot", "No *_raster_burst_plot.svg found.")

    tabs = []
    panels = []
    for i, f in enumerate(svgs):
        label = f.stem.replace("_raster_burst_plot", "").replace("_", " ").strip() or f.stem
        tabs.append(html.Button(label, id={"raster-tab": i}, n_clicks=0,
                                className="btn" + (" active" if i == 0 else ""),
                                style={"fontSize": "11px", "padding": "3px 10px"}))
        panels.append(html.Div(
            html.Img(src=_svg_data_uri(f), style={"maxWidth": "100%", "display": "block"}),
            style={"display": "block" if i == 0 else "none"},
            id={"raster-panel": i},
        ))

    return html.Div(
        [
            html.Div(
                [html.Span("raster plot", className="h-title"),
                 html.Div(tabs, className="toggle-group h-actions")],
                className="card-head",
            ),
            html.Div(panels, className="card-body"),
        ],
        className="card",
    )


def _svg_data_uri(path: Path) -> str:
    import base64
    data = base64.b64encode(path.read_bytes()).decode()
    return f"data:image/svg+xml;base64,{data}"


def _info_card(title: str, msg: str) -> html.Div:
    return html.Div(
        [
            html.Div([html.Span(title, className="h-title")], className="card-head"),
            html.Div(msg, className="card-body",
                     style={"color": "var(--ink-3)", "fontFamily": "var(--font-mono)",
                            "fontSize": "12px"}),
        ],
        className="card",
    )


def _empty_msg(msg: str = "No data.") -> html.Div:
    return html.Div(msg, style={"padding": "24px 0", "color": "var(--ink-3)",
                                "fontFamily": "var(--font-mono)", "fontSize": "12px"})
