"""Burst inspector — single-well network burst detail."""

from __future__ import annotations

import json
from pathlib import Path

import dash
from dash import Input, Output, callback, dcc, html
from flask import current_app

from dashboards.components import no_config_banner
from dashboards.data import load_network_rows

dash.register_page(__name__, path="/burst-inspector", name="Burst Inspector", order=6)

layout = html.Div(
    [
        html.Div(
            [
                html.H1("Burst Inspector"),
                html.Div("Single-well network burst detail", className="subtitle"),
            ],
            className="view-head",
        ),
        html.Div(id="binsp-no-config"),
        dcc.Dropdown(
            id="binsp-well-select",
            placeholder="Choose a well…",
            clearable=False,
            style={"fontFamily": "var(--font-mono)", "fontSize": "12px",
                   "marginBottom": "16px"},
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

    checkpoint_dir = ctx.get("checkpoint_dir")
    if checkpoint_dir:
        rows = load_network_rows(checkpoint_dir, from_checkpoints=True)
    else:
        output_root = ((ctx.get("config") or {}).get("io") or {}).get("output_dir") or ""
        if not output_root:
            return html.Div("io.output_dir not set.", className="banner warn"), [], None
        rows = load_network_rows(output_root, from_checkpoints=False)

    options = [
        {"label": f"{r['run']} / {r['well']} ({r['chip']})", "value": r["path"]}
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
        return _msg("Select a well above.")

    p = Path(well_path)
    if not p.exists():
        return _msg(f"Directory not found: {well_path}")

    # ── network_results.json ─────────────────────────────────────────────────
    nr_file = p / "network_results.json"
    nr = {}
    if nr_file.exists():
        try:
            nr = json.loads(nr_file.read_text())
        except Exception:
            pass

    bl_m = (nr.get("burst_fragments") or {}).get("metrics") or {}
    nb_m = (nr.get("network_bursts")  or {}).get("metrics") or {}
    sb_m = (nr.get("superbursts")     or {}).get("metrics") or {}
    diag = nr.get("diagnostics") or {}

    def _s(m, key, stat):
        return (m.get(key) or {}).get(stat)

    def _f(v, p=4):
        if v is None:
            return "—"
        return f"{v:.{p}g}" if isinstance(v, float) else str(v)

    # ── detection diagnostics ────────────────────────────────────────────────
    diag_rows = [
        ("n_units",              nr.get("n_units")),
        ("n_bursty_units",       diag.get("n_bursty_units")),
        ("bin_size_ms",          diag.get("bin_size_ms")),
        ("reference_isi_s",      diag.get("reference_isi_s")),
        ("isi_source",           diag.get("reference_isi_source")),
        ("participation_baseline", diag.get("participation_baseline")),
        ("participation_mad",    diag.get("participation_mad")),
        ("participation_bc",     diag.get("participation_bc")),
        ("detection_threshold",  diag.get("detection_threshold")),
        ("threshold_source",     diag.get("threshold_source")),
        ("min_units_for_burst",  diag.get("min_units_for_burst")),
        ("fragment_merge_gap_s", diag.get("fragment_merge_gap_s")),
        ("frag_gap_source",      diag.get("fragment_merge_gap_source")),
        ("nb_merge_gap_s",       diag.get("nb_merge_gap_s")),
    ]

    diag_table = html.Table(
        [html.Tbody([
            html.Tr([html.Td(k, className="mono"), html.Td(_f(v), className="num")])
            for k, v in diag_rows if v is not None
        ])],
        className="tbl",
    )

    # ── burst metrics ────────────────────────────────────────────────────────
    metric_rows = [
        ("count",              bl_m.get("burst_count"),       nb_m.get("burst_count"),       sb_m.get("burst_count")),
        ("rate (Hz)",          bl_m.get("burst_rate_hz"),     nb_m.get("burst_rate_hz"),     sb_m.get("burst_rate_hz")),
        ("duration mean (s)",  _s(bl_m,"burst_duration_s","mean"),  _s(nb_m,"burst_duration_s","mean"),  _s(sb_m,"burst_duration_s","mean")),
        ("duration CV",        _s(bl_m,"burst_duration_s","cv"),    _s(nb_m,"burst_duration_s","cv"),    _s(sb_m,"burst_duration_s","cv")),
        ("IBI mean (s)",       _s(bl_m,"ifbi_s","mean"),            _s(nb_m,"ibi_s","mean"),             _s(sb_m,"isbi_s","mean")),
        ("IBI CV",             _s(bl_m,"ifbi_s","cv"),              _s(nb_m,"ibi_s","cv"),               _s(sb_m,"isbi_s","cv")),
        ("participation mean", _s(bl_m,"participation_fraction","mean"), _s(nb_m,"participation_fraction","mean"), _s(sb_m,"participation_fraction","mean")),
        ("spikes/burst mean",  _s(bl_m,"spike_count_per_burst","mean"), _s(nb_m,"spike_count_per_burst","mean"),  _s(sb_m,"spike_count_per_burst","mean")),
        ("peak rate mean (Hz)",_s(bl_m,"peak_population_firing_rate_hz","mean"), _s(nb_m,"peak_population_firing_rate_hz","mean"), _s(sb_m,"peak_population_firing_rate_hz","mean")),
        ("burst area mean",    _s(bl_m,"burst_area","mean"),         _s(nb_m,"burst_area","mean"),        _s(sb_m,"burst_area","mean")),
        ("peak participation", _s(bl_m,"peak_participation_fraction","mean"), _s(nb_m,"peak_participation_fraction","mean"), _s(sb_m,"peak_participation_fraction","mean")),
    ]

    metric_table = html.Table(
        [
            html.Thead(html.Tr([
                html.Th("metric", style={"textAlign": "left"}),
                html.Th("fragments"),
                html.Th("network bursts"),
                html.Th("superbursts"),
            ])),
            html.Tbody([
                html.Tr([
                    html.Td(label, className="mono", style={"paddingRight": "16px"}),
                    html.Td(_f(bf), className="num"),
                    html.Td(_f(nb), className="num"),
                    html.Td(_f(sb), className="num"),
                ])
                for label, bf, nb, sb in metric_rows
                if not (bf is None and nb is None and sb is None)
            ]),
        ],
        className="tbl",
    )

    # ── participation signal scalars (no chart) ──────────────────────────────
    npz_file = p / "network_plot_data.npz"
    npz_section = None
    if npz_file.exists():
        try:
            import numpy as np
            d = np.load(str(npz_file))
            baseline  = float(d["participation_baseline"]) if "participation_baseline"        in d else None
            threshold = float(d["detection_threshold"])    if "detection_threshold"           in d else None
            n_bins    = int(d["time_s"].shape[0])          if "time_s"                        in d else None
            duration  = float(d["time_s"][-1])             if "time_s"                        in d else None
            sig_max   = float(d["participation_fraction_signal"].max()) if "participation_fraction_signal" in d else None
            npz_rows = [
                ("baseline",         baseline),
                ("threshold",        threshold),
                ("signal max",       sig_max),
                ("duration (s)",     duration),
                ("time bins",        n_bins),
            ]
            npz_section = html.Div([
                html.Div("participation signal (scalars)", style={"fontWeight": "600",
                         "marginBottom": "6px", "fontFamily": "var(--font-mono)", "fontSize": "11px"}),
                html.Table(
                    [html.Tbody([
                        html.Tr([html.Td(k, className="mono"), html.Td(_f(v), className="num")])
                        for k, v in npz_rows if v is not None
                    ])],
                    className="tbl",
                ),
            ], style={"marginTop": "16px"})
        except Exception:
            pass

    # ── raster file list (no embedding) ─────────────────────────────────────
    svgs = sorted(p.glob("*raster_burst_plot.svg"))
    raster_section = None
    if svgs:
        raster_section = html.Div([
            html.Div("raster files", style={"fontWeight": "600", "marginBottom": "6px",
                     "fontFamily": "var(--font-mono)", "fontSize": "11px"}),
            html.Ul([
                html.Li(f.name, style={"fontFamily": "var(--font-mono)", "fontSize": "11px"})
                for f in svgs
            ]),
            html.Div(str(p), style={"fontFamily": "var(--font-mono)", "fontSize": "10px",
                                    "color": "var(--ink-3)", "marginTop": "4px"}),
        ], style={"marginTop": "16px"})

    return html.Div(
        [
            html.Div("detection diagnostics", style={"fontWeight": "600", "marginBottom": "6px",
                     "fontFamily": "var(--font-mono)", "fontSize": "11px"}),
            diag_table,
            html.Div("burst metrics", style={"fontWeight": "600", "marginTop": "16px",
                     "marginBottom": "6px", "fontFamily": "var(--font-mono)", "fontSize": "11px"}),
            html.Div(metric_table, className="tbl-wrap"),
            npz_section or html.Div(),
            raster_section or html.Div(),
        ],
        style={"marginTop": "8px"},
    )


def _msg(text: str) -> html.Div:
    return html.Div(text, style={"padding": "16px 0", "color": "var(--ink-3)",
                                 "fontFamily": "var(--font-mono)", "fontSize": "12px"})
