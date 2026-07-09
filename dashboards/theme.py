"""Plotly template matching the dashboard's clean neutral design.

Hex values mirror the CSS custom properties in `assets/styles.css`:
- `--bg`     → `#f6f8fa`  (app background)
- `--bg-elev`→ `#ffffff`  (plot area / card surface)
- `--ink`    → `#1a1d21`  (text)
- `--ink-3`  → `#6e757c`  (axis labels)
- `--line`   → `#d0d3d9`  (gridlines)
- status:  ok `#1f9d55` · run `#e67700` · fail `#e03131` · info `#1971c2`
"""

from __future__ import annotations

import plotly.graph_objects as go
import plotly.io as pio


TEMPLATE_NAME = "mea_paper"

_PAPER = "#f6f8fa"
_PLOT = "#ffffff"
_INK = "#1a1d21"
_INK3 = "#6e757c"
_LINE = "#d0d3d9"
_LINE_SOFT = "#e5e7eb"

_STATUS_PALETTE = ["#1f9d55", "#e67700", "#e03131", "#1971c2", "#a0a6ad"]


def _build_template() -> go.layout.Template:
    return go.layout.Template(
        layout=dict(
            paper_bgcolor=_PAPER,
            plot_bgcolor=_PLOT,
            font=dict(
                family='"Geist", ui-sans-serif, system-ui, sans-serif',
                size=12,
                color=_INK,
            ),
            colorway=_STATUS_PALETTE,
            xaxis=dict(
                gridcolor=_LINE_SOFT,
                linecolor=_LINE,
                tickcolor=_LINE,
                tickfont=dict(color=_INK3, family='"Geist Mono", monospace', size=10),
                zerolinecolor=_LINE,
            ),
            yaxis=dict(
                gridcolor=_LINE_SOFT,
                linecolor=_LINE,
                tickcolor=_LINE,
                tickfont=dict(color=_INK3, family='"Geist Mono", monospace', size=10),
                zerolinecolor=_LINE,
            ),
            legend=dict(
                bgcolor=_PAPER,
                bordercolor=_LINE,
                borderwidth=1,
                font=dict(size=11, color=_INK),
            ),
            margin=dict(l=48, r=24, t=32, b=40),
            hoverlabel=dict(
                bgcolor=_PAPER,
                bordercolor=_LINE,
                font=dict(color=_INK, family='"Geist Mono", monospace', size=11),
            ),
        )
    )


def apply_default_theme() -> None:
    """Register `mea_paper` and set it as the Plotly default.

    Safe to call more than once: re-registering overwrites in place.
    """
    pio.templates[TEMPLATE_NAME] = _build_template()
    pio.templates.default = TEMPLATE_NAME
