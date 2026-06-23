"""Dash app factory: `build_app(config_path, checkpoint_dir)` → Dash."""

from __future__ import annotations

import sys
from pathlib import Path

# Repo root on path so config_loader / mea_checkpoint are importable.
_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from dash import Dash  # noqa: E402

from .components.layout import build_layout  # noqa: E402
from .theme import apply_default_theme  # noqa: E402


_INDEX_STRING = """\
<!DOCTYPE html>
<html data-theme="warm">
    <head>
        {%metas%}
        <title>{%title%}</title>
        {%favicon%}
        {%css%}
    </head>
    <body>
        {%app_entry%}
        <footer>
            {%config%}
            {%scripts%}
            {%renderer%}
        </footer>
    </body>
</html>
"""


def build_app(
    config_path: str | Path = "mea_config.json",
    checkpoint_dir: str | Path | None = None,
) -> Dash:
    """Build the MEA Analysis Dash app.

    config_path    — path to mea_config.json (may not exist yet)
    checkpoint_dir — overrides io.checkpoint_dir from config when provided
    """
    from config_loader import load_config  # lazy: keeps import errors local

    config_path = Path(config_path)
    config_exists = config_path.exists()
    config: dict = load_config(config_path) if config_exists else {}

    if checkpoint_dir is None:
        _cp = config.get("io", {}).get("checkpoint_dir")
        checkpoint_dir = Path(_cp) if _cp else None
    else:
        checkpoint_dir = Path(checkpoint_dir)

    apply_default_theme()

    app = Dash(
        __name__,
        use_pages=True,
        pages_folder=str(Path(__file__).parent / "pages"),
        title="MEA Analysis Dashboard",
        suppress_callback_exceptions=True,
        index_string=_INDEX_STRING,
    )
    app.server.config["MEA"] = {
        "config_path": config_path,
        "config_exists": config_exists,
        "config": config,
        "checkpoint_dir": checkpoint_dir,
    }
    app.layout = build_layout()
    return app
