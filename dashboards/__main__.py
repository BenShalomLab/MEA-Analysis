"""Entry point: python -m dashboards [options]"""

import argparse
from pathlib import Path


def main():
    p = argparse.ArgumentParser(description="MEA Analysis Dashboard")
    p.add_argument("--config", default="mea_config.json",
                   help="Path to mea_config.json (default: mea_config.json)")
    p.add_argument("--checkpoint-dir", default=None,
                   help="Checkpoint directory (overrides config io.checkpoint_dir)")
    p.add_argument("--port", type=int, default=8050)
    p.add_argument("--host", default="127.0.0.1")
    p.add_argument("--debug", action="store_true")
    args = p.parse_args()

    from dashboards.app import build_app
    app = build_app(
        config_path=Path(args.config),
        checkpoint_dir=Path(args.checkpoint_dir) if args.checkpoint_dir else None,
    )
    app.run(debug=args.debug, host=args.host, port=args.port)


if __name__ == "__main__":
    main()
