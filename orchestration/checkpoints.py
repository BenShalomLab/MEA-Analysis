#!/usr/bin/env python3
"""
Read the pipeline's own checkpoint files to report true per-well status.

Why this exists
---------------
The watcher only knows a subprocess exit code. That is a poor description of
what happened: ``run_pipeline_driver.py`` launches one subprocess per
recording x well, so a run can finish with most wells analyzed and a few failed
— for example when a single well hits a CUDA OOM. Exit code 0 does not mean
every well succeeded, and exit code 1 does not mean none did.

``MEAPipeline`` writes a checkpoint per well after every stage:

    <output>/<project>/<date>/<chip>/<run>/<well>/checkpoints/
        <project>_<run>_<well>_checkpoint.json

containing ``stage`` (a ``ProcessingStage`` value), ``failed_stage``, ``error``,
``last_updated``, and ``data_dir`` — the recording the well came from. That last
field is what lets a checkpoint be matched back to the folder the watcher
dispatched, without having to reproduce the driver's output-path logic.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Optional

LOG = logging.getLogger("mea.checkpoints")

# Mirrors mea_checkpoint.ProcessingStage. Duplicated rather than imported so the
# orchestration layer stays importable on machines without the pipeline's deps.
STAGES = {
    0: "Not started",
    1: "Preprocessing", 2: "Preprocessing complete",
    3: "Sorting", 4: "Sorting complete",
    5: "Merge", 6: "Merge complete",
    7: "Analyzer", 8: "Analyzer complete",
    9: "Reports", 10: "Reports complete",
}
FINAL_STAGE = 10


def _norm(state: dict, path: Path) -> dict:
    """Turn a raw checkpoint into the shape the UI consumes."""
    stage = int(state.get("stage", 0) or 0)
    failed = state.get("failed_stage")
    error = state.get("error")

    if failed is not None or error:
        status = "failed"
    elif stage >= FINAL_STAGE:
        status = "complete"
    elif stage == 0:
        status = "pending"
    else:
        status = "running"

    return {
        "well": state.get("well", ""),
        "run_id": state.get("run_id", ""),
        "chip_id": state.get("chip_id", ""),
        "project": state.get("project", ""),
        "date": state.get("date", ""),
        "stage": stage,
        "stage_name": STAGES.get(stage, f"stage {stage}"),
        "progress": round(min(stage / FINAL_STAGE, 1.0), 3),
        "status": status,
        "failed_stage": STAGES.get(failed, failed) if failed is not None else None,
        "error": (str(error)[:600] if error else None),
        "last_updated": state.get("last_updated"),
        "data_dir": state.get("data_dir", ""),
        "output_dir": state.get("output_dir", ""),
        "checkpoint_file": str(path),
    }


def read_checkpoints(search_roots: list[Path], run_dir: Optional[Path] = None,
                     limit: int = 4000) -> list[dict]:
    """Collect checkpoints under ``search_roots``, optionally for one run folder.

    ``run_dir`` filters by the recording each well came from (``data_dir``),
    which is exact — no need to guess how the driver laid out its output.
    """
    wanted = str(Path(run_dir).resolve()) if run_dir else None
    out: list[dict] = []
    seen: set[str] = set()

    for root in search_roots:
        if not root or not Path(root).is_dir():
            continue
        for fp in Path(root).rglob("*_checkpoint.json"):
            key = str(fp.resolve())
            if key in seen:
                continue
            seen.add(key)
            try:
                state = json.loads(fp.read_text())
            except (OSError, json.JSONDecodeError):
                continue
            rec = _norm(state, fp)
            if wanted:
                d = rec.get("data_dir") or ""
                try:
                    if wanted not in str(Path(d).resolve()):
                        continue
                except (OSError, RuntimeError):
                    if wanted not in d:
                        continue
            out.append(rec)
            if len(out) >= limit:
                LOG.warning("Checkpoint scan hit the %d-file limit", limit)
                break

    out.sort(key=lambda r: (r.get("run_id", ""), r.get("well", "")))
    return out


def summarise(rows: list[dict]) -> dict[str, Any]:
    """Roll per-well checkpoints up into a one-line verdict for a run."""
    if not rows:
        return {"wells": 0}

    counts: dict[str, int] = {}
    for r in rows:
        counts[r["status"]] = counts.get(r["status"], 0) + 1

    complete, failed = counts.get("complete", 0), counts.get("failed", 0)
    total = len(rows)

    if failed and complete:
        verdict = "partial"          # the case an exit code hides
    elif failed:
        verdict = "failed"
    elif complete == total:
        verdict = "complete"
    else:
        verdict = "running"

    return {
        "wells": total,
        "complete": complete,
        "failed": failed,
        "running": counts.get("running", 0),
        "pending": counts.get("pending", 0),
        "verdict": verdict,
        "progress": round(sum(r["progress"] for r in rows) / total, 3),
        "errors": [{"well": r["well"], "failed_stage": r["failed_stage"], "error": r["error"]}
                   for r in rows if r["status"] == "failed"][:20],
    }


def main(argv=None) -> None:
    import argparse
    p = argparse.ArgumentParser(description="Report per-well pipeline status from checkpoints")
    p.add_argument("output_dir", type=Path, help="Pipeline --output-dir")
    p.add_argument("--run-dir", type=Path, default=None, help="Only this recording folder")
    p.add_argument("--json", action="store_true")
    a = p.parse_args(argv)

    logging.basicConfig(level=logging.INFO, format="%(message)s")
    rows = read_checkpoints([a.output_dir], a.run_dir)
    if not rows:
        raise SystemExit(f"No checkpoint files found under {a.output_dir}")

    if a.json:
        print(json.dumps({"summary": summarise(rows), "wells": rows}, indent=2))
        return

    s = summarise(rows)
    print(f"{s['wells']} well(s): {s['complete']} complete, {s['failed']} failed, "
          f"{s['running']} running  →  {s['verdict'].upper()}")
    print()
    for r in rows:
        mark = {"complete": "OK  ", "failed": "FAIL", "running": "... ", "pending": "--  "}[r["status"]]
        print(f"  {mark} {r['chip_id']}/{r['run_id']}/{r['well']:<8} "
              f"{r['stage_name']:<22} {r['last_updated'] or ''}")
        if r["error"]:
            first = r["error"].strip().splitlines()[0][:110]
            print(f"       ↳ {r['failed_stage'] or 'error'}: {first}")


if __name__ == "__main__":
    main()
