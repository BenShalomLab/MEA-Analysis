import subprocess
import sys
from pathlib import Path

import pytest


def test_driver_dry_run_with_dummy_h5(tmp_path):
    h5py = pytest.importorskip("h5py")
    if not callable(getattr(h5py, "File", None)):
        pytest.skip("real h5py is unavailable in this environment")

    dummy_file = tmp_path / "data.raw.h5"
    with h5py.File(dummy_file, "w") as h5f:
        recordings = h5f.create_group("recordings")
        rec = recordings.create_group("rec0001")
        rec.create_group("well000")
        rec.create_group("well001")

    repo_root = Path(__file__).resolve().parents[1]
    driver_script = repo_root / "run_pipeline_driver.py"
    proc = subprocess.run(
        [sys.executable, str(driver_script), str(dummy_file), "--dry"],
        capture_output=True,
        text=True,
        check=False,
    )

    assert proc.returncode == 0, proc.stderr
    assert "[DRY-RUN] Would process" in proc.stdout
