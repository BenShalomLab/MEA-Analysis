# ActivityScan extraction

`orchestration/activity_scan.py` turns the normally-discarded ActivityScan data
into whole-array activity maps, per-well quality metrics, and figures.

## Why it's worth doing

A MaxOne/MaxTwo chip has ~26,400 electrodes but records ~1,020 at once. The
ActivityScan sweeps the whole array to decide which ~1,020 to keep, so the
Network file you spike-sort sees roughly **4% of the chip** — the scan is the
only view of the remaining 96%.

## Why it's cheap

MaxWell stores online threshold-crossing spikes as `(frameno, channel,
amplitude)` next to a `settings/mapping` table with each channel's electrode id
and x/y position in µm. Both are plain uncompressed HDF5, so this needs:

* **no spike sorting** — the spikes are already detected on-chip
* **no GPU**
* **no MaxWell HDF5 compression plugin** — only raw voltage traces need that,
  and we never read them

A 6 GB, 6-well recording processes in about 4 seconds.

## Usage

```bash
# One recording
python orchestration/activity_scan.py /data/240605/M06804/ActivityScan/000170/data.raw.h5

# A whole session (finds every data.raw.h5 under an ActivityScan folder)
python orchestration/activity_scan.py /data/240605 --output-dir /data/scan_out

# Overlay which electrodes the Network recording actually kept
python orchestration/activity_scan.py /data/240605/M06804/ActivityScan/000170 \
    --selection-from /data/240605/M06804/Network/000175/data.raw.h5

# Metrics only, no plots
python orchestration/activity_scan.py <path> --no-figures

# Works on Network recordings too — pass an empty assay filter
python orchestration/activity_scan.py <network>/data.raw.h5 --assay-subfolder ""
```

Requires `h5py`, `numpy`, `matplotlib` — all already in `requirements.txt`.

## Output

Per run, under `<output>/<chip>/<run_id>/`:

| File | Contents |
|---|---|
| `summary.json` | per-well metrics, group labels, scan parameters, QC verdicts |
| `per_electrode.csv` | electrode, x, y, spikes, seconds, rate, amplitude |
| `well<NNN>_activity.png` | firing-rate map, amplitude map, rate distribution |
| `plate_overview.png` | every well on a shared colour scale |
| `group_comparison.png` | metrics by experimental group, wells shown individually |

Plus `activity_scan_index.json` at the top level, aggregating every run.

## Metrics

Per well: electrodes scanned and active, active fraction, array coverage,
mean/median/p90/max firing rate, spike amplitude stats, activity centroid and
dispersion, occupied area (mm²), and a clustering index (how often active
electrodes neighbour each other — near 1 means contiguous tissue, near 0 means
scattered isolated units).

Firing rate is computed per electrode against the time **that electrode** was
actually routed, which differs between electrodes in a multi-block scan.

Group labels (`MxWT`, `FxHET`, …) are read from `wellplate/well*/group_name`
inside the recording, so no external plate map is needed.

With `--selection-from`, it also reports how the recorded electrodes compare
with the full array — `selection_enrichment` above 1 means the selection favours
active electrodes, which is the intent, but also biases downstream firing rates
upward. Useful to state explicitly when reporting Network results.

## QC verdicts

Each well gets `pass`, `warn`, or `fail` with reasons, based on active electrode
count, mean rate, and active fraction. Thresholds are conservative defaults
(`--active-hz`, and the constants in `quality_flag`) and should be tuned once
you have scans from known-good and known-bad cultures.

The intended use is a **gate before expensive analysis**: the scan runs before
the Network recording, so a well that fails here is unlikely to repay hours of
Kilosort time.

## Caveats

* Thresholded spikes are not sorted units — a bursting neuron near several
  electrodes contributes to each. These are electrode-level activity measures,
  not neuron counts.
* Amplitude is the on-chip detection amplitude, useful for relative comparison
  rather than absolute waveform analysis.
* The clustering index assumes dense sampling; on a Network recording, where
  electrodes are deliberately spread out, it is near zero by construction and
  should be ignored.
