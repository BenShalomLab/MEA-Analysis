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

## What the timing analysis adds

The counting pass already loads `frameno` timestamps at the sampling rate, so
temporal analysis costs no extra I/O. On a 6 GB, 6-well recording the whole
thing takes about 8 seconds.

**Per electrode** — ISI CV (≈0 metronomic, ~1 Poisson, >1 bursty), burst
fraction (share of intervals under 100 ms), and rate stability (CV of rate
across 10 s windows, which catches cultures drifting or dying mid-recording).

**Per well** — population firing rate, network burst detection (count, rate,
duration, spikes per burst, % of spikes in bursts, interburst interval) and a
synchrony Fano factor. On real data this works well: one well showed 163 bursts
at 0.54 Hz with 56% of spikes inside bursts and Fano 82.8 — clear network
bursting, from unsorted threshold crossings.

**Functional connectivity** — pairwise correlation of binned spike trains among
the busiest electrodes, giving mean/median/p95 correlation and mean degree, plus
a spatial map of connectivity hubs.

> **Important constraint.** Electrodes in different scan blocks were never
> recorded at the same time, so anything relational — correlation, synchrony,
> network bursts — is computed **within a block only** and then aggregated.
> Cross-block correlations would be meaningless and are not attempted.

These are an early, sorting-free preview. The Network pipeline's burst detector
works on sorted units and remains the authority; this is for triage and for
covering the whole array.

## Longitudinal trends

`activity_trends.py` aggregates `summary.json` files across sessions:

```bash
python orchestration/activity_trends.py /data/scan_out
python orchestration/activity_trends.py /data/scan_out --metrics rate_mean_hz synchrony_fano
```

Writes `trends.csv`, `trends_by_group.png` (group means ± SD with wells shown
individually), `trends_by_well.png` (per-well trajectories), and
`group_stats.json` (Mann-Whitney U per timepoint).

DIV comes from plating date and session date when both are present, else from a
`DIV<n>` path component, else session ordering.

This is where scans beat Network recordings for longitudinal work: Network runs
use a different electrode selection each session, so their rates compare
different samples of the array. Scans measure on the same terms every time.

With few wells per group the statistics are reported as
`"too few wells per group to test"` rather than as a p-value — with n=2 there is
nothing meaningful to say, and a number would be misleading.

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
