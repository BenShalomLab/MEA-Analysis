# Implementation integrity audit vs. reference methods

Date 2026-09-14. Companion to `sota_review_and_plan.md`. Each row: what the code does now, what the reference implementation or paper does, verdict, fix. Verdict key: **OK** sound; **BUG** wrong; **DRIFT** differs from stated reference; **VERIFY** cannot confirm without running on data; **GAP** missing.

## 1. Preprocessing (`mea_preprocessing.py`)

| Step | Current | Reference | Verdict | Fix |
|---|---|---|---|---|
| Trim last 1 s | `end_frame = floor(total_frames)` then `frame_slice(0, end_frame)`; log says "removed last 1s" | Intent: drop Maxwell end-of-file artefact | **BUG** no-op | `end_frame = total_frames - int(fs)`; or delete |
| uint16 → signed | `spre.unsigned_to_signed` when dtype kind 'u' | SI Maxwell extractor returns uint16 with gain/offset; conversion correct | OK | — |
| Highpass | Butterworth 300 Hz, no lowpass | KS4 applies its own 300 Hz highpass again (`highpass_cutoff=300`, `skip_kilosort_preprocessing=False`); nicespike uses 300–6000 bandpass | OK, double filter harmless | Optionally set `skip_kilosort_preprocessing=True` to avoid double filter + double CAR |
| Bad channels | `detect_bad_channels(method='mad')`, std_mad_threshold 5 (SI default), 100 chunks × 0.3 s; refuse if >30 % flagged; fall back to 'std' | SI 'mad' = per-channel MAD vs median over channels; IBL 'coherence+psd' needs dense pitch. Reasoning in code comment correct for Maxwell sparse layouts | OK | Persist `n_bad / n_total` into JSON (A6) |
| Local CMR | `common_reference(reference='local', operator='median', local_radius=(250,250))` | SI: annulus `a < d ≤ b`. With a = b annulus is empty → every channel "no neighbours" → SI leaves channel **unreferenced**, warns, no exception. Bare `except` would only trigger on missing locations | **BUG** CMR is silently a no-op | `local_radius=(30, 200)`; assert kernel non-empty; log fraction of channels with < `min_local_neighbors` |
| KS4 CAR | `do_CAR=True` default inside KS4 → global median CAR applied after (no-op) local CMR | Global CAR on Maxwell sparse 1024-ch config is acceptable; local CMR preferred for spatially clustered noise | DRIFT from intent | Once local CMR fixed, set `do_CAR=False` |
| Save | float32 binary, chunk 1 s, n_jobs 16 | int16 would halve disk; KS4 reads float32 fine | OK | — |
| Sampling rate | `metadata['fs']` set here | — | OK | — |

## 2. Sorting (`mea_sorting.py`)

| Param | Current | KS4 default / reference | Verdict | Note |
|---|---|---|---|---|
| `invert_sign` | **True** | False. Docstring: "flip positive/negative values in data to conform to standard expected by Kilosort4" (KS4 expects negative spikes). Maxwell traces after `unsigned_to_signed` are standard extracellular, negative spikes | **VERIFY** likely wrong | Run one well with False vs True; compare n_units, mean template polarity at extremum channel, fraction rp-violations. If True inverts real negative spikes, KS4 sorts on positive rebounds and loses units. Analyzer templates (computed from original recording, `peak_sign='neg'`) would hide this |
| `batch_size` | fs × 2 (40 000 @ 20 kHz) or fs × 0.5 low-VRAM | 60 000 | OK | fine |
| `nblocks` | 0 | 1 | OK | no drift in cultures; correct |
| `do_correction` | False | True | OK | consistent with nblocks 0 |
| `dmin` | 17 (high-VRAM) / auto (low-VRAM) | auto = median contact distance | DRIFT between VRAM paths | Maxwell pitch 17.5 µm; median NN in sparse configs 55–65 µm → auto gives ~60. Two GPUs give different template grids → different unit counts for same well. Pin one value for both paths |
| `dminx` | default 32 | 32 | OK | — |
| `cluster_downsampling` | 20 / 30 | 20 | DRIFT between paths | pin |
| `max_cluster_subset` | None / 50 000 | 25 000 | DRIFT between paths | pin |
| `Th_universal`, `Th_learned` | default 9 / 8 | 9 / 8 | OK | may be high for low-SNR iPSC; sweep 7/6 once |
| `nearest_chans` | default 10 | 10 | VERIFY | sparse Maxwell layout: 10 nearest can span >150 µm; consider 6 |
| post | `remove_excess_spikes`, `remove_empty_units` | standard | OK | — |
| VRAM branching | params depend on GPU memory ≥ 14 GB | none | **BUG** for reproducibility | provenance stamp (A7) + single param set |
| Spike-detection fallback | `detect_peaks(by_channel, thr 5, neg, exclude_sweep 0.1 ms)`; drop channels > 100 Hz | nicespike/CASCADE: 5–6 × MAD, de-duplication across neighbours | OK for QC only | never use for phenotyping; label outputs `mode=threshold` |

## 3. Merge (`mea_merge.py`, `UnitMatch/`)

| Item | Current | Reference | Verdict | Fix |
|---|---|---|---|---|
| auto_merge | SI `auto_merge_units` preset `x_contaminations`, template_diff 0.05/0.15/0.25 recursive | SI recommended preset; thresholds loose at 0.25 | OK, off by default | keep off unless validated |
| UnitMatch | "DeepUnitMatch clone" imported from hardcoded `/home/adamm/dev/pkgs/UnitMatch/UnitMatchPy` (`runner.py:20`, adapter) | UnitMatch (van Beest 2024) is waveform-based *cross-session* tracker; intra-session oversplit merging is off-label | DRIFT + portability bug | Make clone path config; reserve UnitMatch for cross-DIV tracking (F1) |
| Merge stage order | merge before analyzer, so analyzer/QC computed on merged sorting | correct | OK | — |

## 4. Analyzer (`mea_analyzer.py`)

| Item | Current | Reference | Verdict | Fix |
|---|---|---|---|---|
| Sparsity | radius 50 µm, `peak_sign='neg'` | Maxwell footprints 50–150 µm; median NN 55–65 µm → 1–3 channels per unit | **DRIFT** | radius 100 µm; or `method='snr'`. Affects template_metrics, unit_locations, UnitMatch |
| unit_locations | `monopolar_triangulation` | needs ≥ 4–6 channels for stable fit | VERIFY | after sparsity fix; else `center_of_mass` |
| waveforms | 1 ms before / 2 ms after | DeePhys 1/2; fine | OK | — |
| quality_metrics | SI defaults: num_spikes, firing_rate, presence_ratio, snr, isi_violation, rp_violation, sliding_rp_violation, amplitude_cutoff, amplitude_median, amplitude_cv, synchrony, firing_range, drift, sd_ratio, noise_cutoff | Curation uses 4 of these | OK | export all (already in xlsx) |
| template_metrics | computed, written to xlsx, never used | DeePhys 8 waveform features | GAP | E1 |
| Extensions absent | no `correlograms`, `isi_histograms`, `spike_locations` | DeePhys uses CCH graph | GAP | add `correlograms` (needed for D-series alt) |

## 5. Curation (`mea_reports._apply_curation_logic`)

| Rule | Current | SI semantics | Verdict | Fix |
|---|---|---|---|---|
| presence_ratio < 0.75 | bin 60 s (SI default) | Sparse/bursting cultures: unit silent for 15 min of 60 → reject. Mossink QC is well-level, not unit-level | DRIFT, genotype-biased | 0.5, bin 120 s; log per-genotype rejection counts (A6) |
| rp_contamination > 0.15 | refractory 1 ms, censored 0 | Llobet/SI standard; 1 ms fine | OK | also export `isi_violations_ratio` (1.5 ms) |
| firing_rate < 0.05 Hz | rate over recording | fine as noise floor | OK | — |
| firing_rate > 100 Hz | — | ok | OK | — |
| amplitude_median > −20 | assumes **signed** negative median; SI 0.103 docstring says "in absolute value", code takes plain median of signed `spike_amplitudes` (`peak_sign='neg'`) → negative in practice | version-fragile | **VERIFY** | use `abs(amplitude_median) < 20` |
| amplitude_cv_median 0.5 | in config, never applied (TODO) | SI metric name `amplitude_cv_median` | **BUG** silent | apply or remove from config |
| iteration | `iterrows` + `row.get(..., default)` → NaN metric passes | ok | OK | — |

## 6. Network burst detector A: `parameter_free_burst_detector.py`

| Step | Current | Reference | Verdict | Fix |
|---|---|---|---|---|
| Duration base | `total_dur = last_spike − first_spike` | recording duration | **BUG** inflates rates in sparse wells | A3 |
| Per-unit stats | CV, CV2 (Holt 1996), Lv (Shinomoto 2009), BC on log-ISI (Sarle: > 0.555) | formulas correct; BC uses sample-size correction | OK | — |
| is_bursty | BC > 0.555 and (Lv > 1 or NaN) | reasonable heuristic; not literature-standard | VERIFY | compare against logISI/MI burst-fraction (C1) |
| reference ISI | first peak of smoothed log-ISI histogram of bursty units, else 15th pct | Pasquale 2010 uses intra-burst peak ≤ 100 ms | OK | — |
| bin size | clip(reference ISI, 20, 100 ms) | Chiappalone 2005 25 ms; MaxLab 10–20 ms; organoids need larger | OK for 2D; DRIFT for organoids | prep preset (B3) |
| Signals | participation = active units / n_units per bin; rate normalised per unit | Wagenaar/Chiappalone use pooled rate; participation used by Bakkum 2013 (fraction of electrodes) | OK, well justified | — |
| Smoothing | σ 1–2 bins participation, 3–8 bins rate | ok | OK | — |
| Threshold (bimodal) | median + 0.75·MAD, floor 0.03 | literature mean + 2–3 SD; 0.75 MAD ≈ 0.5 SD | DRIFT, permissive | acceptable because prominence (2·MAD) and synchrony floor (≥ max(3, 10 %) units) gate; document as "fragment" threshold |
| Threshold (unimodal) | percentile 95–99.5 scaled by n_units | ad hoc | VERIFY on synthetic sparse case (I2) | — |
| Fragment extent | walk while signal ≥ max(threshold, 0.30·peak) | MaxLab/Wagenaar use % of peak | OK | — |
| burst_area | Σ per-unit-normalised rate × bin (Hz·s/unit) | `peak_population_firing_rate_hz` is pooled Hz | **DRIFT** mixed units in one event dict | rename `burst_area_per_unit`; add pooled |
| participation per event | O(units × spikes) mask per event, called twice (fragment + finalize) | fine < 500 units; slow for 2000 | perf | `np.searchsorted` |
| fragment → NB merge | gap ≤ p95 intra-burst ISI of bursty units (fallback 3·ISI) and valley ≥ threshold | "same supra-threshold region"; sound | OK | — |
| NB → superburst | gap ≤ antimode of inter-fragment intervals (floor max(0.75, 0.3)); `min_components=1`; min dur 2.5 s | Wagenaar 2006 superburst = cluster of bursts; single long NB ≠ superburst | **DRIFT** | A4 |
| IBI | start-to-start | Axion/Mossink IBI = end-to-start | DRIFT in naming | export both `ibi_onset_s`, `ibi_gap_s` |
| Metrics | mean/std/CV per level; empty → 0 not NaN | 0 vs NaN: 0 biases group means for silent wells | **DRIFT** | NaN + separate `burst_count=0` |
| Missing | rise/decay time, % spikes in NB, NB spike rate, CV of NB duration | Mossink set | GAP | C5 |
| Diagnostics | thorough | good | OK | — |

## 7. Network burst detector B: `gaussianNetworkBursts.py`

| Step | Current | Reference | Verdict | Fix |
|---|---|---|---|---|
| Rate signal | pooled counts / bin / n_units, Gaussian σ 100 ms, bin 10 ms | Chiappalone 2005 / MaxLab computeNetworkAct | OK | — |
| Peak gate | prominence = SD of whole smoothed signal, distance 1 s, **no height gate** by design (MATLAB parity) | Chiappalone: threshold = mean + N·SD or RMS × factor; MaxLab GUI default Threshold 1.2 × RMS | **BUG** by fidelity to buggy MATLAB; silent wells produce noise peaks; SD of bursty signal is dominated by bursts so prominence varies with burst rate | A5 |
| Edge rule | `edge_level = peak × (1 − 0.3) = 0.7 × peak` | MaxLab `thresholdStartStop = 0.3` means edges where rate drops **below 30 % of peak** (Wagenaar 2006 style). Current rule keeps only crest above 70 % → NB durations ~3× too short | **BUG** likely inverted | `edge_level = peak × onset_offset_peak_frac`; confirm against a MATLAB-analysed well |
| Duration base | first→last spike | recording | BUG | A3 |
| Participation | computed for plotting only | ok | OK | — |
| Single tier | fragments/superbursts empty | ok, documented | OK | — |
| MinPeakDistance 1 s | merges reverberations < 1 s | fine for MATLAB parity, hides organoid/hippocampal reverberation | note | preset |

## 8. Shared metrics (`burst_common.py`) and collector

| Item | Current | Verdict | Fix |
|---|---|---|---|
| `stats([])` → zeros | biases means | DRIFT | NaN |
| `level_metrics` IBI = diff(starts) | see above | DRIFT | both |
| `collect_network_jsons._parse_path_metadata` | expects `…/<chip>/Network/<run>/<well>`; actual output tree is `<project>/<date>/<chip>/<run>/<well>` (`relative_pattern` = parts[-6:-1] of input path) | **BUG** project/date/chip columns shifted | read ids from JSON (`project`, `date`, `chip_id`, `run_id`, `well` already written) not path |
| unit_stats.csv | not collected | GAP | C4 |
| `_EVT_DISTRIBUTION_FIELDS` percentiles | fine | OK | — |

## 9. Reports / plots (`mea_reports.py`, `helper_functions.py`)

| Item | Current | Verdict | Fix |
|---|---|---|---|
| Curation before burst analysis | yes | OK | — |
| `spike_times.npy` saved after curation | yes; used by `--reanalyze-bursts` and raw template extraction | OK | — |
| Waveform PDF `best_ch = argmin` over sparse channel index; title shows index | misleading id | minor | map to channel id |
| `plot_probe_locations` | ok | OK | — |
| Raster sort `location_y` advertised in CLI, not implemented (`_sort_units_for_raster` handles none/firing_rate/unit_id) | BUG | implement with `unit_locations` |
| y-max summary JSON written by every subprocess (race across parallel SLURM jobs) | BUG under parallel batch | write per-well file, aggregate later |
| `fixed_y` replot omits burst hierarchy and uses `plot_clean_raster` without `sorted_units` | inconsistent | minor |
| `helper.detect_bursts_statistics` fixed-ISI bursts + `np.cov` on 1-D (returns variance, labelled cov); `plot_network_activity` legacy | dead, wrong | delete (C3) |
| `mark_burst_hierarchy` hides superbursts < 2.5 s though detector already filters | redundant | — |

## 10. Raw template extraction (`mea_waveform.py`)

| Item | Current | Verdict |
|---|---|---|
| Reads raw file, recenters on trough ±0.5 ms, 200 spikes/unit, streaming block reads | sound; mirrors DeePhys raw template step | OK |
| Uses **raw** (unfiltered) traces | fine for morphology; note DC offset; highpass before averaging is standard | minor |
| Primary channel from dense template argmin | ok | OK |

## 11. Driver / infra

| Item | Current | Verdict | Fix |
|---|---|---|---|
| Metadata from path regex `/(\d+)/data.raw.h5` and parts[-6:-4]; `.metadata` overrides | fragile; no genotype/DIV | GAP | B1–B2 |
| Reference Excel filter (Run #, Assay) | ok | OK | — |
| Checkpoint schema migration | ok | OK | — |
| Provenance: no git SHA, detector version, KS params in outputs | GAP | A7 |
| Tests: 7 contract tests on detector schema; none on preprocessing, curation, collector | GAP | I1, I2 |

## 12. Priority list from this audit

1. CMR annulus no-op (1.5) and `invert_sign` (2.1): both can change every unit count already produced. Verify on one well before any new analysis.
2. Gaussian edge inversion (7.3) and no height gate (7.2): any MATLAB-parity comparison done so far is suspect.
3. Duration base (6.1), superburst definition (6.11), zeros-for-empty (8.1): bias group statistics.
4. Collector path parse (8.3): mislabels project/date/chip in every CSV.
5. VRAM-dependent KS4 params (2.10): reproducibility.
6. presence_ratio (5.1) and amplitude sign (5.5): curation bias / version fragility.
7. y-max race, raster_sort location_y, dead helpers: hygiene.
