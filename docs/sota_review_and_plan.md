# HD-MEA NDD phenotyping: SOTA survey and task plan

Date: 2026-09-14. Scope: what the field and the current tools do, where this pipeline stands, and a task list sized for single-agent sessions.

## 1. Tools landscape

| Tool | Lang | Input | Core strengths | Gaps for us |
|---|---|---|---|---|
| **MEA-NAP** (Sit et al., Cell Rep Methods 2024; SAND-Lab/MEA-NAP) | MATLAB (+Python loader) | electrode spike trains, MCS/Axion | Batch genotype × DIV design, STTC connectivity with probabilistic thresholding, graph metrics (BCT) with size normalisation, node cartography, NMF activity patterns, effective rank, control-theory metrics, auto stats + plots | Electrode-level not unit-level; no Maxwell reader; MATLAB |
| **DeePhys** (Hornauer et al., Stem Cell Reports 2024) | MATLAB | phy-format sorted units (HD-MEA) | >50 features: 8 waveform + 4 spike-train regularity per unit, burst, cross-correlogram graph features, 100 ms time-series; Louvain cell-type clustering; RF classification, feature importance, age regression; unit tracking across recordings | MATLAB; ground-truth cell type still open (their 2026 bioRxiv follows up with chemogenetic labels) |
| **spikeNburst / nicespike** (Wolff et al., STAR Protoc 2025) | Python | sorted units (3Brain) or raw | ISI-violation unit filter; per-unit burst: Chiappalone, Pasquale logISI, hybrid; NB with absolute min-unit count; STTC, SPIKE-distance, phase synchrony; soma vs dendrite unit split by template polarity | 3Brain-centric IO; no stats layer |
| **MEA-ToolBox** (Hu et al., Neuroinformatics 2022) | MATLAB | raw MCS | Literature-standard spike/burst/NB/connectivity, no manual steps | Low-density arrays |
| **meaRtools** (Gelfman et al., PLoS CB 2018) | R | Axion spike lists | Well-level stats over DIV, permutation tests, burst (MI/logISI), STTC, "entropy/mutual info" of network spikes | Axion-centric |
| **autoMEA** (Hernandes et al., Front Neurosci 2024) | Python | MCS | ML burst detector matched to human expert; reverberations | Low-density |
| **CASCADE** (Avila et al., Bioinformatics 2026) | Python/bash | 3Brain, Axion, Cortical Labs, **Maxwell**, MCS | Cross-vendor pipeline, avalanche/criticality metrics | Threshold spikes only |
| **MaxLab Live** (Maxwell) | closed | raw | Network Burst Shape Metrics + superburst, AxonTracking, activity scan | Closed; no unit-level stats export you control |
| **Axion AxIS / NAS** | closed | raw | De-facto industry metric list (wMFR, ISI CV, NB metrics, synchrony index 20 ms, FWHH), Neural Activity Score composite | Closed |
| **SpikeInterface + Kilosort4 + UnitMatch** | Python | raw | Sorting, QC metrics, template metrics; UnitMatch tracks units across days by waveform (van Beest 2024) | Nothing above the unit level |
| **Elephant** | Python | spike trains | STTC, CCH, SPIKE-distance, unitary events, avalanches | Library, not pipeline |

This repo already has the part most tools lack: Maxwell IO, Kilosort4 sorting, checkpointed batch runs, SLURM, and a defensible adaptive NB detector. It lacks everything MEA-NAP and DeePhys have *above* the burst layer.

## 2. What the literature says a phenotyping readout needs

**Standard feature set (Mossink et al. 2021, Stem Cell Reports; used by Nadif Kasri/Frega labs across Kleefstra, CACNA1A 2026, SHANK2 2024, KCNQ2 2025):** MFR, % random spikes, burst rate, burst duration, burst spike rate, IBI, NB rate, NB duration, NIBI, CV of NIBI, NB rise time, NB decay time, correlation C0, link weight. They found MEA batch explains 69 % of variance, astrocyte batch 32 %; MFR, CV NIBI, C0, link weight have CV > 50 % so never use one alone. Minimum design: 12 wells per condition over ≥ 2 batches, ≥ 3 lines, isogenic where possible, QC gate MFR > 0.1 Hz, BR > 0.4/min, NBR > 1/min.

**Per-unit burst detection (Cotterill et al., J Neurophysiol 2016):** MaxInterval first choice, logISI (Pasquale 2010) second; run two detectors and report agreement. Fixed-ISI threshold (what `helper.detect_bursts_statistics` does) is the method that paper argues against.

**NB detection:** Chiappalone 2005 / Wagenaar 2006 Gaussian-rate methods and adaptive methods (Välkki 2017) are all accepted. Reviewers care that the detector is (a) parameter-transparent, (b) validated on synthetic ground truth, (c) same across genotypes. Ours is fine. MaxWell's new "Network Burst Shape Metrics" (rise/decay/area/superburst) matches our three-tier design; add rise and decay time to match the Mossink set.

**Synchrony and connectivity:** STTC (Cutts & Eglen 2014) is the standard because it is rate-independent; MEA-NAP adds a shuffled-null probabilistic threshold. Ort et al. 2026 (bioRxiv) show graph metrics differ across methods as much as biology does, so report method, window (10 or 25 ms), threshold, and null model. Spike-contrast / SPIKE-distance are timescale-free complements (Ciba 2020).

**Cell type:** template trough-to-peak + half-width splits FS vs RS in cortical cultures; DeePhys clusters on 8 waveform features. Hornauer et al. 2026 (bioRxiv) is the first ground-truth study on HD-MEA (chemogenetic labelling, defined E/I mixes). E/I imbalance is the dominant hypothesis in SHANK, MECP2, SCN2A, SYNGAP1, FMR1 models, so per-class metrics matter.

**Developmental trajectory:** NDD phenotypes are usually shifted trajectories (SHANK2 deviates from isogenic curve; hPSC excitability falls then plateaus). Effective excitability α = f(mean IBI, CV IBI, mean burst duration) (Vinogradov et al. 2024) is a single scalar that tracks development and KCl perturbation and needs nothing beyond burst stats.

**Organoids (Trujillo et al. 2019; Sharf 2022; Martin-Burgos 2024):** LFP oscillations 1–100 Hz, nested theta/gamma, cross-frequency coupling, aperiodic 1/f slope and neuronal timescale from spectra. Needs a low-frequency branch we discard at the 300 Hz highpass.

**Criticality:** avalanche size/duration exponents, branching ratio (CASCADE, pyAvalanches). Used in NDD organoid papers; optional.

**Statistics:** well is the replicate, nested in chip and batch; linear mixed models with batch and line as random effects, BH across features, effect sizes. Mossink used variance partitioning to show batch dominates; MEA-NAP and meaRtools bake stats into the pipeline. Ours has none.

## 3. Our pipeline vs. SOTA

| Layer | Ours | SOTA | Gap |
|---|---|---|---|
| IO / sorting | Maxwell + KS4, checkpoints, SLURM | same (nicespike, CASCADE) | none |
| Unit QC | presence, rp_contam, FR, amp | + ISI violations, SNR, soma/dendrite split, rejection-vs-genotype audit | small |
| Per-unit burst | **done (C)** MaxInterval + logISI with agreement, exported per unit | MI + logISI, agreement | none |
| Network burst | 3-tier adaptive + Gaussian baseline | Chiappalone / Wagenaar / shape metrics | add rise/decay/area-norm, fix duration base |
| Synchrony / graph | **done (D)** STTC at 10/25 ms + surrogate threshold, degree, clustering, path length, modularity, hubs, small-world sigma, burst propagation | STTC + null, degree, clustering, path length, modularity, hubs, NMF | NMF activity patterns and effective rank still missing |
| Cell type / E-I | **done (E)** gated 2-component mixture on waveform shape, non-somatic excluded, per-class metrics | waveform clustering, per-class metrics | ground truth for the proxy (chemogenetic or defined mixtures) |
| Trajectory | **done (F)** waveform tracking across recordings, DIV slope/onset/plateau/AUC per well | DIV curves, unit tracking (UnitMatch across days), α | calibrated match probabilities |
| LFP / oscillations | discarded | organoid standard | large, optional |
| Stats / report | **done (G)** MixedLM with batch random effect, BH, Hedges' g, variance partition, PCA, RF importance, HTML | LMM, BH, PCA/UMAP, HTML | none |
| Validation | synthetic tests for NB | ground-truth + regression sets | small |

## 4. Task plan

Rules: one task = one agent session, one PR, ≤ ~300 lines, one runnable check. Every new metric is a pure function on `SpikeTimes` dict (+ optional locations/templates) returning a flat dict that `mea_reports.py` merges into `network_results.json` and `collect_network_jsons.py` flattens. Detector schema in `burst_common.py` stays frozen. Deps listed as T-numbers.

### Epic A — correctness fixes — DONE (2026-09-14)

| ID | Task | Outcome |
|---|---|---|
| A1 | Fix no-op trim | `TRIM_TAIL_S = 1.0` actually removed, with a guard for short recordings; duration recorded in metadata |
| A2 | Local CMR radius | `(30, 200)` µm plus `_local_reference_coverage()`, which falls back to global referencing when the layout is too sparse and records the mode in `bad_channels.json` |
| A3 | Recording-duration rates | `duration_s` threaded from the recording into both detectors; detection still uses the active span, `duration_source` reported |
| A4 | Superburst definition | `min_superburst_components=2`; long single bursts surface as `burst_duration_p95_s` / `burst_duration_max_s`; both IBI conventions reported |
| A5 | Gaussian height gate | `min_height_sd=2.0` default, 0 restores MATLAB parity; edge rule corrected to `frac * peak`; `detection_threshold_hz` now populated |
| A6 | Curation audit | per-rule rejection counts in `curation_summary.json` and the JSON/checkpoint; `amplitude_median` compared in absolute value; `amplitude_cv_median` applied |
| A7 | Provenance | git commit/branch/dirty plus package versions on every checkpoint and `network_results.json`; resolved sorter parameters in `sorting_params.json` |
| A8 | Exception hygiene | no bare `except:` left; silent `pass` handlers now log |

Checks: `tests/test_epic_a_fixes.py` (23 tests) plus the existing suite. Run with the `mea` conda env: `python -m pytest tests/ -q`.

Still open from the audit and deliberately not changed here: the Kilosort4 `invert_sign` question (needs a real-data A/B, now a one-flag change via `--kilosort-params`), the `presence_ratio` threshold (a scientific choice, now measurable through the rejection counts), and the collector path parsing (task C4).

### Epic B — metadata and design — B1, B2 DONE (2026-09-14)

| ID | Task | Outcome |
|---|---|---|
| B1 | `samples.csv` schema | `docs/samples_schema.md` plus `Configfiles/samples_example.csv`. CSV, TSV or XLSX; key columns project/date/chip/run/well with blank cells as wildcards; metadata columns line, genotype, prep_type, batch, plating_date, div, density, media, treatment; unknown columns carried through |
| B2 | Sample lookup wired through | `samples.py` resolves one row per well, most specific match wins. `io.samples_file` / `--samples-file` on driver and routine. The record lands in `network_results.json` under `sample`, in the checkpoint, and as `sample_*` columns in the collector. DIV derived from `plating_date` or taken from an explicit `div` column, with `div_source` recorded |
| B3 | `prep_type` presets for detector defaults | **Deferred**, as in the ordering below. `prep_type` is now carried per well, so this is a small follow-up, but the organoid numbers (bin ceiling, sigma, minimum burst duration) should be chosen against real organoid recordings rather than guessed |

Also fixed here, because it corrupts the same columns B2 adds: the collector read the run id one level above the well, which is the assay folder, so every row's `run` was the literal string `Network`. Ids recorded inside the JSON now take precedence over directory names, and the path fallback reads the right level. Curation counts, detector name and git commit also became columns.

Checks: `tests/test_samples.py` (24 tests), `tests/test_collect_network_jsons.py` (7 tests).

### Epic C — unit-level features — DONE (2026-09-14)

| ID | Task | Outcome |
|---|---|---|
| C1 | `unit_bursts.py` | MaxInterval and logISI (Pasquale 2010 antimode with a 100 ms fallback, reporting which was used), per-unit burst rate, duration, spikes per burst, intra-burst rate, fraction of spikes in bursts, IBI mean/CV/gap, and a Jaccard agreement between the two detectors as Cotterill et al. 2016 advise |
| C2 | Wired into reports | `unit_stats.csv` now carries the detector's ISI statistics and both detectors' burst features per unit; `network_results.json` gains a `unit_level` block with the across-unit summary and the parameters used. Non-fatal: a failure here logs and leaves the network results intact |
| C3 | Dead helpers | **Not deleted.** `detect_bursts_statistics` and `plot_network_activity` are called by three notebooks under `workbooks/`. Both are documented as deprecated in favour of `unit_bursts.py` and the pipeline detectors, and a real defect was fixed: their `cov_*` fields returned `np.cov` of a 1-D array, which is a variance in seconds squared, not a coefficient of variation |
| C4 | Collector | `ul_*` columns from the unit-level summary and `sp_*` from spike participation, ordered ahead of the burst tiers |
| C5 | NB shape | `rise_time_s` and `decay_time_s` per event, summarised per tier; `spike_participation` block with the fraction of spikes inside network bursts and its complement, percent random spikes |
| C6 | Effective excitability | `duty_cycle` per tier, the time fraction spent bursting. This is the parameter-free factor of the effective excitability in Vinogradov et al. 2024; their alpha is this times a model scale constant, which is deliberately not guessed |

Checks: `tests/test_unit_bursts.py` (24 tests), `tests/test_burst_shape.py` (13 tests).

### Epic D — synchrony and connectivity — DONE (2026-09-14)

| ID | Task | Outcome |
|---|---|---|
| D1 | `synchrony.py` STTC | Spike time tiling coefficient at 10 ms and 25 ms, with tiling fractions precomputed once per unit. Significance threshold from circularly shifted surrogates, pooled across sampled pairs rather than per pair, which is the approximation that keeps the cost affordable at a few hundred units; the pooling is documented in the output and in the config |
| D2 | Graph topology | Density, mean degree, degree CV, hub fraction, clustering (binary and Onnela weighted), characteristic path length, global efficiency, components, Louvain modularity, and clustering and path length normalised against random graphs of the same size and edge count, plus small-world sigma |
| D3 | `propagation.py` | Per-unit latency within each network burst measured from the first unit to fire, a leader score, and a propagation speed from regressing distance on latency with an R² gate. Recovers a synthetic 100 µm/ms wave to within 5% and refuses to fit randomly ordered firing |
| D4 | Wiring | `connectivity` and `propagation` blocks in `network_results.json`, `sttc_matrices.npz` per well, per-unit `prop_*` columns in `unit_stats.csv`, and `conn_<window>_*` / `prop_*` columns in the collector |
| D5 | Plots | `connectivity_summary.svg/.png`: STTC heatmap beside a map of burst leader scores on the array |

Verified against Elephant was not possible (not installed), so the STTC is checked against the Cutts & Eglen definition computed by hand plus its analytic limits: identical trains give 1, a shift inside the window still gives 1, a shift past it gives 0, and independent Poisson trains at 20 Hz and 2 Hz give 0, confirming rate independence.

Cost: quadratic in unit count. Roughly 10 s per well at 200 units and 45 s at 400, for both windows together, on ~2000 spikes per unit over 5 minutes. Graph metrics were moved from networkx to numpy matrix products for this reason, which made that stage about 6 times faster and stopped it dominating at 400 units. `connectivity.max_units`, `--connectivity-max-units` and `--no-connectivity` control the cost.

Checks: `tests/test_synchrony.py` (25 tests), `tests/test_propagation.py` (14 tests).

### Epic E — cell type — DONE (2026-09-14)

| ID | Task | Outcome |
|---|---|---|
| E1 | `celltype.py` | Two-component Gaussian mixture on trough-to-peak duration and half width, behind two gates: Sarle's bimodality coefficient above 0.555 and a BIC improvement of at least 10 over a one-component fit. Non-somatic templates (positive peak dominant, Wolff et al. 2025) are excluded before fitting. Handles both SpikeInterface spellings of the duration metric and both unit conventions |
| E2 | Per-class metrics | `cell_types` block in the JSON with counts, fast-spiking fraction, the regular-to-fast ratio, and per-class summaries of firing rate, burst rate, burst duration, fraction of spikes in bursts and leader score; `cell_class` and its probability per unit in `unit_stats.csv`; `ct_*` columns in the collector |
| E3 | Waveform PDF | Mean waveforms coloured by class, and the channel label fixed: it showed the per-unit sparse column index, so two units on the same electrode appeared to be on different ones |

Naming is deliberate. These are waveform classes, not validated cell types, so nothing is labelled excitatory or inhibitory. The narrow/broad split is a proxy whose correspondence to inhibitory identity is imperfect and has only recently begun to be checked against ground truth in culture (Hornauer et al. 2026, chemogenetic labelling and defined mixtures). The ratio most people will read as E/I is exported as `regular_to_fast_ratio`.

The classifier refuses to split when the distribution is not bimodal, which is common in young iPSC cultures. A mixture model always returns two clusters, so without that gate the pipeline would manufacture a cell-type difference between wells out of noise. `ct_classified` distinguishes "not bimodal" from "no fast-spiking units".

Checks: `tests/test_celltype.py` (20 tests).

### Epic F — trajectory — DONE (2026-09-14)

| ID | Task | Outcome |
|---|---|---|
| F1 | `track_units.py` | Matches units across recordings of one well on mean waveform shape, gated by distance on the array and assigned one-to-one by optimal matching. CLI takes well directories in recording order and writes `unit_tracking.csv` plus a summary. Recovered 16 of 16 persisting units across three synthetic recordings, correctly starting new tracks for units that appeared later |
| F2 | `trajectory.py` | Per well and feature: slope against DIV with its R², the age at which activity first appears, the age at which it stops rising, and area under the curve normalised by the span. `trajectory_feature_table` reshapes these into the well-by-feature form `stats_report.py` consumes, so a trajectory becomes just another per-well measurement to compare between genotypes |

**Deviation from the plan on F1.** It called for using UnitMatch. The repository's UnitMatch integration imports a clone from a hardcoded absolute path (`/home/adamm/dev/pkgs/UnitMatch/UnitMatchPy`) that exists on one machine, and it is wired for merging over-split units inside a single recording, which is a different problem. Rather than depend on an unavailable backend, `track_units.py` implements the same idea directly: match on waveform, because functional properties are what changes between recordings and so cannot establish identity. It is simpler than UnitMatch's probabilistic model, producing a similarity rather than a calibrated match probability, so the threshold is a visible choice rather than a posterior.

Validated on a synthetic delayed genotype: a seven-day shift in a sigmoid activity curve was recovered as 6.8 days at onset and 6.9 days at plateau, while the two genotypes overlap at individual timepoints.

Checks: `tests/test_trajectory.py` (22 tests).

### Epic G — statistics and report — DONE (2026-09-14)

| ID | Task | Outcome |
|---|---|---|
| G1 | `qc_wells.py` | Gates on unit count, firing rate, burst rate, network burst rate and bad-channel fraction, with the Mossink et al. 2021 thresholds. Exclusions are counted per group and an unbalanced exclusion rate is flagged, because it biases every comparison that follows. A missing metric skips its rule rather than failing the well, and the number of rules actually applied is recorded |
| G2 | `stats_report.py` models | Linear mixed model per feature with a random intercept for batch, the grouping variable fixed and DIV as a covariate; falls back to OLS with a single batch and says so. Benjamini-Hochberg across features, Hedges' g with 95% intervals, and variance partitioning as the intraclass correlation per nuisance factor |
| G3 | Multivariate | PCA of the standardised feature matrix coloured by group, and random forest permutation importance with cross-validated accuracy |
| G4 | HTML report | Single self-contained file with figures embedded as base64: QC exclusions, the corrected comparison table, per-feature strip plots showing every well, the variance partition and the multivariate panels |

Two design details worth knowing. Statsmodels warns constantly on small unbalanced designs; those warnings are captured and reported as a per-feature `converged` flag instead of burying the results. And a nuisance factor collinear with the grouping variable is skipped in the variance partition, since its variance share would be the genotype effect wearing a nuisance factor's name, which is easy to misread as the result being explained away.

Verified on a synthetic experiment of 48 wells across 3 batches with a real effect on two features: both were recovered as significant after correction with Hedges' g of 1.97 and 0.99, and the three null features were not.

Checks: `tests/test_stats_report.py` (26 tests). Adds `statsmodels` to requirements.

### Epic H — organoid LFP (optional)

| ID | Task | Files | Check | Deps |
|---|---|---|---|---|
| H1 | Preprocessing branch: 1–300 Hz bandpass, decimate to 1 kHz, save `lfp.npy` (config flag) | `mea_preprocessing.py` | file size | — |
| H2 | `lfp_features.py`: band power (delta–gamma), 1/f slope (fooof/specparam), theta–gamma PAC, NB-locked LFP | new | synthetic sines | H1 |

### Epic I — validation and drift protection

| ID | Task | Files | Check | Deps |
|---|---|---|---|---|
| I1 | Frozen regression set: 3 wells × prep type; expected metric ranges JSON; pytest that fails on drift | `tests/` | CI green | A-series |
| I2 | Synthetic ground-truth generator for NB (Välkki 2017 style) with precision/recall for both detectors | `tests/` | report | — |
| I3 | Metric dictionary `docs/metrics.md`: name, unit, formula, reference, detector dependence | docs | reviewed | C, D, E |

### Order

A1–A8 → B1–B2 → C1, D1, E1 in parallel → C2, C4, C5, C6 → D2–D5, E2–E3 → G1–G4 → F1–F2, B3 → I1–I3 → H optional.

## 5. Key references

- Sit et al. 2024, MEA-NAP, Cell Rep Methods 4:100901. github.com/SAND-Lab/MEA-NAP
- Hornauer et al. 2024, DeePhys, Stem Cell Reports. github.com/hornauerp/DeePhys
- Hornauer et al. 2026, Functional cell-type identification on HD-MEA, bioRxiv 2026.04.30.721923
- Wolff et al. 2025, spikeNburst / nicespike, STAR Protocols
- Mossink et al. 2021, Human neuronal networks on MEA are robust..., Stem Cell Reports (PMC8452490)
- Cotterill et al. 2016, Comparison of burst detection methods, J Neurophysiol (PMC4969396)
- Cutts & Eglen 2014, STTC, J Neurosci
- Ort et al. 2026, Graph theory for MEA recordings: framework and benchmarking, bioRxiv
- Vinogradov et al. 2024, Effective excitability, bioRxiv 2024.08.21.608974
- Trujillo et al. 2019, Cortical organoid oscillations, Cell Stem Cell
- Martin-Burgos et al. 2024, Neuronal timescales in organoids, J Neurophysiol
- van Beest et al. 2024, UnitMatch, Nat Methods
- Pachitariu et al. 2024, Kilosort4, Nat Methods
- Hernandes et al. 2024, autoMEA, Front Neurosci
- Avila et al. 2026, CASCADE, Bioinformatics
- Hu et al. 2022, MEA-ToolBox, Neuroinformatics
- Gelfman et al. 2018, meaRtools, PLoS Comput Biol
