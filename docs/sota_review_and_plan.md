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
| Per-unit burst | computed, not exported; fixed-ISI helper unused | MI + logISI, agreement | medium |
| Network burst | 3-tier adaptive + Gaussian baseline | Chiappalone / Wagenaar / shape metrics | add rise/decay/area-norm, fix duration base |
| Synchrony / graph | none | STTC + null, degree, clustering, path length, modularity, hubs, NMF | large |
| Cell type / E-I | template metrics written, unused | waveform clustering, per-class metrics | medium |
| Trajectory | none | DIV curves, unit tracking (UnitMatch across days), α | medium |
| LFP / oscillations | discarded | organoid standard | large, optional |
| Stats / report | ad-hoc notebooks | LMM, BH, PCA/UMAP, HTML | large, mandatory |
| Validation | synthetic tests for NB | ground-truth + regression sets | small |

## 4. Task plan

Rules: one task = one agent session, one PR, ≤ ~300 lines, one runnable check. Every new metric is a pure function on `SpikeTimes` dict (+ optional locations/templates) returning a flat dict that `mea_reports.py` merges into `network_results.json` and `collect_network_jsons.py` flattens. Detector schema in `burst_common.py` stays frozen. Deps listed as T-numbers.

### Epic A — correctness fixes (do first)

| ID | Task | Files | Check |
|---|---|---|---|
| A1 | Fix no-op trim: `end_frame = total_frames - int(fs)` or delete block | `mea_preprocessing.py` | log line matches frames removed |
| A2 | Local CMR radius: `(30, 200)`; remove bare `except`, log fallback reason | `mea_preprocessing.py` | unit test on synthetic recording with locations |
| A3 | Pass `duration_s` from recording into both detectors; use it for rates instead of first-to-last spike | `parameter_free_burst_detector.py`, `gaussianNetworkBursts.py`, `mea_reports.py` | test: sparse train, rate = n/duration |
| A4 | Superburst `min_components=2` default; keep single long NBs as `nb_duration_p95` | `parameter_free_burst_detector.py`, `config_loader.py` | existing tests + one new |
| A5 | Gaussian detector: add optional `min_height_sd` gate (mean + N·SD); default on, document MATLAB-parity flag | `gaussianNetworkBursts.py`, `config_loader.py` | silent-well synthetic gives 0 NBs |
| A6 | Curation audit: write `n_rejected_by_reason` into JSON; apply `amplitude_cv_median`; add `isi_violations_ratio`, `snr` to log | `mea_reports.py` | keys present in JSON |
| A7 | Stamp git SHA, detector name, KS4 param set into JSON and checkpoint | `mea_infra.py`, `mea_reports.py` | keys present |
| A8 | Replace remaining bare `except:` with logged exceptions | all mixins | grep returns 0 |

### Epic B — metadata and design

| ID | Task | Files | Check |
|---|---|---|---|
| B1 | `samples.csv` schema: chip, well, project, line, genotype, prep_type (mouse/ipsc/organoid), batch, plating_date, density, media, treatment | `docs/samples_schema.md`, example file | schema doc + example |
| B2 | Driver loads `samples.csv` (config `io.samples_file`), merges into JSON `sample` block; path parse stays fallback; compute DIV from plating_date + recording date | `run_pipeline_driver.py`, `config_loader.py`, `mea_reports.py` | JSON has `sample.genotype`, `sample.div` |
| B3 | `prep_type` presets for detector defaults (organoid: bin ceiling 250 ms, sigma up, min NB dur) | `config_loader.py`, detectors | preset name in diagnostics |

### Epic C — unit-level features

| ID | Task | Files | Check | Deps |
|---|---|---|---|---|
| C1 | `unit_bursts.py`: MaxInterval + logISI per-unit burst detectors (Cotterill 2016 params); returns per-unit burst rate, duration, spikes/burst, intra-burst FR, % spikes in bursts, IBI CV, method agreement | new | synthetic bursty vs Poisson train | — |
| C2 | Wire C1 into reports: `unit_stats.csv` gains columns; JSON `unit_level` block with mean/median/CV across units | `mea_reports.py` | JSON keys | C1 |
| C3 | Delete dead `helper.detect_bursts_statistics`, `plot_network_activity` | `helper_functions.py` | grep 0 | C2 |
| C4 | Extend collector with `unit_level` and `sample` sections | `collect_network_jsons.py` | CSV columns | B2, C2 |
| C5 | NB shape: rise time, decay time, area normalised by n_units, % spikes in NBs, PRS | `burst_common.py` (additive), detectors | tests | A3 |
| C6 | Effective excitability α from NB stats (Vinogradov 2024) | `burst_common.py` | value on synthetic | A3 |

### Epic D — synchrony and connectivity

| ID | Task | Files | Check | Deps |
|---|---|---|---|---|
| D1 | `synchrony.py`: STTC matrix (dt 10 and 25 ms), numpy, O(n²) with early exit; mean, median, fraction significant vs 100 spike-time-shuffled nulls | new | matches Elephant on 5 pairs | — |
| D2 | Graph metrics on thresholded STTC via `networkx`: density, mean degree, clustering, path length, modularity (Louvain), small-world σ, hub fraction; size-normalised vs random graphs | `synchrony.py` | test on ring vs random | D1 |
| D3 | NB propagation: per-unit onset latency inside each NB, leader/follower score, propagation speed from `unit_locations` | `propagation.py` | synthetic wave | A3 |
| D4 | Wire D1–D3 into reports + collector; save STTC matrix `.npz` | `mea_reports.py`, `collect_network_jsons.py` | JSON keys | D1–D3 |
| D5 | Plots: STTC heatmap, graph layout on probe map, latency map | `helper_functions.py` | files written | D4 |

### Epic E — cell type

| ID | Task | Files | Check | Deps |
|---|---|---|---|---|
| E1 | `celltype.py`: read `template_metrics` (peak_to_valley, half_width, repolarization_slope, recovery_slope); soma/dendrite split by polarity (Wolff 2025); 2-component GMM on log(ptv), half_width; labels FS / RS / unclassified with bimodality check | new | synthetic bimodal | — |
| E2 | Per-class C1/C5 metrics and E/I ratio in JSON; `unit_stats.csv` gains `cell_class` | `mea_reports.py` | keys | E1, C2 |
| E3 | Waveform PDF: colour by class, fix channel-id title | `mea_reports.py` | visual | E1 |

### Epic F — trajectory

| ID | Task | Files | Check | Deps |
|---|---|---|---|---|
| F1 | UnitMatch across recordings of same chip/well (not intra-recording oversplits): CLI `track_units.py` producing `unit_tracking.csv` | `UnitMatch/`, new script | 2 recordings synthetic | B2 |
| F2 | Trajectory features per chip/well: DIV of first NB, slope of NB rate, α vs DIV, plateau DIV | `trajectory.py` | csv | C4, C6 |

### Epic G — statistics and report

| ID | Task | Files | Check | Deps |
|---|---|---|---|---|
| G1 | `qc_wells.py`: exclusion flags (n_units < 10, active frac, MFR, NBR, bad-channel frac) with counts per genotype | new | flags on collector CSV | C4 |
| G2 | `stats_report.py`: statsmodels MixedLM per feature, genotype fixed, batch/line random, DIV covariate; BH; Hedges g with CI; variance partitioning (batch vs line vs well) | new | runs on synthetic CSV | G1 |
| G3 | PCA/UMAP on z-scored feature vector, coloured by genotype and DIV; RF classifier with permutation importance (DeePhys style) | `stats_report.py` | figures | G2 |
| G4 | HTML report per project (jinja2 + matplotlib): feature table, box/strip per DIV, LMM table, PCA, QC exclusions | `stats_report.py`, template | HTML opens | G2, G3 |

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
