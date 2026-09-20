# MEA pipeline: review, verdict, and what was rebuilt

Branch `pipeline-fable5-review`, September 2026. Companion documents: `sota_review_and_plan.md` (what the field does and the task plan), `integrity_audit.md` (per-module comparison against reference implementations), `blocks.md` (the independent-block architecture), `samples_schema.md` (experimental metadata format).

**Read this first if you read nothing else.** Every change described here is validated by unit tests and synthetic data with known ground truth. **None of it has been run on a real recording.** Two open questions in section 6 can only be settled with your data, and one of them affects every unit the sorter has ever produced.

---

## 1. Verdict

### The short version

The engineering was sound and the science stopped too early.

Checkpointing, the mixin decomposition, config precedence, Docker, SLURM batching and the Maxwell IO were all in good shape. What the pipeline produced, though, was a network burst summary per well and nothing else. That is one feature family out of the five a neurodevelopmental phenotyping study needs, and it is the one that answers the least interesting question. A network can keep its burst rate while its neurons burst differently, while it loses connections, while its excitatory and inhibitory populations diverge, or while its whole developmental trajectory shifts by two weeks. None of those were measurable.

Underneath that, six defects were silently corrupting the numbers that were produced. Four of them biased results in a direction that matters: they made quiet wells look more active than they were, which is precisely the comparison most knockout studies depend on.

### What was already good

- **Spike sorting rather than threshold crossings.** Most published MEA work on disease models still uses per-electrode threshold spikes. Sorted units give real firing rates, real interspike intervals and waveform features. This was the right call and it is what makes everything else in this document possible.
- **The parameter-free burst detector.** Using participation fraction as the detection signal is well judged. Per-unit ISI calibration of bin size, MAD-based thresholding, the bimodality-gated method switch and the three-tier fragment / network burst / superburst hierarchy are all defensible, and the diagnostics are recorded per well.
- **The bad-channel reasoning.** The comment explaining why `mad` is used rather than `coherence+psd` on sparse Maxwell layouts is correct, non-obvious, and the kind of thing that normally lives only in someone's head.
- **Curation with an auditable rejection log.** Thresholds in config, rejections written out.
- **Per-unit ISI statistics.** CV, CV2, Lv and the bimodality coefficient were already computed inside the detector. They were simply never exported.

### What was wrong

Six defects, in rough order of how much damage they did.

| Defect | Consequence |
|---|---|
| Local common reference used `local_radius=(250, 250)` | An empty annulus. SpikeInterface leaves such channels unreferenced and only warns, so the referencing step did nothing at all. |
| Kilosort4 run with `invert_sign=True` | Kilosort expects negative-going spikes and Maxwell traces already are. If this is wrong the sorter has been clustering on positive rebounds. **Still unresolved**, see section 6. |
| Both burst detectors divided rates by the span between the first and last spike | A well silent for half the recording was credited with double its true rate. Directly biases hypoactive genotypes towards looking normal. |
| Gaussian detector placed burst edges at 70% of peak | Should be 30%. Burst durations were roughly three times too short. |
| Gaussian detector applied no height gate | Reproduced the MATLAB code as shipped, where the intended threshold is commented out. Silent wells still produced "network bursts". |
| Empty metrics serialised as `0.0` | A well with no bursts entered group means as a well with very short bursts. |

Plus smaller ones: the "remove last 1 second" trim was a no-op, `amplitude_cv_median` sat in the config unused, the amplitude rule broke on newer SpikeInterface amplitude conventions, the collector's `run` column contained the literal string `Network` for every row, and a helper used by three notebooks returned a variance while labelling it a coefficient of variation.

### What was missing

Measured against MEA-NAP (Sit et al. 2024) and DeePhys (Hornauer et al. 2024), the two reference tools in this space:

| Layer | Before |
|---|---|
| Per-unit burst metrics | Computed internally, never exported |
| Functional connectivity and network topology | Absent |
| Cell-type separation and excitation/inhibition balance | Template metrics computed, written to a spreadsheet, ignored |
| Developmental trajectories | Absent |
| Experimental metadata (genotype, line, batch, DIV) | Absent, so no comparison was possible at all |
| Group statistics | Absent; analysis happened in ad-hoc notebooks |
| Organoid LFP and oscillations | Discarded at the 300 Hz highpass |

The metadata gap was the binding constraint. Without genotype and batch attached to each well, the pipeline could say "here are the metrics for well000 of chip 16657" and nothing more.

---

## 2. What was rebuilt

Seven work packages, each its own commit with its own tests.

### A. Correctness fixes — `8f4fdd7`

All six defects above, plus the smaller ones. The trim now removes a real second. The common reference uses 30 to 200 µm with a coverage check that falls back to global referencing when the electrode layout is too sparse, and records which mode ran. Both detectors take the recording duration and use it as the rate denominator, while detection still runs on the active span so which bursts are found does not change. Superbursts now require at least two component bursts, following Wagenaar et al. 2006; previously any burst over 2.5 seconds was also counted as one. Empty metrics serialise as null beside a real `burst_count`.

The Gaussian detector's edge rule and height gate were both corrected. **Burst durations from that detector will be longer than in any earlier run.** Setting `gaussian_min_height_sd` to 0 restores the old MATLAB-parity behaviour if you need to reproduce a previous analysis.

Provenance now goes into every checkpoint and result file: git commit, branch, dirty flag, package versions, and the Kilosort parameters that actually ran.

### B. Experimental metadata — `66a61ab`

A `samples.csv` carries genotype, line, prep type, batch and plating date into every result. Key columns are project, date, chip, run and well; a blank key cell is a wildcard and the row pinning the most keys wins, so a chip-wide default and a per-well exception coexist. DIV is derived from the plating date or taken from an explicit column, with the source recorded.

Failure is never fatal. A missing table logs a warning and leaves `sample.matched` false, so "wild type" stays distinguishable from "nobody recorded what this well was".

Also fixed here: the collector read the run id one level too shallow, so every row's `run` column held the literal string `Network`.

### C. Per-unit features — `56fbdae`

Single-unit burst detection with both methods Cotterill et al. 2016 recommend, reporting their agreement rather than presenting one as truth: MaxInterval with fixed thresholds, and logISI which derives each unit's threshold from the antimode of its own log-ISI distribution and falls back to a fixed cutoff when the distribution is not clearly bimodal. Which path each unit took is recorded, because a dataset where most units fall back is one where logISI is just MaxInterval with one threshold.

Per unit: burst rate, duration, spikes per burst, intra-burst rate, fraction of spikes in bursts, interval mean, CV and gap, plus a Jaccard agreement between the detectors.

Network bursts gained rise and decay times, a spike participation block giving the share of spiking inside bursts and its complement, and a duty cycle. The duty cycle is the parameter-free factor of the effective excitability in Vinogradov et al. 2024; their scale constant is deliberately not guessed.

### D. Connectivity and propagation — `34f3fc7`

Spike time tiling coefficient at 10 ms and 25 ms. That measure rather than a correlation index because it is not confounded by firing rate, which matters directly: a hypoactive genotype would otherwise look less connected for the wrong reason. Both windows are kept separate, since collapsing them hides a timescale effect inside a connectivity number.

Graph topology on the thresholded matrix: density, degree CV, hub fraction, clustering, path length, global efficiency, Louvain modularity, small-world sigma, and clustering and path length normalised against random graphs of the same size and edge count. That normalisation matters because both depend strongly on network size, so comparing raw values between wells with different unit counts compares the unit counts.

Burst propagation measures each unit's latency within a burst from the first unit to fire, giving a leader score and a wave speed from regressing distance on latency.

**One honest limitation.** The significance threshold is pooled across sampled pairs rather than computed per pair. A per-pair null is affordable for a 60-electrode array and is not for a few hundred sorted units. Treat `fraction_significant_pairs` as comparative between wells analysed identically, not as an absolute count of real connections.

### E. Cell type — `3092251`

A two-component Gaussian mixture on trough-to-peak duration and half width, behind two gates that must both pass: Sarle's bimodality coefficient above 0.555, and a BIC improvement of at least 10 over a one-component fit. Non-somatic templates, where the positive peak dominates, are excluded before fitting.

The load-bearing behaviour is the refusal to split. A mixture model always returns two clusters, and young iPSC cultures frequently have no narrow-spiking population. When the gates fail, every unit comes back unclassified with the reason recorded.

**Naming is deliberate.** These are waveform classes, not validated cell types. The link between narrow spikes and inhibitory identity is a proxy that has only recently begun to be tested against ground truth in culture. Nothing is labelled excitatory or inhibitory, and the ratio you would read as E/I is exported as `regular_to_fast_ratio`.

### G. Statistics and report — `3fff20e`

Quality gates on unit count, firing rate, burst rate, network burst rate and bad-channel fraction, using the Mossink et al. 2021 thresholds. The accounting matters more than the gating: exclusions are counted per group and an unbalanced rate is flagged, because if a knockout loses twice as many wells as its control, the survivors are its healthiest and every comparison is biased towards no effect.

One linear mixed model per feature, with batch as a random intercept because wells on a chip share a dissection, plating and feeding history. With a single batch the random effect is not identifiable, so the model falls back to least squares and says so. Benjamini-Hochberg across features, Hedges' g with intervals, variance partitioning per nuisance factor, PCA, random forest permutation importance, and a single self-contained HTML report.

### F. Trajectories — `39b021e`

Unit tracking across recordings of one well by waveform shape, since functional properties are exactly what changes and so cannot establish identity. Distance gate, cosine similarity with lag tolerance, one-to-one optimal assignment. A unit missing from an intermediate recording starts a new track rather than being bridged, because a gap is where a false match is most likely and a false match fabricates a trajectory for a neuron that does not exist.

Per well and feature: slope against DIV, interpolated onset and plateau ages, and normalised area under the curve.

**Deviation from plan.** The plan called for UnitMatch. Your UnitMatch integration imports a clone from a hardcoded path that exists on one machine, and targets merging over-split units inside a single recording, a different problem. The same idea is implemented directly instead. It gives a similarity, not a calibrated match probability, so the threshold is a visible choice.

---

## 3. Evidence

Everything below is synthetic data with known ground truth, or a hand computation.

| Check | Expected | Got |
|---|---|---|
| STTC of identical trains | 1.0 | 1.0 |
| STTC, shift past the window | 0 | −0.01 |
| STTC, independent Poisson at 20 Hz vs 2 Hz | 0 | 0.005 |
| STTC against the Cutts & Eglen definition by hand | exact | exact |
| Propagation speed of a 100 µm/ms wave | 100 | 99.8, R² 0.996 |
| Leader score, first and last unit of a wave | 1.0 and 0.0 | 1.0 and 0.0 |
| Cell-type split of 30 fast and 70 regular | 30 / 70 | 30 / 70 |
| Cell-type split of a unimodal population | refuse | refused, coefficient 0.399 |
| Per-unit burst metrics on an exact train | 5 bursts, 140 ms, 8 spikes | matched |
| Gaussian detector on Poisson input, gate on vs off | fewer | 10 vs 28 |
| Developmental delay of 7 days | 7 | 6.8 at onset, 6.9 at plateau |
| Group comparison, 2 real effects among 5 features | 2 significant | 2, at g = 1.97 and 0.99 |

232 tests across 11 files. Run them with the `mea` conda environment; the base Anaconda install has numpy 2 alongside a matplotlib compiled against numpy 1, which cannot both load.

```bash
/Users/mandarmp/anaconda3/envs/mea/bin/python -m pytest tests/ -q
```

---

## 4. The architecture question

Your requirement to make modules independent for later conversion to Nextflow blocks was explored on a separate branch, `blocks-refactor`, in its own worktree at `../blocks-refactor`. It is **not** on this branch.

### What that branch contains

Thirteen analysis modules moved to `mea/blocks/`, each gaining a command-line entry point so the same code runs three ways identically: standalone, as a library, and in-process. A shared file contract handles where spike times live, how duration is resolved, and atomic writes so a killed job never leaves a truncated file. Blocks depend on each other only through files; propagation consumes the bursts the burst block wrote and refuses to run before they exist.

Verified by running four blocks standalone on a synthetic well and watching them chain through files. 30 additional tests pin the contract itself: every block exposes an entry point, none imports the pipeline, each runs from a directory containing only its input, and outputs can be staged elsewhere the way Nextflow does.

### The honest assessment

**Folder layout is not what blocks Nextflow.** The obstacle is `MEAPipeline`, one class assembled from eight mixins that share `self.recording`, `self.sorting`, `self.analyzer` and `self.state` in memory. Mixins cannot become independent processes while they pass live objects between themselves, and moving them into folders would leave that coupling exactly as it is, just harder to see.

The good news is that the hard half is already done. Every heavy stage persists to a folder, and `--resume-from` already proves a stage can start from disk state alone:

| Stage | Reads | Writes |
|---|---|---|
| preprocess | raw `.h5` | `binary/`, `bad_channels.json` |
| sort | `binary/` | `sorter_output/`, `sorting_params.json` |
| analyze | `binary/`, `sorter_output/` | `analyzer_output/` |
| report | `analyzer_output/` | `spike_times.npy`, metrics, plots |

So converting those four is per-stage entry-point work, not a rewrite. They are also the four stages with no test coverage, which is why that work deserves its own careful pass rather than riding along with a file move.

### Deployment constraint

Both entry points must stay at the repository root. Your SLURM scripts reference them by absolute path under `MEA_Analysis/IPNAnalysis/`, and `PYTHONPATH` points at that same directory. Moving either means updating `sbatch*.sh` in the same change.

---

## 5. Where things stand

Branch `pipeline-fable5-review` is 8 commits ahead of `main`: 40 files changed, 7504 insertions, 156 deletions. Nothing is pushed and nothing is merged.

New analysis modules: `unit_bursts`, `synchrony`, `propagation`, `celltype`, `trajectory`, `track_units`, `samples`, `qc_wells`, `stats_report`.

A well directory now also contains `unit_stats.csv` with per-unit burst, class and propagation columns, `sttc_matrices.npz`, `curation_summary.json`, `sorting_params.json`, `connectivity_summary.svg`, and a `network_results.json` carrying sample, curation, provenance, unit-level, connectivity, propagation and cell-type blocks.

The end-to-end path from recordings to a genotype comparison:

```bash
python run_pipeline_driver.py /data/experiment --config mea_config.json \
    --samples-file Configfiles/samples_example.csv
python collect_network_jsons.py --root AnalyzedData --out-dir metrics
python stats_report.py --table metrics/network_metrics_ALL.csv --out-dir reports
```

---

## 6. Open questions that need your data

These cannot be settled from code. The first one matters most.

### Kilosort `invert_sign`

Set to `True` in both parameter tiers. Kilosort's own documentation says the flag flips polarity "to conform to standard expected by Kilosort4", and Maxwell traces after the unsigned-to-signed conversion are already negative-going. If the flag is wrong, the sorter has been clustering on positive rebounds and losing units, on every run the lab has done.

I left it untouched, as you asked. Testing it is now one flag:

```bash
python mea_analysis_routine.py <file> --well well000 --config mea_config.json \
    --kilosort-params '{"invert_sign": false}'
```

Compare unit count, the polarity of mean templates at the extremum channel, and the fraction of refractory violations. Note that the analyzer computes templates from the original recording with `peak_sign='neg'`, so it would hide the problem rather than reveal it.

### Analyzer sparsity radius

Set to 50 µm. Median nearest-neighbour distance on sparse Maxwell configurations is 55 to 65 µm, so most units get one to three channels. That weakens template metrics, monopolar triangulation and any waveform matching. A radius of 100 µm is likely better, but the right value depends on your electrode selection.

### The presence ratio threshold

Left at 0.75 deliberately, because it is a scientific choice rather than a bug. It does reject units that fire only inside bursts, which a hypoactive genotype has most of. Rather than change it, every well now writes per-rule rejection counts. Compare those across genotypes before trusting any group statistics; if the knockout is losing more units to that rule, the threshold needs lowering.

### The two Kilosort parameter tiers

They differ in `dmin`, `cluster_downsampling` and `max_cluster_subset`, which change the sorting result, not just memory use. The tier is now logged and the resolved parameters recorded, but the values are not pinned. Pin them through `sorting.kilosort_params` for any dataset processed on more than one GPU model.

---

## 7. What remains

In the order I would do it.

1. **Settle `invert_sign` on one well.** Everything downstream inherits the answer.
2. **Frozen regression set.** Three to five real wells per preparation type with expected metric ranges, run in CI. Detector edits silently changing metrics is the largest ongoing risk to a longitudinal study.
3. **Metrics dictionary.** One table: name, unit, formula, reference, which detector it depends on. Reviewers ask for this and it is tedious to reconstruct later.
4. **Preparation presets.** Dissociated mouse, iPSC and organoid cultures have different burst timescales. `prep_type` is already carried per well, so this is small, but the organoid numbers should be chosen against real organoid recordings rather than guessed.
5. **Pipeline stages as blocks**, if the Nextflow plan goes ahead. Per-stage entry points for preprocess, sort, analyze and report, with tests written first since those stages currently have none.
6. **Organoid LFP.** A 1 to 100 Hz branch for band power, the aperiodic slope and phase-amplitude coupling. Only worth it if organoids become a main line of work.

---

## 8. References

Mossink et al. 2021, *Stem Cell Reports* — the design and feature-set reference for this assay, including the finding that MEA batch explained 69% of variance.
Cotterill et al. 2016, *J Neurophysiol* — burst detector comparison; MaxInterval first, logISI second, run both.
Pasquale et al. 2010 — logISI burst detection.
Cutts & Eglen 2014, *J Neurosci* — spike time tiling coefficient.
Wagenaar et al. 2006; Chiappalone et al. 2005 — network burst and superburst definitions.
Sit et al. 2024, *Cell Reports Methods* — MEA-NAP.
Hornauer et al. 2024, *Stem Cell Reports*; Hornauer et al. 2026, bioRxiv — DeePhys, and the first ground-truth work on HD-MEA cell typing.
Vinogradov et al. 2024, bioRxiv — effective excitability.
Ort et al. 2026, bioRxiv — graph metrics vary with method as much as with biology.
van Beest et al. 2024, *Nat Methods* — UnitMatch.
Wolff et al. 2025, *STAR Protocols* — spikeNburst and nicespike; somatic versus non-somatic templates.
Trujillo et al. 2019, *Cell Stem Cell* — organoid oscillations.
