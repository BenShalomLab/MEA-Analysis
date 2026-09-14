# samples.csv — what was cultured in each well

The pipeline reads project, date, chip, run and well from the directory layout. Nothing in a recording says which genotype, cell line or batch it came from, and nothing says how old the culture was. That has to come from a table the lab maintains.

Point the pipeline at one with `io.samples_file` in the config, or `--samples-file` on either the driver or the routine. Every well then writes a `sample` block into `network_results.json` and its checkpoint, and `collect_network_jsons.py` turns that block into `sample_*` columns next to the ids.

Without this file the pipeline still runs. It logs a warning, `sample.matched` is `false`, and the results identify the well but not the experiment.

## Format

CSV, TSV or XLSX. One row per group of wells. Header names are matched case-insensitively with spaces treated as underscores, so `Plating Date` and `plating_date` are the same column. Columns the pipeline does not know about are carried through to the output unchanged, so you can keep lab-specific fields in the same file.

## Key columns

| Column | Meaning |
|---|---|
| `project` | Project folder name |
| `date` | Recording date, any common format |
| `chip` | Chip / MEA serial |
| `run` | Run id |
| `well` | `well000`, `000` and `0` all mean the same well |

A row matches a recording when every key column it fills in matches. **Blank key cells are wildcards.** The row that pins the most key columns wins, so broad defaults and per-well exceptions can live in the same file:

```csv
chip,well,genotype
16657,,WT          <- every well of chip 16657 is WT
16657,well003,KO   <- except well003
```

If two rows match equally specifically, the first is used and a warning names the count. Add a key column to disambiguate.

A key the pipeline could not infer from the path blocks any row that names it. A row specifying `project` will not match a recording whose project is unknown, because nothing would have verified the association. Leave a key column out of the sheet entirely if you do not want it checked.

## Metadata columns

All optional. Fill in what the analysis needs to group by.

| Column | Meaning |
|---|---|
| `genotype` | The grouping variable for most comparisons, e.g. `WT`, `KO`, `R306C` |
| `line` | Cell line or animal line identifier |
| `prep_type` | `dissociated_mouse`, `ipsc`, `organoid` |
| `batch` | Differentiation, dissection or plating batch. Needed as a random effect: batch explained most of the between-well variance in Mossink et al. 2021 |
| `plating_date` | Used to derive DIV |
| `div` | Days in vitro, if you count from something other than `plating_date` (a thaw or differentiation date). Overrides the derived value |
| `density_cells_per_mm2` | Plating density |
| `media` | Media formulation or feeding schedule |
| `treatment` | Drug or vehicle |
| `treatment_concentration` | Dose, with units, e.g. `10uM` |
| `notes` | Free text |

## Dates and DIV

Accepted date formats: `2025-01-30`, `2025/01/30`, `2025_01_30`, `20250130`, `250130`, `30-01-2025`, `01/30/2025`. An eight-digit run of digits is read as `YYYYMMDD`, six digits as `YYMMDD`. A date embedded in a longer directory name, such as `2025-01-30_plate2`, is also found.

`div` is `recording_date - plating_date` in days. The `sample` block records `div_source` as either `column` or `plating_date` so you can tell which. A negative DIV is reported rather than hidden: it means the table or the folder name is wrong, and it should not reach the statistics unnoticed.

## Example

`Configfiles/samples_example.csv`:

```csv
project,chip,well,line,genotype,prep_type,batch,plating_date,density_cells_per_mm2,media,treatment,notes
CDKL5_cortex,16657,,C57BL6_WT,WT,dissociated_mouse,B12,2025-01-15,1200,BrainPhys,vehicle,
CDKL5_cortex,16657,well003,Cdkl5_KO_3,KO,dissociated_mouse,B12,2025-01-15,1200,BrainPhys,vehicle,edge well
CDKL5_cortex,16658,,Cdkl5_KO_3,KO,dissociated_mouse,B12,2025-01-15,1200,BrainPhys,vehicle,
SHANK3_ipsc,17201,,SHANK3_iso_ctrl,control,ipsc,D7,2024-11-04,900,BrainPhys+astro,vehicle,
SHANK3_ipsc,17202,,SHANK3_het,SHANK3+/-,ipsc,D7,2024-11-04,900,BrainPhys+astro,vehicle,
ORG_pilot,17410,,H9,WT,organoid,ORG5,2024-06-02,,BrainPhys,vehicle,sliced organoid
```

## Design notes for grouping

Experimental design constrains what the statistics can do later, so it is worth getting these columns right at the start rather than reconstructing them from lab notebooks:

- The **well** is the replicate, nested in chip, nested in batch. Wells on one chip are not independent.
- Fill in `batch` even when it feels redundant. Without it, batch effects are indistinguishable from genotype effects whenever a genotype was cultured mostly in one batch.
- Mossink et al. 2021 recommend at least 12 wells per condition across 2 or more independent batches, and at least 3 lines per genotype, using isogenic pairs where possible.
- Keep `prep_type` accurate. Dissociated mouse, iPSC-derived and organoid cultures have different burst timescales, and detector presets keyed on this field are planned (task B3).
