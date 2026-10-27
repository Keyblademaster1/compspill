# compspill

Estimate a flow cytometry **compensation (spillover) matrix from single-stain controls**,
AutoSpill-style, and get a single HTML report that tells you where to look before you trust it.

- Reads FCS files directly; detectors are taken from the files' `$SPILLOVER` keyword.
- Matches each control file to its detector by file name (`Alexa Fluor 488`, `AF488`,
  `PE-Cy7`, ...), by a table in a config file, or, as a last resort, by where it is brightest.
- Iterative robust-slope refinement (the AutoSpill idea), starting from the classic
  median estimate.
- Warnings for the usual control problems: two particle populations in one control,
  too few positives, off-scale events, **dim controls** that pick up autofluorescence.
- Optional **check on stained samples**: for every spillover value, what do your real
  samples say? This catches controls that overestimate spill.
- Writes the matrix as CSV, FlowJo `.mtx` and `.npy`, plus residual and spreading tables,
  per-control plots and `report.html` (self-contained, can be emailed).

> **Status:** 0.1.0, research software. The matrix is a well-founded starting point, not a
> verdict. Look at the report.

## Install

Requires Python ≥ 3.11.

```bash
git clone https://github.com/Keyblademaster1/compspill.git
cd compspill
python3 -m venv .venv
source .venv/bin/activate
pip install -e '.[test]'
```

Dependencies: `numpy`, `flowio`, `matplotlib` (and `pytest` for the tests).

## Quick start with example data

The repository contains no real data. `examples/make_example_data.py` simulates a small
5-colour experiment with a **known** spillover matrix and some deliberate problems:

```bash
python examples/make_example_data.py          # writes example_data/ (git-ignored)
compspill run --controls example_data/controls \
              --out example_output \
              --config examples/config_example.toml \
              --samples example_data/samples
open example_output/report.html               # macOS; or open it in any browser
```

`example_data/true_spillover.csv` holds the true matrix, so you can compare. The report
should show:

1. **dim control** for FITC. That control is made of cells whose positive cells are more
   autofluorescent than the negatives, so its row comes out too high.
2. **two particle populations** for APC: the control holds small particles with a
   different spectrum. The example config keeps only the beads (`[particles."APC-A"]`).
3. **off-scale** events in the PE control, which are dropped.
4. In the sample check, FITC → PE and FITC → PE-Cy7 are flagged: the samples say 18 % and
   1 % (the true values), the dim control said 33 % and 6 %.
5. A manual override (FITC → BV421 set to 0) with its reason.

Run it on your own data:

```bash
compspill run --controls "path/to/Compensation Controls" --out comp_out
# or a quoted glob if the folder holds other files too:
compspill run --controls "data/Compensation Controls_*.fcs" --out comp_out --samples data/samples/
```

`python -m compspill run ...` works the same. Options: `--unstained FILE`, `--no-unstained`,
`--config FILE`, `--samples DIR_OR_GLOB [...]`, `--samples-max-events N`, `--no-plots`, `-q`.

## How it works, in plain language

Each fluorochrome is measured mostly in its own (*primary*) detector, but some of its
light also lands in other detectors. The spillover matrix `S` records those fractions:
`S[i, j]` is how much of fluor *i* shows up in detector *j*, relative to its primary
detector (`S[i, i] = 1`). Compensation undoes it: `compensated = raw @ inv(S)`. This is the
same layout as the FCS `$SPILLOVER` keyword and FlowJo's matrices.

1. **Gate every control.** Events that hit the top of the scale (≥ 99 % of `$PnR`) are
   dropped, because their true value is unknown. *Negatives* and *positives* are split at
   the 99.5th percentile of the unstained control in the primary detector. A scatter gate
   (± 3.5 robust SD in log FSC / log SSC) is put around **each population separately**,
   because stained beads or cells can differ in size from unstained ones. The brightest
   0.5 % of positives (aggregates) are dropped.
2. **Classic start.** `S[i, j] = (median_pos_j − median_neg_j) / (median_pos_i − median_neg_i)`.
3. **Iterate.** Compensate every control with the current `S`. If `S` were right, the
   positives would go *straight up*: compensated detector *j* would not change as
   compensated primary *i* gets brighter. The robust (Huber) slope of *j* vs *i* is the
   remaining error `E[i, j]`. Update `S ← S + E @ S`, set the diagonal back to 1, and repeat
   until the largest slope is below 1e-4 (usually 2–5 rounds). Slopes use every event
   rather than two medians, so a few odd events move the answer less.
4. **Spreading matrix** (Nguyen et al. 2013): how much a bright fluor *widens* the
   negatives in another detector after compensation. No matrix can remove that.
5. **Sample check** (optional, `--samples`): for each fluor *i*, compensate the stained
   samples with the matrix but leave *i* itself uncompensated. Among the events brightest
   in *i*, the robust slope of each detector vs *i* gives fluor *i*'s spill row *as the
   samples see it*, converted back through the matrix so that errors elsewhere in the row
   do not leak in. The report lists "matrix says X %, samples say Y %" (median and
   interquartile range over samples) and flags clear, consistent disagreements.

**No unstained control?** If there is no `--unstained` file and no control file with
"unstained" in its name, each control's own negative population is used. Its primary
detector is split in two (Otsu threshold on arcsinh-transformed values). This works for
controls with a clear negative peak. Pass `--no-unstained` to force it.

**Control ↔ detector matching** is by *detector*, the channel name in the file. In some
labs the channel is named after a reference dye (e.g. `Alexa Fluor 700-A` holding an R718
stain); that is fine, because the control for that channel is still found by that channel
name. Order of precedence: config `[match]` table → detector name in the file name →
marker name (from `[markers]`) in the file name → brightest detector. The last one is
refused if the runner-up is within 80 %. Two controls for one detector, a detector without
a control, or a name that fits two detectors are errors with a clear message. The
matching used is shown in the terminal and the report.

## Reading the report

- **Warnings first.**
  - *dim control*: positives < 10× negatives in the primary detector. With
    **cell-based** controls this is the classic trap: positive cells are often larger or
    more autofluorescent than negative cells, and that difference is mistaken for
    spillover. Check the row against samples or redo the control with beads.
  - *two particle populations*: the positives form two FSC populations (doublets at 2×
    FSC are ignored). The warning says how different the two populations' spill vectors
    are and suggests an FSC threshold. Use `[particles."DETECTOR"]` to keep one of them.
    For cell controls this is often debris or dead cells vs intact cells.
  - *few positives* (< 200), *off-scale* (> 2 % of events), *negative spill* (< −1 %:
    physically impossible, so usually a gating or control problem), *not converged*.
- **Spillover matrix** in %. Purple outline = manual override, dashed red = samples
  disagree.
- **Controls table**: what matched what, how many events ended up negative / positive,
  medians, FSC of each population.
- **Compensated controls** (click to expand): in every panel, the black positives should
  rise straight up along the red line. A lean to the right means under-compensation, a
  lean to the left over-compensation.
- **Sample check**: a flag is **a prompt to look, not proof**. Biology also makes slopes:
  if cells bright for marker *i* also express marker *j* (e.g. CD3 and CD56 on NKT cells),
  the samples show "spill" that compensation must not remove. Dead cells are bright in
  everything; set `samples.viability_detector` to drop them. Flags are most convincing
  when the samples say ≈ 0 and the controls say clearly more.

## Configuration

All settings are optional. See [`examples/config_example.toml`](examples/config_example.toml)
for every key with comments. The main ones:

```toml
[markers]                 # labels in the report; FlowJo infixes like Comp-CD3-BB700 in the .mtx
"BB700-A" = "CD3"

[match]                   # control file -> detector, when the file name does not say it
"tube 7.fcs" = "APC-A"

[particles."BV605-A"]     # keep one particle population of a control
fsc_threshold = 31623     # e.g. log10 FSC-A < 4.5
keep = "below"            # or "above"
apply_to = "positives"    # or "both"

[[override]]              # replace one value; the reason is printed in the report
source = "Alexa Fluor 488-A"
detector = "BV605-A"
value = 0.0
reason = "Samples show no AF488 signal in BV605; the dim cell control picked up autofluorescence."

[samples]
viability_detector = "APC-Cy7-A"
```

Unknown keys are an error, so a typo cannot silently change nothing.

## Python API

```python
from compspill import estimate_spillover, sample_check
from compspill.report import write_report

res = estimate_spillover("path/to/controls", config="settings.toml", log=print)
res.S, res.detectors          # the matrix and its row/column order
res.warnings                  # list of dicts: kind, subject, message
res.controls[0].stats         # event counts, medians, ... per control

rows, plots = sample_check(res, "path/to/samples")
write_report(res, "out", rows, plots)
```

`controls` can be a folder, a glob, a list of files, or a dict `{detector: file}` (no
matching needed). In-memory data can be passed as `compspill.io.Events` objects.
`compspill.synthetic` simulates controls and samples with a known matrix.

## Limitations

- **Human judgment is required.** The tool cannot know whether a slope in the samples is
  spillover or biology, whether a dim control is trustworthy, or which of two particle
  populations has the right spectrum. It points at the places to look.
- Single-stain controls must be as bright as, or brighter than, the samples, and must use
  the same fluorochrome lot (tandem dyes differ between lots and degrade).
- Gating is deliberately simple (percentile split, rectangular robust scatter gate). Very
  messy controls (lots of debris, several cell types) may need pre-gating in other
  software and an export of the gated events.
- Conventional compensation only, with one detector per fluor. No spectral unmixing and no
  autofluorescence extraction.
- Only linear-scale FCS data are supported (normal for modern digital cytometers).

## Relation to AutoSpill

The idea of refining the matrix iteratively, using robust slopes of compensated controls
until the positive populations stand upright, comes from **AutoSpill**:

> Roca CP, Burton OT, Gergelits V, Prezzemolo T, Whyte CE, Halpert R, Kreft Ł, Collier J,
> Botzki A, Spidlen J, Humblet-Baron S, Liston A. AutoSpill is a principled framework that simplifies the analysis of
> multichromatic flow cytometry data. *Nat Commun* 12, 2890 (2021).
> https://doi.org/10.1038/s41467-021-23126-8

If you use compspill in published work, please cite AutoSpill for the method (and this
repository for the implementation, see `CITATION.cff`).

compspill is an **independent, simplified Python implementation** of that idea, not a port
of the AutoSpill R code, and its results will not be identical. Differences:

- Simpler gating: a percentile split against the unstained control and separate robust
  scatter gates for positives and negatives.
- No autofluorescence correction. Instead there are explicit warnings for dim controls, and
  the sample check to catch rows a control got wrong.
- Additions aimed at day-to-day use: tolerant control↔detector matching, detection of
  two particle populations, manual overrides with mandatory reasons, a spillover spreading
  matrix, FlowJo `.mtx` export and a single self-contained HTML report.

## Tests

```bash
pytest
```

The tests simulate controls (beads, autofluorescent cells, a two-particle control,
off-scale events, a dim cell control) and samples with a known matrix. They check matrix
recovery, file-name matching, `.mtx`/CSV/FCS round trips, overrides, the warnings and the
sample check.

## License

MIT; see [LICENSE](LICENSE).
