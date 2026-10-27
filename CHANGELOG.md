# Changelog

## 0.1.0 (unreleased)

First version, generalised from lab-specific scripts.

- AutoSpill-style iterative estimation of the spillover matrix from single-stain controls.
- Control-to-detector matching by file name, marker name, config table, or brightest detector.
- Warnings: two particle populations (FSC), few positives, off-scale events, dim controls,
  negative spill, non-convergence.
- Per-control particle selection by FSC threshold; manual overrides with mandatory reasons.
- Check of every spill row against stained samples (optional viability exclusion).
- Outputs: CSV, FlowJo .mtx, .npy, residuals, spreading matrix, plots, self-contained HTML report.
- Synthetic data generator, example config and pytest suite.
