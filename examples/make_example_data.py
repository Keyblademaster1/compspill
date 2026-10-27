"""Write a small simulated dataset to try compspill on (no real data needed).

    python examples/make_example_data.py            # -> example_data/
    compspill run --controls example_data/controls --out example_output \
        --config examples/config_example.toml --samples example_data/samples

Five bead controls (FITC, PE, PE-Cy7, APC, BV421) plus an unstained control, with
a known true spillover matrix (saved as example_data/true_spillover.csv so you can
compare).  Built-in problems for the report to find:

* the APC control contains a second, smaller particle type with another spectrum;
* the PE control has off-scale events;
* the FITC control is a dim *cell* control whose positive cells are more
  autofluorescent than its negatives, so it overestimates FITC's spill;
  the three stained samples reveal this in the sample check.
"""
import argparse
from pathlib import Path

from compspill.io import write_csv
from compspill.synthetic import DETECTORS, make_control, make_sample, make_unstained, write_fcs

ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
ap.add_argument('--out', default='example_data')
a = ap.parse_args()

out = Path(a.out)
(out / 'controls').mkdir(parents=True, exist_ok=True)
(out / 'samples').mkdir(exist_ok=True)

from compspill.synthetic import TRUE_S  # noqa: E402

alt = TRUE_S[3].copy()
alt[2] = 0.35
controls = {
    # deliberately mixed naming styles: matching is by detector name in the file name
    'FITC stained.fcs': make_control(0, kind='cells', brightness=600, pos_af_boost=3.0, frac_pos=0.6, seed=1),
    'Comp_PE.fcs': make_control(1, n_offscale=150, seed=2),
    'PE-Cy7 single stain.fcs': make_control(2, seed=3),
    'APC beads.fcs': make_control(3, second_population=dict(fraction=0.4, spectrum=alt), seed=4),
    'BV421.fcs': make_control(4, seed=5),
}
for name, ev in controls.items():
    write_fcs(ev, out / 'controls' / name)
write_fcs(make_unstained(kind='cells', seed=6), out / 'controls' / 'Unstained.fcs')
for k in range(3):
    write_fcs(make_sample(n=40000, seed=10 + k), out / 'samples' / f'sample_{k + 1}.fcs')
write_csv(TRUE_S, DETECTORS, out / 'true_spillover.csv')
print(f'wrote {len(controls) + 1} controls and 3 samples to {out}/')
