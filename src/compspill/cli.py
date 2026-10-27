"""Command line:  python -m compspill run --controls DIR --out DIR [options]"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

from . import __version__
from .config import ConfigError, load_config
from .estimate import estimate_spillover, short_name
from .matching import MatchError


def main(argv=None):
    ap = argparse.ArgumentParser(prog='compspill', description='Estimate a flow cytometry spillover '
                                 '(compensation) matrix from single-stain controls, AutoSpill-style.')
    ap.add_argument('--version', action='version', version=f'compspill {__version__}')
    sub = ap.add_subparsers(dest='cmd', required=True)
    r = sub.add_parser('run', help='estimate the matrix and write the report')
    r.add_argument('--controls', required=True,
                   help='folder with the single-stain control FCS files, or a quoted glob pattern')
    r.add_argument('--out', required=True, help='output folder (created if needed)')
    r.add_argument('--unstained', help='unstained control FCS file (default: a control file with '
                   '"unstained" in its name; if none, each control\'s own negatives are used)')
    r.add_argument('--no-unstained', action='store_true',
                   help='ignore any unstained file and use each control\'s own negatives')
    r.add_argument('--config', help='TOML settings file (see examples/config_example.toml)')
    r.add_argument('--samples', nargs='+', help='stained sample FCS files: folder(s), glob(s) or files, '
                   'for the independent check of the matrix')
    r.add_argument('--samples-max-events', type=int, help='random subsample per sample file (speed)')
    r.add_argument('--no-plots', action='store_true', help='skip the per-control plots (faster)')
    r.add_argument('-q', '--quiet', action='store_true')
    a = ap.parse_args(argv)

    log = (lambda *x: None) if a.quiet else print
    try:
        cfg = load_config(a.config)
        if a.samples_max_events:
            cfg['samples']['max_events'] = a.samples_max_events
        if a.no_plots:
            cfg['plots']['enabled'] = False
        unstained = False if a.no_unstained else a.unstained
        result = estimate_spillover(a.controls, unstained=unstained, config=cfg, log=log)
    except (MatchError, ConfigError, ValueError, KeyError, FileNotFoundError) as e:
        print(f'error: {e}', file=sys.stderr)
        return 2

    log('\nControl matching:')
    for c in result.controls:
        log(f'  {c.name}  ->  {c.detector}   ({c.match_method})')
    if result.warnings:
        log(f'\n{len(result.warnings)} warning(s):')
        for w in result.warnings:
            log(f'  [{w["kind"]}] {w["subject"]}: {w["message"]}')

    rows = plots = None
    if a.samples:
        from .samples import sample_check
        log('\nChecking against stained samples:')
        try:
            rows, plots = sample_check(result, a.samples, log=log)
        except (FileNotFoundError, KeyError, ValueError) as e:
            print(f'error: {e}', file=sys.stderr)
            return 2
        flagged = [x for x in rows if x['flagged']]
        log(f'\n{len(flagged)} spill value(s) where samples disagree with the matrix:')
        for x in flagged:
            log(f'  {short_name(x["source"]):>12s} -> {short_name(x["detector"]):<12s} '
                f'matrix {x["matrix"]:7.2%}   samples {x["samples"]:7.2%}   {x["hint"]}')

    from .report import write_report
    inputs = {'controls': a.controls, 'unstained': result.unstained or '(none)',
              'config': a.config or '(defaults)', 'samples': ' '.join(a.samples) if a.samples else '(none)'}
    path = write_report(result, a.out, rows, plots, inputs, log=None)
    log(f'\nWrote {Path(a.out) / "spillover.csv"}, spillover.mtx, spillover.npy and {path}')
    return 0


if __name__ == '__main__':
    sys.exit(main())
