"""Settings, with defaults, read from an optional TOML file.

Unknown keys are an error rather than silently ignored: a typo in a gating
parameter should not quietly change nothing.  See ``examples/config_example.toml``
for every option with comments.
"""
from __future__ import annotations

import copy
import tomllib

DEFAULTS = {
    # restrict/reorder the detectors to compensate (default: from the files)
    'detectors': None,
    'gating': {
        'negative_percentile': 99.5,   # unstained percentile that separates neg / pos
        'scatter_k': 3.5,              # scatter gate half-width, robust SDs of log FSC/SSC
        'top_trim_percent': 0.5,       # drop this % brightest positives (aggregates)
        'saturation_fraction': 0.99,   # events >= this x $PnR in any detector are off-scale
        'fsc_channel': None,           # default: 'FSC-A' (or first FSC*-A)
        'ssc_channel': None,           # default: 'SSC-A' (or first SSC*-A)
    },
    'iteration': {'tolerance': 1e-4, 'max_iterations': 100},
    'warnings': {
        'min_positives': 200,          # fewer gated positives -> warning
        'dim_ratio': 10.0,             # positive median < this x negative median -> 'dim'
        'saturated_percent': 2.0,      # more off-scale events than this % -> warning
        'bimodal_check': True,         # look for two particle sizes among positives
        'negative_spill': -0.01,       # spill values below this are suspicious
    },
    'plots': {'enabled': True, 'cofactor': 150.0, 'dpi': 60, 'max_points': 20000},
    'samples': {
        'max_events': 50000,           # per sample file (random subsample, for speed)
        'top_percent': 2.0,            # 'bright in i' = top % of compensated i ...
        'negative_percentile': 99.0,   # ... and above this percentile of the negatives
        'min_spill': 0.01,             # check every |S[i,j]| >= this (plus warned controls)
        'abs_tolerance': 0.02,         # flag if |samples - matrix| > this ...
        'rel_tolerance': 0.3,          # ... and > this fraction of |matrix value|
        'min_bright_events': 50,       # per sample, else that sample is skipped
        'viability_detector': None,    # drop dead cells (positive here); its own row is skipped
        'seed': 0,
        'max_plots': 30,
    },
    'markers': {},     # detector -> marker name (labels, .mtx infixes)
    'match': {},       # control file name -> detector (explicit, wins over file name)
    'particles': {},   # detector -> {fsc_threshold, keep='below'|'above', apply_to='positives'|'both'}
    'override': [],    # [{source, detector, value, reason}]
}


class ConfigError(ValueError):
    pass


def _merge(base, new, where=''):
    for k, v in new.items():
        if k not in base:
            raise ConfigError(f'unknown config key {where}{k!r}; known: {", ".join(base)}')
        if isinstance(base[k], dict) and base[k] and isinstance(v, dict):
            _merge(base[k], v, f'{where}{k}.')
        else:
            base[k] = v


def load_config(source=None):
    """Return the full settings dict.  ``source``: None, a path to a TOML file, or a
    (possibly partial) dict with the same structure."""
    cfg = copy.deepcopy(DEFAULTS)
    if source is None:
        return cfg
    if not isinstance(source, dict):
        with open(source, 'rb') as fh:
            source = tomllib.load(fh)
    _merge(cfg, source)
    for det, p in cfg['particles'].items():
        extra = set(p) - {'fsc_threshold', 'keep', 'apply_to'}
        if extra or 'fsc_threshold' not in p:
            raise ConfigError(f'[particles.{det!r}] needs fsc_threshold (and optionally keep, '
                              f'apply_to); unexpected: {sorted(extra)}')
        if p.setdefault('keep', 'below') not in ('below', 'above'):
            raise ConfigError(f'[particles.{det!r}] keep must be "below" or "above"')
        if p.setdefault('apply_to', 'positives') not in ('positives', 'both'):
            raise ConfigError(f'[particles.{det!r}] apply_to must be "positives" or "both"')
    for o in cfg['override']:
        missing = {'source', 'detector', 'value', 'reason'} - set(o)
        if missing:
            raise ConfigError(f'override {o} is missing {sorted(missing)}')
        if not str(o['reason']).strip():
            raise ConfigError(f'override {o["source"]} -> {o["detector"]} needs a non-empty reason')
    return cfg
