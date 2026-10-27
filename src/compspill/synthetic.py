"""Simulated single-stain controls and stained samples with a known spillover matrix.

Used by the tests and by ``examples/make_example_data.py``.  The model is the one
compensation assumes, plus the things that go wrong in practice:

    raw = true @ S  +  autofluorescence(particle) * AF_SPECTRUM  +  noise

* beads: tight FSC/SSC, almost no autofluorescence; ~8 % doublets at 2x FSC
* cells: broader scatter, autofluorescence that grows with cell size
* optional second particle type with a *different* spectrum (two-particle control)
* optional off-scale events, and dim cell controls whose positive cells are more
  autofluorescent than the negatives (makes the control overestimate spill)
"""
from __future__ import annotations

import numpy as np

from .io import Events

DETECTORS = ['FITC-A', 'PE-A', 'PE-Cy7-A', 'APC-A', 'BV421-A']
TRUE_S = np.array([
    [1.000, 0.180, 0.010, 0.000, 0.000],   # FITC
    [0.020, 1.000, 0.060, 0.005, 0.000],   # PE
    [0.000, 0.030, 1.000, 0.040, 0.000],   # PE-Cy7
    [0.000, 0.004, 0.120, 1.000, 0.000],   # APC
    [0.090, 0.020, 0.002, 0.000, 1.000],   # BV421
])
AF_SPECTRUM = np.array([1.0, 0.6, 0.15, 0.05, 0.9])
RANGE = 262144.0


def _scatter(rng, n, kind):
    """(fsc, ssc, autofluorescence level) for n particles."""
    if kind == 'beads':
        fsc = 6e4 * 10 ** rng.normal(0, 0.03, n)
        ssc = 2e4 * 10 ** rng.normal(0, 0.04, n)
        af = 20 * 10 ** rng.normal(0, 0.1, n)
        dbl = rng.random(n) < 0.08
        fsc[dbl] *= 2
        ssc[dbl] *= 1.8
    elif kind == 'small':
        fsc = 1.2e4 * 10 ** rng.normal(0, 0.12, n)
        ssc = 4e3 * 10 ** rng.normal(0, 0.15, n)
        af = 10 * 10 ** rng.normal(0, 0.1, n)
    elif kind == 'cells':
        fsc = 5e4 * 10 ** rng.normal(0, 0.07, n)
        ssc = 1.5e4 * 10 ** rng.normal(0, 0.12, n)
        af = 150 * (fsc / 5e4) ** 2 * 10 ** rng.normal(0, 0.15, n)
    else:
        raise ValueError(kind)
    return fsc, ssc, af


def _measure(rng, true, S, af):
    signal = true @ S + af[:, None] * AF_SPECTRUM
    noise = rng.normal(0, 1, signal.shape) * np.sqrt(25 ** 2 + 0.8 * np.abs(signal))
    return np.minimum(signal + noise, RANGE)


def _events(name, X, fsc, ssc, detectors):
    ch = {'FSC-A': fsc, 'SSC-A': ssc}
    ch.update({d: X[:, k] for k, d in enumerate(detectors)})
    ch['Time'] = np.arange(len(fsc), dtype=float)
    return Events(name, ch, {c: RANGE for c in ch}, list(detectors))


def make_control(i, n=5000, frac_pos=0.5, kind='beads', brightness=2e4, S=None, detectors=None,
                 name=None, seed=0, second_population=None, n_offscale=0, pos_af_boost=1.0):
    """One single-stain control for fluor ``i``.

    second_population: dict(fraction, spectrum (row vector, S[i,i]=1), kind='small')
        -> that fraction of positives are other particles with another spectrum.
    n_offscale: that many positives are made far too bright (off-scale).
    pos_af_boost: positive particles get this x more autofluorescence (dim cell controls).
    """
    S = TRUE_S if S is None else S
    detectors = detectors or DETECTORS
    rng = np.random.default_rng(seed)
    npos = int(n * frac_pos)
    fsc, ssc, af = _scatter(rng, n, kind)
    true = np.zeros((n, len(S)))
    true[:npos, i] = brightness * 10 ** rng.normal(0, 0.15, npos)
    af[:npos] *= pos_af_boost
    X = _measure(rng, true, S, af)
    if second_population:
        k = int(npos * second_population['fraction'])
        f2, s2, a2 = _scatter(rng, k, second_population.get('kind', 'small'))
        S2 = S.copy()
        S2[i] = second_population['spectrum']
        X[:k] = _measure(rng, true[:k], S2, a2)
        fsc[:k], ssc[:k] = f2, s2
    if n_offscale:
        X[npos - n_offscale:npos, i] = RANGE
    return _events(name or f'{detectors[i][:-2]} stained.fcs', X, fsc, ssc, detectors)


def make_unstained(n=5000, kind='beads', S=None, detectors=None, seed=99):
    S = TRUE_S if S is None else S
    rng = np.random.default_rng(seed)
    fsc, ssc, af = _scatter(rng, n, kind)
    return _events('Unstained.fcs', _measure(rng, np.zeros((n, len(S))), S, af), fsc, ssc,
                   detectors or DETECTORS)


def make_sample(n=30000, S=None, detectors=None, seed=1, name='sample.fcs'):
    """Stained 'cells': a few cell types, each expressing a set of markers.
    Expression is independent of autofluorescence, and no cell type co-expresses
    FITC with anything, so FITC's row can be checked cleanly."""
    S = TRUE_S if S is None else S
    m = len(S)
    rng = np.random.default_rng(seed)
    fsc, ssc, af = _scatter(rng, n, 'cells')
    types = rng.integers(0, 4, n)
    true = np.zeros((n, m))
    patterns = [[0], [1, 2], [3], [4, 1]]
    for t, markers in enumerate(patterns):
        sel = types == t
        for k in markers:
            if k < m:
                true[sel, k] = 10 ** rng.normal(3.8, 0.4, sel.sum())
    return _events(name, _measure(rng, true, S, af), fsc, ssc, detectors or DETECTORS)


def standard_set(seed=0, two_particle=True, dim_fitc=False, offscale=True):
    """A full set: {name: Events} for the 5 controls + unstained, and the true S.

    two_particle: the APC control holds 40 % small particles with a different spectrum.
    dim_fitc: the FITC control is cell-based, dim, and its positives are 3x more
              autofluorescent -> it overestimates FITC's spill (like a dim cell control).
    """
    out = {}
    for i, d in enumerate(DETECTORS):
        kw = dict(seed=seed + i)
        if i == 3 and two_particle:
            alt = TRUE_S[3].copy()
            alt[2] = 0.35
            kw['second_population'] = dict(fraction=0.4, spectrum=alt)
        if i == 1 and offscale:
            kw['n_offscale'] = 150
        if i == 0 and dim_fitc:
            kw.update(kind='cells', brightness=600, pos_af_boost=3.0, frac_pos=0.6)
        ev = make_control(i, **kw)
        out[ev.name] = ev
    un = make_unstained(seed=seed + 50)
    return out, un, TRUE_S.copy()


def write_fcs(ev: Events, path, extra=None):
    """Write Events to an FCS 3.1 file (float32) with flowio, incl. $PnR and $SPILLOVER."""
    import flowio
    names = list(ev.channels)
    data = np.column_stack([ev.channels[c] for c in names]).astype(np.float32)
    meta = {f'p{k + 1}r': str(int(ev.ranges.get(c, RANGE))) for k, c in enumerate(names)}
    if ev.spill_detectors:
        d = ev.spill_detectors
        eye = np.eye(len(d))
        meta['spillover'] = ','.join([str(len(d))] + d + [f'{v:g}' for v in eye.ravel()])
    meta.update(extra or {})
    with open(path, 'wb') as fh:
        flowio.create_fcs(fh, data.ravel().tolist(), names, metadata_dict=meta)
