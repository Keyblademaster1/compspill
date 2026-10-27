"""Independent check of the matrix on fully stained samples.

For each fluor i we compensate the samples with the final matrix except that row
i is left out (S0 = S with the off-diagonal of row i set to 0).  Then fluor i is
not compensated at all, while everything else is.  Among the events brightest
in i, the robust slope of each compensated detector j vs compensated i measures
how much of i lands in j.  Converting those slopes back through S0 gives the
whole spill row of fluor i "as the samples see it" (see ``_row_from_slopes``).

Why the whole row at once: zeroing a single S[i, j] while other wrong values in
row i stay in place lets their error leak into j via the other fluors' spill
(e.g. a wrong AF488->BV711 value travels on into AF700 through BV711's large
spill into AF700).

Biology also produces slopes: if cells bright for marker i also express marker
j, the slope is positive even with perfect compensation.  Dead cells bind
everything a little and are bright in every detector.  So a disagreement is a
prompt to look at the plot, not proof that the controls are wrong.  It is most
convincing when the samples say ~0 and the controls clearly more, e.g. a dim
cell-based control that picked up autofluorescence.
"""
from __future__ import annotations

import glob
from pathlib import Path

import numpy as np

from .io import as_events
from .stats import robust_slope


def find_samples(spec):
    """Folder, glob pattern, single file, or list of those -> list of paths."""
    specs = spec if isinstance(spec, (list, tuple)) else [spec]
    out = []
    for s in specs:
        p = Path(str(s))
        if p.is_dir():
            out += sorted(q for q in p.iterdir() if q.suffix.lower() == '.fcs' and not q.name.startswith('.'))
        elif glob.has_magic(str(s)):
            out += sorted(Path(q) for q in glob.glob(str(s)) if q.lower().endswith('.fcs'))
        else:
            out.append(p)
    if not out:
        raise FileNotFoundError(f'no sample FCS files found in {spec}')
    return out


def load_samples(paths, detectors, max_events, seed=0, saturation_fraction=0.99, log=None):
    """Raw detector matrices of each sample (off-scale events removed, subsampled)."""
    rng = np.random.default_rng(seed)
    out = []
    for p in paths:
        ev = as_events(p).subsample(max_events, rng)
        X = ev.matrix(detectors)
        top = np.array([ev.ranges.get(d, np.inf) for d in detectors])
        X = X[(X < saturation_fraction * top).all(1)]
        out.append((ev.name, X))
        if log:
            log(f'  {ev.name}: {len(X)} events')
    return out


def _row_from_slopes(g, S0, i):
    """Slopes g (g[i] = 1) measured after compensating with S0 -> spill row of i.

    Pure fluor-i events are compensated to t * (e_i + r) @ inv(S0), with r the
    true off-diagonal spill row, so g is proportional to (e_i + r) @ inv(S0).
    Multiplying back by S0 and rescaling the diagonal to 1 recovers e_i + r.
    """
    v = g @ S0
    return v / v[i]


def sample_check(result, samples, config=None, log=None):
    """Estimate every spill row from stained samples and compare with the matrix.

    ``samples``: folder / glob / list of paths, or list of (name, raw matrix in
    ``result.detectors`` order).  Returns (rows, plot_data).  Rows are reported
    for |S[i, j]| >= min_spill, for every pair in rows of warned controls, and
    for every flagged pair.  A pair is flagged when the samples' median differs
    from the matrix by more than ``abs_tolerance`` and ``rel_tolerance`` x |S|,
    and the matrix value lies outside the samples' interquartile range (i.e.
    most samples agree with each other, not with the matrix).
    """
    from .estimate import short_name
    cfg = (config or result.config)['samples']
    det, S, n = result.detectors, result.S, len(result.detectors)
    if samples and isinstance(samples, list) and isinstance(samples[0], tuple):
        loaded = samples
    else:
        loaded = load_samples(find_samples(samples), det, cfg['max_events'], cfg['seed'],
                              result.config['gating']['saturation_fraction'], log)
    neg = result.negative_reference
    via = cfg.get('viability_detector')
    if via:
        from .matching import resolve_detector
        v = det.index(resolve_detector(via, det, result.markers))
        Sinv = np.linalg.inv(S)
        vcut = np.percentile((neg @ Sinv)[:, v], 99.5)
        loaded = [(nm, X[(X @ Sinv)[:, v] < vcut]) for nm, X in loaded]
    warned = {w['subject'] for w in result.warnings if w['kind'] in ('dim control', 'two particle populations')}

    rows, plots = [], {}
    for i in range(n):
        if via and i == v:
            continue  # bright = dead cells: bright in everything, not checkable
        S0 = S.copy()
        S0[i] = 0
        S0[i, i] = 1
        M = np.linalg.inv(S0)
        neg_cut = np.percentile((neg @ M)[:, i], cfg['negative_percentile'])
        ests, pooled = [], []
        for name, X in loaded:
            C = X @ M
            bright = C[:, i] > max(neg_cut, np.percentile(C[:, i], 100 - cfg['top_percent']))
            if bright.sum() < cfg['min_bright_events']:
                continue
            g = np.array([1.0 if j == i else robust_slope(C[bright, i], C[bright, j])[1] for j in range(n)])
            ests.append(_row_from_slopes(g, S0, i))
            pooled.append(C[bright])
        ests = np.array(ests)
        for j in range(n):
            if j == i:
                continue
            val = float(S[i, j])
            if len(ests):
                med, lo, hi = np.median(ests[:, j]), *np.percentile(ests[:, j], [25, 75])
            else:
                med = lo = hi = np.nan
            diff = med - val
            flagged = bool(len(ests)) and abs(diff) > cfg['abs_tolerance'] \
                and abs(diff) > cfg['rel_tolerance'] * abs(val) and not (lo <= val <= hi)
            if not (flagged or abs(val) >= cfg['min_spill'] or short_name(det[i]) in warned):
                continue
            if not flagged:
                hint = ''
            elif abs(med) < cfg['abs_tolerance'] / 2:
                hint = 'samples ~0: control may overestimate (dim / autofluorescent control?)'
            elif med > val:
                hint = 'samples higher: under-compensated, or co-expression / dead cells'
            else:
                hint = 'samples lower: matrix may over-compensate (dim / autofluorescent control?)'
            rows.append(dict(source=det[i], detector=det[j], i=i, j=j, matrix=val,
                             controls=float(result.S_estimated[i, j]), samples=float(med),
                             samples_q1=float(lo), samples_q3=float(hi), n_samples=len(ests),
                             flagged=flagged, hint=hint))
            if flagged:
                allC = np.vstack([X[:3000] @ M for _, X in loaded])
                g_mat = (np.eye(n)[i] + (S[i] - np.eye(n)[i])) @ M
                g_smp = np.median(ests, 0) @ M
                plots[(i, j)] = dict(all=allC[:, [i, j]], bright=np.vstack(pooled)[:30000][:, [i, j]],
                                     g_matrix=g_mat[j] / g_mat[i], g_samples=g_smp[j] / g_smp[i],
                                     matrix=val, samples=float(med))
    rows.sort(key=lambda r: (not r['flagged'], -abs(r['samples'] - r['matrix']) if r['n_samples'] else 0))
    return rows, plots
