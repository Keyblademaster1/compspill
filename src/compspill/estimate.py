"""Spillover estimation from single-stain controls, AutoSpill-style.

Outline (Roca et al. 2021, Nat Commun 12:2890, simplified):

1. Gate every control: drop off-scale events; split negatives / positives at a
   high percentile of the unstained control in the primary detector; put a scatter
   gate around each population *separately* (stained particles may differ in size
   from the unstained ones); drop the brightest 0.5 % of positives (aggregates).
2. Start from the classic estimate: (median pos - median neg) / same in primary.
3. Iterate.  Compensate every control with the current S.  If S were right, the
   positive population would go "straight up": compensated detector j would not
   change with compensated primary i.  The robust slope of comp-j vs comp-i is the
   remaining error E[i, j];  S <- S + E @ S, renormalise the diagonal, repeat
   until max |E| < tolerance.  Slopes use every event, not just two medians, so
   one odd sub-population moves the answer less.
"""
from __future__ import annotations

import glob
import os
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from .config import load_config
from .io import Events, as_events, find_scatter, fluorescence_channels
from .matching import MatchError, by_filename, by_signal, resolve_detector
from .stats import mad, otsu_threshold, robust_sd, robust_slope, two_gaussians


@dataclass
class Control:
    """One single-stain control after gating."""
    name: str
    detector: str
    index: int
    match_method: str
    X: np.ndarray            # raw events x detectors
    fsc: np.ndarray | None
    ssc: np.ndarray | None
    pos: np.ndarray          # boolean masks into X
    neg: np.ndarray
    cut: float               # neg/pos threshold in the primary detector
    n_saturated: int
    particle_rule: str = ''
    bimodal: dict | None = None
    stats: dict = field(default_factory=dict)


@dataclass
class SpilloverResult:
    S: np.ndarray                    # final matrix (after overrides)
    detectors: list
    S_estimated: np.ndarray          # before overrides
    controls: list
    history: list                    # max |E| per iteration
    converged: bool
    residual_slopes: np.ndarray      # remaining E with the final S
    median_offsets: np.ndarray       # % of primary signal, pos vs neg medians
    spreading: np.ndarray            # spillover spreading matrix
    warnings: list
    overrides: list
    markers: dict
    unstained: str | None
    negative_reference: np.ndarray   # raw events used as 'negative' for the sample check
    config: dict

    def label(self, d):
        """Short display label: 'CD3 (BB700)' or 'AF488'."""
        short = d.replace('Alexa Fluor ', 'AF')
        short = short[:-2] if short.endswith('-A') else short
        return f'{self.markers[d]} ({short})' if self.markers.get(d) else short


def short_name(d):
    d = d.replace('Alexa Fluor ', 'AF')
    return d[:-2] if d.endswith('-A') else d


# ------------------------------------------------------------------- inputs

def _list_fcs(folder):
    return sorted(p for p in Path(folder).iterdir() if p.is_file() and p.suffix.lower() == '.fcs'
                  and not p.name.startswith('.'))


def collect_inputs(controls, unstained=None):
    """Normalise the many ways to pass controls -> (list of (name, Events or path,
    detector hint or None), unstained path/Events or None)."""
    if isinstance(controls, (str, os.PathLike)) and (Path(controls).is_dir() or glob.has_magic(str(controls))):
        files = _list_fcs(controls) if Path(controls).is_dir() else \
            sorted(Path(p) for p in glob.glob(str(controls)) if p.lower().endswith('.fcs'))
        if unstained is None:
            un = [p for p in files if 'unstained' in p.name.lower()]
            if len(un) > 1:
                raise MatchError(f'several files look unstained: {[p.name for p in un]}; pass one '
                                 f'explicitly (--unstained)')
            unstained = un[0] if un else None
        elif unstained is not False:
            files = [p for p in files if Path(p).resolve() != Path(str(unstained)).resolve()] \
                if not isinstance(unstained, Events) else files
        files = [p for p in files if 'unstained' not in p.name.lower()]
        items = [(p.name, p, None) for p in files]
    elif isinstance(controls, dict):
        items = [(v.name if isinstance(v, Events) else Path(v).name, v, k) for k, v in controls.items()]
    else:
        items = [(v.name if isinstance(v, Events) else Path(v).name, v, None) for v in controls]
    return items, (None if unstained is False else unstained)


# ------------------------------------------------------------------- gating

def scatter_gate(fsc, ssc, sel, k):
    """Events within k robust SDs (log10 FSC/SSC) of the centre of the events in ``sel``."""
    sc = np.column_stack([np.log10(np.clip(fsc, 1, None)), np.log10(np.clip(ssc, 1, None))])
    med = np.median(sc[sel], 0)
    spread = mad(sc[sel], axis=0) + 1e-9
    return (np.abs((sc - med) / spread) < k).all(1)


def gate_control(ev, detectors, i, cut, cfg, particle=None):
    """Split one control into gated positive and negative events."""
    g = cfg['gating']
    X = ev.matrix(detectors)
    fsc_name = find_scatter(ev, 'FSC', g['fsc_channel'])
    ssc_name = find_scatter(ev, 'SSC', g['ssc_channel'])
    fsc = np.asarray(ev.channels[fsc_name], float) if fsc_name else None
    ssc = np.asarray(ev.channels[ssc_name], float) if ssc_name else None
    top = np.array([ev.ranges.get(d, np.inf) for d in detectors])
    ok = (X < g['saturation_fraction'] * top).all(1)
    rawpos = ok & (X[:, i] > cut)
    rawneg = ok & (X[:, i] <= cut)

    bimodal = None
    if cfg['warnings']['bimodal_check'] and fsc is not None and rawpos.sum() >= 50:
        bimodal = fsc_bimodality(X, fsc, rawpos, rawneg, i)

    rule = ''
    if particle is not None:
        if fsc is None:
            raise ValueError(f'{ev.name}: particle selection needs an FSC channel')
        thr = float(particle['fsc_threshold'])
        keep = fsc < thr if particle['keep'] == 'below' else fsc >= thr
        rawpos &= keep
        if particle['apply_to'] == 'both':
            rawneg &= keep
        rule = f"FSC {'<' if particle['keep'] == 'below' else '>='} {thr:g} ({particle['apply_to']})"
    if rawpos.sum() < 3 or rawneg.sum() < 3:
        raise ValueError(f'{ev.name}: only {rawpos.sum()} positive / {rawneg.sum()} negative events '
                         f'left after gating; is this the right control / unstained file?')
    if fsc is not None and ssc is not None:
        pos = rawpos & scatter_gate(fsc, ssc, rawpos, g['scatter_k'])
        neg = rawneg & scatter_gate(fsc, ssc, rawneg, g['scatter_k'])
    else:
        pos, neg = rawpos, rawneg
    if pos.sum() >= 3 and g['top_trim_percent'] > 0:
        pos &= X[:, i] < np.percentile(X[pos, i], 100 - g['top_trim_percent'])
    return X, fsc, ssc, pos, neg, int((~ok).sum()), rule, bimodal


def fsc_bimodality(X, fsc, pos, neg, i):
    """Are the positives two particle populations of different size?

    Two particle types (e.g. capture beads plus small free-dye particles, or cells
    plus debris) can carry different spectra, and a single slope through both is
    then wrong for both.  A two-Gaussian fit on log10 FSC that is clearly better
    than one Gaussian (BIC), with well separated components (Ashman's D > 2) of
    which the smaller still holds >= 5 % of events, counts as bimodal.  We also
    report how different the two populations' spill vectors are.

    Doublets (two particles measured as one) sit at ~2x the FSC of the main
    population (+0.3 decades) in nearly every control; they share its spectrum
    and are removed by the scatter gate, so events more than 0.2 decades above
    the main FSC mode are left out of this check.
    """
    lf = np.log10(np.clip(fsc[pos], 1, None))
    h, edges = np.histogram(lf, bins=np.arange(lf.min(), lf.max() + 0.05, 0.05))
    h = np.convolve(h, [1, 2, 3, 2, 1], 'same') if len(h) > 5 else h
    mode = edges[np.argmax(h)] + 0.025
    keep = lf <= mode + 0.2
    lf = lf[keep]
    fit = two_gaussians(lf)
    fit['doublet_cut'] = float(10 ** (mode + 0.2))
    fit['bimodal'] = bool(fit['delta_bic'] > 10 and fit['ashman_d'] > 2 and min(fit['weights']) >= 0.05)
    fit['threshold'] = float(10 ** fit['valley'])
    fit['spill_difference'] = None
    if fit['bimodal']:
        small = fsc < fit['threshold']
        pos = pos & (fsc <= fit['doublet_cut'])
        mn = np.median(X[neg], 0)
        vecs = []
        for part in (pos & small, pos & ~small):
            if part.sum() < 30:
                return fit
            mp = np.median(X[part], 0)
            vecs.append((mp - mn) / (mp[i] - mn[i]))
        diff = np.abs(vecs[0] - vecs[1])
        diff[i] = 0
        fit['spill_difference'] = (float(diff.max()), int(diff.argmax()))
        fit['n_small'], fit['n_large'] = int((pos & small).sum()), int((pos & ~small).sum())
    return fit


def own_negative_cut(x, cofactor=150.0):
    """No unstained control: split the control's own primary channel in two
    (Otsu threshold on arcsinh-transformed values) and use the lower group as the
    negatives."""
    return float(np.sinh(otsu_threshold(np.arcsinh(x / cofactor))) * cofactor)


# ------------------------------------------------------------------- the maths

def classic_estimate(controls, n):
    S = np.eye(n)
    for c in controls:
        mp, mn = np.median(c.X[c.pos], 0), np.median(c.X[c.neg], 0)
        S[c.index] = (mp - mn) / (mp[c.index] - mn[c.index])
    return S


def residual_slopes(controls, S):
    """E[i, j] = robust slope of compensated j vs compensated i in control i."""
    n = len(S)
    Sinv = np.linalg.inv(S)
    E = np.zeros((n, n))
    for c in controls:
        i = c.index
        C = c.X[c.pos | c.neg] @ Sinv
        for j in range(n):
            if j != i:
                E[i, j] = robust_slope(C[:, i], C[:, j])[1]
    return E


def iterate(controls, S, tolerance=1e-4, max_iterations=100, log=None):
    history = []
    for it in range(max_iterations):
        E = residual_slopes(controls, S)
        history.append(float(np.abs(E).max()))
        if log:
            log(f'  iteration {it + 1:2d}: largest remaining slope = {history[-1]:.5f}')
        if history[-1] < tolerance:
            return S, history, True
        S = S + E @ S
        S = S / np.diag(S)[:, None]
    return S, history, False


def median_offsets(controls, S):
    """Remaining (median pos - median neg) in each detector, % of primary signal."""
    n = len(S)
    Sinv = np.linalg.inv(S)
    R = np.zeros((n, n))
    for c in controls:
        i = c.index
        C = c.X @ Sinv
        p, q = C[c.pos], C[c.neg]
        dF = np.median(p[:, i]) - np.median(q[:, i])
        R[i] = (np.median(p, 0) - np.median(q, 0)) / dF * 100
        R[i, i] = 0
    return R


def spreading_matrix(controls, S):
    """Spillover spreading matrix (Nguyen et al. 2013, Cytometry A 83:306).

    SS[i, j] = sqrt(rSD_pos(comp j)^2 - rSD_neg(comp j)^2) / sqrt(median_pos(i) - median_neg(i))

    Compensation removes spillover on average but not its photon noise: cells
    bright in fluor i come out *wider* in detector j.  No matrix can fix that;
    it tells you which marker pairs to avoid putting dim markers on.
    """
    n = len(S)
    Sinv = np.linalg.inv(S)
    SS = np.zeros((n, n))
    for c in controls:
        i = c.index
        C = c.X @ Sinv
        p, q = C[c.pos], C[c.neg]
        dF = max(np.median(p[:, i]) - np.median(q[:, i]), 1e-9)
        for j in range(n):
            if j != i:
                SS[i, j] = np.sqrt(max(robust_sd(p[:, j]) ** 2 - robust_sd(q[:, j]) ** 2, 0)) / np.sqrt(dF)
    return SS


# ------------------------------------------------------------------- driver

def _warn(warnings, kind, subject, message):
    warnings.append(dict(kind=kind, subject=subject, message=message))


def estimate_spillover(controls, unstained=None, detectors=None, config=None, log=None):
    """Estimate a spillover matrix from single-stain controls.

    Parameters
    ----------
    controls : folder, list of FCS paths / Events, or dict {detector: path/Events}
        A folder is scanned for .fcs files; a file with 'unstained' in its name is
        used as the unstained control unless ``unstained`` is given.
    unstained : path, Events, None or False
        False (or no unstained file found) = use each control's own negatives,
        found by splitting its primary channel in two (Otsu).
    detectors : list of detector (channel) names; default from the files.
    config : path to a TOML file, a dict, or None (defaults).
    log : callable for progress messages (e.g. ``print``).

    Returns a :class:`SpilloverResult`.
    """
    cfg = config if isinstance(config, dict) and 'gating' in config and 'override' in config \
        else load_config(config)
    say = log or (lambda *a: None)
    items, unstained = collect_inputs(controls, unstained)
    if not items:
        raise MatchError('no control files found')
    loaded = [(name, as_events(src), hint) for name, src, hint in items]
    un = as_events(unstained) if unstained is not None else None

    if detectors is None:
        detectors = cfg['detectors'] or fluorescence_channels(loaded[0][1])
    detectors = list(detectors)
    n = len(detectors)
    markers = {resolve_detector(k, detectors): v for k, v in cfg['markers'].items()}
    match_cfg = {k.lower(): v for k, v in cfg['match'].items()}

    # --- match controls to detectors
    matches, pending = {}, []
    for name, ev, hint in loaded:
        explicit = hint or match_cfg.get(name.lower()) or match_cfg.get(Path(name).stem.lower())
        if explicit:
            det, how = resolve_detector(explicit, detectors, markers), 'given' if hint else 'config [match]'
        else:
            det, how = by_filename(name, detectors, markers)
        if det is None:
            pending.append((name, ev))
            continue
        if det in matches:
            raise MatchError(f'two controls for {det}: {matches[det][0]} and {name}')
        matches[det] = (name, ev, how)
    if un is not None:
        baseline = np.median(un.matrix(detectors), 0)
    for name, ev in pending:
        X = ev.matrix(detectors)
        try:
            det, how = by_signal(X, baseline if un is not None else np.percentile(X, 5, 0),
                                 detectors, taken=set(matches))
        except MatchError as e:
            raise MatchError(f'cannot match control {name!r} to a detector: {e}. '
                             f'Add it to the [match] table of the config file.') from None
        matches[det] = (name, ev, how)
    missing = [d for d in detectors if d not in matches]
    if missing:
        raise MatchError(f'no control for detector(s) {", ".join(missing)}. Provide them, or restrict '
                         f'the detector list with detectors = [...] in the config.')

    # --- negative / positive cut per detector
    pct = cfg['gating']['negative_percentile']
    if un is not None:
        U = un.matrix(detectors)
        cuts = np.percentile(U, pct, axis=0)
        neg_reference = U
    else:
        say('No unstained control: each control\'s own negative population is used.')
        cuts = np.array([own_negative_cut(matches[d][1].matrix([d])[:, 0], cfg['plots']['cofactor'])
                         for d in detectors])

    particles = {resolve_detector(k, detectors, markers): v for k, v in cfg['particles'].items()}
    warnings, controls_out = [], []
    w = cfg['warnings']
    say(f"{'control':28s} {'detector':18s} {'neg':>6s} {'pos':>6s} {'off-scale':>9s}")
    for i, d in enumerate(detectors):
        name, ev, how = matches[d]
        X, fsc, ssc, pos, neg, nsat, rule, bim = gate_control(ev, detectors, i, cuts[i], cfg,
                                                               particles.get(d))
        c = Control(name, d, i, how, X, fsc, ssc, pos, neg, float(cuts[i]), nsat, rule, bim)
        mp, mn = np.median(X[pos, i]), np.median(X[neg, i])
        c.stats = dict(n_events=len(X), n_pos=int(pos.sum()), n_neg=int(neg.sum()),
                       median_pos=float(mp), median_neg=float(mn),
                       ratio=float(mp / mn) if mn > 0 else np.inf,
                       separation=float((mp - mn) / (robust_sd(X[neg, i]) + 1e-9)),
                       fsc_pos=float(np.median(fsc[pos])) if fsc is not None else np.nan,
                       fsc_neg=float(np.median(fsc[neg])) if fsc is not None else np.nan)
        controls_out.append(c)
        say(f'{name[:28]:28s} {d:18s} {neg.sum():6d} {pos.sum():6d} {nsat:9d}')

        lab = short_name(d)
        if pos.sum() < w['min_positives']:
            _warn(warnings, 'few positives', lab,
                  f'only {pos.sum()} gated positive events (< {w["min_positives"]}); the slope is noisy.')
        if nsat > w['saturated_percent'] / 100 * len(X):
            _warn(warnings, 'off-scale', lab,
                  f'{nsat} events ({nsat / len(X):.1%}) are off-scale in at least one detector and were '
                  f'dropped; consider a dimmer stain or lower voltage if many positives are affected.')
        if (mn > 0 and mp < w['dim_ratio'] * mn) or (mn <= 0 and c.stats['separation'] < w['dim_ratio']):
            _warn(warnings, 'dim control', lab,
                  f'positives are only {mp / mn:.1f}x the negatives in {d} (median {mp:.0f} vs {mn:.0f}). '
                  f'Dim controls, especially cell-based ones, can pick up autofluorescence differences '
                  f'between positive and negative cells and then overestimate spill into other '
                  f'detectors. Check this row against stained samples (--samples) or use beads.'
                  if mn > 0 else
                  f'positives are only {c.stats["separation"]:.1f} negative SDs above the negatives in {d}. '
                  f'Dim controls can overestimate spill; check this row against stained samples.')
        if bim and bim['bimodal']:
            sd = bim['spill_difference']
            extra = (f' Their spill vectors differ by up to {sd[0]:.3f} (in {short_name(detectors[sd[1]])}).'
                     if sd else '')
            handled = f' Handled by config: {rule}.' if rule else (
                f' To use only one of them add to the config:  [particles."{d}"] '
                f'fsc_threshold = {bim["threshold"]:.0f}, keep = "below" (or "above").')
            _warn(warnings, 'two particle populations', lab,
                  f'positives form two populations in FSC (medians ~{10 ** bim["means"][0]:.0f} and '
                  f'~{10 ** bim["means"][1]:.0f}, {bim["weights"][0]:.0%} / {bim["weights"][1]:.0%}).'
                  + extra + handled)

    # --- estimate
    S0 = classic_estimate(controls_out, n)
    say('Iterating:')
    S, history, converged = iterate(controls_out, S0, cfg['iteration']['tolerance'],
                                    cfg['iteration']['max_iterations'], say)
    if not converged:
        _warn(warnings, 'not converged', 'matrix',
              f'stopped after {len(history)} iterations with max remaining slope {history[-1]:.2g}.')
    S_est = S.copy()

    overrides = []
    for o in cfg['override']:
        a, b = resolve_detector(o['source'], detectors, markers), resolve_detector(o['detector'], detectors, markers)
        ia, ib = detectors.index(a), detectors.index(b)
        if ia == ib:
            raise ValueError('overrides of the diagonal are not allowed (it is 1 by definition)')
        overrides.append(dict(source=a, detector=b, estimated=float(S[ia, ib]), value=float(o['value']),
                              reason=str(o['reason'])))
        S[ia, ib] = float(o['value'])

    for i in range(n):
        for j in range(n):
            if i != j and S[i, j] < w['negative_spill']:
                _warn(warnings, 'negative spill', f'{short_name(detectors[i])} -> {short_name(detectors[j])}',
                      f'S = {S[i, j]:.4f}; real spillover cannot be negative, so values well below 0 '
                      f'usually mean a gating or control problem.')

    if un is None:
        neg_reference = np.vstack([c.X[c.neg] for c in controls_out])
    return SpilloverResult(
        S=S, detectors=detectors, S_estimated=S_est, controls=controls_out, history=history,
        converged=converged, residual_slopes=residual_slopes(controls_out, S),
        median_offsets=median_offsets(controls_out, S), spreading=spreading_matrix(controls_out, S),
        warnings=warnings, overrides=overrides, markers=markers,
        unstained=un.name if un is not None else None, negative_reference=neg_reference, config=cfg)
