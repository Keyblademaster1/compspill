"""Write all outputs: matrix files, tables, plots and one self-contained HTML report."""
from __future__ import annotations

import base64
import datetime
import html
import io
import json
from pathlib import Path

import numpy as np

from . import __version__
from .estimate import short_name
from .io import write_csv, write_flowjo_mtx


# ------------------------------------------------------------------- plots

def _plt():
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    return plt


def _png(fig, dpi):
    buf = io.BytesIO()
    fig.savefig(buf, format='png', dpi=dpi)
    _plt().close(fig)
    return buf.getvalue()


def plot_control(result, c, cofactor, dpi, max_points=20000):
    """Every compensated detector vs the compensated primary of one control.
    Correct compensation: the positives rise straight up along the red line
    (the negatives' median)."""
    plt = _plt()
    n = len(result.detectors)
    C = c.X @ np.linalg.inv(result.S)
    rng = np.random.default_rng(0)

    def pick(mask):
        idx = np.flatnonzero(mask)
        return idx if len(idx) <= max_points // 2 else rng.choice(idx, max_points // 2, replace=False)

    p, q = pick(c.pos), pick(c.neg)
    a = lambda v: np.arcsinh(v / cofactor)
    others = [j for j in range(n) if j != c.index]
    ncol = min(7, max(1, n - 1))
    nrow = -(-len(others) // ncol)
    fig, ax = plt.subplots(nrow, ncol, figsize=(2.6 * ncol, 2.6 * nrow), squeeze=False)
    for k, j in enumerate(others):
        axk = ax.ravel()[k]
        axk.scatter(a(C[q, j]), a(C[q, c.index]), s=.4, c='0.65', alpha=.5, rasterized=True)
        axk.scatter(a(C[p, j]), a(C[p, c.index]), s=.4, c='k', alpha=.5, rasterized=True)
        axk.axvline(a(np.median(C[c.neg, j])), color='r', lw=1, zorder=3)
        axk.set_xlabel('Comp ' + short_name(result.detectors[j]), fontsize=8)
        axk.tick_params(labelsize=6)
        if k % ncol == 0:
            axk.set_ylabel('Comp ' + short_name(c.detector), fontsize=8)
    for axk in ax.ravel()[len(others):]:
        axk.axis('off')
    fig.suptitle(f'{result.label(c.detector)} control ({c.name}): positives (black) should rise '
                 f'straight up along the red line', fontsize=9)
    fig.tight_layout()
    return _png(fig, dpi)


def plot_convergence(history, dpi):
    plt = _plt()
    fig, ax = plt.subplots(figsize=(4, 2.6))
    ax.semilogy(range(1, len(history) + 1), history, 'o-', color='C0')
    ax.set_xlabel('iteration')
    ax.set_ylabel('largest remaining slope')
    ax.grid(alpha=.3)
    fig.tight_layout()
    return _png(fig, dpi * 1.5)


def plot_fsc(c, dpi):
    """log10 FSC of the positives, with the suggested split."""
    plt = _plt()
    raw = c.X[:, c.index] > c.cut
    fig, ax = plt.subplots(figsize=(4.2, 2.6))
    ax.hist(np.log10(np.clip(c.fsc[raw], 1, None)), bins=80, color='0.4')
    ax.axvline(np.log10(c.bimodal['threshold']), color='r', lw=1, label='suggested split')
    ax.axvline(np.log10(c.bimodal['doublet_cut']), color='0.6', lw=1, ls='--', label='doublets above')
    ax.set_xlabel('log10 FSC (positives)')
    ax.set_title(f'{short_name(c.detector)} control', fontsize=9)
    ax.legend(fontsize=7)
    fig.tight_layout()
    return _png(fig, dpi * 1.5)


def plot_sample_pair(result, i, j, d, cofactor, dpi):
    plt = _plt()
    a = lambda v: np.arcsinh(v / cofactor)
    fig, ax = plt.subplots(figsize=(4.2, 3.6))
    ax.scatter(a(d['all'][:, 0]), a(d['all'][:, 1]), s=.3, c='0.75', alpha=.4, rasterized=True)
    ax.scatter(a(d['bright'][:, 0]), a(d['bright'][:, 1]), s=.4, c='C0', alpha=.5, rasterized=True,
               label='bright in source')
    x = np.linspace(max(d['bright'][:, 0].min(), 1), d['bright'][:, 0].max(), 100)
    off = np.median(d['bright'][:, 1] - d['g_samples'] * d['bright'][:, 0])
    ax.plot(a(x), a(off + d['g_matrix'] * x), 'r-', lw=1.2, label=f"matrix {d['matrix']:.1%}")
    ax.plot(a(x), a(off + d['g_samples'] * x), 'k--', lw=1.2, label=f"samples {d['samples']:.1%}")
    ax.set_xlabel(f'{short_name(result.detectors[i])} (not compensated for itself)', fontsize=8)
    ax.set_ylabel(f'Comp {short_name(result.detectors[j])}', fontsize=8)
    ax.legend(fontsize=7, loc='upper left', markerscale=8)
    ax.set_title(f'{short_name(result.detectors[i])} -> {short_name(result.detectors[j])}', fontsize=9)
    fig.tight_layout()
    return _png(fig, dpi * 1.5)


# ------------------------------------------------------------------- tables

def _write_tables(result, out, sample_rows):
    det = result.detectors
    with open(out / 'controls.csv', 'w') as fh:
        fh.write('file,detector,marker,matched_by,events,negatives,positives,off_scale,cut,'
                 'median_pos,median_neg,pos_neg_ratio,fsc_median_pos,fsc_median_neg,particle_rule\n')
        for c in result.controls:
            s = c.stats
            fh.write(f'"{c.name}",{c.detector},{result.markers.get(c.detector, "")},"{c.match_method}",'
                     f'{s["n_events"]},{s["n_neg"]},{s["n_pos"]},{c.n_saturated},{c.cut:.1f},'
                     f'{s["median_pos"]:.1f},{s["median_neg"]:.1f},{s["ratio"]:.2f},{s["fsc_pos"]:.0f},'
                     f'{s["fsc_neg"]:.0f},"{c.particle_rule}"\n')
    with open(out / 'residuals.csv', 'w') as fh:
        fh.write('control,detector,spill,remaining_slope,median_offset_percent\n')
        for i, a in enumerate(det):
            for j, b in enumerate(det):
                if i != j:
                    fh.write(f'{a},{b},{result.S[i, j]:.6f},{result.residual_slopes[i, j]:.6f},'
                             f'{result.median_offsets[i, j]:.3f}\n')
    write_csv(result.spreading, det, out / 'spreading.csv')
    with open(out / 'warnings.txt', 'w') as fh:
        for w in result.warnings:
            fh.write(f'[{w["kind"]}] {w["subject"]}: {w["message"]}\n')
    if sample_rows is not None:
        with open(out / 'sample_check.csv', 'w') as fh:
            fh.write('source,detector,matrix,controls_estimate,samples_median,samples_q1,samples_q3,'
                     'n_samples,flagged,hint\n')
            for r in sample_rows:
                fh.write(f'{r["source"]},{r["detector"]},{r["matrix"]:.6f},{r["controls"]:.6f},'
                         f'{r["samples"]:.6f},{r["samples_q1"]:.6f},{r["samples_q3"]:.6f},'
                         f'{r["n_samples"]},{r["flagged"]},"{r["hint"]}"\n')


# ------------------------------------------------------------------- HTML

CSS = """
:root { --fg:#1d232b; --muted:#5b6672; --bg:#fbfbfa; --card:#fff; --line:#dde1e5; --warn:#a15c00;
        --warnbg:#fff4e0; --flag:#b42318; }
body { font: 14px/1.5 -apple-system, system-ui, 'Segoe UI', sans-serif; color:var(--fg);
       background:var(--bg); margin:0; }
main { max-width: 1180px; margin: 0 auto; padding: 24px 16px 64px; }
h1 { font-size: 24px; margin: 0 0 4px; } h2 { font-size: 18px; margin: 36px 0 8px;
     border-bottom: 1px solid var(--line); padding-bottom: 4px; }
.muted { color: var(--muted); } .small { font-size: 12px; }
table { border-collapse: collapse; font-size: 12px; margin: 8px 0; }
th, td { border: 1px solid var(--line); padding: 3px 6px; text-align: right; white-space: nowrap; }
th { background: #f1f3f5; font-weight: 600; } td.l, th.l { text-align: left; }
.scroll { overflow-x: auto; }
.box { background: var(--card); border: 1px solid var(--line); border-radius: 6px; padding: 10px 14px; }
.warn { background: var(--warnbg); border-left: 4px solid var(--warn); padding: 6px 10px; margin: 6px 0; }
.flag { color: var(--flag); font-weight: 600; }
td.ovr { outline: 2px solid #6941c6; outline-offset: -2px; }
td.flg { outline: 2px dashed var(--flag); outline-offset: -2px; }
img { max-width: 100%; height: auto; } details { margin: 6px 0; } summary { cursor: pointer; }
.grid { display: flex; flex-wrap: wrap; gap: 12px; }
code { background: #f1f3f5; padding: 1px 4px; border-radius: 3px; }
"""


def _img(png, alt):
    return f'<img alt="{html.escape(alt)}" src="data:image/png;base64,{base64.b64encode(png).decode()}">'


def _color(v, scale):
    a = min(1.0, abs(v) / scale) ** 0.5
    rgb = (226, 110, 30) if v > 0 else (40, 110, 200)
    txt = '#fff' if a > 0.75 else 'inherit'
    return f'background: rgba({rgb[0]},{rgb[1]},{rgb[2]},{a:.2f}); color:{txt}'


def _matrix_table(result, M, fmt, scale, marks=None):
    marks = marks or {}
    det = result.detectors
    h = ['<div class="scroll"><table><tr><th class="l">fluor \\ detector</th>'
         + ''.join(f'<th>{html.escape(short_name(d))}</th>' for d in det) + '</tr>']
    for i, d in enumerate(det):
        h.append(f'<tr><th class="l">{html.escape(result.label(d))}</th>')
        for j in range(len(det)):
            if i == j:
                h.append('<td style="background:#e9ecef">-</td>')
            else:
                cls = marks.get((i, j), '')
                h.append(f'<td class="{cls}" style="{_color(M[i, j], scale)}">{fmt(M[i, j])}</td>')
        h.append('</tr>')
    return '\n'.join(h) + '</table></div>'


def write_report(result, out, sample_rows=None, sample_plots=None, inputs=None, log=None):
    """Write every output file into ``out`` and return the path of report.html."""
    out = Path(out)
    (out / 'plots').mkdir(parents=True, exist_ok=True)
    det, cfg = result.detectors, result.config
    P = cfg['plots']
    np.save(out / 'spillover.npy', result.S)
    write_csv(result.S, det, out / 'spillover.csv')
    write_flowjo_mtx(result.S, det, out / 'spillover.mtx', 'compspill', result.markers)
    if result.overrides:
        write_csv(result.S_estimated, det, out / 'spillover_before_overrides.csv')
    _write_tables(result, out, sample_rows)

    def save(png, name):
        (out / 'plots' / name).write_bytes(png)
        return png

    ctrl_png, fsc_png = {}, {}
    if P['enabled']:
        for c in result.controls:
            if log:
                log(f'  plotting {short_name(c.detector)}')
            ctrl_png[c.index] = save(plot_control(result, c, P['cofactor'], P['dpi'], P['max_points']),
                                     f'control_{short_name(c.detector)}.png')
            if c.bimodal and c.bimodal['bimodal']:
                fsc_png[c.index] = save(plot_fsc(c, P['dpi']), f'fsc_{short_name(c.detector)}.png')
    conv_png = plot_convergence(result.history, P['dpi'])

    e = html.escape
    h = [f'<!doctype html><html lang="en"><head><meta charset="utf-8">'
         f'<meta name="viewport" content="width=device-width, initial-scale=1">'
         f'<title>Compensation report</title><style>{CSS}</style></head><body><main>',
         '<h1>Compensation report</h1>',
         f'<p class="muted">compspill {__version__} &middot; {datetime.datetime.now():%Y-%m-%d %H:%M} &middot; '
         f'{len(det)} detectors &middot; '
         + (f'converged in {len(result.history)} iterations' if result.converged
            else f'<span class="flag">NOT converged after {len(result.history)} iterations</span>')
         + '</p>']
    if inputs:
        h.append('<div class="box small">' + '<br>'.join(f'<b>{e(k)}:</b> {e(str(v))}'
                                                       for k, v in inputs.items()) + '</div>')
    h.append('''<h2>How to read this report</h2><div class="box">
<p>The matrix below was estimated from your single-stain controls only. It is a <b>starting point that
needs your judgment</b>: check the warnings, look at the per-control plots (positives should rise straight
up), and, if stained samples were given, look at every flagged row of the sample check.
Values are shown in %: row = fluorochrome (its primary detector), column = detector it spills into.</p>
<p class="small muted">Outlined purple = manual override (reason listed below); dashed red = samples
disagree with this value.</p></div>''')

    # warnings
    h.append(f'<h2>Warnings ({len(result.warnings)})</h2>')
    if not result.warnings:
        h.append('<p>None.</p>')
    for w in result.warnings:
        h.append(f'<div class="warn"><b>{e(w["kind"])}</b> &middot; {e(w["subject"])}: {e(w["message"])}</div>')
    for i, png in fsc_png.items():
        h.append(_img(png, f'FSC histogram {det[i]}'))

    # matrix
    marks = {(det.index(o['source']), det.index(o['detector'])): 'ovr' for o in result.overrides}
    for r in sample_rows or []:
        if r['flagged']:
            marks[(r['i'], r['j'])] = (marks.get((r['i'], r['j']), '') + ' flg').strip()
    h.append('<h2>Spillover matrix (%)</h2>')
    h.append(_matrix_table(result, result.S * 100, lambda v: f'{v:.2f}', 50, marks))
    h.append('<p class="small muted">Files: <code>spillover.csv</code>, <code>spillover.mtx</code> (FlowJo: '
             'import in the compensation editor), <code>spillover.npy</code>. Convention: raw = true &times; S, '
             'compensated = raw &times; S<sup>-1</sup>.</p>')

    h.append(f'<h2>Manual overrides ({len(result.overrides)})</h2>')
    if result.overrides:
        h.append('<table><tr><th class="l">fluor</th><th class="l">detector</th><th>estimated</th>'
                 '<th>used</th><th class="l">reason</th></tr>')
        for o in result.overrides:
            h.append(f'<tr><td class="l">{e(result.label(o["source"]))}</td><td class="l">{e(o["detector"])}</td>'
                     f'<td>{o["estimated"]:.2%}</td><td>{o["value"]:.2%}</td><td class="l">{e(o["reason"])}</td></tr>')
        h.append('</table>')
    else:
        h.append('<p>None.</p>')

    # controls
    h.append('<h2>Controls: matching and populations</h2>')
    h.append(f'<p class="small">Unstained control: <b>{e(result.unstained or "none - each control&#39;s own negatives (Otsu split of its primary detector)")}</b>. '
             f'Positive = above the {cfg["gating"]["negative_percentile"]}th percentile of the unstained in the '
             f'primary detector; scatter gate &plusmn;{cfg["gating"]["scatter_k"]} robust SD around each population; '
             f'brightest {cfg["gating"]["top_trim_percent"]}% of positives dropped.</p>')
    h.append('<div class="scroll"><table><tr><th class="l">control file</th><th class="l">detector</th>'
             '<th class="l">matched by</th><th>events</th><th>neg</th><th>pos</th><th>off-scale</th>'
             '<th>median pos</th><th>median neg</th><th>pos/neg</th><th>FSC pos</th><th>FSC neg</th>'
             '<th class="l">particle rule</th></tr>')
    for c in result.controls:
        s = c.stats
        h.append(f'<tr><td class="l">{e(c.name)}</td><td class="l">{e(result.label(c.detector))}</td>'
                 f'<td class="l">{e(c.match_method)}</td><td>{s["n_events"]}</td><td>{s["n_neg"]}</td>'
                 f'<td>{s["n_pos"]}</td><td>{c.n_saturated}</td><td>{s["median_pos"]:.0f}</td>'
                 f'<td>{s["median_neg"]:.0f}</td><td>{s["ratio"]:.1f}</td><td>{s["fsc_pos"]:.0f}</td>'
                 f'<td>{s["fsc_neg"]:.0f}</td><td class="l">{e(c.particle_rule)}</td></tr>')
    h.append('</table></div>')

    # convergence + residuals
    h.append('<h2>Convergence and remaining error</h2><div class="grid">')
    h.append(_img(conv_png, 'convergence'))
    off = sorted(((abs(result.median_offsets[i, j]), i, j) for i in range(len(det)) for j in range(len(det))
                  if i != j), reverse=True)[:10]
    h.append('<div><p class="small">Largest remaining median offsets (positives vs negatives, % of the '
             'primary signal). Below ~1% is usually invisible.</p><table><tr><th class="l">control</th>'
             '<th class="l">detector</th><th>offset</th><th>slope</th></tr>')
    for v, i, j in off:
        h.append(f'<tr><td class="l">{e(short_name(det[i]))}</td><td class="l">{e(short_name(det[j]))}</td>'
                 f'<td>{result.median_offsets[i, j]:.2f}%</td><td>{result.residual_slopes[i, j]:.4f}</td></tr>')
    h.append('</table></div></div>')

    # spreading
    h.append('<h2>Spillover spreading matrix</h2><p class="small">How much a bright fluor (row) '
             'widens the negatives in another detector (column) after compensation, in units of '
             'sqrt(signal). Compensation cannot remove this spread; avoid dim markers in columns that '
             'receive a lot of spread. Rough rule: above ~5-10 is a problematic pair.</p>')
    h.append(_matrix_table(result, result.spreading, lambda v: f'{v:.1f}', 10))

    # per-control plots
    if ctrl_png:
        h.append('<h2>Compensated controls</h2><p class="small">Each panel: one compensated detector (x) '
                 'vs the compensated primary detector (y), arcsinh scale (cofactor '
                 f'{P["cofactor"]:g}). Black = gated positives, grey = negatives.</p>')
        for i, png in ctrl_png.items():
            h.append(f'<details><summary>{e(result.label(det[i]))}</summary>{_img(png, det[i])}</details>')

    # samples
    if sample_rows is not None:
        sc = cfg['samples']
        nflag = sum(r['flagged'] for r in sample_rows)
        h.append(f'<h2>Check on stained samples ({nflag} flagged)</h2>')
        h.append(f'''<div class="box small"><p>For each fluor i the samples are compensated with this matrix
except row i, so fluor i is left uncompensated. Among the events brightest in i (top {sc["top_percent"]}%
and above the {sc["negative_percentile"]}th percentile of the negatives) the robust slope of every other
compensated detector vs i gives the spill row of i <i>as the samples see it</i> (median over samples,
interquartile range across samples). Flagged: differs from the matrix by &gt; {sc["abs_tolerance"]:.0%}
(absolute) and &gt; {sc["rel_tolerance"]:.0%} (relative), with the matrix value outside the samples' IQR.</p>
<p><b>A flag is a prompt to look, not proof.</b> Biology also creates slopes: if cells bright for marker i
also express marker j, the samples show "spill" that compensation must not remove. Dead cells are bright in
everything (set <code>samples.viability_detector</code> to exclude them). Flags are most convincing when the
samples say ~0 and the controls clearly more: typical for a dim, cell-based control whose positive cells are
more autofluorescent than its negatives.</p>
<p>Viability exclusion: <b>{e(str(sc.get("viability_detector") or "off"))}</b>.</p></div>''')
        h.append('<div class="scroll"><table><tr><th class="l">fluor</th><th class="l">detector</th>'
                 '<th>matrix</th><th>controls</th><th>samples (median)</th><th>IQR</th><th>n</th>'
                 '<th class="l">flag</th></tr>')
        for r in sample_rows:
            h.append(f'<tr><td class="l">{e(result.label(r["source"]))}</td><td class="l">{e(short_name(r["detector"]))}</td>'
                     f'<td>{r["matrix"]:.2%}</td><td>{r["controls"]:.2%}</td><td>{r["samples"]:.2%}</td>'
                     f'<td>{r["samples_q1"]:.2%} &ndash; {r["samples_q3"]:.2%}</td><td>{r["n_samples"]}</td>'
                     f'<td class="l {"flag" if r["flagged"] else ""}">{e(r["hint"]) if r["flagged"] else ""}</td></tr>')
        h.append('</table></div>')
        if sample_plots:
            h.append('<p class="small">Flagged pairs: grey = all events, blue = bright in the source fluor; '
                     'red = slope the matrix predicts, dashed = slope the samples show.</p><div class="grid">')
            for (i, j), d in list(sample_plots.items())[:sc['max_plots']]:
                png = save(plot_sample_pair(result, i, j, d, P['cofactor'], P['dpi']),
                           f'samples_{short_name(det[i])}_to_{short_name(det[j])}.png')
                h.append(_img(png, f'{det[i]} to {det[j]}'))
            h.append('</div>')

    cfg_dump = {k: v for k, v in cfg.items() if k != 'override'}
    h.append('<h2>Method and settings</h2><div class="box small"><p>AutoSpill-style estimation '
             '(Roca CP et al. 2021, <i>Nat Commun</i> 12:2890): gate each control, start from the classic '
             'median estimate, then iteratively correct the matrix with robust (Huber) slopes of each '
             'compensated detector vs the compensated primary until the largest slope is below '
             f'{cfg["iteration"]["tolerance"]:g}. Spreading matrix after Nguyen et al. 2013, '
             '<i>Cytometry A</i> 83:306.</p>'
             f'<pre>{e(json.dumps(cfg_dump, indent=1, default=str))}</pre></div>')
    h.append('</main></body></html>')
    path = out / 'report.html'
    path.write_text('\n'.join(h))
    return path
