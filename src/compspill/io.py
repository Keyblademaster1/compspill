"""Reading FCS files and reading/writing spillover matrices (CSV, FlowJo .mtx, .npy).

Convention used everywhere in compspill
---------------------------------------
The spillover matrix ``S`` has one row per fluorochrome (named by its *primary*
detector) and one column per detector::

    raw = true @ S          compensated = raw @ inv(S)

``S[i, i] == 1`` and ``S[i, j]`` is the fraction of fluor i's primary-detector
signal that also lands in detector j.  This is the layout of the FCS
``$SPILLOVER`` keyword and of FlowJo's spillover matrices.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

# channels that are never fluorescence detectors
_NON_FLUOR = re.compile(r'(^|[^a-z])(fsc|ssc|time)([^a-z]|$)', re.I)


@dataclass
class Events:
    """One FCS file in memory: channel name -> values, plus the bits of metadata we use.

    ``ranges`` holds $PnR (the top of the measurement scale) per channel; events at
    or near it are off-scale and their true value is unknown.
    """
    name: str
    channels: dict
    ranges: dict = field(default_factory=dict)
    spill_detectors: list | None = None
    path: str | None = None

    @property
    def n(self):
        return len(next(iter(self.channels.values()))) if self.channels else 0

    def matrix(self, detectors):
        missing = [d for d in detectors if d not in self.channels]
        if missing:
            raise KeyError(f'{self.name}: channels {missing} not found (has {list(self.channels)})')
        return np.column_stack([np.asarray(self.channels[d], float) for d in detectors])

    def subsample(self, max_events, rng):
        if max_events is None or self.n <= max_events:
            return self
        idx = np.sort(rng.choice(self.n, int(max_events), replace=False))
        return Events(self.name, {k: v[idx] for k, v in self.channels.items()}, self.ranges,
                      self.spill_detectors, self.path)


def parse_spill(text):
    """Parse a $SPILLOVER / $SPILL string -> (detector names, matrix)."""
    parts = [p.strip() for p in text.strip().split(',')]
    n = int(parts[0])
    names = parts[1:n + 1]
    vals = np.array([float(v) for v in parts[n + 1:n + 1 + n * n]]).reshape(n, n)
    return names, vals


def load_fcs(path) -> Events:
    """Read an FCS file with flowio.  Values are the stored (linear) event values."""
    import flowio
    try:
        f = flowio.FlowData(str(path))
    except Exception as e:
        raise ValueError(f'cannot read FCS file {path}: {e}') from e
    a = np.reshape(np.asarray(f.events, dtype=float), (-1, f.channel_count))
    channels = dict(zip(f.pnn_labels, a.T))
    ranges = {}
    for k, ch in enumerate(f.pnn_labels):
        try:
            ranges[ch] = float(f.text[f'p{k + 1}r'])
        except (KeyError, ValueError):
            pass
    spill = None
    for key in ('spillover', 'spill', 'comp'):
        if key in f.text:
            try:
                spill = parse_spill(f.text[key])[0]
                break
            except (ValueError, IndexError):
                pass
    return Events(Path(path).name, channels, ranges, spill, str(path))


def as_events(x) -> Events:
    return x if isinstance(x, Events) else load_fcs(x)


def fluorescence_channels(ev: Events):
    """Detectors to compensate: $SPILLOVER order if present, else all '-A' channels
    that are not scatter or time."""
    if ev.spill_detectors:
        return list(ev.spill_detectors)
    return [c for c in ev.channels if c.upper().endswith('-A') and not _NON_FLUOR.search(c)]


def find_scatter(ev: Events, kind, preferred=None):
    """Name of the forward/side scatter area channel ('FSC'/'SSC'), or None.
    Prefers exactly 'FSC-A' / 'SSC-A' (e.g. not 'SP SSC-A')."""
    if preferred:
        if preferred not in ev.channels:
            raise KeyError(f'{ev.name}: scatter channel {preferred!r} not found')
        return preferred
    if f'{kind}-A' in ev.channels:
        return f'{kind}-A'
    cands = [c for c in ev.channels if kind in c.upper() and c.upper().endswith('-A')]
    cands = cands or [c for c in ev.channels if c.upper().startswith(kind)]
    return cands[0] if cands else None


# ---------------------------------------------------------------- matrix files

def write_csv(S, detectors, path, row_labels=None):
    """Rows = fluorochrome (primary detector), columns = detectors."""
    rows = row_labels or detectors
    with open(path, 'w') as fh:
        fh.write(',' + ','.join(detectors) + '\n')
        for r, vals in zip(rows, S):
            fh.write(r + ',' + ','.join(f'{v:.6f}' for v in vals) + '\n')


def read_csv(path):
    """Read a matrix written by write_csv -> (detectors, S)."""
    lines = [l.rstrip('\n') for l in open(path) if l.strip()]
    detectors = lines[0].split(',')[1:]
    S = np.array([[float(v) for v in l.split(',')[1:]] for l in lines[1:]])
    return detectors, S


def _stem(detector):
    return re.sub(r'-A$', '', detector)


def write_flowjo_mtx(S, detectors, path, name='compspill', markers=None):
    """Write S as a FlowJo-importable gatingML spillover matrix (.mtx).

    With ``markers`` (detector -> marker) the compensated parameters get FlowJo
    infixes like ``Comp-CD3-BB700``, which is how FlowJo itself stores labels.
    """
    markers = markers or {}

    def infix(d):
        return f'Comp-{markers[d]}-{_stem(d)}' if markers.get(d) else f'Comp-{_stem(d)}'

    out = ['<?xml version="1.0" encoding="UTF-8"?>',
           ' <gating:gatingML>',
           f'   <transforms:spilloverMatrix spectral="0"  prefix="Comp-"  name="{name}"  editable="1"  '
           f'matrixType="userDefined"  version="FlowJo-10.10.1"  status="FINALIZED"  suffix="" >',
           '     <data-type:parameters>']
    for d in detectors:
        out.append(f'       <data-type:parameter data-type:name="{d}"  userProvidedCompInfix="{infix(d)}" />')
    out.append('     </data-type:parameters>')
    for i, d in enumerate(detectors):
        out.append(f'     <transforms:spillover data-type:parameter="{d}"  userProvidedCompInfix="{infix(d)}" >')
        for j, d2 in enumerate(detectors):
            out.append(f'       <transforms:coefficient data-type:parameter="{d2}"  '
                       f'transforms:value="{S[i, j]:.10g}" />')
        out.append('     </transforms:spillover>')
    out += ['   </transforms:spilloverMatrix>', ' </gating:gatingML>']
    Path(path).write_text('\n'.join(out) + '\n')


def read_flowjo_mtx(path, detectors=None):
    """Parse a FlowJo .mtx -> (detectors, S).  If ``detectors`` is given, S is
    reordered to that order (missing rows/columns stay identity)."""
    txt = Path(path).read_text()
    order = re.findall(r'<data-type:parameter data-type:name="([^"]+)"', txt)
    detectors = list(detectors) if detectors is not None else order
    S = np.eye(len(detectors))
    for m in re.finditer(r'<transforms:spillover data-type:parameter="([^"]+)"(.*?)</transforms:spillover>',
                         txt, flags=re.S):
        if m.group(1) not in detectors:
            continue
        i = detectors.index(m.group(1))
        for cm in re.finditer(r'data-type:parameter="([^"]+)"\s+transforms:value="([^"]+)"', m.group(2)):
            if cm.group(1) in detectors:
                S[i, detectors.index(cm.group(1))] = float(cm.group(2))
    return detectors, S
