"""Which control file belongs to which detector?

Matching is by *detector* (the channel that measures the fluor), never by the fluor
name a person happened to type.  Names are compared as token sequences, so
'Alexa Fluor 488', 'AF488', 'af-488' and 'Alexa Fluor 488-A' are all the same,
while 'PE' does not match inside 'PE-Cy7' (the longest match wins).
"""
from __future__ import annotations

import re

import numpy as np


class MatchError(ValueError):
    """Raised when controls cannot be matched to detectors unambiguously."""


def tokens(s):
    """'Alexa Fluor 488-A' -> ['af', '488'];  'PE-Cy5.5' -> ['pe', 'cy', '5', '5']."""
    s = re.sub(r'\.fcs$', '', s, flags=re.I).lower()
    s = re.sub(r'alexa[\s_-]*fluor|alexa', ' af ', s)
    return re.findall(r'[a-z]+|[0-9]+', s)


def detector_tokens(detector):
    return tokens(re.sub(r'-[AHW]$', '', detector))


def _spans(hay, needle):
    k = len(needle)
    return [(s, s + k) for s in range(len(hay) - k + 1) if k and hay[s:s + k] == needle]


def match_name(name, candidates):
    """Detectors whose token sequence occurs in ``name``.

    ``candidates`` maps detector -> list of token lists (detector name, marker,
    ...).  Matches lying inside a longer match are dropped, so for a file called
    'PE-Cy7 stained' only PE-Cy7 is returned, not PE.
    """
    hay = tokens(name)
    hits = [(s, e, det) for det, toks in candidates.items() for t in toks for s, e in _spans(hay, t)]
    keep = {det for s, e, det in hits
            if not any(s2 <= s and e <= e2 and (s2, e2) != (s, e) for s2, e2, _ in hits)}
    return sorted(keep)


def resolve_detector(name, detectors, markers=None):
    """Map a user-written name (config key) to one detector, tolerant to case, '-A'
    and 'Alexa Fluor' spelling, or to a marker name.  Raises MatchError."""
    if name in detectors:
        return name
    t = tokens(re.sub(r'-[AHW]$', '', name))
    hits = [d for d in detectors if detector_tokens(d) == t]
    if not hits and markers:
        hits = [d for d, m in markers.items() if d in detectors and tokens(m) == t]
    if len(hits) != 1:
        raise MatchError(f'{name!r} does not name exactly one detector '
                         f'({"ambiguous: " + ", ".join(hits) if hits else "no match"}); '
                         f'detectors are: {", ".join(detectors)}')
    return hits[0]


def by_filename(name, detectors, markers=None):
    """(detector or None, method).  Raises MatchError if the name fits several."""
    hits = match_name(name, {d: [detector_tokens(d)] for d in detectors})
    if len(hits) > 1:
        raise MatchError(f'Control {name!r} matches several detectors by name: {", ".join(hits)}. '
                         f'Add it to the [match] table of the config file.')
    if hits:
        return hits[0], 'file name'
    if markers:
        hits = match_name(name, {d: [tokens(m)] for d, m in markers.items() if d in detectors and m})
        if len(hits) > 1:
            raise MatchError(f'Control {name!r} matches several markers: {", ".join(hits)}. '
                             f'Add it to the [match] table of the config file.')
        if hits:
            return hits[0], 'marker name'
    return None, None


def by_signal(X, baseline, detectors, taken=(), margin=0.8):
    """Fallback: the detector where the control is brightest above background.

    The primary detector of a fluor is, by definition, the one that sees most of
    its light, so the background-subtracted 99th percentile should be highest
    there.  Refuses (MatchError) when the runner-up is within ``margin`` of the
    winner, because then the guess would be a coin flip.
    """
    score = np.percentile(X, 99, axis=0) - baseline
    score = np.where([d in taken for d in detectors], -np.inf, score)
    order = np.argsort(-score)
    best, second = score[order[0]], score[order[1]] if len(order) > 1 else -np.inf
    if not np.isfinite(best) or best <= 0 or second > margin * best:
        raise MatchError('ambiguous by signal: brightest detectors are '
                         f'{detectors[order[0]]} ({best:.0f}) and '
                         f'{detectors[order[1]] if len(order) > 1 else "-"} ({second:.0f})')
    runner = f', runner-up {detectors[order[1]]} at {second / best:.0%}' if np.isfinite(second) else ''
    return detectors[order[0]], f'signal{runner}'
