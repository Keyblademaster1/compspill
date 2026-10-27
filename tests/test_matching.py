import numpy as np
import pytest

from compspill.matching import MatchError, by_filename, by_signal, resolve_detector, tokens

DET = ['Alexa Fluor 488-A', 'PE-A', 'PE-Cy7-A', 'PE-Cy5-5-A', 'PE-CF594-A', 'APC-A', 'APC-Cy7-A',
       'BV421-A', 'PerCP-Cy5-5-A']


def test_tokens():
    assert tokens('Alexa Fluor 488-A') == ['af', '488', 'a']
    assert tokens('AF488') == tokens('alexa-fluor 488') == ['af', '488']
    assert tokens('PE-Cy5.5') == tokens('PE-Cy5-5') == ['pe', 'cy', '5', '5']


@pytest.mark.parametrize('name, det', [
    ('Compensation Controls_Alexa Fluor 488 Stained Control.fcs', 'Alexa Fluor 488-A'),
    ('af488.fcs', 'Alexa Fluor 488-A'),
    ('Compensation Controls_PE Stained Control.fcs', 'PE-A'),
    ('Compensation Controls_PE-Cy7 Stained Control.fcs', 'PE-Cy7-A'),
    ('pe cy5.5 beads.fcs', 'PE-Cy5-5-A'),
    ('PE-CF594-A.fcs', 'PE-CF594-A'),
    ('APC_single.fcs', 'APC-A'),
    ('APC-Cy7 stained.fcs', 'APC-Cy7-A'),
    ('PerCP-Cy5.5 CD3.fcs', 'PerCP-Cy5-5-A'),
])
def test_filename_matching(name, det):
    assert by_filename(name, DET)[0] == det


def test_marker_fallback_and_none():
    assert by_filename('CD3 control.fcs', DET, {'BV421-A': 'CD3'}) == ('BV421-A', 'marker name')
    assert by_filename('mystery.fcs', DET) == (None, None)


def test_ambiguous_filename_refused():
    with pytest.raises(MatchError):
        by_filename('PE and BV421 mix.fcs', DET)


def test_resolve_detector():
    assert resolve_detector('AF488', DET) == 'Alexa Fluor 488-A'
    assert resolve_detector('pe-cy7', DET) == 'PE-Cy7-A'
    assert resolve_detector('CD3', DET, {'BV421-A': 'CD3'}) == 'BV421-A'
    with pytest.raises(MatchError):
        resolve_detector('FITC', DET)


def test_signal_fallback():
    rng = np.random.default_rng(0)
    X = rng.normal(100, 10, (1000, 3))
    X[:500, 1] += 5000
    X[:500, 2] += 1000
    assert by_signal(X, np.full(3, 100.), ['a', 'b', 'c'])[0] == 'b'
    X[:500, 2] += 3500   # now nearly as bright as b -> refuse
    with pytest.raises(MatchError):
        by_signal(X, np.full(3, 100.), ['a', 'b', 'c'])
