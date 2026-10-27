import numpy as np
import pytest

from compspill import estimate_spillover, load_config
from compspill.config import ConfigError
from compspill.matching import MatchError
from compspill.synthetic import DETECTORS, TRUE_S, make_control, make_unstained, standard_set


def kinds(r):
    return {w['kind'] for w in r.warnings}


def test_recovers_known_matrix(clean_set):
    controls, un, S = clean_set
    r = estimate_spillover(list(controls.values()), unstained=un)
    assert r.converged
    assert np.abs(r.S - S).max() < 0.003
    assert np.allclose(np.diag(r.S), 1)
    assert np.abs(r.residual_slopes).max() < 1e-3
    assert not kinds(r)


def test_without_unstained_uses_own_negatives(clean_set):
    controls, _, S = clean_set
    r = estimate_spillover(list(controls.values()), unstained=False)
    assert r.unstained is None
    assert np.abs(r.S - S).max() < 0.003


def test_dict_input_skips_matching(clean_set):
    controls, un, S = clean_set
    by_det = {d: ev for d, ev in zip(DETECTORS, controls.values())}
    r = estimate_spillover(by_det, unstained=un)
    assert [c.match_method for c in r.controls] == ['given'] * len(DETECTORS)
    assert np.abs(r.S - S).max() < 0.003


def test_two_particle_warning_and_particle_selection():
    controls, un, S = standard_set(offscale=False)
    r = estimate_spillover(list(controls.values()), unstained=un)
    w = [w for w in r.warnings if w['kind'] == 'two particle populations']
    assert len(w) == 1 and w[0]['subject'] == 'APC'
    assert 'fsc_threshold' in w[0]['message']
    # keep only the small particles: their (different) spectrum is recovered
    r2 = estimate_spillover(list(controls.values()), unstained=un,
                            config={'particles': {'APC': {'fsc_threshold': 30000, 'keep': 'below'}}})
    assert abs(r2.S[3, 2] - 0.35) < 0.01
    assert 'Handled by config' in [w for w in r2.warnings if w['kind'] == 'two particle populations'][0]['message']
    # ... or only the beads
    r3 = estimate_spillover(list(controls.values()), unstained=un,
                            config={'particles': {'APC-A': {'fsc_threshold': 30000, 'keep': 'above'}}})
    assert abs(r3.S[3, 2] - TRUE_S[3, 2]) < 0.005


def test_offscale_and_few_positives_warnings():
    controls, un, _ = standard_set(two_particle=False)          # PE has 150 off-scale events
    controls['FITC stained.fcs'] = make_control(0, n=600, frac_pos=0.2)
    r = estimate_spillover(list(controls.values()), unstained=un)
    assert {'off-scale', 'few positives'} <= kinds(r)
    pe = r.controls[1]
    assert pe.n_saturated >= 150 and not (pe.pos & (pe.X[:, 1] >= 0.99 * 262144)).any()


def test_dim_control_warning():
    controls, _, _ = standard_set(dim_fitc=True, two_particle=False, offscale=False)
    r = estimate_spillover(list(controls.values()), unstained=make_unstained(kind='cells'))
    dim = [w for w in r.warnings if w['kind'] == 'dim control']
    assert [w['subject'] for w in dim] == ['FITC']
    assert r.S[0, 4] > 0.1     # the overestimate the warning is about


def test_override_applied_with_reason(clean_set):
    controls, un, _ = clean_set
    cfg = {'override': [{'source': 'FITC', 'detector': 'BV421-A', 'value': 0.0, 'reason': 'checked on samples'}]}
    r = estimate_spillover(list(controls.values()), unstained=un, config=cfg)
    assert r.S[0, 4] == 0.0
    assert r.overrides[0]['reason'] == 'checked on samples'
    assert r.S_estimated[0, 4] == pytest.approx(r.S[0, 4] + r.overrides[0]['estimated'])
    with pytest.raises(ConfigError):
        load_config({'override': [{'source': 'FITC', 'detector': 'PE', 'value': 0, 'reason': ' '}]})


def test_config_rejects_typos():
    with pytest.raises(ConfigError):
        load_config({'gating': {'scater_k': 3}})
    with pytest.raises(ConfigError):
        load_config({'particles': {'APC': {'fsc_threshold': 1, 'keep': 'middle'}}})


def test_missing_and_duplicate_controls(clean_set):
    controls, un, _ = clean_set
    evs = list(controls.values())
    with pytest.raises(MatchError, match='no control for'):
        estimate_spillover(evs[:-1], unstained=un)
    dup = make_control(1, name='PE again.fcs')
    with pytest.raises(MatchError, match='two controls'):
        estimate_spillover(evs + [dup], unstained=un)
    # a subset of detectors is fine when asked for explicitly
    r = estimate_spillover(evs[:-1], unstained=un, detectors=DETECTORS[:-1])
    assert r.S.shape == (4, 4)
