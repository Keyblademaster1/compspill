
from compspill import estimate_spillover, sample_check
from compspill.synthetic import make_sample, make_unstained, standard_set


def _samples(det, k=3):
    return [(f's{i}', make_sample(seed=10 + i).matrix(det)) for i in range(k)]


def test_sample_check_flags_overestimating_dim_control():
    controls, _, S = standard_set(dim_fitc=True, two_particle=False, offscale=False)
    r = estimate_spillover(list(controls.values()), unstained=make_unstained(kind='cells'))
    rows, plots = sample_check(r, _samples(r.detectors))
    flagged = {(x['source'], x['detector']) for x in rows if x['flagged']}
    assert ('FITC-A', 'BV421-A') in flagged          # true spill 0, control says ~30 %
    row = [x for x in rows if (x['source'], x['detector']) == ('FITC-A', 'BV421-A')][0]
    assert abs(row['samples']) < 0.01 and 'samples ~0' in row['hint']
    assert all(x['source'] == 'FITC-A' for x in rows if x['flagged'])
    assert (0, 4) in plots


def test_sample_check_agrees_with_good_matrix(clean_set):
    controls, un, S = clean_set
    r = estimate_spillover(list(controls.values()), unstained=un)
    rows, _ = sample_check(r, _samples(r.detectors))
    assert not any(x['flagged'] for x in rows)
    coexpressed = {(1, 2), (2, 1), (4, 1), (1, 4)}   # cell types in make_sample: PE+PE-Cy7, BV421+PE
    for x in rows:
        if (x['i'], x['j']) not in coexpressed:
            assert abs(x['samples'] - S[x['i'], x['j']]) < 0.01
    # biology shows up as extra 'spill' (here below the flag thresholds)
    bv421_pe = [x for x in rows if (x['i'], x['j']) == (4, 1)][0]
    assert bv421_pe['samples'] > S[4, 1] + 0.005
