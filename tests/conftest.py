import pytest

from compspill.synthetic import standard_set, write_fcs


@pytest.fixture(scope='session')
def clean_set():
    controls, un, S = standard_set(two_particle=False, offscale=False)
    return controls, un, S


@pytest.fixture
def fcs_folder(tmp_path):
    """Controls + unstained written as FCS files with lab-style names."""
    controls, un, S = standard_set(two_particle=False)
    names = {'FITC-A': 'Compensation Controls_FITC Stained Control.fcs',
             'PE-A': 'Compensation Controls_PE Stained Control.fcs',
             'PE-Cy7-A': 'Compensation Controls_PE-Cy7 Stained Control.fcs',
             'APC-A': 'apc single stain.FCS',
             'BV421-A': 'Compensation Controls_BV421 Stained Control.fcs'}
    for ev in controls.values():
        det = [d for d in names if ev.name.startswith(d[:-2] + ' ')][0]
        write_fcs(ev, tmp_path / names[det])
    write_fcs(un, tmp_path / 'Compensation Controls_Unstained Control.fcs')
    return tmp_path, S
