import numpy as np

from compspill import estimate_spillover, load_fcs, read_csv, read_flowjo_mtx, write_csv, write_flowjo_mtx
from compspill.cli import main
from compspill.synthetic import DETECTORS, make_sample, write_fcs


def test_mtx_and_csv_round_trip(tmp_path):
    det = ['Alexa Fluor 488-A', 'PE-A', 'BV421-A']
    S = np.array([[1, .12345678, -.001], [.02, 1, 0], [.3, 1e-5, 1]])
    write_flowjo_mtx(S, det, tmp_path / 'm.mtx', 'test', markers={'PE-A': 'CD3'})
    d2, S2 = read_flowjo_mtx(tmp_path / 'm.mtx')
    assert d2 == det and np.allclose(S, S2, atol=1e-12)
    assert 'Comp-CD3-PE' in (tmp_path / 'm.mtx').read_text()
    # reorder on read
    _, S3 = read_flowjo_mtx(tmp_path / 'm.mtx', detectors=det[::-1])
    assert np.allclose(S3, S[::-1, ::-1])
    write_csv(S, det, tmp_path / 'm.csv')
    d4, S4 = read_csv(tmp_path / 'm.csv')
    assert d4 == det and np.allclose(S, S4, atol=1e-6)


def test_fcs_round_trip(tmp_path):
    ev = make_sample(n=500)
    write_fcs(ev, tmp_path / 's.fcs')
    back = load_fcs(tmp_path / 's.fcs')
    assert back.spill_detectors == DETECTORS
    assert back.ranges['PE-A'] == 262144
    assert np.allclose(back.channels['PE-A'], ev.channels['PE-A'].astype(np.float32))


def test_folder_input_matches_by_filename(fcs_folder):
    folder, S = fcs_folder
    r = estimate_spillover(folder)
    assert r.unstained == 'Compensation Controls_Unstained Control.fcs'
    assert [c.detector for c in r.controls] == DETECTORS
    assert {c.match_method for c in r.controls} == {'file name'}
    assert np.abs(r.S - S).max() < 0.003


def test_config_match_table_wins(fcs_folder, tmp_path):
    folder, _ = fcs_folder
    (folder / 'apc single stain.FCS').rename(folder / 'tube 7.fcs')
    cfg = tmp_path / 'c.toml'
    cfg.write_text('[match]\n"tube 7.fcs" = "APC-A"\n')
    r = estimate_spillover(folder, config=cfg)
    assert r.controls[3].match_method == 'config [match]'


def test_cli_end_to_end(fcs_folder, tmp_path):
    folder, S = fcs_folder
    samples = tmp_path / 'samples'
    samples.mkdir()
    for k in range(2):
        write_fcs(make_sample(n=20000, seed=k), samples / f's{k}.fcs')
    out = tmp_path / 'out'
    cfg = tmp_path / 'c.toml'
    cfg.write_text('[markers]\n"PE-A" = "CD3"\n[[override]]\nsource = "FITC"\ndetector = "APC"\n'
                   'value = 0.0\nreason = "unit test"\n')
    rc = main(['run', '--controls', str(folder), '--out', str(out), '--config', str(cfg),
               '--samples', str(samples), '-q'])
    assert rc == 0
    for f in ['spillover.csv', 'spillover.mtx', 'spillover.npy', 'residuals.csv', 'spreading.csv',
              'controls.csv', 'sample_check.csv', 'report.html', 'plots/control_PE.png']:
        assert (out / f).exists(), f
    assert np.abs(np.load(out / 'spillover.npy') - S).max() < 0.003
    html = (out / 'report.html').read_text()
    assert 'unit test' in html and 'CD3 (PE)' in html and 'data:image/png;base64' in html


def test_cli_reports_match_errors(tmp_path, capsys):
    write_fcs(make_sample(n=200), tmp_path / 'mystery.fcs')
    rc = main(['run', '--controls', str(tmp_path), '--out', str(tmp_path / 'o')])
    assert rc == 2
    assert 'error' in capsys.readouterr().err
