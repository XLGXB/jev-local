import torch
from decision_ft.data import atomize
from decision_ft.make_demo import rows
from decision_ft.metrics import report
from decision_ft.calibrate import fit_temperature


def test_calibration_fit_does_not_increase_fit_nll():
    atoms = atomize(rows(12))
    logits = []
    for i, a in enumerate(atoms):
        z = torch.zeros(len(a.keys))
        z[a.label if i % 3 else (a.label + 1) % len(a.keys)] = 8.
        logits.append(z)
    temperatures, _ = fit_temperature(atoms, logits, min_count=2)
    assert report(atoms, logits, temperatures)['all']['nll_target'] <= report(atoms, logits)['all']['nll_target'] + 1e-5


def test_unverified_teacher_not_counted_as_empirical_calibration():
    atoms = atomize(rows(2))
    for atom in atoms:
        atom.label_source = 'teacher_unverified'
    result = report(atoms, [torch.zeros(len(a.keys)) for a in atoms])
    assert result['all']['n_verified'] == 0
    assert 'ece_verified_15_bins' not in result['all']


def test_no_acceptance_does_not_report_zero_error():
    atoms = atomize(rows(2))
    result = report(atoms, [torch.zeros(len(a.keys)) for a in atoms])
    assert result['all']['risk_coverage_verified']['0.99']['error_rate'] is None
