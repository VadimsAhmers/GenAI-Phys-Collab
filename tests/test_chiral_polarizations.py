"""Two circular channels: basis signs, single solve, failure and CSV resume."""
import csv
import math
import sys
from types import SimpleNamespace
from zipfile import ZipFile
import xml.etree.ElementTree as ET

import numpy as np
import pytest

from src.opt import ChiralSolver, FakeChiralSolver, ChiralReflectanceObjective, CHIRAL_REFERENCE
from src.opt.solvers import _ll_expression_from_rr
from scripts.sweep_freq import FIELDS, geom_key, load_done, open_csv, _scan
from scripts import sweep_freq


def shipped_rr_expression():
    with ZipFile('models/chiral.mph') as archive:
        root = ET.fromstring(archive.read('dmodel.xml'))
    node = next(n for n in root.iter('NumericalFeature') if n.get('name') == 'r_RR')
    prop = next(p for p in node.iter('propertyValue') if p.get('name') == 'p:expr')
    return prop.get('valueMatrix').split("'", 1)[1][:-1].replace('\\', '')


def evaluate_expression(expression, matrix):
    # Evaluate only the known expression read from our model/test generator.
    scope = dict(abs=abs, freq=5.8, setval=lambda *args: None,
                 withsol=lambda solution, value, selection: value,
                 ewfd=SimpleNamespace(S11=matrix[0, 0], S12=matrix[0, 1],
                                      S21=matrix[1, 0], S22=matrix[1, 1]))
    return eval(expression.replace('1i', '1j').replace('^', '**'), {'__builtins__': {}}, scope)


def test_opposite_basis_matches_matrix_projection_and_is_not_conjugated_rr():
    matrix = np.array([[0.2 + 0.1j, 0.2 - 0.15j], [0.1 + 0.1j, -0.1 + 0.3j]])
    rr = shipped_rr_expression()
    ll = _ll_expression_from_rr(rr)
    e_r = np.array([1, -1j]) / np.sqrt(2)
    e_l = e_r.conj()
    # Reflected-wave convention of the shipped model uses these transpose
    # projections, since the outgoing wave propagates in the opposite direction.
    assert evaluate_expression(rr, matrix) == pytest.approx(abs(e_r @ matrix @ e_r)**2)
    assert evaluate_expression(ll, matrix) == pytest.approx(abs(e_l @ matrix @ e_l)**2)
    assert evaluate_expression(ll, matrix) != pytest.approx(evaluate_expression(rr, matrix))


def test_unknown_model_convention_is_rejected():
    with pytest.raises(ValueError, match='Unrecognized'):
        _ll_expression_from_rr('abs(ewfd.S11)^2')


class EvaluationNode:
    def __init__(self, fail=False):
        self.original = [shipped_rr_expression()]
        self.expression = self.original
        self.fail = fail
        self.java = self

    def property(self, name, value=None):
        assert name == 'expr'
        if value is None:
            return self.expression
        self.expression = value

    def getReal(self):
        if self.fail:
            raise RuntimeError('LL evaluation failed')
        return [[0.71]]


class Model:
    def __init__(self, node):
        self.node = node

    def __truediv__(self, name):
        return self.node if name == 'r_RR' else self


class CountingSolver(ChiralSolver):
    def __init__(self, node):
        super().__init__()
        self._model = Model(node)
        self.calls = 0

    def __call__(self, params):
        self.calls += 1
        return 0.91


@pytest.mark.parametrize('fail', [False, True])
def test_one_solve_and_expression_restored_even_on_failure(fail):
    node = EvaluationNode(fail)
    solver = CountingSolver(node)
    if fail:
        with pytest.raises(RuntimeError, match='LL evaluation'):
            solver.polarizations({})
    else:
        assert solver.polarizations({}) == {'abs_r_RR_sq': 0.91, 'abs_r_LL_sq': 0.71}
    assert solver.calls == 1
    assert node.expression == node.original


def test_objective_records_ll_without_changing_rr_score():
    solver = FakeChiralSolver()
    params = {**CHIRAL_REFERENCE, 'eps_r': 7.1, 'freq_ghz': 5.8}
    single = ChiralReflectanceObjective(solver).evaluate(params)
    both = ChiralReflectanceObjective(solver, both_polarizations=True).evaluate(params)
    assert both.score == single.score
    assert both.aux['abs_r_LL_sq'] != both.aux['abs_r_RR_sq']
    assert 'LL is recorded but is not part of this score' in both.feedback


def test_ll_evaluation_error_is_not_recorded_as_zero_reflection():
    obj = ChiralReflectanceObjective(CountingSolver(EvaluationNode(fail=True)),
                                    both_polarizations=True)
    with pytest.raises(RuntimeError, match='LL evaluation'):
        obj.evaluate({})


def test_csv_mode_switch_refused_without_modifying_original(tmp_path):
    path = tmp_path / 'old.csv'
    fh, writer = open_csv(path)
    fh.close()
    original = path.read_bytes()
    for operation in (load_done, open_csv):
        with pytest.raises(ValueError, match='new --out'):
            operation(path, both_polarizations=True)
    assert path.read_bytes() == original


def test_resume_requires_both_channels_and_never_repeats_complete_points(tmp_path):
    path = tmp_path / 'both.csv'
    params = {**CHIRAL_REFERENCE, 'eps_r': 7.1}

    class Counter(FakeChiralSolver):
        calls = 0

        def polarizations(self, params):
            self.calls += 1
            return super().polarizations(params)

    solver = Counter()
    obj = ChiralReflectanceObjective(solver, both_polarizations=True)
    fh, writer = open_csv(path, True)
    try:
        assert not _scan(obj, params, [5.77, 5.8], 'base', 1.0, {}, writer, fh, {}, 'coarse')
    finally:
        fh.close()
    assert solver.calls == 2
    done = load_done(path, True)
    entry = done[geom_key(params)]
    assert set(entry['values']) == set(entry['values_ll']) == {5.77, 5.8}
    assert not _scan(obj, params, [5.77, 5.8], 'base', 1.0, {}, None, None, done, 'coarse')
    assert solver.calls == 2

    # A row whose RR is present but LL is missing must be recalculated.
    with path.open(newline='') as fh:
        rows = list(csv.DictReader(fh))
    rows[0]['abs_r_LL_sq'] = ''
    with path.open('w', newline='') as fh:
        writer = csv.DictWriter(fh, fieldnames=FIELDS + ['abs_r_LL_sq'])
        writer.writeheader()
        writer.writerows(rows)
    done = load_done(path, True)
    assert set(done[geom_key(params)]['values']) == {5.8}


def test_sweep_refines_both_distinct_peaks_and_resumes_without_solves(tmp_path, monkeypatch):
    class SeparatedPeaks:
        calls = 0

        def polarizations(self, params):
            type(self).calls += 1
            f = params['freq_ghz']
            return {'abs_r_RR_sq': 0.9 * math.exp(-((f - 5.65) / 0.1)**2),
                    'abs_r_LL_sq': 0.85 * math.exp(-((f - 5.85) / 0.1)**2)}

    path = tmp_path / 'separated.csv'
    monkeypatch.setattr(sweep_freq, 'FakeChiralSolver', SeparatedPeaks)
    monkeypatch.setattr(sys, 'argv', ['sweep_freq', '--solver', 'fake', '--both-polarizations',
                                    '--base', 'eps_r=7.1', '--band', '5.5:6.0',
                                    '--step', '0.1', '--refine', '10', '--out', str(path)])
    assert sweep_freq.main() == 0
    with path.open(newline='') as fh:
        rows = list(csv.DictReader(fh))
    by_f = {float(r['freq_ghz']): r for r in rows}
    assert {5.65, 5.85} <= by_f.keys()
    assert float(by_f[5.65]['abs_r_RR_sq']) == pytest.approx(0.9)
    assert float(by_f[5.85]['abs_r_LL_sq']) == pytest.approx(0.85)
    assert len(rows) == len(by_f) == SeparatedPeaks.calls
    before = path.read_bytes()
    assert sweep_freq.main() == 0
    assert path.read_bytes() == before
    assert SeparatedPeaks.calls == len(rows)
