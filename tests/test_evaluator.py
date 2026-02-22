"""Test the evaluator pipeline against known results from Bartel et al. (2019).

Ground truth comes from reproduce_evidence/evidence1_abx3_classification.py,
which uses the paper's pre-computed predictions on 576 ABX3 compounds:
  - Goldschmidt t: 73.8% overall, FPR=50.6%
  - New τ:         91.7% overall, FPR=10.6%, test=94%
"""

from pathlib import Path

import pytest

from evaluator import evaluate_candidate, load_dataset

DATA_PATH = Path(__file__).parent.parent / "perovskite-stability" / "TableS1.csv"

GOLDSCHMIDT_T = """\
def descriptor(rA, rB, rX, nA, nB, nX):
    import numpy as np
    return (rA + rX) / (np.sqrt(2) * (rB + rX))
"""

BARTEL_TAU = """\
def descriptor(rA, rB, rX, nA, nB, nX):
    import numpy as np
    return rX / rB - nA * (nA - (rA / rB) / np.log(rA / rB))
"""


@pytest.fixture(scope="module")
def df():
    return load_dataset(str(DATA_PATH))


@pytest.fixture(scope="module")
def plot_dir(tmp_path_factory):
    return tmp_path_factory.mktemp("plots")


@pytest.fixture(scope="module")
def result_t(df, plot_dir):
    return evaluate_candidate(GOLDSCHMIDT_T, df, plot_dir, "test_t")


@pytest.fixture(scope="module")
def result_tau(df, plot_dir):
    return evaluate_candidate(BARTEL_TAU, df, plot_dir, "test_tau")


class TestGoldschmidtT:
    def test_no_error(self, result_t):
        assert not result_t.error

    def test_overall_accuracy(self, result_t):
        assert result_t.accuracy == pytest.approx(0.738, abs=0.02)

    def test_false_positive_rate(self, result_t):
        assert result_t.false_positive_rate == pytest.approx(0.506, abs=0.05)

    def test_plot_generated(self, result_t):
        assert Path(result_t.plot_path).exists()

    def test_descriptor_values_count(self, result_t):
        assert len(result_t.descriptor_values) == 576


class TestBartelTau:
    def test_no_error(self, result_tau):
        assert not result_tau.error

    def test_overall_accuracy(self, result_tau):
        assert result_tau.accuracy == pytest.approx(0.917, abs=0.01)

    def test_test_accuracy(self, result_tau):
        assert result_tau.test_accuracy == pytest.approx(0.94, abs=0.03)

    def test_false_positive_rate(self, result_tau):
        assert result_tau.false_positive_rate == pytest.approx(0.106, abs=0.03)

    def test_per_anion_all_above_88pct(self, result_tau):
        for anion, acc in result_tau.per_anion_accuracy.items():
            assert acc >= 0.88, f"Anion {anion} accuracy {acc:.1%} below 88%"

    def test_per_anion_oxygen(self, result_tau):
        assert result_tau.per_anion_accuracy["O"] == pytest.approx(0.92, abs=0.02)

    def test_per_anion_chloride(self, result_tau):
        assert result_tau.per_anion_accuracy["Cl"] == pytest.approx(0.90, abs=0.03)

    def test_plot_generated(self, result_tau):
        assert Path(result_tau.plot_path).exists()

    def test_descriptor_values_count(self, result_tau):
        assert len(result_tau.descriptor_values) == 576

    def test_tau_much_better_than_t(self, result_t, result_tau):
        assert result_tau.accuracy - result_t.accuracy > 0.15


class TestBrokenFunction:
    def test_syntax_error(self, df, plot_dir):
        code = "def descriptor(rA, rB, rX, nA, nB, nX):\n    return asdf("
        result = evaluate_candidate(code, df, plot_dir, "broken_syntax")
        assert result.error
        assert result.accuracy == 0.0

    def test_missing_function_name(self, df, plot_dir):
        code = "def my_func(rA, rB, rX, nA, nB, nX):\n    return rA"
        result = evaluate_candidate(code, df, plot_dir, "wrong_name")
        assert result.error
        assert result.accuracy == 0.0

    def test_returns_nan(self, df, plot_dir):
        code = "def descriptor(rA, rB, rX, nA, nB, nX):\n    return float('nan')"
        result = evaluate_candidate(code, df, plot_dir, "nan_result")
        assert result.error
        assert result.accuracy == 0.0
