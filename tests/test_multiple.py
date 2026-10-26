"""Multiple-testing procedures against textbook examples."""

import numpy as np
import pytest
from scipy import stats

from btaudit.multiple import benjamini_hochberg, bonferroni, holm, one_sided_t_pvalues

# The 15 p-values of the worked example in Benjamini and Hochberg (1995, section 4).
BH_EXAMPLE = np.array(
    [
        0.0001,
        0.0004,
        0.0019,
        0.0095,
        0.0201,
        0.0278,
        0.0298,
        0.0344,
        0.0459,
        0.3240,
        0.4262,
        0.5719,
        0.6528,
        0.7590,
        1.0000,
    ]
)


def test_bh_example_rejects_four() -> None:
    result = benjamini_hochberg(BH_EXAMPLE, 0.05)
    assert result.n_rejected == 4
    assert np.all(result.reject[:4]) and not np.any(result.reject[4:])


def test_bonferroni_example_rejects_three() -> None:
    assert bonferroni(BH_EXAMPLE, 0.05).n_rejected == 3


def test_holm_example_rejects_three() -> None:
    assert holm(BH_EXAMPLE, 0.05).n_rejected == 3


def test_adjusted_p_values_hand_computed() -> None:
    p = np.array([0.01, 0.04, 0.03, 0.005])
    assert np.allclose(bonferroni(p).adjusted, [0.04, 0.16, 0.12, 0.02])
    # Holm: sorted [0.005, 0.01, 0.03, 0.04] x [4, 3, 2, 1] = [0.02, 0.03, 0.06, 0.04],
    # running max [0.02, 0.03, 0.06, 0.06], mapped back to input order.
    assert np.allclose(holm(p).adjusted, [0.03, 0.06, 0.06, 0.02])
    # BH: sorted x 4 / rank = [0.02, 0.02, 0.04, 0.04], running min from the top.
    assert np.allclose(benjamini_hochberg(p).adjusted, [0.02, 0.04, 0.04, 0.02])


def test_procedures_are_nested() -> None:
    rng = np.random.default_rng(0)
    p = np.concatenate([rng.uniform(0, 0.01, 10), rng.uniform(size=90)])
    b, h, bh = bonferroni(p), holm(p), benjamini_hochberg(p)
    assert np.all(b.reject <= h.reject) and np.all(h.reject <= bh.reject)
    assert np.all(h.adjusted <= b.adjusted + 1e-15)


def test_results_follow_input_order() -> None:
    rng = np.random.default_rng(1)
    p = rng.uniform(0, 0.2, 30)
    perm = rng.permutation(30)
    for procedure in (bonferroni, holm, benjamini_hochberg):
        assert np.allclose(procedure(p[perm]).adjusted, procedure(p).adjusted[perm])


@pytest.mark.parametrize("bad", [[-0.1, 0.5], [0.2, 1.5], [np.nan], []])
def test_invalid_p_values_raise(bad: list) -> None:
    with pytest.raises(ValueError):
        holm(bad)


def test_one_sided_t_pvalues_match_scipy() -> None:
    x = np.random.default_rng(2).normal(0.1, 1.0, size=(80, 4))
    expected = [stats.ttest_1samp(x[:, k], 0.0, alternative="greater").pvalue for k in range(4)]
    assert np.allclose(one_sided_t_pvalues(x), expected)


def test_fwer_of_holm_under_independent_nulls() -> None:
    rng = np.random.default_rng(3)
    p = rng.uniform(size=(20_000, 50))
    fwer = np.mean([holm(row).n_rejected > 0 for row in p])
    assert fwer == pytest.approx(1 - (1 - 0.05 / 50) ** 50, abs=0.006)
