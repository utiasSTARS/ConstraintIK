import numpy as np

from constraint_ik import id_estimation as ide

# Worked example of App. D.1 (UR5, t = 15, K = 50; regenerate with scripts/eval_id.py --worked_example).
SE3 = [153.9, 144.2, 129.3, 125.3, 114.9, 105.7, 101.3, 88.5, 82.5, 74.6, 66.2, 55.6]
LINE = [146.8, 138.5, 127.0, 112.4, 107.3, 95.2, 83.8, 70.2, 9.6, 4.3, 3.4, 3.1]


def test_gap_rules():
    assert ide.gap_rank(SE3) == 2 and ide.gap_rank(SE3, dagger=True) == 12
    assert ide.id_from_spectrum(np.array(SE3), 12) == (10, 0)
    assert ide.id_from_spectrum(np.array(LINE), 12) == (4, 4)
    assert ide.gap_rank([5.0, 1e-13]) == 1


def test_neighbourhood_estimators_on_a_plane():
    rng = np.random.default_rng(0)
    X = np.zeros((600, 6))
    X[:, :2] = rng.random((600, 2))
    assert abs(ide.local_mle(X, 10) - 2) < 0.3
    assert ide.local_pca(X, 10, 0.9) == 2
    assert abs(ide.twonn(X) - 2 / np.log(2)) < 0.3    # median of per-point 1/log(r2/r1) is d/ln 2
    # everything isolated: MLE has no terms, PCA and TwoNN count every point as 0
    far = X * 1e6
    assert ide.local_mle(far) == ide.local_pca(far) == ide.twonn(far) == 0.0
    assert abs(ide.local_mle(far, iso=None) - 2) < 0.3


def test_anchor_rules():
    np.random.seed(7)
    expected = np.random.choice(40, 5, replace=False)
    assert np.array_equal(ide.uniform_anchors(40, 5, np.random.RandomState(7)), expected)
    sols = np.random.default_rng(1).uniform(-3, 3, (40, 6))
    q_valid = sols[[3, 17]] + 1e-3
    assert set(ide.model_anchors(sols, q_valid, 2, None)) == {3, 17}
    assert len(ide.model_anchors(sols, q_valid[:0], 4, np.random.default_rng(0))) == 4
