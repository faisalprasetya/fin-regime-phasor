import numpy as np

from fin_regime_phasor.discretize import apply_vq_codebook, fit_vq_codebook


def _two_blobs(seed=0, n=200):
    rng = np.random.default_rng(seed)
    blob_a = rng.normal(loc=(0.1, -1.0), scale=0.01, size=(n, 2))
    blob_b = rng.normal(loc=(0.9, 1.0), scale=0.01, size=(n, 2))
    features = np.concatenate([blob_a, blob_b], axis=0)
    return features[:, 0], features[:, 1]


def test_fit_vq_codebook_recovers_two_separated_clusters():
    sigma, theta = _two_blobs()
    codebook = fit_vq_codebook(sigma, theta, n_symbols=2)
    symbols = apply_vq_codebook(codebook, sigma, theta)

    n = len(sigma) // 2
    # each half should be assigned overwhelmingly to a single (possibly different) symbol
    first_half_mode = np.bincount(symbols[:n]).max()
    second_half_mode = np.bincount(symbols[n:]).max()
    assert first_half_mode >= n * 0.95
    assert second_half_mode >= n * 0.95


def test_codebook_symbol_count_matches_n_symbols():
    sigma, theta = _two_blobs()
    codebook = fit_vq_codebook(sigma, theta, n_symbols=4)
    assert codebook.n_symbols == 4


def test_apply_vq_codebook_symbols_within_range():
    sigma, theta = _two_blobs()
    codebook = fit_vq_codebook(sigma, theta, n_symbols=3)
    symbols = apply_vq_codebook(codebook, sigma, theta)
    assert symbols.min() >= 0
    assert symbols.max() < 3


def test_apply_vq_codebook_on_held_out_points_nearest_center():
    sigma, theta = _two_blobs()
    codebook = fit_vq_codebook(sigma, theta, n_symbols=2)
    # a fresh point close to blob A's mean (0.1, -1.0) should land on the same symbol as blob A
    held_out_sigma = np.array([0.1])
    held_out_theta = np.array([-1.0])
    symbol = apply_vq_codebook(codebook, held_out_sigma, held_out_theta)
    blob_a_symbol = apply_vq_codebook(codebook, np.array([0.1]), np.array([-1.0]))
    assert symbol[0] == blob_a_symbol[0]


def test_fit_vq_codebook_is_deterministic_given_random_state():
    sigma, theta = _two_blobs()
    codebook_1 = fit_vq_codebook(sigma, theta, n_symbols=2, random_state=42)
    codebook_2 = fit_vq_codebook(sigma, theta, n_symbols=2, random_state=42)
    np.testing.assert_allclose(sorted(codebook_1.centers[:, 0]), sorted(codebook_2.centers[:, 0]))
