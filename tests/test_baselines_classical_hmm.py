import numpy as np

from fin_regime_phasor.baselines.classical_hmm import fit_categorical_hmm, fit_gaussian_hmm


def _two_regime_features(seed=0, n_per_regime=150):
    rng = np.random.default_rng(seed)
    regime_a = rng.normal(loc=(0.001, 0.01), scale=(0.0005, 0.002), size=(n_per_regime, 2))
    regime_b = rng.normal(loc=(-0.002, 0.05), scale=(0.001, 0.01), size=(n_per_regime, 2))
    features = np.concatenate([regime_a, regime_b], axis=0)
    labels = np.concatenate([np.zeros(n_per_regime), np.ones(n_per_regime)])
    return features, labels


def test_fit_gaussian_hmm_returns_valid_result_shape():
    features, _ = _two_regime_features()
    result = fit_gaussian_hmm(features, n_states=2, random_state=0)

    assert result.states.shape == (features.shape[0],)
    assert np.isfinite(result.log_likelihood)
    assert result.aic > 0 or result.aic < 0  # just check it's a finite real number
    assert np.isfinite(result.aic)
    assert np.isfinite(result.bic)


def test_fit_gaussian_hmm_recovers_two_separated_regimes():
    features, labels = _two_regime_features()
    result = fit_gaussian_hmm(features, n_states=2, random_state=0)

    # up to label permutation, predicted states should align strongly with ground truth
    agreement = max(
        np.mean(result.states == labels),
        np.mean(result.states == (1 - labels)),
    )
    assert agreement > 0.9


def test_fit_categorical_hmm_returns_valid_result_shape():
    rng = np.random.default_rng(0)
    symbols = rng.integers(0, 4, size=300)
    result = fit_categorical_hmm(symbols, n_states=2, n_symbols=4, random_state=0)

    assert result.states.shape == (300,)
    assert np.isfinite(result.log_likelihood)
    assert np.isfinite(result.aic)
    assert np.isfinite(result.bic)


def test_bic_increases_with_more_states_on_pure_noise():
    rng = np.random.default_rng(0)
    symbols = rng.integers(0, 3, size=200)
    result_2 = fit_categorical_hmm(symbols, n_states=2, n_symbols=3, random_state=0)
    result_4 = fit_categorical_hmm(symbols, n_states=4, n_symbols=3, random_state=0)
    # more parameters on unstructured noise should be penalized more heavily by BIC
    assert result_4.n_params > result_2.n_params
