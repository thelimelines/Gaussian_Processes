from __future__ import annotations

import numpy as np


def ensure_2d(X: np.ndarray) -> np.ndarray:
    X = np.asarray(X, dtype=float)
    if X.ndim == 1:
        return X[:, None]
    return X


def standard_normal_pdf(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=float)
    return (1.0 / np.sqrt(2.0 * np.pi)) * np.exp(-0.5 * x**2)


def normal_pdf(x: np.ndarray, mu: float, sigma: float) -> np.ndarray:
    x = np.asarray(x, dtype=float)
    sigma = max(float(sigma), 1e-10)
    z = (x - mu) / sigma
    return standard_normal_pdf(z) / sigma


def rbf_kernel(
    X1: np.ndarray,
    X2: np.ndarray,
    *,
    length_scale: float = 1.0,
    variance: float = 1.0,
) -> np.ndarray:
    X1 = ensure_2d(X1)
    X2 = ensure_2d(X2)
    length_scale = max(float(length_scale), 1e-10)
    variance = max(float(variance), 1e-12)
    sqdist = np.sum((X1[:, None, :] - X2[None, :, :]) ** 2, axis=2)
    return variance * np.exp(-0.5 * sqdist / (length_scale**2))


def matern32_kernel(
    X1: np.ndarray,
    X2: np.ndarray,
    *,
    length_scale: float = 1.0,
    variance: float = 1.0,
) -> np.ndarray:
    X1 = ensure_2d(X1)
    X2 = ensure_2d(X2)
    length_scale = max(float(length_scale), 1e-10)
    variance = max(float(variance), 1e-12)
    dist = np.sqrt(np.sum((X1[:, None, :] - X2[None, :, :]) ** 2, axis=2))
    scaled = np.sqrt(3.0) * dist / length_scale
    return variance * (1.0 + scaled) * np.exp(-scaled)


def periodic_kernel(
    X1: np.ndarray,
    X2: np.ndarray,
    *,
    length_scale: float = 1.0,
    variance: float = 1.0,
    period: float = 1.0,
) -> np.ndarray:
    X1 = ensure_2d(X1)
    X2 = ensure_2d(X2)
    length_scale = max(float(length_scale), 1e-10)
    variance = max(float(variance), 1e-12)
    period = max(float(period), 1e-10)
    dist = np.sqrt(np.sum((X1[:, None, :] - X2[None, :, :]) ** 2, axis=2))
    sin_term = np.sin(np.pi * dist / period)
    return variance * np.exp(-2.0 * sin_term**2 / (length_scale**2))


def cholesky_with_jitter(K: np.ndarray, base_jitter: float = 1e-8) -> np.ndarray:
    K = np.asarray(K, dtype=float)
    jitter = max(float(base_jitter), 1e-12)
    for _ in range(6):
        try:
            return np.linalg.cholesky(K + jitter * np.eye(K.shape[0]))
        except np.linalg.LinAlgError:
            jitter *= 10.0
    raise np.linalg.LinAlgError("Cholesky decomposition failed even after jitter escalation.")


def sample_mvn(
    mu: np.ndarray,
    cov: np.ndarray,
    n_samples: int,
    rng: np.random.Generator,
    *,
    jitter: float = 1e-8,
) -> np.ndarray:
    mu = np.asarray(mu, dtype=float).reshape(-1)
    cov = np.asarray(cov, dtype=float)
    L = cholesky_with_jitter(cov, base_jitter=jitter)
    z = rng.standard_normal((mu.shape[0], int(n_samples)))
    return (mu[:, None] + L @ z).T


def sample_gp_prior(
    X: np.ndarray,
    kernel_fn,
    *,
    n_samples: int,
    rng: np.random.Generator,
    mean: np.ndarray | None = None,
    jitter: float = 1e-8,
) -> np.ndarray:
    X = ensure_2d(X)
    K = kernel_fn(X, X)
    K = 0.5 * (K + K.T)
    mean_vector = (
        np.zeros(X.shape[0], dtype=float)
        if mean is None
        else np.asarray(mean, dtype=float)
    )
    return sample_mvn(mean_vector, K, n_samples=n_samples, rng=rng, jitter=jitter)


def gp_posterior_predictive(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    kernel_fn,
    *,
    noise_variance: float | np.ndarray = 1e-2,
    jitter: float = 1e-8,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    X_train = ensure_2d(X_train)
    X_test = ensure_2d(X_test)
    y_train = np.asarray(y_train, dtype=float).reshape(-1, 1)

    jitter = max(float(jitter), 1e-12)

    noise_values = np.asarray(noise_variance, dtype=float)
    if noise_values.ndim == 0:
        noise_diagonal = np.full(
            X_train.shape[0],
            max(float(noise_values.item()), 1e-12),
        )
    else:
        noise_diagonal = np.maximum(noise_values.reshape(-1), 1e-12)
        if noise_diagonal.shape[0] != X_train.shape[0]:
            raise ValueError("noise_variance must be a scalar or have one value per observation.")

    K_xx = kernel_fn(X_train, X_train)
    K_xx = 0.5 * (K_xx + K_xx.T) + np.diag(noise_diagonal)
    K_xs = kernel_fn(X_train, X_test)
    K_ss = kernel_fn(X_test, X_test)
    K_ss = 0.5 * (K_ss + K_ss.T)

    L = cholesky_with_jitter(K_xx, base_jitter=jitter)
    alpha = np.linalg.solve(L.T, np.linalg.solve(L, y_train))
    mean = (K_xs.T @ alpha).reshape(-1)

    v = np.linalg.solve(L, K_xs)
    cov = K_ss - v.T @ v
    cov = 0.5 * (cov + cov.T)
    std = np.sqrt(np.maximum(np.diag(cov), 0.0))
    return mean, cov, std
