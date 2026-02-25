from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
import streamlit as st

from gp_book.gp_math import cholesky_with_jitter

TITLE = "3) Multivariate Normal and Cholesky Factor"


def render() -> None:
    st.header(TITLE)
    st.markdown(
        "This is the key step from scalar statistics to Gaussian processes: "
        "we now model multiple random variables jointly."
    )
    st.markdown("For a vector random variable:")
    st.latex(r"\mathbf{x} \sim \mathcal{N}(\boldsymbol{\mu}, \Sigma)")
    st.markdown(r"$\Sigma$ captures covariance structure between dimensions.")
    st.markdown("Notation:")
    st.markdown(r"- $\mathbf{x} \in \mathbb{R}^d$: random vector.")
    st.markdown(r"- $\boldsymbol{\mu} \in \mathbb{R}^d$: mean vector.")
    st.markdown(r"- $\Sigma \in \mathbb{R}^{d \times d}$: covariance matrix.")
    st.markdown("Cholesky factorization writes:")
    st.latex(r"\Sigma = L L^\top")
    st.markdown(
        r"Here $L$ is a lower-triangular matrix (positive diagonal entries). "
        r"We use $L$ because sampling with $L\mathbf{z}$ is stable and efficient."
    )
    st.markdown(
        r"which lets us sample by "
        r"$\mathbf{x} = \boldsymbol{\mu} + L\mathbf{z}$, "
        r"$\mathbf{z} \sim \mathcal{N}(0, I)$."
    )
    st.latex(
        r"\Sigma = \begin{bmatrix}\sigma_1^2 & \rho\sigma_1\sigma_2 \\ "
        r"\rho\sigma_1\sigma_2 & \sigma_2^2\end{bmatrix}, \quad "
        r"L = \begin{bmatrix}\ell_{11} & 0 \\ \ell_{21} & \ell_{22}\end{bmatrix}"
    )

    col1, col2, col3, col4 = st.columns(4)
    with col1:
        sigma1 = st.slider("sigma_1", 0.2, 3.0, 1.0, 0.1)
    with col2:
        sigma2 = st.slider("sigma_2", 0.2, 3.0, 1.0, 0.1)
    with col3:
        rho = st.slider("correlation rho", -0.95, 0.95, 0.5, 0.01)
    with col4:
        n_samples = st.slider("samples", 100, 4000, 800, 100)

    cov = np.array(
        [
            [sigma1**2, rho * sigma1 * sigma2],
            [rho * sigma1 * sigma2, sigma2**2],
        ],
        dtype=float,
    )
    mu = np.array([0.0, 0.0], dtype=float)

    rng = np.random.default_rng(7)
    L = cholesky_with_jitter(cov)
    z = rng.standard_normal((2, n_samples))
    samples = (mu[:, None] + L @ z).T

    fig, ax = plt.subplots(figsize=(7, 6))
    ax.scatter(samples[:, 0], samples[:, 1], s=10, alpha=0.25)
    ax.set_title("Samples from a 2D Gaussian")
    ax.set_xlabel("x1")
    ax.set_ylabel("x2")
    ax.set_aspect("equal", adjustable="box")
    st.pyplot(fig)

    st.subheader("Covariance matrix")
    st.write(cov)
    st.subheader("Cholesky factor L")
    st.write(L)
    st.info(
        "Bridge to Chapter 1.1: if we treat function evaluations "
        r"$[f(x_1),\dots,f(x_n)]^\top$ as a random vector, "
        "the same multivariate Gaussian machinery applies."
    )

    st.code(
        "L = np.linalg.cholesky(Sigma)\n"
        "z = np.random.randn(dimension, n_samples)\n"
        "x = mu[:, None] + L @ z",
        language="python",
    )
