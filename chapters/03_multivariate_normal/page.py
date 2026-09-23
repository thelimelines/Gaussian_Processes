from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
import streamlit as st
from matplotlib.legend_handler import HandlerTuple
from matplotlib.patches import Ellipse, Rectangle

from gp_book.gp_math import cholesky_with_jitter

TITLE = "3) Multivariate Normal"


def render() -> None:
    st.header(TITLE)
    st.markdown(
        "A covariance matrix describes the size and direction of a Gaussian cloud. "
        "This is the key step from one random variable to many."
    )
    st.markdown("For a random vector:")
    st.latex(r"\mathbf{x} \sim \mathcal{N}(\boldsymbol{\mu}, \Sigma)")
    st.latex(
        r"\Sigma = \begin{bmatrix}\sigma_1^2 & \rho\sigma_1\sigma_2 \\ "
        r"\rho\sigma_1\sigma_2 & \sigma_2^2\end{bmatrix}"
    )
    st.markdown(
        r"The diagonal terms $\sigma_1^2$ and $\sigma_2^2$ set the spread along each axis. "
        r"The off-diagonal term sets how strongly the two variables move together."
    )

    col1, col2, col3, col4 = st.columns(4)
    with col1:
        sigma1 = st.slider("sigma_1", 0.2, 3.0, 1.0, 0.1, key="ch3_sigma1")
    with col2:
        sigma2 = st.slider("sigma_2", 0.2, 3.0, 1.0, 0.1, key="ch3_sigma2")
    with col3:
        rho = st.slider("correlation rho", -0.95, 0.95, 0.5, 0.01, key="ch3_rho")
    with col4:
        n_samples = st.slider("samples", 100, 4000, 800, 100, key="ch3_samples")

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
    eigenvalues, eigenvectors = np.linalg.eigh(cov)
    order = np.argsort(eigenvalues)[::-1]
    eigenvalues = eigenvalues[order]
    principal_vector = eigenvectors[:, order[0]]
    angle = np.degrees(np.arctan2(principal_vector[1], principal_vector[0]))

    for standard_deviations, alpha in ((3, 0.2), (2, 0.5), (1, 0.8)):
        ax.add_patch(
            Ellipse(
                (0.0, 0.0),
                width=2.0 * standard_deviations * np.sqrt(eigenvalues[0]),
                height=2.0 * standard_deviations * np.sqrt(eigenvalues[1]),
                angle=angle,
                facecolor="#f4a261",
                edgecolor="none",
                alpha=alpha,
                label="_nolegend_",
            )
        )

    samples_artist = ax.scatter(
        samples[:, 0],
        samples[:, 1],
        s=10,
        alpha=0.5,
        color="#1f77b4",
        label="samples",
        zorder=3,
    )
    ax.axhline(0.0, color="0.8", linewidth=0.8)
    ax.axvline(0.0, color="0.8", linewidth=0.8)
    ax.set_title("Covariance controls shape and orientation")
    ax.set_xlabel("x1")
    ax.set_ylabel("x2")
    ax.set_aspect("equal", adjustable="box")
    sd_legend_handle = tuple(
        Rectangle((0.0, 0.0), 1.0, 1.0, facecolor="#f4a261", edgecolor="none", alpha=alpha)
        for alpha in (0.8, 0.5, 0.2)
    )
    ax.legend(
        handles=[sd_legend_handle, samples_artist],
        labels=["1, 2, and 3 SD regions", "samples"],
        handler_map={tuple: HandlerTuple(ndivide=3)},
        loc="upper left",
    )
    st.pyplot(fig)

    st.subheader("Covariance matrix")
    st.write(cov)
    direction = "upward" if rho > 0.05 else "downward" if rho < -0.05 else "almost no"
    st.markdown(
        f"- Horizontal variance: **{sigma1**2:.2f}**. Vertical variance: **{sigma2**2:.2f}**.\n"
        f"- Correlation: **{rho:.2f}**, giving {direction} tilt.\n"
        "- Change a diagonal scale to stretch the cloud. Change correlation to rotate it."
    )

    with st.expander("Optional: how Cholesky generates correlated samples"):
        st.markdown(
            "The multivariate normal above is the distribution we care about. "
            "Cholesky is one useful way to generate samples from it."
        )
        st.markdown(
            r"Start with independent standard-normal values $\mathbf{z}\sim\mathcal{N}(0,I)$. "
            r"They form a circular cloud with no preferred direction."
        )
        st.markdown(
            "Cholesky rewrites the covariance in a computationally useful form: "
            "it finds a lower-triangular matrix $L$ such that"
        )
        st.latex(r"L L^\top = \Sigma")
        st.markdown(
            "This format is useful because multiplying independent noise by $L$ "
            "creates samples with the spread and correlations encoded by $\\Sigma$. "
            "The triangular structure also makes the transformation efficient to solve."
        )
        st.latex(r"\mathbf{x} = \boldsymbol{\mu} + L\mathbf{z}")
        st.markdown("The transformed samples have the covariance we asked for:")
        st.latex(
            r"\operatorname{Cov}(\mathbf{x})"
            r"=L\operatorname{Cov}(\mathbf{z})L^\top"
            r"=LIL^\top=\Sigma"
        )
        st.markdown(
            "So Cholesky is not a different distribution; it is a practical "
            "representation for sampling from the multivariate normal."
        )

        st.subheader("Cholesky factor L")
        st.write(L)
        st.code(
            "L = np.linalg.cholesky(Sigma)\n"
            "z = np.random.randn(dimension, n_samples)\n"
            "x = mu[:, None] + L @ z",
            language="python",
        )

    st.info(
        "Next, we treat function values "
        r"$[f(x_1),\dots,f(x_n)]^\top$ as a random vector, "
        "so the same covariance machinery applies."
    )
