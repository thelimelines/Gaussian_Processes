from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st

TITLE = "4) Latent Function Values Form a Vector"


def _base_function(x: np.ndarray, fn_name: str) -> np.ndarray:
    if fn_name == "sine":
        return np.sin(x)
    if fn_name == "cosine":
        return np.cos(x)
    if fn_name == "quadratic":
        return 0.15 * (x**2) - 0.5
    return 0.4 * x


def render() -> None:
    st.header(TITLE)
    st.markdown(
        "A Gaussian process describes an unknown, noise-free function. "
        "We call this the **latent function** because it is not observed directly."
    )
    st.markdown(r"At inputs $x_1,\ldots,x_n$, its latent values form the vector")
    st.latex(r"\mathbf f=[f(x_1),\ldots,f(x_n)]^\top.")
    st.markdown(
        r"The GP prior is a distribution over $\mathbf f$. It says which sets of latent "
        "function values are plausible before any measurements are used."
    )

    col1, col2 = st.columns(2)
    with col1:
        fn_name = st.selectbox(
            "Latent function",
            ["sine", "cosine", "quadratic", "linear"],
            index=0,
            key="ch4_function",
        )
    with col2:
        n_points = st.slider("Input locations", 5, 120, 25, 1, key="ch4_n_points")

    x = np.linspace(-3.5, 3.5, n_points)
    f = _base_function(x, fn_name)

    st.subheader("Measurements add noise")
    st.markdown(r"A measurement $y_i$ is the latent value plus observational noise:")
    st.latex(r"y_i=f(x_i)+\epsilon_i,\qquad \epsilon_i\sim\mathcal N(0,\sigma_n^2).")
    noise_std = st.slider(
        "Observation noise standard deviation",
        0.0,
        1.0,
        0.1,
        0.01,
        key="ch4_noise_std",
    )

    rng = np.random.default_rng(3)
    y = f + noise_std * rng.standard_normal(n_points)
    x_dense = np.linspace(-3.5, 3.5, 400)
    f_dense = _base_function(x_dense, fn_name)

    fig, ax = plt.subplots(figsize=(9, 4))
    ax.plot(x_dense, f_dense, color="#1f77b4", linewidth=2.5, label="latent function f(x)")
    ax.scatter(
        x,
        f,
        color="#1f77b4",
        s=24,
        zorder=3,
        label="latent values f(x_i)",
    )
    ax.scatter(
        x,
        y,
        color="#d62728",
        marker="x",
        s=38,
        zorder=4,
        label="observations y_i",
    )
    ax.set_xlabel("x")
    ax.set_ylabel("value")
    ax.set_title("Latent function values and noisy observations")
    ax.legend(loc="upper right")
    st.pyplot(fig)

    preview = pd.DataFrame(
        {
            "x_i": x[:10],
            "latent f(x_i)": f[:10],
            "observation y_i": y[:10],
        }
    )
    st.write("First 10 latent values and observations:")
    st.dataframe(preview, width="stretch")
    st.info(
        r"Keep the roles separate: the GP prior models latent values $\mathbf f$. "
        r"The observations $\mathbf y$ also contain measurement noise."
    )
    st.info(
        r"Bridge to Section 1.2: at a finite set of inputs, a Gaussian process becomes "
        r"a multivariate normal distribution over the latent vector $\mathbf f$. "
        r"We now need a covariance matrix $K$ to describe how those values vary together. "
        r"A kernel builds that matrix from the input locations: "
        r"$K_{ij}=k(x_i,x_j)$, so $\mathbf f\sim\mathcal{N}(\mathbf m,K)$."
    )

    st.code(
        "x = np.linspace(-3.5, 3.5, n)\n"
        "f_vec = f(x)                 # latent function values\n"
        "y = f_vec + observation_noise  # measured observations",
        language="python",
    )
