from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st

from gp_book.gp_math import (
    gp_posterior_predictive,
    matern32_kernel,
    periodic_kernel,
    rbf_kernel,
    sample_mvn,
)

TITLE = "7) Interactive GP Regressor"


def _synthetic_fn(x: np.ndarray, mode: str) -> np.ndarray:
    if mode == "sine":
        return np.sin(x)
    if mode == "tan":
        return np.clip(np.tan(x), -3.0, 3.0)
    return np.sin(0.8 * x) + 0.2 * x


def render() -> None:
    st.header(TITLE)
    st.markdown(
        "Now we switch from 'what a GP is' to 'how a GP predicts'. "
        "This page runs the full posterior update each time you move a slider."
    )
    st.markdown("Posterior GP regression (noisy observations):")
    st.latex(
        r"f_* \mid X, y, X_* \sim \mathcal{N}\left("
        r"\mu_*, \Sigma_*"
        r"\right)"
    )
    st.markdown(r"with $\mu_*, \Sigma_*$ built from the kernel matrices.")
    st.markdown("Paper-style notation for this page:")
    st.markdown(r"- $X=[x_1,\dots,x_n]$: observed input locations.")
    st.markdown(r"- $y=[y_1,\dots,y_n]^\top$: observed targets.")
    st.markdown(r"- $X_*=[x^*_1,\dots,x^*_m]$: test/query locations.")
    st.markdown(r"- $*$ means test/query, not multiplication.")
    st.markdown(r"- $K_{XX}$, $K_{X_*X}$, $K_{X_*X_*}$ are kernel blocks.")
    with st.expander("What the symbols mean on this page"):
        st.markdown(r"- $X$: training inputs you already observed.")
        st.markdown(r"- $y$: observed outputs at those inputs.")
        st.markdown(r"- $X_*$: new query inputs where you want predictions.")
        st.markdown(r"- $\mu_*$: posterior mean curve (best estimate).")
        st.markdown(r"- $\Sigma_*$: posterior covariance (uncertainty).")
        st.markdown(
            r"- `assumed obs noise std`: this is $\sigma_n$, and we use $\sigma_n^2 I$ "
            r"inside $K_{XX} + \sigma_n^2 I$."
        )
        st.markdown(
            "You can read the flow as: build kernel matrices -> combine with noise -> "
            "condition a joint Gaussian -> get mean + uncertainty at each test point."
        )
        st.markdown("Single-point predictive form (matches the slide notation):")
        st.latex(
            r"\mu(x) = m(x) + K(x, X)\left[K(X, X) + \sigma_n^2 I\right]^{-1}\left(y - m(X)\right)"
        )
        st.latex(
            r"\sigma^2(x) = K(x, x) - K(x, X)\left[K(X, X) + \sigma_n^2 I\right]^{-1}K(X, x)"
        )
        st.latex(
            r"K(x,X)=\begin{bmatrix}K(x,x_1)&\cdots&K(x,x_t)\end{bmatrix}"
        )
        st.latex(
            r"K(X,X)=\begin{bmatrix}"
            r"K(x_1,x_1)&\cdots&K(x_1,x_t)\\"
            r"\vdots&\ddots&\vdots\\"
            r"K(x_t,x_1)&\cdots&K(x_t,x_t)"
            r"\end{bmatrix}"
        )

    data_mode = st.radio(
        "Data source",
        ["Synthetic: sine", "Synthetic: tan", "Synthetic: mixed", "Custom table"],
        horizontal=True,
    )

    true_curve = None
    rng = np.random.default_rng(123)

    if data_mode != "Custom table":
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            n_train = st.slider("train points", 6, 80, 18, 1)
        with col2:
            x_min = st.slider("x min", -8.0, 0.0, -4.0, 0.1)
        with col3:
            x_max = st.slider("x max", 0.0, 8.0, 4.0, 0.1)
        with col4:
            data_noise_std = st.slider("data noise std", 0.0, 1.0, 0.2, 0.01)

        mode_lookup = {
            "Synthetic: sine": "sine",
            "Synthetic: tan": "tan",
            "Synthetic: mixed": "mixed",
        }
        x_train = np.linspace(x_min, x_max, n_train)
        true_curve = _synthetic_fn(x_train, mode_lookup[data_mode])
        y_train = true_curve + data_noise_std * rng.standard_normal(n_train)
    else:
        st.write("Edit your training data table:")
        default_df = pd.DataFrame(
            {
                "x": [-3.0, -2.0, -1.0, 0.0, 1.0, 2.0, 3.0],
                "y": [0.2, -0.7, -0.9, -0.1, 0.8, 0.7, 0.3],
            }
        )
        edited = st.data_editor(default_df, num_rows="dynamic", width="stretch", key="gp_table")
        clean = edited[["x", "y"]].apply(pd.to_numeric, errors="coerce").dropna()
        clean = clean.sort_values("x")
        if clean.shape[0] < 2:
            st.warning("Add at least 2 valid points in the custom table.")
            return
        x_train = clean["x"].to_numpy(dtype=float)
        y_train = clean["y"].to_numpy(dtype=float)
        x_min = float(np.min(x_train) - 1.0)
        x_max = float(np.max(x_train) + 1.0)

    st.subheader("Kernel and model settings")
    col5, col6, col7, col8 = st.columns(4)
    with col5:
        kernel_name = st.selectbox("kernel", ["RBF", "Matern 3/2", "Periodic"], index=0)
    with col6:
        length_scale = st.slider("length-scale", 0.05, 3.0, 0.8, 0.05)
    with col7:
        variance = st.slider("variance", 0.05, 4.0, 1.0, 0.05)
    with col8:
        obs_noise_std = st.slider("assumed obs noise std", 0.001, 1.0, 0.2, 0.001)

    period = st.slider("period (Periodic kernel only)", 0.2, 5.0, 1.0, 0.05)

    if kernel_name == "RBF":
        kernel_fn = lambda a, b: rbf_kernel(a, b, length_scale=length_scale, variance=variance)
    elif kernel_name == "Matern 3/2":
        kernel_fn = lambda a, b: matern32_kernel(a, b, length_scale=length_scale, variance=variance)
    else:
        kernel_fn = lambda a, b: periodic_kernel(
            a, b, length_scale=length_scale, variance=variance, period=period
        )

    n_test = st.slider("prediction grid points", 80, 600, 240, 10)
    n_posterior_draws = st.slider("posterior function draws", 0, 10, 3, 1)

    X_train = x_train[:, None]
    X_test = np.linspace(x_min, x_max, n_test)[:, None]

    mean, cov, std = gp_posterior_predictive(
        X_train,
        y_train,
        X_test,
        kernel_fn,
        noise_variance=obs_noise_std**2,
    )

    fig, ax = plt.subplots(figsize=(11, 4.5))
    x_test = X_test[:, 0]
    ax.scatter(x_train, y_train, color="black", label="train points", zorder=3)
    ax.plot(x_test, mean, color="#1f77b4", linewidth=2.0, label="posterior mean")
    ax.fill_between(
        x_test,
        mean - 1.96 * std,
        mean + 1.96 * std,
        color="#1f77b4",
        alpha=0.2,
        label="95% interval",
    )

    if true_curve is not None:
        mode_lookup = {
            "Synthetic: sine": "sine",
            "Synthetic: tan": "tan",
            "Synthetic: mixed": "mixed",
        }
        ax.plot(
            x_test,
            _synthetic_fn(x_test, mode_lookup[data_mode]),
            color="#2ca02c",
            linestyle="--",
            linewidth=1.5,
            label="true function",
        )

    if n_posterior_draws > 0:
        draw_rng = np.random.default_rng(321)
        draws = sample_mvn(mean, cov, n_samples=n_posterior_draws, rng=draw_rng)
        for idx, draw in enumerate(draws):
            ax.plot(x_test, draw, linewidth=1.0, alpha=0.8, label=f"posterior draw {idx + 1}")

    ax.set_title("Gaussian Process Regression")
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.legend(loc="upper right", fontsize=8, ncol=2)
    st.pyplot(fig)

    preview = pd.DataFrame({"x*": x_test[:12], "mean": mean[:12], "std": std[:12]})
    st.write("Preview of posterior predictions:")
    st.dataframe(preview, width="stretch")
