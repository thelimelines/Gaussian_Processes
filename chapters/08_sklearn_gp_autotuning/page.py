from __future__ import annotations

import warnings

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st
from sklearn.exceptions import ConvergenceWarning
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import ConstantKernel, ExpSineSquared, Matern, RBF, WhiteKernel

TITLE = "8) Scikit-Learn GP Regressor (Autotuned Hyperparameters)"


def _synthetic_fn(x: np.ndarray, mode: str) -> np.ndarray:
    if mode == "sine":
        return np.sin(x)
    if mode == "tan":
        return np.clip(np.tan(x), -3.0, 3.0)
    return np.sin(0.8 * x) + 0.2 * x


def _build_kernel(
    kernel_name: str,
    length_scale: float,
    variance: float,
    period: float,
    noise_std: float,
):
    if kernel_name == "RBF":
        base = RBF(length_scale=length_scale, length_scale_bounds=(1e-2, 1e2))
    elif kernel_name == "Matern 3/2":
        base = Matern(length_scale=length_scale, length_scale_bounds=(1e-2, 1e2), nu=1.5)
    else:
        base = ExpSineSquared(
            length_scale=length_scale,
            periodicity=period,
            length_scale_bounds=(1e-2, 1e2),
            periodicity_bounds=(1e-2, 1e2),
        )

    return ConstantKernel(variance, (1e-3, 1e3)) * base + WhiteKernel(
        noise_level=noise_std**2,
        noise_level_bounds=(1e-8, 1e1),
    )


def render() -> None:
    st.header(TITLE)
    st.markdown(
        "This chapter uses scikit-learn to fit the same GP model style, but now "
        "hyperparameters are tuned automatically instead of being fixed by hand."
    )
    st.latex(r"\theta^* = \arg\max_{\theta} \log p(y \mid X, \theta)")
    st.markdown(
        "Per scikit-learn docs, fitting optimizes log-marginal-likelihood, "
        "default optimizer is L-BFGS-B, and `n_restarts_optimizer` adds random restarts."
    )
    st.markdown(
        "When this same GP machinery is wrapped in a strategy for selecting the next point to evaluate, "
        "you get Bayesian optimization."
    )
    st.markdown("Notation carried from Chapter 7:")
    st.markdown(r"- $X$: observed inputs, $y$: observed outputs, $X_*$: prediction inputs.")
    st.markdown(r"- $\theta$: kernel hyperparameters (length-scale, variance, noise, periodicity).")
    st.markdown(r"- $*$ marks test/query points.")
    with st.expander("How autotuning works (plain-language view)"):
        st.markdown(
            r"1. Pick an initial kernel with parameters $\theta$ (length-scale, variance, noise, etc.)."
        )
        st.markdown(
            r"2. Build $K_\theta(X, X)$ from your current $\theta$, then evaluate "
            r"$\log p(y \mid X, \theta)$."
        )
        st.markdown(
            "3. Optimizer updates parameters to increase that score, repeating until convergence."
        )
        st.markdown(
            "4. The final kernel is used for prediction, giving a posterior mean and uncertainty band."
        )
        st.markdown(
            "Intuition: the model searches for hyperparameters that best explain observed data "
            "while respecting the structure implied by the chosen kernel family."
        )
    st.markdown(
        "Docs: "
        "[GaussianProcessRegressor API](https://scikit-learn.org/stable/modules/generated/"
        "sklearn.gaussian_process.GaussianProcessRegressor.html) and "
        "[Gaussian process user guide](https://scikit-learn.org/stable/modules/gaussian_process.html)."
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
        edited = st.data_editor(
            default_df,
            num_rows="dynamic",
            width="stretch",
            key="gp_table_sklearn",
        )
        clean = edited[["x", "y"]].apply(pd.to_numeric, errors="coerce").dropna()
        clean = clean.sort_values("x")
        if clean.shape[0] < 2:
            st.warning("Add at least 2 valid points in the custom table.")
            return
        x_train = clean["x"].to_numpy(dtype=float)
        y_train = clean["y"].to_numpy(dtype=float)
        x_min = float(np.min(x_train) - 1.0)
        x_max = float(np.max(x_train) + 1.0)

    st.subheader("Kernel and optimizer settings")
    col5, col6, col7, col8 = st.columns(4)
    with col5:
        kernel_name = st.selectbox("kernel", ["RBF", "Matern 3/2", "Periodic"], index=0)
    with col6:
        initial_length_scale = st.slider("initial length-scale", 0.05, 3.0, 0.8, 0.05)
    with col7:
        initial_variance = st.slider("initial variance", 0.05, 4.0, 1.0, 0.05)
    with col8:
        initial_noise_std = st.slider("initial noise std", 0.001, 1.0, 0.2, 0.001)

    col9, col10, col11, col12 = st.columns(4)
    with col9:
        period = st.slider("initial period (Periodic only)", 0.2, 5.0, 1.0, 0.05)
    with col10:
        normalize_y = st.checkbox("normalize y", value=True)
    with col11:
        optimizer_mode = st.selectbox("optimizer", ["L-BFGS-B", "None (fixed kernel)"], index=0)
    with col12:
        n_restarts_optimizer = st.slider("optimizer restarts", 0, 12, 4, 1)

    alpha = st.select_slider(
        "alpha (diagonal jitter/noise term)",
        options=[1e-10, 1e-9, 1e-8, 1e-7, 1e-6, 1e-5, 1e-4, 1e-3],
        value=1e-8,
    )

    n_test = st.slider("prediction grid points", 80, 600, 240, 10)
    n_posterior_draws = st.slider("posterior function draws", 0, 10, 3, 1)

    X_train = x_train[:, None]
    X_test = np.linspace(x_min, x_max, n_test)[:, None]
    x_test = X_test[:, 0]

    kernel = _build_kernel(
        kernel_name=kernel_name,
        length_scale=initial_length_scale,
        variance=initial_variance,
        period=period,
        noise_std=initial_noise_std,
    )

    optimizer = "fmin_l_bfgs_b" if optimizer_mode == "L-BFGS-B" else None
    restarts = n_restarts_optimizer if optimizer is not None else 0

    with warnings.catch_warnings(record=True) as caught_warnings:
        warnings.simplefilter("always", ConvergenceWarning)
        gpr = GaussianProcessRegressor(
            kernel=kernel,
            alpha=float(alpha),
            optimizer=optimizer,
            n_restarts_optimizer=restarts,
            normalize_y=normalize_y,
            random_state=321,
        )
        gpr.fit(X_train, y_train)

    convergence_messages = [
        str(warning.message)
        for warning in caught_warnings
        if issubclass(warning.category, ConvergenceWarning)
    ]
    if convergence_messages:
        st.warning("Convergence warnings were raised:\n- " + "\n- ".join(convergence_messages))

    mean, std = gpr.predict(X_test, return_std=True)

    title_fontsize = 24
    label_fontsize = 20
    tick_fontsize = 18
    legend_fontsize = 16

    fig, ax = plt.subplots(figsize=(11, 4.5))
    ax.fill_between(
        x_test,
        mean - 1.96 * std,
        mean + 1.96 * std,
        color="#1f77b4",
        alpha=0.2,
        label="95% interval",
        zorder=1,
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
            linewidth=3.0,
            label="true function",
            zorder=4,
        )

    if n_posterior_draws > 0:
        sampled = gpr.sample_y(X_test, n_samples=n_posterior_draws, random_state=321)
        if sampled.ndim == 1:
            sampled = sampled[:, None]
        if sampled.ndim == 3:
            sampled = sampled[:, :, 0]
        for idx in range(sampled.shape[1]):
            ax.plot(
                x_test,
                sampled[:, idx],
                color="#7a7a7a",
                linewidth=1.2,
                alpha=0.75,
                label="posterior draws" if idx == 0 else None,
                zorder=2,
            )

    ax.plot(x_test, mean, color="#1f77b4", linewidth=3.5, label="posterior mean", zorder=4)
    ax.scatter(x_train, y_train, color="black", label="train points", zorder=5)

    ax.set_title("Scikit-learn Gaussian Process Regression", fontsize=title_fontsize)
    ax.set_xlabel("x", fontsize=label_fontsize)
    ax.set_ylabel("y", fontsize=label_fontsize)
    ax.tick_params(axis="both", labelsize=tick_fontsize)
    ax.legend(loc="upper right", fontsize=legend_fontsize, ncol=2)
    st.pyplot(fig)

    st.subheader("Learned model")
    st.code(str(gpr.kernel_), language="text")
    colm1, colm2 = st.columns(2)
    with colm1:
        st.metric("Log-marginal likelihood", f"{gpr.log_marginal_likelihood_value_:.3f}")
    with colm2:
        st.metric("Optimizer restarts used", f"{restarts}")

    preview = pd.DataFrame({"x*": x_test[:12], "mean": mean[:12], "std": std[:12]})
    st.write("Preview of posterior predictions:")
    st.dataframe(preview, width="stretch")
