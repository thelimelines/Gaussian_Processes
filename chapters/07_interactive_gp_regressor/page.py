from __future__ import annotations

from typing import TypedDict

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


class Observation(TypedDict):
    x: float
    y: float
    noise_std: float


def _synthetic_fn(x: np.ndarray, mode: str) -> np.ndarray:
    if mode == "sine":
        return np.sin(x)
    if mode == "tan":
        return np.clip(np.tan(x), -3.0, 3.0)
    return np.sin(0.8 * x) + 0.2 * x


def _initial_observations() -> list[Observation]:
    return [
        {"x": -1.5, "y": float(np.sin(-1.5)), "noise_std": 0.1},
        {"x": 1.5, "y": float(np.sin(1.5)), "noise_std": 0.1},
    ]


def _conditioning_posterior(
    observations: list[Observation],
    X_test: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    X_train = np.array([item["x"] for item in observations], dtype=float)[:, None]
    y_train = np.array([item["y"] for item in observations], dtype=float)
    noise_variance = np.square(
        np.array([item["noise_std"] for item in observations], dtype=float)
    )
    kernel_fn = lambda a, b: rbf_kernel(a, b, length_scale=1.0, variance=1.0)
    return gp_posterior_predictive(
        X_train,
        y_train,
        X_test,
        kernel_fn,
        noise_variance=noise_variance,
    )


def _render_conditioning_experiment() -> None:
    st.subheader("Conditioning experiment")
    st.markdown(
        "Conditioning means updating a prediction after seeing an observation. "
        "Choose a point, predict the effect, then add it and inspect the change."
    )
    st.info("Cycle: **predict → change → observe → explain**. Add several points and repeat.")

    if "ch7_observations" not in st.session_state:
        st.session_state["ch7_observations"] = _initial_observations()
    if "ch7_candidate_x" not in st.session_state:
        st.session_state["ch7_candidate_x"] = 0.0
    if "ch7_candidate_y" not in st.session_state:
        st.session_state["ch7_candidate_y"] = 0.0
    if "ch7_candidate_noise" not in st.session_state:
        st.session_state["ch7_candidate_noise"] = 0.05
    if "ch7_last_addition" not in st.session_state:
        st.session_state["ch7_last_addition"] = None

    st.markdown("**Try a contrast**")
    preset_columns = st.columns(4)
    presets = [
        ("Near + low noise", 1.8, 0.05),
        ("Far + low noise", 4.2, 0.05),
        ("Near + high noise", -1.8, 0.8),
        ("Far + high noise", -4.2, 0.8),
    ]
    for column, (label, x_value, noise_value) in zip(preset_columns, presets):
        with column:
            if st.button(label, key=f"ch7_preset_{label}", width="stretch"):
                st.session_state["ch7_candidate_x"] = x_value
                st.session_state["ch7_candidate_y"] = float(np.sin(x_value))
                st.session_state["ch7_candidate_noise"] = noise_value
                st.rerun()

    candidate_columns = st.columns(3)
    with candidate_columns[0]:
        candidate_x = st.slider(
            "Observation location x",
            -5.0,
            5.0,
            step=0.1,
            key="ch7_candidate_x",
        )
    with candidate_columns[1]:
        candidate_y = st.slider(
            "Observed value y",
            -3.0,
            3.0,
            step=0.05,
            key="ch7_candidate_y",
        )
    with candidate_columns[2]:
        candidate_noise = st.slider(
            "Observation noise std",
            0.01,
            1.0,
            step=0.01,
            key="ch7_candidate_noise",
        )

    observations: list[Observation] = st.session_state["ch7_observations"]
    nearest_distance = min(abs(candidate_x - item["x"]) for item in observations)
    st.caption(
        f"The candidate is {nearest_distance:.2f} units from the nearest observation. "
        "The fixed RBF length scale is 1.0."
    )

    prediction_columns = st.columns(2)
    with prediction_columns[0]:
        mean_prediction = st.radio(
            "Predict the posterior mean",
            ["Move toward the new y", "Move away from the new y", "Stay unchanged"],
            index=None,
            key="ch7_mean_prediction",
        )
    with prediction_columns[1]:
        uncertainty_prediction = st.radio(
            "Predict latent-function uncertainty near this x",
            ["Decrease", "Increase", "Stay unchanged"],
            index=None,
            key="ch7_uncertainty_prediction",
        )

    button_columns = st.columns([1, 1, 3])
    with button_columns[0]:
        add_observation = st.button(
            "Add observation",
            type="primary",
            key="ch7_add_observation",
            width="stretch",
        )
    with button_columns[1]:
        reset_observations = st.button(
            "Reset",
            key="ch7_reset_observations",
            width="stretch",
        )

    if reset_observations:
        st.session_state["ch7_observations"] = _initial_observations()
        st.session_state["ch7_last_addition"] = None
        st.rerun()

    if add_observation:
        updated_observations = [*observations]
        updated_observations.append(
            {
                "x": float(candidate_x),
                "y": float(candidate_y),
                "noise_std": float(candidate_noise),
            }
        )
        st.session_state["ch7_observations"] = updated_observations
        st.session_state["ch7_last_addition"] = {
            "previous_count": len(observations),
            "x": float(candidate_x),
            "y": float(candidate_y),
            "noise_std": float(candidate_noise),
            "nearest_distance": float(nearest_distance),
            "mean_prediction": mean_prediction,
            "uncertainty_prediction": uncertainty_prediction,
        }
        st.rerun()

    X_test = np.linspace(-5.0, 5.0, 300)[:, None]
    x_test = X_test[:, 0]
    mean, _, latent_std = _conditioning_posterior(observations, X_test)

    previous_mean = None
    previous_std = None
    last_addition = st.session_state["ch7_last_addition"]
    if last_addition is not None:
        previous_count = int(last_addition["previous_count"])
        previous_observations = observations[:previous_count]
        previous_mean, _, previous_std = _conditioning_posterior(previous_observations, X_test)

    observation_x = np.array([item["x"] for item in observations], dtype=float)
    observation_y = np.array([item["y"] for item in observations], dtype=float)
    observation_noise = np.array([item["noise_std"] for item in observations], dtype=float)

    fig, ax = plt.subplots(figsize=(11, 4.5))
    ax.plot(
        x_test,
        np.sin(x_test),
        color="#2ca02c",
        linestyle="--",
        linewidth=1.5,
        label="simulated latent function",
    )
    if previous_mean is not None:
        ax.plot(
            x_test,
            previous_mean,
            color="0.45",
            linestyle=":",
            linewidth=2.0,
            label="posterior mean before last point",
        )
    ax.plot(x_test, mean, color="#1f77b4", linewidth=2.5, label="latent posterior mean")
    ax.fill_between(
        x_test,
        mean - 1.96 * latent_std,
        mean + 1.96 * latent_std,
        color="#1f77b4",
        alpha=0.2,
        label="95% latent-function interval",
    )
    ax.errorbar(
        observation_x,
        observation_y,
        yerr=2.0 * observation_noise,
        fmt="o",
        color="black",
        capsize=3,
        label="observations ± 2 noise std",
        zorder=4,
    )
    ax.scatter(
        [candidate_x],
        [candidate_y],
        marker="*",
        s=160,
        facecolors="none",
        edgecolors="#d62728",
        linewidths=1.8,
        label="candidate",
        zorder=5,
    )
    ax.set_title("Posterior after conditioning")
    ax.set_xlabel("x")
    ax.set_ylabel("latent value or observation")
    ax.legend(loc="upper right", fontsize=8, ncol=2)
    st.pyplot(fig)

    if last_addition is not None and previous_mean is not None and previous_std is not None:
        last_x = float(last_addition["x"])
        last_y = float(last_addition["y"])
        point_index = int(np.argmin(np.abs(x_test - last_x)))
        before_mean = float(previous_mean[point_index])
        after_mean = float(mean[point_index])
        before_std = float(previous_std[point_index])
        after_std = float(latent_std[point_index])
        location_description = (
            "near existing data"
            if float(last_addition["nearest_distance"]) <= 1.0
            else "far from existing data"
        )
        noise_description = (
            "Low noise makes the observation more trustworthy."
            if float(last_addition["noise_std"]) <= 0.15
            else "High noise makes the update more cautious."
        )
        st.success(
            f"Last update at x={last_x:.2f}: mean {before_mean:.2f} → {after_mean:.2f}; "
            f"latent std {before_std:.2f} → {after_std:.2f}."
        )
        st.markdown(
            f"The point was **{location_description}**. The mean moved toward its observed "
            f"value, {last_y:.2f}, because the kernel links nearby inputs. {noise_description}"
        )
        if last_addition["mean_prediction"] is not None:
            st.caption(
                "Your prediction — mean: "
                f"{last_addition['mean_prediction']}; uncertainty: "
                f"{last_addition['uncertainty_prediction'] or 'not selected'}."
            )

    st.write("Current observations:")
    st.dataframe(pd.DataFrame(observations), width="stretch", hide_index=True)


def render() -> None:
    st.header(TITLE)
    st.markdown(
        "A GP prediction changes when we add observations. Nearby, precise observations "
        "usually have the strongest effect."
    )
    _render_conditioning_experiment()

    with st.expander("The conditioning equations"):
        st.markdown("For noisy observations, the updated latent values follow")
        st.latex(
            r"f_* \mid X, y, X_* \sim \mathcal{N}\left("
            r"\mu_*, \Sigma_*"
            r"\right)."
        )
        st.markdown(r"- $X$: training inputs you already observed.")
        st.markdown(r"- $y$: observed outputs at those inputs.")
        st.markdown(r"- $X_*$: new query inputs where you want predictions.")
        st.markdown(r"- $\mu_*=\mathbb E[f_*\mid X,y]$: posterior mean of the latent function.")
        st.markdown(r"- $\Sigma_*$: posterior covariance of the latent function.")
        st.markdown(
            r"- `assumed observation noise std`: this is $\sigma_n$, and we use $\sigma_n^2 I$ "
            r"inside $K_{XX} + \sigma_n^2 I$."
        )
        st.markdown(
            r"The plotted band describes uncertainty in latent $f_*$. A future noisy "
            r"observation $y_*$ has extra measurement variance:"
        )
        st.latex(r"\operatorname{Var}(y_*)=\operatorname{Var}(f_*)+\sigma_n^2.")
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

    st.divider()
    st.subheader("Full data-set explorer")
    st.markdown(
        "Now change a complete data set and the model assumptions. "
        "Each control reruns the same conditioning calculation."
    )

    data_mode = st.radio(
        "Data source",
        ["Synthetic: sine", "Synthetic: tan", "Synthetic: mixed", "Custom table"],
        horizontal=True,
        key="ch7_data_source",
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
            data_noise_std = st.slider("simulated observation noise std", 0.0, 1.0, 0.2, 0.01)

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
        obs_noise_std = st.slider("assumed observation noise std", 0.001, 1.0, 0.2, 0.001)

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
    n_posterior_draws = st.slider("latent posterior function draws", 0, 10, 3, 1)

    X_train = x_train[:, None]
    X_test = np.linspace(x_min, x_max, n_test)[:, None]

    mean, cov, latent_std = gp_posterior_predictive(
        X_train,
        y_train,
        X_test,
        kernel_fn,
        noise_variance=obs_noise_std**2,
    )

    fig, ax = plt.subplots(figsize=(11, 4.5))
    x_test = X_test[:, 0]
    ax.scatter(x_train, y_train, color="black", label="observations y", zorder=3)
    ax.plot(x_test, mean, color="#1f77b4", linewidth=2.0, label="latent posterior mean")
    ax.fill_between(
        x_test,
        mean - 1.96 * latent_std,
        mean + 1.96 * latent_std,
        color="#1f77b4",
        alpha=0.2,
        label="95% latent-function interval",
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
            label="simulated latent function",
        )

    if n_posterior_draws > 0:
        draw_rng = np.random.default_rng(321)
        draws = sample_mvn(mean, cov, n_samples=n_posterior_draws, rng=draw_rng)
        for idx, draw in enumerate(draws):
            ax.plot(
                x_test,
                draw,
                linewidth=1.0,
                alpha=0.8,
                label=f"latent posterior draw {idx + 1}",
            )

    ax.set_title("Gaussian Process Regression")
    ax.set_xlabel("x")
    ax.set_ylabel("latent value or observation")
    ax.legend(loc="upper right", fontsize=8, ncol=2)
    st.pyplot(fig)
    st.caption(
        r"The blue band and posterior draws describe latent $f_*$. "
        "A future noisy observation would have additional measurement variance."
    )

    preview = pd.DataFrame(
        {"x*": x_test[:12], "latent mean": mean[:12], "latent std": latent_std[:12]}
    )
    st.write("Preview of latent-function predictions:")
    st.dataframe(preview, width="stretch")
