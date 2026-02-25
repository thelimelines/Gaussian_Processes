from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st

TITLE = "4) A Sampled Function Is a Vector"


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
        "Gaussian processes are about random functions, but computation happens on finite "
        "grids. This chapter makes that bridge explicit."
    )
    st.markdown(r"If we evaluate a function at inputs $x_1, \dots, x_n$, we get:")
    st.latex(r"\mathbf{f} = [f(x_1), f(x_2), \dots, f(x_n)]^\top")
    st.markdown("Notation:")
    st.markdown(r"- $x_i$: the $i$th input location.")
    st.markdown(r"- $f(x_i)$: function value at that input.")
    st.markdown(r"- $\mathbf{f}$: stacked vector of function values.")
    st.markdown(
        'So a "random function" can be understood as a random vector at any finite set of inputs. '
        "This is the bridge to Gaussian processes."
    )

    col1, col2, col3 = st.columns(3)
    with col1:
        fn_name = st.selectbox("Base function", ["sine", "cosine", "quadratic", "linear"], index=0)
    with col2:
        n_points = st.slider("Number of x points", 5, 120, 25, 1)
    with col3:
        noise_std = st.slider("Noise std", 0.0, 1.0, 0.1, 0.01)

    x = np.linspace(-3.5, 3.5, n_points)
    base = _base_function(x, fn_name)
    rng = np.random.default_rng(3)
    y = base + noise_std * rng.standard_normal(n_points)

    fig, ax = plt.subplots(figsize=(9, 4))
    ax.plot(x, y, marker="o", linewidth=1.2, markersize=3, label="vector entries")
    ax.set_xlabel("x")
    ax.set_ylabel("f(x)")
    ax.set_title("A finite function sample is a vector")
    ax.legend(loc="upper right")
    st.pyplot(fig)

    preview = pd.DataFrame({"x_i": x[:10], "f(x_i)": y[:10]})
    st.write("First 10 entries of the vector representation:")
    st.dataframe(preview, width="stretch")
    st.info(
        "Bridge to Chapter 1.2: in a GP, covariance between entries of this vector "
        r"is specified by a kernel $k(x_i, x_j)$."
    )

    st.code(
        "x = np.linspace(-3.5, 3.5, n)\n"
        "f_vec = f(x)  # [f(x1), ..., f(xn)]",
        language="python",
    )
