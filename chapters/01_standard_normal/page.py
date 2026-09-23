from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
import streamlit as st
from scipy.stats import norm

from gp_book.gp_math import standard_normal_pdf

TITLE = "1) Standard Normal Distribution"


def render() -> None:
    st.header(TITLE)
    st.markdown(
        "This chapter introduces the base object used throughout the book: a single "
        "Gaussian random variable."
    )
    st.markdown("The standard normal is:")
    st.latex(
        r"Z \sim \mathcal{N}(0, 1), \quad "
        r"\phi(z) = \frac{1}{\sqrt{2\pi}} \exp\left(-\frac{z^2}{2}\right)."
    )
    st.markdown(
        r"Here, $\phi(\cdot)$ denotes the standard normal probability density "
        r"function (pdf)."
    )
    st.markdown("Notation:")
    st.markdown(r"- $Z$: random variable.")
    st.markdown(r"- $z$: a realized value of $Z$.")
    st.markdown(r"- $\phi(z)$: standard normal density evaluated at $z$.")

    col1, col2 = st.columns(2)
    with col1:
        x_min, x_max = st.slider("Plot range", -8.0, 8.0, (-4.0, 4.0), 0.1)
    with col2:
        a, b = st.slider("Interval for P(a <= Z <= b)", -4.0, 4.0, (-1.0, 1.0), 0.1)

    x = np.linspace(x_min, x_max, 600)
    y = standard_normal_pdf(x)
    prob = float(norm.cdf(b) - norm.cdf(a))

    fig, ax = plt.subplots(figsize=(9, 4))
    ax.plot(x, y, label="Standard normal pdf")
    mask = (x >= a) & (x <= b)
    ax.fill_between(x[mask], 0.0, y[mask], alpha=0.25, label=f"P({a:.1f} <= Z <= {b:.1f})")
    ax.set_title("Standard normal density")
    ax.set_xlabel("z")
    ax.set_ylabel("density")
    ax.legend(loc="upper right")
    st.pyplot(fig)

    st.metric("Interval probability", f"{prob:.4f}")
    st.info(
        "Bridge to Chapter 0.2: we next move from the standard Gaussian "
        r"$\mathcal{N}(0,1)$ to the general form $\mathcal{N}(\mu,\sigma^2)$."
    )

    st.code(
        "from scipy.stats import norm\n"
        "prob = norm.cdf(b) - norm.cdf(a)\n"
        "print(prob)",
        language="python",
    )
