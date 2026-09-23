from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
import streamlit as st
from scipy.stats import norm

from gp_book.gp_math import normal_pdf, standard_normal_pdf

TITLE = "2) General Normal"


def render() -> None:
    st.header(TITLE)
    st.markdown(
        "Here we generalize the standard normal by introducing location and scale. "
        "This is still a single random variable, but with tunable mean and variance."
    )
    st.markdown("General normal:")
    st.latex(
        r"X \sim \mathcal{N}(\mu, \sigma^2), \quad "
        r"p(x) = \frac{1}{\sigma} \phi\left(\frac{x - \mu}{\sigma}\right)"
    )
    st.markdown(
        r"Reminder: $\phi(z) = \frac{1}{\sqrt{2\pi}} "
        r"\exp\left(-\frac{z^2}{2}\right)$ is the standard normal density."
    )
    st.markdown("Notation:")
    st.markdown(r"- $X$: random variable.")
    st.markdown(r"- $\mu$: mean (location).")
    st.markdown(r"- $\sigma^2$: variance, with $\sigma > 0$ as standard deviation.")
    st.markdown(r"- $x$: value at which density/CDF are evaluated.")

    col1, col2, col3 = st.columns(3)
    with col1:
        mu = st.slider("Mean (mu)", -4.0, 4.0, 0.0, 0.1)
    with col2:
        sigma = st.slider("Std dev (sigma)", 0.2, 4.0, 1.0, 0.1)
    with col3:
        a, b = st.slider("Interval for P(a <= X <= b)", -4.0, 4.0, (-1.0, 1.0), 0.1)

    x = np.linspace(-8.0, 8.0, 700)
    y = normal_pdf(x, mu, sigma)
    y_std = standard_normal_pdf(x)

    fig, ax = plt.subplots(figsize=(9, 4))
    ax.plot(x, y, label=f"N({mu:.1f}, {sigma:.1f}^2)")
    ax.plot(x, y_std, linestyle="--", label="N(0, 1)")
    mask = (x >= a) & (x <= b)
    ax.fill_between(x[mask], 0.0, y[mask], alpha=0.25, label=f"P({a:.1f} <= X <= {b:.1f})")
    ax.set_xlabel("x")
    ax.set_ylabel("density")
    ax.set_title("General normal vs standard normal")
    ax.legend(loc="upper right")
    st.pyplot(fig)

    interval_prob = float(norm.cdf(b, loc=mu, scale=sigma) - norm.cdf(a, loc=mu, scale=sigma))
    standard_interval_prob = float(norm.cdf(b) - norm.cdf(a))

    probability_col, standard_probability_col = st.columns(2)
    with probability_col:
        st.metric("P(a <= X <= b)", f"{interval_prob:.4f}")
    with standard_probability_col:
        st.metric("P(a <= Z <= b) for N(0, 1)", f"{standard_interval_prob:.4f}")
    st.info(
        "Bridge to Chapter 0.3: once we stack multiple Gaussian variables into a vector, "
        "covariance structure appears as a matrix."
    )

    st.code(
        "p = norm.cdf(b, loc=mu, scale=sigma) - norm.cdf(a, loc=mu, scale=sigma)",
        language="python",
    )
