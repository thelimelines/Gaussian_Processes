from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
import streamlit as st
from scipy.stats import norm

from gp_book.gp_math import normal_pdf, standard_normal_pdf

TITLE = "2) General Normal With Sliders"


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
    st.markdown(r"where $\phi(\cdot)$ is the standard normal density.")
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
        x0 = st.slider("Query x", -8.0, 8.0, 0.0, 0.1)

    x = np.linspace(-8.0, 8.0, 700)
    y = normal_pdf(x, mu, sigma)
    y_std = standard_normal_pdf(x)

    fig, ax = plt.subplots(figsize=(9, 4))
    ax.plot(x, y, label=f"N({mu:.1f}, {sigma:.1f}^2)")
    ax.plot(x, y_std, linestyle="--", label="N(0, 1)")
    ax.axvline(x0, color="black", linewidth=1.0, alpha=0.5)
    ax.set_xlabel("x")
    ax.set_ylabel("density")
    ax.set_title("General normal vs standard normal")
    ax.legend(loc="upper right")
    st.pyplot(fig)

    z0 = (x0 - mu) / sigma
    cdf_val = float(norm.cdf(x0, loc=mu, scale=sigma))

    st.latex(r"z = \frac{x - \mu}{\sigma}")
    st.write(f"For x = {x0:.2f}: z = {z0:.3f}, P(X <= x) = {cdf_val:.4f}")
    st.info(
        "Bridge to Chapter 0.3: once we stack multiple Gaussian variables into a vector, "
        "covariance structure appears as a matrix."
    )

    st.code(
        "z = (x - mu) / sigma\n"
        "p = norm.cdf(x, loc=mu, scale=sigma)",
        language="python",
    )
