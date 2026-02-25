from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
import streamlit as st

from gp_book.gp_math import matern32_kernel, periodic_kernel, rbf_kernel, sample_gp_prior

TITLE = "5) Kernel Covariance Functions"


def render() -> None:
    st.header(TITLE)
    st.markdown(
        "Let's start with a simple idea: points that are close together in input space "
        "usually have similar outputs. A kernel is the rule that turns that idea into math."
    )
    st.markdown("Notation:")
    st.markdown(r"- $x, x'$: two input locations.")
    st.markdown(r"- $k(x, x')$: covariance between $f(x)$ and $f(x')$.")
    st.markdown(r"- $X=[x_1,\dots,x_n]$: stacked training inputs.")
    st.markdown(r"- $K(X,X)$: kernel matrix with entries $K_{ij}=k(x_i,x_j)$.")
    st.markdown(r"A kernel $k(x, x')$ defines covariance between function values:")
    st.latex(r"\text{Cov}(f(x), f(x')) = k(x, x')")
    st.markdown(
        r"Given inputs $X$, the kernel builds the covariance matrix $K(X, X)$, "
        r"which in turn defines a multivariate Gaussian over $\mathbf{f}$."
    )
    st.markdown(
        "Read it like this: each cell in the matrix asks, "
        r"'how much should $f(x_i)$ move together with $f(x_j)$?' "
        "Large value means strongly linked; small value means weakly linked."
    )
    st.markdown("Expanded matrix form:")
    st.latex(
        r"K(X,X)=\begin{bmatrix}"
        r"k(x_1,x_1)&\cdots&k(x_1,x_n)\\"
        r"\vdots&\ddots&\vdots\\"
        r"k(x_n,x_1)&\cdots&k(x_n,x_n)"
        r"\end{bmatrix}"
    )

    col1, col2 = st.columns(2)
    with col1:
        kernel_name = st.selectbox("Kernel", ["RBF", "Matern 3/2", "Periodic"], index=0)
    with col2:
        n_points = st.slider("Input grid size", 20, 160, 80, 5)

    col3, col4, col5 = st.columns(3)
    with col3:
        length_scale = st.slider("length-scale", 0.1, 3.0, 1.0, 0.05)
    with col4:
        variance = st.slider("variance", 0.1, 3.0, 1.0, 0.05)
    with col5:
        period = st.slider("period (Periodic only)", 0.2, 5.0, 1.0, 0.05)

    if kernel_name == "RBF":
        kernel_fn = lambda a, b: rbf_kernel(a, b, length_scale=length_scale, variance=variance)
        st.latex(
            r"k_{\text{RBF}}(x, x') = \sigma^2 \exp\left(-\frac{\|x-x'\|^2}{2\ell^2}\right)"
        )
        st.markdown(
            r"RBF details: $\ell$ controls how quickly correlation fades with distance, "
            r"and $\sigma^2$ sets the vertical scale."
        )
    elif kernel_name == "Matern 3/2":
        kernel_fn = lambda a, b: matern32_kernel(a, b, length_scale=length_scale, variance=variance)
        st.latex(
            r"k_{\nu=3/2}(r) = \sigma^2\left(1 + \frac{\sqrt{3}r}{\ell}\right)"
            r"\exp\left(-\frac{\sqrt{3}r}{\ell}\right)"
        )
        st.markdown(
            "Matern 3/2 allows rougher functions than RBF. "
            "It is a good default when perfectly smooth curves feel too optimistic."
        )
    else:
        kernel_fn = lambda a, b: periodic_kernel(
            a,
            b,
            length_scale=length_scale,
            variance=variance,
            period=period,
        )
        st.latex(
            r"k_{\text{per}}(x, x') = \sigma^2 \exp\left(-\frac{2\sin^2(\pi|x-x'|/p)}{\ell^2}\right)"
        )
        st.markdown(
            r"Periodic adds repeating structure. The parameter $p$ is the repeat period."
        )

    x = np.linspace(-4.0, 4.0, n_points)[:, None]
    K = kernel_fn(x, x)

    fig, ax = plt.subplots(figsize=(5, 4))
    im = ax.imshow(K, origin="lower", aspect="auto")
    ax.set_title("Covariance matrix K(X, X)")
    fig.colorbar(im, ax=ax, shrink=0.85)
    st.pyplot(fig)

    rng = np.random.default_rng(14)
    samples = sample_gp_prior(x, kernel_fn, n_samples=5, rng=rng)

    fig2, ax2 = plt.subplots(figsize=(9, 4))
    for idx, sample in enumerate(samples):
        ax2.plot(x[:, 0], sample, label=f"sample {idx + 1}")
    ax2.set_title(f"Prior function samples ({kernel_name})")
    ax2.set_xlabel("x")
    ax2.set_ylabel("f(x)")
    ax2.legend(loc="upper right", ncol=2, fontsize=8)
    st.pyplot(fig2)

    st.markdown(
        """
References:
- Rasmussen, C.E. and Williams, C.K.I. (2006), *Gaussian Processes for Machine Learning*, MIT Press.
- Duvenaud, D. (2014), *The Kernel Cookbook*.
"""
    )
