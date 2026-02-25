from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
import streamlit as st

from gp_book.gp_math import matern32_kernel, periodic_kernel, rbf_kernel, sample_gp_prior

TITLE = "6) Formal Gaussian Process Definition"


def render() -> None:
    st.header(TITLE)
    st.markdown(
        "Think of a Gaussian Process (GP) as a probability distribution over whole curves, "
        "not just single numbers. Before seeing data, the GP describes which kinds of curves "
        "look plausible."
    )
    st.markdown(
        'Formal definition (RW2006, Definition 2.1): '
        '"A Gaussian process is a collection of random variables, any finite number '
        'of which have a joint Gaussian distribution."'
    )
    st.latex(r"f(x) \sim \mathcal{GP}(m(x), k(x, x'))")
    st.markdown(
        "Important clarification: people often say 'any two points are Gaussian'. "
        "More precisely, any two points are jointly Gaussian as a 2D Gaussian, and this "
        "extends to any finite number of points."
    )
    st.latex(
        r"\begin{bmatrix}f(x_a)\\ f(x_b)\end{bmatrix} \sim "
        r"\mathcal{N}\left("
        r"\begin{bmatrix}m(x_a)\\ m(x_b)\end{bmatrix},"
        r"\begin{bmatrix}k(x_a,x_a)&k(x_a,x_b)\\k(x_b,x_a)&k(x_b,x_b)\end{bmatrix}"
        r"\right)"
    )
    st.markdown("Where:")
    st.markdown(r"- $m(x)$ is the mean function (the baseline trend).")
    st.markdown(r"- $k(x, x')$ is the kernel/covariance function (how points are related).")
    st.markdown(
        "To actually compute with a GP, we choose a finite list of inputs "
        r"$X = [x_1, \dots, x_n]$:"
    )
    st.latex(r"\mathbf{f} = [f(x_1), \dots, f(x_n)]^\top \sim \mathcal{N}(\mathbf{m}, K)")
    st.markdown(r"with $m_i = m(x_i)$ and $K_{ij} = k(x_i, x_j)$.")
    st.markdown(
        "In plain words: evaluate the mean at each point to get a mean vector, and evaluate "
        "the kernel for every pair of points to get a covariance matrix."
    )
    st.markdown(
        "Reference for this definition and notation: Rasmussen, C.E. and Williams, C.K.I. "
        "(2006), *Gaussian Processes for Machine Learning*, MIT Press; free online PDF: "
        "https://gaussianprocess.org/gpml/chapters/RW.pdf"
    )

    with st.expander("Slow walkthrough: from prior to posterior (behind the scenes)"):
        st.markdown("1. Start with prior assumptions")
        st.markdown(
            r"Choose $m(x)$ and $k(x, x')$. This defines a prior over functions "
            r"$f \sim \mathcal{GP}(m, k)$."
        )
        st.markdown("2. Observe noisy data")
        st.latex(r"y_i = f(x_i) + \epsilon_i, \quad \epsilon_i \sim \mathcal{N}(0, \sigma_n^2)")
        st.markdown("3. Build kernel blocks")
        st.latex(
            r"K_{XX},\quad K_{X X_*},\quad K_{X_* X},\quad K_{X_* X_*}"
        )
        st.markdown(
            r"Notation for $*$: the star means 'test/query points'. "
            r"So $X$ is observed inputs and $X_*$ is prediction inputs."
        )
        st.latex(
            r"K_{XX}=\begin{bmatrix}k(x_1,x_1)&\cdots&k(x_1,x_n)\\"
            r"\vdots&\ddots&\vdots\\"
            r"k(x_n,x_1)&\cdots&k(x_n,x_n)\end{bmatrix}"
        )
        st.latex(
            r"K_{X_*X}=\begin{bmatrix}k(x^*_1,x_1)&\cdots&k(x^*_1,x_n)\\"
            r"\vdots&\ddots&\vdots\\"
            r"k(x^*_m,x_1)&\cdots&k(x^*_m,x_n)\end{bmatrix}"
        )
        st.markdown("4. Condition the joint Gaussian to get predictions")
        st.latex(
            r"\begin{bmatrix}y\\ f_*\end{bmatrix}\sim\mathcal{N}\left("
            r"\begin{bmatrix}m(X)\\ m(X_*)\end{bmatrix},"
            r"\begin{bmatrix}K_{XX}+\sigma_n^2I&K_{XX_*}\\K_{X_*X}&K_{X_*X_*}\end{bmatrix}"
            r"\right)"
        )
        st.latex(
            r"\mu_* = m(X_*) + K_{X_*X}(K_{XX} + \sigma_n^2 I)^{-1}(y - m(X))"
        )
        st.latex(
            r"\Sigma_* = K_{X_*X_*} - K_{X_*X}(K_{XX} + \sigma_n^2 I)^{-1}K_{XX_*}"
        )
        st.markdown(
            "These two equations are the engine of GP regression: "
            r"$\mu_*$ is the best estimate and $\Sigma_*$ is uncertainty."
        )
        st.markdown("5. Why runtime grows quickly")
        st.markdown(
            r"Inverting/factorizing the $n \times n$ matrix is about $O(n^3)$ time and "
            r"$O(n^2)$ memory, which is why large datasets need approximations."
        )

    col1, col2 = st.columns(2)
    with col1:
        mean_name = st.selectbox("Mean function", ["Zero mean", "Linear mean"], index=0)
    with col2:
        kernel_name = st.selectbox("Kernel", ["RBF", "Matern 3/2", "Periodic"], index=0)

    col3, col4, col5 = st.columns(3)
    with col3:
        length_scale = st.slider("length-scale", 0.1, 3.0, 1.0, 0.05)
    with col4:
        variance = st.slider("variance", 0.1, 4.0, 1.0, 0.05)
    with col5:
        period = st.slider("period (Periodic only)", 0.2, 5.0, 1.0, 0.05)

    if kernel_name == "RBF":
        kernel_fn = lambda a, b: rbf_kernel(a, b, length_scale=length_scale, variance=variance)
    elif kernel_name == "Matern 3/2":
        kernel_fn = lambda a, b: matern32_kernel(a, b, length_scale=length_scale, variance=variance)
    else:
        kernel_fn = lambda a, b: periodic_kernel(
            a, b, length_scale=length_scale, variance=variance, period=period
        )

    x = np.linspace(-5.0, 5.0, 120)[:, None]
    if mean_name == "Zero mean":
        mean = np.zeros(x.shape[0], dtype=float)
    else:
        mean = 0.25 * x[:, 0]

    rng = np.random.default_rng(28)
    samples = sample_gp_prior(x, kernel_fn, n_samples=6, rng=rng, mean=mean)

    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(x[:, 0], mean, color="black", linewidth=2.0, linestyle="--", label="mean m(x)")
    for idx, sample in enumerate(samples):
        ax.plot(x[:, 0], sample, alpha=0.85, label=f"sample {idx + 1}")
    ax.set_title("Finite draws implied by the GP definition")
    ax.set_xlabel("x")
    ax.set_ylabel("f(x)")
    ax.legend(loc="upper right", ncol=2, fontsize=8)
    st.pyplot(fig)

    st.code(
        "K = k(X, X)\n"
        "f ~ N(m(X), K)\n"
        "# This finite MVN is the operational GP view",
        language="python",
    )
