# Gaussian Processes Interactive Explainer

Standalone interactive "book-like" app for learning Gaussian processes with:
- Markdown narrative
- LaTeX equations
- Python code snippets
- Interactive controls (sliders, table editor)
- Graph outputs

## Structure

```
app.py
chapters/
  01_standard_normal/page.py
  02_general_normal_with_sliders/page.py
  03_multivariate_normal_and_cholesky/page.py
  04_functions_as_vectors/page.py
  05_kernels_and_covariance/page.py
  06_formal_gp_definition/page.py
  07_interactive_gp_regressor/page.py
  08_sklearn_gp_autotuning/page.py
src/gp_book/gp_math.py
```

Each chapter is a folder with a `page.py` renderer, so you can expand the book page-by-page.

## Setup (uv)

```powershell
uv venv
uv sync
```

## Run

```powershell
uv run streamlit run app.py
```

## Chapter Outline

1. Standard normal distribution
2. General normal with interactive mean/variance controls
3. Multivariate normal and Cholesky factor
4. Why sampled functions can be treated as vectors
5. Kernel covariance functions (RBF, Matern 3/2, Periodic) with references
6. Formal GP definition with visual finite draws
7. Interactive GP regressor with synthetic functions (`sine`, `tan`, `mixed`) or custom table data
8. Scikit-learn GP regressor with automatic hyperparameter tuning via log-marginal-likelihood optimization
