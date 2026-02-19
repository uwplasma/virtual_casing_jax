# virtual_casing_jax

JAX implementation of the virtual casing principle with high-order singular quadrature.

Status: early development with validated surface geometry and kernel operators,
plus baseline direct-sum boundary integrals and singular-corrected Laplace
FxdU on-surface evaluation (off-surface uses upsampled quadrature).

Planned features:
- On-surface and off-surface virtual casing evaluation.
- High-order singular quadrature matching BIEST.
- JAX autodiff support.

Docs
----

Sphinx documentation lives in `docs/` and is configured for ReadTheDocs.
It includes the equations, numerics, implementation details, and validation
strategy. Run locally:

```bash
pip install -r docs/requirements.txt
sphinx-build -b html docs docs/_build/html
```
