Parity Datasets
===============

This project uses small parity datasets generated from the reference
``virtual-casing`` code and SIMSOPT examples/tests.

Dataset Sources
---------------

- ``virtual-casing`` test utilities (e.g. ``VirtualCasingTestData``)
- SIMSOPT tests in ``simsopt/tests/mhd/test_virtual_casing.py``

Current Dataset Prefixes
------------------------

- ``case_vc``: external field and GradB parity (small internal test case).
- ``case_vc_int``: internal field and GradB parity (small internal test case).
- ``case_vc_large``: external field and GradB parity (larger Nt/Np).
- ``case_simsopt``: external field and GradB parity (SIMSOPT VMEC case).
- ``case_simsopt_int``: internal field and GradB parity (SIMSOPT VMEC case).
- ``case_simsopt_large``: external field and GradB parity (larger VMEC grids).
- ``case_vc_w7x``: external field and GradB parity (W7-X geometry, NFP=5).
- ``case_vc_w7x_large``: external field and GradB parity (larger W7-X grid).
- ``case_vc``/``case_vc_large``: include off-surface B/GradB targets.
- ``case_testdata_axisym``: VirtualCasingTestData B/GradB parity (axisymmetric).

Dataset Format
--------------

Each dataset consists of binary arrays with JSON metadata:

- ``<prefix>_<name>.bin``
- ``<prefix>_<name>.json``

The metadata contains ``dtype`` and ``shape`` fields to reconstruct
arrays in NumPy or JAX.

Reproduction
------------

Parity datasets are generated with:

- ``VC_DUMP_DIR`` environment variable for C++ dumps.
- ``tools/make_parity_dumps.py`` in this repository.

The datasets are intentionally small to keep the repository size
reasonable.
