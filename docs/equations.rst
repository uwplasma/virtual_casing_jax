Equations
=========

Virtual Casing Principle
------------------------

Let ``Gamma`` be a closed toroidal surface with outward unit normal ``n``.
Let ``B`` be the total magnetic field on ``Gamma``, decomposed into
interior and exterior contributions:

.. math::

   \mathbf{B} = \mathbf{B}_{\mathrm{int}} + \mathbf{B}_{\mathrm{ext}}.

The virtual casing principle provides an explicit expression for
``B_ext`` on the surface in terms of singular layer potentials:

.. math::

   \mathbf{B}_{\mathrm{ext}}(\mathbf{r}) = \frac{1}{2}\mathbf{B}(\mathbf{r})
   + \nabla G[\mathbf{B}\cdot\mathbf{n}](\mathbf{r})
   + \nabla \times G[\mathbf{n}\times\mathbf{B}](\mathbf{r}),

where ``G`` denotes the Laplace single-layer potential:

.. math::

   G[\sigma](\mathbf{r}) = \frac{1}{4\pi}\int_{\Gamma}
   \frac{\sigma(\mathbf{r}')}{\lVert \mathbf{r} - \mathbf{r}' \rVert}
   \, d a(\mathbf{r}').

The interior field is obtained by reversing the signs:

.. math::

   \mathbf{B}_{\mathrm{int}}(\mathbf{r}) = \frac{1}{2}\mathbf{B}(\mathbf{r})
   - \nabla G[\mathbf{B}\cdot\mathbf{n}](\mathbf{r})
   - \nabla \times G[\mathbf{n}\times\mathbf{B}](\mathbf{r}).

The two vector layer potentials used are:

- A Laplace single-layer gradient (``FxdU`` in the reference code)
- A Biot-Savart single-layer (``FxU``)

The kernels are (up to scaling) derivatives of ``1 / |r|``.

Off-Surface Evaluation
----------------------

For off-surface targets ``r``:

.. math::

   \mathbf{B}_{\mathrm{ext}}(\mathbf{r}) = \nabla G[\mathbf{B}\cdot\mathbf{n}](\mathbf{r})
   + \nabla \times G[\mathbf{n}\times\mathbf{B}](\mathbf{r}).

The singular ``+1/2`` jump term is absent off the surface.

Field-Period Symmetry
---------------------

The surface and field may be defined on a half field period using
stellarator symmetry. The toroidal grid is shifted by half a grid point
for the symmetric case.

A full description of the grid conventions is in the reference code
and reproduced in the :doc:`numerics` section.
