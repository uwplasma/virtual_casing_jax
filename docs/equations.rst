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
   + \nabla \times G[\mathbf{B}\times\mathbf{n}](\mathbf{r}),

where ``G`` denotes the Laplace single-layer potential:

.. math::

   G[\sigma](\mathbf{r}) = \frac{1}{4\pi}\int_{\Gamma}
   \frac{\sigma(\mathbf{r}')}{\lVert \mathbf{r} - \mathbf{r}' \rVert}
   \, d a(\mathbf{r}').

The interior field is obtained by reversing the signs:

.. math::

   \mathbf{B}_{\mathrm{int}}(\mathbf{r}) = \frac{1}{2}\mathbf{B}(\mathbf{r})
   - \nabla G[\mathbf{B}\cdot\mathbf{n}](\mathbf{r})
   - \nabla \times G[\mathbf{B}\times\mathbf{n}](\mathbf{r}).

Internal vs External Off-Surface
--------------------------------

For off-surface targets, the jump term is absent. The external and
internal fields are:

.. math::

   \mathbf{B}_{\mathrm{ext}}(\mathbf{r}) =
   \nabla G[\mathbf{B}\cdot\mathbf{n}](\mathbf{r})
   + \nabla \times G[\mathbf{B}\times\mathbf{n}](\mathbf{r}),

.. math::

   \mathbf{B}_{\mathrm{int}}(\mathbf{r}) =
   -\nabla G[\mathbf{B}\cdot\mathbf{n}](\mathbf{r})
   - \nabla \times G[\mathbf{B}\times\mathbf{n}](\mathbf{r}).

The off-surface gradients are obtained by differentiating these
expressions and using the second-derivative Laplace kernel.

The two vector layer potentials used are:

- A Laplace single-layer gradient (``FxdU`` in the reference code)
- A Biot-Savart single-layer (``FxU``)

The kernels are (up to scaling) derivatives of ``1 / |r|``.

Kernel Formulas
---------------

Let ``r = x - x'`` be the displacement from a source to a target point.
The kernels implement the current `BIEST
<https://github.com/dmalhotra/BIEST>`_ convention with a ``1/(4*pi)`` factor.

Laplace single-layer:

.. math::

   G(r) = \frac{1}{4\pi \lVert r \rVert}

Gradient of single-layer:

.. math::

   \nabla G(r) = -\frac{r}{4\pi \lVert r \rVert^3}

Second derivatives:

.. math::

   \partial_i\partial_j G(r) = \frac{1}{4\pi}\left(-\delta_{ij}\lVert r \rVert^{-3} + 3 r_i r_j \lVert r \rVert^{-5}\right)

Biot-Savart (single-layer):

.. math::

   \mathbf{K}(r) = \frac{1}{4\pi} \frac{\mathbf{f} \times r}{\lVert r \rVert^3}

The outward-normal Laplace double layer is

.. math::

   D[\mu](x) = \frac{1}{4\pi}\int_\Gamma
   \frac{n(x')\cdot(x-x')}{\lVert x-x'\rVert^3}\mu(x')\,da(x').

This sign gives ``D[1] = -1`` inside a closed surface, ``0`` outside, and the
principal value ``-1/2`` on the surface. These identities are tested directly
on a torus and follow the SCTL convention adopted by BIEST in August 2026.

The derivative ``FxdU`` for Biot-Savart is implemented explicitly to match the
reference code in ``biest/kernel.hpp``.

Off-Surface Evaluation
----------------------

For off-surface targets ``r``:

.. math::

   \mathbf{B}_{\mathrm{ext}}(\mathbf{r}) = \nabla G[\mathbf{B}\cdot\mathbf{n}](\mathbf{r})
   + \nabla \times G[\mathbf{B}\times\mathbf{n}](\mathbf{r}).

The singular ``+1/2`` jump term is absent off the surface.

Field-Period Symmetry
---------------------

The surface and field may be defined on a half field period using
stellarator symmetry. The toroidal grid is shifted by half a grid point
for the symmetric case.

A full description of the grid conventions is in the reference code
and reproduced in the :doc:`numerics` section.
