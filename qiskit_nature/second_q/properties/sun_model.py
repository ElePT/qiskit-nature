# This code is part of Qiskit.
#
# (C) Copyright IBM 2022.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

"""The Wilson model for a non-abelian SU(n) gauge theory and a two component spinor
field ona a 1 dimensional lattice."""
import logging
from fractions import Fraction
from typing import Optional

import numpy as np

from qiskit_nature.operators.second_quantization import SpinOp
from qiskit_nature.problems.second_quantization.lattice.lattices import Lattice

from .lattice_model import LatticeModel

logger = logging.getLogger(__name__)


class SUNModel(LatticeModel):
    """The SUN model."""

    def coupling_matrix(self) -> np.ndarray:
        """Return the coupling matrix."""
        return self.interaction_matrix()

    @classmethod
    def uniform_parameters(
        cls,
        lattice: Lattice,
        uniform_interaction: complex,
        uniform_onsite_potential: complex,
    ) -> "SUNModel":
        """Set a uniform interaction parameter and on-site potential over the input lattice.

        Args:
            lattice: Lattice on which the model is defined.
            uniform_interaction: The interaction parameter.
            uniform_onsite_potential: The on-site potential.

        Returns:
            The Lattice model with uniform parameters.
        """
        return cls(
            cls._generate_lattice_from_uniform_parameters(
                lattice, uniform_interaction, uniform_onsite_potential
            )
        )

    @classmethod
    def from_parameters(
        cls,
        interaction_matrix: np.ndarray,
    ) -> "SUNModel":
        """Return the Hamiltonian of the Lattice model
        from the given interaction matrix and on-site interaction.

        Args:
            interaction_matrix: A real or complex valued square matrix.

        Returns:
            LatticeModel: The Lattice model generated from the given interaction
                matrix and on-site interaction.

        Raises:
            ValueError: If the interaction matrix is not square matrix, it is invalid.
        """
        return cls(cls._generate_lattice_from_parameters(interaction_matrix))

    def second_q_ops(self, display_format: Optional[str] = None) -> SpinOp:
        """Return the Staggered Hamiltonian.

        Args:
            display_format: Not supported for Spin operators. If specified, it will be ignored.

        Returns:
            SpinOp: The Hamiltonian of the Ising model.
        """

        return SpinOp(ham, spin=Fraction(1, 2), register_length=register_length)


########################
def string_hamilton_staggered(lattice, group_dim, params, output="abstract"):
    """
    Constructs the `Staggered` Hamiltonian for for a non-abelian SU(n) gauge theory and
    a one component spinor field ona a 1 dimensional lattice.
    (With some minor changes, arbitrary compact continuous gauge groups can be simulated)

    Args:
        lattice (Lattice): The lattice object (must be of dimension 1)
        group_dim (int): The degree of the special unitary group SU(n)
        params (dict): The dictonary of physical parameters for the Hamiltonian. Must contain:
            t (float): A strength parameter for the hopping
            a (float): The lattice spacing.
            m (float): The bare mass of the Wilson fermions (mass parameter in the Hamiltonian)
            g (float): The coupling constant
        output(str): The desired output format.
            Must be one of ['abstract', 'qiskit', 'qutip', 'matrix', 'spmatrix']

    Returns:
        FermionicOperator or qiskit.aqua.Operator or qutip.Qobj or np.ndarray
    """
    assert lattice.ndim == 1, "The lattice object must be of dimension 1"
    assert lattice.boundary_cond == "closed", "The Lattice boundary conditions must be 'closed'"
    assert isinstance(group_dim, (int, np.integer)) and group_dim > 0, (
        "The degree `group_dim` of the group SU(n) " "must be a positive integer"
    )
    # 1. Extract the relevant parameters
    t = params["t"]
    a = params["a"]
    m = params["m"]
    g = params["g"]

    ncolors = group_dim

    # 2. Build the mass term
    mass_terms = []
    for color_idx in range(ncolors):
        for site in lattice.sites:
            # define the index for the phase depending on the summation index `n`
            n = site[0] + 1

            mass_summand = (
                (-1) ** n
                * psidag_color(
                    site,
                    color_idx,
                    ncolors=ncolors,
                    lattice=lattice,
                    spinor_component=0,
                    ncomponents=1,
                )
                * psi_color(
                    site,
                    color_idx,
                    ncolors=ncolors,
                    lattice=lattice,
                    spinor_component=0,
                    ncomponents=1,
                )
            )
            mass_terms.append(mass_summand)

    # 2.1 Finalizing the mass_terms
    mass = m * operator_sum(mass_terms)
    if not output == "abstract":
        mass = mass.to_qubit_operator(output=output)

    # 3. Build the hopping term
    hopping_terms = []
    for color_idx in range(ncolors):
        for site in lattice.sites:
            # Treat closed boundary conditions (skip term if edge is at boundary)
            site_is_at_boundary = lattice.is_boundary_along(site, 0, direction="positive")
            if site_is_at_boundary:
                continue

            next_site = lattice.project(site + 1)
            hopping_summand = psidag_color(
                site, color_idx, ncolors=ncolors, lattice=lattice, spinor_component=0, ncomponents=1
            ) * psi_color(
                next_site,
                color_idx,
                ncolors=ncolors,
                lattice=lattice,
                spinor_component=0,
                ncomponents=1,
            )
            hopping_terms.append(hopping_summand)
            hopping_terms.append(hopping_summand.dag())

    # 3.1 Finalizing the mass_terms
    hopping = t / (2 * a) * operator_sum(hopping_terms)
    if not output == "abstract":
        hopping = hopping.to_qubit_operator(output=output)

    # 4. Build the longrange-interaction term
    # 4.1 Get a list of the generators of the gauge group SU(n) in the fundamental representation
    generators = get_generators_SU(ncolors)
    ngenerators = len(generators)

    # 4.2 Construct the summands
    longrange_terms = []
    for site in lattice.sites:

        # Treat closed boundary conditions (skip term if edge is at boundary)
        site_is_at_boundary = lattice.is_boundary_along(site, 0, direction="positive")
        if site_is_at_boundary:
            continue

        # Sum over the conserved charges
        for gen_idx in range(ngenerators):
            longrange_inner = []
            for inner_site in lattice.sites:

                # the summation index `long_site` goes from 0 to `site`
                if inner_site[0] > site[0]:
                    break

                inner_summand = conserved_charge(
                    inner_site, generators[gen_idx], lattice, ncomponents=1
                )
                longrange_inner.append(inner_summand)

            longrange_summand = (operator_sum(longrange_inner)) ** 2
            longrange_terms.append(longrange_summand)

    # 4.3 Finalizing the longrange_terms
    longrange = g**2 * a / 2.0 * operator_sum(longrange_terms)
    if not output == "abstract":
        longrange = longrange.to_qubit_operator(output=output)

    return mass + hopping + longrange
