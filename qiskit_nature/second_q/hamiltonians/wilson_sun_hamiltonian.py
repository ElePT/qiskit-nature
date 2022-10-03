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

"""The Fermi-Hubbard model"""
import numpy as np
from qiskit_nature.second_q.operators.spin_op import SpinOp

from qiskit_nature.second_q.operators import MixedOp
from qiskit_nature.second_q.properties import LatticeModel
from qiskit_nature.second_q.operators import FermionicOp

from qiskit_nature.second_q.properties.lattices import HyperCubicLattice

from qiskit_nature.second_q.hamiltonians.basic_operators import FermionicSpinor, QLM

class WilsonModel(LatticeModel):
    """The Wilson Model.



    Attributes:
        lattice (HyperCubicLattice): Lattice on which the model is defined.
        lattice_constant (complex): Lattice Constant.
        wilson_parameter (complex): Wilson Parameter.
        mass: Mass.
        constraint_coefficient (float): Coefficient added in front of the regularization term in
            the hamiltonian
        representation (list(np.array)): Matrix representation of the dirac operators for the spinnor product.
        flavours (int): Number of flavours.
        fermionic_spinor (FermionicSpinor): Represents the fermionic part of the system.
        QLM_spin (QLM): Represents the bosonic part of the system.

    """

    def __init__(
        self,
        lattice: HyperCubicLattice,
        lattice_constant: complex,
        wilson_parameter: complex,
        charge: float,
        mass: float,
        constraint_coefficient: float,
        representation: list[np.ndarray],
        electric_field: list[float],
        flavours: int,
        spin: int,
    ):
        """
        Args:
            lattice: Lattice on which the model is defined.
            lattice_constant: Lattice Constant.
            wilson_parameter: Wilson Parameter.
            charge: Charge
            mass: Mass.
            constraint_coefficient: Coefficient added in front of the regularization term in
                the hamiltonian
            representation: Matrix representation of the dirac operators for the spinnor product.
            electric_field: External electric field applied on the system.
            flavours: Number of flavours.
            spin: Spin of a bosonic operator.
        """

        super().__init__(lattice)
        self.lattice_constant = lattice_constant
        self.wilson_parameter = wilson_parameter
        self.mass = mass
        self.constraint_coefficient = constraint_coefficient
        self.representation = representation
        self.flavours = flavours
        self.fermionic_spinor = FermionicSpinor(
            spinor_size=self.flavours * int(2 ** ((self.dimension + 1) // 2)), lattice=lattice
        )
        self.bosonic_qlm = QLM(
            spin=spin,
            lattice=lattice,
            charge=charge,
            electric_field=electric_field,
        )

    @property
    def lattice(self) -> HyperCubicLattice:
        """Return a copy of the input lattice."""
        return self._lattice.copy()

    @property
    def spin(self):
        """Returns the spin of the Quantum Link Model"""
        return self.bosonic_qlm.spin

    @property
    def electric_field(self):
        """Returns the spin of the Quantum Link Model"""
        return self.bosonic_qlm.electric_field

    @property
    def charge(self):
        """Returns the charge of the Quantum Link Model"""
        return self.bosonic_qlm.charge

    @property
    def dimension(self):
        """Dimension of the lattice"""
        return self.lattice.dim

    def mass_term(self) -> FermionicOp:
        """Creates the mass term for the wilson hamiltonian."""
        mass_terms = []
        for site in self.lattice.node_indexes:
            mass_terms.append(
                self.fermionic_spinor.spinor_product(site, site, self.representation[0])
            )
        return (self.mass + self.wilson_parameter * self.dimension / self.lattice_constant) * sum(
            mass_terms
        )

    def hopping_term(self) -> MixedOp:
        """Creates the hopping term in the hamiltonian."""
        hopping_terms = []

        tensor_size = self.representation[0].shape[0]
        for edge_index, (node_a, node_b) in enumerate(self.lattice.graph.edge_list()):
            k = self.lattice.direction((node_a, node_b))
            fermionic_tensor = self.representation[0] @ (
                self.wilson_parameter * np.eye(tensor_size) + 1.0j * self.representation[np.abs(k)]
            )
            # print(f"Tensor for nodes {node_a}, {node_b} is:")
            # print(fermionic_tensor)
            fermionic_part = self.fermionic_spinor.spinor_product(node_a, node_b, fermionic_tensor)
            # fermionic_part = self.fermionic_spinor.spinor_product(node_b, node_a, fermionic_tensor)

            bosonic_part = self.bosonic_qlm.operator_u(edge_index=edge_index)

            mix_op = MixedOp(([fermionic_part, bosonic_part], 1 / (2 * self.lattice_constant)))
            mix_op_dag = MixedOp(
                (
                    [fermionic_part.adjoint(), bosonic_part.adjoint()],
                    1 / (2 * self.lattice_constant),
                )
            )
            hopping_terms.append(mix_op)
            hopping_terms.append(mix_op_dag)
        return sum(hopping_terms)

    def plaquette_term(self) -> SpinOp:
        """Creates the plaquette terms for the hamiltonian."""

        if self.dimension == 1:
            return None

        plaquette_terms = []
        for plaquette in self.lattice.get_plaquettes():
            plaquette_terms.append(self.bosonic_qlm.operator_plaquette(*plaquette))
            plaquette_terms.append(self.bosonic_qlm.operator_plaquette(*plaquette).adjoint())

        return (-1) / (4 * self.charge) * sum(plaquette_terms)

    def link_term(self) -> SpinOp:
        """Creates the link terms of the hamiltonian."""
        link_terms = []
        for edge_index, (node_a, node_b) in enumerate(self.lattice.graph.edge_list()):
            operator_e = self.bosonic_qlm.operator_e(edge_index=edge_index)
            operatror_e2 = self.bosonic_qlm.operator_e2(edge_index=edge_index)
            idnty = self.bosonic_qlm.idnty()
            k = self.lattice.direction((node_a, node_b))
            field = self.bosonic_qlm.electric_field[k - 1]
            link_terms.append(operatror_e2 + 2 * field * operator_e + field**2 * idnty)

        return self.charge**2 / 2 * sum(link_terms)

    def gauss_operators(self) -> list[MixedOp]:
        """Returns a list of the gauss operators imposing constraints over each node of the graph."""
        charge_offset = -self.charge
        gauss_terms = []
        for site in self.lattice.node_indexes:
            gauss_term = MixedOp(
                ([self.fermionic_spinor.idnty(), self.bosonic_qlm.operator_divergence(site)], 1)
            )
            gauss_term += -self.charge * self.fermionic_spinor.spinor_product(site, site, None)
            gauss_term += MixedOp(
                ([self.fermionic_spinor.idnty(), self.bosonic_qlm.idnty()], charge_offset)
            )
            gauss_terms.append(gauss_term)
        return gauss_terms

    def second_q_ops(self, display_format: str|None = None)-> MixedOp:
        """Returns the Hamiltonian of the Wilson Model in terms of `MixedOp`.
        Args:
            display_format: We need to wait until MixedOps is finished
        """
        hamiltonian = self.hopping_term()
        hamiltonian += self.mass_term()
        hamiltonian += self.link_term()
        plaquette_term = self.plaquette_term()

        if plaquette_term is not None:
            hamiltonian += plaquette_term
        
        return self.lattice_constant**self.dimension * hamiltonian