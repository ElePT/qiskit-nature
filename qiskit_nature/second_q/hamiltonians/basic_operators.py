# This code is part of Qiskit.
#
# (C) Copyright IBM 2021, 2022.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

"""Basic operators for the wilson hamiltonian"""
import numpy as np
from qiskit_nature.second_q.operators import FermionicOp, SpinOp
from qiskit_nature.second_q.properties.lattices import HyperCubicLattice



class FermionicSpinor:
    """Represents the Fermionic Spinors sitting on the nodes of a
    :class"`~.qiskit_nature.second_q.properties.lattices.HyperCubicLattice`.

    Depending on the dimension of the lattice, the fermion sitting at a given
    node will be represented by spinors with `spinor_size` ammount of components.

    The resulting operator will have a size of `spinor_size * num_nodes`. The first
    `spinor_size` registers will correspond to the spinor at node 0, then the spinor
    at node 1 and so on for all the nodes in the lattice.

    """

    def __init__(self, spinor_size: int, lattice: HyperCubicLattice) -> None:
        self.spinor_size = spinor_size
        self.register_lenght = lattice.num_nodes * spinor_size

    def spinor_product(
        self, site_left: int, site_right: int, operator: np.ndarray | None
    ) -> FermionicOp:
        """Returns the spinor product of the fermions in two lattice nodes with respect
        to a given operator.
        Args:
            site_left: Site of the leftmost spinor.
            site_right: Site of the rightmost spinor.
            operator: Operator to to do the tensor product of both spinors. If `None` is
            given the operator will default to the identity.
        """
        if operator is None:
            operator = np.eye(N=self.spinor_size, M=self.spinor_size)
        fermionic_sum = []
        for (a, b), v in np.ndenumerate(operator):
            if v != 0:
                index_a = self.spinor_size * site_left + a
                index_b = self.spinor_size * site_right + b
                fermionic_sum.append(
                    v
                    * (
                        FermionicOp(f"+_{index_a}", register_length=self.register_lenght)
                        @ FermionicOp(f"-_{index_b}", register_length=self.register_lenght)
                    )
                )
        return sum(fermionic_sum)

    def idnty(self) -> FermionicOp:
        """Returns the identity on the Fermionic system."""
        return FermionicOp("", register_length=self.register_lenght)


class QLM:
    """Class representing the Bosons sitting on the edges of a
        :class"`~.qiskit_nature.second_q.properties.lattices.HyperCubicLattice`.

        This Bosons are actually represented by `~.qiskit_nature.second_q.operators.SpinOp` and we
        will have only one operator by edge.

        The resulting operator will have as many spin operators as edges there are in the lattice.
        The ordering of the registers corresponds to the ordering of the edges in the lattice, so
        the nth spin operator will sit in the edge with index `n` of the lattice. In order to see
        which index corresponds to each edge, one can draw the lattice
        :meth:`~.qiskit_nature.second_q.properties.lattices.HyperCubicLattice.indexed_graph`.
    )
    """

    def __init__(
        self, spin: int, lattice: HyperCubicLattice, charge: float, electric_field: list[float]
    ) -> None:

        self.spin = spin
        self.lattice = lattice
        self.edges = len(lattice.weighted_edge_list)
        self.charge = charge
        self.electric_field = electric_field

    def idnty(self):
        """Returns the identity on the Bosonic system."""
        return SpinOp("I_0", spin=self.spin, register_length=self.edges)

    def operator_u(self, edge_index: int):
        """Returns the Operator U in the Wilson Hamiltonian.
        Args:
            edge_index: The operator will correspond to the Boson sitting at `edge_index`.
        """
        return (self.spin * (self.spin + 1)) ** (-0.5) * SpinOp(
            f"+_{edge_index}", spin=self.spin, register_length=self.edges
        )

    def operator_e(self, edge_index:int):
        """Returns the Operator E in the Wilson Hamiltonian.
        Args:
            edge_index: The operator will correspond to the Boson sitting at `edge_index`.
        """
        return self.charge * SpinOp(f"Z_{edge_index}", spin=self.spin, register_length=self.edges)

    def operator_e2(self, edge_index:int):
        """Returns the Operator E^2 in the Wilson Hamiltonian.
        Args:
            edge_index: The operator will correspond to the Boson sitting at `edge_index`.
        """
        return (self.charge**2) * SpinOp(
            f"Z_{edge_index}^2", spin=self.spin, register_length=self.edges
        )

    # This operator could potentially create sign issues. We need to get it checked.
    def operator_plaquette(self, nodes: tuple[int,int,int,int]):
        """Returns the plaquette operator for a given plaquette.

        The plaquette operator for a given square consists of the product of the botom
        and right `U` operators and the adjoint of the top and left edges.

        Args:
            nodes: Tupple with the nodes that will form the plaquette term in the
                following order (bottom,right,top,left).
        """
        return (self.spin * (self.spin + 1)) ** (-8) * SpinOp(
            f"+_{nodes[0]} +_{nodes[1]} -_{nodes[2]} -_{nodes[3]}",
            spin=self.spin,
            register_length=self.edges,
        )

    def operator_divergence(self, node:int):
        """Returns the divergence for the field arround a given node.

        In case of being next to a border with open boundary conditions the `unexisting`
        edge will not be added.

        Args:
            node: Node arround which the divergence is computed.
        """
        indexed_graph = self.lattice.indexed_graph()
        divergence = []
        for neighbor in self.lattice.graph.neighbors(node):
            directn = self.lattice.direction((node, neighbor))
            edge_index = indexed_graph.get_edge_data(node, neighbor)
            divergence.append(np.sign(-directn) * self.operator_e(edge_index))
        return sum(divergence)
