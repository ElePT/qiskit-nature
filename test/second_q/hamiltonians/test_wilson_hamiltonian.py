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

from qiskit_nature.second_q.operators.spin_op import SpinOp
from qiskit_nature.second_q.properties.lattices.hyper_cubic_lattice import HyperCubicLattice
from test import QiskitNatureTestCase

import numpy as np
from ddt import ddt, data, unpack

from qiskit_nature.second_q.mappers import QubitConverter
from qiskit_nature.second_q.mappers import JordanWignerMapper
from qiskit_nature.second_q.hamiltonians import QuadraticHamiltonian
from qiskit_nature.second_q.operators import FermionicOp
from qiskit_nature.second_q.hamiltonians.wilson_sun_hamiltonian import WilsonModel


@ddt
class TestWilsonHamiltonian(QiskitNatureTestCase):

    representation4 = [
        np.array([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, -1, 0], [0, 0, 0, -1]]),
        np.array([[0, 0, 0, 1], [0, 0, 1, 0], [0, -1, 0, 0], [-1, 0, 0, 0]]),
        np.array([[0, 0, 0, -1j], [0, 0, 1j, 0], [0, 1j, 0, 0], [-1j, 0, 0, 0]]),
        np.array([[0, 0, 1, 0], [0, 0, 0, -1], [-1, 0, 0, 0], [0, 1, 0, 0]]),
    ]

    representation2 = [
        np.array([[1, 0], [0, -1]]),
        np.array([[0, 1j], [-1j, 0]]),
        np.array([[0, 1], [1, 0]]),
    ]

    @unpack
    @data(
        {
            "lattice_size": (2, 3),
            "representation": representation2,
            "expected_mass_term": sum([FermionicOp((f"N_{i}", 3 * (-1) ** i)) for i in range(12)]),
        },
    )
    def test_mass_term(self, lattice_size, representation, expected_mass_term):

        lattice = HyperCubicLattice(size=lattice_size, self_loops=False)

        w_model = WilsonModel(
            lattice=lattice,
            a=1,
            r=1,
            mass=1,
            representation=representation,
            electric_field=lattice_size,
            flavours=1,
            spin=3,
            e_value=(3 * (3 + 1)) ** (0.5) / (8),
        )

        with self.subTest("Mass Term"):
            self.assertEqual(
                w_model.mass_term().simplify().to_list(), expected_mass_term.simplify().to_list()
            )

    @unpack
    @data(
        {
            "lattice_size": (2, 2),
            "representation": representation2,
            "expected_plaquette_term": sum([SpinOp("+_0 +_3 -_2 -_1", spin=3, register_length=4)]),
        },
    )
    def test_plaquette_term(self, lattice_size, representation, expected_plaquette_term):

        lattice = HyperCubicLattice(size=lattice_size, self_loops=False)

        expected_plaquette_term = -0.25 * (3 * (3 + 1)) ** (-2) * expected_plaquette_term
        expected_plaquette_term = expected_plaquette_term + expected_plaquette_term.adjoint()

        w_model = WilsonModel(
            lattice=lattice,
            a=1,
            r=1,
            mass=1,
            representation=representation,
            electric_field=lattice_size,
            flavours=1,
            spin=3,
            e_value=1,
        )

        self.assertEqual(
            w_model.plaquette_term().simplify().to_list(),
            expected_plaquette_term.simplify().to_list(),
        )

    def test_hopping(self):
        pass

    def test_link(self):
        pass
