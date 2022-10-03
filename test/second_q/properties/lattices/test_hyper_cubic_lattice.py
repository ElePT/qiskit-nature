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

"""Test for HyperCubicLattice."""
from test import QiskitNatureTestCase
import numpy as np
from numpy.testing import assert_array_equal
from retworkx import PyGraph, is_isomorphic
from qiskit_nature.second_q.properties.lattices import (
    BoundaryCondition,
    HyperCubicLattice,
)


from ddt import ddt, data, unpack


@ddt
class TestHyperCubic(QiskitNatureTestCase):
    """Test HyperCubicLattice."""

    def test_init(self):
        """Test init."""
        size = (2, 2, 2)
        edge_parameter = (1.0 + 1.0j, 0.0, -2.0 - 2.0j)
        onsite_parameter = 5.0
        boundary_condition = (
            BoundaryCondition.OPEN,
            BoundaryCondition.PERIODIC,
            BoundaryCondition.OPEN,
        )
        hyper_cubic = HyperCubicLattice(size, edge_parameter, onsite_parameter, boundary_condition)

        with self.subTest("Check the graph."):
            target_graph = PyGraph(multigraph=False)
            target_graph.add_nodes_from(range(8))
            weighted_edge_list = [
                (0, 1, 1.0 + 1.0j),
                (2, 3, 1.0 + 1.0j),
                (4, 5, 1.0 + 1.0j),
                (6, 7, 1.0 + 1.0j),
                (0, 2, 0.0),
                (1, 3, 0.0),
                (4, 6, 0.0),
                (5, 7, 0.0),
                (0, 4, -2.0 - 2.0j),
                (1, 5, -2.0 - 2.0j),
                (2, 6, -2.0 - 2.0j),
                (3, 7, -2.0 - 2.0j),
                (0, 0, 5.0),
                (1, 1, 5.0),
                (2, 2, 5.0),
                (3, 3, 5.0),
                (4, 4, 5.0),
                (5, 5, 5.0),
                (6, 6, 5.0),
                (7, 7, 5.0),
            ]
            target_graph.add_edges_from(weighted_edge_list)
            self.assertTrue(
                is_isomorphic(hyper_cubic.graph, target_graph, edge_matcher=lambda x, y: x == y)
            )

        with self.subTest("Check the number of nodes."):
            self.assertEqual(hyper_cubic.num_nodes, 8)

        with self.subTest("Check the set of nodes."):
            self.assertSetEqual(set(hyper_cubic.node_indexes), set(range(8)))

        with self.subTest("Check the set of weights."):
            target_set = {
                (0, 1, 1.0 + 1.0j),
                (2, 3, 1.0 + 1.0j),
                (4, 5, 1.0 + 1.0j),
                (6, 7, 1.0 + 1.0j),
                (0, 2, 0.0),
                (1, 3, 0.0),
                (4, 6, 0.0),
                (5, 7, 0.0),
                (0, 4, -2.0 - 2.0j),
                (1, 5, -2.0 - 2.0j),
                (2, 6, -2.0 - 2.0j),
                (3, 7, -2.0 - 2.0j),
                (0, 0, 5.0),
                (1, 1, 5.0),
                (2, 2, 5.0),
                (3, 3, 5.0),
                (4, 4, 5.0),
                (5, 5, 5.0),
                (6, 6, 5.0),
                (7, 7, 5.0),
            }
            self.assertSetEqual(set(hyper_cubic.weighted_edge_list), target_set)

        with self.subTest("Check the adjacency matrix."):
            target_matrix = np.array(
                [
                    [5.0, 1.0 + 1.0j, 0.0, 0.0, -2.0 - 2.0j, 0.0, 0.0, 0.0],
                    [1.0 - 1.0j, 5.0, 0.0, 0.0, 0.0, -2.0 - 2.0j, 0.0, 0.0],
                    [0.0, 0.0, 5.0, 1.0 + 1.0j, 0.0, 0.0, -2.0 - 2.0j, 0.0],
                    [0.0, 0.0, 1.0 - 1.0j, 5.0, 0.0, 0.0, 0.0, -2.0 - 2.0j],
                    [-2.0 + 2.0j, 0.0, 0.0, 0.0, 5.0, 1.0 + 1.0j, 0.0, 0.0],
                    [0.0, -2.0 + 2.0j, 0.0, 0.0, 1.0 - 1.0j, 5.0, 0.0, 0.0],
                    [0.0, 0.0, -2.0 + 2.0j, 0.0, 0.0, 0.0, 5.0, 1.0 + 1.0j],
                    [0.0, 0.0, 0.0, -2.0 + 2.0j, 0.0, 0.0, 1.0 - 1.0j, 5.0],
                ]
            )

            assert_array_equal(hyper_cubic.to_adjacency_matrix(weighted=True), target_matrix)

    @unpack
    @data(
        ((2, 2), BoundaryCondition.OPEN, {(0, 3, 2, 1)}),
        ((2, 2), BoundaryCondition.PERIODIC, {(0, 3, 2, 1)}),
        (
            (3, 3),
            BoundaryCondition.OPEN,
            {(2, 8, 4, 3), (0, 6, 2, 1), (7, 11, 9, 8), (5, 10, 7, 6)},
        ),
        (
            (3, 3),
            BoundaryCondition.PERIODIC,
            {
                (2, 8, 4, 3),
                (7, 11, 9, 8),
                (0, 6, 2, 1),
                (5, 17, 9, 16),
                (12, 10, 13, 1),
                (13, 11, 14, 3),
                (0, 16, 4, 15),
                (12, 17, 14, 15),
                (5, 10, 7, 6),
            },
        ),
        (
            (2, 2, 2),
            BoundaryCondition.OPEN,
            {
                (1, 6, 4, 2),
                (5, 11, 7, 6),
                (8, 11, 10, 9),
                (0, 8, 5, 1),
                (3, 10, 7, 4),
                (0, 9, 3, 2),
            },
        ),
    )
    def test_plaquettes(self, size, boundary, expected_result):
        lattice = HyperCubicLattice(size=size, self_loops=False, boundary_condition=boundary)
        np.testing.assert_equal(expected_result, lattice.get_plaquettes())

    @unpack
    @data(
        ((10, 10), BoundaryCondition.OPEN, 9 * 9),
        ((7, 6, 4), BoundaryCondition.OPEN, 6 * 5 * 4 + 6 * 6 * 3 + 7 * 5 * 3),
        ((2, 2), BoundaryCondition.PERIODIC, 1),
        ((2, 2, 2), BoundaryCondition.OPEN, 6),
        ((2, 2, 2), BoundaryCondition.PERIODIC, 6),
        ((3, 4), [BoundaryCondition.OPEN, BoundaryCondition.PERIODIC], 2 * 3 + 2),
    )
    def test_complex_plaquettes(self, size, boundary, expected_n_plaquettes):
        lattice = HyperCubicLattice(size=size, self_loops=False, boundary_condition=boundary)
        self.assertEqual(expected_n_plaquettes, len(lattice.get_plaquettes()))
