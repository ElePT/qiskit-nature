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

"""The HFInitialPoint class to compute an initial point for the VQE Ansatz parameters."""

from __future__ import annotations

import warnings

import numpy as np

from qiskit_nature.circuit.library import UCC
from qiskit_nature.second_quantization.operator_factories.electronic import ElectronicEnergy
from qiskit_nature.second_quantization.operator_factories.second_quantized_property import (
    GroupedSecondQuantizedProperty,
)
from qiskit_nature.exceptions import QiskitNatureError

from .initial_point import InitialPoint


class HFInitialPoint(InitialPoint):
    r"""Compute the Hartree-Fock (HF) initial point.

    A class that provides an all-zero initial point for the ``VQE`` parameter values.

    If used in concert with the
    :class:`~qiskit.circuit.library.initial_states.hartree_fock.HartreeFock` initial state (which
    will be prepended to the :class:`~qiskit.circuit.library.ansatzes.ucc.UCC` circuit) the all-zero
    initial point will correspond to the HF initial point.

    The excitation list generated by the :class:`~qiskit.circuit.library.ansatzes.ucc.UCC` ansatz is
    obtained to ensure that the shape of the initial point is appropriate.
    """

    def __init__(self) -> None:
        super().__init__()
        self._ansatz: UCC | None = None
        self._excitation_list: list[tuple[tuple[int, ...], tuple[int, ...]]] | None = None
        self._reference_energy: float = 0.0
        self._parameters: np.ndarray | None = None

    @property
    def grouped_property(self) -> GroupedSecondQuantizedProperty | None:
        """The grouped property.

        The grouped property is not required to compute the HF initial point. If it is provided we
        will attempt to obtain the HF ``reference_energy``.
        """
        return self._grouped_property

    @grouped_property.setter
    def grouped_property(self, grouped_property: GroupedSecondQuantizedProperty) -> None:
        electronic_energy: ElectronicEnergy | None = grouped_property.get_property(ElectronicEnergy)
        if electronic_energy is None:
            warnings.warn(
                "The ElectronicEnergy was not obtained from the grouped_property. "
                "The grouped_property and reference_energy will not be set."
            )
            return

        self._reference_energy = electronic_energy.reference_energy if not None else 0.0
        self._grouped_property = grouped_property

    @property
    def ansatz(self) -> UCC:
        """The UCC ansatz.

        This is used to ensure that the :attr:`excitation_list` matches with the UCC ansatz that
        will be used with the VQE algorithm.
        """
        return self._ansatz

    @ansatz.setter
    def ansatz(self, ansatz: UCC) -> None:
        # Operators must be built early to compute the excitation list.
        _ = ansatz.operators

        # Invalidate any previous computation.
        self._parameters = None

        self._excitation_list = ansatz.excitation_list
        self._ansatz = ansatz

    @property
    def excitation_list(self) -> list[tuple[tuple[int, ...], tuple[int, ...]]]:
        """The list of excitations.

        Setting this will overwrite the excitation list from the ansatz.
        """
        return self._excitation_list

    @excitation_list.setter
    def excitation_list(self, excitations: list[tuple[tuple[int, ...], tuple[int, ...]]]):
        # Invalidate any previous computation.
        self._parameters = None

        self._excitation_list = excitations

    def to_numpy_array(self) -> np.ndarray:
        """The initial point as an array."""
        if self._parameters is None:
            self.compute()
        return self._parameters

    def compute(
        self,
        ansatz: UCC | None = None,
        grouped_property: GroupedSecondQuantizedProperty | None = None,
    ) -> None:
        """Compute the coefficients and energy corrections.

        See further up for more information.

        Args:
            grouped_property: A grouped second-quantized property that may optionally contain the
                Hartree-Fock reference energy. This is for consistency with other initial points.
            ansatz: The UCC ansatz. Required to set the :attr:`excitation_list` to ensure that the
                coefficients are mapped correctly in the initial point array.

        Raises:
            QiskitNatureError: If :attr`ansatz` is not set.
        """
        if grouped_property is not None:
            self.grouped_property = grouped_property

        if ansatz is not None:
            self.ansatz = ansatz

        if self._excitation_list is None:
            raise QiskitNatureError(
                "The excitation list has not been set directly or via the ansatz. "
                "Not enough information has been provided to compute the initial point. "
                "Set the ansatz or call compute with it as an argument. "
                "The ansatz is not required if the excitation list has been set directly."
            )

        self._compute()

    def _compute(self) -> None:
        self._parameters = np.zeros(len(self._excitation_list))

    def get_energy(self) -> float:
        """The reference energy.

        If the reference energy was not obtained from
        :class:`~qiskit_nature.properties.second_quantization.electronic.ElectronicEnergy`
        this will be equal to zero.
        """
        return self._reference_energy
