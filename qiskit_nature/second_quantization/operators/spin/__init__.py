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
"""
Second-Quantization Spin Operators (:mod:`qiskit_nature.second_quantization.operators.spin`)
================================================================

.. currentmodule:: qiskit_nature.second_quantization.operators.spin

Operators for spin systems.

Spin Mappers
============

.. autosummary::
   :toctree: ../stubs/

   LinearMapper
   LogarithmicMapper
"""

from .mappers.linear_mapper import LinearMapper
from .mappers.logarithmic_mapper import LogarithmicMapper

__all__ = [
    "LinearMapper",
    "LogarithmicMapper",
]
