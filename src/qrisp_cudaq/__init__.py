# ********************************************************************************
# * Copyright (c) 2026 the Qrisp authors
# *
# * This program and the accompanying materials are made available under the
# * terms of the Eclipse Public License 2.0 which is available at
# * http://www.eclipse.org/legal/epl-2.0.
# *
# * This Source Code may also be made available under the following Secondary
# * Licenses when the conditions for such availability set forth in the Eclipse
# * Public License, v. 2.0 are satisfied: GNU General Public License, version 2
# * with the GNU Classpath Exception which is
# * available at https://www.gnu.org/software/classpath/license.html.
# *
# * SPDX-License-Identifier: EPL-2.0 OR GPL-2.0 WITH Classpath-exception-2.0
# ********************************************************************************

"""Provide the CUDA-Q backend interface for Qrisp Jasp programs."""

# qrisp_cudaq — CUDA-Q backend for Qrisp/Jasp.
# =============================================
#
# Provides the CUDA-Q execution tools for Qrisp functions compiled via
# Jasp/Quake MLIR.

from qrisp_cudaq.annotations import FixedShapeNDArray
from qrisp_cudaq.cudaq_kernel import cudaq_kernel
from qrisp_cudaq.quake_lowering import to_quake_mlir

__all__ = [
    "cudaq_kernel",
    "FixedShapeNDArray",
    "to_quake_mlir",
]
