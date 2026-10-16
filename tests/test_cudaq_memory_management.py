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

"""Tests for CUDA-Q quantum memory management."""

import pytest
import cudaq

from qrisp import QuantumFloat, cx, x, reset, measure
from qrisp.jasp import jrange, qache
from qrisp_cudaq import cudaq_kernel


@pytest.mark.timeout(30)
def test_cudaq_memory_management():
    """Test for CUDA-Q quantum memory management."""

    @cudaq_kernel
    def main():
        a = QuantumFloat(10)
        x(a[0])
        x(a[0])
        reset(a)
        a.delete()

        b = QuantumFloat(10)
        x(b[0])
        x(b[0])
        reset(b)
        b.delete()

        c = QuantumFloat(10)
        x(c[0])
        return measure(c[0])

    cudaq.run(main, shots_count=10)


@pytest.mark.timeout(30)
def test_cudaq_memory_management_arithmetic():
    """Test for CUDA-Q quantum memory management using Qrisp's QuantumFloat arithmetic.
    Each inplace addition allocates auxiliary qubits, which are uncomputed, and must be deallocated properly.
    Otherwise, the total number of allocated qubits grows, leading to prohibitive simulation costs."""

    @cudaq_kernel
    def main():
        a = QuantumFloat(6)
        a[:] = 4

        a += 4
        a += 4
        a += 4
        a += 6

        return measure(a)

    cudaq.run(main, shots_count=10)


@qache
def _copy_into_new_register(a):
    s = QuantumFloat(a.size)
    for i in jrange(a.size):
        cx(a[i], s[i])
    return s


@qache
def _copy_and_flip_lowest_bit(a):
    s = _copy_into_new_register(a)
    x(s[0])
    return s


@pytest.mark.timeout(30)
def test_register_returned_from_qached_function():
    """A register allocated inside a @qache function stays allocated after the function returns."""

    @cudaq_kernel
    def main():
        a = QuantumFloat(2)
        a[:] = 3
        return measure(_copy_into_new_register(a))

    assert set(cudaq.run(main, shots_count=20)) == {3.0}


@pytest.mark.timeout(30)
def test_register_returned_through_nested_qached_functions():
    """A register allocated in a nested @qache call survives being returned through both functions."""

    @cudaq_kernel
    def main():
        a = QuantumFloat(2)
        a[:] = 3
        return measure(_copy_and_flip_lowest_bit(a))

    assert set(cudaq.run(main, shots_count=20)) == {2.0}


@pytest.mark.timeout(60)
def test_quantum_float_multiplication():
    """QuantumFloat multiplication returns its product register from a @qache function."""

    @cudaq_kernel
    def main():
        a = QuantumFloat(2)
        b = QuantumFloat(2)
        a[:] = 3
        b[:] = 2
        return measure(a * b)

    assert set(cudaq.run(main, shots_count=20)) == {6.0}


@qache
def _new_register_and_passthrough(a, k):
    s = QuantumFloat(a.size)
    for i in jrange(a.size):
        cx(a[i], s[i])
    return s, k


@pytest.mark.timeout(30)
def test_classical_argument_returned_with_new_register():
    """A classical argument returned unchanged next to a new register keeps its value."""

    @cudaq_kernel
    def main(k: int):
        a = QuantumFloat(2)
        a[:] = 3
        s, k_out = _new_register_and_passthrough(a, k)
        return measure(s) + k_out

    assert set(cudaq.run(main, 5, shots_count=10)) == {8.0}


@pytest.mark.timeout(30)
def test_register_returned_from_qached_function_inside_loop():
    """A register returned from a @qache function called in a loop body stays allocated."""

    @cudaq_kernel
    def main():
        a = QuantumFloat(2)
        a[:] = 3
        out = QuantumFloat(2)
        for i in jrange(2):
            s = _copy_into_new_register(a)
            cx(s[i], out[i])
        return measure(out)

    assert set(cudaq.run(main, shots_count=20)) == {3.0}
