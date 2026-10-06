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

"""Lower Jaspr programs to CUDA-Q Quake MLIR."""

# Entry point for the Jasp → Quake (memory-semantics) lowering pipeline.
# =========================================================================
#
# Pipeline
# --------
# The lowering consists of the following passes:
#
# 0. Emission (mlir_emission) – Translate the Jaspr to an initial xDSL
#    builtin.ModuleOp via jaspr_to_mlir.
# 0a. Safeguard (safeguard_no_ranked_tensor_linalg) – Reject any module
#     that contains linalg.generic operations on ranked tensors before lowering begins.
# 1. JASP → Quake (jasp_to_quake) – Replace jasp.* operations with
#    Quake equivalents and eliminate !jasp.QuantumState threading.
# 2. SCF → CC (scf_to_cc) – Replace structured control flow with
#    cc.if and cc.loop operations.
# 3. Scalar tensor unwrapping (scalar_tensor_unwrap) – Fold trivial
#    rank-0 tensor constants, extracts, and wrappers into scalars.
# 4. Static register allocation (static_veq_alloca) – Rewrite
#    constant-sized !quake.veq<?> allocations as !quake.veq<N>.
# 5. Ranked tensor → CC array (ranked_tensor_to_array) – Lower ranked
#    tensor constants, accesses, signatures, and calls to CC arrays.
# 6. Array → sequence (array_to_sequence) – Rewrite entrypoint array
#    pointers to !cc.sequence<T> for CUDA-Q runtime compatibility.
#
# The returned ModuleOp contains only the dialects and operations supported by
# the CUDA-Q ingestion layer; no !jasp.* types or tensor operations remain.

from qrisp.jasp.jasp_expression import Jaspr
from qrisp.jasp.mlir.mlir_emission import jaspr_to_mlir
from xdsl.dialects.builtin import ModuleOp

from qrisp_cudaq.quake_lowering.lowering_passes.array_to_sequence import (
    _lower_array_to_sequence,
)
from qrisp_cudaq.quake_lowering.lowering_passes.jasp_to_quake.jasp_to_quake import (
    _jasp_to_quake,
)
from qrisp_cudaq.quake_lowering.lowering_passes.ranked_tensor_to_array import (
    _lower_ranked_tensors,
)
from qrisp_cudaq.quake_lowering.lowering_passes.safeguard_no_ranked_tensor_linalg import (
    _verify_no_ranked_tensor_linalg,
)
from qrisp_cudaq.quake_lowering.lowering_passes.scalar_tensor_unwrap import (
    _unwrap_scalar_tensors,
)
from qrisp_cudaq.quake_lowering.lowering_passes.scf_to_cc import _lower_scf_to_cc
from qrisp_cudaq.quake_lowering.lowering_passes.static_veq_alloca import (
    _staticize_veq_alloca,
)
from qrisp_cudaq.quake_lowering.pass_manager import (
    _LoweringPass,
    _run_pass_pipeline,
)


def to_quake_mlir(jaspr: Jaspr, execution_mode: str = "run") -> ModuleOp:
    """
    Compiles a Jaspr to MLIR using the `Quake dialect <https://nvidia.github.io/cuda-quantum/latest/specification/quake-dialect.html>`__.

    Parameters
    ----------
    jaspr : Jaspr
        The Jaspr to compile.
    execution_mode : {"run", "sample"}, default "run"
        Controls how quantum measurements are lowered and how the function
        signature is generated.  Two values are accepted:

        ``"run"``
            Targets ``cudaq.run``.  Array measurements are lowered to a
            ``cc.loop`` that extracts each qubit, calls ``quake.mz`` +
            ``quake.discriminate``, and packs the resulting bits into an
            ``i64`` accumulator.  Single-qubit measurements are lowered to
            ``quake.mz`` + ``quake.discriminate`` returning ``i1``.
            Classical return values are preserved in the function signature.

        ``"sample"``
            Targets ``cudaq.sample``.  Every ``quake.mz`` is emitted on the
            full operand (``!quake.ref`` or ``!quake.veq<?>``), leaving the
            ``!cc.measure_handle`` / ``!cc.sequence<!cc.measure_handle>`` result for the
            CUDA-Q runtime to collect across shots.  A zero dummy constant
            (``i1`` for single qubits, ``i64`` for arrays)
            keeps SSA valid only for pure computation that feeds the entry
            function's return and is subsequently stripped.  If a measurement
            result reaches classical control or another side-effecting operation,
            lowering raises ``NotImplementedError``.  All classical return values
            are stripped from ``func.return`` and the function signature so that
            the kernel has a ``void`` return type, as required by ``cudaq.sample``.

    Returns
    -------
    xdsl.dialects.builtin.ModuleOp
        An xDSL module representing the quantum computation in Quake and CC dialects.

    Examples
    --------
    We create a simple script and inspect the MLIR string:

    ::

        from qrisp import QuantumFloat, cx, t, measure
        from qrisp.jasp import make_jaspr
        from qrisp_cudaq import to_quake_mlir

        def example_function(i):

            qv = QuantumFloat(i)
            cx(qv[0], qv[1])
            t(qv[1])
            meas_res = measure(qv)
            meas_res += 1
            return meas_res

        jaspr = make_jaspr(example_function)(2)
        xdsl_module = to_quake_mlir(jaspr)
        print(xdsl_module)

    .. code-block:: none

        builtin.module @jasp_module {
          func.func public @main(%0: i64) -> (f64) attributes {"cudaq-entrypoint", "cudaq-kernel"} {
            %1 = quake.alloca !quake.veq<?>[%0 : i64]
            %2 = arith.constant 0 : i64
            %3 = quake.veq_size %1 : (!quake.veq<?>) -> i64
            %4 = arith.constant 0 : i64
            %5 = arith.cmpi slt, %2, %4 : i64
            %6 = arith.addi %2, %3 : i64
            %7 = arith.select %5, %6, %2 : i64
            %8 = quake.extract_ref %1[%7] : (!quake.veq<?>, i64) -> !quake.ref
            %9 = arith.constant 1 : i64
            %10 = quake.veq_size %1 : (!quake.veq<?>) -> i64
            %11 = arith.constant 0 : i64
            %12 = arith.cmpi slt, %9, %11 : i64
            %13 = arith.addi %9, %10 : i64
            %14 = arith.select %12, %13, %9 : i64
            %15 = quake.extract_ref %1[%14] : (!quake.veq<?>, i64) -> !quake.ref
            quake.x [%8] %15 : (!quake.ref, !quake.ref) -> ()
            quake.t %15 : (!quake.ref) -> ()
            %16 = quake.veq_size %1 : (!quake.veq<?>) -> i64
            %17 = arith.constant 0 : i64
            %18 = arith.constant 1 : i64
            %19, %20 = cc.loop while ((%21 = %17, %22 = %17) -> (i64, i64)) {
            %23 = arith.cmpi slt, %21, %16 : i64
            cc.condition %23(%21, %22 : i64, i64)
            } do {
            ^bb0(%24: i64, %25: i64):
            %26 = quake.extract_ref %1[%24] : (!quake.veq<?>, i64) -> !quake.ref
            %27 = quake.mz %26 : (!quake.ref) -> !quake.measure
            %28 = quake.discriminate %27 : (!quake.measure) -> i1
            %29 = arith.extui %28 : i1 to i64
            %30 = arith.shli %29, %24 : i64
            %31 = arith.ori %25, %30 : i64
            cc.continue %24, %31 : i64, i64
            } step {
            ^bb1(%32: i64, %33: i64):
            %34 = arith.addi %32, %18 : i64
            cc.continue %34, %33 : i64, i64
            }
            %35 = arith.sitofp %20 : i64 to f64
            %36 = arith.constant 1.000000e+00 : f64
            %37 = arith.mulf %35, %36 : f64
            %38 = arith.constant 1.000000e+00 : f64
            %39 = arith.addf %37, %38 : f64
            func.return %39 : f64
          }
        }

    """
    if execution_mode not in ("run", "sample"):
        raise ValueError(f"Unknown execution_mode: {execution_mode!r}. Supported: 'run', 'sample'.")

    # Step 0 – Produce the initial xDSL module with Jasp IR.
    module: ModuleOp = jaspr_to_mlir(jaspr, lower_stableHLO=True)

    _run_pass_pipeline(
        module,
        (
            _LoweringPass("verify-no-ranked-tensor-linalg", _verify_no_ranked_tensor_linalg),
            _LoweringPass(
                "jasp-to-quake",
                lambda current_module: _jasp_to_quake(current_module, execution_mode),
            ),
            _LoweringPass("scf-to-cc", _lower_scf_to_cc),
            _LoweringPass("scalar-tensor-unwrap", _unwrap_scalar_tensors),
            _LoweringPass("staticize-veq-alloca", _staticize_veq_alloca),
            _LoweringPass("ranked-tensor-to-array", _lower_ranked_tensors),
            _LoweringPass("array-to-sequence", _lower_array_to_sequence),
        ),
    )

    return module
