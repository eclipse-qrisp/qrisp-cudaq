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

"""Inline calls to functions that return qubits."""

# Inlining of qubit-returning calls.
# ==================================
#
# CUDA-Q releases the qubits a function allocates when the function returns.
# This pass inlines every call to a function whose results include a
# !quake.veq or !quake.ref, so that the qubits it returns are allocated in the
# caller, and erases the inlined functions afterwards.

from xdsl.dialects import func
from xdsl.dialects.builtin import ModuleOp
from xdsl.rewriter import InsertPoint, Rewriter

from qrisp_cudaq.quake_lowering.dialects.quake_dialect import QuakeRefType, QuakeVeqType


def _returns_qubits(func_op: func.FuncOp) -> bool:
    """Return whether the function returns a qubit or a qubit register."""
    # Qrisp does not let a @qache function return one of its quantum arguments,
    # so every qubit a function returns is one it allocated.
    return any(isinstance(t, (QuakeVeqType, QuakeRefType)) for t in func_op.function_type.outputs.data)


def _inline_call(call: func.CallOp, callee: func.FuncOp) -> None:
    """Replace the call by a copy of the callee's body."""
    body = callee.body.clone().block
    return_op = body.last_op
    assert isinstance(return_op, func.ReturnOp)
    # A returned block argument becomes the corresponding call operand.
    operand_of_arg = {id(arg): operand for arg, operand in zip(body.args, call.arguments)}
    returned = [operand_of_arg.get(id(value), value) for value in return_op.operands]
    body.erase_op(return_op)
    Rewriter.inline_block(body, InsertPoint.before(call), call.arguments)
    for result, value in zip(call.results, returned):
        result.replace_all_uses_with(value)
    call.parent_block().erase_op(call)


def _inline_qubit_returning_calls(module: ModuleOp) -> None:
    """Inline every call to a function that returns qubits and erase the inlined functions."""
    funcs = {op.sym_name.data: op for op in module.body.block.ops if isinstance(op, func.FuncOp)}
    targets = {name for name, func_op in funcs.items() if _returns_qubits(func_op)}
    inlined = set()
    # Inlining can expose calls from the inlined body, hence the repeated walks.
    while calls := [
        op for op in module.walk() if isinstance(op, func.CallOp) and op.callee.root_reference.data in targets
    ]:
        for call in calls:
            name = call.callee.root_reference.data
            _inline_call(call, funcs[name])
            inlined.add(name)
    for name in inlined:
        module.body.block.erase_op(funcs[name])
