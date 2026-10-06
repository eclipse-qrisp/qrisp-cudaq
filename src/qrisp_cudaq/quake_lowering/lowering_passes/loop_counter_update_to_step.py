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

"""Move loop counter updates into the cc.loop step region."""

# Loop counter updates.
# =====================
#
# Jasp loops (jrange, q_fori_loop, q_while_loop) update their counters in the
# loop body, so the step region of the lowered cc.loop only passes its arguments
# on. For every loop-carried value whose next value is ``counter + c`` or
# ``counter - c`` (for jrange, wrapped in a call to _jrange_marker), this pass
# moves the update into the step region and lets the body pass the counter on
# unchanged, the form native CUDA-Q emits for counted loops. An update is moved
# only if the step region can compute it: ``c`` (and the marker's bound) must be
# a constant, defined outside the loop, or passed on unchanged by the body.
#
# This works around a CUDA-Q bug (NVIDIA/cuda-quantum#5561): with the update in
# the body, CUDA-Q's cse can merge it with an equal value such as an index
# i + 1, and cc-loop-normalize then rewrites that index along with the counter
# for loops that do not start at 0, so a[i + 1] addresses a[i]. The pass can be
# dropped once the minimum supported CUDA-Q version contains a fix.
#
# The pass runs after scalar_tensor_unwrap, which turns the 0-d tensor
# round-trips around these updates into plain i64 arithmetic.

from xdsl.dialects import arith, func
from xdsl.dialects.builtin import ModuleOp
from xdsl.ir import Operation, SSAValue

from qrisp_cudaq.quake_lowering.dialects.cc_dialect import CcContinueOp, CcLoopOp


def _is_jrange_marker(op: Operation | None) -> bool:
    """Return whether the operation is a call to a jrange marker function."""
    return isinstance(op, func.CallOp) and op.callee.root_reference.data.startswith("_jrange_marker")


def _defined_in(loop: CcLoopOp, value: SSAValue) -> bool:
    """Return whether the value is defined inside one of the loop's regions."""
    owner = value.owner
    op = owner if isinstance(owner, Operation) else owner.parent_op()
    return op is not None and loop.is_ancestor(op)


def _move_counter_updates_to_step(loop: CcLoopOp) -> None:
    """Move the counter updates of one loop into its step region."""
    body = loop.body_region.block
    step = loop.step_region.block
    body_continue = body.last_op
    step_continue = step.last_op
    assert isinstance(body_continue, CcContinueOp) and isinstance(step_continue, CcContinueOp)

    for position in range(len(body_continue.operands)):
        counter = body.args[position]
        marker = body_continue.operands[position].owner
        if not _is_jrange_marker(marker):
            marker = None
        update = (marker.arguments[0] if marker else body_continue.operands[position]).owner
        if isinstance(update, arith.AddiOp) and counter in (update.lhs, update.rhs):
            amount = update.rhs if update.lhs is counter else update.lhs
        elif isinstance(update, arith.SubiOp) and update.lhs is counter:
            amount = update.rhs
        else:
            continue
        new_ops = []

        def in_step(value: SSAValue) -> SSAValue | None:
            """Return the value as seen from the step region, if it can be."""
            if value.owner is body:
                # The step region receives what the body passes on, so only
                # values the body leaves unchanged keep their meaning there.
                passed_on = body_continue.operands[value.index] is value
                return step.args[value.index] if passed_on else None
            if not _defined_in(loop, value):
                return value
            if isinstance(value.owner, arith.ConstantOp):
                constant = value.owner.clone()
                new_ops.append(constant)
                return constant.result
            return None

        amount_in_step = in_step(amount)
        bound_in_step = in_step(marker.arguments[1]) if marker else None
        if amount_in_step is None or (marker and bound_in_step is None):
            continue

        new_update = type(update)(step.args[position], amount_in_step)
        new_ops.append(new_update)
        next_value = new_update.result
        if marker:
            new_marker = func.CallOp(marker.callee, [next_value, bound_in_step], [marker.results[0].type])
            new_ops.append(new_marker)
            next_value = new_marker.results[0]
        step.insert_ops_before(new_ops, step_continue)
        step_continue.operands = [
            next_value if i == position else operand for i, operand in enumerate(step_continue.operands)
        ]
        body_continue.operands = [
            counter if i == position else operand for i, operand in enumerate(body_continue.operands)
        ]
        for op in (marker, update):
            if op is not None and not any(result.uses for result in op.results):
                body.erase_op(op)


def _move_loop_counter_updates_to_step(module: ModuleOp) -> None:
    """Move the counter updates of every loop into its step region."""
    for loop in [op for op in module.walk() if isinstance(op, CcLoopOp)]:
        _move_counter_updates_to_step(loop)
