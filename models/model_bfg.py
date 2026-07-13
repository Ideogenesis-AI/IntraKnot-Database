# Copyright (C) 2026 Changkai Zhang.
#
# This file is part of IntraKnot.
#
# IntraKnot is free software: you can redistribute it and/or modify it
# under the terms of the GNU General Public License as published
# by the Free Software Foundation, either version 3 of the License,
# or (at your option) any later version.
#
# IntraKnot is distributed in the hope that it will be useful, but
# WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU
# General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with IntraKnot. If not, see <https://www.gnu.org/licenses/>.


"""Balents-Fisher-Girvin model builder for Alice."""

from __future__ import annotations

from typing import Callable, Dict, List, Optional, Tuple

from nicole import Index, Tensor
from nicole import einsum, identity, oplus

from alice.network.interaction import Interaction, Interaction1Site, Interaction2Site
from alice.physics.system import build_bosonic


def _make_intermid4(string_op: Tensor, leading_tnsr: Tensor) -> Tensor:
    """Build a bosonic intermediate-site tensor for one operator channel."""
    op_idx_out = leading_tnsr.indices[1]
    op_id = identity(op_idx_out.flip())
    return einsum('op,rs->oprs', op_id, string_op)


def build_bfg(
    interactions: List[Interaction],
    L: int = 0,
    *,
    symmetry: str = 'U1',
    spin: float = 0.5,
    Jz: float = 1.0,
    Jperp: float = 0.1,
    include_hex_onsite: bool = True,
    space_fn: Optional[Callable] = None,
    **_ignored,
) -> Tuple[Index, Dict[str, Tensor]]:
    r"""Populate interactions for the spin-1/2 BFG kagome model.

    The implemented Hamiltonian is

    .. math::

        H = -J_\perp \sum_{\langle i,j\rangle}
            (S_i^+ S_j^- + S_i^- S_j^+)
            + \frac{J_z}{2} \sum_\hexagon
            \left(\sum_{i \in \hexagon} S_i^z\right)^2.

    By default, the squared hexagon term is represented exactly, including
    its constant on-site contribution for spin one half. Set
    ``include_hex_onsite=False`` to omit that constant energy shift.
    """
    del L

    if symmetry != 'U1':
        raise ValueError(f"BFG model requires symmetry='U1', got '{symmetry}'")
    if spin != 0.5:
        raise ValueError(f"BFG model requires spin=0.5, got {spin}")

    _space = space_fn if space_fn is not None else build_bosonic
    spc, ops = _space(symmetry, spin)

    xy4       = oplus(ops['Sp4'], ops['Sm4'], axes=1)
    xy4dag    = oplus(ops['Sp4dag'], ops['Sm4dag'], axes=0)
    xy4mid    = _make_intermid4(identity(spc), xy4)
    sz4       = ops['Sz4']
    sz4dag    = ops['Sz4dag']
    sz4mid    = _make_intermid4(identity(spc), sz4)
    identity4 = ops['I4']

    for intr in interactions:
        if isinstance(intr, Interaction1Site):
            if include_hex_onsite and 'HEX_ONSITE' in intr.label:
                intr.cpl  = Jz / 8.0
                intr.tnsr = identity4.clone()
            continue

        if not isinstance(intr, Interaction2Site):
            continue

        if 'XY' in intr.label:
            # Nicole's U1 Sp/Sm are spherical components:
            # Sp = -S+ / sqrt(2), Sm = S- / sqrt(2). The two tensor
            # products therefore carry a factor 1/2, so use -2*Jperp
            # to implement -Jperp * (S+_i S-_j + S-_i S+_j).
            intr.cpl           = -2.0 * Jperp
            intr.leading_tnsr  = xy4.clone()
            intr.terminal_tnsr = xy4dag.clone()
            if intr.terminal_site > intr.leading_site + 1:
                intr.intermid_tnsr = xy4mid.clone()

        elif 'HEX_PAIR' in intr.label:
            intr.cpl           = Jz
            intr.leading_tnsr  = sz4.clone()
            intr.terminal_tnsr = sz4dag.clone()
            if intr.terminal_site > intr.leading_site + 1:
                intr.intermid_tnsr = sz4mid.clone()

    return spc, ops
