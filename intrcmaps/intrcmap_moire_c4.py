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


"""Interaction map for the LLL-projected C4 moire model on a cylinder chain."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, List

from alice.network.interaction import (
    Interaction,
    Interaction1Site,
    Interaction2Site,
    InteractionNSite,
)

if TYPE_CHECKING:
    from alice.physics.geometry import Geometry

logger = logging.getLogger(__name__)


def intrcmap_moire_c4(geo: Geometry) -> List[Interaction]:
    """Enumerate one-body and ordered-pair interaction slots for moire_c4.

    Reads from `geo.cfg`:

    - `lx` — number of retained Landau-gauge orbitals `N`
    - `ny` — cylinder periods per flux cell (hopping distance for C4)

    The ordered-pair enumeration is complete: every `(m, n, k, l)` admitted by
    momentum conservation and the orbital bounds is emitted. Truncation belongs
    to the model builder, where the coefficients are known — see `cpl_rtol` in
    `build_moire_c4`. Do not reintroduce index cutoffs here: this stage only
    receives `geo`, so it cannot see the interaction parameters that decide
    which terms are negligible, and a cutoff imposed here is invisible in the
    logs because the excluded terms are never enumerated at all.

    All returned interactions have `cpl = 0.0` and unset tensors; the model
    builder fills them.
    """
    N = int(geo.lx)
    ny = int(geo.cfg.get('ny', 2))

    interactions: List[Interaction] = []

    logger.info("─" * 60)
    logger.info("Moire C4 Interaction Map".center(60))
    logger.info("─" * 60)
    logger.info("")
    logger.info(f"  Orbitals N = {N},  ny = {ny}")
    logger.info("")

    # --- One-body diagonal potential slots ---
    for site in range(N):
        interactions.append(Interaction1Site(
            label=['B', 'DIAG'],
            site=site,
        ))

    # --- One-body hopping at separation ny (C4 y-harmonics) ---
    n_hop = 0
    for n in range(N - ny):
        interactions.append(Interaction2Site(
            label=['B', 'HOP'],
            leading_site=n,
            terminal_site=n + ny,
        ))
        n_hop += 1

    # --- Ordered-pair interaction: (m, p, r) with p = m-l, r = m-k ---
    # The ordered-pair constraints are r < p and p + r < 0. For integer p, r
    # these are jointly equivalent to r < 0 together with |r| >= |p| + 1, so
    # writing q = -r > 0 the admissible triples can be enumerated directly from
    # the orbital bounds without generating and rejecting candidates:
    #
    #   l = m - p     in [0, N)  ->  m - N < p <= m
    #   k = m + q     in [0, N)  ->  q <= N - 1 - m
    #   n = m - p + q in [0, N)  ->  q <= N - 1 - m + p    (binds only for p < 0)
    #
    # m < n and l < k both reduce to q > p and so need no separate check.
    # When p == 0 the term is density-density n_m n_n with n = m + q.
    n_pair = 0
    n_dens = 0
    for m in range(N):
        for p in range(m - N + 1, m + 1):
            q_hi = N - 1 - m + min(p, 0)
            for q in range(abs(p) + 1, q_hi + 1):
                l = m - p
                k = m + q
                n = m - p + q
                if p == 0:
                    label = ['V', 'DENS']
                    n_dens += 1
                else:
                    label = ['V', 'PAIR']
                    n_pair += 1
                interactions.append(InteractionNSite(
                    label=label,
                    sites=[m, n, k, l],
                ))

    logger.info(f"  Diagonal one-body:     {N}")
    logger.info(f"  Hopping (sep={ny}):     {n_hop}")
    logger.info(f"  Density-density:       {n_dens}")
    logger.info(f"  Ordered-pair terms:    {n_pair}")
    logger.info(f"  Total interactions:    {len(interactions)}")
    logger.info("")

    return interactions
