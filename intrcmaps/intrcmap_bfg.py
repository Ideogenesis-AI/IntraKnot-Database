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


"""Interaction map for the Balents-Fisher-Girvin model on kagome.

The returned interactions encode

    H = -Jperp sum_<ij> (S+_i S-_j + S-_i S+_j)
        + Jz / 2 sum_hexagon (sum_i S^z_i)^2.

The model plugin assigns the couplings and operator tensors. Expanding the
hexagon term produces six on-site terms and fifteen pair terms per hexagon.
"""

from __future__ import annotations

import logging
from itertools import combinations
from typing import List

from alice.network.interaction import Interaction, Interaction1Site, Interaction2Site
from alice.physics.geometry import Geometry
from alice.physics.kagome import intrcmap_kagome

# Alice loads file plugins under the generic module name ``_alice_plugin``.
# Use an explicit Alice-child logger so INFO records inherit Alice's handlers
# and are written to both alice.log and the runner's combined iknot.log.
logger = logging.getLogger('alice.plugins.intrcmap_bfg')


def _log_pairs(pairs: list[str], indent: int = 3, max_per_line: int = 6) -> None:
    """Log interaction pairs with the wrapped layout used by Alice."""
    indent_str = " " * indent
    for i in range(0, len(pairs), max_per_line):
        logger.info(indent_str + ", ".join(pairs[i:i + max_per_line]))


def intrcmap_bfg(geo: Geometry) -> List[Interaction]:
    """Generate nearest-neighbor XY and hexagon-charge interactions.

    Parameters
    ----------
    geo:
        Fully-resolved kagome geometry. Complete hexagons are included along
        open directions and wrapped along periodic directions.

    Returns
    -------
    list[Interaction]
        Bare interactions with labels identifying ``XY``, ``HEX_ONSITE``,
        and ``HEX_PAIR`` terms. Couplings and tensors are left unset.
    """
    if geo.lattice != 'kagome':
        raise ValueError(
            f"BFG interaction map requires lattice='kagome', got '{geo.lattice}'"
        )

    lx  = geo.lx
    ly  = geo.ly
    bcx = geo.cfg.get('bcx', 'OBC').upper()
    bcy = geo.cfg.get('bcy', 'OBC').upper()

    if bcx not in {'OBC', 'PBC'} or bcy not in {'OBC', 'PBC'}:
        raise ValueError("BFG interaction map supports only OBC and PBC")
    if (bcx == 'PBC' and lx < 2) or (bcy == 'PBC' and ly < 2):
        raise ValueError("Periodic BFG geometries require at least two cells")

    interactions: List[Interaction] = []

    # The transverse part of the BFG Hamiltonian acts on every nearest-neighbor
    # kagome bond. Reuse Alice's tested kagome map and add an XY label so the
    # model plugin can distinguish these bonds from the longitudinal hexagon
    # pairs generated below.
    for intr in intrcmap_kagome(geo):
        intr.label.insert(0, 'XY')
        interactions.append(intr)

    # A hexagon anchored at (row, col) extends to row + 1 and col + 1.
    # Under OBC the final row/column cannot anchor a complete hexagon. Under
    # PBC every unit cell is a valid anchor and the neighboring coordinate is
    # wrapped with modulo arithmetic.
    cols = range(lx) if bcx == 'PBC' else range(max(0, lx - 1))
    rows = range(ly) if bcy == 'PBC' else range(max(0, ly - 1))

    logger.info("")
    logger.info(" BFG hexagon interactions:")

    n_hexagons = 0
    for row in rows:
        for col in cols:
            row_next = (row + 1) % ly
            col_next = (col + 1) % lx

            # Ordered loop around the hexagon:
            #
            #   B(row,col) -> C(row,col) -> A(row+1,col)
            #   -> B(row+1,col) -> C(row,col+1) -> A(row,col+1).
            #
            # This ordering is used only for diagnostics. The expanded
            # Hamiltonian below contains all unordered pairs on the loop.
            hexagon = [
                geo.to_1d((row,      col,      1)),
                geo.to_1d((row,      col,      2)),
                geo.to_1d((row_next, col,      0)),
                geo.to_1d((row_next, col,      1)),
                geo.to_1d((row,      col_next, 2)),
                geo.to_1d((row,      col_next, 0)),
            ]
            if len(set(hexagon)) != 6:
                raise ValueError(
                    f"Degenerate BFG hexagon at unit cell ({row}, {col})"
                )

            logger.info("")
            logger.info(
                " Hexagon %d at cell (%d,%d), ordered sites: %s",
                n_hexagons + 1,
                row,
                col,
                " -> ".join(f"{site:02d}" for site in hexagon),
            )

            # Expand (Jz / 2) * (sum_i Sz_i)^2:
            #
            #   (Jz / 2) * sum_i (Sz_i)^2 + Jz * sum_{i<j} Sz_i Sz_j.
            #
            # For spin one half, (Sz_i)^2 = I / 4. Keep one on-site entry
            # per site per hexagon because sites shared by multiple hexagons
            # must receive each hexagon's constant contribution separately.
            for site in hexagon:
                interactions.append(Interaction1Site(
                    label=['HEX_ONSITE'],
                    site=site,
                ))

            # combinations(hexagon, 2) generates the fifteen unordered pairs
            # in one six-site hexagon. Store sites in MPS order, matching the
            # Interaction2Site contract, and log the exact generated map.
            pairs = []
            for site_a, site_b in combinations(hexagon, 2):
                lo, hi = min(site_a, site_b), max(site_a, site_b)
                interactions.append(Interaction2Site(
                    label=['HEX_PAIR'],
                    leading_site=lo,
                    terminal_site=hi,
                ))
                pairs.append(f"({lo:02d},{hi:02d})")
            logger.info(" HEX_PAIR interactions:")
            _log_pairs(pairs)

            n_hexagons += 1

    # Sorting mixed one-site and two-site interactions makes plugin output
    # deterministic across traversals and Python versions.
    interactions.sort(key=_interaction_sort_key)

    logger.info("")
    logger.info("BFG hexagons: %d", n_hexagons)
    logger.info("BFG interactions: %d", len(interactions))
    logger.info("")

    return interactions


def _interaction_sort_key(intr: Interaction) -> tuple[int, int, str]:
    """Return a deterministic site-ordering key for mixed interactions."""
    if isinstance(intr, Interaction1Site):
        return intr.site, intr.site, intr.label[0]
    elif isinstance(intr, Interaction2Site):
        return intr.leading_site, intr.terminal_site, intr.label[0]
    else:
        raise ValueError(f"Unsupported interaction type: {type(intr)}")
