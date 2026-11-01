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


"""Interaction map for the Balents-Fisher-Girvin model on an infinite cylinder.

The returned interactions encode

    H = -Jperp sum_<ij> (S+_i S-_j + S-_i S+_j)
        + Jz / 2 sum_hexagon (sum_i S^z_i)^2.

The model plugin `build_ibfg` assigns the couplings and operator tensors.
Expanding the hexagon term produces six on-site terms and fifteen pair
terms per hexagon.

The geometry is infinite along x and periodic along y: `bcx='IBC'` and
`bcy='PBC'`. `lx` is the number of kagome columns in one matrix-product
unit cell (one column is enough when the state shares the lattice's
translation), and `ly` is the circumference in unit cells. Sequential
traversal walks around that ring and then along the cylinder, which is
the ordering the cell is tiled in. A bond that steps out of the cell
through its right edge keeps a terminal site at or beyond `geo.L`;
`build_ihamiltonian` carries the operator string into the next cell from
there. `bcy='OBC'` is the same construction on an infinite strip.
"""

from __future__ import annotations

import logging
from itertools import combinations
from typing import List, Sequence

from alice.network.interaction import Interaction, Interaction1Site, Interaction2Site
from alice.physics.geometry import Geometry

# Alice loads file plugins under the generic module name ``_alice_plugin``.
# Use an explicit Alice-child logger so INFO records inherit Alice's handlers
# and are written to both alice.log and the runner's combined iknot.log.
logger = logging.getLogger('alice.plugins.intrcmap_ibfg')


def _log_pairs(pairs: list[str], indent: int = 3, max_per_line: int = 6) -> None:
    """Log interaction pairs with the wrapped layout used by Alice."""
    indent_str = " " * indent
    for i in range(0, len(pairs), max_per_line):
        logger.info(indent_str + ", ".join(pairs[i:i + max_per_line]))


def _unit_cell_index(geo: Geometry, row: int, col: int, u: int) -> int:
    """Return the MPS index of a kagome site, including the next column.

    Column `lx` is the first column of the following unit cell. The infinite
    chain is the cell repeated, so that column has the same internal order
    as column 0 and its indices are shifted by the cell length. A negative
    column is the previous cell; the infinite-cylinder builder does not emit
    those, but the same rule names them.
    """
    cell, col_in = divmod(col, geo.lx)
    return geo.to_1d((row, col_in, u)) + cell * geo.L


def _order_bond(left: int, right: int, n_sites: int) -> tuple[int, int]:
    """Return a bond as `(leading, terminal)` with the leading site in-cell.

    `build_ihamiltonian` names every bond by the unit-cell site that leads
    it, and reads a terminal site past the cell as the partner in a later
    cell. A bond that lies wholly in the next cell is the translate of one
    in this cell, so it is folded back by the cell length. A bond with only
    its right end past the cell is already in the form the builder wants.
    """
    leading, terminal = (left, right) if left < right else (right, left)
    if leading >= n_sites:
        shift = (leading // n_sites) * n_sites
        leading -= shift
        terminal -= shift
    if leading < 0 or leading >= n_sites or terminal <= leading:
        raise ValueError(
            f"iBFG bond {left}--{right} is not an infinite-chain bond "
            f"on a cell of {n_sites} sites"
        )
    return leading, terminal


def _intrcmap_ibfg(geo: Geometry) -> List[Interaction]:
    """Nearest-neighbor XY and hexagon terms of an infinite kagome lattice.

    `bcx` is `IBC`: every column of the cell, including the last, has a
    neighbor in the following cell. `bcy` is either `PBC` (cylinder) or
    `OBC` (strip). Kagome bond 6 runs to the column on the left, so on the
    first column that partner would sit in the previous cell. Emitting it
    from the other end, the B site of this cell toward the C site of the
    next column, keeps the leading site inside the cell.
    """
    lx = geo.lx
    ly = geo.ly
    n_sites = geo.L
    bcy = geo.cfg.get('bcy', 'OBC').upper()
    interactions: List[Interaction] = []

    def _emit(coord_a: Sequence[int], coord_b: Sequence[int], label: List[str]) -> str:
        left = _unit_cell_index(geo, coord_a[0], coord_a[1], coord_a[2])
        right = _unit_cell_index(geo, coord_b[0], coord_b[1], coord_b[2])
        leading, terminal = _order_bond(left, right, n_sites)
        tags = list(label)
        if terminal >= n_sites:
            tags.append('IBC')
        interactions.append(Interaction2Site(
            label=tags,
            leading_site=leading,
            terminal_site=terminal,
        ))
        return f"({leading:02d},{terminal:02d})"

    def _emit_onsite(row: int, col: int, u: int) -> None:
        site = _unit_cell_index(geo, row, col, u)
        if site >= n_sites:
            site -= (site // n_sites) * n_sites
        interactions.append(Interaction1Site(label=['HEX_ONSITE'], site=site))

    # Rows that have a neighbor below them. On a cylinder the top row is
    # that neighbor of the bottom row; on a strip the last row is an edge.
    if bcy == 'PBC':
        y_rows = range(ly)
    else:
        y_rows = range(max(0, ly - 1))

    logger.info("")
    kind = "cylinder" if bcy == 'PBC' else "strip"
    logger.info(" iBFG infinite %s (bcx=IBC, bcy=%s):", kind, bcy)
    logger.info(" Unit cell: %d column(s) by %d row(s), %d sites", lx, ly, n_sites)

    logger.info("")
    logger.info(" XY N2U interactions (upward triangles):")
    pairs: List[str] = []
    for row in range(ly):
        for col in range(lx):
            for u, v in ((0, 1), (0, 2), (1, 2)):
                pairs.append(_emit(
                    (row, col, u), (row, col, v), ['XY', 'NN', 'N2U'],
                ))
    _log_pairs(pairs)

    logger.info("")
    logger.info(" XY N2D bond 4 (horizontal, into the next column):")
    pairs = []
    for row in range(ly):
        for col in range(lx):
            pairs.append(_emit(
                (row, col, 1), (row, col + 1, 0), ['XY', 'NN', 'N2D'],
            ))
    _log_pairs(pairs)

    logger.info("")
    logger.info(" XY N2D bond 5 (C to A of the next row):")
    pairs = []
    for row in y_rows:
        row_next = (row + 1) % ly if bcy == 'PBC' else row + 1
        label = ['XY', 'NN', 'N2D', 'PBC'] if row_next < row else ['XY', 'NN', 'N2D']
        for col in range(lx):
            pairs.append(_emit((row, col, 2), (row_next, col, 0), label))
    _log_pairs(pairs)

    # Bond 6 written from its right-hand end: B(row_next, col) -- C(row, col+1).
    # When col is the last column, C lies in the next unit cell.
    logger.info("")
    logger.info(" XY N2D bond 6 (B to C of the next column):")
    pairs = []
    for row in y_rows:
        row_next = (row + 1) % ly if bcy == 'PBC' else row + 1
        label = ['XY', 'NN', 'N2D', 'PBC'] if row_next < row else ['XY', 'NN', 'N2D']
        for col in range(lx):
            pairs.append(_emit((row_next, col, 1), (row, col + 1, 2), label))
    _log_pairs(pairs)

    logger.info("")
    logger.info(" iBFG hexagon interactions:")

    n_hexagons = 0
    for row in y_rows:
        row_next = (row + 1) % ly if bcy == 'PBC' else row + 1
        for col in range(lx):
            col_next = col + 1
            # Same loop the finite map uses. col_next is not reduced modulo
            # lx: the two sites in the following column belong to the next
            # cell, and an on-site term there is recorded on the matching
            # site of this cell because every cell carries the same operator.
            hexagon = [
                (row,      col,      1),
                (row,      col,      2),
                (row_next, col,      0),
                (row_next, col,      1),
                (row,      col_next, 2),
                (row,      col_next, 0),
            ]
            if len(set(hexagon)) != 6:
                raise ValueError(
                    f"Degenerate iBFG hexagon at unit cell ({row}, {col})"
                )

            indices = [
                _unit_cell_index(geo, r, c, u) for r, c, u in hexagon
            ]
            logger.info("")
            logger.info(
                " Hexagon %d at cell (%d,%d), sites: %s",
                n_hexagons + 1,
                row,
                col,
                " -> ".join(f"{site:02d}" for site in indices),
            )

            for r, c, u in hexagon:
                _emit_onsite(r, c, u)

            pair_logs = []
            for (r1, c1, u1), (r2, c2, u2) in combinations(hexagon, 2):
                pair_logs.append(_emit(
                    (r1, c1, u1), (r2, c2, u2), ['HEX_PAIR'],
                ))
            logger.info(" HEX_PAIR interactions:")
            _log_pairs(pair_logs)
            n_hexagons += 1

    interactions.sort(key=_interaction_sort_key)

    logger.info("")
    logger.info("iBFG hexagons: %d", n_hexagons)
    logger.info("iBFG interactions: %d", len(interactions))
    logger.info("")
    return interactions


def intrcmap_ibfg(geo: Geometry) -> List[Interaction]:
    """Generate the infinite-cylinder BFG interaction map.

    Parameters
    ----------
    geo:
        Kagome geometry with `bcx='IBC'`. `bcy='PBC'` closes the cylinder
        and `bcy='OBC'` leaves an infinite strip. The `lx` columns are one
        unit cell.

    Returns
    -------
    list[Interaction]
        Bare interactions labeled `XY`, `HEX_ONSITE`, and `HEX_PAIR`.
        Couplings and tensors are left unset. A bond into the next unit
        cell also carries `IBC`.
    """
    if geo.lattice != 'kagome':
        raise ValueError(
            f"iBFG interaction map requires lattice='kagome', got '{geo.lattice}'"
        )

    ly = geo.ly
    bcx = geo.cfg.get('bcx', 'OBC').upper()
    bcy = geo.cfg.get('bcy', 'OBC').upper()

    if bcx != 'IBC' or bcy not in {'OBC', 'PBC'}:
        raise ValueError(
            "iBFG interaction map requires bcx='IBC' and bcy in {OBC, PBC}, "
            f"got bcx='{bcx}', bcy='{bcy}'"
        )
    if bcy == 'PBC' and ly < 2:
        raise ValueError("A periodic iBFG circumference needs at least two cells")

    return _intrcmap_ibfg(geo)


def _interaction_sort_key(intr: Interaction) -> tuple[int, int, str]:
    """Return a deterministic site-ordering key for mixed interactions."""
    if isinstance(intr, Interaction1Site):
        return intr.site, intr.site, intr.label[0]
    elif isinstance(intr, Interaction2Site):
        return intr.leading_site, intr.terminal_site, intr.label[0]
    else:
        raise ValueError(f"Unsupported interaction type: {type(intr)}")
