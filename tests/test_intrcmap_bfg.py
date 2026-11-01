"""Finite BFG map and the separate infinite-cylinder iBFG map."""

import importlib.util
import logging
import unittest
from collections import Counter
from itertools import combinations
from pathlib import Path

from alice.network.interaction import Interaction1Site, Interaction2Site
from alice.physics.geometry import build_geometry
from alice.physics.kagome import intrcmap_kagome


ROOT = Path(__file__).resolve().parents[1]


def load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


bfg_map = load_module("bfg_map", ROOT / "intrcmaps/intrcmap_bfg.py")
ibfg_map = load_module("ibfg_map", ROOT / "intrcmaps/intrcmap_ibfg.py")
ibfg_model = load_module("ibfg_model", ROOT / "models/model_ibfg.py")


def geometry(lx, ly, bcx, bcy, traverse="sequential"):
    return build_geometry({
        "lattice": "kagome",
        "lx": lx,
        "ly": ly,
        "bcx": bcx,
        "bcy": bcy,
        "traverse": traverse,
    })


def _index(geo, row, col, u):
    """MPS index, with columns outside [0, lx) living in another cell."""
    cell, col_in = divmod(col, geo.lx)
    return geo.to_1d((row, col_in, u)) + cell * geo.L


def _order(left, right, n_sites):
    """Fold a bond into the form build_ihamiltonian accepts."""
    leading, terminal = sorted((left, right))
    if leading < 0:
        shift = ((-leading + n_sites - 1) // n_sites) * n_sites
        leading += shift
        terminal += shift
    if leading >= n_sites:
        shift = (leading // n_sites) * n_sites
        leading -= shift
        terminal -= shift
    return leading, terminal


def expected_infinite(geo, bcy):
    """Lattice bonds of one infinite cell, independent of the plugin's loops.

    Bond 6 is written in its textbook direction, toward the column on the
    left, which may be the previous cell. That is the opposite end from the
    one the plugin emits, so a matching multiset checks the shift.
    """
    lx, ly, n_sites = geo.lx, geo.ly, geo.L
    xy = []
    for row in range(ly):
        for col in range(lx):
            for u, v in ((0, 1), (0, 2), (1, 2)):
                xy.append(_order(
                    _index(geo, row, col, u), _index(geo, row, col, v), n_sites,
                ))
            xy.append(_order(
                _index(geo, row, col, 1), _index(geo, row, col + 1, 0), n_sites,
            ))

    y_rows = range(ly) if bcy == "PBC" else range(ly - 1)
    for row in y_rows:
        row_next = (row + 1) % ly if bcy == "PBC" else row + 1
        for col in range(lx):
            xy.append(_order(
                _index(geo, row, col, 2),
                _index(geo, row_next, col, 0),
                n_sites,
            ))
            xy.append(_order(
                _index(geo, row, col, 2),
                _index(geo, row_next, col - 1, 1),
                n_sites,
            ))

    onsites = []
    pairs = []
    for row in y_rows:
        row_next = (row + 1) % ly if bcy == "PBC" else row + 1
        for col in range(lx):
            hexagon = [
                _index(geo, row, col, 1),
                _index(geo, row, col, 2),
                _index(geo, row_next, col, 0),
                _index(geo, row_next, col, 1),
                _index(geo, row, col + 1, 2),
                _index(geo, row, col + 1, 0),
            ]
            for site in hexagon:
                onsites.append(site % n_sites)
            for left, right in combinations(hexagon, 2):
                pairs.append(_order(left, right, n_sites))
    return xy, onsites, pairs


class MapTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        logging.disable(logging.INFO)

    @classmethod
    def tearDownClass(cls):
        logging.disable(logging.NOTSET)

    def test_finite_cylinder_stays_inside_the_patch(self):
        geo = geometry(3, 2, "OBC", "PBC")
        terms = bfg_map.intrcmap_bfg(geo)
        xy = [t for t in terms if "XY" in t.label]
        onsites = [t for t in terms if isinstance(t, Interaction1Site)]
        pairs = [t for t in terms if "HEX_PAIR" in t.label]

        self.assertEqual(len(xy), len(intrcmap_kagome(geo)))
        self.assertEqual(len(onsites), 6 * (3 - 1) * 2)
        self.assertEqual(len(pairs), 15 * (3 - 1) * 2)
        for term in terms:
            if isinstance(term, Interaction2Site):
                self.assertLess(term.terminal_site, geo.L)
                self.assertNotIn("IBC", term.label)

    def test_rejects_unknown_and_short_periodic_directions(self):
        with self.assertRaises(ValueError):
            bfg_map.intrcmap_bfg(geometry(2, 2, "IBC", "PBC"))
        with self.assertRaisesRegex(ValueError, "at least two cells"):
            bfg_map.intrcmap_bfg(geometry(2, 1, "OBC", "PBC"))
        with self.assertRaisesRegex(ValueError, "at least two cells"):
            bfg_map.intrcmap_bfg(geometry(1, 2, "PBC", "OBC"))
        with self.assertRaisesRegex(ValueError, "bcx='IBC'"):
            ibfg_map.intrcmap_ibfg(geometry(2, 2, "OBC", "PBC"))

    def test_one_column_cylinder_matches_hand_count(self):
        geo = geometry(1, 2, "IBC", "PBC")
        terms = ibfg_map.intrcmap_ibfg(geo)
        xy = {
            (t.leading_site, t.terminal_site): t
            for t in terms if "XY" in t.label
        }
        self.assertEqual(set(xy), {
            (0, 1), (0, 2), (1, 2),
            (3, 4), (3, 5), (4, 5),
            (1, 6), (4, 9),
            (2, 3), (0, 5),
            (4, 8), (1, 11),
        })
        self.assertIn("IBC", xy[(1, 6)].label)
        self.assertIn("PBC", xy[(0, 5)].label)
        self.assertNotIn("IBC", xy[(0, 5)].label)
        self.assertIn("IBC", xy[(1, 11)].label)
        self.assertIn("PBC", xy[(1, 11)].label)
        self.assertEqual(
            Counter(t.site for t in terms if isinstance(t, Interaction1Site)),
            Counter({site: 2 for site in range(6)}),
        )

    def test_infinite_bonds_match_the_lattice(self):
        cases = [
            (1, 2, "PBC", "sequential"),
            (1, 2, "PBC", "serpentine"),
            (2, 3, "PBC", "sequential"),
            (2, 2, "PBC", "serpentine"),
            (3, 2, "OBC", "sequential"),
            (1, 3, "OBC", "sequential"),
        ]
        for lx, ly, bcy, traverse in cases:
            with self.subTest(lx=lx, ly=ly, bcy=bcy, traverse=traverse):
                geo = geometry(lx, ly, "IBC", bcy, traverse)
                terms = ibfg_map.intrcmap_ibfg(geo)
                xy, onsites, pairs = expected_infinite(geo, bcy)
                got_xy = [
                    (t.leading_site, t.terminal_site)
                    for t in terms if "XY" in t.label
                ]
                got_on = [t.site for t in terms if isinstance(t, Interaction1Site)]
                got_pairs = [
                    (t.leading_site, t.terminal_site)
                    for t in terms if "HEX_PAIR" in t.label
                ]
                self.assertEqual(Counter(got_xy), Counter(xy))
                self.assertEqual(Counter(got_on), Counter(onsites))
                self.assertEqual(Counter(got_pairs), Counter(pairs))
                self.assertTrue(any(t >= geo.L for _, t in got_xy))
                for term in terms:
                    if isinstance(term, Interaction2Site):
                        self.assertGreaterEqual(term.leading_site, 0)
                        self.assertLess(term.leading_site, geo.L)
                        self.assertGreater(term.terminal_site, term.leading_site)
                        crossed = term.terminal_site >= geo.L
                        self.assertEqual(crossed, "IBC" in term.label)


class HamiltonianTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        logging.disable(logging.INFO)

    @classmethod
    def tearDownClass(cls):
        logging.disable(logging.NOTSET)

    def test_product_state_energy_is_the_hexagon_sum(self):
        from alice import build_ihamiltonian, init_imps, vumps

        geo = geometry(1, 2, "IBC", "PBC")
        terms = ibfg_map.intrcmap_ibfg(geo)
        spc, ops = ibfg_model.build_ibfg(
            terms, geo.L, symmetry="U1", spin=0.5, Jz=1.0, Jperp=0.25,
        )
        charges = [sector.charge for sector in spc.sectors]
        self.assertEqual(charges, [-1, 1])

        # Every infinite-boundary bond longer than one site needs the
        # identity string. The corner bond of this cell has span 10.
        corner = next(
            t for t in terms
            if "XY" in t.label and t.leading_site == 1 and t.terminal_site == 11
        )
        self.assertIsNotNone(corner.intermid_tnsr)
        self.assertAlmostEqual(corner.cpl, -0.5)

        for config in ([1, 0, 1, 0, 1, 0], [1, 1, 1, 0, 0, 0]):
            with self.subTest(config=config):
                expected = _hexagon_energy(geo, config, spc) / geo.L
                impo = build_ihamiltonian(terms, geo.L, spc)
                imps = init_imps(geo.L, spc, ops, config=config)
                density = vumps.energy_density(imps, impo, tol=1e-12, maxiter=32)
                self.assertAlmostEqual(density, expected, places=8)


def _hexagon_energy(geo, config, spc):
    """(Jz/2) sum of squared hexagon charges for Jz = 1, XY omitted.

    A product state in the Sz basis has vanishing transverse energy, and
    the squared charge factors into the sum of the on-site eigenvalues.
    """
    def sz(row, col, u):
        _cell, col_in = divmod(col, geo.lx)
        site = geo.to_1d((row, col_in, u))
        return 0.5 * spc.sectors[config[site]].charge

    total = 0.0
    for row in range(geo.ly):
        row_next = (row + 1) % geo.ly
        for col in range(geo.lx):
            charge = sum(sz(*coord) for coord in (
                (row, col, 1),
                (row, col, 2),
                (row_next, col, 0),
                (row_next, col, 1),
                (row, col + 1, 2),
                (row, col + 1, 0),
            ))
            total += 0.5 * charge * charge
    return total


if __name__ == "__main__":
    unittest.main()
