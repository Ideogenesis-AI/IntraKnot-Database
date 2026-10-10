"""Check the finite-cylinder particle-hole correction in the model builder."""

import importlib.util
import itertools
import math
import unittest
from pathlib import Path
from types import SimpleNamespace

from alice.network.interaction import Interaction1Site, InteractionNSite


ROOT = Path(__file__).resolve().parents[1]


def load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


model = load_module("moire_ph_model", ROOT / "models/model_moire_c4.py")
intrcmap = load_module("moire_ph_map", ROOT / "intrcmaps/intrcmap_moire_c4.py")


def coefficients(**parameters):
    terms = intrcmap.intrcmap_moire_c4(SimpleNamespace(lx=4, cfg={"ny": 2}))
    _, info = model.build_moire_c4(
        terms, 4, ny=2, lam=4.0, int_tol=1e-10, **parameters
    )
    return terms, info


class ChemicalPotentialTests(unittest.TestCase):
    def test_u0_matches_legacy_c_and_rejects_ambiguous_config(self):
        legacy, _ = coefficients(C=1.0, V0=0.2, mu=0.3)
        updated, _ = coefficients(
            U0=1 / math.sqrt(2 * math.pi), V0=0.2, mu=0.3
        )
        self.assertEqual(
            [term.cpl for term in legacy], [term.cpl for term in updated]
        )
        with self.assertRaisesRegex(ValueError, "not both"):
            coefficients(U0=1.0, C=1.0)
        with self.assertRaisesRegex(ValueError, "finite number"):
            coefficients(U0=1.0, mu="ph")
        with self.assertRaisesRegex(ValueError, "boolean"):
            coefficients(U0=1.0, ph_correction="false")

    def test_ph_centered_density_energy_is_complement_symmetric(self):
        terms, info = coefficients(U0=1 / math.sqrt(2 * math.pi), V0=0.0, mu=0.0)
        density = {}
        diagonal = {}
        for term in terms:
            if isinstance(term, Interaction1Site):
                diagonal[term.site] = term.cpl
            elif isinstance(term, InteractionNSite) and "DENS" in term.label:
                i, j, _, _ = term.sites
                density[i, j] = term.cpl

        def energy(occupation):
            return sum(
                value * occupation[i] * occupation[j]
                for (i, j), value in density.items()
            ) + sum(
                diagonal[i] * occupation[i] for i in range(4)
            )

        for occupation in itertools.product((0, 1), repeat=4):
            complement = tuple(1 - value for value in occupation)
            self.assertAlmostEqual(energy(occupation), energy(complement))
        profile = info["_moire_mu_profile"]
        self.assertGreater(max(profile) - min(profile), 0)

    def test_uniform_mu_shifts_only_diagonal_number_terms(self):
        centered, _ = coefficients(U0=1 / math.sqrt(2 * math.pi), V0=0.0, mu=0.0)
        shifted, info = coefficients(U0=1 / math.sqrt(2 * math.pi), V0=0.0, mu=0.3)
        for original, offset in zip(centered, shifted):
            expected = -0.3 if isinstance(original, Interaction1Site) else 0.0
            self.assertAlmostEqual(offset.cpl - original.cpl, expected)
        for total, ph in zip(info["_moire_mu_profile"], info["_moire_ph_mu_profile"]):
            self.assertAlmostEqual(total - ph, 0.3)

    def test_ph_flag_removes_only_correction_and_defaults_on(self):
        parameters = {"U0": 1 / math.sqrt(2 * math.pi), "V0": 0.2, "mu": 0.3}
        default, _ = coefficients(**parameters)
        enabled, info_on = coefficients(**parameters, ph_correction=True)
        disabled, info_off = coefficients(**parameters, ph_correction=False)
        self.assertEqual(
            [term.cpl for term in default], [term.cpl for term in enabled]
        )

        def key(term):
            if isinstance(term, Interaction1Site):
                return ("diag", term.site)
            if isinstance(term, InteractionNSite):
                return ("pair", tuple(term.sites))
            return ("hop", term.leading_site, term.terminal_site)

        disabled_by_key = {key(term): term for term in disabled}
        for on in enabled:
            off = disabled_by_key[key(on)]
            expected = (
                info_on["_moire_ph_mu_profile"][on.site]
                if isinstance(on, Interaction1Site)
                else 0.0
            )
            self.assertAlmostEqual(off.cpl - on.cpl, expected)
            if isinstance(off, Interaction1Site):
                self.assertAlmostEqual(
                    off.cpl, info_off["_moire_B"][off.site, off.site].real - 0.3
                )
        for total in info_off["_moire_mu_profile"]:
            self.assertAlmostEqual(total, 0.3)


if __name__ == "__main__":
    unittest.main()
