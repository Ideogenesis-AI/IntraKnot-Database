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


"""LLL-projected C4 moire model builder for Alice.

Implements the cylinder one-body matrix B from reciprocal harmonics and the
Yukawa interaction table V_{p,r}, then populates Interaction1Site /
Interaction2Site / InteractionNSite objects for AutoMPO construction.
"""

from __future__ import annotations

import logging
from typing import Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np
from numpy.polynomial.legendre import leggauss

from nicole import Direction, Index, Tensor, einsum, identity

from alice.network.interaction import (
    Interaction,
    Interaction1Site,
    Interaction2Site,
    InteractionNSite,
)
from alice.physics.system import build_fermionic

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Local 4-order MPO helpers (mirrors alice.physics.system private helpers)
# ---------------------------------------------------------------------------

def _make_leading4(op3: Tensor) -> Tensor:
    op4 = op3.clone()
    op4.insert_index(0, direction=Direction.IN, itag='_aux_')
    return op4.permute([0, 3, 1, 2])


def _make_terminal4(op3: Tensor) -> Tensor:
    op4 = op3.clone()
    op4.insert_index(3, direction=Direction.OUT, itag='_aux_')
    return op4.permute([2, 3, 0, 1])


def _make_intermid4(string_op: Tensor, leading_tnsr: Tensor) -> Tensor:
    op_idx_out = leading_tnsr.indices[1]
    op_id = identity(op_idx_out.flip())
    return einsum('op,rs->oprs', op_id, string_op)


def _channel_templates(ops: Dict[str, Tensor]) -> Dict[str, Tensor]:
    """Derive single-channel fermionic templates from build_fermionic ops."""
    C4 = _make_leading4(ops['C'])
    C4dag = _make_terminal4(ops['Cd'])
    F4 = _make_leading4(ops['F'])
    F4dag = _make_terminal4(ops['Fd'])
    return {
        'C4': C4,
        'C4dag': C4dag,
        'C4mid': _make_intermid4(ops['Z'], C4),
        'F4': F4,
        'F4dag': F4dag,
        'F4mid': _make_intermid4(ops['Z'], F4),
        'I4': ops['I4'],
        'N4': ops['N4'],
        'G4': ops['G4'],
        'G4dag': ops['G4dag'],
        'Z4mid': ops['Z4mid'],
    }


def _onebody_window(i: int, j: int, ch: Dict[str, Tensor]) -> List[Tensor]:
    """MPO window for c†_j c_i on the contiguous span [min(i,j), max(i,j)].

    Alice's JW-dressed channels match G4 with positive coupling for both
    directions, so no extra minus is applied.
    """
    if i == j:
        return [ch['N4'].clone()]
    left, right = (i, j) if i < j else (j, i)
    span = right - left + 1
    if j > i:
        leading, mid, terminal = ch['F4'], ch['F4mid'], ch['F4dag']
    else:
        leading, mid, terminal = ch['C4'], ch['C4mid'], ch['C4dag']
    window = [leading.clone()]
    for _ in range(span - 2):
        window.append(mid.clone())
    window.append(terminal.clone())
    return window


def _four_op_window(m: int, n: int, k: int, l: int, ch: Dict[str, Tensor]) -> List[Tensor]:
    """Concatenate two disjoint one-body windows for (c†_m c_l)(c†_n c_k)."""
    a_hi = max(m, l)
    b_lo = min(n, k)
    if a_hi >= b_lo:
        raise ValueError(
            f"Factors not disjoint/ordered for (m,n,k,l)=({m},{n},{k},{l})"
        )
    win_a = _onebody_window(l, m, ch)
    win_b = _onebody_window(k, n, ch)
    gap = [ch['I4'].clone() for _ in range(b_lo - a_hi - 1)]
    return win_a + gap + win_b


# ---------------------------------------------------------------------------
# Geometry / harmonic helpers
# ---------------------------------------------------------------------------

def _c4_geometry(lB: float, ny: int) -> Tuple[float, float, float, float]:
    """Return (a0, Ly, alpha, g0) for one flux per square cell."""
    a0 = np.sqrt(2.0 * np.pi) * lB
    Ly = ny * a0
    alpha = 2.0 * np.pi * lB / Ly
    g0 = 2.0 * np.pi / a0
    return float(a0), float(Ly), float(alpha), float(g0)


def _c4_harmonics(V0: float, g0: float) -> List[Tuple[float, float, complex]]:
    """Four C4 harmonics with amplitude V0 at (±g0, 0) and (0, ±g0)."""
    return [
        (g0, 0.0, complex(V0)),
        (-g0, 0.0, complex(V0)),
        (0.0, g0, complex(V0)),
        (0.0, -g0, complex(V0)),
    ]


# ---------------------------------------------------------------------------
# One-body matrix B (Eq. 10)
# ---------------------------------------------------------------------------

def onebody_matrix(
    N: int,
    Ly: float,
    lB: float,
    harmonics: Sequence[Tuple[float, float, complex]],
    *,
    comm_tol: float = 1e-10,
    imag_tol: float = 1e-10,
) -> np.ndarray:
    """Build the complex one-body matrix B from reciprocal harmonics.

    Parameters
    ----------
    N:
        Number of retained orbitals.
    Ly:
        Cylinder circumference.
    lB:
        Magnetic length.
    harmonics:
        Sequence of `(gx, gy, Vg)` triples.
    comm_tol:
        Tolerance for the commensurability check on `gy * Ly / (2π)`.
    imag_tol:
        Maximum allowed imaginary part after assembly (Alice `cpl` is real).

    Returns
    -------
    ndarray
        Complex `(N, N)` matrix. For the C4 potential it is real up to
        `imag_tol`.
    """
    B = np.zeros((N, N), dtype=complex)
    X = 2.0 * np.pi * (lB ** 2) * np.arange(N) / Ly

    for gx, gy, Vg in harmonics:
        s_raw = gy * Ly / (2.0 * np.pi)
        s = int(round(s_raw))
        if abs(s_raw - s) > comm_tol:
            raise ValueError(
                f"Incommensurate harmonic gy={gy}: s_raw={s_raw} not near "
                f"integer (tol={comm_tol})"
            )
        # Use the commensurate gy to remove roundoff.
        gy_c = 2.0 * np.pi * s / Ly
        form = np.exp(-0.25 * (lB ** 2) * (gx ** 2 + gy_c ** 2))
        for n in range(N):
            m = n + s
            if 0 <= m < N:
                phase = np.exp(1j * gx * (X[m] + X[n]) / 2.0)
                B[m, n] += Vg * form * phase

    if np.max(np.abs(B - B.conj().T)) > imag_tol:
        raise ValueError("One-body matrix B is not Hermitian within tolerance")
    if np.max(np.abs(B.imag)) > imag_tol:
        raise ValueError(
            f"One-body matrix B has imaginary parts up to "
            f"{np.max(np.abs(B.imag))}; Alice couplings must be real"
        )
    return B


# ---------------------------------------------------------------------------
# Yukawa interaction table V_{p,r} (Eq. 22–24)
# ---------------------------------------------------------------------------

def _integrate_I(a: float, b: float, tol: float) -> Tuple[float, float]:
    """Evaluate the nonoscillatory integral I(a,b) of Eq. (24).

    Uses Gauss-Legendre quadrature on [0, v_max] with v_max = asinh(w_max / b),
    doubling the node count until successive estimates differ by less than
    `tol`.
    """
    if b <= 0.0:
        raise ValueError("Finite screening requires b > 0")

    # Tail bound: ΔV <= P * exp(-w_max^2/2) / w_max^2; for the dimensionless
    # integral allocate tol/2 to the tail → choose w_max large enough that
    # exp(-w^2/2)/w^2 <= tol/2 (P handled by the caller).
    w_max = 8.0
    while True:
        tail = np.exp(-0.5 * w_max ** 2) / (w_max ** 2)
        if tail <= tol / 2.0:
            break
        w_max += 2.0
        if w_max > 40.0:
            break
    v_max = float(np.arcsinh(w_max / b))

    def integrand(v: np.ndarray) -> np.ndarray:
        return np.exp(
            -0.5 * (b ** 2) * np.sinh(v) ** 2
            - 0.5 * (a ** 2) / np.cosh(v) ** 2
        )

    n_nodes = 16
    prev = None
    estimate = 0.0
    for _ in range(12):
        x, w = leggauss(n_nodes)
        # Map [-1, 1] → [0, v_max]
        v = 0.5 * v_max * (x + 1.0)
        jac = 0.5 * v_max
        estimate = float(jac * np.dot(w, integrand(v)))
        if prev is not None and abs(estimate - prev) < tol / 2.0:
            err = abs(estimate - prev) + float(tail)
            return estimate, err
        prev = estimate
        n_nodes *= 2

    return estimate, abs(estimate - (prev or estimate)) + float(tail)


def interaction_table(
    N: int,
    Ly: float,
    lB: float,
    C: float,
    lam: float,
    *,
    int_tol: float = 1e-12,
) -> Tuple[np.ndarray, np.ndarray]:
    """Precompute the N×N Yukawa table V_{p,r} for p,r = 0…N-1.

    Returns
    -------
    V:
        Real table of shape `(N, N)`.
    err:
        Estimated absolute error per entry, same shape.
    """
    alpha = 2.0 * np.pi * lB / Ly
    kappa = lB / lam
    V = np.zeros((N, N), dtype=float)
    err = np.zeros((N, N), dtype=float)

    if C == 0.0:
        return V, err

    for p in range(N):
        Pp = (2.0 * C / Ly) * np.exp(-0.5 * (alpha * p) ** 2)
        b = np.sqrt((alpha * p) ** 2 + kappa ** 2)
        # Absolute tolerance on I: int_tol / (2 Pp) when Pp > 0.
        i_tol = int_tol / (2.0 * Pp) if Pp > 0.0 else int_tol
        for r in range(N):
            a = alpha * abs(r)
            I_val, I_err = _integrate_I(a, b, i_tol)
            V[p, r] = Pp * I_val
            err[p, r] = Pp * I_err

    return V, err


def lookup_V(V: np.ndarray, p: int, r: int) -> float:
    """Retrieve V_{p,r} using absolute indices (isotropic symmetries)."""
    return float(V[abs(p), abs(r)])


def ordered_pair_W(V: np.ndarray, m: int, n: int, k: int, l: int) -> float:
    """Ordered-pair coefficient W_{mnkl} (Eq. 32), or 0 if m+n ≠ k+l.

    Parameters
    ----------
    V:
        Yukawa table from `interaction_table`, indexed by `(|p|, |r|)`.
    m, n, k, l:
        Orbital indices of `c†_m c†_n c_k c_l`, ordered as `m < n` and `l < k`.

    Returns
    -------
    float
        Coefficient multiplying `c†_m c†_n c_k c_l` in the ordered-pair sum.

    Notes
    -----
    The two-body term is carried as a sum over ordered pairs alone,

        H_int = Σ_{m<n, l<k} W_{mnkl} c†_m c†_n c_k c_l,

    which is the half-weighted convention: it equals
    `1/2 Σ_{mnkl} A_{mnkl} c†_m c†_n c_k c_l` with all four indices summed
    freely, where `A_{mnkl} = V[m-l, m-k]` is the bare matrix element.
    Collapsing that unrestricted sum onto its ordered representatives gathers
    the four permutations `(m,n,k,l)`, `(n,m,k,l)`, `(m,n,l,k)` and `(n,m,l,k)`
    — whose operator strings coincide up to sign — into
    `2 (A_{mnkl} - A_{mnlk})`. The `1/2` cancels that factor and leaves the
    antisymmetrized difference returned here.

    The convention is observable, not cosmetic: it fixes the two-body scale `C`
    relative to the one-body scale `V0`. A source writing the interaction
    unrestricted, `H_int = Σ_{mnkl} A_{mnkl} c†c†cc`, is twice as strong at
    equal `C`, so couplings taken from one convention must be halved or doubled
    before being used in the other.
    """
    if m + n != k + l:
        return 0.0
    return lookup_V(V, m - l, m - k) - lookup_V(V, m - k, m - l)


# ---------------------------------------------------------------------------
# Discard accounting
# ---------------------------------------------------------------------------

class _ChannelTally:
    """Running discard accounting for one coupling channel.

    The threshold is relative to a channel reference scale so that the same
    `cpl_rtol` means the same relative accuracy at every point of a parameter
    scan: `max|B|` is linear in `V0` and `V[0,0]` is linear in `C`, so an
    absolute threshold would drift in aggressiveness as either is varied.

    The reported figure of merit is the summed discarded magnitude, not the
    largest one: the energy error from dropping terms is bounded by that sum
    (occupation factors are at most one), and a maximum alone cannot
    distinguish a handful of negligible terms from millions of them.
    """

    def __init__(self, name: str, ref: float, cpl_rtol: float) -> None:
        self.name = name
        self.ref = ref
        self.thresh = cpl_rtol * ref
        self.n_total = 0
        self.n_drop = 0
        self.sum_total = 0.0
        self.sum_drop = 0.0
        self.max_drop = 0.0

    def keep(self, cpl: float) -> bool:
        """Record `cpl` and report whether it survives the threshold.

        A zero reference means the channel vanishes identically (`V0 = 0` or
        `C = 0`), in which case every term is dropped. That path must be taken
        explicitly: a relative comparison against a zero reference would
        otherwise retain every coupling as an explicit zero in the MPO.
        """
        mag = abs(cpl)
        self.n_total += 1
        self.sum_total += mag
        if self.ref == 0.0 or mag < self.thresh:
            self.n_drop += 1
            self.sum_drop += mag
            self.max_drop = max(self.max_drop, mag)
            return False
        return True

    @property
    def fraction(self) -> float:
        """Discarded share of the total coupling weight in this channel."""
        if self.sum_total == 0.0:
            return 0.0
        return self.sum_drop / self.sum_total

    def report(self, warn_frac: float) -> None:
        """Log the discard budget, warning when it exceeds `warn_frac`."""
        logger.info(
            f"  {self.name}: kept {self.n_total - self.n_drop}/{self.n_total}, "
            f"ref={self.ref:.3e}, thresh={self.thresh:.3e}"
        )
        logger.info(
            f"    discarded weight {self.sum_drop:.3e} of {self.sum_total:.3e} "
            f"(fraction {self.fraction:.2e}, largest {self.max_drop:.3e})"
        )
        if self.fraction > warn_frac:
            logger.warning(
                f"  {self.name} truncation discards {self.fraction:.2%} of the "
                f"total coupling weight, above warn_frac={warn_frac:.1e}. The "
                f"energy error is bounded by {self.sum_drop:.3e}; lower cpl_rtol."
            )


# ---------------------------------------------------------------------------
# Model builder
# ---------------------------------------------------------------------------

def build_moire_c4(
    interactions: List[Interaction],
    L: int,
    *,
    symmetry: str = 'U1',
    lB: float = 1.0,
    ny: int = 2,
    V0: float = 0.3,
    C: float = 1.0,
    lam: float = 2.0,
    cpl_rtol: float = 1e-10,
    warn_frac: float = 1e-6,
    comm_tol: float = 1e-10,
    int_tol: float = 1e-12,
    space_fn: Optional[Callable] = None,
    **_ignored,
) -> Tuple[Index, Dict[str, Tensor]]:
    """Populate interactions for the LLL-projected C4 moire model.

    Parameters
    ----------
    interactions:
        List from `intrcmap_moire_c4`. Modified in place.
    L:
        Chain length (number of orbitals). Must equal `N` used by the
        interaction map.
    symmetry:
        Fermion symmetry; must be `'U1'`.
    lB:
        Magnetic length.
    ny:
        Number of flux cells along the cylinder circumference.
    V0:
        C4 potential amplitude.
    C:
        Yukawa interaction strength. Its scale relative to `V0` depends on the
        half-weighted two-body convention documented in `ordered_pair_W`.
    lam:
        Screening length λ, as a bare length in the same units as `lB`. Sources
        that quote λ in units of the orbital spacing `2π lB² / Ly` must be
        converted before being passed here.
    cpl_rtol:
        Relative coupling-truncation threshold. A term is discarded when its
        magnitude falls below `cpl_rtol` times its channel reference scale:
        `max|B|` for the one-body terms and `V[0,0]` for the two-body terms.
        This is the only truncation control — the interaction map enumerates
        every admissible term, since only this stage knows the coefficients.
    warn_frac:
        Warn when the discarded coupling weight of either channel exceeds this
        fraction of that channel's total weight.
    comm_tol:
        Commensurability tolerance for potential harmonics.
    int_tol:
        Absolute integration tolerance per V-table entry.
    space_fn:
        Optional replacement for `build_fermionic`.
    """
    if symmetry != 'U1':
        raise ValueError(f"moire_c4 requires symmetry='U1', got '{symmetry}'")
    # `**_ignored` would otherwise swallow a stale `vtol`, silently falling back
    # to the `cpl_rtol` default. Since the two differ in meaning (absolute vs
    # relative to a channel reference), that would change the Hamiltonian
    # without any indication in the log.
    if 'vtol' in _ignored:
        raise ValueError(
            "'vtol' has been replaced by 'cpl_rtol', which is relative to the "
            "channel reference scale (max|B| for one-body, V[0,0] for two-body) "
            "rather than absolute. Set 'cpl_rtol' explicitly."
        )

    _space = space_fn if space_fn is not None else build_fermionic
    spc, ops = _space(symmetry)
    ch = _channel_templates(ops)

    a0, Ly, alpha, g0 = _c4_geometry(lB, ny)
    harmonics = _c4_harmonics(V0, g0)
    B = onebody_matrix(L, Ly, lB, harmonics, comm_tol=comm_tol)
    Vtable, Verr = interaction_table(L, Ly, lB, C, lam, int_tol=int_tol)

    logger.info("─" * 60)
    logger.info("Moire C4 Coefficients".center(60))
    logger.info("─" * 60)
    logger.info("")
    logger.info(f"  lB={lB}, ny={ny}, a0={a0:.6f}, Ly={Ly:.6f}")
    logger.info(f"  alpha={alpha:.6f}, V0={V0}, C={C}, lam={lam}")
    logger.info(f"  max|B|={np.max(np.abs(B)):.6e},  V00={Vtable[0, 0]:.6e}")
    logger.info(f"  cpl_rtol={cpl_rtol:.1e}, warn_frac={warn_frac:.1e}")
    logger.info("")

    onebody = _ChannelTally('one-body', float(np.max(np.abs(B))), cpl_rtol)
    twobody = _ChannelTally('two-body', float(Vtable[0, 0]), cpl_rtol)

    # Cached only for the density-density channel (p == 0), where the window is
    # [N4, I4 ... I4, N4] and its span |r| + 1 is fixed by r alone. Pair-hopping
    # windows depend on the absolute site positions, not just (p, r), so they
    # cannot be shared and are rebuilt per term.
    dens_window_cache: Dict[int, List[Tensor]] = {}

    for intr in interactions:
        if isinstance(intr, Interaction1Site) and 'DIAG' in intr.label:
            # Diagonal B_nn is real for C4.
            cpl = float(B[intr.site, intr.site].real)
            if onebody.keep(cpl):
                intr.cpl = cpl
                intr.tnsr = ch['N4'].clone()
            else:
                intr.cpl = 0.0
            continue

        if isinstance(intr, Interaction2Site) and 'HOP' in intr.label:
            i, j = intr.leading_site, intr.terminal_site
            # Hermitian hopping: store B_ji as cpl with G4/G4dag (= c†_i c_j + h.c.
            # channel sum). For real C4, B_ji = B_ij = V0 e^{-π/2}.
            cpl = float(B[j, i].real)
            if onebody.keep(cpl):
                intr.cpl = cpl
                intr.leading_tnsr = ch['G4'].clone()
                intr.terminal_tnsr = ch['G4dag'].clone()
                if j > i + 1:
                    intr.intermid_tnsr = ch['Z4mid'].clone()
            else:
                intr.cpl = 0.0
            continue

        if isinstance(intr, InteractionNSite) and (
            'PAIR' in intr.label or 'DENS' in intr.label
        ):
            if len(intr.sites) != 4:
                raise ValueError(
                    f"Expected sites=[m,n,k,l], got {intr.sites}"
                )
            m, n, k, l = intr.sites
            W = ordered_pair_W(Vtable, m, n, k, l)
            if not twobody.keep(W):
                intr.cpl = 0.0
                continue
            intr.cpl = float(W)

            p = m - l
            if p == 0:
                # Density-density n_m n_n on [m, n]. The window is fixed by the
                # separation n - m = |r| alone, so one template serves every m
                # at that separation. Pair hopping (p != 0) has no such
                # invariance: its window depends on the absolute positions.
                key = n - m
                if key not in dens_window_cache:
                    win = [ch['N4'].clone()]
                    for _ in range(key - 1):
                        win.append(ch['I4'].clone())
                    win.append(ch['N4'].clone())
                    dens_window_cache[key] = win
                intr.tnsrs = [t.clone() for t in dens_window_cache[key]]
            else:
                intr.tnsrs = _four_op_window(m, n, k, l, ch)
            continue

    onebody.report(warn_frac)
    twobody.report(warn_frac)
    logger.info("")

    # Attach coefficient tables on the ops dict for tests / inspection.
    ops['_moire_B'] = B
    ops['_moire_V'] = Vtable
    ops['_moire_Verr'] = Verr
    # Expose the discard budget so it can be audited programmatically rather
    # than only scraped from the log.
    ops['_moire_discard'] = {
        t.name: {
            'ref': t.ref, 'thresh': t.thresh,
            'n_total': t.n_total, 'n_drop': t.n_drop,
            'sum_total': t.sum_total, 'sum_drop': t.sum_drop,
            'max_drop': t.max_drop, 'fraction': t.fraction,
        }
        for t in (onebody, twobody)
    }
    ops['_moire_params'] = {
        'lB': lB, 'ny': ny, 'a0': a0, 'Ly': Ly, 'alpha': alpha,
        'V0': V0, 'C': C, 'lam': lam, 'g0': g0,
    }

    return spc, ops
