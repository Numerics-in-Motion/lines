# -*- coding: utf-8 -*-
"""How hard a washing line pulls on its posts -- the registered solver.

REGISTRATION (frozen by the design review before any registered cell was computed:
`analyze/260906/044_registration.md`, ruling MINOR REVISION on 044d). Nothing in this file may be
changed to suit a result.

THE QUESTION

One rope between two posts, one load of washing. Take the slack out in registered steps: how does
the horizontal pull on the posts change, and how large does it get compared with the weight of the
washing itself?

THE MODEL

Geometrically exact, tension-only elastic bar network between two fixed, no-slip endpoints at
equal elevation. Equilibrium is the minimiser of

    U = sum_e (EA / 2 l0e)(l_e - l0e)^2  [only where l_e > l0e]  +  sum_i W_i y_i

found by Newton on the analytic tangent stiffness, with mesh continuation. Garments are point
weights at fixed MATERIAL coordinates, so a node sits exactly at every garment. Sag is the support
elevation minus the lowest point. H is the horizontal component of the end-element force, which is
NOT the total endpoint force T.

THE SPECIMEN IS SOURCED, NOT ASSUMED. S3i Group stainless wire-rope tables, read 2026-09-17:
7x7 rope modulus E = 57.3 kN/mm^2 (1x19 107.5, 7x19 47.5 -- a CONSTRUCTION constant, not the
200 GPa of the wire material), 3 mm nominal, MBL 5030 N, published SWL 85 kg, mass 3.46 kg/100 m.
The supplier's stretch formula defines A = pi D^2 / 4, so EA = E x NOMINAL area = 405 030 N.
An earlier registration used a metallic area derived from the mass because the smaller number
looked like the conservative reading; the source states the definition, and conservatism is not a
licence to override it.

TWO EARLIER SOLVERS PRODUCED NUMBERS THAT WERE NOT EQUILIBRIA. Steepest descent returned H = 0 for
every case. Newton without a convergence check returned a full table of plausible values taken
from non-converged states (||grad|| up to 275 N against a 36 N load). `solve()` therefore RAISES
rather than return a non-converged state, and its tolerance is the double-precision round-off
floor rather than a number chosen here. The controls in `line_probe_checks.py` (C1-C8) are what
caught both.
"""
from __future__ import annotations

import numpy as np

G = 9.80665

# ---------------------------------------------------------------- frozen registration
SPAN = 6.000                     # m, both ends fixed, no-slip, equal elevation
N_BASE = 400                     # base intervals; + 8 garment nodes = 408 elements
GARMENT_KG = 0.45
N_GARMENTS = 8
GARMENTS = [(k / 9.0, GARMENT_KG * G) for k in range(1, N_GARMENTS + 1)]
W_WASHING = N_GARMENTS * GARMENT_KG * G          # 35.304 N -- the denominator of H/W
L0S = (6.300, 6.200, 6.120, 6.080, 6.050, 6.030, 6.010, 6.000)

RHO_STEEL = 7850.0
E_ROPE = {"7x7": 57.3e9, "1x19": 107.5e9, "7x19": 47.5e9}
# name,            d mm, construction, MBL N, mass kg/m, published SWL kg
ROPES = [
    ("7x7 1.5 mm", 1.5, "7x7", 1260.0, 0.0086, 21.3),
    ("7x7 2 mm",   2.0, "7x7", 2240.0, 0.0154, 38.0),
    ("7x7 3 mm",   3.0, "7x7", 5030.0, 0.0346, 85.0),
    ("7x7 4 mm",   4.0, "7x7", 8940.0, 0.0614, 152.0),
]
REGISTERED_ROPE = "7x7 3 mm"

U_REG = 1.629e-3                 # N, measured (C7): max(round-off floor, mesh diff, start spread)
SEP_MULT = 5.0                   # adjacent reported H must differ by >= 5u
MESH_TOL = 1e-3                  # |H(400) - H(1600)| / H
H_W_SHIP = 5.0                   # THE SHIP LINE, declared before any registered cell was run


def rope(name=REGISTERED_ROPE):
    """Sourced rope properties. EA = E x NOMINAL area, as the supplier's stretch formula defines
    it; SWL is the supplier's PUBLISHED column, not MBL/6 recomputed here. The fill factor is a
    diagnostic that the supplier's masses and diameters are mutually consistent -- it is NOT used
    in EA."""
    for (nm, d_mm, con, mbl, mass, swl_kg) in ROPES:
        if nm != name:
            continue
        A_nom = np.pi * (d_mm * 1e-3) ** 2 / 4
        E = E_ROPE[con]
        return dict(name=nm, d_mm=d_mm, A_nom=A_nom, E=E, EA=E * A_nom,
                    fill=mass / (RHO_STEEL * A_nom),
                    w=mass * G, MBL=mbl, SWL=swl_kg * G, mass=mass)
    raise KeyError(name)


class Line:
    def __init__(self, span, L0, EA, w_line, garments, n=200):
        """span, L0 in m; EA in N; w_line in N per m of UNSTRETCHED length; garments as
        [(material fraction, weight N), ...]."""
        self.span, self.L0, self.EA, self.w = span, L0, EA, w_line
        fracs = sorted({0.0, 1.0} | {f for f, _ in garments})
        # nodes at every garment and evenly between, so every element has one material length
        s = np.linspace(0.0, 1.0, n + 1)
        s = np.unique(np.round(np.concatenate([s, np.array(fracs)]), 12))
        self.s = s
        self.l0 = np.diff(s) * L0
        self.W = np.zeros(s.size)
        for i in range(s.size):
            lo = self.l0[i - 1] / 2 if i > 0 else 0.0
            hi = self.l0[i] / 2 if i < self.l0.size else 0.0
            self.W[i] += self.w * (lo + hi)
        for f, wt in garments:
            j = int(np.argmin(np.abs(s - f)))
            self.W[j] += wt
        self.free = np.arange(1, s.size - 1)

    def _xy(self, q):
        x = np.empty(self.s.size)
        y = np.empty(self.s.size)
        x[0], y[0] = 0.0, 0.0
        x[-1], y[-1] = self.span, 0.0
        x[self.free] = q[0::2]
        y[self.free] = q[1::2]
        return x, y

    def energy_grad(self, q):
        x, y = self._xy(q)
        dx, dy = np.diff(x), np.diff(y)
        l = np.hypot(dx, dy)
        stretch = l - self.l0
        k = self.EA / self.l0
        t = np.where(stretch > 0, k * stretch, 0.0)              # tension-only
        U = float(np.sum(np.where(stretch > 0, 0.5 * k * stretch ** 2, 0.0)) + np.sum(self.W * y))
        gx = np.zeros_like(x)
        gy = np.zeros_like(y)
        ux, uy = dx / l, dy / l
        np.add.at(gx, np.arange(l.size), -t * ux)
        np.add.at(gx, np.arange(l.size) + 1, t * ux)
        np.add.at(gy, np.arange(l.size), -t * uy)
        np.add.at(gy, np.arange(l.size) + 1, t * uy)
        gy += self.W
        g = np.empty(2 * self.free.size)
        g[0::2] = gx[self.free]
        g[1::2] = gy[self.free]
        return U, g, t, l

    def hessian(self, q):
        """Tangent stiffness of the bar network: k u u^T + (t/l)(I - u u^T) per element.

        Steepest descent could not solve this at all (the first probe returned H = 0 everywhere):
        the elements are ~4e5 N/m stiff while the loads are ~0.4 N, so the problem is far too
        ill-conditioned for a gradient step."""
        from scipy.sparse import coo_matrix
        x, y = self._xy(q)
        dx, dy = np.diff(x), np.diff(y)
        l = np.hypot(dx, dy)
        k = self.EA / self.l0
        t = np.where(l - self.l0 > 0, k * (l - self.l0), 0.0)
        kk = np.where(l - self.l0 > 0, k, 0.0)
        ux, uy = dx / l, dy / l
        idx = {n: i for i, n in enumerate(self.free)}
        rows, cols, vals = [], [], []
        for e in range(l.size):
            uu = np.array([[ux[e] * ux[e], ux[e] * uy[e]], [ux[e] * uy[e], uy[e] * uy[e]]])
            ke = kk[e] * uu + (t[e] / l[e]) * (np.eye(2) - uu)
            for (a, sa) in ((e, -1.0), (e + 1, +1.0)):
                for (b, sb) in ((e, -1.0), (e + 1, +1.0)):
                    if a not in idx or b not in idx:
                        continue
                    for r in range(2):
                        for c in range(2):
                            rows.append(2 * idx[a] + r)
                            cols.append(2 * idx[b] + c)
                            vals.append(sa * sb * ke[r, c])
        n = 2 * self.free.size
        return coo_matrix((vals, (rows, cols)), shape=(n, n)).tocsc()

    def solve(self, q0=None, iters=600, tol=1e-9, require=True):
        """Newton with a line search. RAISES if it does not converge.

        The first version of this probe reported H = 0 for every case, and the second reported a
        whole table of numbers that came from NON-CONVERGED states (|grad| up to 275 N against a
        36 N load) because solve() returned whatever the iteration happened to hold. A solver that
        can return a non-converged state silently will eventually have that state published."""
        from scipy.sparse import identity
        from scipy.sparse.linalg import spsolve
        if q0 is None:
            # start from the inextensible parabola that uses up the slack
            slack = max(self.L0 - self.span, 1e-6)
            sag = self.span * np.sqrt(3.0 * slack / (8.0 * self.span))
            xs = self.s * self.span
            ys = -4.0 * sag * self.s * (1.0 - self.s)
            q = np.empty(2 * self.free.size)
            q[0::2], q[1::2] = xs[self.free], ys[self.free]
        else:
            q = q0.copy()
        # Round-off floor on the gradient. Element forces are k (l - l0) with k = EA/l0 up to
        # 4e5 N/m, so representing a node position in double precision already costs
        # eps * k * span of force per node; asking for less than that is asking for a number the
        # arithmetic cannot produce (043 froze three tolerances below this floor and had to
        # re-register). The criterion is the LOOSER of the physical one and this one, both reported.
        scale = max(float(np.sum(self.W)), 1e-9)
        floor = float(np.finfo(float).eps * (self.EA / self.l0.min()) * self.span
                      * np.sqrt(2.0 * self.free.size))
        self.tol_abs = max(tol * scale, floor)
        self.floor = floor
        self.converged = False
        for it in range(iters):
            U, g, t, l = self.energy_grad(q)
            if float(np.linalg.norm(g)) < self.tol_abs:
                self.converged = True
                break
            K = self.hessian(q)
            lam = 1e-8 * abs(K.diagonal()).max()
            for _ in range(40):                       # Levenberg-style regularisation if needed
                try:
                    d = spsolve((K + lam * identity(K.shape[0], format="csc")).tocsc(), -g)
                except Exception:
                    d = -g
                if np.all(np.isfinite(d)) and float(d @ g) < 0:
                    break
                lam *= 10.0
            # Near the minimum the predicted energy drop (~0.5 g'K^-1 g) falls BELOW the
            # round-off in U itself, so a pure Armijo test can no longer see an improvement and
            # halves the step to nothing. Accept a step that leaves the energy flat to round-off
            # provided it reduces the residual -- that is the quantity actually being driven to 0.
            Ue = 16.0 * np.finfo(float).eps * max(abs(U), 1.0)
            gn = float(np.linalg.norm(g))
            a, taken = 1.0, False
            for _ in range(80):
                U2, g2 = self.energy_grad(q + a * d)[:2]
                if U2 <= U - 1e-4 * a * abs(float(d @ g)):
                    taken = True
                    break
                if U2 <= U + Ue and float(np.linalg.norm(g2)) < gn:
                    taken = True
                    break
                a *= 0.5
            if not taken:
                break                                  # stalled; the check below decides
            q = q + a * d
        self.q = q
        self.U, self.g, self.t, self.l = self.energy_grad(q)
        self.gnorm = float(np.linalg.norm(self.g))
        self.iters = it
        if not self.converged:
            self.converged = self.gnorm < self.tol_abs
        if require and not self.converged:
            raise RuntimeError("line solve did not converge: |grad| = %.3e N after %d iterations "
                               "(tol %.3e = max(%.3e physical, %.3e round-off floor), n = %d)"
                               % (self.gnorm, it + 1, self.tol_abs, tol * scale, self.floor,
                                  self.l0.size))
        return self


    # ---------------------------------------------------------------- results
    def results(self):
        x, y = self._xy(self.q)
        dx, dy = np.diff(x), np.diff(y)
        l = np.hypot(dx, dy)
        H = float(self.t[0] * abs(dx[0]) / l[0])
        V = float(self.t[0] * abs(dy[0]) / l[0])
        strain = float(np.max((l - self.l0) / self.l0))
        return dict(H=H, V=V, T_max=float(self.t.max()), sag=float(-y.min()),
                    strain=strain, W=float(np.sum(self.W)), grad=float(np.linalg.norm(self.g)),
                    tol_abs=float(self.tol_abs), floor=float(self.floor),
                    W_span=float(np.sum(self.W[1:-1])), converged=bool(self.converged),
                    x=x, y=y)


def solve_line(span, L0, EA, w, garments, n=400, ladder=(50, 100, 200, 400, 800, 1600, 3200)):
    """Mesh continuation: solve coarse, interpolate onto the next mesh by MATERIAL coordinate,
    Newton again. Started cold, fine meshes let elements go slack (zero tangent stiffness, so the
    Hessian is singular in those directions) and the iteration stalls -- that is what produced the
    non-converged tables. Each rung starts close enough that no element ever goes slack."""
    prev = None
    for m in [k for k in ladder if k < n] + [n]:
        ln = Line(span, L0, EA, w, garments, n=m)
        q0 = None
        if prev is not None:
            xs = np.interp(ln.s, prev.s, prev._xy(prev.q)[0])
            ys = np.interp(ln.s, prev.s, prev._xy(prev.q)[1])
            q0 = np.empty(2 * ln.free.size)
            q0[0::2], q0[1::2] = xs[ln.free], ys[ln.free]
        prev = ln.solve(q0=q0)
    return prev
