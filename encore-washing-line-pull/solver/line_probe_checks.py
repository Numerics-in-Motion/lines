# -*- coding: utf-8 -*-
"""Controls for the washing-line probe -- the five the design review required BEFORE registration:

    "Require exact force equilibrium, a central-point-load identity, an appropriate uniform-load
     control, discretization convergence, and agreement between reaction and energy derivatives."

Each one is printed with its measured error, and each is run in BOTH directions where that is
possible: a control that only ever passes has not been shown to be able to fail.

C1  global equilibrium      sum of reactions == total weight, and the two H's equal and opposite
C2  central point load      weightless line, one load at midspan: H = P S / (4 d) exactly
C3  uniform load            self-weight only, against the ELASTIC CATENARY closed form (Irvine),
                            which is a different method, not a re-run of this one
C4  discretization          n = 100 .. 1600, H must converge at the expected order
C5  reaction vs energy      H measured from the end element vs dU/dS by central difference
"""
from __future__ import annotations

import numpy as np
from scipy.optimize import brentq

from washing_line import G, Line, rope, solve_line


# --------------------------------------------------------------------------- closed forms
def catenary_H(span, L0, EA, w):
    """Elastic catenary, level ends, weight w per UNSTRETCHED length. Irvine (1981):
        x(L0) = H L0 / EA + (2H/w) asinh(w L0 / 2H) = span."""
    def f(H):
        return H * L0 / EA + (2.0 * H / w) * np.arcsinh(w * L0 / (2.0 * H)) - span
    lo, hi = 1e-6, 1e9
    return brentq(f, lo, hi, xtol=1e-14, rtol=1e-15, maxiter=500)


def catenary_sag(span, L0, EA, w):
    H = catenary_H(span, L0, EA, w)
    V0 = w * L0 / 2.0
    return w * L0 ** 2 / (8.0 * EA) + (np.hypot(H, V0) - H) / w, H


# --------------------------------------------------------------------------- controls
def reactions(ln):
    """End forces from the two end elements, in the sense the posts push on the line."""
    x, y = ln._xy(ln.q)
    dx, dy = np.diff(x), np.diff(y)
    l = np.hypot(dx, dy)
    HL = ln.t[0] * dx[0] / l[0]
    VL = ln.t[0] * dy[0] / l[0]
    HR = -ln.t[-1] * dx[-1] / l[-1]
    VR = -ln.t[-1] * dy[-1] / l[-1]
    return HL, VL, HR, VR


def c1_equilibrium(span, L0, EA, w, garments, n=400):
    """The end elements carry everything EXCEPT the lumped weight sitting on the support nodes
    themselves, which rests directly on the posts. Comparing against the full W left a 4.5e-5
    residual that is exactly W[0] + W[-1]; both numbers are printed so the claim is checkable."""
    ln = solve_line(span, L0, EA, w, garments, n=n)
    HL, VL, HR, VR = reactions(ln)
    W = float(np.sum(ln.W))
    W_line = float(np.sum(ln.W[1:-1]))
    r_v = abs((-VL - VR) - W_line)
    r_h = abs(HL + HR)
    # The reaction sum equals the line weight MINUS the sum of the free nodes' residual forces, so
    # it is a sum over n_free nodes each of which may carry up to the convergence tolerance:
    # |r_v| <= sqrt(n_free) * ||g|| <= sqrt(n_free) * tol. Judging it against tol alone (as a first
    # version did) demands the sum be as small as one term of it.
    nf = ln.free.size
    bound = np.sqrt(nf) * ln.tol_abs
    return dict(r_v=r_v, r_h=r_h, tol=ln.tol_abs, bound=bound, nf=nf,
                e_v=r_v / W_line, e_h=r_h / max(abs(HL), 1e-12),
                H=abs(HL), W=W, W_line=W_line, on_posts=W - W_line)


def c2_point_load(span=6.0, L0=6.02, EA=405030.0, P=50.0, n=400):
    """Weightless line, one load at midspan. H = P S / (4 d) is an identity for ANY
    tension-only equilibrium, so a solver that satisfies it is doing statics correctly."""
    ln = solve_line(span, L0, EA, 0.0, [(0.5, P)], n=n)
    r = ln.results()
    H_id = P * span / (4.0 * r["sag"])
    return dict(H=r["H"], H_id=H_id, rel=abs(r["H"] - H_id) / H_id, sag=r["sag"])


def c3_uniform(span=6.0, L0=6.05, EA=405030.0, w=0.33930, n=400):
    """Run at the REGISTERED mesh (400 base intervals), not at a finer one.

    At n = 1600 this same comparison is 1.11e-4, a THOUSAND times worse than at n = 400 (1.11e-7).
    The round-off floor grows with the mesh (it scales with EA / l0_min and with sqrt(ndof), i.e.
    about n^1.5) while the discretisation error falls as n^-2, so beyond a few hundred elements a
    finer mesh is LESS accurate, not more. A control validates the configuration that ships."""
    ln = solve_line(span, L0, EA, w, [], n=n)
    r = ln.results()
    sag_c, H_c = catenary_sag(span, L0, EA, w)
    return dict(H=r["H"], H_c=H_c, rel_H=abs(r["H"] - H_c) / H_c,
                sag=r["sag"], sag_c=sag_c, rel_sag=abs(r["sag"] - sag_c) / sag_c)


def c4_convergence(span, L0, EA, w, garments, ns=(100, 200, 400, 800, 1600)):
    out = []
    for n in ns:
        r = solve_line(span, L0, EA, w, garments, n=n).results()
        out.append((n, r["H"], r["sag"]))
    return out


def c5_energy(span, L0, EA, w, garments, n=400, d=2e-4):
    """H = dU/dS at equilibrium (Castigliano). Central difference in the span."""
    ln = solve_line(span, L0, EA, w, garments, n=n)
    H = ln.results()["H"]
    Up = solve_line(span + d, L0, EA, w, garments, n=n).U
    Um = solve_line(span - d, L0, EA, w, garments, n=n).U
    H_e = (Up - Um) / (2.0 * d)
    return dict(H=H, H_e=H_e, rel=abs(H - H_e) / abs(H))


def main():
    # Every control runs on the REGISTERED specimen. An earlier version left C1, C4 and C5 on the
    # polypropylene case they had been written against, so the controls were not testing the rope
    # the registration ships.
    R = rope("7x7 3 mm")
    span, EA, w = 6.00, R["EA"], R["w"]
    garments = [(k / 9.0, 0.45 * G) for k in range(1, 9)]
    ok = True
    print("registered specimen: %s  EA %.0f N  w %.5f N/m  SWL %.1f N"
          % (R["name"], R["EA"], R["w"], R["SWL"]))
    print("mesh: n = 400 base intervals + 8 garment nodes = %d elements"
          % Line(span, 6.05, EA, w, garments, n=400).l0.size)

    print("C1  global equilibrium (span 6.00, L0 6.05, 8 garments)")
    r = c1_equilibrium(span, 6.05, EA, w, garments)
    print("    sum V vs line   %.2e N (rel %.1e)   W %.4f N, on posts %.4f N, H = %.2f N"
          % (r["r_v"], r["e_v"], r["W"], r["on_posts"], r["H"]))
    print("    HL + HR         %.2e N (rel %.1e)   solve tolerance %.2e N"
          % (r["r_h"], r["e_h"], r["tol"]))
    print("    bound sqrt(%d) x tol = %.2e N -- both residuals must sit under it" % (r["nf"], r["bound"]))
    ok &= r["r_v"] < r["bound"] and r["r_h"] < r["bound"]
    print("C1b known-BAD: charge the posts' OWN lumped weight to the line (the v1 mistake)")
    bad_v = abs(r["r_v"] - r["on_posts"])
    print("    residual %.2e N vs bound %.2e N -> control %s"
          % (bad_v, r["bound"], "FAILS as it must" if bad_v > r["bound"] else "DID NOT FAIL"))
    ok &= bad_v > r["bound"]

    print("C2  central point load, weightless line -- H = P S / 4d")
    r = c2_point_load(EA=EA)
    print("    solver   %10.4f N" % r["H"])
    print("    identity %10.4f N   rel %.2e   (sag %.4f m)" % (r["H_id"], r["rel"], r["sag"]))
    ok &= r["rel"] < 1e-9

    print("C3  uniform self-weight vs ELASTIC CATENARY closed form (at the REGISTERED mesh)")
    r = c3_uniform(EA=EA, w=w)
    print("    H    solver %10.4f N   catenary %10.4f N   rel %.2e" % (r["H"], r["H_c"], r["rel_H"]))
    print("    sag  solver %10.6f m   catenary %10.6f m   rel %.2e" % (r["sag"], r["sag_c"], r["rel_sag"]))
    ok &= r["rel_H"] < 2e-5 and r["rel_sag"] < 2e-5

    print("C3b known-BAD: the same comparison with the lumping deliberately halved")
    ln = Line(6.0, 6.05, EA, w, [], n=400)
    ln.W *= 0.5
    ln.solve()
    _, H_c = catenary_sag(6.0, 6.05, EA, w)
    bad = abs(ln.results()["H"] - H_c) / H_c
    print("    rel %.2e  -> control %s" % (bad, "FAILS as it must" if bad > 2e-5 else "DID NOT FAIL"))
    ok &= bad > 2e-5

    print("C4  discretization convergence (L0 6.05, 8 garments)")
    rows = c4_convergence(span, 6.05, EA, w, garments)
    Href = rows[-1][1]
    for (n, H, sag) in rows:
        print("    n %5d   H %10.5f N   sag %8.6f m   rel to finest %.2e"
              % (n, H, sag, abs(H - Href) / Href))
    r400 = [h for (n, h, _) in rows if n == 400][0]
    ok &= abs(r400 - Href) / Href < 1e-3

    print("C4b discretization ORDER on the curved case (uniform load, vs catenary)")
    print("     -- C4 is flat to 3e-8 because with point loads the exact shape is piecewise")
    print("        straight, so any mesh holding the load nodes is exact; it tests nothing.")
    _, H_c = catenary_sag(6.0, 6.05, EA, w)
    es = []
    for n in (25, 50, 100, 200, 400):
        H = solve_line(6.0, 6.05, EA, w, [], n=n).results()["H"]
        es.append((n, abs(H - H_c) / H_c, H))
    # An order can only be measured where the DISCRETISATION error dominates the solver's own
    # precision. With this rope the n=400 error is 1.1e-7, at the solver floor, and its ratio comes
    # out 104 -- the error passing through zero, not a convergence rate. Pairs are used only where
    # both errors exceed 1e-6, ~50x the mesh-to-mesh spread seen in C4.
    FLOOR = 1e-6
    ratios = []
    for i in range(1, len(es)):
        (n0, e0, _), (n1, e1, H1) = es[i - 1], es[i]
        tag = ""
        if e0 > FLOOR and e1 > FLOOR:
            ratios.append(e0 / e1)
            tag = "   ratio %5.2f" % (e0 / e1)
        elif i:
            tag = "   (at the solver floor -- no order measurable)"
        print("    n %5d   H %10.6f N   rel %.3e%s" % (n1, H1, e1, tag))
    print("    expected 4.00 for second order; measured " +
          ", ".join("%.2f" % r for r in ratios))
    ok &= all(3.5 < r < 4.5 for r in ratios) and len(ratios) >= 2

    print("C5  reaction vs energy derivative dU/dS")
    r = c5_energy(span, 6.05, EA, w, garments)
    print("    reaction %10.5f N   dU/dS %10.5f N   rel %.2e" % (r["H"], r["H_e"], r["rel"]))
    ok &= r["rel"] < 1e-4

    print("C6  uniqueness: independent starting shapes must reach the SAME equilibrium")
    R = rope("7x7 3 mm")
    Hs = []
    for (tag, shape) in (("parabola x0.5", 0.5), ("parabola x1", 1.0), ("parabola x2", 2.0),
                         ("triangle", "tri"), ("cosine", "cos")):
        ln = Line(span, 6.000, R["EA"], R["w"], garments, n=400)
        sl = max(6.000 - span, 1e-6)
        d0 = span * np.sqrt(3.0 * sl / (8.0 * span)) + 0.10
        u = ln.s
        if shape == "tri":
            ys = -d0 * (1.0 - np.abs(2.0 * u - 1.0))
        elif shape == "cos":
            ys = -d0 * np.sin(np.pi * u)
        else:
            ys = -4.0 * d0 * shape * u * (1.0 - u)
        q0 = np.empty(2 * ln.free.size)
        q0[0::2], q0[1::2] = (u * span)[ln.free], ys[ln.free]
        H = ln.solve(q0=q0).results()["H"]
        Hs.append(H)
        print("    %-15s H %12.7f N" % (tag, H))
    spread = max(Hs) - min(Hs)
    print("    spread over 5 independent starts: %.2e N (rel %.1e)" % (spread, spread / np.mean(Hs)))

    print("C7  numerical uncertainty u for the REGISTERED specimen (7x7 3 mm, tightest cell)")
    r400 = solve_line(span, 6.000, R["EA"], R["w"], garments, n=400).results()
    r1600 = solve_line(span, 6.000, R["EA"], R["w"], garments, n=1600).results()
    mesh = abs(r400["H"] - r1600["H"])
    print("    residual floor  n=400  %.3e N     n=1600 %.3e N" % (r400["floor"], r1600["floor"]))
    print("    mesh difference |H400 - H1600|  %.3e N  (rel %.2e)" % (mesh, mesh / r400["H"]))
    print("    start spread                    %.3e N" % spread)
    u_reg = max(r400["floor"], r1600["floor"], mesh, spread)
    print("    -> u must be frozen at >= %.3e N ; 5u = %.3e N" % (u_reg, 5 * u_reg))
    Hsweep = [solve_line(span, L0, R["EA"], R["w"], garments, n=400).results()["H"]
              for L0 in (6.300, 6.200, 6.120, 6.080, 6.050, 6.030, 6.010, 6.000)]
    gaps = np.diff(Hsweep)
    adj = float(gaps.min())
    print("    registered sweep H: " + " ".join("%.1f" % h for h in Hsweep))
    print("    SMALLEST adjacent gap %.2f N  -> clears 5u by %.0fx" % (adj, adj / (5 * u_reg)))
    # the gate is the registration's own separation rule, not a number invented here
    ok &= 5 * u_reg < adj

    print("C8  EVERY registered cell against EVERY per-cell gate (not just the tightest)")
    print("    %-8s %-10s %-9s %-9s %-9s %-11s %-11s %s"
          % ("L0 (m)", "H (N)", "sag (m)", "T_max", "strain %", "min T (N)", "mesh rel", "gates"))
    cells = (6.300, 6.200, 6.120, 6.080, 6.050, 6.030, 6.010, 6.000)
    allpass = True
    for L0 in cells:
        a = solve_line(span, L0, EA, w, garments, n=400)
        b = solve_line(span, L0, EA, w, garments, n=1600)
        ra, rb = a.results(), b.results()
        mesh = abs(ra["H"] - rb["H"]) / ra["H"]
        minT = float(a.t.min())
        g = []
        if not ra["converged"]:
            g.append("NOT CONVERGED")
        if minT <= 0.0:
            g.append("SLACK SEGMENT")
        if ra["T_max"] > R["SWL"]:
            g.append("OVER SWL")
        if mesh >= 1e-3:
            g.append("MESH")
        allpass &= not g
        print("    %-8.3f %-10.4f %-9.4f %-9.1f %-9.4f %-11.3f %-11.2e %s"
              % (L0, ra["H"], ra["sag"], ra["T_max"], 100 * ra["strain"], minT, mesh,
                 ",".join(g) if g else "pass"))
    print("    all eight cells pass every per-cell gate: %s" % ("yes" if allpass else "NO"))
    ok &= allpass

    print()
    print("ALL CONTROLS PASS" if ok else "*** A CONTROL FAILED ***")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
