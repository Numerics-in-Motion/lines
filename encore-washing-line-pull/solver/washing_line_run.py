# -*- coding: utf-8 -*-
"""044 production run: solve the eight registered cells and apply the registered ship/kill line.

    python src/washing_line_run.py          # -> <data>/cells.json

Every gate in `analyze/260906/044_registration.md` §4 is re-applied here to every cell, not just
the tightest one. A cell that fails any gate kills the registered sweep; no tighter case is
substituted afterwards.
"""
from __future__ import annotations

import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

import numpy as np                                            # noqa: E402

import washing_line as WL                                     # noqa: E402

# No internal numbering in anything that ships: the published package is read by people
# who have never seen this repository's folder names.
DATA = os.environ.get("LINE_DATA") or os.path.abspath("line_data")


def run():
    R = WL.rope()
    cells = []
    for L0 in WL.L0S:
        a = WL.solve_line(WL.SPAN, L0, R["EA"], R["w"], WL.GARMENTS, n=WL.N_BASE)
        b = WL.solve_line(WL.SPAN, L0, R["EA"], R["w"], WL.GARMENTS, n=4 * WL.N_BASE)
        ra, rb = a.results(), b.results()
        mesh = abs(ra["H"] - rb["H"]) / ra["H"]
        fail = []
        if not ra["converged"]:
            fail.append("not converged")
        if float(a.t.min()) <= 0.0:
            fail.append("slack segment")
        if ra["T_max"] > R["SWL"]:
            fail.append("over SWL")
        if mesh >= WL.MESH_TOL:
            fail.append("mesh")
        cells.append(dict(L0=L0, H=ra["H"], sag=ra["sag"], T_max=ra["T_max"], V=ra["V"],
                          strain=ra["strain"], min_T=float(a.t.min()), mesh=mesh,
                          HW=ra["H"] / WL.W_WASHING, elements=int(a.l0.size),
                          x=list(map(float, ra["x"])), y=list(map(float, ra["y"])),
                          fail=fail))

    Hs = [c["H"] for c in cells]
    gaps = np.diff(Hs)
    sep_ok = bool(gaps.min() >= WL.SEP_MULT * WL.U_REG)
    tight = cells[-1]
    ship = (not any(c["fail"] for c in cells)) and sep_ok and tight["HW"] >= WL.H_W_SHIP

    out = dict(rope={k: (v if not isinstance(v, float) else float(v))
                     for k, v in R.items()},
               span=WL.SPAN, W=WL.W_WASHING, u=WL.U_REG, h_w_ship=WL.H_W_SHIP,
               min_gap=float(gaps.min()), sep_ok=sep_ok, ship=bool(ship), cells=cells)
    os.makedirs(DATA, exist_ok=True)
    with open(os.path.join(DATA, "cells.json"), "w", encoding="utf-8") as f:
        json.dump(out, f, indent=1)
    return out


def main():
    out = run()
    R = out["rope"]
    print("Registered sweep: how hard a washing line pulls on its posts")
    print("%s  EA %.0f N  published SWL %.1f N  span %.3f m  washing %.3f N"
          % (R["name"], R["EA"], R["SWL"], out["span"], out["W"]))
    print("%-8s %-9s %-10s %-8s %-9s %-9s %-10s %s"
          % ("L0 (m)", "sag (m)", "H (N)", "H/W", "T_max", "strain %", "mesh rel", "gates"))
    for c in out["cells"]:
        print("%-8.3f %-9.4f %-10.4f %-8.2f %-9.1f %-9.4f %-10.2e %s"
              % (c["L0"], c["sag"], c["H"], c["HW"], c["T_max"], 100 * c["strain"], c["mesh"],
                 ",".join(c["fail"]) if c["fail"] else "pass"))
    print("smallest adjacent gap %.3f N vs 5u = %.3e N -> %s"
          % (out["min_gap"], WL.SEP_MULT * out["u"], "ok" if out["sep_ok"] else "FAIL"))
    print("tightest cell H/W = %.3f vs registered ship line %.1f -> %s"
          % (out["cells"][-1]["HW"], out["h_w_ship"], "SHIPS" if out["ship"] else "KILLED"))
    print("elements per solve: %d" % out["cells"][0]["elements"])
    return 0 if out["ship"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
