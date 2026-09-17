# -*- coding: utf-8 -*-
"""Re-solve the registered washing-line cases and check every published number.

    pip install -r ../requirements.txt
    python reproduce.py               # the eight registered cells + the controls
    python reproduce.py --quick       # the eight registered cells only

Exits non-zero if any published value has moved, if any registered gate fails, or if the SHA-256
of a solver module no longer matches the frozen reference. Nothing here is fitted to the answer:
the reference was written from the canonical run before this file existed.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "solver"))

REF = os.path.join(HERE, "reference", "reference_canonical.json")


def load_reference():
    with open(REF, encoding="utf-8") as f:
        return json.load(f)


def check_sha(ref):
    """SHA-256 over LF-normalised bytes, so a CRLF checkout still matches."""
    out = []
    for name, want in sorted(ref["source_sha256"].items()):
        p = os.path.join(HERE, "solver", name + ".py")
        with open(p, "rb") as f:
            got = hashlib.sha256(f.read().replace(b"\r\n", b"\n")).hexdigest()
        out.append(dict(module=name, ok=(got == want), got=got, want=want))
    return out


def solve_registered():
    import washing_line as WL
    rope = WL.rope()
    rows = []
    for L0 in WL.L0S:
        r = WL.solve_line(WL.SPAN, L0, rope["EA"], rope["w"], WL.GARMENTS,
                          n=WL.N_BASE).results()
        rows.append(dict(L0=L0, H=r["H"], sag=r["sag"], T_max=r["T_max"],
                         HW=r["H"] / WL.W_WASHING, strain=r["strain"]))
    return rope, rows


def compare(ref, rows):
    tol = ref["tolerances"]
    bad = []
    for row in rows:
        want = ref["primary"]["%.3f" % row["L0"]]
        if abs(row["H"] - want["H"]) > tol["H_abs_N"]:
            bad.append("L0 %.3f: H %.6f, reference %.6f" % (row["L0"], row["H"], want["H"]))
        if abs(row["sag"] - want["sag"]) > tol["sag_abs_m"]:
            bad.append("L0 %.3f: sag %.8f, reference %.8f" % (row["L0"], row["sag"], want["sag"]))
    return bad


def check_gates(ref, rope, rows):
    """The registered ship/kill line, re-applied."""
    import washing_line as WL
    g = []
    g.append(("every cell in tension", all(r["T_max"] > 0 for r in rows)))
    g.append(("no cell over the published rope SWL",
              all(r["T_max"] <= rope["SWL"] for r in rows)))
    g.append(("H rises at every step", all(rows[i + 1]["H"] > rows[i]["H"]
                                           for i in range(len(rows) - 1))))
    g.append(("tightest cell clears the declared H/W >= %.1f" % WL.H_W_SHIP,
              rows[-1]["HW"] >= WL.H_W_SHIP))
    return g


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true",
                    help="skip the controls (C1-C8), which take a few minutes")
    a = ap.parse_args(argv)

    ref = load_reference()
    print("How hard a washing line pulls on its posts")
    print("registration: %s" % ref["registration"])
    print()

    print("solver SHA-256 (LF-normalised)")
    shas = check_sha(ref)
    for s in shas:
        print("   %-20s %s %s" % (s["module"], "ok " if s["ok"] else "MOVED", s["got"][:16]))
    print()

    rope, rows = solve_registered()
    print("specimen: %s   EA %.0f N   published SWL %.1f N" % (rope["name"], rope["EA"],
                                                              rope["SWL"]))
    print("%-9s %-10s %-9s %-8s %-9s %s" % ("L0 (m)", "H (N)", "sag (m)", "H/W", "T_max", "strain %"))
    for r in rows:
        print("%-9.3f %-10.4f %-9.4f %-8.2f %-9.1f %.4f"
              % (r["L0"], r["H"], r["sag"], r["HW"], r["T_max"], 100 * r["strain"]))
    print()

    bad = compare(ref, rows)
    gates = check_gates(ref, rope, rows)
    for name, ok in gates:
        print("   %-48s %s" % (name, "ok" if ok else "FAILED"))
    print()

    ok = all(s["ok"] for s in shas) and not bad and all(o for _, o in gates)
    for b in bad:
        print("MOVED: %s" % b)

    if not a.quick:
        print("controls C1-C8 ...")
        env = dict(os.environ)
        env["PYTHONPATH"] = os.path.join(HERE, "solver")
        env["PYTHONIOENCODING"] = "utf-8"
        r = subprocess.run([sys.executable,
                            os.path.join(HERE, "solver", "line_probe_checks.py")],
                           capture_output=True, text=True, env=env)
        sys.stdout.write(r.stdout)
        ok = ok and r.returncode == 0 and "ALL CONTROLS PASS" in r.stdout

    print()
    print("EVERYTHING REPRODUCES" if ok else "*** SOMETHING MOVED ***")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
