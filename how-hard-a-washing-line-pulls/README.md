# How hard a washing line pulls on its posts

One rope between two posts, one load of washing. Take the slack out in registered steps and
measure the horizontal pull **H** at the post.

**For this fixed-end model of this rope and these loads, shortening the unstretched line from
6.300 m to 6.000 m changed the horizontal end reaction from 36.4 N to 307.2 N** -- 8.70 times the
weight of the washing itself.

*This does not rate posts, anchors, knots, clips, or real clothesline installations.*

## The specimen

Not assumed -- read from S3i Group stainless wire-rope tables, read 2026-09-17:

| | |
|---|---|
| rope | 7x7 3 mm |
| rope modulus | 57.3 kN/mm² (7x7; the table also gives 1x19 107.5, 7x19 47.5) |
| EA | 405030 N — modulus × **nominal** area πD²/4, as the supplier's own stretch formula defines it |
| minimum breaking load | 5030 N |
| published SWL | 833.6 N (85 kg) — the supplier's column, not MBL/6 recomputed |
| mass | 0.0346 kg/m |

The rope modulus is a **construction** constant, not the 200 GPa of the wire material.

## The scenario

Span 6.000 m, both ends fixed and no-slip at equal elevation. Eight garments of 0.45 kg
(35.304 N total) at fixed material coordinates k/9. No pretension: every registered length is at
least the span. 408 elements.

## Results

| L₀ (m) | sag (m) | H (N) | H / W | T_max (N) | strain (%) |
|---|---|---|---|---|---|
| 6.300 | 0.8312 | 36.43 | 1.03 | 41.0 | 0.0101 |
| 6.200 | 0.6756 | 45.14 | 1.28 | 48.9 | 0.0121 |
| 6.120 | 0.5221 | 58.75 | 1.66 | 61.6 | 0.0152 |
| 6.080 | 0.4267 | 72.10 | 2.04 | 74.5 | 0.0184 |
| 6.050 | 0.3390 | 90.93 | 2.58 | 92.8 | 0.0229 |
| 6.030 | 0.2662 | 115.98 | 3.29 | 117.5 | 0.0290 |
| 6.010 | 0.1684 | 183.58 | 5.20 | 184.5 | 0.0456 |
| 6.000 | 0.1007 | 307.23 | 8.70 | 307.8 | 0.0760 |

Peak rope tension 307.8 N is 37 % of the published SWL: the ceiling here is the rope's stretch,
not its strength.

## Reproduce

```
python solver/washing_line_run.py        # the eight registered cells
python solver/line_probe_checks.py       # C1-C8, the controls
```

`reference/reference_canonical.json` holds the canonical values. SHA-256 in that file is over
LF-normalised bytes.

## Why the controls ship with the solver

The first solver written for this study returned H = 0 N for every case. The second returned a
full table of plausible numbers taken from **non-converged** states -- the residual reached 275 N
against a 36 N load -- because it returned whatever the iteration happened to hold. Neither was
caught by reading the code. Both were caught by the controls, which check global equilibrium, a
central-point-load identity, the elastic catenary closed form, convergence order, the reaction
against dU/dS, uniqueness from five independent starting shapes, and every registered cell against
every gate. `solve()` now raises rather than return a state it has not converged.
