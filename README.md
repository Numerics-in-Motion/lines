# lines

Ropes and cables between fixed points, solved from the equations and published with everything
needed to run them again.

| study | question |
|---|---|
| [`how-hard-a-washing-line-pulls/`](how-hard-a-washing-line-pulls/) | Eight garments on a six metre line: as the slack comes out, how large does the horizontal pull on the posts get compared with the weight of the washing itself? |

Each study folder holds its solver, a frozen reference of every reported number with the SHA-256
of the modules that produced it, and a `reproduce.py` that re-solves the registered cases and
stops with an error if anything has moved.

```
pip install -r requirements.txt
cd how-hard-a-washing-line-pulls
python reproduce.py --quick
```

The controls ship with the solver, and they are the reason the numbers are believable: the first
solver written for this rig returned zero horizontal pull for every case, and the second returned
a full table of plausible values taken from states it had never converged. Neither was caught by
reading the code.
