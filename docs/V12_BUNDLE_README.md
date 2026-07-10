# The Witness Machine — V12 release bundle

This archive is a compact, public, non-score-bearing release artifact. It
contains the runnable marimo notebook, the Python modules reachable from that
notebook, compact locked evidence, and custody metadata. It does not contain
the comma.ai source video, frozen scorer weights, private run directories, or
an exact score claim.

## Run the notebook

From the extracted `witness_machine_v12` directory:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e .
marimo run notebooks/witness_machine_v12.py --headless
```

For a finite execution smoke instead of a live server:

```bash
marimo export html notebooks/witness_machine_v12.py \
  --no-include-code --force -o /tmp/witness-machine-smoke.html
```

`RELEASE_MANIFEST.json` records every shipped source byte. The fuller research
archive additionally contains the manuscript, locked figures, review packet,
and reproducibility reports. Rebuilding the release archives and running the
repository-wide gate suite require the source repository; those commands are
deliberately not advertised as archive-local operations.

## Authority boundary

Public claims use `TOY`, `ADVISORY`, `EMPIRICAL`, `DERIVATION`, `EXTERNAL`,
`EXACT-CANDIDATE`, or `EXACT`. No `EXACT` score is available without an exact
score card tied to exact archive bytes and an unmodified upstream evaluator
transcript. Public submit, push, upload, PR, publication, or payment remains
stopped until explicit operator GO; login or credentials remain an operator
gate.

MIT licensed. Built in public by `adpena`.
