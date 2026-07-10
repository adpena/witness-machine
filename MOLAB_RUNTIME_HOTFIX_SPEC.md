# Molab mirror runtime hotfix

## Objective

Make the existing public Molab URL

`https://molab.marimo.io/github/adpena/witness-machine/blob/main/notebooks/witness_machine_v12.py/server`

run from a fresh Molab server sandbox when the notebook is mirrored from GitHub and executed as a synthetic `notebook.py`.

## Confirmed failure

The first cell derives `project_root` from synthetic `__file__`, inserts a nonexistent sibling `src/`, and then raises:

`ModuleNotFoundError: No module named 'molab_witness_machine'`

Later cells also require the locked files under `artifacts/v12_public`, so installing only the Python package is insufficient.

## Required repair

1. Keep the normal local-repository fast path when a candidate root contains both:
   - `src/molab_witness_machine`
   - `artifacts/v12_public`
2. Otherwise bootstrap only from the immutable RC2 Molab release bundle:
   - URL: `https://github.com/adpena/witness-machine/releases/download/v1.2.0-rc2/witness_machine_v12_molab_bundle.zip`
   - expected bytes: `3704001`
   - SHA-256: `baf3e1e50b21ec229dcdc62ddde1451d42a50833ffa1c5ffc98778430edd2439`
3. Use stdlib only for bootstrap. Download with a bounded timeout, verify exact size and SHA-256 before extraction, reject absolute paths and `..` traversal, require the single expected top-level directory `witness_machine_v12`, extract to a temporary directory, and atomically promote into a SHA-keyed writable cache.
4. Validate a cache before reuse with a marker containing the bundle SHA and with the required package/assets present. A corrupt or incomplete cache must not be trusted.
5. Set `project_root` to the validated local or extracted bundle root and insert its `src` into `sys.path` before loading the notebook helpers.
6. Avoid marimo's static missing-package prompt for `molab_witness_machine` if that prompt is caused by literal import analysis. Bind the helper symbols through `importlib.import_module` after bootstrap while preserving the same names used by all descendant cells.
7. Fail closed with a concise actionable exception containing the immutable URL and expected/actual digest or size. Never fall back to mutable `main` or unverified bytes.

## Scope constraints

- Do not change the notebook's mathematics, claims, evidence, copy, visual design, controls, or interaction behavior.
- Do not alter the RC2 tag or existing release asset.
- Do not edit the upstream comma.ai evaluator or modules.
- Do not commit, tag, push, publish, upload, or modify GitHub state. The parent agent owns review and public mutation.
- Keep the patch minimal and readable.
- Add focused regression coverage in this compact public repository if practical. Tests must simulate a synthetic notebook path/no sibling repo and exercise at least cache validation, hash rejection, traversal rejection, and required asset presence without relying on the mutable network in every test.
- Do not add a generated marimo session file unless it can be produced by a clean successful run and is demonstrably required for the server URL; runtime correctness is P0.

## Acceptance gates

- `marimo check notebooks/witness_machine_v12.py` passes.
- A fresh isolated execution with no repository `src` on `sys.path` can acquire the sealed bundle, load every helper module, and locate every required `artifacts/v12_public` file.
- A second isolated execution succeeds from the validated cache with the network disabled.
- Tampered bytes and unsafe ZIP members fail closed.
- Existing focused tests pass.
- The parent agent will then verify the exact public `/server` URL in an authenticated fresh Molab kernel, confirm zero runtime errors, exercise an interaction, and leave it live.

## Concrete implementation shape

Act on this shape directly; do not spend another pass surveying unrelated helper modules.

1. Keep the bootstrap inside the second marimo cell so a synthetic one-file mirror is sufficient. Define small underscore-prefixed helpers there for root validation, safe ZIP member validation, sealed download/extraction, and root resolution.
2. Root validation should require at least all of:
   - `src/molab_witness_machine/v12_copy.py`
   - `src/molab_witness_machine/v12_gpu.py`
   - `artifacts/v12_public/v12_temporal_transport_display.svg`
   - `artifacts/v12_public/v12_temporal_transport_display.manifest.json`
   - `artifacts/v12_public/v12_real_frozen_scorer_display.npz`
   - `artifacts/v12_public/v12_real_frozen_scorer_display.manifest.json`
   - `artifacts/v12_public/v12_real_frozen_scorer_evidence.npz`
   - `artifacts/v12_public/v12_real_frozen_scorer_evidence.manifest.json`
3. Search a small deduplicated set of local candidates derived from resolved `__file__` and `Path.cwd()`, walking their parents. If none validate, materialize the sealed bundle.
4. Use a cache slot such as `${XDG_CACHE_HOME:-~/.cache}/witness-machine/<full-sha>/`. Put a marker containing the full SHA beside the extracted `witness_machine_v12/` root. Trust only a matching marker plus a fully validated root. Build in `tempfile.mkdtemp(dir=cache_parent)`, write the marker before atomic promotion, and handle a concurrent winner by validating the winner. Removing an invalid cache slot is permitted because it is explicitly rebuildable cache data.
5. Read at most `expected_bytes + 1` from `urllib.request.urlopen(..., timeout=30)`. Reject incorrect byte count before digest verification. Reject an incorrect digest before opening the ZIP.
6. Before `extractall`, validate every `ZipInfo.filename` with `PurePosixPath`: no absolute path, no empty/dot/dot-dot component, and the only top-level component must be `witness_machine_v12`. The pinned digest is primary custody, but retain this structural guard and test it.
7. After inserting the verified `src` at `sys.path[0]`, load notebook helper modules with `importlib.import_module` and bind the same public symbols currently returned by the cell. Do not leave any literal `from molab_witness_machine` or `import molab_witness_machine` statement anywhere in the notebook, including the lazy GPU worker near the end; use dynamic import there as well.
8. Tests may parse the notebook AST, extract the second cell body before the first `import numpy`, and execute the underscore-prefixed bootstrap helpers in a controlled namespace. Prefer this over creating a separate helper module that the synthetic mirror could not import.
