from __future__ import annotations

import ast
from contextlib import contextmanager
import hashlib
import importlib
from pathlib import Path
import sys
import zipfile

import pytest


ROOT = Path(__file__).resolve().parents[1]
NOTEBOOK = ROOT / "notebooks" / "witness_machine_v12.py"
SESSION = ROOT / "notebooks" / "__marimo__" / "session" / "witness_machine_v12.py.json"
PACKAGE_DIR = ROOT / "src" / "molab_witness_machine"
ASSET_DIR = ROOT / "artifacts" / "v12_public"
HELPER_MODULES = (
    "molab_witness_machine.v12_copy",
    "molab_witness_machine.v12_control",
    "molab_witness_machine.v12_geometry",
    "molab_witness_machine.v12_visuals",
    "molab_witness_machine.v12_stac",
    "molab_witness_machine.v12_policy_compare",
    "molab_witness_machine.v12_real_evidence",
    "molab_witness_machine.v12_temporal",
    "molab_witness_machine.v12_gpu",
)
REQUIRED_PACKAGE_FILES = (
    "__init__.py",
    "score_law.py",
    "v12_copy.py",
    "v12_control.py",
    "v12_geometry.py",
    "v12_gpu.py",
    "v12_policy_compare.py",
    "v12_real_evidence.py",
    "v12_real_frames.py",
    "v12_stac.py",
    "v12_temporal.py",
    "v12_visuals.py",
)
REQUIRED_ASSETS = (
    "v12_temporal_transport_display.svg",
    "v12_temporal_transport_display.manifest.json",
    "v12_real_frozen_scorer_display.npz",
    "v12_real_frozen_scorer_display.manifest.json",
    "v12_real_frozen_scorer_evidence.npz",
    "v12_real_frozen_scorer_evidence.manifest.json",
    "v12_real_frame_display.npz",
    "v12_real_frame_display.manifest.json",
    "v12_boundary_evolution.npz",
    "v12_boundary_evolution.manifest.json",
)


def _is_app_cell_decorator(decorator: ast.expr) -> bool:
    """Match both ``@app.cell`` and ``@app.cell(hide_code=True)`` forms."""
    if isinstance(decorator, ast.Call):
        decorator = decorator.func
    return (
        isinstance(decorator, ast.Attribute)
        and isinstance(decorator.value, ast.Name)
        and decorator.value.id == "app"
        and decorator.attr == "cell"
    )


def _bootstrap_namespace() -> dict[str, object]:
    tree = ast.parse(NOTEBOOK.read_text(encoding="utf-8"), filename=str(NOTEBOOK))
    app_cells = [
        node
        for node in tree.body
        if isinstance(node, ast.FunctionDef)
        and any(
            _is_app_cell_decorator(decorator)
            for decorator in node.decorator_list
        )
    ]
    bootstrap_cell = next(
        cell
        for cell in app_cells
        if any(
            isinstance(statement, ast.Assign)
            and any(
                isinstance(target, ast.Name) and target.id == "BUNDLE_URL"
                for target in statement.targets
            )
            for statement in cell.body
        )
    )
    selected_body: list[ast.stmt] = []
    for statement in bootstrap_cell.body:
        if isinstance(statement, ast.Assign):
            target_names = {
                target.id for target in statement.targets if isinstance(target, ast.Name)
            }
            if "project_root" in target_names:
                break
        if isinstance(statement, ast.Import):
            if any(alias.name == "numpy" for alias in statement.names):
                break
        selected_body.append(statement)
    namespace: dict[str, object] = {}
    module = ast.Module(body=selected_body, type_ignores=[])
    ast.fix_missing_locations(module)
    exec(compile(module, str(NOTEBOOK), "exec"), namespace)
    return namespace


def _write_bundle(
    destination: Path,
    *,
    omitted_paths: set[str] | None = None,
    extra_members: tuple[tuple[str, bytes], ...] = (),
) -> Path:
    omitted = omitted_paths or set()
    with zipfile.ZipFile(destination, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for package_name in REQUIRED_PACKAGE_FILES:
            source = PACKAGE_DIR / package_name
            relative = source.relative_to(ROOT).as_posix()
            if relative in omitted:
                continue
            archive.write(source, arcname=f"witness_machine_v12/{relative}")
        for asset_name in REQUIRED_ASSETS:
            relative = (ASSET_DIR / asset_name).relative_to(ROOT).as_posix()
            if relative in omitted:
                continue
            archive.write(ASSET_DIR / asset_name, arcname=f"witness_machine_v12/{relative}")
        for member_name, payload in extra_members:
            archive.writestr(member_name, payload)
    return destination


def _configure_bundle(namespace: dict[str, object], bundle_path: Path) -> None:
    payload = bundle_path.read_bytes()
    namespace["BUNDLE_URL"] = f"test://{bundle_path.name}"
    namespace["BUNDLE_BYTES"] = len(payload)
    namespace["BUNDLE_SHA256"] = hashlib.sha256(payload).hexdigest()


def _bundle_urlopen(bundle_path: Path, calls: list[str]):
    def _opener(url: str, timeout: int = 30):
        calls.append(f"{url}|{timeout}")
        return bundle_path.open("rb")

    return _opener


def _seed_invalid_cache_slot(namespace: dict[str, object], cache_slot: Path) -> None:
    bundle_root = cache_slot / "witness_machine_v12"
    bundle_root.mkdir(parents=True, exist_ok=True)
    (cache_slot / namespace["BUNDLE_MARKER"]).write_text(
        f"{namespace['BUNDLE_SHA256']}\n",
        encoding="utf-8",
    )
    relative = "src/molab_witness_machine/v12_copy.py"
    target = bundle_root / relative
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text((ROOT / relative).read_text(encoding="utf-8"), encoding="utf-8")


def _assert_no_static_molab_imports() -> None:
    tree = ast.parse(NOTEBOOK.read_text(encoding="utf-8"), filename=str(NOTEBOOK))
    offenders: list[str] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                if alias.name == "molab_witness_machine" or alias.name.startswith(
                    "molab_witness_machine."
                ):
                    offenders.append(alias.name)
        elif isinstance(node, ast.ImportFrom):
            module_name = node.module or ""
            if module_name == "molab_witness_machine" or module_name.startswith(
                "molab_witness_machine."
            ):
                offenders.append(module_name)
    assert offenders == []


@contextmanager
def _isolated_bundle_imports(bundle_root: Path):
    original_sys_path = sys.path[:]
    original_modules = {
        name: module
        for name, module in sys.modules.items()
        if name == "molab_witness_machine" or name.startswith("molab_witness_machine.")
    }
    for name in list(original_modules):
        sys.modules.pop(name, None)
    blocked_paths = {ROOT.resolve(), (ROOT / "src").resolve()}
    filtered_sys_path: list[str] = []
    for entry in original_sys_path:
        try:
            resolved = Path(entry or ".").resolve()
        except OSError:
            filtered_sys_path.append(entry)
            continue
        if resolved not in blocked_paths:
            filtered_sys_path.append(entry)
    sys.path[:] = [str(bundle_root / "src"), *filtered_sys_path]
    try:
        yield
    finally:
        for name in list(sys.modules):
            if name == "molab_witness_machine" or name.startswith("molab_witness_machine."):
                sys.modules.pop(name, None)
        sys.modules.update(original_modules)
        sys.path[:] = original_sys_path


def _assert_bundle_runtime_loads(bundle_root: Path) -> None:
    evidence_root = bundle_root / "artifacts" / "v12_public"
    with _isolated_bundle_imports(bundle_root):
        imported_modules = {
            name: importlib.import_module(name)
            for name in HELPER_MODULES
        }
        temporal_module = imported_modules["molab_witness_machine.v12_temporal"]
        display = temporal_module.load_temporal_display(
            evidence_root / "v12_temporal_transport_display.svg",
            evidence_root / "v12_temporal_transport_display.manifest.json",
        )
        assert display.svg.lstrip().startswith("<svg")
        real_evidence_module = imported_modules["molab_witness_machine.v12_real_evidence"]
        derivative = real_evidence_module.load_display_derivative(
            evidence_root / "v12_real_frozen_scorer_display.npz",
            evidence_root / "v12_real_frozen_scorer_display.manifest.json",
            parent_artifact_path=evidence_root / "v12_real_frozen_scorer_evidence.npz",
            parent_manifest_path=evidence_root
            / "v12_real_frozen_scorer_evidence.manifest.json",
        )
        assert derivative.arrays


def test_bootstrap_rebuilds_invalid_cache_and_reuses_validated_cache(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    namespace = _bootstrap_namespace()
    bundle_path = _write_bundle(tmp_path / "bundle.zip")
    _configure_bundle(namespace, bundle_path)
    sandbox = tmp_path / "sandbox"
    sandbox.mkdir()
    monkeypatch.chdir(sandbox)
    monkeypatch.setenv("XDG_CACHE_HOME", str(tmp_path / "xdg-cache"))
    cache_slot = Path(namespace["_cache_slot"]())
    _seed_invalid_cache_slot(namespace, cache_slot)
    calls: list[str] = []
    project_root = namespace["_resolve_project_root"](
        sandbox / "notebook.py",
        urlopen=_bundle_urlopen(bundle_path, calls),
    )
    assert project_root == cache_slot / "witness_machine_v12"
    assert len(calls) == 1
    assert Path(project_root, "artifacts/v12_public/v12_real_frozen_scorer_evidence.npz").is_file()
    _assert_bundle_runtime_loads(Path(project_root))

    def _network_disabled(url: str, timeout: int = 30):
        raise AssertionError(f"unexpected network access for {url} timeout={timeout}")

    cached_project_root = namespace["_resolve_project_root"](
        sandbox / "notebook.py",
        urlopen=_network_disabled,
    )
    assert cached_project_root == project_root


def test_bootstrap_redownloads_when_marked_cache_loses_required_module(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    namespace = _bootstrap_namespace()
    bundle_path = _write_bundle(tmp_path / "bundle.zip")
    _configure_bundle(namespace, bundle_path)
    sandbox = tmp_path / "sandbox"
    sandbox.mkdir()
    monkeypatch.chdir(sandbox)
    monkeypatch.setenv("XDG_CACHE_HOME", str(tmp_path / "xdg-cache"))
    calls: list[str] = []
    project_root = namespace["_resolve_project_root"](
        sandbox / "notebook.py",
        urlopen=_bundle_urlopen(bundle_path, calls),
    )
    damaged_path = Path(project_root) / "src/molab_witness_machine/v12_visuals.py"
    damaged_path.unlink()
    recovered_root = namespace["_resolve_project_root"](
        sandbox / "notebook.py",
        urlopen=_bundle_urlopen(bundle_path, calls),
    )
    assert recovered_root == project_root
    assert len(calls) == 2
    assert damaged_path.is_file()
    _assert_bundle_runtime_loads(Path(recovered_root))


def test_bootstrap_rejects_hash_mismatch(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    namespace = _bootstrap_namespace()
    bundle_path = _write_bundle(tmp_path / "bundle.zip")
    _configure_bundle(namespace, bundle_path)
    namespace["BUNDLE_SHA256"] = "0" * 64
    sandbox = tmp_path / "sandbox"
    sandbox.mkdir()
    monkeypatch.chdir(sandbox)
    monkeypatch.setenv("XDG_CACHE_HOME", str(tmp_path / "xdg-cache"))
    with pytest.raises(RuntimeError, match="expected SHA-256"):
        namespace["_resolve_project_root"](
            sandbox / "notebook.py",
            urlopen=_bundle_urlopen(bundle_path, []),
        )


def test_bootstrap_rejects_traversal_member(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    namespace = _bootstrap_namespace()
    bundle_path = _write_bundle(
        tmp_path / "bundle.zip",
        extra_members=(("witness_machine_v12/../escape.txt", b"escape"),),
    )
    _configure_bundle(namespace, bundle_path)
    sandbox = tmp_path / "sandbox"
    sandbox.mkdir()
    monkeypatch.chdir(sandbox)
    monkeypatch.setenv("XDG_CACHE_HOME", str(tmp_path / "xdg-cache"))
    with pytest.raises(RuntimeError, match="unsafe ZIP member"):
        namespace["_resolve_project_root"](
            sandbox / "notebook.py",
            urlopen=_bundle_urlopen(bundle_path, []),
        )


def test_bootstrap_rejects_missing_required_asset(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    namespace = _bootstrap_namespace()
    missing_relative = "artifacts/v12_public/v12_real_frozen_scorer_evidence.manifest.json"
    bundle_path = _write_bundle(
        tmp_path / "bundle.zip",
        omitted_paths={missing_relative},
    )
    _configure_bundle(namespace, bundle_path)
    sandbox = tmp_path / "sandbox"
    sandbox.mkdir()
    monkeypatch.chdir(sandbox)
    monkeypatch.setenv("XDG_CACHE_HOME", str(tmp_path / "xdg-cache"))
    with pytest.raises(RuntimeError, match="missing .*v12_real_frozen_scorer_evidence.manifest.json"):
        namespace["_resolve_project_root"](
            sandbox / "notebook.py",
            urlopen=_bundle_urlopen(bundle_path, []),
        )


def test_notebook_has_no_static_molab_namespace_imports() -> None:
    _assert_no_static_molab_imports()


def test_release_builder_keeps_static_session_in_runtime_bundle() -> None:
    release_source = (PACKAGE_DIR / "v12_release.py").read_text(encoding="utf-8")
    assert SESSION.relative_to(ROOT).as_posix() in release_source
