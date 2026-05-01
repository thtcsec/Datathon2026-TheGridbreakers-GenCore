#!/usr/bin/env python3
"""Inject embedded neural module + offline loader into notebook 14."""
from __future__ import annotations

import json
import pathlib

ROOT = pathlib.Path(__file__).resolve().parents[1]
NB = ROOT / "notebooks" / "14_Final_LB_Optimization_Journey.ipynb"
LIT = ROOT / "tools" / "neural_blend_b39_zlib_b64_fragment.py"

LOADER = '''

def _materialize_neural_from_bundle(dest: Path) -> None:
    import base64 as _b64
    import zlib as _zlib
    raw = _zlib.decompress(_b64.b64decode(_NEURAL_B39_ZLIB_B64))
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_bytes(raw)


def _resolve_neural_blend_py_path(out_dir: Path) -> Path:
    """Local repo ``src/`` if present; else materialize embedded zlib+b64 into OUT (no extras, no network)."""
    name = "neural_blend_refined_b39.py"
    seen = set()
    cands = []
    for p in (
        Path("/kaggle/working") / name,
        Path("/kaggle/working") / "src" / name,
        out_dir / name,
        out_dir / "src" / name,
        *(c / "src" / name for c in [Path.cwd(), *Path.cwd().parents]),
    ):
        rp = p.resolve()
        if rp not in seen:
            seen.add(rp)
            cands.append(p)
    for p in cands:
        if p.exists() and p.stat().st_size >= 1800:
            return p.resolve()
    dest = (out_dir / name).resolve()
    _materialize_neural_from_bundle(dest)
    if not dest.exists() or dest.stat().st_size < 1800:
        raise FileNotFoundError(f"Cannot materialize {name}")
    return dest


def _load_neural_blend_module():
    py_file = _resolve_neural_blend_py_path(OUT)
    spec = importlib.util.spec_from_file_location("_neural_blend_b39", py_file)
    module = importlib.util.module_from_spec(spec)
    if spec.loader is None:
        raise RuntimeError("spec.loader missing for neural blend module")
    spec.loader.exec_module(module)
    return module

'''


def main() -> None:
    literal_body = LIT.read_text(encoding="utf-8").strip()
    nb = json.loads(NB.read_text(encoding="utf-8"))
    assert nb["cells"][2]["cell_type"] == "code"
    src = "".join(nb["cells"][2]["source"])
    if "_NEURAL_B39_ZLIB_B64" in src and "_materialize_neural_from_github" not in src and "_materialize_neural_from_bundle" in src:
        print("Skipping: notebook cell already matches offline neural bundle loader.")
        return

    key = "\nDATA, OUT, ENV_KIND = _find_data_and_out()"
    src = "".join(nb["cells"][2]["source"])
    if "\nimport urllib.request\n" in src:
        src = src.replace("\nimport urllib.request\n", "\n")

    j2 = src.index(key)

    gh = src.find("\ndef _materialize_neural_from_github(dest:")
    fb = src.find("\ndef _materialize_neural_from_bundle(dest:")
    if gh != -1 and fb != -1:
        src = src[:gh] + src[fb:]  # drop GitHub stub
        j2 = src.index(key)

    needle_start = src.find("\ndef _load_neural_blend_module():", 0, src.index(key))
    if needle_start < 0:
        raise RuntimeError("Could not find _load_neural_blend_module anchor in notebook")

    new_src = src[:needle_start].rstrip() + "\n\n\n" + literal_body + "\n\n" + LOADER.strip() + "\n\n" + src[j2:]
    if "import urllib.request" in new_src:
        new_src = new_src.replace("\nimport urllib.request\n", "\n")

    nb["cells"][2]["source"] = [blk + "\n" for blk in new_src.splitlines()]
    NB.write_text(json.dumps(nb, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print("Patched:", NB, "lines", len(nb["cells"][2]["source"]))


if __name__ == "__main__":
    main()
