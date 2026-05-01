"""Emit _NEURAL_B39_ZLIB_B64 literal blocks for embedding in notebook 14."""
from __future__ import annotations

import base64
import pathlib
import textwrap
import zlib

root = pathlib.Path(__file__).resolve().parents[1]
raw = (root / "src" / "neural_blend_refined_b39.py").read_bytes()
b64 = base64.b64encode(zlib.compress(raw, level=9)).decode("ascii")
parts = textwrap.wrap(b64, 72)

lines = ["_NEURAL_B39_ZLIB_B64 = ("]
for p in parts:
    lines.append(f'    "{p}"')
lines.append(")")

(root / "tools" / "neural_blend_b39_zlib_b64_fragment.py").write_text("\n".join(lines), encoding="utf-8")
print("parts", len(parts), "literal lines", len(lines))
