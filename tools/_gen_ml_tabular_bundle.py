"""Emit zlib+b64 fragment for notebooks/ml_tabular_blend materialization."""
import base64
import pathlib
import zlib

ROOT = pathlib.Path(__file__).resolve().parents[1]
src = ROOT / "src" / "ml_tabular_blend.py"
raw = src.read_bytes()
b64 = base64.b64encode(zlib.compress(raw, 9)).decode()
chunks = [b64[i : i + 76] for i in range(0, len(b64), 76)]
lines = ['    "' + c + '",' for c in chunks]
out = ROOT / "tools" / "_ml_tabular_zlib_b64_fragment.txt"
out.write_text("_ML_TABULAR_ZLIB_B64 = (\n" + "\n".join(lines) + "\n)\n", encoding="utf-8")
print(out, "lines", len(chunks), "b64 len", len(b64))
