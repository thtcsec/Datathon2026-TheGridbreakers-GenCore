"""Replace _ML_TABULAR_ZLIB_B64 block in notebook 14 from tools/_ml_tabular_zlib_b64_fragment.txt."""
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
NB = ROOT / "notebooks" / "14_Final_LB_Optimization_Journey.ipynb"
FRAG = ROOT / "tools" / "_ml_tabular_zlib_b64_fragment.txt"


def main() -> None:
    frag = FRAG.read_text(encoding="utf-8").strip()
    nb = json.loads(NB.read_text(encoding="utf-8"))
    replaced = False
    for c in nb["cells"]:
        src = c.get("source")
        if not src:
            continue
        text = "".join(src)
        if "_ML_TABULAR_ZLIB_B64 = (" not in text:
            continue
        mark = "_ML_TABULAR_ZLIB_B64 = ("
        start = text.index(mark)
        open_idx = start + len(mark) - 1
        assert text[open_idx] == "(", repr(text[open_idx : open_idx + 3])
        depth = 1
        i = open_idx + 1
        while i < len(text) and depth > 0:
            ch = text[i]
            if ch == "(":
                depth += 1
            elif ch == ")":
                depth -= 1
            i += 1
        if depth != 0:
            raise RuntimeError("Unbalanced parens in notebook ML bundle")
        while i < len(text) and text[i] in "\r\n":
            i += 1
        new_text = text[:start] + frag + "\n" + text[i:]
        c["source"] = new_text.splitlines(keepends=True)
        replaced = True
        break
    if not replaced:
        raise SystemExit("No _ML_TABULAR_ZLIB_B64 cell found")
    NB.write_text(json.dumps(nb, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print("Updated ML bundle in", NB)


if __name__ == "__main__":
    main()
