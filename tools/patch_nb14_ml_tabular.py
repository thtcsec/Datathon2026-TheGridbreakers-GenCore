"""Insert markdown + code cells (GBDT tabular blend) into notebook 14 after setup cell."""
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
NB = ROOT / "notebooks" / "14_Final_LB_Optimization_Journey.ipynb"
FRAG = ROOT / "tools" / "_ml_tabular_zlib_b64_fragment.txt"


def main() -> None:
    frag_txt = FRAG.read_text(encoding="utf-8").strip()
    frag_lines = frag_txt.splitlines()

    rest = r'''def _materialize_ml_tabular_from_bundle(dest: Path) -> None:
    import base64 as _b64
    import zlib as _zlib
    _p = _ML_TABULAR_ZLIB_B64
    if isinstance(_p, tuple):
        _p = "".join(_p)
    raw = _zlib.decompress(_b64.b64decode(_p))
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_bytes(raw)


def _resolve_ml_tabular_blend_py_path(out_dir: Path) -> Path:
    """Repo ``src/`` if present; else materialize embedded bundle into OUT (Kaggle-safe)."""
    name = "ml_tabular_blend.py"
    seen = set()
    cands = []
    for p in (
        *(c / "src" / name for c in [Path.cwd(), *Path.cwd().parents]),
        Path("/kaggle/working") / name,
        Path("/kaggle/working") / "src" / name,
        out_dir / name,
        out_dir / "src" / name,
    ):
        rp = p.resolve()
        if rp not in seen:
            seen.add(rp)
            cands.append(p)
    for p in cands:
        if p.exists() and p.stat().st_size >= 800:
            return p.resolve()
    dest = (out_dir / name).resolve()
    _materialize_ml_tabular_from_bundle(dest)
    if not dest.exists() or dest.stat().st_size < 800:
        raise FileNotFoundError(f"Cannot materialize {name}")
    return dest


def _load_ml_tabular_blend_module():
    py_file = _resolve_ml_tabular_blend_py_path(OUT)
    spec = importlib.util.spec_from_file_location("_ml_tabular_blend", py_file)
    module = importlib.util.module_from_spec(spec)
    if spec.loader is None:
        raise RuntimeError("spec.loader missing for ml tabular module")
    spec.loader.exec_module(module)
    return module


_ml_tab_mod = _load_ml_tabular_blend_module()
ML_TABULAR_WEIGHT = 0.18  # higher = more GBDT vs neural anchor; tune conservatively for LB scale

print("\\nGBDT tabular (XGB+LGB, TimeSeriesSplit CV) blended into b39 anchor...")
anchor_b39, ML_DIAG = _ml_tab_mod.run_ml_blend_into_anchor(
    DATA,
    OUT,
    anchor_b39,
    sample,
    ml_weight=ML_TABULAR_WEIGHT,
)
if not sample["Date"].equals(anchor_b39["Date"]):
    anchor_b39 = anchor_b39.set_index("Date").loc[sample["Date"]].reset_index()

print("ML_DIAG keys:", sorted(ML_DIAG.keys()))
for k in sorted(ML_DIAG.keys()):
    if k == "feature_cols":
        print(f"  {k}: {len(ML_DIAG[k])} columns")
    else:
        print(f"  {k}: {ML_DIAG[k]}")
print(f"b39+GBDT anchor signature: {frame_signature(anchor_b39)}")
'''

    code_source = [ln + "\n" for ln in frag_lines] + [ln + "\n" for ln in rest.strip().splitlines()]

    md_source = [
        "## 1b. GBDT tabular (XGBoost + LightGBM) + TimeSeriesSplit CV\n",
        "\n",
        "Module `src/ml_tabular_blend.py` huấn luyện **gradient boosting** trên lag / calendar / `web_traffic` / `inventory`, in **MAE trung bình các fold** (split thời gian), rồi **blend** vào anchor neural b39 **trước** các bước LB-guided (v20–v41).\n",
        "\n",
        "- **Mặc định** `ML_TABULAR_WEIGHT = 0.18`: giữ anchor neural là chủ đạo; tăng nếu muốn nặng tabular hơn.\n",
        "- **Cảnh báo Kaggle**: public LB ~673k và `ALPHA_V23 = 4.30` là kết quả **tuning trên leaderboard** — private có thể khác. CV MAE in ra đây là **metric nội bộ**, không phải private score.\n",
        "- Pipeline neural **b39** (`neural_blend_refined_b39.py`) đã có stacking LGB/XGB/CatBoost/NN; bước này thêm một **forecast GBDT độc lập** cùng kiểu `notebooks/03` / `src/models.py`.\n",
        "- Chuỗi `_ML_TABULAR_ZLIB_B64` chỉ là **mã nguồn** nén (giống bundle b39), không phải dữ liệu ngoài BTC.\n",
    ]

    nb = json.loads(NB.read_text(encoding="utf-8"))
    # Avoid duplicate patch
    for c in nb["cells"]:
        src = "".join(c.get("source", []))
        if "1b. GBDT tabular" in src and c["cell_type"] == "markdown":
            print("Notebook already patched (markdown 1b found). Skipping.")
            return

    md_cell = {
        "cell_type": "markdown",
        "id": "ml_tabular_md_1b",
        "metadata": {},
        "source": md_source,
    }
    code_cell = {
        "cell_type": "code",
        "execution_count": None,
        "id": "ml_tabular_code_1b",
        "metadata": {},
        "outputs": [],
        "source": code_source,
    }
    nb["cells"].insert(3, md_cell)
    nb["cells"].insert(4, code_cell)
    NB.write_text(json.dumps(nb, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print("Patched", NB, "inserted cells at index 3-4")


if __name__ == "__main__":
    main()
