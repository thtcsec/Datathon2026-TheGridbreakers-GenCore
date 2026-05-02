"""Insert §1c walk-forward + replace ML_DIAG printing in notebook 14."""
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
NB = ROOT / "notebooks" / "14_Final_LB_Optimization_Journey.ipynb"

OLD_ML_DIAG = (
    'print("ML_DIAG keys:", sorted(ML_DIAG.keys()))\n'
    'for k in sorted(ML_DIAG.keys()):\n'
    '    if k == "feature_cols":\n'
    '        print(f"  {k}: {len(ML_DIAG[k])} columns")\n'
    '    else:\n'
    '        print(f"  {k}: {ML_DIAG[k]}")\n'
    'print(f"b39+GBDT anchor signature: {frame_signature(anchor_b39)}")\n'
)

NEW_ML_DIAG = (
    'print("ML_DIAG (scalar metrics + feature count):")\n'
    'for k in sorted(ML_DIAG.keys()):\n'
    '    if k in ("feature_cols", "cv_fold_detail"):\n'
    '        continue\n'
    '    print(f"  {k}: {ML_DIAG[k]}")\n'
    'print(f"  feature_cols: {len(ML_DIAG[\\"feature_cols\\"])} columns")\n'
    'if "cv_fold_detail" in ML_DIAG:\n'
    '    print("\\nTimeSeriesSplit — GBDT tabular: MAE mean/std/min/max theo target×model:")\n'
    '    display(ML_DIAG["cv_fold_detail"].groupby(["target", "model"])["mae"].agg(["mean", "std", "min", "max"]))\n'
    '    print("Chi tiết từng fold (expanding window):")\n'
    '    display(ML_DIAG["cv_fold_detail"])\n'
    'print(f"b39+GBDT anchor signature: {frame_signature(anchor_b39)}")\n'
)


def main() -> None:
    nb = json.loads(NB.read_text(encoding="utf-8"))

    patched = False
    for c in nb["cells"]:
        if c.get("id") != "ml_tabular_code_1b":
            continue
        text = "".join(c["source"])
        if OLD_ML_DIAG not in text:
            if NEW_ML_DIAG.split("\n")[0] in text:
                print("ML_DIAG block already updated.")
                patched = True
                break
            raise SystemExit("Expected OLD_ML_DIAG not found in ml_tabular_code_1b")
        c["source"] = text.replace(OLD_ML_DIAG, NEW_ML_DIAG).splitlines(keepends=True)
        patched = True
        break
    if not patched:
        raise SystemExit("ml_tabular_code_1b cell not found")

    full = "".join(json.dumps(nb))
    if "1c. Validation nội bộ" in full and "WF_REPORT" in full:
        print("§1c cells already present.")
        NB.write_text(json.dumps(nb, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        return

    md_cell = {
        "cell_type": "markdown",
        "id": "ml_val_1c_md",
        "metadata": {},
        "source": [
            "## 1c. Validation nội bộ — walk-forward (không nhìn leaderboard)\n",
            "\n",
            "**TimeSeriesSplit / từng fold** đã in ở §1b (`ML_DIAG[\"cv_fold_detail\"]`).\n",
            "\n",
            "Đoạn dưới chạy **rolling-origin backtest** trên các cửa sổ cuối 2022 (còn nhãn trong `sales.csv`): chỉ dùng quá khứ đến `train_cutoff`, dự báo các ngày test, so MAE với thực tế. Đây là **đánh giá học máy** tách biệt với các hệ số chỉnh theo public LB (§3–§13).\n",
            "\n",
            "- `RUN_WALK_FORWARD = True`: thêm vài phút CPU; đặt `False` trên Kaggle nếu cần rút ngắn (submission cuối không đổi).\n",
        ],
    }
    code_cell = {
        "cell_type": "code",
        "execution_count": None,
        "id": "ml_val_1c_code",
        "metadata": {},
        "outputs": [],
        "source": [
            "RUN_WALK_FORWARD = True  # False = bỏ qua backtest (nhanh hơn trên Kaggle)\n",
            "\n",
            "if RUN_WALK_FORWARD:\n",
            "    WF_REPORT = _ml_tab_mod.walk_forward_gbdt_evaluation(\n",
            "        DATA,\n",
            "        windows=_ml_tab_mod.default_walk_forward_windows_2022(),\n",
            "        date_col=\"Date\",\n",
            "    )\n",
            "    print(\"Walk-forward GBDT ensemble vs actuals (late 2022 windows):\")\n",
            "    display(WF_REPORT)\n",
            "else:\n",
            "    print(\"Walk-forward skipped (RUN_WALK_FORWARD=False).\")\n",
            "    WF_REPORT = None\n",
        ],
    }

    idx = None
    for i, c in enumerate(nb["cells"]):
        if c.get("id") == "ml_tabular_code_1b":
            idx = i + 1
            break
    if idx is None:
        raise SystemExit("ml_tabular_code_1b not found")

    nb["cells"].insert(idx, md_cell)
    nb["cells"].insert(idx + 1, code_cell)

    NB.write_text(json.dumps(nb, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print("§1c inserted at index", idx, "+ ML_DIAG patch")


if __name__ == "__main__":
    main()
