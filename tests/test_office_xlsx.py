from pathlib import Path

import openpyxl

from src.rag.corpus import FileRef
from src.rag.extract.office import extract_xlsx, normalized_xlsx_rows


def _ref(path: Path) -> FileRef:
    return FileRef(path=path, project="p", category="plan", rel=path.name,
                   name=path.name, ext="xlsx")


def test_grouping_columns_forward_fill_without_inventing_other_cells(tmp_path: Path) -> None:
    path = tmp_path / "schedule.xlsx"
    wb = openpyxl.Workbook()
    ws = wb.active
    ws.title = "WBS"
    ws.append(["タスクID", "フェーズNo.", "フェーズ名", "タスク名", "担当者", "開始日"])
    ws.append(["T23", 6, "最終成果物化", "最終モデル確定", "鈴木", "2025-11-01"])
    ws.append(["T24", None, None, "報告書作成", None, "2025-11-03"])
    ws.append(["T27", None, None, "最終報告・成果物提出・検収会", "佐藤", "2025-11-11"])
    ws.append(["T28", 7, "クローズ", "検収結果反映", "佐藤", "2025-11-12"])
    wb.save(path)

    loaded = openpyxl.load_workbook(path, data_only=True)
    rows = normalized_xlsx_rows(loaded["WBS"])
    assert rows[2].values[1:3] == (6, "最終成果物化")
    assert rows[3].values[1:3] == (6, "最終成果物化")
    assert rows[2].values[4] is None  # non-grouping owner column remains genuinely blank

    text = extract_xlsx(_ref(path), None)
    assert "【グルーピング列を前方補完: フェーズNo.、フェーズ名】" in text
    assert "T27 | 6 | 最終成果物化 | 最終報告・成果物提出・検収会" in text
