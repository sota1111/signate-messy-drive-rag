"""SOT-2694 — formula_apply_store の純関数テスト（LLM/corpus 不要）。

記載式の変数束縛・Decimal 評価（idx68 型のページ本文）と、docx 入れ子統計量表の検出（idx50 型）を
合成データで固定する。全て build 成果ではなく抽出ロジックの単体検証。
"""
from __future__ import annotations

from decimal import Decimal
from pathlib import Path

from src.rag.corpus import FileRef
from src.rag.index import formula_apply_store as S

# idx68 の実ページ本文（image_ocr ページ5 と同型の並び）。
_PAGE5 = (
    "生成AIの経済的インパクトと圧倒的な投資収益率（ROI）\n\n"
    "3.7倍\n圧倒的なROI\n生成AIへの投資1ドルあたりの平均リターン\n\n"
    "+15.2% / +22.6%\nコスト削減 / 生産性向上\n初期導入企業の平均。一部では最大80%達成\n\n"
    "注釈：投資実装係数＝（生産性向上率＋コスト削減率）×ROI倍率\n"
)


def _ref():
    return FileRef(path=Path("/x/未来予測.pdf"), project="東都人材プラットフォーム",
                   category="00.提案", rel="x/未来予測.pdf", name="未来予測.pdf", ext="pdf")


def test_eval_expr_decimal_exact():
    env = {"a": Decimal("0.378"), "b": Decimal("3.7")}
    assert S._eval_expr("a×b", env) == Decimal("1.3986")
    assert S._eval_expr("(1+2)*3", {}) == Decimal("9")
    assert S._eval_expr("10/4", {}) == Decimal("2.5")


def test_decimal_of_percent_and_mult():
    assert S._decimal_of("22.6", "%") == Decimal("0.226")
    assert S._decimal_of("3.7", "倍") == Decimal("3.7")


def test_formula_record_from_page():
    recs = S._formula_records_for_page(_ref(), "ページ5", _PAGE5)
    assert len(recs) == 1
    rec = recs[0]
    assert rec["formula_name"] == "投資実装係数"
    assert rec["value"] == "1.3986"
    assert {k: v["value"] for k, v in rec["bindings"].items()} == {
        "生産性向上率": "0.226", "コスト削減率": "0.152", "ROI倍率": "3.7"}


def test_no_formula_record_when_binding_incomplete():
    # ROI の 倍 値が無い ⇒ 束縛不完全 ⇒ レコードを焼かない（precision-first）。
    page = "投資実装係数＝（生産性向上率＋コスト削減率）×ROI倍率\n+15.2% / +22.6%\nコスト削減 / 生産性向上\n"
    assert S._formula_records_for_page(_ref(), "p", page) == []


def test_non_derived_name_ignored():
    # 名称が係数/指数等でない普通の等式は式レコードにしない。
    page = "合計＝(3+4)×2\n"
    assert S._formula_records_for_page(_ref(), "p", page) == []


class _FakeCell:
    def __init__(self, text):
        self.text = text
        self.tables = []


class _FakeRow:
    def __init__(self, cells):
        self.cells = [_FakeCell(c) for c in cells]


class _FakeTable:
    def __init__(self, rows):
        self.rows = [_FakeRow(r) for r in rows]


def test_stat_table_record_detects_nested_shape():
    ref = FileRef(path=Path("/x/調査.docx"), project="東都", category="00.提案",
                  rel="x/調査.docx", name="調査.docx", ext="docx")
    tbl = _FakeTable([
        ["情報源・調査主体", "調査・予測基準時期", "中央値・平均値（米ドル）",
         "下位10%・最低水準", "上位90%・最高水準"],
        ["Salary.com", "2025年予測", "123,778", "112,000", "137,000"],
        ["Indeed", "2025年5月予測", "127,689", "80,000", "204,000"],
    ])
    rec = S._stat_table_record(ref, tbl)
    assert rec is not None and rec["kind"] == "stat_table"
    assert rec["unit"] == "ドル"
    salary = next(r for r in rec["rows"] if r["key"] == "Salary.com")
    # ヘッダは NFKC 正規化される（全角括弧→半角）。
    vals = {c["header"]: c["value"] for c in salary["cells"]}
    assert vals["中央値・平均値(米ドル)"] == 123778
    assert vals["上位90%・最高水準"] == 137000


def test_stat_table_ignores_non_stat_table():
    ref = FileRef(path=Path("/x/調査.docx"), project="東都", category="c",
                  rel="x/調査.docx", name="調査.docx", ext="docx")
    tbl = _FakeTable([["氏名", "所属"], ["田中", "営業"]])
    assert S._stat_table_record(ref, tbl) is None
