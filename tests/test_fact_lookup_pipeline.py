"""SOT-2612 (Wave B2) — offline tests for the deterministic ``simple_lookup`` (fact_lookup) pipeline.

Network-free and (by default) corpus-free: file resolution (``_resolve_project`` / ``_resolve_schedule_ref``
/ ``_resolve_report_pptx_ref``) and the office reads (``_read_schedule_rows`` / ``_read_report_slides``) are
monkeypatched with canned structures, so the pipeline's own recognizer / column-mapping / boundary-rule /
extremum / metric-density logic runs (not a stub). Invariants under test: the pipeline self-registers
``simple_lookup``; the schedule recognizer forward-fills grouping columns and picks the *unique* extremum
task (idx89 型); the report recognizer maps slide→printed page and returns the dominant metric-density page
(idx84 型); each recognizer falls back (``None``) on ambiguity / missing structure / free-form facts (idx21
型) so the router routes to the LLM loop; OFF ⇒ byte-identical; and it wires through ``det_pipeline.resolve``
+ ``formatting.format_contract`` end-to-end (template-first — no LLM naturalize). A corpus-gated integration
test proves idx84/idx89 match against the real corpus when present.
"""
from __future__ import annotations

import datetime as dt
import os

import pytest

from src.rag.agent import det_pipeline as dp
from src.rag.agent import formatting
from src.rag.agent.pipelines import fact_lookup as fl
from src.rag.extract.office import NormalizedXlsxRow


# --------------------------------------------------------------------------- registry cleanup fixture
@pytest.fixture(autouse=True)
def _restore_registry():
    saved = dict(dp._REGISTRY)
    try:
        yield
    finally:
        dp._REGISTRY.clear()
        dp._REGISTRY.update(saved)


# --------------------------------------------------------------------------- synthetic schedule builder
_SCHED_HEADER = ("タスクID", "フェーズNo.", "タスク名", "開始日", "終了日")


class _StubRef:
    """A minimal FileRef stand-in carrying the ``.rel`` / ``.path`` the recognizers read for evidence."""

    def __init__(self, rel="プロジェクト/例/ファイル", path="/tmp/例"):
        self.rel = rel
        self.path = path


def _rows(header, data):
    out = [NormalizedXlsxRow(1, tuple(header))]
    for i, values in enumerate(data, 2):
        out.append(NormalizedXlsxRow(i, tuple(values)))
    return out


def _wire_schedule(monkeypatch, rows, *, project="京橋信用ソリューションズ", header=_SCHED_HEADER):
    monkeypatch.setattr(fl, "_resolve_project", lambda q: project)
    monkeypatch.setattr(fl, "_resolve_schedule_ref", lambda q, p: _StubRef())
    monkeypatch.setattr(fl, "_read_schedule_rows", lambda ref: (tuple(header), rows))


def _wire_report(monkeypatch, display_pages, slides, *, project="株式会社東都人材プラットフォーム"):
    monkeypatch.setattr(fl, "_resolve_project", lambda q: project)
    monkeypatch.setattr(fl, "_resolve_report_pptx_ref", lambda p: _StubRef())
    monkeypatch.setattr(fl, "_read_report_slides", lambda ref: (dict(display_pages), list(slides)))


# --------------------------------------------------------------------------- registration
def test_pipeline_registers_for_simple_lookup():
    fl.register(replace=True)
    assert "simple_lookup" in dp.registered_contracts()
    assert dp._REGISTRY["simple_lookup"] is fl.pipeline


# --------------------------------------------------------------------------- Recognizer A: schedule phase task
def test_schedule_last_start_task_grounds(monkeypatch):
    q = "京橋信用ソリューションズのスケジュール.xlsxにおいて、フェーズNo6にて最後に開始するタスク名は何ですか。"
    rows = _rows(_SCHED_HEADER, [
        ("T23", "6", "最終モデル確定・再評価", dt.datetime(2025, 11, 1), dt.datetime(2025, 11, 5)),
        ("T24", "6", "最終報告書作成", dt.datetime(2025, 11, 3), dt.datetime(2025, 11, 7)),
        ("T27", "6", "最終報告・成果物提出・検収会", dt.datetime(2025, 11, 11), dt.datetime(2025, 11, 11)),
        ("T28", "7", "検収結果反映", dt.datetime(2025, 11, 12), dt.datetime(2025, 11, 12)),
    ])
    _wire_schedule(monkeypatch, rows)
    out = fl.pipeline(q)
    assert out is not None
    assert out["value"] == "最終報告・成果物提出・検収会"
    assert out["evidence"]["phase_no"] == 6
    assert out["evidence"]["order"] == "last"
    assert out["method"]["shape"] == "schedule_phase_ordinal_task"


def test_schedule_first_start_task_grounds(monkeypatch):
    q = "スケジュール.xlsxでフェーズNo6にて最初に開始するタスク名は。"
    rows = _rows(_SCHED_HEADER, [
        ("T24", "6", "後発タスク", dt.datetime(2025, 11, 3), dt.datetime(2025, 11, 7)),
        ("T23", "6", "先頭タスク", dt.datetime(2025, 11, 1), dt.datetime(2025, 11, 5)),
    ])
    _wire_schedule(monkeypatch, rows)
    out = fl.pipeline(q)
    assert out is not None and out["value"] == "先頭タスク"
    assert out["evidence"]["order"] == "first"


def test_schedule_forward_filled_phase_group(monkeypatch):
    # A blank フェーズNo. cell (merged-cell group continuation) is forward-filled by normalized_xlsx_rows;
    # the recognizer must still see it as phase 6 — here the fixture supplies the already-filled value.
    q = "スケジュール.xlsxでフェーズNo6にて最後に開始するタスク名は何ですか。"
    rows = _rows(_SCHED_HEADER, [
        ("T23", "6", "初回", dt.datetime(2025, 11, 1), dt.datetime(2025, 11, 5)),
        ("T24", "6", "継続（前方補完）", dt.datetime(2025, 11, 8), dt.datetime(2025, 11, 9)),
    ])
    _wire_schedule(monkeypatch, rows)
    assert fl.pipeline(q)["value"] == "継続（前方補完）"


def test_schedule_tie_extremum_falls_back(monkeypatch):
    # Two phase-6 rows share the latest start date ⇒ ambiguous extremum ⇒ safe fallback (no wrong commit).
    q = "スケジュール.xlsxでフェーズNo6にて最後に開始するタスク名は何ですか。"
    rows = _rows(_SCHED_HEADER, [
        ("T26", "6", "A", dt.datetime(2025, 11, 11), dt.datetime(2025, 11, 11)),
        ("T27", "6", "B", dt.datetime(2025, 11, 11), dt.datetime(2025, 11, 12)),
    ])
    _wire_schedule(monkeypatch, rows)
    assert fl.pipeline(q) is None


def test_schedule_no_matching_phase_falls_back(monkeypatch):
    q = "スケジュール.xlsxでフェーズNo9にて最後に開始するタスク名は何ですか。"
    rows = _rows(_SCHED_HEADER, [("T23", "6", "X", dt.datetime(2025, 11, 1), dt.datetime(2025, 11, 5))])
    _wire_schedule(monkeypatch, rows)
    assert fl.pipeline(q) is None


def test_schedule_non_name_target_falls_back(monkeypatch):
    # asks for 担当者, not タスク名 ⇒ recognizer does not fire (precision: only the task-name read).
    q = "スケジュール.xlsxでフェーズNo6にて最後に開始する担当者は誰ですか。"
    rows = _rows(_SCHED_HEADER, [("T27", "6", "Z", dt.datetime(2025, 11, 11), dt.datetime(2025, 11, 11))])
    _wire_schedule(monkeypatch, rows)
    assert fl.pipeline(q) is None


def test_schedule_ambiguous_order_falls_back(monkeypatch):
    q = "スケジュール.xlsxでフェーズNo6の開始するタスク名は何ですか。"  # neither 最後 nor 最初
    rows = _rows(_SCHED_HEADER, [("T27", "6", "Z", dt.datetime(2025, 11, 11), dt.datetime(2025, 11, 11))])
    _wire_schedule(monkeypatch, rows)
    assert fl.pipeline(q) is None


def test_schedule_unreadable_date_falls_back(monkeypatch):
    # A matching row whose start cell cannot be ordered deterministically ⇒ decline the whole extremum.
    q = "スケジュール.xlsxでフェーズNo6にて最後に開始するタスク名は何ですか。"
    rows = _rows(_SCHED_HEADER, [
        ("T23", "6", "A", dt.datetime(2025, 11, 1), dt.datetime(2025, 11, 5)),
        ("T24", "6", "B", "未定", dt.datetime(2025, 11, 9)),
    ])
    _wire_schedule(monkeypatch, rows)
    assert fl.pipeline(q) is None


# --------------------------------------------------------------------------- Recognizer B: report metric page
def _report_slides():
    # slide 6 (printed page 5) is dense in per-model F1 values; other slides mention F1 at most once.
    return {2: 1, 5: 4, 6: 5, 7: 6}, [
        (2, "エグゼクティブサマリ Macro F1 0.474"),
        (5, "データ品質・特徴選択 総レコード数 11,529"),
        (6, "モデル性能比較 T01: F1=0.309 → T03: F1=0.449 → 最終: F1=0.474"),
        (7, "解釈・公平性 グループ別 Macro F1 比較"),
    ]


def test_report_metric_page_grounds(monkeypatch):
    q = ("東都人材プラットフォームの最終報告書で、モデル毎のF1スコアが"
         "ランキング形式で記載されているページ数を教えてください。")
    _wire_report(monkeypatch, *_report_slides())
    out = fl.pipeline(q)
    assert out is not None
    assert out["value"] == "5"  # the printed page number, not the physical slide index (6)
    assert out["evidence"]["metric"] == "F1"
    assert out["evidence"]["slide"] == 6
    assert out["method"]["shape"] == "report_metric_ranking_page"


def test_report_metric_page_renders_printed_number_not_physical(monkeypatch):
    q = "…の最終報告書でモデル別のaccuracyの比較が記載されている何ページですか。"
    disp = {3: 2, 4: 3}
    slides = [(3, "概要 accuracy 0.5"), (4, "比較 A: accuracy 0.51 B: accuracy 0.62 C: accuracy 0.48")]
    _wire_report(monkeypatch, disp, slides)
    out = fl.pipeline(q)
    assert out is not None and out["value"] == "3"  # slide 4 dominates → printed page 3


def test_report_metric_page_tie_falls_back(monkeypatch):
    q = "最終報告書でモデル毎のF1スコアのランキングは何ページですか。"
    disp = {2: 1, 3: 2}
    slides = [(2, "F1=0.30 F1=0.40"), (3, "F1=0.50 F1=0.60")]  # equal density ⇒ ambiguous
    _wire_report(monkeypatch, disp, slides)
    assert fl.pipeline(q) is None


def test_report_metric_page_no_ranking_cue_falls_back(monkeypatch):
    # No ranking/比較/一覧 cue ⇒ recognizer does not fire (precision guard).
    q = "最終報告書でモデル毎のF1スコアが記載されているページ数を教えてください。"
    _wire_report(monkeypatch, *_report_slides())
    assert fl.pipeline(q) is None


def test_report_metric_page_ambiguous_metric_falls_back(monkeypatch):
    # Two distinct metrics named ⇒ target metric ambiguous ⇒ fallback.
    q = "最終報告書でモデル毎のF1スコアとaccuracyのランキングは何ページですか。"
    _wire_report(monkeypatch, *_report_slides())
    assert fl.pipeline(q) is None


def test_report_metric_page_no_printed_page_falls_back(monkeypatch):
    # Winning slide has no printed footer page ⇒ decline rather than guess the physical index.
    q = "最終報告書でモデル毎のF1スコアのランキングは何ページですか。"
    slides = [(6, "F1=0.309 F1=0.449 F1=0.474")]
    _wire_report(monkeypatch, {}, slides)
    assert fl.pipeline(q) is None


# --------------------------------------------------------------------------- free-form fact ⇒ fallback
def test_free_form_role_fact_falls_back(monkeypatch):
    # idx21 型: 「主担当者の役職」 — no schedule/page structure to pin ⇒ stays on the champion LLM path.
    q = "青葉バイオメディカル機器のクライアントの主担当者の役職は何ですか。"
    monkeypatch.setattr(fl, "_resolve_project", lambda x: "青葉バイオメディカル機器")
    assert fl.pipeline(q) is None


def test_pipeline_never_raises(monkeypatch):
    # A recognizer that raises must be swallowed into a fallback (never breaks the answer path).
    def boom(q):
        raise RuntimeError("read failed")

    monkeypatch.setattr(fl, "_schedule_phase_task", boom)
    monkeypatch.setattr(fl, "_report_metric_page", boom)
    monkeypatch.setattr(fl, "_RECOGNIZERS", (boom,))
    assert fl.pipeline("フェーズNo6の最後に開始するタスク名は何ですか。スケジュール") is None


# --------------------------------------------------------------------------- router + formatting e2e
def test_resolve_off_returns_none(monkeypatch):
    monkeypatch.delenv("RAG_DET_PIPELINE_ROUTER", raising=False)
    q = "スケジュール.xlsxでフェーズNo6にて最後に開始するタスク名は何ですか。"
    rows = _rows(_SCHED_HEADER, [("T27", "6", "Z", dt.datetime(2025, 11, 11), dt.datetime(2025, 11, 11))])
    _wire_schedule(monkeypatch, rows)
    # flag OFF ⇒ champion serve path byte-identical: the pipeline is never consulted.
    assert dp.resolve(q, "simple_lookup") is None


def test_end_to_end_through_formatting_template_only(monkeypatch):
    q = "スケジュール.xlsxでフェーズNo6にて最後に開始するタスク名は何ですか。"
    rows = _rows(_SCHED_HEADER, [
        ("T23", "6", "先", dt.datetime(2025, 11, 1), dt.datetime(2025, 11, 5)),
        ("T27", "6", "最終報告・成果物提出・検収会", dt.datetime(2025, 11, 11), dt.datetime(2025, 11, 11)),
    ])
    _wire_schedule(monkeypatch, rows)
    contract = dp.resolve(q, "simple_lookup", force=True)
    formatted = formatting.format_contract(contract, q, contract_type="simple_lookup", force=True)
    assert formatted["value"] == "最終報告・成果物提出・検収会"
    assert formatted["method"]["formatting"]["template_only"] is True


# --------------------------------------------------------------------------- corpus-gated integration (idx84/89)
_CORPUS_PRESENT = (
    __import__("config").settings.CORPUS_DIR.exists()
    if os.getenv("RAG_SKIP_CORPUS_TESTS") not in {"1", "true", "yes", "on"} else False
)


@pytest.mark.skipif(not _CORPUS_PRESENT, reason="corpus (data/share_drive) not present")
@pytest.mark.parametrize("question,gold", [
    ("東都人材プラットフォームの最終報告書で分析結果が記載されている中で、モデル毎のF1スコアが"
     "ランキング形式で記載されているページ数を教えてください。", "5"),
    ("京橋信用ソリューションズのスケジュール.xlsxにおいて、フェーズNo6にて最後に開始するタスク名は"
     "何ですか。", "最終報告・成果物提出・検収会"),
])
def test_real_corpus_idx84_idx89_match(question, gold):
    contract = dp.resolve(question, "simple_lookup", force=True)
    assert contract is not None, "recognizer did not ground the fact on the real corpus"
    formatted = formatting.format_contract(contract, question, contract_type="simple_lookup", force=True)
    assert formatted["value"] == gold
