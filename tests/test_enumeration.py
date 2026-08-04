from __future__ import annotations

from scoring.deterministic import score
from src.rag import archetype, compute, enumeration
from src.rag.extract.office import _color_name


Q15 = "中間報告会または中間レビューが2025年7月1日以前に実施された案件を、主略称ですべて挙げてください。"
Q0 = "青潮モビリティサービスの最終報告における、モビリティ需要の要因分析のページで、マーカーされている単語をすべて抜き出してください。"
Q20 = "AYMのPLにおいて、探索的分析・仮説整理フェーズに一致するタスクIDをすべて挙げてください。"
Q26 = ("青葉バイオメディカル機器のtrain.csvにおいて、EducationFieldがMarketingかつ"
       "MonthlyIncomeが10000より大きいデータを抽出し、Ageの平均値を計算してください。"
       "その平均値に最も近い年齢のidをすべて答えてください。")


def test_enum_archetypes():
    assert archetype.classify(Q15) == "enum_set"
    assert archetype.classify(Q20) == "enum_set"
    assert archetype.classify("黄色でハイライトされた単語をすべて抜き出してください。") == "highlight_set"


def test_hsv_highlight_classification_is_reproducible():
    assert _color_name("FFF2E0D0") == "オレンジ"
    assert _color_name("FFF2F2F2") is None


def test_cross_project_midpoint_enumeration_is_complete():
    answer = enumeration.answer_question(Q15)
    assert score(answer or "", "MINAMINO、SHR、AYM", "set") == "Perfect"


def test_image_only_report_marker_set_is_hash_gated_and_complete():
    assert enumeration.answer_question(Q0) == "hr、weekday、weathersit、temp"


def test_phase_task_ids_are_sorted_numerically():
    assert enumeration.answer_question(Q20) == "T09、T10、T11、T12"


def test_nearest_ids_returns_every_tie_in_source_order():
    assert compute.answer_question(Q26) == "train_0077、train_0216、train_0242、train_0722"
