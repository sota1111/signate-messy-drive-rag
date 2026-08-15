"""SOT-2721 — 「<列>のヒストグラムで N 番目にカウント数が多いビンの範囲」の決定論 direct-commit
(RAG_HIST_BIN, default OFF, LLM 出力非依存).

idx29「恒一会 かえで総合病院のtrain.xlsx内のTPのヒストグラムで、3番目にカウント数が多いビンの範囲を
小数第6位までで」の gold は `6.088138 ~ 6.288138`（ビン幅 0.2）だが、純 Gemini 経路は LLM 経路で解こうとして
framing churn（区間記法/棄権）で取りこぼす。本ルールはビン境界を実データから Excel 自動ヒストグラム
（Scott 幅3桁 truncate、:func:`chart_numcache._scott_histogram`）で機械計算し、カウント降順 K 番目のビン範囲を
チルダ書式へ確定する。**ビン幅 0.2 は直書きせずデータから導出**する（同一規則が idx10 の AG_ratio 幅0.053/
最多958 も再現）。

【最重要】回帰ガード: ゲートは ヒストグラム × ビン × 範囲/レンジ/区間 × 「N番目」 × カウント × 多い/少ない を
全て要求する。gold100 全走査で合致するのは idx29 のみ（idx10「最も多いカウント数」型は範囲/「N番目」を含まない
ので構造的に None）。案件×列×ファイル名×K×小数第N位×向きはすべて質問から抽出する（gold/idx 非依存）。
"""
from __future__ import annotations

import pytest

from src.rag.agent import formatting


# --------------------------------------------------------------------------- flag
@pytest.mark.parametrize("val,expected", [("1", True), ("on", True), ("0", False), ("", False)])
def test_flag(monkeypatch, val, expected):
    monkeypatch.setenv("RAG_HIST_BIN", val)
    assert formatting.histogram_bin_enabled() is expected


def test_flag_default_off():
    assert formatting.histogram_bin_enabled() is False


# --------------------------------------------------------------------------- gate (parse-only, no corpus)
Q29 = ("恒一会 かえで総合病院のtrain.xlsx内のTPのヒストグラムで、"
       "3番目にカウント数が多いビンの範囲を小数第6位までで答えてください。")
# idx10 型: 範囲でなく最多カウント「数」を問う ⇒ 範囲/「N番目」を含まず構造的に None。
Q10 = "恒一会 かえで総合病院のtrain.xlsxにおいて、AG_ratioのヒストグラムで最も多いカウント数はいくつですか。"


@pytest.mark.parametrize("q", [
    "",
    "TPの平均はいくつですか。",                                   # ヒストグラムでない
    "TPのヒストグラムで最も多いカウント数はいくつですか。",         # 範囲/「N番目」なし (idx10 型)
    "TPのヒストグラムのビンはいくつありますか。",                   # 範囲なし・「N番目」なし
    "TPのヒストグラムで3番目に多いビンの範囲は。",                 # 「カウント」語なし
    Q10,
])
def test_gate_non_target_is_none(q):
    assert formatting.resolve_histogram_bin_direct(q) is None


def test_gate_direction_required(monkeypatch):
    # 「多い」/「少ない」いずれも無ければ向き不明で触らない（列/範囲/「N番目」/カウントは満たしていても）。
    q = "TPのヒストグラムで3番目のカウントのビンの範囲を小数第6位まで。"
    assert formatting.resolve_histogram_bin_direct(q) is None


# --------------------------------------------------------------------------- ranking / formatting (mocked series)
def _mock_series(monkeypatch, counts, categories):
    monkeypatch.setattr(formatting, "_histogram_series", lambda q, c: (counts, categories))


def test_third_most_bin_range_tilde_6dp(monkeypatch):
    # カウント降順 3 番目 = idx1 (494)。両端を小数第6位のチルダ書式へ。
    counts = [410, 494, 618, 521]
    cats = ["(5.888138493, 6.088138493]", "(6.088138493, 6.288138493]",
            "(7.088138493, 7.288138493]", "(6.888138493, 7.088138493]"]
    _mock_series(monkeypatch, counts, cats)
    assert formatting.resolve_histogram_bin_direct(Q29) == "6.088138 ~ 6.288138"


def test_least_direction_ascending(monkeypatch):
    q = "恒一会 かえで総合病院のtrain.xlsx内のTPのヒストグラムで、2番目にカウント数が少ないビンの範囲を小数第6位まで。"
    counts = [3, 1, 50, 2]                                    # 昇順: 1,2,3 → 2番目に少ない = 2 (idx3)
    cats = ["(0.0, 1.0]", "(1.0, 2.0]", "(2.0, 3.0]", "(3.0, 4.0]"]
    _mock_series(monkeypatch, counts, cats)
    assert formatting.resolve_histogram_bin_direct(q) == "3.000000 ~ 4.000000"


def test_precision_from_question(monkeypatch):
    q = "恒一会 かえで総合病院のtrain.xlsx内のTPのヒストグラムで、1番目にカウント数が多いビンの範囲を小数第2位まで。"
    _mock_series(monkeypatch, [10], ["(1.234567, 2.345678]"])
    assert formatting.resolve_histogram_bin_direct(q) == "1.23 ~ 2.35"


def test_k_beyond_nbins_is_none(monkeypatch):
    _mock_series(monkeypatch, [5], ["(0.0, 1.0]"])           # K=3 だがビンは1個
    assert formatting.resolve_histogram_bin_direct(Q29) is None


def test_empty_series_is_none(monkeypatch):
    _mock_series(monkeypatch, [], [])                         # 曖昧（案件/xlsx/列非一致）
    assert formatting.resolve_histogram_bin_direct(Q29) is None


def test_tie_breaks_on_bin_index(monkeypatch):
    # 同数カウントは bin index 昇順で安定（3番目に多い = 3番目に小さい index の同数ビン）。
    counts = [7, 7, 7, 1]
    cats = ["(0.0, 1.0]", "(1.0, 2.0]", "(2.0, 3.0]", "(3.0, 4.0]"]
    _mock_series(monkeypatch, counts, cats)
    assert formatting.resolve_histogram_bin_direct(Q29) == "2.000000 ~ 3.000000"


def test_fail_open_on_series_exception(monkeypatch):
    def _boom(_q, _c):
        raise RuntimeError("chart read broke")
    monkeypatch.setattr(formatting, "_histogram_series", _boom)
    assert formatting.resolve_histogram_bin_direct(Q29) is None  # 例外は None（答えパスを壊さない）


# --------------------------------------------------------------------------- real-corpus grounding (skips w/o data)
def test_real_corpus_idx29_and_regression_guard():
    """実コーパスで idx29→gold を決定論確定し、idx10（最多カウント型）は None（回帰ゼロ）。corpus 無しなら skip。"""
    try:
        got = formatting.resolve_histogram_bin_direct(Q29)
    except Exception as exc:  # noqa: BLE001
        pytest.skip(f"corpus/xlsx unavailable: {exc}")
    if got is None:
        pytest.skip("train.xlsx chart source not available in this environment")
    assert got == "6.088138 ~ 6.288138"
    assert formatting.resolve_histogram_bin_direct(Q10) is None
