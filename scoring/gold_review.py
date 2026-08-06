"""gold-100 レビュー表 (md/csv) + 実行履歴の自動生成 (SOT-2491).

``scoring.gold_offline`` が gold-100 を採点して :class:`~scoring.gold_offline.Report` を組み立てた
直後にこのモジュールを呼ぶと、人間レビュー用の表を再生成する:

* ``artifacts/gold_100_review.md`` / ``.csv`` — 1 問 1 行。列 =
  ``index, question, gold_v3, system_answer, status(MATCH/ABSTAIN/WRONG), archetype, difficulty,
  state_code, unmet_obligation, reason, confidence``。``state_code`` / ``unmet_obligation`` は
  SOT-2492 の棄権台帳から棄権行にのみ補完（SOT-2497）。**設問全文＋正答を含むため Private 限定**（公開禁止）。
* ``docs/gold_offline_history.jsonl`` — per-run 1 行を追記（上書きしない）。**メトリクスのみ**
  （設問・正答・システム解答を含めない）。原因別棄権集計 ``abstain_state_codes``（コード別カウント・
  SOT-2497）を含むので、改善の効果を run 間でコード別に比較できる。

``reason`` / ``confidence`` は resolve/investigator が書き出す ``*.details.jsonl``
（``status`` / ``reason`` / ``confidence`` 等）から index で補完する。details が無い項目は空欄。

adhoc スクリプトを置き換えるモジュール化なので、``gold_offline`` 以外からも
``generate(report, ...)`` を直接呼べる。
"""
from __future__ import annotations

import csv
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import TYPE_CHECKING

from config import settings

if TYPE_CHECKING:  # 循環 import を避けるため型注釈のみ
    from scoring.gold_offline import Report

# --- 表示ラベル -----------------------------------------------------------------
# verdict バケット -> レビュー表 status。
_STATUS_BY_BUCKET = {"match": "MATCH", "abstain": "ABSTAIN", "wrong": "WRONG"}
# status -> 難しさ列の日本語サフィックス（システム挙動の要約）。
_STATUS_LABEL = {"MATCH": "システム正答", "ABSTAIN": "システム棄権", "WRONG": "システム過信誤り"}

# archetype -> 基本難易度。導出/集計/差分/ハイライト系は「高」、それ以外は「中」。
_HARD_ARCHETYPES = {
    "derived_calculation", "highlight_set", "version_diff", "cross_aggregate",
    "contract_amount", "csv_column_mean", "csv_column_max", "pivot_condition",
}

# 状態コード列 (SOT-2497): 棄権行に SOT-2492 台帳の state_code と未充足義務の要約を表示。MATCH/WRONG は空欄。
_MD_COLUMNS = ["#", "質問", "正答(v3 gold)", "現在の解答(system)", "status", "型", "難しさ",
               "状態コード", "未充足義務", "理由", "確信度"]
_CSV_COLUMNS = ["index", "question", "gold_v3", "system_answer", "status", "archetype",
                "difficulty", "state_code", "unmet_obligation", "reason", "confidence"]

DEFAULT_MD = settings.ARTIFACTS_DIR / "gold_100_review.md"
DEFAULT_CSV = settings.ARTIFACTS_DIR / "gold_100_review.csv"
DEFAULT_HISTORY = settings.REPO_ROOT / "docs" / "gold_offline_history.jsonl"


def _difficulty(archetype: str, status: str) -> str:
    base = "高" if archetype in _HARD_ARCHETYPES else "中"
    return f"{base}（{_STATUS_LABEL.get(status, status)}）"


def _confidence_str(value) -> str:
    """details の confidence を表示用文字列に。float は 2 桁、文字列/None はそのまま。"""
    if value is None or value == "":
        return ""
    if isinstance(value, (int, float)):
        return f"{float(value):.2f}"
    return str(value)


def load_details(path: Path | None) -> dict[int, dict]:
    """``*.details.jsonl`` を ``{index: {answer,status,confidence,reason}}`` に読み込む。

    investigator 経路（``status``/``reason`` 無し）と resolve 経路（両方あり）の双方を許容する。
    ファイルが無い / パスが csv などは空辞書を返す（reason/confidence が空欄になるだけ）。"""
    out: dict[int, dict] = {}
    if path is None or path.suffix.lower() != ".jsonl" or not path.exists():
        return out
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            rec = json.loads(line)
            idx = int(rec["index"])
        except (json.JSONDecodeError, KeyError, TypeError, ValueError):
            continue
        out[idx] = {
            "answer": rec.get("answer", ""),
            "status": rec.get("status", ""),
            "confidence": rec.get("confidence", ""),
            "reason": rec.get("reason", ""),
        }
    return out


def build_rows(report: "Report", details: dict[int, dict] | None = None) -> list[dict]:
    """:class:`Report` の各 Item を CSV 行 dict（``_CSV_COLUMNS`` キー）に変換する。"""
    from scoring.gold_offline import _bucket  # 遅延 import（循環回避）

    details = details or {}
    # SOT-2497: per-index abstain diagnosis ({state_code, obligation}) joined from the abstain ledger.
    abstain_codes = getattr(report, "abstain_codes", None) or {}
    rows: list[dict] = []
    for it in sorted(report.items, key=lambda x: x.index):
        status = _STATUS_BY_BUCKET.get(_bucket(it.verdict), it.verdict)
        d = details.get(it.index, {})
        code = abstain_codes.get(it.index, {})
        rows.append({
            "index": it.index,
            "question": it.question,
            "gold_v3": it.gold,
            "system_answer": it.answer,
            "status": status,
            "archetype": it.archetype,
            "difficulty": _difficulty(it.archetype, status),
            "state_code": str(code.get("state_code", "") or ""),
            "unmet_obligation": str(code.get("obligation", "") or ""),
            "reason": str(d.get("reason", "") or ""),
            "confidence": _confidence_str(d.get("confidence", "")),
        })
    return rows


def _md_cell(text: str) -> str:
    """markdown テーブルセル用に改行/パイプをエスケープ。"""
    return str(text).replace("\\", "＼").replace("|", "\\|").replace("\n", " ").replace("\r", " ")


def render_md(rows: list[dict], report: "Report", gen: str | None = None) -> str:
    d = report.to_dict()
    m, a, w = d["match"]["count"], d["abstain"]["count"], d["wrong"]["count"]
    meets = d["baseline"]["meets"]
    manual = d["baseline"]["manual_match_rate"]
    label = gen or "system"
    header = [
        f"# gold-100 レビュー表（質問／正答(v3)／現在の解答({label})／status／型／難しさ）",
        "",
        "> ⚠️ **Private 限定**: この表は設問全文と正答(v3)を含む。SIGNATE 配布物由来のため公開禁止。",
        "",
        f"system: 一致{m} / 棄権{a} / 誤り{w}"
        f"（手動{manual:.4f}に{'到達' if meets else '未達'}, n={d['n']}, gen={label}）",
        "",
        "| " + " | ".join(_MD_COLUMNS) + " |",
        "|" + "|".join(["---"] * len(_MD_COLUMNS)) + "|",
    ]
    for r in rows:
        cells = [r["index"], r["question"], r["gold_v3"], r["system_answer"], r["status"],
                 r["archetype"], r["difficulty"], r["state_code"], r["unmet_obligation"],
                 r["reason"], r["confidence"]]
        header.append("| " + " | ".join(_md_cell(c) for c in cells) + " |")
    return "\n".join(header) + "\n"


def write_csv(rows: list[dict], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8-sig", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=_CSV_COLUMNS)
        w.writeheader()
        for r in rows:
            w.writerow(r)


def _history_entry(report: "Report", gen: str | None, out_md: Path, out_csv: Path,
                   now: str) -> dict:
    """履歴 1 行 — メトリクスのみ（設問・正答・システム解答は含めない）。"""
    d = report.to_dict()
    by_type = {arch: {"n": b["n"], "match": b["match"], "abstain": b["abstain"],
                      "wrong": b["wrong"], "match_rate": b["match_rate"]}
               for arch, b in d["by_type"].items()}
    return {
        "recordedAt": now,
        "gen": gen or "system",
        "n": d["n"],
        "match": d["match"]["count"],
        "abstain": d["abstain"]["count"],
        "wrong": d["wrong"]["count"],
        "match_rate": d["match"]["rate"],
        "abstain_rate": d["abstain"]["rate"],
        "wrong_rate": d["wrong"]["rate"],
        "cost_usd": d["cost_usd"],
        "baseline_meets": d["baseline"]["meets"],
        "by_type": by_type,
        # SOT-2497: 原因別の棄権集計（メトリクスのみ・設問/正答/解答は非含有）。改善の効果を
        # コード別（NOT_RETRIEVED/EVIDENCE_INCOMPLETE/BUDGET_EXHAUSTED …）の増減で run 間比較できる。
        "abstain_state_codes": d.get("abstain_state_codes", {}),
        "out_md": str(out_md.relative_to(settings.REPO_ROOT))
        if out_md.is_absolute() and _under_repo(out_md) else str(out_md),
        "out_csv": str(out_csv.relative_to(settings.REPO_ROOT))
        if out_csv.is_absolute() and _under_repo(out_csv) else str(out_csv),
    }


def _under_repo(p: Path) -> bool:
    try:
        p.relative_to(settings.REPO_ROOT)
        return True
    except ValueError:
        return False


def append_history(report: "Report", gen: str | None, out_md: Path, out_csv: Path,
                   history_path: Path, now: str | None = None) -> dict:
    """履歴 jsonl に 1 行追記（上書きしない）し、追記した entry を返す。"""
    now = now or datetime.now(timezone.utc).isoformat()
    entry = _history_entry(report, gen, out_md, out_csv, now)
    history_path.parent.mkdir(parents=True, exist_ok=True)
    with history_path.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(entry, ensure_ascii=False) + "\n")
    return entry


def generate(report: "Report", *, details: dict[int, dict] | Path | None = None,
             gen: str | None = None,
             out_md: Path = DEFAULT_MD, out_csv: Path = DEFAULT_CSV,
             history_path: Path = DEFAULT_HISTORY, now: str | None = None) -> dict:
    """レビュー md/csv を再生成し、履歴 jsonl に 1 行追記する。

    ``details`` は details.jsonl パス、読み込み済み dict、または None を受け付ける。
    戻り値は追記した履歴 entry（呼び出し側のログ用）。"""
    if isinstance(details, Path):
        details = load_details(details)
    rows = build_rows(report, details)
    out_md.parent.mkdir(parents=True, exist_ok=True)
    out_md.write_text(render_md(rows, report, gen), encoding="utf-8")
    write_csv(rows, out_csv)
    return append_history(report, gen, out_md, out_csv, history_path, now=now)
