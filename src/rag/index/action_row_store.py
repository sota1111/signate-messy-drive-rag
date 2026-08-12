"""スキャンPDFの action-row（AI/タスク行）構造化抽出ストア（SOT-2652 / 事前計算事実層 追補 — クラスタA）.

Sonnet gold100 cycle4 の abstain のうち idx20/34/70/93（全て simple_lookup / BUDGET_EXHAUSTED, スキャン
PDF 由来）は、証拠が「ページテキスト」ではなく **構造化された行動行フィールド**（ID / 内容 / 担当 / 期日 /
状態 / 出典）であり、file_grep 全走査では 1 発で到達できないことが主因だった
(`docs/ai/sonnet_cycle_analysis/cycle4.md` クラスタA)。cycle3 申し送り「precision を緩めず action-row
フィールド化」を実装する。

本モジュールは build 時に一度だけ、案件×文書ごとに **質問を見ずに全数** で

* **アクション行テーブル** — 会議録の ``ID | Action | Owner | Due Date | Status`` 表（OCR/text-layer とも
  パイプ区切りの同型表）を全行、原文の内容 / 担当 / 期日 / 状態 付きで抽出。
* **優先タスク一覧** — 報告資料の「優先タスク(責任者・期限)」節の ``<内容> (担当：X/Y) →期限：<日付>``
  行を全件、内容（原文そのまま）/ 担当集合 / 期日 付きで抽出。
* **優先フォロー注記** — 報告資料本文の「優先フォロー: AI-03(…)、AI-07(…)」に列挙されたアクションID群
  （報告時点で Open としてフォロー対象に挙げられたもの）。

を属性化し ``artifacts/action_row_store.jsonl`` に焼く。serve path は事前計算済みテキストを読むだけで
**genai 呼び出しゼロ**。読み出し/配線は :mod:`src.rag.agent.action_row_lane`。

Design invariants（sibling の report_attr_store / case_finance_store / ocr_store と同一）
--------------------------------------------------------------------------------------
* **Opt-in at serve time.** :func:`enabled` (``RAG_ACTION_ROW_STORE``) が runtime 参照のみを gate する。
  default OFF ⇒ champion serve path は byte-identical。
* **Build は LLM フリー・追加的.** 参照は既存の永続 OCR（前処理で確定済み）と text-layer/extract のみ。
  build で新たな genai 呼び出しはしない。読めない文書は 1 件スキップし build は継続（fail-open）。
* **Question-independent.** universe は構造的（会議/報告 category の pdf/docx/pptx）。質問も gold も参照せず、
  表の全行・全タスクを網羅抽出する。値は毎セル出典（doc/page）付き。
* **Fail-open.** artifact 欠落・解析不能はすべて空へフォールバック（回帰ゼロ）。
"""
from __future__ import annotations

import json
import os
import re
import unicodedata
from pathlib import Path
from typing import Any, Mapping, Sequence

from config import settings
from src.rag.corpus import FileRef, nfc, walk

SCHEMA = "action-row-store"
SCHEMA_VERSION = 1

_ON = {"1", "true", "yes", "on"}

# 会議/報告 文書のみを対象（05.会議 = meeting, 06.報告書 = report）。
_DOC_CATEGORIES = {"meeting", "report"}
_DOC_EXTS = {"pdf", "docx", "pptx"}

_PAGE_MARK = re.compile(r"\[ページ\s*(\d+)\]")
_DATE_IN_NAME = re.compile(r"(\d{4})-(\d{2})-(\d{2})")

# アクションID: AI-01 / A10 / A-10 / M01 / T09 等（区切り非依存で正規化してから比較）。
_ACTION_ID = re.compile(r"\b([A-Za-z]{1,3}-?\d{1,3})\b")

# パイプ区切りのアクション行テーブル: "AI-05 | 内容 | 担当 | 2025-05-20 | Open(…)"。
# 先頭セル=ID, 末尾セル=状態, 内部に日付セルがあれば期日。列数は 3〜6 を許容。
_PIPE_ROW = re.compile(r"^\s*([A-Za-z]{1,3}-?\d{1,3})\s*[|｜]\s*(.+)$")

# 優先タスク行: "<内容> (担当：渡辺遥/藤田) →期限：2025-08-21"。担当は最後の "(担当" を採用（内容側の
# 括弧を誤検出しないため）。期限は全角/半角コロン・任意の矢印を許容。
_TASK_ROW = re.compile(
    r"[(（]\s*担当\s*[：:]\s*([^)）]+?)\s*[)）]\s*(?:[→\-–—ー]+\s*)?期限\s*[：:]\s*"
    r"(\d{4}\s*-\s*\d{2}\s*-\s*\d{2})")

_STATUS_DONE = re.compile(r"clos|完了|済|done|resolved", re.IGNORECASE)
_STATUS_OPEN = re.compile(r"open|未完了|未着手|進行中|フォロー", re.IGNORECASE)

_PRIORITY_FOLLOW_CUE = re.compile(r"優先フォロー")
_OWNER_SPLIT = re.compile(r"[/／、,，・]")


def enabled() -> bool:
    """True when the serve path may consult the action-row store (default OFF — opt-in)."""
    return os.getenv("RAG_ACTION_ROW_STORE", "0").strip().lower() in _ON


def default_out_path() -> Path:
    return settings.ARTIFACTS_DIR / "action_row_store.jsonl"


def default_report_path() -> Path:
    return settings.ARTIFACTS_DIR / "action_row_store_build_report.json"


def norm_id(value: Any) -> str:
    """アクションIDを区切り非依存の比較キーへ（'AI-05'='ai05', 'A10'='a10'）。"""
    return re.sub(r"[\s\-_]", "", unicodedata.normalize("NFKC", str(value or ""))).lower()


def _z(s: str) -> str:
    return unicodedata.normalize("NFKC", s or "")


def _doc_text(ref: FileRef) -> str:
    """文書本文: スキャンPDF は永続 OCR、その他は text-layer/extract。genai 呼び出しはしない。"""
    try:
        from src.rag.index import ocr_store
        ocr = ocr_store.lookup(ref)
        if ocr and ocr.strip():
            return nfc(ocr)
    except Exception:  # noqa: BLE001
        pass
    try:
        from src.rag.extract import extract
        doc = extract(ref, caption_images=False)
        return nfc(getattr(doc, "text", "") or "")
    except Exception:  # noqa: BLE001
        return ""


def _doc_date(ref: FileRef) -> str | None:
    m = _DATE_IN_NAME.search(nfc(ref.name))
    return f"{m.group(1)}-{m.group(2)}-{m.group(3)}" if m else None


# --------------------------------------------------------------------------- extractors
def _pages(text: str) -> list[tuple[int, str]]:
    """テキストを [ページN] マーカで分割 → [(page, chunk)]（マーカ無しは page=0）。"""
    out: list[tuple[int, str]] = []
    page = 0
    buf: list[str] = []
    for line in text.splitlines():
        pm = _PAGE_MARK.search(line)
        if pm:
            if buf:
                out.append((page, "\n".join(buf)))
                buf = []
            page = int(pm.group(1))
            continue
        buf.append(line)
    if buf:
        out.append((page, "\n".join(buf)))
    return out


def _status_kind(status_text: str) -> str:
    """状態セルの語から done/open/unknown を判定（完了扱いを保守的に）。"""
    s = _z(status_text)
    if _STATUS_DONE.search(s):
        return "done"
    if _STATUS_OPEN.search(s):
        return "open"
    return "unknown"


def _pipe_rows(text: str, doc_rel: str) -> list[dict[str, Any]]:
    """パイプ区切りのアクション行テーブルを全行抽出（ID/内容/担当/期日/状態/ページ）。"""
    rows: list[dict[str, Any]] = []
    for page, chunk in _pages(text):
        for raw in chunk.splitlines():
            line = _z(raw).strip()
            m = _PIPE_ROW.match(line)
            if not m:
                continue
            cells = [c.strip() for c in re.split(r"[|｜]", line)]
            if len(cells) < 3:
                continue
            action_id = cells[0]
            status = cells[-1]
            # 期日: 末尾から2番目以前に日付セルがあれば採用。
            due = None
            for c in cells[1:-1]:
                dm = _DATE_IN_NAME.search(c)
                if dm:
                    due = f"{dm.group(1)}-{dm.group(2)}-{dm.group(3)}"
            # 内容 = ID の次セル、担当 = 期日セルの手前（推定）。日付を含むセルは担当から除外。
            middle = cells[1:-1]
            content = middle[0] if middle else ""
            owner = ""
            for c in reversed(middle[1:]):
                if not _DATE_IN_NAME.search(c) and c:
                    owner = c
                    break
            rows.append({
                "id": action_id, "id_key": norm_id(action_id), "content": content,
                "owner": owner, "due": due, "status": status,
                "status_kind": _status_kind(status),
                "source": {"doc_id": doc_rel, "page": page},
            })
    return rows


def _priority_tasks(text: str, doc_rel: str) -> list[dict[str, Any]]:
    """報告資料の「優先タスク」行 ``<内容> (担当：X/Y) →期限：日付`` を全件抽出（内容は原文のまま）。"""
    tasks: list[dict[str, Any]] = []
    for page, chunk in _pages(text):
        # ``(担当：X)`` と期限日付は行またぎ（渡辺\n遥 / 2025-\n08-21）を許容するが、内容の境界は
        # 行構造で決めるため改行を潰さず走査する（潰すと項目/内容の切れ目が消える）。
        for m in _TASK_ROW.finditer(chunk):
            owners_raw = _z(m.group(1))
            due = re.sub(r"\s", "", m.group(2))
            content = _task_content(chunk[:m.start()])
            if not content:
                continue
            owners = [re.sub(r"\s+", "", o) for o in _OWNER_SPLIT.split(owners_raw) if o.strip()]
            tasks.append({
                "content": content, "owners": owners,
                "owner_keys": [_owner_key(o) for o in owners],
                "due": due, "source": {"doc_id": doc_rel, "page": page},
            })
    return tasks


# 箇条書きの先頭（番号・記号）を落として内容フレーズだけを取り出す。
_LEADING_BULLET = re.compile(r"^(?:[0-9０-９]+\s*[.\.、）)]\s*|[・*※\-–—●○◦▪◆]\s*)+")


def _task_content(head: str) -> str:
    """"(担当" 直前テキストから当該タスクの内容フレーズを取り出す（同一行末尾＝内容, 箇条番号を除去）。

    ``(担当`` は内容と同じ行にあるため、head の最終行が内容行。空なら直前の非空行へ遡る（見出し行を除く）。
    """
    lines = head.split("\n")
    for pos in range(len(lines) - 1, -1, -1):
        ln = lines[pos]
        frag = _LEADING_BULLET.sub("", ln.strip()).strip()
        # 見出し（「優先タスク...」）や WBS 括弧注記だけの行はスキップして本文行へ。
        if not frag or frag.startswith("優先タスク") or re.fullmatch(r"[(（][^)）]*[)）]", frag):
            continue
        # Scanned report watermarks occasionally bleed into the first table row.  The
        # 東都 PDF is one such case: OCR emits ``データ分析計画書...`` although the
        # visible row (and its WBS T03 source) is ``分析計画書...``.  Only remove the
        # known watermark fragment when this is the unnumbered row immediately below
        # the priority-task heading; numbered rows and ordinary ``データ...`` tasks
        # remain untouched.
        prior = [x.strip() for x in lines[:pos] if x.strip()]
        if (frag.startswith("データ") and not _LEADING_BULLET.match(ln.strip())
                and prior and prior[-1].startswith("優先タスク")):
            frag = frag[len("データ"):]
        return frag if len(frag) <= 120 else ""
    return ""


# 行頭アンカ付きの節見出し（"3. 主要な分析結果"）— 列挙範囲の終端。日付中の "26、" 等には誤反応しない。
_SECTION_HEAD = re.compile(r"(?m)^\s*[0-9０-９]{1,2}\s*[.\.]\s*\S")


def _priority_follow(text: str, doc_rel: str) -> list[dict[str, Any]]:
    """報告本文の「優先フォロー: AI-03(…)、AI-07(…)」列挙のアクションID群を抽出（状態注記付き）。"""
    out: list[dict[str, Any]] = []
    for page, chunk in _pages(text):
        for cue in _PRIORITY_FOLLOW_CUE.finditer(chunk):
            # 列挙は「優先フォロー」直後から次の行頭番号見出しまで（行構造で切る）。
            rest = chunk[cue.end():]
            stop = _SECTION_HEAD.search(rest)
            span = rest[: stop.start()] if stop else rest[:600]
            seen: set[str] = set()
            for idm in _ACTION_ID.finditer(span):
                aid = idm.group(1)
                key = norm_id(aid)
                if not re.match(r"[A-Za-z]", aid) or key in seen:
                    continue
                seen.add(key)
                # ID 直後の括弧注記内の status: 語（Open/Closed）を拾う（改行込み）。
                near = span[idm.end(): idm.end() + 60]
                out.append({"id": aid, "id_key": key, "status_in_report": _status_kind(near),
                            "source": {"doc_id": doc_rel, "page": page}})
    return out


def compute_doc(ref: FileRef) -> dict[str, Any] | None:
    """1文書から action-row レコードを組む（何も抽出できなければ None — 欠測を偽装しない）。"""
    text = _doc_text(ref)
    if not text.strip():
        return None
    doc_rel = nfc(ref.rel)
    rows = _pipe_rows(text, doc_rel)
    tasks = _priority_tasks(text, doc_rel)
    follow = _priority_follow(text, doc_rel)
    if not (rows or tasks or follow):
        return None
    rec: dict[str, Any] = {
        "doc_id": doc_rel, "project": nfc(ref.project),
        "category": ref.category, "date": _doc_date(ref),
        "doc_name": nfc(ref.name),
    }
    if rows:
        rec["action_rows"] = rows
    if tasks:
        rec["priority_tasks"] = tasks
    if follow:
        rec["priority_follow"] = follow
    return rec


# --------------------------------------------------------------------------- universe / build / io
def _universe(refs: Sequence[FileRef]) -> list[FileRef]:
    out: list[FileRef] = []
    for r in refs:
        if r.category not in _DOC_CATEGORIES or r.ext not in _DOC_EXTS:
            continue
        name = nfc(r.name).lower()
        if name.startswith("~$") or "old" in name:
            continue
        out.append(r)
    out.sort(key=lambda r: r.rel)
    return out


def write_store(records: Sequence[Mapping[str, Any]], path: Path | None = None) -> dict[str, int]:
    """Atomically write a reproducible JSONL store (schema header + doc_id-sorted rows)."""
    out = Path(path) if path is not None else default_out_path()
    out.parent.mkdir(parents=True, exist_ok=True)
    ordered = sorted(records, key=lambda r: r.get("doc_id", ""))
    tmp = out.with_suffix(out.suffix + ".tmp")
    with tmp.open("w", encoding="utf-8", newline="\n") as handle:
        handle.write(json.dumps({"schema": SCHEMA, "version": SCHEMA_VERSION},
                                ensure_ascii=False, sort_keys=True) + "\n")
        for rec in ordered:
            handle.write(json.dumps(rec, ensure_ascii=False, sort_keys=True) + "\n")
    tmp.replace(out)
    reset_cache()
    return {"docs": len(ordered)}


def build(refs: Sequence[FileRef] | None = None, *, out: Path | None = None,
          write_report: bool = True) -> dict[str, Any]:
    """会議/報告 文書を全走査して action-row ストアを焼く（質問非依存・LLM フリー・fail-open）。"""
    refs = list(refs) if refs is not None else list(walk())
    universe = _universe(refs)
    records: list[dict[str, Any]] = []
    skipped: list[str] = []
    for ref in universe:
        try:
            rec = compute_doc(ref)
        except Exception:  # noqa: BLE001 — one bad doc must not sink the build
            rec = None
        if rec is not None:
            records.append(rec)
        else:
            skipped.append(nfc(ref.rel))
    stats = write_store(records, out)
    report = {
        "schema": SCHEMA, "version": SCHEMA_VERSION,
        "universe": len(universe), "records": len(records), "skipped": skipped,
        "action_rows": sum(len(r.get("action_rows", [])) for r in records),
        "priority_tasks": sum(len(r.get("priority_tasks", [])) for r in records),
        "priority_follow": sum(len(r.get("priority_follow", [])) for r in records),
    }
    if write_report:
        rp = default_report_path()
        rp.parent.mkdir(parents=True, exist_ok=True)
        rp.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    return {"records": len(records), "docs": stats["docs"], "report": report}


# --------------------------------------------------------------------------- load / read (minimal API)
_LOAD_CACHE: dict[str, list[dict[str, Any]]] = {}


def reset_cache() -> None:
    _LOAD_CACHE.clear()


def load(path: Path | None = None) -> list[dict[str, Any]]:
    """Load the action-row rows (memoized). ``[]`` when absent/unreadable/schema-mismatch (回帰ゼロ)."""
    out = Path(path) if path is not None else default_out_path()
    key = str(out)
    if key in _LOAD_CACHE:
        return _LOAD_CACHE[key]
    rows: list[dict[str, Any]] = []
    try:
        with open(out, encoding="utf-8") as handle:
            header = handle.readline()
            meta = json.loads(header) if header.strip() else {}
            if isinstance(meta, dict) and meta.get("schema") == SCHEMA \
                    and meta.get("version") == SCHEMA_VERSION:
                for line in handle:
                    line = line.strip()
                    if line:
                        rows.append(json.loads(line))
    except Exception:  # noqa: BLE001
        rows = []
    _LOAD_CACHE[key] = rows
    return rows


def docs_for_project(project_hint: str, *, path: Path | None = None) -> list[dict[str, Any]]:
    """案件名(一部一致可)にマッチする文書レコード群（コーパス順）。"""
    hint = _owner_key(project_hint)
    rows = load(path)
    if not hint:
        return list(rows)
    return [r for r in rows if hint in _owner_key(r.get("project", ""))
            or _owner_key(r.get("project", "")) in hint]


# 案件名/担当名の識別キー（法人格・空白・ケースを吸収）。法人格は接頭・接尾どちらも除去
# （"白峰信用リスク評価株式会社" → "白峰信用リスク評価", 質問の略称に一致させる）。
_CORP = r"(?:株式会社|医療法人社団|一般社団法人|一般財団法人|有限会社|合同会社|合資会社)"
_CORP_PREFIX = re.compile(rf"^{_CORP}\s*")
_CORP_SUFFIX = re.compile(rf"\s*{_CORP}$")


def _owner_key(value: Any) -> str:
    s = unicodedata.normalize("NFKC", str(value or ""))
    s = _CORP_PREFIX.sub("", s)
    s = _CORP_SUFFIX.sub("", s)
    return re.sub(r"[\s　]", "", s).lower()


if __name__ == "__main__":
    summary = build()
    print(json.dumps(summary["report"], ensure_ascii=False, indent=2))
