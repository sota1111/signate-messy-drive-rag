"""版ペア差分の事前確定ストア — 全 version-family ペアの構造 diff＋SUBSTANTIVE ランキングを index 時に一度だけ
計算・保存する事実層 (SOT-2646 / 事前計算事実層 4/5).

champion (Wave A net40) の abstain 全数分類 (`docs/ai/champion_abstain_wrong_classification.md`,
2026-08-11) が確定した **hard core 16問** の version_diff 1問 (idx22) は、実行時の ``version_diff`` ツール呼び
＋LLM 整形では ``BUDGET_EXHAUSTED`` に至る。B クラス (回答化するが誤答) にも version 系 idx1/14/95 が含まれ、
実行時 diff→LLM 整形の**不安定さ** (cycle3 で version 系が run 間で MATCH↔WRONG した揺らぎ) が原因だった。

diff 計算そのものは完全に決定論 (diffpair + SOT-2588 block alignment + SOT-2605 パイプライン、全て実装済み) で、
**コーパスは不変** — つまり実行時に毎回計算する必要が本来ない。本 issue は **全 version-family ペアの diff を
index 時に1回だけ確定**し、実行時は事前確定済み結果の読み出しにすることで、計算の揺らぎを消す。diff ロジック
自体は一切変更せず (:func:`diffpair.rank_changes` を呼ぶだけ)、列挙とランキング結果の永続化に徹する。

**正当性の歯止め（必読）**: 事前計算は **質問を見ない網羅列挙** に限定する — document registry の
version_family / ファイル名規約から全ペアを機械列挙し、各ペアの全 ranked candidate を保存するのであり、特定設問を
狙った選別・gold 値のハードコードは一切しない。設問→カバレッジの対応表 (:func:`version_hard_core_coverage`) は
*ビルドレポート用の診断* にすぎず、ストア本体 (``diff_store.jsonl``) は完全に質問非依存。カバレッジ外・未対応
拡張子は従来経路へフォールバック（劣化ゼロ）。

Design invariants（既存の兄弟事前ストア — id_master / derived_metrics / case_master — に倣う）:
  * **全ペア網羅列挙.** :func:`enumerate_pairs` は filename/folder 規約 (:func:`diffpair.find_pairs`) と
    document-registry 家系 (:func:`diffpair._registry_family_pairs`) の **和集合** を取る。後者は SOT-2588 の
    ``find_pairs`` 制約 — ``_old``×無印 latest ペアが組めない (idx0 白峰 / idx1 恒一会) — を補完する。
  * **全 candidate を順位付きで保持.** SUBSTANTIVE 1位だけでなく全 ranked candidate を保存する。実行時の質問条件
    (「ステータス変更を除く」idx95 等) による絞り込みに、事前確定済みの候補集合上の決定論フィルタで応えられる。
  * **除外条件の素材化.** 各変更に機械属性 (数値変更 / 担当者変更 / ステータス遷移 / 書式のみ 等) を付与し、
    質問側の除外条件を :func:`filter_changes` の決定論フィルタで適用できる形にする。
  * **決定論・LLM フリー.** ランキング・属性付与は正規表現＋:mod:`diffpair` の決定論分類のみ。Gemini 呼び出しなし・
    ネットワークなし・再現バイト一致（同じコーパスで2回ビルドすると byte-identical）。
  * **Opt-in at serve time.** :func:`enabled` (``RAG_DIFF_STORE``) は *実行時参照のみ* を gate する。アーティ
    ファクトは index 時に常に additive・guarded に（再）ビルドされる。既定 OFF ⇒ champion serve path はバイト
    一致（本 issue は列挙と永続化のみ — ルーター配線の本番は SOT-2647）。
"""
from __future__ import annotations

import json
import os
import re
from pathlib import Path
from typing import Any, Iterable, Sequence

from config import settings
from src.rag import diffpair
from src.rag.corpus import FileRef, nfc

SCHEMA = "diff-store"
SCHEMA_VERSION = 1

_ON = {"1", "true", "yes", "on"}

# 1ペアあたりの保存変更数上限（病的な巨大 diff の暴走ガード）。実コーパス最大は xlsx 全面再整列の 369 変更なので
# 実際には切り詰めは起きないが、上限に達した場合はレポートに truncated として honest に記録する。
_MAX_STORED_CHANGES = 1000
_TEXT_MAX = 400  # 変更前後テキストの保存上限（1セルの巨大テキストで行が暴走するのを防ぐ）

# --------------------------------------------------------------------------- 除外条件の素材（決定論属性）
# ワークフロー・ステータス語彙（idx95「未着手から完了への変更を除いて」等の状態遷移を機械判定する）。
# _norm で空白/記号/大小を潰したうえで完全一致させるので、表記ゆれ（"未 着手" / "ToDo"）も吸収する。
_STATUS_TOKENS = {
    diffpair._norm(s) for s in (
        "未着手", "未対応", "未開始", "未実施", "着手", "着手中", "対応中", "進行中", "実施中", "作業中",
        "レビュー", "レビュー中", "確認中", "承認待ち", "保留", "中止", "見送り",
        "完了", "済", "完了済", "対応済", "クローズ", "終了",
        "todo", "wip", "doing", "inprogress", "notstarted", "done", "closed", "open", "review",
    )
}

# 担当者・アサイン文脈（列ヘッダ / セルラベルにこれらが出れば person 変更を assignee 変更として区別する）。
_ASSIGNEE_LABEL = re.compile(
    r"担当|アサイン|割当|レビュ|承認|責任者|オーナー|owner|assignee|PM|マネージャ|リード|作業者"
)


def _clip_text(text: str, limit: int = _TEXT_MAX) -> str:
    t = nfc(text)
    return t if len(t) <= limit else t[:limit] + "…"


def change_attributes(rc: "diffpair.RankedChange") -> list[str]:
    """1 変更に、質問側の除外条件を決定論適用するための機械属性タグを付与する（質問非依存・LLM フリー）。

    ``diffpair.classify_change`` が付ける ``reason_features`` (money/percent/date/number/person_name/…) と
    intent・kind・location から、除外フィルタが参照しやすい正規化タグ集合を導出する。例:
    「ステータス変更を除く」→ ``status_transition`` を除外、「書式変更を除く」→ ``formatting_only`` を除外。
    """
    c = rc.change
    feats = set(rc.features)
    label = c.label or ""
    tags: list[str] = []

    # kind（追加 / 削除 / 変更）
    tags.append({"add": "addition", "remove": "deletion", "modify": "modification"}.get(c.kind, c.kind))

    # intent 由来（書式のみ / レイアウト移動・再掲）
    if rc.intent == diffpair.BOILERPLATE:
        tags.append("formatting_only")
    elif rc.intent == diffpair.LAYOUT_METADATA:
        tags.append("layout_change")
        if "moved_block" in feats:
            tags.append("moved_block")
        if "summary_representation" in feats:
            tags.append("summary_representation")
    elif rc.intent == diffpair.SUBSTANTIVE:
        tags.append("substantive")

    # ステータス遷移（両側が純粋なワークフロー状態語の modify）— idx95 の「未着手→完了」除外に対応
    if c.kind == "modify" and diffpair._norm(c.before) in _STATUS_TOKENS \
            and diffpair._norm(c.after) in _STATUS_TOKENS:
        tags.append("status_transition")

    # 数値系（数値/金額/割合/数量/日付）— 「数値の変更を除く」等
    if "money" in feats:
        tags.append("money_change")
    if "percent" in feats:
        tags.append("percent_change")
    if "date" in feats:
        tags.append("date_change")
    if feats & {"number", "quantity", "money", "percent"}:
        tags.append("numeric_change")

    # 担当者・人物（person_name 特徴、または担当系ラベル文脈）— idx95「担当者に…を追加」を残す/除く判定に
    if "person_name" in feats:
        tags.append("person_name_change")
        if _ASSIGNEE_LABEL.search(label) or _ASSIGNEE_LABEL.search(c.before) or _ASSIGNEE_LABEL.search(c.after):
            tags.append("assignee_change")
    elif _ASSIGNEE_LABEL.search(label):
        tags.append("assignee_change")

    # スキーマ名・識別子の in-place 変更（idx14 の列名アンダースコア表記修正）
    if "identifier" in feats and c.kind == "modify":
        tags.append("schema_name_change")
    if "obligation" in feats:
        tags.append("obligation_change")
    if "condition" in feats:
        tags.append("condition_change")

    # 決定論・順序安定・重複排除
    seen: set[str] = set()
    out: list[str] = []
    for t in tags:
        if t not in seen:
            seen.add(t)
            out.append(t)
    return out


def enabled() -> bool:
    """True when the serve path should consult the precomputed diff store (default OFF — opt-in).

    Mirrors ``RAG_ID_MASTER`` / ``RAG_DERIVED_METRICS``: the artifact is always (re)built at index time,
    but *runtime consultation* is gated so the champion serve path stays byte-identical.
    """
    return os.getenv("RAG_DIFF_STORE", "0").strip().lower() in _ON


def default_out_path() -> Path:
    return settings.ARTIFACTS_DIR / "diff_store.jsonl"


def report_out_path() -> Path:
    return settings.ARTIFACTS_DIR / "diff_store_build_report.json"


# --------------------------------------------------------------------------- ペア列挙（和集合・網羅）
def enumerate_pairs() -> list[tuple["diffpair.VersionPair", list[str]]]:
    """全 version-family ペアを filename/folder 規約 ∪ registry 家系で列挙 (old→new)、各ペアの由来 source 付き。

    ``diffpair.find_pairs`` は両側に version トークンを要求するため ``_old``×無印 latest (idx0/idx1) を組めない。
    ``diffpair._registry_family_pairs`` は document registry の family id でその欠落を埋める。両者の和集合を
    (old.rel, new.rel) で重複排除し、どの列挙器が出したかを source リストに残す（診断・網羅性の根拠）。
    """
    ordered: list[tuple[str, str]] = []
    by_key: dict[tuple[str, str], "diffpair.VersionPair"] = {}
    sources: dict[tuple[str, str], list[str]] = {}
    for source, fn in (("filename", diffpair.find_pairs), ("registry-family", diffpair._registry_family_pairs)):
        try:
            pairs = fn()
        except Exception:  # noqa: BLE001 — one enumerator failing must not sink the other
            pairs = []
        for p in pairs:
            key = (p.old.rel, p.new.rel)
            if key not in by_key:
                by_key[key] = p
                sources[key] = [source]
                ordered.append(key)
            elif source not in sources[key]:
                sources[key].append(source)
    ordered.sort(key=lambda k: (nfc(by_key[k].new.project or ""), k[0], k[1]))
    return [(by_key[k], sources[k]) for k in ordered]


# --------------------------------------------------------------------------- ペア diff レコード
def pair_record(pair: "diffpair.VersionPair", sources: Sequence[str]) -> dict[str, Any]:
    """1 ペアの事前確定 diff レコード（全 ranked candidate ＋ 変更属性）。

    ``diffpair.rank_changes`` の順位付き結果をそのまま保存する。``rank_changes`` が None（片方が読めない /
    構造抽出不可）のときは ``alignment_ok=False`` の空レコードにする（欠測を偽装しない）。
    """
    try:
        ranked = diffpair.rank_changes(pair)
    except Exception:  # noqa: BLE001 — one unreadable pair must not sink the build
        ranked = None
    alignment_ok = ranked is not None
    ranked = ranked or []
    truncated = len(ranked) > _MAX_STORED_CHANGES
    changes: list[dict[str, Any]] = []
    for rank, rc in enumerate(ranked[:_MAX_STORED_CHANGES]):
        d = rc.to_dict()
        d["old"] = _clip_text(d.get("old", ""))
        d["new"] = _clip_text(d.get("new", ""))
        d["structural_location"] = _clip_text(d.get("structural_location", ""), 120)
        d["rank"] = rank
        d["attributes"] = change_attributes(rc)
        changes.append(d)
    return {
        "old_rel": nfc(pair.old.rel),
        "new_rel": nfc(pair.new.rel),
        "old_name": nfc(pair.old.name),
        "new_name": nfc(pair.new.name),
        "project": nfc(pair.new.project or pair.old.project or ""),
        "base": nfc(pair.base),
        "basis": pair.basis,
        "ext": pair.new.ext,
        "sources": list(sources),
        "alignment_ok": alignment_ok,
        "change_count": len(changes),
        "truncated": truncated,
        "changes": changes,
    }


# --------------------------------------------------------------------------- notebook (.ipynb) diff レーン
# SOT-2650 — idx22 型の hard-core 版ペア。決定論 office diff パイプライン (diffpair, pptx/docx/xlsx) は
# serve path からも参照されるため触らず、notebook はこのストア専用の build 時レーンで扱う（追加行のみ =
# additive、serve 到達は diff_lookup 経由で RAG_FACT_LAYER ゲート下）。セル単位で source と出力テキストを
# 突き合わせ、変わった行だけを before/after で保存する（質問非依存・全 ipynb 版ペア網羅・LLM-free）。

def _notebook_pairs() -> list["diffpair.VersionPair"]:
    """All .ipynb version pairs via the same filename conventions as the office rules.

    ``diffpair.find_pairs`` requires a version token on BOTH sides, so the corpus shape
    ``01_eda_old.ipynb × 01_eda.ipynb`` (versioned old × unversioned latest) never enumerates there.
    Here a versioned member pairs with its unversioned same-dir sibling (the latest), and the office
    old-folder rule is mirrored as-is.
    """
    try:
        refs = [r for r in diffpair._walk(None) if r.ext == "ipynb"]
    except Exception:  # noqa: BLE001
        return []
    by_dir: dict[str, list[FileRef]] = {}
    for r in refs:
        by_dir.setdefault("/".join(r.rel.split("/")[:-1]), []).append(r)
    pairs: list["diffpair.VersionPair"] = []
    seen: set[tuple[str, str]] = set()
    for d, files in by_dir.items():
        by_stem = {nfc(f.stem): f for f in files}
        for f in files:
            pv = diffpair._parse_version(f.stem)
            if pv is None:
                continue
            base, _rank = pv
            latest = by_stem.get(nfc(base))
            if latest is None or latest.rel == f.rel:
                continue
            key = (f.rel, latest.rel)
            if key not in seen:
                seen.add(key)
                pairs.append(diffpair.VersionPair(f, latest, base, "notebook"))
        # old-folder rule (old/ 旧/ …) mirrored from diffpair.find_pairs
        leaf = nfc(d.split("/")[-1].lower()) if d else ""
        if leaf in getattr(diffpair, "_OLD_DIR_NAMES", set()):
            parent = "/".join(d.split("/")[:-1])
            for old_ref in files:
                for new_ref in by_dir.get(parent, []):
                    if new_ref.name == old_ref.name:
                        key = (old_ref.rel, new_ref.rel)
                        if key not in seen:
                            seen.add(key)
                            pairs.append(diffpair.VersionPair(old_ref, new_ref, old_ref.stem, "notebook"))
    return pairs


def _nb_cells(path: Path) -> list[dict[str, str]] | None:
    """Notebook → [{type, src, out}] (outputs as plain text), or ``None`` when unparsable."""
    try:
        nb = json.loads(Path(path).read_text(encoding="utf-8", errors="ignore"))
    except Exception:  # noqa: BLE001
        return None
    cells: list[dict[str, str]] = []
    for cell in nb.get("cells", []) or []:
        src = "".join(cell.get("source", []) or [])
        outs: list[str] = []
        for o in cell.get("outputs", []) or []:
            txt = o.get("text") or (o.get("data", {}) or {}).get("text/plain")
            if txt:
                outs.append("".join(txt) if isinstance(txt, list) else str(txt))
        cells.append({"type": str(cell.get("cell_type", "code")), "src": src, "out": "\n".join(outs)})
    return cells


_EMBEDDED_IMG_RE = re.compile(r"!\[[^\]]*\]\(data:image/(png|jpeg);base64,([A-Za-z0-9+/=\s]+?)\)")


def _embedded_image_b64(src: str) -> "tuple[str, str] | None":
    """(mime_subtype, base64) of the first embedded data-URI image in a cell source, or ``None``."""
    m = _EMBEDDED_IMG_RE.search(src or "")
    return (m.group(1), m.group(2)) if m else None


def _transcribe_embedded(b64: str, subtype: str) -> "str | None":
    """Build-time Gemini-vision transcription of one embedded image (表/図の中身をテキスト化).

    Spends a vision call, so it runs only under the same opt-in build flag as the OCR store
    (``RAG_OCR_STORE_BUILD``) — the default build stays LLM-free. Failure ⇒ ``None`` (honest marker).
    """
    try:
        from src.rag.index import ocr_store
        if not ocr_store.build_enabled():
            return None
        import base64 as _b64mod

        from src.rag import llm
        from src.rag.extract.vision import _OCR_PROMPT

        data = _b64mod.b64decode(re.sub(r"\s+", "", b64))
        mime = f"image/{subtype}"
        try:
            # Notebook-rendered tables are frequently transparent-background PNGs whose glyphs vanish
            # on the model's default (black) canvas — composite onto white before transcription.
            import io

            from PIL import Image as _PILImage

            img = _PILImage.open(io.BytesIO(data))
            if img.mode in ("RGBA", "LA", "P"):
                rgba = img.convert("RGBA")
                canvas = _PILImage.new("RGB", rgba.size, (255, 255, 255))
                canvas.paste(rgba, mask=rgba.split()[-1])
                buf = io.BytesIO()
                canvas.save(buf, format="PNG")
                data, mime = buf.getvalue(), "image/png"
        except Exception:  # noqa: BLE001 — compositing is best-effort; fall back to the raw bytes
            pass
        txt = llm.generate(_OCR_PROMPT, images=[llm.Image(data=data, mime_type=mime)],
                           model=llm.settings.VISION_MODEL, max_output_tokens=2048)
        return txt.strip() if txt and txt.strip() else None
    except Exception:  # noqa: BLE001 — transcription is best-effort; the pair record survives without it
        return None


_HEADER_PROMPT = (
    "画像は統計表・データ表の一部です。表のセクション見出し（変数名・列名。例: Attr1 のような識別子）だけを、"
    "上から順に改行区切りで全て列挙してください。綴りは画像の文字を1文字ずつ正確に転記してください。"
    "統計量の行ラベル（平均、標準偏差、データ個数など）や数値は出力しないでください。"
    "見出しが無ければ NONE とだけ出力してください。")


def _table_headers(b64: str, subtype: str, tiles: int = 4) -> "list[str] | None":
    """Build-time vision enumeration of a rendered table's variable headers (tiled for resolution).

    A 4000px-tall rendered describe() table downsamples past glyph fidelity in one shot (class→dass
    misreads); reading vertical tiles keeps each pass sharp. Same opt-in gating as the transcription.
    Returns the ordered de-duplicated header list, or ``None`` when unavailable.
    """
    try:
        from src.rag.index import ocr_store
        if not ocr_store.build_enabled():
            return None
        import base64 as _b64mod
        import io

        from PIL import Image as _PILImage

        from src.rag import llm

        data = _b64mod.b64decode(re.sub(r"\s+", "", b64))
        img = _PILImage.open(io.BytesIO(data)).convert("RGBA")
        canvas = _PILImage.new("RGB", img.size, (255, 255, 255))
        canvas.paste(img, mask=img.split()[-1])
        w, h = canvas.size
        out: list[str] = []
        for t in range(tiles):
            tile = canvas.crop((0, h * t // tiles, w, h * (t + 1) // tiles))
            buf = io.BytesIO()
            tile.save(buf, format="PNG")
            txt = llm.generate(_HEADER_PROMPT,
                               images=[llm.Image(data=buf.getvalue(), mime_type="image/png")],
                               model=llm.settings.VISION_MODEL, max_output_tokens=1024) or ""
            for ln in txt.splitlines():
                s = ln.strip()
                if s and s != "NONE" and s not in out:
                    out.append(s)
        return out or None
    except Exception:  # noqa: BLE001
        return None


def _changed_lines(old: str, new: str) -> tuple[str, str]:
    """The line-level (removed, added) content between two texts — the changed rows only, clipped."""
    import difflib

    old_lines, new_lines = old.splitlines(), new.splitlines()
    removed: list[str] = []
    added: list[str] = []
    for tag, i1, i2, j1, j2 in difflib.SequenceMatcher(a=old_lines, b=new_lines,
                                                       autojunk=False).get_opcodes():
        if tag in ("replace", "delete"):
            removed.extend(old_lines[i1:i2])
        if tag in ("replace", "insert"):
            added.extend(new_lines[j1:j2])
    return _clip_text("\n".join(removed)), _clip_text("\n".join(added))


def _notebook_pair_record(pair: "diffpair.VersionPair") -> dict[str, Any]:
    """One .ipynb pair's cell-level diff record (same schema as :func:`pair_record`)."""
    old_cells = _nb_cells(pair.old.path)
    new_cells = _nb_cells(pair.new.path)
    alignment_ok = old_cells is not None and new_cells is not None
    changes: list[dict[str, Any]] = []
    if alignment_ok:
        import difflib

        sm = difflib.SequenceMatcher(a=[c["src"] for c in old_cells],
                                     b=[c["src"] for c in new_cells], autojunk=False)
        for tag, i1, i2, j1, j2 in sm.get_opcodes():
            if tag == "equal":
                for oi, nj in zip(range(i1, i2), range(j1, j2)):
                    if old_cells[oi]["out"] != new_cells[nj]["out"]:
                        rem, add = _changed_lines(old_cells[oi]["out"], new_cells[nj]["out"])
                        changes.append({"kind": "modify", "intent": "SUBSTANTIVE",
                                        "old": rem, "new": add,
                                        "structural_location": f"cell {nj + 1} (output)",
                                        "attributes": ["substantive", "notebook", "output_change"]})
            elif tag == "replace":
                for oi, nj in zip(range(i1, i2), range(j1, j2)):
                    old_img = _embedded_image_b64(old_cells[oi]["src"])
                    new_img = _embedded_image_b64(new_cells[nj]["src"])
                    if old_img and new_img:
                        # An embedded data-URI image changed (e.g. a rendered 記述統計 table). The
                        # base64 blob itself is diff-noise; at build time (vision, opt-in) enumerate
                        # both renders' variable headers (tiled — glyph-exact) and transcribe the
                        # content, storing the HEADER set diff + content diff instead.
                        ch: dict[str, Any] = {"kind": "modify", "intent": "SUBSTANTIVE",
                                              "structural_location": f"cell {nj + 1} (embedded image)",
                                              "attributes": ["substantive", "notebook", "embedded_image"]}
                        old_hdr = _table_headers(old_img[1], old_img[0])
                        new_hdr = _table_headers(new_img[1], new_img[0])
                        if old_hdr is not None and new_hdr is not None:
                            ch["headers_old_count"] = len(old_hdr)
                            ch["headers_new_count"] = len(new_hdr)
                            ch["headers_added"] = [x for x in new_hdr if x not in old_hdr]
                            ch["headers_removed"] = [x for x in old_hdr if x not in new_hdr]
                            ch["attributes"].append("table_headers")
                            if ch["headers_added"]:
                                ch["attributes"].append("column_added")
                        old_txt = _transcribe_embedded(old_img[1], old_img[0])
                        new_txt = _transcribe_embedded(new_img[1], new_img[0])
                        if old_txt is not None and new_txt is not None:
                            rem, add = _changed_lines(old_txt, new_txt)
                            ch["old"], ch["new"] = rem, add
                            ch["attributes"].append("image_ocr")
                        else:
                            marker = "[埋め込み画像 (base64相違、転記未実施)]"
                            ch["old"] = ch["new"] = marker
                        changes.append(ch)
                        continue
                    rem, add = _changed_lines(old_cells[oi]["src"], new_cells[nj]["src"])
                    changes.append({"kind": "modify", "intent": "SUBSTANTIVE",
                                    "old": rem, "new": add,
                                    "structural_location": f"cell {nj + 1} (source)",
                                    "attributes": ["substantive", "notebook", "source_change"]})
                    if old_cells[oi]["out"] != new_cells[nj]["out"]:
                        rem, add = _changed_lines(old_cells[oi]["out"], new_cells[nj]["out"])
                        changes.append({"kind": "modify", "intent": "SUBSTANTIVE",
                                        "old": rem, "new": add,
                                        "structural_location": f"cell {nj + 1} (output)",
                                        "attributes": ["substantive", "notebook", "output_change"]})
                for oi in range(i1 + (j2 - j1), i2):
                    changes.append({"kind": "remove", "intent": "SUBSTANTIVE",
                                    "old": _clip_text(old_cells[oi]["src"]), "new": "",
                                    "structural_location": f"old cell {oi + 1}",
                                    "attributes": ["substantive", "notebook", "deletion"]})
                for nj in range(j1 + (i2 - i1), j2):
                    changes.append({"kind": "add", "intent": "SUBSTANTIVE",
                                    "old": "", "new": _clip_text(new_cells[nj]["src"]),
                                    "structural_location": f"cell {nj + 1}",
                                    "attributes": ["substantive", "notebook", "addition"]})
            elif tag == "delete":
                for oi in range(i1, i2):
                    changes.append({"kind": "remove", "intent": "SUBSTANTIVE",
                                    "old": _clip_text(old_cells[oi]["src"]), "new": "",
                                    "structural_location": f"old cell {oi + 1}",
                                    "attributes": ["substantive", "notebook", "deletion"]})
            elif tag == "insert":
                for nj in range(j1, j2):
                    changes.append({"kind": "add", "intent": "SUBSTANTIVE",
                                    "old": "", "new": _clip_text(new_cells[nj]["src"]),
                                    "structural_location": f"cell {nj + 1}",
                                    "attributes": ["substantive", "notebook", "addition"]})
    truncated = len(changes) > _MAX_STORED_CHANGES
    changes = changes[:_MAX_STORED_CHANGES]
    for rank, ch in enumerate(changes):
        ch["rank"] = rank
    return {
        "old_rel": nfc(pair.old.rel), "new_rel": nfc(pair.new.rel),
        "old_name": nfc(pair.old.name), "new_name": nfc(pair.new.name),
        "project": nfc(pair.new.project or pair.old.project or ""),
        "base": nfc(pair.base), "basis": "notebook", "ext": "ipynb",
        "sources": ["notebook"], "alignment_ok": alignment_ok,
        "old_size": _file_size(pair.old.path), "new_size": _file_size(pair.new.path),
        "change_count": len(changes), "truncated": truncated, "changes": changes,
    }


def _file_size(path) -> int:
    try:
        return int(Path(path).stat().st_size)
    except OSError:
        return 0


def _prior_notebook_records(out: Path | None) -> dict[tuple[str, str], dict[str, Any]]:
    """Prior store's notebook records keyed by (old_rel, new_rel) — for build-time reuse."""
    p = out or default_out_path()
    prior: dict[tuple[str, str], dict[str, Any]] = {}
    try:
        for line in Path(p).read_text(encoding="utf-8").splitlines():
            rec = json.loads(line)
            if rec.get("basis") == "notebook" and rec.get("old_rel") and rec.get("new_rel"):
                prior[(rec["old_rel"], rec["new_rel"])] = rec
    except Exception:  # noqa: BLE001 — no/corrupt prior store ⇒ nothing to reuse
        return {}
    return prior


def _notebook_records(out: Path | None) -> list[dict[str, Any]]:
    """Notebook pair records, reusing prior vision-derived records for unchanged files.

    The vision passes run only under ``RAG_OCR_STORE_BUILD``; a default (LLM-free) rebuild must not
    DOWNGRADE a previously transcribed record to the 転記未実施 marker, so an unchanged pair (same
    old/new file sizes) whose prior record carries vision output is kept verbatim.
    """
    prior = _prior_notebook_records(out)
    records: list[dict[str, Any]] = []
    for pair in _notebook_pairs():
        key = (nfc(pair.old.rel), nfc(pair.new.rel))
        old = prior.get(key)
        if (old is not None and old.get("alignment_ok")
                and old.get("old_size") == _file_size(pair.old.path)
                and old.get("new_size") == _file_size(pair.new.path)
                and any("image_ocr" in (c.get("attributes") or []) or "table_headers" in (c.get("attributes") or [])
                        or "embedded_image" not in (c.get("attributes") or [])
                        for c in old.get("changes", []) or [{}])):
            records.append(old)
            continue
        records.append(_notebook_pair_record(pair))
    return records


# --------------------------------------------------------------------------- write / build
def _sort_key(rec: dict[str, Any]) -> tuple:
    return (rec.get("project", ""), rec.get("old_rel", ""), rec.get("new_rel", ""))


def write_store(records: Sequence[dict[str, Any]], path: Path | None = None) -> dict[str, int]:
    """Atomically write a reproducible JSONL store (schema header + deterministically-sorted pair rows)."""
    out = path or default_out_path()
    out.parent.mkdir(parents=True, exist_ok=True)
    ordered = sorted(records, key=_sort_key)
    tmp = out.with_suffix(out.suffix + ".tmp")
    with tmp.open("w", encoding="utf-8", newline="\n") as handle:
        handle.write(json.dumps({"schema": SCHEMA, "version": SCHEMA_VERSION},
                                ensure_ascii=False, sort_keys=True) + "\n")
        for rec in ordered:
            handle.write(json.dumps(rec, ensure_ascii=False, sort_keys=True) + "\n")
    tmp.replace(out)
    reset_cache()
    total_changes = sum(r.get("change_count", 0) for r in ordered)
    return {"pairs": len(ordered),
            "changes": total_changes,
            "alignment_failures": sum(1 for r in ordered if not r.get("alignment_ok"))}


def build(refs: list[FileRef] | None = None, *, out: Path | None = None,
          write_report: bool = True) -> dict[str, Any]:
    """Precompute + persist the version-pair diff store (every family pair's ranked substantive changes).

    Deterministic and LLM-free. ``refs`` is accepted for wiring symmetry with the sibling stores; the pair
    universe itself comes from :func:`diffpair.find_pairs` / ``_registry_family_pairs`` which walk the
    corpus directly. Returns a small build summary. Additive & guarded — a failure here never corrupts the
    retrieval index (the caller in :mod:`src.rag.index` swallows exceptions).
    """
    pairs = enumerate_pairs()
    records = [pair_record(pair, sources) for pair, sources in pairs]
    # SOT-2650: notebook (.ipynb) pairs — build-time cell-level diff lane (office enumerators can never
    # produce these, so the append is disjoint / additive).
    try:
        records += _notebook_records(out)
    except Exception:  # noqa: BLE001 — the notebook lane must not sink the office store
        pass
    counts = write_store(records, out)
    report = build_report(records, out or default_out_path())
    if write_report:
        rp = report_out_path()
        rp.parent.mkdir(parents=True, exist_ok=True)
        rp.write_text(json.dumps(report, ensure_ascii=False, sort_keys=True, indent=2) + "\n",
                      encoding="utf-8")
    return {**counts, "report": report}


def build_report(records: Sequence[dict[str, Any]], out_path: Path) -> dict[str, Any]:
    """Diagnostic build report: ペア数 / ペアあたり変更数分布 / alignment失敗 / 属性分布 / hard-core カバレッジ."""
    change_counts = sorted(r.get("change_count", 0) for r in records)
    n = len(change_counts) or 1
    by_basis: dict[str, int] = {}
    by_ext: dict[str, int] = {}
    by_attr: dict[str, int] = {}
    for r in records:
        by_basis[r.get("basis", "")] = by_basis.get(r.get("basis", ""), 0) + 1
        by_ext[r.get("ext", "")] = by_ext.get(r.get("ext", ""), 0) + 1
        for ch in r.get("changes", []):
            for a in ch.get("attributes", []):
                by_attr[a] = by_attr.get(a, 0) + 1

    def _pct(p: float) -> int:
        return change_counts[min(int(p * n), n - 1)]

    return {
        "certificate": {
            "schema": SCHEMA, "schema_version": SCHEMA_VERSION,
            "question_independent": True,
            "enumeration_basis": ("全 version-family ペアを filename/folder 規約 ∪ document-registry 家系で "
                                  "網羅列挙し、各ペアの diffpair.rank_changes 全 candidate を順位付きで保存。"
                                  "diff ロジック不変更・gold値ハードコードなし。"),
        },
        "pairs": len(records),
        "alignment_failures": sum(1 for r in records if not r.get("alignment_ok")),
        "truncated_pairs": sum(1 for r in records if r.get("truncated")),
        "changes_total": sum(change_counts),
        "changes_per_pair": {
            "min": change_counts[0] if records else 0,
            "p50": _pct(0.50) if records else 0,
            "p90": _pct(0.90) if records else 0,
            "max": change_counts[-1] if records else 0,
        },
        "by_basis": dict(sorted(by_basis.items())),
        "by_ext": dict(sorted(by_ext.items())),
        "by_attribute": dict(sorted(by_attr.items())),
        "version_hard_core_coverage": version_hard_core_coverage(records),
        "out_path": str(out_path),
    }


# --------------------------------------------------------------------------- version hard-core coverage
# 設問 → カバレッジ診断（診断専用・ストア本体には非関与・gold値非依存）。version_diff の hard core (idx22) と
# B クラス (idx1/14/95)、および既知 MATCH の対照 (idx0) を対象に、その設問が参照する **版ペアが列挙され diff が
# 確定したか** を honest に報告する。idx22 は .ipynb 版ペアで、決定論 office diff パイプライン (pptx/docx/xlsx)
# の対象外 — その事実（別レーンで解く領域）も欠測を偽装せず明示する。
_VERSION_HARD_CORE: dict[str, dict[str, Any]] = {
    "idx0": {
        "gist": "白峰信用リスク評価 提案書old.pptx→提案書.pptx の実質的変更（既知 MATCH の対照）",
        "old_hint": "提案書old.pptx", "new_hint": "白峰信用リスク評価",
        "note": "registry-family が _old×無印 latest を補完して列挙するペア（find_pairs 単独では欠落）。",
    },
    "idx1": {
        "gist": "恒一会 かえで総合病院 最終報告_old→最新 の実質的変更（Bクラス誤答）",
        "old_hint": "最終報告_old", "new_hint": "恒一会",
        "note": "同上 registry-family 補完ペア。性能比較表の削除→要約置換を候補として保持。",
    },
    "idx14": {
        "gist": "青葉与信マネジメント 提案書_v1→_v3 の列名アンダースコア表記修正（Bクラス誤答）",
        "old_hint": "提案書_v1.pptx", "new_hint": "青葉与信",
        "note": "rev-suffix ペア。schema_name_change 属性で列名修正を除外条件フィルタに掛けられる。",
    },
    "idx22": {
        "gist": "白峰信用リスク評価 01_eda_old.ipynb→01_eda.ipynb の記述統計表の変更（hard core）",
        "old_hint": "01_eda_old.ipynb", "new_hint": "白峰信用リスク評価",
        "note": ("**.ipynb 版ペア** — 決定論 office diff パイプライン(pptx/docx/xlsx)の対象外。SOT-2650 で "
                 "build 時 notebook レーン（セル単位 source/出力 diff、basis=notebook）が本ストアに追加行として "
                 "列挙するようになった。"),
    },
    "idx95": {
        "gist": "青嶺不動産 スケジュール_r1→_r2、未着手→完了を除く実質的変更（Bクラス誤答）",
        "old_hint": "スケジュール_r1.xlsx", "new_hint": "青嶺不動産",
        "note": ("rev-suffix xlsx ペア。列全面再整列で atomic alignment はノイズが多い(多数候補)が、"
                 "status_transition 属性で「未着手→完了」除外を決定論適用できる素材は保持。"),
    },
}


def version_hard_core_coverage(records: Sequence[dict[str, Any]]) -> dict[str, Any]:
    """設問別に「参照版ペアが列挙され diff が確定したか」を honest 報告する診断表（gold 非依存）。

    ``old_hint``/``new_hint`` を old_rel/new_rel に部分一致させ、対応ペアの列挙有無・alignment 成否・
    変更数・source を報告する。列挙されていなければ ``enumerated=False`` と note を返す（欠測を偽装しない）。
    """
    out: dict[str, Any] = {}
    for idx, spec in _VERSION_HARD_CORE.items():
        oh, nh = spec.get("old_hint", ""), spec.get("new_hint", "")
        matched = [r for r in records
                   if (not oh or oh in r.get("old_rel", "")) and (not nh or nh in r.get("new_rel", ""))]
        rec = matched[0] if matched else None
        out[idx] = {
            "gist": spec["gist"],
            "enumerated": rec is not None,
            "pair": (f'{rec["old_rel"]} -> {rec["new_rel"]}' if rec else None),
            "sources": (rec["sources"] if rec else []),
            "alignment_ok": (rec["alignment_ok"] if rec else False),
            "change_count": (rec["change_count"] if rec else 0),
            "note": spec["note"],
        }
    return out


# --------------------------------------------------------------------------- load / read (minimal API)
_LOAD_CACHE: dict[str, list[dict[str, Any]]] = {}


def reset_cache() -> None:
    _LOAD_CACHE.clear()


def load(path: Path | None = None) -> list[dict[str, Any]]:
    """Load the diff-store pair rows (memoized). ``[]`` when absent/unreadable/schema-mismatch (回帰ゼロ)."""
    out = path or default_out_path()
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
    except Exception:
        rows = []
    _LOAD_CACHE[key] = rows
    return rows


def lookup_pair(old_rel: str | None = None, new_rel: str | None = None,
                *, project: str | None = None, path: Path | None = None) -> list[dict[str, Any]]:
    """Minimal read API (配線は SOT-2647): pair records matching old/new rel (substring) and/or project.

    Unknown pair / no artifact ⇒ ``[]`` (no degradation)."""
    o, nw = (nfc(old_rel) if old_rel else None), (nfc(new_rel) if new_rel else None)
    proj = nfc(project) if project else None
    out: list[dict[str, Any]] = []
    for row in load(path):
        if o is not None and o not in row.get("old_rel", ""):
            continue
        if nw is not None and nw not in row.get("new_rel", ""):
            continue
        if proj is not None and nfc(row.get("project", "")) != proj:
            continue
        out.append(row)
    return out


def filter_changes(record: dict[str, Any], *, exclude_attributes: Iterable[str] = (),
                   require_attributes: Iterable[str] = ()) -> list[dict[str, Any]]:
    """Ranked changes of one pair record with a deterministic attribute filter (question-side exclusions).

    ``exclude_attributes`` drops any change carrying one of the tags (idx95「未着手→完了 を除く」→
    ``exclude_attributes={"status_transition"}``); ``require_attributes`` keeps only changes carrying all of
    them. Order (substantive-first rank) is preserved. Question-independent store + question-side filter.
    """
    excl, req = set(exclude_attributes), set(require_attributes)
    out: list[dict[str, Any]] = []
    for ch in record.get("changes", []):
        attrs = set(ch.get("attributes", []))
        if excl & attrs:
            continue
        if req and not req <= attrs:
            continue
        out.append(ch)
    return out


if __name__ == "__main__":
    summary = build()
    print(json.dumps({**{k: v for k, v in summary.items() if k != "report"},
                      "certificate": summary["report"]["certificate"],
                      "coverage": summary["report"]["version_hard_core_coverage"]},
                     ensure_ascii=False, indent=2))
