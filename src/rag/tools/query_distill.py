"""query_distill — deterministic pre-search query distillation (SOT-2672).

Cerebras/ESG 型検索基盤の移植 4/4 (parent PLAN SOT-2602). 2つの独立コンペ解法が収束した手法の移植:
「会社名や回答条件を除き核となる情報を抽出して検索」/「〈〇〇の〉〈〇〇において〉といった企業情報を除去」。
例: 『ABC社の2019年度売上高を知りたい』→ scope=ABC社 + query『2019年度の売上高』。

生の質問文/LLM 判断のまま FTS(:mod:`src.rag.index.text_fts`)へ渡すと、会社名・修飾句・疑問句の低IDF
トークンがヒットを薄め、希少な核トークン(ID・指標名・固有語)が埋もれる。本モジュールは検索の *前段* に、
LLM を一切使わない決定論のクエリ蒸留を挟む:

* **(a) 会社名 → スコープ化 (捨てずに転用)**  当プロジェクトには registry の案件スコープ解決
  (:func:`src.rag.tools.canonical_route.resolve_project`, 用語集 glossary 適応層) が既にある。会社名/略称
  を検出したら、クエリ語からは *除去* しつつ、解決できた案件フォルダを FTS の ``project`` スコープ
  フィルタへ *移す* — 解法より一歩進んだ「除去ではなくスコープ絞り」。スコープに解決できない会社名は
  除去しない(誤ってスコープを失わないための保守的分岐)。
* **(b) 条件句・疑問句の除去**  「〜を教えてください / 〜は何ですか / いくつありますか / すべて挙げて」等の
  定型スキャフォールドを決定論の除去テーブル(正規表現)で落とす。内容名詞は落とさない。
* **(c) 希少トークン保持**  ID 形(EXT1234 / APR-M3 / M-01)の希少トークンは、除去で誤って落ちないよう
  常に温存する。
* **コーパス語彙ブリッジ (別コンペ・キーワードスコアリング型解法の追加手法)**  蒸留クエリでヒットしない
  場合に、質問語をそのまま使わず「本文中に実際に登場する近義・関連トークン」(IDF 表 = コーパス語彙)へ
  置換して 1 回だけ再検索する決定論版。:func:`corpus_vocab_retry` が text_fts の IDF 語彙を引き、語彙外の
  クエリトークンを最も希少(高IDF)な部分文字列一致語へ置換する。

Serve-time invariants (兄弟ツールと同一):
  * **Opt-in.** :func:`enabled` (``RAG_QUERY_DISTILL``) 既定 OFF ⇒ 生クエリのまま = champion serve は
    byte-identical。text_search / unified_search は OFF 時に蒸留を一切呼ばない。
  * **決定論・LLM/Gemini-free.**  正規表現 + glossary/registry 照合 + IDF 表 lookup のみ。埋め込み無し。
  * **Fail-open.**  glossary/registry/IDF の欠損・例外は「蒸留せず生クエリ」に落ちる(検索は必ず走る)。
"""
from __future__ import annotations

import os
import re
from dataclasses import dataclass, field
from typing import Any

from src.rag.corpus import nfc

_ON = {"1", "true", "yes", "on"}


def enabled() -> bool:
    """True when the serve path should distill the query before searching (default OFF — opt-in)."""
    return os.getenv("RAG_QUERY_DISTILL", "0").strip().lower() in _ON


# --------------------------------------------------------------------------- rare/ID token retention
# A rare literal: letters+digits joined by -_. (EXT1234 / APR-M3 / M-01 / WBS_12 / AI08) or digits+letters.
# These carry the highest IDF and must never be dropped by phrase removal (受け入れ条件: 希少トークン保持).
_ID_TOKEN_RE = re.compile(r"[A-Za-z]+(?:[-_.][A-Za-z]+)*[-_.]?\d{1,5}|\d{1,5}[A-Za-z]{1,}")


def _id_tokens(text: str) -> list[str]:
    seen: set[str] = set()
    out: list[str] = []
    for m in _ID_TOKEN_RE.finditer(text):
        t = m.group(0)
        if t not in seen:
            seen.add(t)
            out.append(t)
    return out


# --------------------------------------------------------------------------- condition/question removal
# Deterministic removal table. Each entry strips a *formulaic* interrogative/politeness scaffold — never a
# content noun. Ordered so longer compound tails are removed before their fragments. Applied over NFC text.
_CONDITION_PATTERNS: tuple[re.Pattern[str], ...] = tuple(re.compile(p) for p in (
    # compound scaffolds first, so a bare fragment ("挙げてください") never eats before its compound.
    r"(?:を|は)?(?:すべて|全て)(?:挙げて|列挙して|教えて)(?:ください|下さい)?",
    r"について(?:は)?(?:教えて|知りたい)?(?:ください|下さい)?",
    r"に(?:関して|関する|係る|ついて)(?:教えて)?(?:ください|下さい)?",
    r"を(?:教えて|知りたい|調べたい|確認したい|示して|挙げて|列挙して)(?:ください|下さい)?",
    r"(?:教えて|示して|挙げて|列挙して)(?:ください|下さい)",
    r"(?:を|は)?(?:知りたい|調べたい|確認したい)",
    r"は?\s*(?:何|なに)(?:個|件|人|社|種類|種|通り|台|円|年|回|度)?(?:ありますか|あるか|ある|ですか|でしょうか|か)",
    r"は?\s*いくつ(?:ありますか|あるか|ある|ですか|か)?",
    r"とは(?:何|なに)?(?:ですか|か)?",
    r"(?:ですか|でしょうか|ますか|のか)(?=[。\?？!！]?$)",
    r"ください|下さい",
))

# Trailing possessive/scope particle immediately after a removed company token: 『ABC社の…』→ drop 『の』too.
_TRAILING_PARTICLE = re.compile(
    r"^(?:のうち|における|において|に関する|に関して|に係る|の場合|向けの|向け|の|は|が|を|で|と)")

# Whitespace normaliser for the distilled output.
_WS = re.compile(r"\s+")


@dataclass
class DistillResult:
    """Outcome of :func:`distill` — the distilled query plus the scope/diagnostics it produced."""

    original: str
    query: str
    scope_project: str | None = None
    removed_company: str | None = None
    removed_phrases: list[str] = field(default_factory=list)
    kept_id_tokens: list[str] = field(default_factory=list)
    changed: bool = False

    def as_diagnostic(self) -> dict[str, Any]:
        """Compact record of 蒸留前後 for coverage/evidence (診断可能に; 受け入れ条件)."""
        return {
            "before": self.original,
            "after": self.query,
            "scope_project": self.scope_project,
            "removed_company": self.removed_company,
            "removed_phrases": self.removed_phrases,
            "kept_id_tokens": self.kept_id_tokens,
            "changed": self.changed,
        }


def _match_company(text: str, glossary: Any) -> "tuple[str, str] | None":
    """Longest glossary alias occurring in ``text`` → (canonical_company, matched_alias_substring).

    Mirrors :meth:`Glossary.company_of` (longest-alias-first to avoid 青葉/青葉与信 collisions) but also
    returns the *matched alias string* so the caller can excise exactly that span from the query.
    """
    best_company: str | None = None
    best_alias = ""
    for company, aliases in getattr(glossary, "company_aliases", {}).items():
        for a in aliases:
            if a and a in text and len(a) > len(best_alias):
                best_company, best_alias = company, a
    if best_company is None:
        return None
    return best_company, best_alias


def distill(query: str, *, project: str | None = None, glossary: Any | None = None,
            corpus_dir: Any | None = None) -> DistillResult:
    """Deterministically distill ``query`` for FTS: company→scope, condition removal, rare-token retention.

    Parameters
    ----------
    query
        The raw natural-language question.
    project
        An explicit project hint. When given it wins — no company is stripped and no scope is derived (the
        caller already scoped), only condition-phrase removal applies.
    glossary, corpus_dir
        DI seams (mirroring ``canonical_route``): tests inject a synthetic glossary / mini-corpus;
        production leaves both ``None`` (real corpus + cached glossary).

    Returns a :class:`DistillResult`. Fail-open: any glossary/registry error degrades to
    condition-removal-only (never raises into the search path).
    """
    original = query
    t = nfc(query)
    ids_before = _id_tokens(t)
    removed_company: str | None = None
    scope_project: str | None = None

    # (a) company → scope, only when we can resolve a real corpus folder (else keep the token, never guess).
    if project is None:
        try:
            import importlib

            from src.rag.extract import glossary as _glossary
            # ``from src.rag.tools import canonical_route`` binds the re-exported *function* (it shadows the
            # submodule in ``tools.__init__``); import the actual module to reach ``resolve_project``.
            _cr = importlib.import_module("src.rag.tools.canonical_route")

            g = glossary if glossary is not None else _glossary.load()
            hit = _match_company(t, g)
            if hit is not None:
                company, alias = hit
                resolved = _cr.resolve_project(None, company, corpus_dir=corpus_dir, glossary=g)
                if resolved:
                    scope_project = resolved
                    # excise the matched alias (+ an immediately trailing scope particle) from the query.
                    idx = t.find(alias)
                    if idx != -1:
                        rest = t[idx + len(alias):]
                        rest = _TRAILING_PARTICLE.sub("", rest, count=1)
                        t = t[:idx] + rest
                        removed_company = alias
        except Exception:  # noqa: BLE001 — fail-open: distillation degrades to condition removal only
            pass

    # (b) condition / question phrase removal (deterministic table).
    removed_phrases: list[str] = []

    def _capture(m: "re.Match[str]") -> str:
        removed_phrases.append(m.group(0))
        return ""

    for pat in _CONDITION_PATTERNS:
        t = pat.sub(_capture, t)

    # trailing/standalone punctuation + dangling grammatical particle left behind by removal.
    t = re.sub(r"[。\?？!！、，,\s]+$", "", t)
    t = re.sub(r"(?:の|は|が|を|に|で|と|へ|や)+$", "", t)
    t = _WS.sub(" ", t).strip()

    # (c) rare/ID token retention: re-append any ID-shaped token that removal accidentally dropped.
    kept_ids: list[str] = []
    for tok in ids_before:
        if tok in original:
            kept_ids.append(tok)
            if tok not in t:
                t = (t + " " + tok).strip()

    # never return an empty distilled query — if removal ate everything, keep the original.
    if not t:
        t = nfc(original).strip()

    changed = t != nfc(original).strip() or scope_project is not None
    return DistillResult(
        original=original, query=t, scope_project=scope_project,
        removed_company=removed_company,
        removed_phrases=[p for p in removed_phrases if p.strip()],
        kept_id_tokens=kept_ids, changed=changed,
    )


# --------------------------------------------------------------------------- corpus-vocabulary retry
def _bridge_tokens(tokens: list[str], vocab: dict[str, float]) -> "tuple[list[str], list[tuple[str, str]]]":
    """Replace out-of-vocabulary query tokens with the rarest body-present token that shares a substring.

    Deterministic: for a query token absent from the corpus IDF vocabulary, candidate vocab tokens are those
    that contain it or are contained by it (len≥2); the rarest (max IDF) wins, tie-broken by shorter length
    then lexicographic token. In-vocabulary tokens pass through unchanged. Returns (bridged_tokens, subs).
    """
    if not vocab:
        return tokens, []
    out: list[str] = []
    subs: list[tuple[str, str]] = []
    for tok in tokens:
        if tok in vocab:
            out.append(tok)
            continue
        cands = [v for v in vocab if len(v) >= 2 and v != tok and (tok in v or v in tok)]
        if not cands:
            out.append(tok)
            continue
        best = min(cands, key=lambda v: (-vocab[v], len(v), v))
        out.append(best)
        subs.append((tok, best))
    return out, subs


def corpus_vocab_retry(query: str, *, path: Any | None = None) -> "tuple[str, list[tuple[str, str]]]":
    """Deterministic 1-shot corpus-vocabulary rewrite of a zero-hit query.

    Tokenises ``query`` with text_fts's own tokenizer, reads the corpus IDF vocabulary, and bridges any
    out-of-vocabulary token to the rarest body-present near-token (:func:`_bridge_tokens`). Returns the
    rewritten query string (space-joined bridged tokens) and the substitution list; the query is unchanged
    (empty subs) when nothing could be bridged. Fail-open on any error.
    """
    try:
        from src.rag.index import text_fts

        toks = list(dict.fromkeys(text_fts.tokenize(query)))
        if not toks:
            return query, []
        vocab = text_fts.idf_vocab(path)
        bridged, subs = _bridge_tokens(toks, vocab)
        if not subs:
            return query, []
        return " ".join(bridged), subs
    except Exception:  # noqa: BLE001 — retry is a best-effort fast path, never a raised answer path
        return query, []
