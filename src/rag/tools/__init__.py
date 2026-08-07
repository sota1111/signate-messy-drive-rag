"""Agent-callable deterministic tools with a unified ``{value, evidence, method}`` contract.

* :mod:`~src.rag.tools.contract` — the shared result contract (``ToolResult`` / ``is_contract``).
* :mod:`~src.rag.tools.profile` — the ``corpus_profile.json`` adaptation layer (runtime cache of
  self-discovered passwords / aliases / formats; raw secrets never committed).
* :mod:`~src.rag.tools.compute_sandbox` — restricted pandas execution (``run``).
* :mod:`~src.rag.tools.corpus_aggregate` — 全プロジェクト横断の決定論集約 (契約金額/着手金/行数/担当者/
  契約期間 の count/max/min/period-filter + 会社名↔主略称正規化) (``corpus_aggregate``).
* :mod:`~src.rag.tools.extract_tools` — thin contract wrappers over corpus/passwords/glossary/
  office/vision extraction.
* :mod:`~src.rag.tools.file_grep` — NFC-aware cross-corpus full-text / cell grep (``file_grep``).
* :mod:`~src.rag.tools.pdf_faux_italic` — detect faux-italic (matrix-shear) emphasis in a PDF.
* :mod:`~src.rag.tools.highlight_extract` — enumerate highlighted/marker words & cells (xlsx/pptx/docx/pdf).
* :mod:`~src.rag.tools.emf_pivot` — reconstruct an embedded EMF PivotTable image (text + highlights).
* :mod:`~src.rag.tools.chart_numcache` — read exact xlsx/pptx chart plot values from ``numCache``.
* :mod:`~src.rag.tools.seating_chart` — 座席表画像 → 氏名⇄内線(EXT)⇄座席 directory + 空間ルックアップ.
* :mod:`~src.rag.tools.canonical_route` — data/計算系質問を案件の canonical データファイル (train.xlsx/
  分析コード/notebook/leaderboard) へ直行させ、chunk 検索を迂回する決定論ルート (``canonical_route``).
"""
from src.rag.tools.compute_sandbox import ComputeError
from src.rag.tools.compute_sandbox import run as compute_run
from src.rag.tools.corpus_aggregate import (
    ProjectContract,
    collect_contracts,
    corpus_aggregate,
)
from src.rag.tools.contract import ContractError, ToolResult, ensure_contract, is_contract, make
from src.rag.tools.extract_tools import (
    caption_figure,
    company_of,
    decrypt,
    expand_terms,
    extract_office,
    find_files,
    resolve_ref,
)
from src.rag.tools.emf_pivot import (
    emf_blobs_from_pptx,
    extract_emf_pivot,
    extract_pptx_pivots,
    highlighted_cells,
    resolve_pivot_semantics,
)
from src.rag.tools.chart_numcache import (
    chart_series,
    chart_xml_members,
    embedded_chart_sources,
    extract_chart_numcache,
    read_chart_values,
    series_values,
)
from src.rag.tools.canonical_route import (
    CanonicalRouteError,
    canonical_route,
    discover as canonical_discover,
    infer_kinds as canonical_infer_kinds,
    resolve_project as canonical_resolve_project,
)
from src.rag.tools.file_grep import FileGrepError, file_grep
from src.rag.tools.pdf_faux_italic import detect_faux_italic, emphasized_words
from src.rag.tools.highlight_extract import highlight_extract, highlight_words
from src.rag.tools.profile import CorpusProfile
from src.rag.tools.seating_chart import (
    Directory,
    Seat,
    build_directory,
    extract_seating_image,
    seating_lookup,
)

# ``run`` kept as the compute_sandbox entry point for backward compatibility.
run = compute_run

__all__ = [
    "run",
    "compute_run",
    "ComputeError",
    "corpus_aggregate",
    "collect_contracts",
    "ProjectContract",
    "ToolResult",
    "ContractError",
    "is_contract",
    "ensure_contract",
    "make",
    "CorpusProfile",
    "find_files",
    "expand_terms",
    "company_of",
    "decrypt",
    "extract_office",
    "caption_figure",
    "resolve_ref",
    "file_grep",
    "FileGrepError",
    "detect_faux_italic",
    "emphasized_words",
    "highlight_extract",
    "highlight_words",
    "extract_emf_pivot",
    "highlighted_cells",
    "emf_blobs_from_pptx",
    "extract_pptx_pivots",
    "extract_chart_numcache",
    "chart_series",
    "chart_xml_members",
    "embedded_chart_sources",
    "read_chart_values",
    "series_values",
    "canonical_route",
    "CanonicalRouteError",
    "canonical_discover",
    "canonical_infer_kinds",
    "canonical_resolve_project",
    "seating_lookup",
    "build_directory",
    "extract_seating_image",
    "Seat",
    "Directory",
]
