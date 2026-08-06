"""Agent-callable deterministic tools with a unified ``{value, evidence, method}`` contract.

* :mod:`~src.rag.tools.contract` — the shared result contract (``ToolResult`` / ``is_contract``).
* :mod:`~src.rag.tools.profile` — the ``corpus_profile.json`` adaptation layer (runtime cache of
  self-discovered passwords / aliases / formats; raw secrets never committed).
* :mod:`~src.rag.tools.compute_sandbox` — restricted pandas execution (``run``).
* :mod:`~src.rag.tools.extract_tools` — thin contract wrappers over corpus/passwords/glossary/
  office/vision extraction.
* :mod:`~src.rag.tools.file_grep` — NFC-aware cross-corpus full-text / cell grep (``file_grep``).
* :mod:`~src.rag.tools.pdf_faux_italic` — detect faux-italic (matrix-shear) emphasis in a PDF.
* :mod:`~src.rag.tools.highlight_extract` — enumerate highlighted/marker words & cells (xlsx/pptx/docx/pdf).
* :mod:`~src.rag.tools.emf_pivot` — reconstruct an embedded EMF PivotTable image (text + highlights).
* :mod:`~src.rag.tools.chart_numcache` — read exact xlsx/pptx chart plot values from ``numCache``.
"""
from src.rag.tools.compute_sandbox import ComputeError
from src.rag.tools.compute_sandbox import run as compute_run
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
)
from src.rag.tools.chart_numcache import (
    chart_series,
    chart_xml_members,
    extract_chart_numcache,
    series_values,
)
from src.rag.tools.file_grep import FileGrepError, file_grep
from src.rag.tools.pdf_faux_italic import detect_faux_italic, emphasized_words
from src.rag.tools.highlight_extract import highlight_extract, highlight_words
from src.rag.tools.profile import CorpusProfile

# ``run`` kept as the compute_sandbox entry point for backward compatibility.
run = compute_run

__all__ = [
    "run",
    "compute_run",
    "ComputeError",
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
    "series_values",
]
