"""SOT-2511 — deterministic PPTX Gantt week-grid extraction."""
from __future__ import annotations

from pptx import Presentation
from pptx.enum.shapes import MSO_CONNECTOR
from pptx.util import Inches

from src.rag.corpus import FileRef
from src.rag.extract.office import WeekCell, extract_gantt_week_ranges, extract_pptx, week_range_for_span


def test_week_span_uses_positive_half_open_overlap_at_boundaries() -> None:
    cells = tuple(WeekCell(week, (week - 1) * 100, week * 100) for week in range(1, 5))
    # Exact boundary: [100, 300) means W2-W3, never W1 or W4.
    assert week_range_for_span(100, 200, cells) == (2, 3)
    # One coordinate epsilon before/after fixes the boundary rule deterministically.
    assert week_range_for_span(99, 201, cells) == (1, 3)
    assert week_range_for_span(101, 200, cells) == (2, 4)
    assert week_range_for_span(400, 10, cells) is None


def _synthetic_gantt() -> Presentation:
    prs = Presentation()
    slide = prs.slides.add_slide(prs.slide_layouts[6])
    for offset, week in enumerate(range(1, 5)):
        header = slide.shapes.add_textbox(
            Inches(4 + offset), Inches(1), Inches(1), Inches(0.4))
        header.text = f"W{week}"
    activity = slide.shapes.add_textbox(Inches(0.5), Inches(2), Inches(3.5), Inches(0.8))
    activity.text = "モデル改善\n説明性分析"
    # Starts exactly at W2 and ends exactly at W4's left edge => W2-W3.
    slide.shapes.add_connector(
        MSO_CONNECTOR.STRAIGHT, Inches(5), Inches(2.4), Inches(7), Inches(2.4))
    return prs


def test_native_shape_geometry_maps_activity_to_week_range() -> None:
    records = extract_gantt_week_ranges(_synthetic_gantt())
    assert records == [{
        "slide": 1,
        "activity": "モデル改善 説明性分析",
        "status": "resolved",
        "start_week": 2,
        "end_week": 3,
        "bar_left": Inches(5),
        "bar_width": Inches(2),
    }]


def test_gantt_annotation_is_prepended_before_flattened_slide_text(tmp_path) -> None:
    path = tmp_path / "schedule.pptx"
    _synthetic_gantt().save(path)
    ref = FileRef(path=path, project="", category="proposal", rel=path.name,
                  name=path.name, ext="pptx")
    text = extract_pptx(ref, None)
    assert text.startswith("【ガント週グリッド:決定論】")
    assert "モデル改善 説明性分析: 第2週目から第3週目" in text
    assert "半開区間" in text


def test_conflicting_native_bars_are_reported_ambiguous() -> None:
    prs = _synthetic_gantt()
    slide = prs.slides[0]
    slide.shapes.add_connector(
        MSO_CONNECTOR.STRAIGHT, Inches(6), Inches(2.4), Inches(7), Inches(2.4))
    records = extract_gantt_week_ranges(prs)
    assert records[0]["status"] == "ambiguous"
    assert records[0]["candidates"] == [
        {"start_week": 2, "end_week": 3},
        {"start_week": 3, "end_week": 3},
    ]
