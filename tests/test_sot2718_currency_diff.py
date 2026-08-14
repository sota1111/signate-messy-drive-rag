from src.rag.agent import formula_apply_lane as lane


def _ocr_record():
    return {
        "project": "株式会社東都人材プラットフォーム",
        "name": "職種給与調査.docx",
        "locus": "image1.emf",
        "full_text": (
            "職務タイトル 米国平均給与・中央値(米ドル) "
            "データエンジニア 平均 125,256 基盤構築 "
            "機械学習( ML )エンジニア 中央値 140,000 平均 143,000 モデル運用"
        ),
    }


def test_ocr_pairwise_average_currency_diff(monkeypatch):
    monkeypatch.setenv("RAG_FORMULA_APPLY", "1")
    monkeypatch.setattr(lane._image_ocr_store, "load", lambda: [_ocr_record()])
    monkeypatch.setattr(lane, "_company_of", lambda _q: "株式会社東都人材プラットフォーム")
    q = "東都人材プラットフォームの米国平均給与における機械学習（ML）エンジニアとデータエンジニアの差はいくらですか。"
    out = lane.resolve(q)
    assert out["value"] == "17,744ドル"
    assert out["method"]["selection"] == "ocr_job_metric_pairwise_diff"


def test_ocr_pairwise_diff_off_is_noop(monkeypatch):
    monkeypatch.delenv("RAG_FORMULA_APPLY", raising=False)
    monkeypatch.setattr(lane._image_ocr_store, "load", lambda: [_ocr_record()])
    assert lane.resolve("機械学習（ML）エンジニアとデータエンジニアの平均給与の差") is None


def test_ocr_pairwise_diff_defers_without_unique_operands(monkeypatch):
    monkeypatch.setenv("RAG_FORMULA_APPLY", "1")
    rec = _ocr_record()
    rec["full_text"] = rec["full_text"].replace("平均 143,000", "平均 143,000 平均 144,000")
    monkeypatch.setattr(lane._image_ocr_store, "load", lambda: [rec])
    monkeypatch.setattr(lane, "_company_of", lambda _q: "株式会社東都人材プラットフォーム")
    q = "米国平均給与における機械学習（ML）エンジニアとデータエンジニアの差はいくらですか。"
    assert lane.resolve(q) is None
