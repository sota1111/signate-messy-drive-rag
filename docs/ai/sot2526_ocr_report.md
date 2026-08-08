# SOT-2526 — 画像スキャンPDFの OCR フォールバック（棄権E / RETRIEVED_NOT_PARSED）

## 真因（診断）

対象4問の実挙動を再現・実データ抽出で確認した結果、issue の初期分類より踏み込んだ真因が判明した。

| idx | gold | 種別 | 真因 |
| --- | --- | --- | --- |
| 28 | BMI | RETRIEVED_NOT_PARSED（＋誤答） | みなみ野の `06.報告書/…最終報告.pdf` が**画像スキャン**でテキスト層なし（抽出122字＝ページマーカのみ）。素朴に train.csv 相関を計算すると Age (0.266) > BMI (0.244) で**確信度1.0の誤答 Age** を出していた。 |
| 34 | A08、A09 | RETRIEVED_NOT_PARSED | 針（アクションアイテム表 A08/A09・担当 伊藤・マイルストーン M01/M02）が `05.会議/会議録/*.pdf` に在るが、これも**画像スキャン**（抽出41字）で pdfplumber/pypdf が本文を読めない。 |
| 87 | AYM | 横断列挙（PDF-parse 非該当） | `社内管理` の APR 定義＋各案件の行数を跨ぐ full_enumeration。canonical/parse の小口対策の対象外。 |
| 91 | campaign | — | 既に正答（canonical_route→compute）。 |

→ 28/34 の共通根本原因は **みなみ野案件の PDF が全て画像スキャン**であること（会議録=41字, 報告資料=34字, 最終報告=122字）。`msoffcrypto`/`pdfplumber`/`pypdf` はテキスト層が無いPDFからは何も取り出せない＝典型的 `RETRIEVED_NOT_PARSED`。

## 対策（一般・ハードコードなし・Gemini のみ）

画像専用PDFを検知し、ページ画像を Gemini vision で転記する **OCR フォールバック**を追加。

- `src/rag/extract/vision.py`
  - `pdf_ocr_enabled()` — env `RAG_PDF_OCR`（default **OFF**、sibling #1–#5 と同規約＝champion serve は byte-identical）。
  - `ocr_image_pdf(path)` — `pdf_page_images()` の**単一フルページ画像のみ**を対象に per-page 転記。混在（テキスト＋図）PDFは対象外→`None`。`(path,size,mtime)` で `lru_cache`。
- `src/rag/extract/plain.py`
  - `extract_pdf()` を、テキスト層が**実質空**（ページマーカ除去後の body が空）**かつ** flag ON の時のみ OCR にフォールバックするよう配線。default OFF＝従来のテキストのみ挙動と **byte-identical**。

## 検証（human指示08-08：gold100 は本issueで実行せず SOT-2527 に委譲）

- 単体: `tests/test_pdf_ocr.py` **16件 green**（flag解釈・ページ結合・非画像/空ページ=None・cache・OFF時OCR非実行で marker-only byte-identical・ON+空層でOCR・ON+テキスト層有りはスキップ・ocr=None時テキストfallback）。
- 非劣化: **offline 全 857 tests green**（既存841＋新16、hang無）。default OFF なので serve パス byte-identical＝関門2非劣化は自明。
- 実データ実証: `RAG_PDF_OCR=1` で `会議録_2025-04-24.pdf` を OCR→ **A01–A06 / 伊藤 / M01 / M02 / Open / Close / アクション** を4270字転記（OFFは41字）。
- focused（flag ON, timeout180）:
  - idx28: baseline **誤答 Age(−1)** → **安全棄権(0)**（OCRで素朴相関ショートカットを抑止）。＝ +1 EV。
  - idx34: **安全棄権維持**（会議録本文へ到達するも A08/A09 の組立ては未達）。回帰なし。
  - idx91: campaign **正答**（無影響）。
  - idx87: 横断列挙で over-enum（PDF-parse 非該当、flag ON 時のみの挙動）。

## 既知の caveat / 後続

- OCR の serve レイテンシが大きい（idx28 が 473s で timeout）。→ 本来は **index 時に一度だけ OCR して永続**（SOT-2528/2529 の precompute パターン）し serve は cache 参照するのが正道。本issueは parser 補修（flag-gated）に留め、index時OCR永続は後続に。
- full gold100 実測・最終 flag 構成判断（`RAG_PDF_OCR=1` を flag セットに追加）は **SOT-2527** の統合単一実行へ委譲。

ledger 帰属: `axis=retrieval-parse-small-bucket`（`docs/ai/experiment_ledger.jsonl`）。
