# Corpus reconnaissance notes

Hard-won facts about the SIGNATE 共有ドライブ corpus that the extraction pipeline depends on.
(Corpus itself is gitignored; this documents its structure.)

## Layout
- `data/share_drive/プロジェクト/<会社名>/` — 10 client projects, each with a fixed taxonomy:
  `00.提案`(pptx, versioned `_v1`/`_final`/`old`) · `01.契約`(docx contract, bold+amounts) ·
  `02.計画`(xlsx schedule, highlighted rows) · `03.データ`(train.csv, カラム説明.md) ·
  `04.分析`(a full ML repo: `notebooks/*.ipynb`, `src/*.py`, `reports/figures/*.png`, configs) ·
  `05.会議`(会議録 pdf, 報告資料 docx) · `06.報告書`(最終報告 pptx, `_old` variants).
- `data/share_drive/社内管理/` — glossary, seat map, approval rules, password rules.
- 418 files: py100 png54 docx46 md31 csv29 pdf28 json28 pptx26 xlsx21 ipynb11 txt10 toml10.

## GOTCHA 1 — filenames are NFD (macOS origin)
Directory/JP names are Unicode **NFD**-normalized. Python string literals are NFC → never
construct JP paths as literals; iterate + `unicodedata.normalize('NFC', name)` for matching.

## GOTCHA 2 — password-protected Office files (2 schemes, both real)
Encrypted OOXML = OLE container (not a zip). Genuinely encrypted (ignore `~$` Office locks):
- `かえで/01.契約/契約書_pw-kaede20250902.docx` → password **`kaede20250902`**
  (the `pw-<token>` in the filename literally *is* the password).
- `かえで/02.計画/スケジュール.xlsx` → password **`DA-KAEDE-20250902-xlsx`**
  (rule `DA-[案件略号]-[契約開始日YYYYMMDD]-[拡張子]`; 案件略号=glossary 主略称, date mined from
  readable project docs — かえで契約開始日 = 2025-09-02).
Resolver must try BOTH schemes + a bounded brute force. Decrypt with `msoffcrypto-tool`.
The `パスワード導出規則.docx` documents only the rule scheme (a partial decoy for the docx).

## GOTCHA 3 — 社内用語集 (glossary) drives query expansion AND passwords
`社内用語集.docx` = 9 tables mapping 正式名称 ↔ 社内用語(略語). Key ones:
- Case codes (Table「組織・案件略称」): KSS=京橋信用, AYM=青葉与信, SHR=白峰, AOSHIO=青潮モビリティ,
  SOHK=ひがし丘, TOTO=東都人材, AOMINE=青嶺不動産, KAEDE=かえで, AOBM=青葉バイオ, MINAMINO=みなみ野;
  CROSS=案件横断, INTERNAL=社内管理共通. Each has 別名候補 (青ソ, 楓病院, …).
- File/artifact codes: PP=提案書, CT=契約書, FR=最終報告書, TX=train.xlsx, MDL=modeling.py,
  EDA1=01_eda.ipynb, FIG=reports/figures, LB=leaderboard.csv, TG (a column) etc.
Questions use these abbreviations (e.g. "KSSのfigure_06.png", "TG平均") → expand via glossary.

## GOTCHA 4 — questions hinge on FORMATTING, not just text
- xlsx/pptx **highlight/fill color** (オレンジ/黄 rows, cells, numbers) → openpyxl `fill.fgColor`,
  pptx run highlight / shape fill.
- docx **bold** runs (契約書の太字箇所) → `run.bold`.
- chart **PNG** (histogram max count, marked words) → Gemini Vision.
- xlsx **PivotTable** aggregation conditions.
- **version diffs** old vs `_final`/最新 (pptx/docx/xlsx).
Extraction must emit these signals as structured, searchable text, not drop them.

## 決裁基準.md (approval rules) — for aggregation questions
契約金額(税込) → 主任/課長/部長/本部長. 医療案件 = +1段階. time_and_materials = 部長以上.
Applied in order: base → medical +1 → T&M floor.

## Scoring (recap)
Official: SIGNATE-side gpt-5.2 CRAG judge. Perfect+1 / Acceptable+0.5 / Missing0 / **Incorrect−1**,
mean of 100. Answer ≤1000 tiktoken tokens. → confidence-gated abstention ("わかりません") on low
evidence beats guessing.
