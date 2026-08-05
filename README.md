# signate-messy-drive-rag

**SIGNATE AI Engineering Challenge — タスク「煩雑な社内ドライブをハックせよ」** に取り組む、
再現可能な RAG パイプラインと、その精度を**客観的に採点する3関門ハーネス**。
回答生成は **Google Cloud / Vertex AI Gemini**、本番バックエンドは toddler-private-rag と同系の
**Cloud Run + Vertex + Firestore + GCS**（Terraform IaC）で構築する。

> データ（`share.zip` 等）は SIGNATE 配布物で再配布不可のため `.gitignore` 済み。
> `bash scripts/fetch_data.sh` で各自の SIGNATE アカウントから取得する。

---

## 課題と採点の仕組み（客観スコアの核心）

- **入力**: 10社の顧客プロジェクト＋社内管理フォルダからなる「散らかった共有ドライブ」（418ファイル・
  PDF/Word/Excel/PowerPoint/画像/コード/Notebook が混在）と、各質問文。
- **出力**: `predictions.csv`（`index,answer` ヘッダなし・100問）→ zip → `signate submit`。
- **採点**: SIGNATE サーバ上の **LLM 審査 (gpt-5.2)** が各回答を模範解答と比較し、
  **100問の平均**を最終スコアとする。

  | 判定 | 点 | 意味 |
  |---|---|---|
  | Perfect | **+1** | 正確・虚偽なし |
  | Acceptable | **+0.5** | 有用だが軽微な誤り |
  | Missing | **0** | 「わかりません」等、具体的な答えなし |
  | Incorrect | **−1** | 誤答・無関係 |

  → **戦略的含意**: 自信のない誤答(−1)は棄権(0)より1点損。**確信度ゲート付きの棄権**が効く。

回答は各問 **1000トークン上限**。前処理＋100問回答で **12時間以内**が目安。
ハードコード（設問別の固定回答・特定案件専用分岐）は**禁止** — 汎用RAGのみ評価対象。

## 質問の性質（document-intelligence 型）

単純な本文検索では解けない。実際の設問例:
- Excel/PPTで**オレンジ/黄にハイライト**されたセル・数値・行
- Word 契約書の**太字**箇所の抽出
- チャートPNG（ヒストグラム等）の**最大カウント読み取り**
- xlsx **PivotTable** の集計条件
- `modeling.py` の**パラメータ値**、`01_eda.ipynb` の**相関**
- old版 vs 最新版の**差分**、全案件**横断集計**
- **社内用語集**での略称展開、**座席表**順の人物列挙

→ 書式・図表・コードまで機械的に解釈する抽出層が勝負。

---

## アーキテクチャ

```
data/share_drive (416 files, gitignored)
        │
        ▼
[1] 抽出  src/rag/extract/        # 形式別の決定論的抽出 + Gemini Vision
        │   docx(太字/ハイライト) · xlsx(セル塗り/PivotTable) · pptx(ハイライト/図)
        │   pdf(text/表/OCR) · csv/xlsx(表) · py/ipynb(コード) · png(チャート→Vision)
        ▼
[2] 索引  src/rag/index.py        # チャンク化 + Vertex embeddings + BM25
        ▼
[3] 検索  src/rag/retrieve.py     # ハイブリッド(dense+sparse) + 用語集展開 + 案件フィルタ
        ▼
[4] 生成  src/rag/generate.py     # Vertex Gemini + 根拠強制 + 確信度ゲート棄権
        ▼
    predictions.csv  ──►  signate submit
```

---

## 課題と設計根拠（Design Rationale）

タスクが突きつける各課題に対し、**なぜその設計にしたか**と**実装の在り処**を対応付ける。
一貫する原則は、①**書式・構造シグナルは平文化せず注記として保存**（検索可能にする）、
②**横断算術・列挙・差分・抽出条件は LLM に計算させず決定論モジュールで確定**、
③採点が Incorrect=−1 / Missing=0 のため**確信の持てない回答は棄権**（−1 を増やさない）、の3点。

### ① 画像・グラフ埋め込み資料を、構造化／非構造化／画像を統合して根拠に基づき回答
- 形式別の**決定論的抽出層**（`src/rag/extract/`）が docx / xlsx / pptx / pdf / csv / py / ipynb /
  json / png を個別処理し、構造化・非構造化・画像を**同一の索引**へ統合する。
- **PNG（チャート等）は2段階**: 索引時に Gemini Vision でキャプション化して内容検索可能にし
  （図の種類・軸・凡例・強調語のみ、創作禁止プロンプト）、回答時には**生画像を生成呼び出しに再添付**
  して「棒の最大カウント」等の精密読取をマルチモーダルで行う（`extract/vision.py`, `generate.py`）。
- 生成は「提供された根拠資料のみ使用・外部知識や推測で補わない」をシステムプロンプトで強制し、
  温度違いの**二重ドラフト自己一致**＋**厳格 verify パス**（疑わしきは supported=false→棄権）でゲートする。
- **根拠**: 埋め込み図表は平文化すると失われる。索引=キャプション（探せる）／回答=生画像（正確に読む）
  と役割を分けることで、検索性と読取精度を両立する。

### ② 複数資料・複数案件をまたぐ検索・照合・集計
- 検索は **dense（Vertex text-embedding-005）+ sparse（BM25）の RRF 融合ハイブリッド**（`retrieve.py`）。
- **横断集計は LLM に計算させない**。`compute.py` が複数レコードにまたがる算術（総額・平均・件数）を
  pandas / `Fraction` で決定論算出。列挙は `enumeration.py` が**ソース全体を走査しきれた場合のみ**回答
  （列挙は部分点なし → 部分列挙は棄権）。verify にも「根拠に直接無い集計値は不採用」を明記。
- **根拠**: LLM の暗算は桁・件数で誤りやすい。決定論算術に委ね、完全性を担保できない列挙は棄権する。

### ③ 社内用語・略称（社内用語集の参照）
- `extract/glossary.py` が社内管理フォルダの `社内用語集.docx`（9表）をパースし、
  **略称↔正式名称**・**案件名↔主略称/別名候補**のマップを構築。用途は3つ:
  (1) **クエリ展開**（略称→正式名称・企業名を検索クエリへ追加）、(2) **案件特定**（最長 alias 一致で
  対象案件を判定→検索ブースト）、(3) パスワード付きファイルの**パスワード導出**（⑥）。
- **根拠**: 略称は検索語としても案件絞り込みキーとしても効く。用語集を1度パースし多目的に使い回す。

### ④ 回答形式・単位・小数桁・丸め・主略称指定・識別子の表記
- システムプロンプトで「質問文が指定する形式・単位・小数桁・丸め・並び順・主略称/通常表現に従う」
  「**指定がない限り通常表現で記載**」「**タスクID/アクションID/列名/パラメータ名等の識別子は資料の
  表記どおり**」を明示強制。verify が「問われた値・要素だけ」の最小・正確形式へ蒸留する。
- **意味保存の正規化レイヤ**（SOT-2448, `normalize.py`）が冗長表現（前置き・「〜です」・単位重複）を
  除去。全数値・全 ASCII 識別子が逐語一致で保存されない候補は**元の回答へフォールバック**するため、
  値や識別子は壊れない。
- **根拠**: 実 gpt-5.2 審査は冗長・書式ずれを Incorrect 側へ倒す傾向（SOT-2446）。ただし正規化で意味を
  壊せば逆効果なので、逐語保存ゲートを通らない変換は捨てる。

### ⑤ 該当する対象が存在しない場合の「該当なし」回答
- 「条件に該当する対象が**資料内に明確に存在しないと確認できる場合のみ**『該当なし』」をプロンプトと
  verify の双方で強制。「見つからなかっただけ」は該当なしと断定せず、低確信→棄権する。
- **根拠**: 存在しない確認と探せなかったは別。誤った「該当なし」は −1 なので、断定は確認できた時だけ。

### ⑥ PNG / Jupyter Notebook / パスワード付きファイル
- **PNG**: ①のとおり（索引時 Vision キャプション + 回答時の生画像添付）。
- **.ipynb**: `plain.py::extract_ipynb` がセル単位でソースと**実行出力**（text/plain）を抽出し索引化
  （`01_eda.ipynb` の相関値等は出力から読める）。
- **パスワード付き Office**: `extract/passwords.py` がコーパスで使われる2方式を自動解決 —
  (A) ファイル名 `..._pw-<token>` → token がそのままパスワード、(B) 規則
  `DA-<案件略号>-<契約開始日YYYYMMDD>-<拡張子>`（略号=用語集の主略称、契約開始日は**同案件の可読資料
  から自動採掘**）。候補集合上の有界ブルートフォースで変種もカバーし、msoffcrypto で透過復号→通常の
  抽出パスへ流す。
- **根拠**: パスワードは資料内の規則から導出できる。総当たりでなく用語集＋契約日から候補を絞る。

### ⑦ Word / PowerPoint の書式情報（太字・文字色など）
- `extract/office.py` が書式を**検索可能なテキスト注記として保存**する: docx 太字 run→`【太字】…`・
  ハイライト→`【ハイライト:色】…`、xlsx セル塗り→HSV 分類による粗い色名（`C5(オレンジ): 値` 等。
  淡いテーマ色も分類、解決不能なテーマ色は誤検出防止のため fail-closed）、pptx は run XML の
  highlight と図形塗り色を抽出。pptx 要素は列挙規則に合わせ**上→下・左→右**でソート。
- **根拠**: 太字・塗り色は問題の答えそのもの。平文化すると消えるため、明示注記として索引に残す。

### ⑧ PDF の画像・図表
- テキスト層は pdfplumber（表はパイプ区切り行として保持）、失敗時 pypdf フォールバック。
- **画像のみ PDF**（テキスト層なし）はページ埋め込みラスタを抽出し、回答時に**レポート全ページを
  Gemini に添付**して読取（無関係な PNG を誤読しないようレポート単位で添付）。目視レビュー済みマーカー
  集合は**ピクセルハッシュでキー化**しソース変化時は fail-closed。画像のみの請求表は型付き抽出で決定論対応。
- **根拠**: スキャン PDF はテキストが無い。レポート単位でのマルチモーダル読取が誤読を抑えつつ精度を出す。

### ⑨ 検索対象ファイルが非常に多い（418ファイル）
- ヒント両方に対応: 質問中の**ファイル名トークン**（`train.xlsx`, `figure_06.png`, `modeling.py`…）を
  検出しファイル名完全一致チャンクを **2.2倍**、パス/識別子一致を1.3倍ブースト。**企業名/略称から用語集
  で案件を特定**し該当案件チャンクを1.6倍ブースト（`retrieve.py`）。
- macOS 由来の **NFD ファイル名は全て NFC 正規化**して照合（`corpus.py`）。
- 追加の検索強化（レア語 IDF 信号・旧版 decay=`RAG_RRF4`、LLM rerank + ファイル毎上限=`RAG_RERANK`）は
  実装済みだが、ローカル proxy が実 LB と非相関だった教訓から**関門3（実提出）で確認するまで既定 OFF の
  opt-in**。
- **根拠**: 418ファイルの全走査は非効率。ファイル名・企業名の手掛かりで検索空間を先に絞る。

### ⑩ テキスト読解に加え、追加処理・推論が必要な質問
- **決定論ハードモジュール群**でルーティング: `compute.py`（横断算術）、`enumeration.py`（完全性ゲート付き
  列挙）、`diffpair.py`（old版 vs 最新版の**構造 diff** — 表セルを行ラベルで対応付け「変更前→変更後」を
  機械特定）、`pivotcond.py`（**PivotTable 定義 / AutoFilter の XML を直接読み**、ソース表上で集計を再計算
  — フラット化セル文字列には存在しない抽出条件に対応）。
- 各モジュールは **sealed hold-out で実証済みの archetype のみ直接コミット**し、未実証は advisory ヒントと
  して LLM+verify が判断（可視案件への過学習が未知案件で誤答コミットする事故を防止）。archetype 別
  trust map により測定済み低精度の型は LLM 呼び出し前に棄権。
- **根拠**: 差分・抽出条件・横断集計は生成では不安定。構造を直読して確定し、未実証型は棄権へ倒す。

### ⑪ Google Document AI / Document AI Layout Parser は使っているか
- **使用していない**。python-docx / openpyxl / python-pptx / pdfplumber / msoffcrypto による
  **OOXML・PDF の直接解析** + Vertex AI Gemini（生成・Vision・埋め込み）の構成。
- **根拠**: 本タスクの勝負所は「太字・セル塗り色・PivotTable 定義・AutoFilter・パスワード解決」といった
  **書式・構造シグナル**で、Document AI 系の出力（レイアウト＋テキスト中心）はこれらを表出しない。
  OOXML 直読の方が決定論的・再現可能・低コストで精度も高い。PDF のレイアウト/図表読取は
  pdfplumber + 回答時 Gemini マルチモーダルで代替済み。テキスト層のないスキャン PDF の OCR が増える場合に
  Layout Parser 追加は将来の選択肢として残す。

---

## 3つの関門（RAG精度を客観測定）

| 関門 | データ | 正解 | 用途 |
|---|---|---|---|
| **関門1 教師データ** | `questions_valid.csv` 30問 | 公式GT (`valid_txt.csv`) | 手元CRAG採点の主シグナル |
| **関門2 推測した未知** | 同ドライブから**自動生成**したQ/A ＋ **1案件を封印** | 機械生成GT | 30問への過学習を検知・汎化測定 |
| **関門3 完全な未知** | `questions_test.csv` 100問 | 非公開 | SIGNATE本提出＝真の汎化KPI |

手元採点は公式 `evaluator.py` の採点ルールをそのまま移植（`scoring/`）。審査バックエンドは
**Codex CLI（`codex exec`, GPT-5.x）が既定**（SOT-2457）: 公式採点者 gpt-5.2 と同系列モデルで
審査するため、Gemini 代理審査で確認された「約0.2の甘さ・実LBと無相関(ρ=−0.09)」のギャップを
詰める。OpenAI キーは不要（Codex CLI の認証を利用）。Codex CLI が無い環境や
`JUDGE_BACKEND=gemini` 指定時は従来の Gemini 審査、Codex 呼び出し失敗（使用上限等）時も
Gemini に自動フォールバックする。チューニングは環境変数
`CODEX_JUDGE_MODEL` / `CODEX_JUDGE_BATCH` / `CODEX_JUDGE_VOTES` / `CODEX_JUDGE_TIMEOUT`
（`scoring/codex_judge.py`）。注意: ローカル採点は依然 proxy であり、採用ゲートは実LB確認のみ。

### 決定論的な自己改善ハーネス（archetype別 trust map, SOT-2407）

実 gpt-5.2 と非相関な Gemini 代理採点への依存を減らすため、**コーパス内の機械可読データから GT を
プログラム抽出したQ/A**を採点する、モデル呼び出し不要（=ノイズゼロ）のループを追加した:

- `scoring/synth.py` — `project_config.json` / `metrics.json` / `train.csv` / 社内用語集 から
  8 archetype・**156問**の GT 付き Q/A を決定論生成（`config_model_type` / `config_hyperparam` /
  `metric_score` / `data_shape` / `csv_column_mean` / `csv_column_max` / `glossary_formal` /
  `glossary_abbrev`）。GTは自己採点で全問 Perfect になることを検証（rubric整合）。
- `scoring/deterministic.py` — 数値/集合/文字列の型付き比較で公式 rubric（Perfect+1/Acceptable+0.5/
  Missing0/Incorrect−1）を**決定論的**に再現。
- `scoring/selfimprove.py` — 実RAGを流し archetype別に **committed精度・coverage** を集計し、
  精度≥閾値の型だけを `config/archetype_trust.json` に *trusted* として記録。
- `src/rag/archetype.py` — 質問を archetype に分類（`generate` の trust ゲート用）。
- `src/rag/generate.py` — trust map を参照し、**測定済みかつ低精度**の archetype のみ LLM 呼び出し前に
  棄権する **加算的ゲート**（棄権=Missing 0 は誤答−1 より良い＝Incorrectを増やさない）。

```bash
python -m scoring.synth                            # 156問の GT 付きベンチを生成
python -m scoring.selfimprove --self-test          # 決定論スコアラの整合を offline 検証（LLM不要）
python -m scoring.selfimprove                       # 実RAG→決定論採点→archetype別 trust map 出力
.venv/bin/python -m pytest scoring/test_harness.py -q   # ハーネスの単体テスト
```

## セットアップ

```bash
python3 -m venv .venv && . .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env            # GCP_PROJECT_ID などを確認
gcloud auth application-default login   # Vertex 用 ADC
bash scripts/fetch_data.sh      # SIGNATE からデータ取得（要 signate CLI 設定）
```

## 使い方（予定）

```bash
python -m src.rag.build_index                    # 抽出 + 索引構築
python -m src.rag.run --split valid              # 関門1: valid30 に回答
python -m scoring.gate1                          # 関門1: 客観採点
python -m scoring.gate2                          # 関門2: 自動生成ホールドアウト採点
python -m src.rag.run --split test               # 関門3: test100 -> predictions.csv
python -m scoring.gate3 --submit                 # 関門3: signate 提出
```

## リポジトリ構成

```
config/settings.py     設定（GCP/モデル/パス）
src/rag/               抽出→索引→検索→生成→バッチ実行
scoring/               3関門の採点ハーネス（CRAG審査: Codex CLI既定 / Geminiフォールバック）
infra/terraform/       GCP バックエンド（Cloud Run/Vertex/Firestore/GCS）
backend/               FastAPI サービス（Cloud Run）
scripts/fetch_data.sh  SIGNATE データ再取得
```
