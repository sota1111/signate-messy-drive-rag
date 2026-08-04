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

## 3つの関門（RAG精度を客観測定）

| 関門 | データ | 正解 | 用途 |
|---|---|---|---|
| **関門1 教師データ** | `questions_valid.csv` 30問 | 公式GT (`valid_txt.csv`) | 手元CRAG採点の主シグナル |
| **関門2 推測した未知** | 同ドライブから**自動生成**したQ/A ＋ **1案件を封印** | 機械生成GT | 30問への過学習を検知・汎化測定 |
| **関門3 完全な未知** | `questions_test.csv` 100問 | 非公開 | SIGNATE本提出＝真の汎化KPI |

手元採点は公式 `evaluator.py` の採点ルールを **Gemini 審査**へ移植（`scoring/`）。公式は gpt-5.2 だが、
提出物にOpenAIキーは不要（採点はSIGNATE側）。境界事例のみ差が出る程度。

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
scoring/               3関門の採点ハーネス（Gemini CRAG）
infra/terraform/       GCP バックエンド（Cloud Run/Vertex/Firestore/GCS）
backend/               FastAPI サービス（Cloud Run）
scripts/fetch_data.sh  SIGNATE データ再取得
```
