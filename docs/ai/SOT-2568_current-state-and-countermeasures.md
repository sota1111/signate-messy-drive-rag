# SOT-2568 — 現状分析とディープリサーチによる対策検討

> 人間指示（`review=human`, 2026-08-09）:
> 「ディープリサーチを用いて、対策を検討する。現在の構成と取っている手法等、現状を説明してください。
> また、今回のSIGNATE課題の説明、現在の採点結果、どんな問いに対して誤りや棄権になっているか、
> なぜ正答できないかを説明してください。」
>
> 本書は **調査・分析ドキュメント**（PLAN 成果物）。コード変更・提出は行っていない。
> 実測の一次ソースは `docs/ai/gold100_sot2568_result.md`（gold100 net32）、
> `docs/ai/abstain_wrong_root_cause_SOT-2550.md`、`docs/ai/sot2563_intime_abstain_root_cause.md`、
> `artifacts/gold_100_review.csv`。対策候補は §6 のディープリサーチ結果に基づく。

---

## 0. 3行サマリ

- **現状**: gold100 オフライン実測で **match 39 / abstain 54 / wrong 7（net = match−wrong = 32）**。
  安全棄権を厚く取り precision を守る設計で、**手動 baseline 86.7% には未達**。実 LB がまだ一次 KPI。
- **失点の構造**: 誤りは少なく（wrong 7）、**取りこぼし（棄権 54）が本丸**。棄権の 83% が
  **BUDGET_EXHAUSTED(27) + UNANSWERABLE(18)** ＝「探索でターン予算を溶かして推論に届かない」＋
  「書式/深掘り抽出が面に出ず“根拠なし”に落ちる」の2大メカニズム。最弱型は
  **derived_calculation（多段数値, 8/32）**。
- **なぜ正答できないか**: ①**探索律速**（証拠の在り処探しに 90% のターンを消費）、②**書式・視覚
  抽出ギャップ**（ハイライト色/コメント/文字装飾/条件付き書式の“ルール”）、③**多段数値の計算到達性・
  正確性**、④**版差分の“実質的変更”選別**、の4系統。

---

## 1. SIGNATE 課題の説明（「煩雑な社内ドライブをハックせよ」）

- **入力**: 10社の顧客プロジェクト＋社内管理フォルダからなる「散らかった共有ドライブ」
  （**418ファイル**・PDF/Word/Excel/PowerPoint/画像/コード/Notebook が混在）と、各質問文。
- **出力**: `predictions.csv`（`index,answer`・100問・ヘッダなし）→ zip → `signate submit`。
- **採点**: SIGNATE サーバ上の **LLM 審査（gpt-5.2）** が各回答を模範解答と比較し **100問平均**で最終スコア。

  | 判定 | 点 | 意味 |
  |---|---|---|
  | Perfect | **+1** | 正確・虚偽なし |
  | Acceptable | **+0.5** | 有用だが軽微な誤り |
  | Missing | **0** | 「わかりません」等、具体的な答えなし |
  | Incorrect | **−1** | 誤答・無関係 |

- **戦略的含意**: 誤答(−1)は棄権(0)より **1点損**。→ **確信度ゲート付きの棄権**が有効（現設計の根幹）。
- **制約**: 各問 1000 トークン上限、前処理＋100問回答で 12時間目安、**ハードコード禁止**（汎用 RAG のみ評価）。

### 質問の性質（document-intelligence 型）
単純な本文検索では解けない。実例:
- Excel/PPT で**オレンジ/黄にハイライト**されたセル・数値・行
- Word 契約書の**太字**箇所抽出、**コメント**箇所抽出
- チャート PNG（ヒストグラム等）の**最大カウント読取**
- xlsx **PivotTable / AutoFilter** の集計条件
- `modeling.py` の**パラメータ値**、`01_eda.ipynb` の**相関**
- old 版 vs 最新版の**差分**、全案件**横断集計**
- **社内用語集**での略称展開、**座席表**順の人物列挙

→ 書式・図表・コードまで機械的に解釈する**抽出層**と、多段の**照合・集計・差分**が勝負所。

---

## 2. 現在の構成・採用手法（アーキテクチャ）

```
data/share_drive (418 files) ─► [1]抽出 ─► [2]索引 ─► [3]検索 ─► [4]生成(agent) ─► predictions.csv
```

- **[1] 抽出（`src/rag/extract/`）** — 形式別の**決定論的抽出**＋Gemini Vision。
  書式・構造シグナルを平文化せず**注記として保存**（検索可能化）: docx 太字→`【太字】`・
  ハイライト→`【ハイライト:色】`、xlsx セル塗り→HSV 粗色名（`C5(オレンジ): 値`）、pptx run XML の
  highlight/図形塗り、PDF は pdfplumber（表=パイプ行）＋画像 PDF はページラスタ、png はチャート、
  ipynb はセル＋**実行出力**、パスワード付き Office は用語集＋契約日から候補導出して透過復号。
- **[2] 索引（`src/rag/index.py`）** — チャンク化＋Vertex embeddings（text-embedding-005）＋BM25。
  **事前処理は単一コマンド `python -m src.rag.index`** で 5 ストア生成:
  retrieval index / evidence_index / canonical_manifest / corpus_profile（事前復号）/ structure_store
  （highlights/charts/pivots/seating/version_pairs）。
- **[3] 検索（`src/rag/retrieve.py`）** — dense+sparse の **RRF 融合ハイブリッド**＋用語集展開＋案件
  フィルタ。ファイル名トークン一致 2.2x、企業名→用語集で案件特定 1.6x ブースト。NFD→NFC 正規化。
- **[4] 生成（`src/rag/generate.py` / investigator エージェント）** — Vertex Gemini（`--gen opus` で
  Claude Opus 生成も可）。**根拠強制**（提供資料のみ・推測禁止）＋温度違い二重ドラフト自己一致＋
  厳格 verify（疑わしきは supported=false→棄権）＋**確信度ゲート棄権**。
- **決定論ハードモジュール**（LLM に計算させない）: `compute.py`（横断算術, pandas/Fraction）、
  `enumeration.py`（完全性ゲート付き列挙・部分列挙は棄権）、`diffpair.py`（old/new 構造 diff）、
  `pivotcond.py`（PivotTable/AutoFilter の XML 直読）。**sealed hold-out で実証済み archetype のみ直接
  commit**、未実証は advisory ヒント（過学習で未知案件を誤答する事故を防止）。
- **エージェントの探索ツール**: find_files / file_grep / canonical_route / read_office / caption_image /
  read_chart_values / compute / corpus_aggregate / version_diff / pdf_emphasis。適応ターン予算
  （`ADAPTIVE_MAX_TURNS=18`）＋spin 検出＋境界再探索＋file_grep per-call デッドライン。
- **3関門ハーネス（客観採点）**: 関門1=教師30問（公式GT）／関門2=自動生成＋1案件封印（過学習検知）／
  関門3=test100（実提出・真の汎化 KPI）。手元採点は公式 `evaluator.py` 移植、審査は Codex CLI(GPT-5.x)。
  **重要な教訓（SOT-2457）**: ローカル proxy は実 LB と **無相関(ρ=−0.09)** だった → 採用ゲートは
  **実 LB 確認のみ**。gold100 は方向性を見るための proxy。

---

## 3. 現在の採点結果（gold100 実測）

最新 = **SOT-2568**（2026-08-09, 全改善フラグ ON, gen=investigator, n=100, 判定=codex, cost $10.31）:

| 指標 | SOT-2550(前回最良) | **SOT-2568(最新)** | 差分 |
| --- | --- | --- | --- |
| match | 34 | **39** | +5 ✅ |
| abstain | 60 | **54** | −6 ✅ |
| wrong | 6 | **7** | +1 |
| **net(match−wrong)** | 28 | **32** | **+4** ✅ |
| baseline(手動 86.7%) | BELOW | BELOW | 未達 |

- 統合改善（回答増フラグ＋型別精度ゲート＋書式抽出強化）は前進。非一致 61 のうち **54(88.5%)は安全
  棄権を維持**し、precision の後退は wrong +1 に留めた。
- **注意**: これはオフライン proxy。実 LB は別途確認が必要（proxy と実 LB は歴史的に非相関）。

### 型別内訳（n / match / abstain / wrong）
```
fact_lookup         n=26  match=14  abstain=11  wrong=1
document_extract    n=24  match=11  abstain=11  wrong=2
derived_calculation n=32  match= 8  abstain=22  wrong=2   ← 最弱・棄権の主軸
enum_set            n= 9  match= 2  abstain= 7  wrong=0
version_diff        n= 6  match= 2  abstain= 2  wrong=2
config_hyperparam / data_shape / highlight_set  各 n=1
```
`fact_lookup`(14/26)・`document_extract`(11/24) が牽引。**derived_calculation は 8/32 と最弱**。

---

## 4. どの問いで誤り／棄権になっているか

### 4-1. 棄権 54 件の状態コード内訳
```
BUDGET_EXHAUSTED      27   ← 探索でターン/時間予算を溶かし推論に届かず
UNANSWERABLE          18   ← 「根拠なし」判定。多くは gold 実在＝抽出できていないだけ
NOT_RETRIEVED          5   ← retrieval miss
RETRIEVED_NOT_PARSED   2
PARSED_AMBIGUOUS       1
EVIDENCE_INCOMPLETE    1
```
**BUDGET_EXHAUSTED(27) + UNANSWERABLE(18) = 45/54 (83%)** が棄権の主軸。

代表的な棄権問（derived_calculation / enum_set）:
- idx6「提案時見込み金額と最終請求金額の差額」（2値を別文書から取り差分）
- idx8「米国平均給与 ML エンジニアとデータエンジニアの差」
- idx17「黄色ハイライトかつ RED の数値の上昇率」（書式条件＋多段計算）
- idx19「スケジュール r2 で 2025-08-11〜09-09 に開始/終了が入る項目」（enum/NOT_RETRIEVED）
- idx27「スコープ対象外項目はいくつあるか」（列挙クロージャ）
- idx30「標準化 loan_amnt<0 かつ purpose=credit_card の行…」（多段条件フィルタ集計）

### 4-2. 誤答 7 件（全件）
| idx | 型 | 誤り方 | pred → gold |
| --- | --- | --- | --- |
| 1 | version_diff | **実質的変更の取り違え**（契約サマリ追記を報告） | 追記した ≠ 性能比較表を削除 |
| 12 | fact_lookup | **指定ファイル未発見のまま別ファイルへ過剰推論** | 「見つからない」→別 doc | (gold=2ページ) |
| 14 | version_diff | **差分の対象取り違え** | Step4 追記等 ≠ 列名アンダースコア化 |
| 16 | document_extract | **抽出漏れ→「存在しない」誤断定**（黄ハイライト∧赤字, 文字色判別不可） | (gold=0.589) |
| 49 | document_extract | **docx コメント抽出機能なし→棄権的誤答** | (gold=WBS・進捗管理台帳確定) |
| 65 | derived_calculation | **条件の言語化を誤り**（件数「8件」で回答） | (gold=相関 <−0.99 のセル) |
| 76 | derived_calculation | **what-if 計算の式/丸め誤り** | 73,260円 ≠ 79,200円の増加 |

誤答の主群は **version_diff の実質変更選別（1/14）** と **過剰推論/機能欠損（12/16/49）** と
**多段数値の計算誤り（65/76）**。

---

## 5. なぜ正答できないか（根本原因の4系統）

1. **探索律速（retrieval-turn starvation）— BUDGET_EXHAUSTED の真因**
   実トレース集計で **tool ターンの 90%（182/202）が retrieval**（find_files/file_grep/canonical_route/
   read_office/caption_image）に消費され、compute/集計/差分など**推論は 10%**。適応予算 18 ターンでも
   探索だけで使い切り、多段の集約/列挙/数値導出に入る前に棄権する。問い文は対象を**固有名で明示**
   （企業名・ファイル名）しているのに、その固有名→canonical 文書の**事前解決が回答ループ内**に残るため、
   探索が予算を溶かす。（`sot2563_intime_abstain_root_cause.md` で検証済。timeout 棄権は PR#101/#103 で
   0 化済＝残るのは純粋なターン浪費。）

2. **書式・視覚抽出ギャップ — UNANSWERABLE の主因**
   UNANSWERABLE 18 の多くは gold が実在し、**抽出できていないだけ**。
   - **条件付き書式（CF）のルール抽出**: idx65 は「黄色＝相関<−0.99」という **CF のルール**を答える問い。
     セルの見た目色は `cell.fill` に出ず **dxf の bgColor＋cfRule の数式**に埋まっており、面に出ていない。
   - **文字色（赤字）** idx16、**docx コメント** idx49 は現状ツール層に抽出能力がない。
   - 複合書式条件（黄ハイライト∧赤字）を同時に満たす対象の特定ができない。

3. **多段数値の計算到達性・正確性 — derived_calculation が最弱(8/32)**
   (a) 探索で予算切れして compute に**到達しない**、(b) 到達しても**対象行/式/丸めを誤る**
   （idx76: 79,200 を 73,260 と誤算）。what-if・差÷差・回帰予測小数第5位・多段条件フィルタ集計は、
   証拠が席次にあっても**推論深さ（compute の連鎖）**が予算・正確性の両面で不足。

4. **版差分の“実質的変更”選別 — version_diff の wrong(1/14)**
   構造 diff は取れているが、**どの atomic 変更が「案件遂行に関連する実質的変更」か**の選別を誤る
   （ボイラープレートのサマリ追記を実質変更として報告し、本命の「性能比較表削除」「列名の
   アンダースコア化」を落とす）。

> **横断する制約**: 採点が Incorrect=−1 のため設計は**疑わしきは棄権**に倒しており、①〜④が解けない問いは
> 誤答ではなく棄権に落ちる。したがって**スコアの伸びしろは棄権 54 の回収**にあり、その回収は
> ①探索圧縮 ②書式抽出 ③計算到達性 ④差分選別 の4系統に一致する。

---

## 6. ディープリサーチに基づく対策候補（優先度付き）

4系統の根本原因（§5）に対し、2023–2026 の SOTA 文献・OOXML 仕様・ライブラリ実装を調査した結果を、
本システムへの適用形にして提示する。**採用は必ず opt-in フラグ→gold100 A/B→実 LB 確認**のゲートを通す
（ローカル proxy は実 LB と非相関＝proxy 単独で昇格しない, SOT-2457）。

### 対策A【最優先】探索圧縮 — 固有名の“回答ループ前”事前解決＋計画先行
> 対象: BUDGET_EXHAUSTED 27（棄権の 50%）。効果最大・リスク最小（既存 canonical_route の前倒し）。

- **A1. エンティティ事前解決（entity pre-resolution）**: 問い文の固有名（企業名「青葉与信マネジメント」・
  ファイル名「train.xlsx」）を回答ループ**開始前**に NER＋文字列/別名類似（用語集 alias, Levenshtein）で
  canonical 文書へ解決し、**該当スライスを席次へ pre-inject**。探索を ~10 ターン→~1–2 ターンへ圧縮し、
  空いた予算を推論へ回す。既存 `canonical_route`（モデルが選ぶツール）を「回答前の決定論ステップ」へ前倒し
  するだけで、SOT-2494/2498 の延長。
  出典: *An Entity Linking Agent for QA*（arXiv:2508.03865）, *RAG + Entity Linking*（arXiv:2512.05967）,
  *Agentic-RAG for Fintech*（用語/略称の事前解決, arXiv:2510.25518）。
- **A2. 計画先行アーキテクチャ（plan-before-execute）**: ReAct の毎ターン再決定を廃し、**Planner が
  ツール鎖を1パスで生成→Worker 実行→Solver 推論**。冗長探索の再プロンプトが消える。
  出典: *ReWOO*（arXiv:2305.18323）, *LLMCompiler*（DAG 化・並列/重複排除）,
  *FAIR-RAG*（必要findingsのチェックリスト化→ギャップだけ targeted sub-query, arXiv:2510.22344）。
- **A3. 予算認識ターン配分（budget-aware）**: 残予算をエージェントに供給し「深掘り vs 撤退」を判断させる。
  予算“増やすだけ”は天井に当たる＝**認識が Pareto を動かす**。失敗軌跡の早期停止で 28–64% トークン削減。
  出典: *Budget-Aware Tool-Use / BATS*（arXiv:2511.17006）, *BAGEN*（arXiv:2606.00198）,
  *ReaLM-Retrieve*（retrieval 呼出 47%減, arXiv:2604.26649）。
- **A4. 空回り検出（anti-flailing）**: 実証則「**2回の retrieval で 5回分の 95% の gains**」＝反復に
  ハード上限。既 grep 済みへの再走査を検知して戦略切替。
  出典: *Dissecting Agentic RAG*（arXiv:2606.21553）, *InferAct*（arXiv:2407.11843）。

### 対策B【高】書式・構造抽出ギャップの解消 — 条件付き書式の“ルール”／コメント／文字色
> 対象: UNANSWERABLE 18 の多く＋idx65/16/49。gold は実在＝抽出できれば match 化。

- **B1. 条件付き書式(CF)の“ルール”抽出**: 色は `cell.fill` に出ず **dxf の bgColor＋cfRule の数式**に埋まる
  （SOT-2564 の gotcha と一致）。openpyxl `ws.conditional_formatting` を走査し `rule.type/operator/formula`
  を読めば「**value < −0.99 のセル**」を**ルールとして復元**（idx65 = 相関<−0.99 が直接解ける）。加えて
  データ領域 dataframe 上で述語を再評価し**実際に該当するセルを列挙**（Excel が焼いた色を鵜呑みにしない）。
  出典: openpyxl formatting/rule API。
- **B2. docx コメント抽出**: python-docx **≥1.2.0** で `Document.comments` が一級対応。範囲は
  `w:commentRangeStart/End` の id で本文 run に対応付く（idx49「コメントがついた部分を抽出」が解ける）。
  旧版は `zipfile`+`ElementTree` で `word/comments.xml` を直読する fallback。
  出典: python-docx comments docs, `docx-comments-to-text`。
- **B3. 文字色＋ハイライトの複合条件**: `run.font.color`(`w:color`) と `run.font.highlight_color`(`w:highlight`)
  を読む。**gotcha**: セル/段落 shading（`w:shd@fill`）は python-docx が露出しない別要素＝raw XML 直読が必要。
  仕様上 highlight は shd に優先。idx16「黄ハイライト∧赤字」は `highlight==YELLOW ∧ color==red`（無ければ
  `w:shd@fill` へ fallback）の複合述語で判定。
- **B4. PivotTable/AutoFilter 条件**: `xl/pivotTables/*.xml`（filters/pageField/dataField）と sheet の
  `<autoFilter>` を直読（`pivotcond.py` の延長）。フラット化テキストに無い集計/抽出条件を面に出す。
- **B5. 表構造の圧縮エンコード**: 面に出した書式を LLM へ渡す際は **SpreadsheetLLM/SheetCompressor** の
  format-aware aggregation（隣接同フォーマット領域をまとめる）で「ハイライト領域」を1属性で表現し、
  セル毎平文化をやめる。出典: *SpreadsheetLLM*（arXiv:2407.09025）。

### 対策C【高】多段数値の計算到達性・正確性 — 強制 PoT lane＋実行結果の多数決＋二重検証
> 対象: derived_calculation 8/32（最弱）・idx76 型の式/丸め誤り。

- **C1. numeric 契約を強制 PoT レーンへ**: 一般エージェント（探索で予算浪費）でなく、**Router で numeric を
  plan-first レーンに固定**。まず**変数束縛表＋名前付き数式**を出し、束縛量だけ retrieval → 必ず compute に
  到達（対策A と相乗で探索律速を断つ）。CoT→PoT は FinQA で **40.4%→64.5%**。
  出典: *PoT*（openreview YfZ4ZPt8zd）, *PAL*, *FinAgent-RAG*（retriever→PoT→executor＋router, arXiv:2605.05409）,
  *PTR/PAR-RAG*（plan→execute→verify, 2–3 LLM 呼で予算上限, arXiv:2604.04131 / 2504.16787）。
- **C2. 実行結果の自己一致（PoT + self-consistency）**: N=3–5 個のプログラムを生成・実行し、**実行された
  数値**で多数決（テキストでなく）。誤った式は多数決で負ける＝idx76 の 73,260 vs 79,200 を捕捉。
  出典: *Universal Self-Consistency*（arXiv:2311.17311）。
- **C3. 二重ライブラリ検証ゲート**: 返却前に sympy 等の**別ライブラリで再計算**＋単位/丸め正規化、不一致は
  再生成（compute.py を `Fraction`/`Decimal` のまま第2実装で照合）。
  出典: *IMP-TIP*（GSM8K-Hard 56→65%, arXiv:2401.05384）, *VSI*（sympy ステップ検証, arXiv:2603.21558）。
- **C4. ステップ毎 rerank**: 対象行/数値が compute に届くよう推論ステップ毎に証拠を rerank（対象行取り違え
  対策）。出典: *DyRRen*（arXiv:2211.12668）。

### 対策D【中】版差分の“実質的変更”選別 — block 単位 diff＋編集意図分類
> 対象: version_diff wrong（idx1/14）。構造 diff は取れているが選別を誤る。

- **D1. ブロック/表サブツリー単位 diff**: 行ラベル行ではなく **table-block / slide-shape 単位**で対応付け、
  「表サブツリーごと削除」を1つの atomic イベントに（`diffpair.py` の粒度を上げる）。tree-edit distance を
  変更規模の特徴量に。出典: *BlockDiff/FuncDiff*（arXiv:2604.27296）, *TSED*（AST edit distance, arXiv:2404.08817）。
- **D2. 編集意図分類（Edit Intent Classification）**: 各 atomic edit を LLM で
  **substantive vs boilerplate**（Wikipedia 編集分類の Content add/delete ⇔ Copy-Edit/Clarification）に分類し、
  boilerplate（サマリ追記等）を降格、substantive（性能比較表削除・列名アンダースコア化）を上位に。keep 閾値は
  conformal で較正。出典: *EIC / Are LLMs Good Classifiers*（arXiv:2410.02028）,
  *Conformal Importance Summarization*（arXiv:2509.20461）。

### 6-1. 推奨実装順（follow-up 子 Issue 候補）

| 順 | 子 | 対象棄権/誤答 | 期待 | リスク |
| --- | --- | --- | --- | --- |
| 1 | **A1 固有名 pre-resolve＋スライス pre-inject** | BUDGET 27 の中核 | 探索圧縮で abstain→match | 低（canonical 前倒し・byte-identical opt-in） |
| 2 | **B1 CF ルール抽出 ＋ B3 文字色/shd** | UNANSWERABLE 18・idx65/16 | 書式問の面出し | 低（抽出層追加, 既定 OFF） |
| 3 | **C1+C2+C3 numeric PoT lane＋二重検証** | derived_calc 8/32・idx76 | 到達性＋正確性 | 中（生成経路変更・A/B 必須） |
| 4 | **B2 docx コメント抽出** | idx49 等 | 機能欠損の解消 | 低（ライブラリ更新） |
| 5 | **A3 budget-aware＋A4 anti-flailing** | BUDGET 残 | 予算再配分・早期棄権 | 中（エージェント制御変更） |
| 6 | **D1+D2 block diff＋編集意図分類** | version_diff idx1/14 | 実質変更選別 | 中（分類器の過学習注意） |

> **検証規律（必須）**: 各子は ①既定 OFF opt-in フラグ（champion serve は byte-identical 維持）、
> ②focused/offline A/B → gold100 A/B、③**採用の最終ゲートは実 LB 確認のみ**（proxy 単独昇格禁止, SOT-2457）、
> ④sealed hold-out 実証済 archetype だけ直接 commit・未実証は advisory（未知案件への過学習事故防止）。
> saturation は blocker ではない＝エスカレーションラダー（局所調整→データ/oracle 再構築→アーキ変更→外部知識）
> を歩む。ハードコード（設問別固定回答）は課題規約違反のため禁止。



---

## 7. 参考（一次ソース）

- `docs/ai/gold100_sot2568_result.md` — 最新実測（net32）
- `docs/ai/abstain_wrong_root_cause_SOT-2550.md` — 棄権/誤答の実トレース分析
- `docs/ai/sot2563_intime_abstain_root_cause.md` — retrieval-turn starvation の検証
- `artifacts/gold_100_review.csv` — 問別 gold/回答/status/state_code（Private-strict, 追跡対象）
- `README.md` — 設計根拠（Design Rationale ①〜⑪）・3関門ハーネス
