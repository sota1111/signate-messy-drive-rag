# cycle2 4フラグ単独アブレーション (SOT-2634)

**目的**: cycle2 統合 (SOT-2622, 4フラグ同時ON, net 40→28 rejected) では交絡で採否未確定だった
`RAG_SPIN_PIVOT` / `RAG_SEARCH_CAP` / `RAG_OPERAND_PREFILL` / `RAG_CONDITION_PREIR` を **単独ON** で
focused 実測し、SOT-2629 の介入テレメトリ (`details.interventions`) で **発火→verdict の因果**を確認して
per-flag に **KEEP / DROP / 要改修** を確定する。cycle3 統合 (SOT-2636) に入れるフラグ集合を決める。

## 測定条件（全て OFFICIAL）

- モデル: `gemini-3.6-flash` @ `global` (GEN/HARD/VISION 全て flash 3.6, judge=codex) — `run_focused_gate.py`
  のモデルガード合格 (`official:true`)。`--dev` 不使用。
- 構成: champion **Wave A net40** ベース (`RAG_DET_PIPELINE_ROUTER=1` + B1-only) に、cycle2 の4フラグを
  **単独で1つずつON**（他3つはOFF）。ドライバ: `scripts/sot2634_ablation.sh <flag>`。
- 共通 focused set (22 target): 各フラグの元 target idx ∪ cycle2 誤答 idx `{4,9,12,24,47,73,83,94,99}`
  = `4,6,8,9,12,24,27,28,32,38,47,50,57,63,67,73,76,83,87,94,98,99`。＋ 番兵10問 (SOT-2623, 自動同梱)。
- **OFF baseline**: 4フラグ全OFFの champion Wave A を同一 focused set で実測し、各 target の OFF基準と
  番兵の非決定ノイズ床を取得（`sot2634_baseline`）。これが無いと単発 flash3.6 の非決定性で
  「改善/回帰」を誤帰属する。
- 発火の読み方 (SOT-2629): `interventions` の該当キーが**存在すれば ON**（値0/false=ON-but-idle,
  不在=OFF/前提非成立）。target 行の `interventions` を `run_focused_gate.py` に carry する追記を実施
  （測定ハーネスのみ・serve path 不変）。

## 番兵回帰は flash3.6 非決定ノイズ（flag起因ではない）

| run | 番兵 MATCH | gate | 回帰 idx (発火状態) |
|---|---|---|---|
| baseline (全OFF) | 9/10 | FAIL | 30 (—, flag無し) |
| spin_pivot | 8/10 | FAIL | 30 (spin_pivot=2), 16 (spin_pivot=0 idle) |
| search_cap | 10/10 | PASS | — |
| operand_prefill | 9/10 | FAIL | 30 (idle, 未発火) |
| condition_preir | 9/10 | FAIL | 16 (idle, 未発火) |

**idx30 は全OFF baseline でも回帰** → idx16/30 は非決定な脆弱番兵で、回帰は flag 起因ではない。
spin_pivot の idx30 は spin_pivot=2 発火と同時だが、同 idx が baseline でも回帰するため発火は交絡で
因果とは言えない（発火因果は下記 target 側で判定する）。**よって「番兵回帰ゼロ」基準は非決定床に
埋もれるため、KEEP/DROP は主に target の OFF→ON 発火因果で判定した。**

## target OFF→ON（baseline=OFF基準・`*n`=当該フラグ発火数）

| idx | baseline(OFF) | spin_pivot | search_cap | operand_prefill | condition_preir |
|---|---|---|---|---|---|
| 4  | Incorrect | Incorrect `*1` | **Perfect** (非発火) | Incorrect | Incorrect |
| 8  | Missing | **Incorrect `*3`** | Incorrect `*1` | Missing | Incorrect |
| 9  | **Perfect** | Perfect | **Missing `*1`** | Perfect | Perfect |
| 12 | Missing | Missing `*2` | Missing `*1` | **Perfect** (非発火) | Missing |
| 24 | Missing | Missing | Missing | Missing | Incorrect |
| 28 | Missing | **Incorrect `*3`** | Missing `*1` | Missing | **Perfect** (非発火) |
| 76 | Missing | Missing `*4` | **Incorrect `*2`** | Missing | Missing |
| 83 | Missing | **Incorrect `*4`** | Missing `*1` | Missing | Missing |
| 94 | **Perfect** | **Incorrect** | **Missing** | **Incorrect** | **Missing** |
| その他(6,27,32,38,47,50,57,63,67,73,87,98,99) | 全 Missing/Incorrect | 改善なし | 改善なし | 改善なし | 改善なし |

target MATCH 到達: baseline `{9,94}` / spin `{9}` / cap `{4}` / prefill `{9,12}` / preir `{9,28}`。
**baseline 比で新規 MATCH に到達した idx はいずれも当該フラグが非発火** → flag-attributable な改善はゼロ。
idx94 (baseline Perfect) は全フラグ run で MATCH を落としており、これは非決定の揺れ幅。

## per-flag verdict

### RAG_SPIN_PIVOT (SOT-2614) → **DROP**
- 発火は広範（全 llm route で pivot 発火 `*1〜*5`）。**発火→WRONG の因果を確認**: baseline で abstain
  だった idx8/28/83 が spin発火 `*3/*3/*4` で **Missing→Incorrect**（棄権すべき所で誤答を commit）。
- flag-attributable な改善ゼロ（target 新規MATCH無し）。**SOT-2614 REJECT を単独ONで再確認**（早期
  打ち切りが precision を崩す）。cycle3 統合に入れない。

### RAG_SEARCH_CAP (SOT-2620) → **DROP**
- 番兵は PASS(10/10) だが **発火→harm の因果を確認**: baseline Perfect の **idx9 が cap `*1` 発火で
  Missing に転落**（必要な search を withhold）、idx76 は cap `*2` 発火で Incorrect。
- 唯一の gain idx4 (Incorrect→Perfect) は **cap 非発火＝非帰属**（model variance）。改善なし＋発火harm
  → DROP。**SOT-2620 REJECT を再確認**。cycle3 統合に入れない。

### RAG_OPERAND_PREFILL (SOT-2616) → **DROP（inert）**
- **focused 22 target で発火0件**。canonical NUMERIC target(76/47/57/6/8/50/63/99) は route=`llm`・
  大半 stop=`max_turns`(BUDGET_EXHAUSTED) だが、operand-prefill preamble は候補を注入していない
  （NUMERIC分類 or operand構造の build-time 前提が champion Wave A で非成立）。
- gain(idx12 Missing→Perfect) は非発火＝variance。**改善が flag に帰属しない** → DROP。
  SUSPECT (SOT-2616/2621) を **証拠付きで DROP に確定**。cycle3 統合に入れない。
- 要改修メモ: 効かせるには NUMERIC route 分類の網羅か structure-store operand 抽出の前提充足が先。
  だが本丸の失点は `max_turns` の retrieval/budget starvation で、本フラグはそれを解消しない。

### RAG_CONDITION_PREIR (SOT-2621) → **DROP（inert）**
- **focused 22 target で発火0件**。what-if 対象 idx76/47/57/6/27 でも `build_condition_ir` が IR を
  構築せず（NUMERIC route 前提が非成立）。gain(idx28 Missing→Perfect) は非発火＝variance。
- 改善が flag に帰属しない → DROP。SUSPECT を **証拠付きで DROP に確定**。cycle3 統合に入れない。
- 要改修メモ: what-if 検出/NUMERIC 分類が champion Wave A の route で発火しないことが根本。効かせるには
  route 前段での condition 検出配線が必要。

## 結論 — cycle3 統合 (SOT-2636) に入れるフラグ集合

**4フラグとも DROP。cycle3 統合構成には cycle2 の4フラグを一切入れない。**

- spin_pivot / search_cap: 発火するが precision を落とす（abstain→WRONG, Perfect→Missing）。
- operand_prefill / condition_preir: champion Wave A の route 前提で発火せず inert（改善は非帰属の variance）。

これは cycle2 統合 net28 rejected と敵対的レビュー（spin/cap 犯人説 trace 反証・SUSPECT暫定）を、単独ON＋
発火テレメトリ＋OFF baseline で**因果的に確定**したもの。cycle3 は trace移植 (SOT-2631-33) と EUゲート
(SOT-2635) に集中し、これら4フラグはレジストリ上 OFF のまま据え置く。

## 生成物

- 実行ドライバ: `scripts/sot2634_ablation.sh`（champion Wave A + 単独フラグ、`run_focused_gate.py` 経由）
- gate JSON: `artifacts/focused_gate_sot2634_{spin_pivot,search_cap,operand_prefill,condition_preir,baseline}.json`
- 実行ログ: `artifacts/sot2634_logs/*.log`
- ledger: `docs/ai/experiment_ledger.jsonl`（gate自動5行 official=true ＋ per-flag verdict 4行）
