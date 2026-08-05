# Gemini-only 早期検証ゲート 結果 (SOT-2467, Step1)

親PLAN **SOT-2460** の早期検証ゲート。**汎用ツールのみ**（pw/略称/書式規則を非注入）を function-calling で
`gemini-2.5-pro` に接続し、型の異なるゴールド5問を **offline で自力再導出**できるかを測定した。

- ハーネス: `scoring/early_gate.py`（`python -m scoring.early_gate` で実行、`scoring/test_early_gate.py`）
- 実行: `gemini-2.5-pro` / `--max-turns 12` / temperature 0 / thinking 1024
- 生成物: `artifacts/early_gate.json`（gitignore対象・実行時再生成）、`docs/ai/experiment_ledger.jsonl` に台帳追記

## 判定: **GO ✅ 4/5**（閾値 4/5）

| type | 結果 | 反復(tool rounds) | tokens | cost(USD) | 主な使用ツール | 備考 |
| --- | --- | --- | --- | --- | --- | --- |
| decrypt (復号)   | ✅ | 4 | 18,786 | 0.032 | find_files→file_grep→read_office | **pwをファイル名から自力発見**して復号し税込見込金額を取得 |
| compute (計算)   | ✅ | 3 | 7,638  | 0.017 | find_files→compute | pandas式で loan_amnt 平均=1526 を厳密計算 |
| enumerate (列挙) | ✅ | 6 | 12,539 | 0.020 | find_files→read_office | PLからタスクID群 T09–T12 を列挙 |
| chart (図表)     | ✅ | 7 | 16,660 | 0.038 | caption_image→compute | vision読取をcomputeで裏取りし「20日」を特定 |
| format (書式)    | ❌ | 12(上限) | 66,088 | 0.116 | grep/pdf_emphasis/read_office | pptxのマーカー語を特定できず棄権（上限到達） |
| **合計** | **4/5** | **32** | **113,635 in + 8,076 out** | **0.223** | — | — |

## Step2 設計への反映（受け入れ条件②）

1. **Gemini-only + ツール駆動は成立**：復号(pw自力発見)/計算/列挙/図表の4型は汎用ツールで再導出可能。
   過去のGemini失敗は「テキスト専用RAG(ツール無し)」起因という仮説を支持する。
2. **プロンプト感度が高い**：初回試行は2/5(compute・chart棄権)。「計算は必ずcompute」「列名/値を先に確認」
   「エラーでも諦めず再試行」「図はread_chart_values→visionの順」を明示しただけで4/5へ改善。Step2の調査
   エージェントは**手続き指示（列確認→式）とエラー回復ループ**を組み込むべき。
3. **未カバー領域＝書式(マーカー/ハイライト)抽出**：pptx内のハイライト語を取る汎用ツールが無く format が
   律速。Step2で **pptx/EMFのハイライト語抽出ツール**（`pdf_faux_italic`/`emf_pivot` の pptx-テキスト
   ハイライト版）を追加する必要がある。
4. **コスト目安**：1問あたり平均 ~$0.045 / ~6.4 tool rounds（`gemini-2.5-pro`, thinking込み）。100問規模で
   ~$4.5/パス。異種検証(2パス)や flash 併用でのコスト最適化はStep3以降で検討。
5. **移植性**：ゴールド/ツールにコーパス固有の秘密(pw・略称)を非同梱。復号pwは実行時に自力発見して
   `corpus_profile.json`（gitignore）にのみキャッシュ。
