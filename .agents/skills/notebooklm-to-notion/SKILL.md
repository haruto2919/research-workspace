---
name: notebooklm-to-notion
description: NotebookLMで12軸分析した論文マークダウンをNotionの「PECL論文 12軸分析DB」に登録しつつローカルにSSOT保存するスキル。必ず `/notebooklm-to-notion` で明示起動する。起動するとpapers/に一時ファイルを作成して貼り付けを待機し、ユーザーが報告したら解析・Notion登録・リネームまで自動で行う。NotebookLMの出力を渡されたとき、「論文をNotionに入れて」「12軸分類の結果を登録して」と言われたときにも提案してよいが、実行は必ず明示的なコマンド起動を求めること（自動実行はしない）。
---

# NotebookLM → Notion 保存スキル

NotebookLM の12軸論文分析プロンプト（[notebooklm-prompt-axis-analysis.md](../../../.company/lab/projects/lora-dynamic-separation/references/notebooklm-prompt-axis-analysis.md)）で得たマークダウンを、Notion の 12軸 Multi-select DB に登録しつつ、ローカルに SSOT（Single Source of Truth）として保存するスキル。

## 原則

- **SSOT はローカル MD**: `papers/<paper-id>-<slug>.md` が正本。Notion は閲覧・フィルタ用のビュー
- **入力ファイルをそのままSSOT化**: リネームのみ行う。内容は NotebookLM 出力そのままを維持し、frontmatter 等は付与しない
- **papers/ への事前配置が前提**: 呼び出す前に NotebookLM 出力を `papers/` フォルダに置いておく
- **明示起動のみ**: 自動トリガーは設けない。誤起動でゴミが増えるリスクを避ける
- **OOS 論文の分離**: `[OUT-OF-SCOPE]` プレフィックスで始まる分析結果は Notion 投入をスキップし、ローカルリネームのみ行う
- **タグは whitelist 照合**: 表記揺れ（`高σ` vs `高 σ`）を Notion の Multi-select が新規タグとして量産するのを防ぐ

---

## 設定

> **更新時の注意**: DB を別のものに差し替える、またはプロパティ名を変更した場合はこのセクションと下記 `PROPERTY_NAMES` / `TAG_WHITELIST` を更新する。

### Notion DB

| 項目 | 値 |
|---|---|
| `NOTION_DATA_SOURCE_ID` | `34a84249-7e4a-8032-af2d-000bb821f6da` |
| `NOTION_DB_URL` | `https://www.notion.so/34a842497e4a806f8c83ca553c38a0b9` |
| `PAPERS_DIR` | `.company/lab/projects/lora-dynamic-separation/papers/` |

### PROPERTY_NAMES（Notion 側の実プロパティ名）

**メタデータ（title + rich_text + number + select + date）**
- `論文ID` (title)
- `論文タイトル` (rich_text)
- `発行年` (number)
- `会議` (select): `ICLR` / `NeurIPS` / `ICML` / `CVPR` / `ECCV` / `ICCV` / `AAAI` / `arXiv` / `その他`
- `手法略称` (rich_text)
- `ベンチマーク` (rich_text)
- `ベースライン` (rich_text)
- `登録日` (date)

**12軸（すべて multi_select）**
- `軸1_保護対象` / `軸2_重要度尺度の性質` / `軸3_情報源` / `軸4_干渉防止機構` / `軸5_モジュール構造` / `軸6_計算タイミング` / `軸7_制約強度` / `軸8a_対象行列` / `軸8b_帯域指向` / `軸8c_処理方向` / `軸9_忘却の方向性` / `軸10_評価プロトコル`

**制御列**
- `スコープ内` (checkbox) — OOS は false（OOS の場合は Notion に投入しないので通常の論文では常に true）
- `ソース` (select): `NotebookLM` / `Manual`

### TAG_WHITELIST（12軸の有効タグ値）

この whitelist と一致しないタグ値を検出したら Phase 3 でユーザーに確認する。

| プロパティ名 | 有効タグ値 |
|---|---|
| `軸1_保護対象` | `PTM知識` / `過去タスク知識` / `両方` / `該当なし` |
| `軸2_重要度尺度の性質` | `静的` / `動的` / `ハイブリッド` |
| `軸3_情報源` | `事前学習W` / `入力共分散` / `タスク勾配` / `全空間Fisher` / `パラメータ直接` |
| `軸4_干渉防止機構` | `勾配射影` / `空間配属` / `損失正則化` / `スパース非衝突` / `重み凍結` / `デュアルブランチ` / `知識分解` |
| `軸5_モジュール構造` | `タスク固有` / `共有単一` / `ハイブリッド` |
| `軸6_計算タイミング` | `事前` / `オンライン` / `事後` |
| `軸7_制約強度` | `ハード` / `ソフト` / `動的` |
| `軸8a_対象行列` | `W` / `ΔW` / `タスク勾配` / `複合` / `区別なし` |
| `軸8b_帯域指向` | `高σ` / `中σ` / `低σ` / `Nullspace` / `全帯域` / `区別なし` |
| `軸8c_処理方向` | `侵入防止` / `更新誘導` / `共有活用` / `限定更新` / `該当なし` |
| `軸9_忘却の方向性` | `Backward Forgetting防止` / `Forward Forgetting防止` / `両立` / `非該当` |
| `軸10_評価プロトコル` | `CIL` / `TIL` / `DIL` / `General-Incremental` |

whitelist の出典: [notebooklm-prompt-axis-analysis.md](../../../.company/lab/projects/lora-dynamic-separation/references/notebooklm-prompt-axis-analysis.md) 末尾「12軸の定義一覧」表。

---

## 起動方法

```
/notebooklm-to-notion
```

引数は不要。起動すると `papers/` に一時ファイルを自動作成し、NotebookLM 出力の貼り付けを待機する。ユーザーが「完了」と報告したら、残りのフロー（解析→Notion登録→リネーム）が自動で走る。

---

## Workflow

### Phase 1: 起動・前提チェック・一時ファイル作成

#### 1.1 前提チェック（起動ブロッカー）

以下が NG なら理由を明示して停止する。

1. **Notion MCP の可用性**: `ToolSearch` で `mcp__claude_ai_Notion__notion-create-pages` を取得できるか確認する。surface されていなければ「Notion MCP が未接続です。`~/.claude/mcp.json` を確認してください」と返して終了。
2. **設定の妥当性**: `NOTION_DATA_SOURCE_ID` がプレースホルダ（空や `TODO`）のままなら「SKILL.md の設定を更新してください」と返して終了。

#### 1.2 一時ファイル作成と貼り付け依頼

`PAPERS_DIR` に一時ファイルを `Write` で作成する。

- **ファイルパス**: `<PAPERS_DIR>/notebooklm-tmp.md`
- **初期内容**: コメント1行 `<!-- ここに NotebookLM 出力を貼り付けてください -->`
- `PAPERS_DIR` が存在しない場合は `Bash(mkdir -p <PAPERS_DIR>)` で作成する

作成後、ユーザーに以下のメッセージを送って**待機する**:

```
準備できました！

📄 以下のファイルに NotebookLM の出力を貼り付けてください:
<PAPERS_DIR>/notebooklm-tmp.md

貼り付けが完了したら「完了」とお知らせください。
```

### Phase 2: 貼り付け確認とマークダウン解析

ユーザーが「完了」（または「貼った」「入れた」など完了を示す言葉）を送ったら処理を再開する。

#### 2.1 ファイル読み込み確認

`Read` で `<PAPERS_DIR>/notebooklm-tmp.md` を開く。内容が空またはコメント行のみの場合は「ファイルが空のようです。貼り付けを確認してから再度「完了」とお知らせください」と返して再待機する。

#### 2.2 OOS 早期判定

ファイル先頭（空白・BOM を除く最初の非空行）が `[OUT-OF-SCOPE]` で始まる場合、Notion 投入はスキップする。Phase 3〜4 を飛ばして Phase 5（リネーム）に直行する。

#### 2.3 通常解析

Part 1 の構造を前提に、以下を抽出する。

- **Part 1-A メタデータ**: `論文ID` / `発行年` / `会議` / `主要ベンチマーク` / `主要ベースライン` / `手法略称` / `論文タイトル`
- **Part 1-B 12軸タグ表**: 各軸のタグ値（`/` 区切りで複数タグあり）

抽出に失敗した場合は「Part X が解析できません。ファイル構造を確認してください」と停止する。

### Phase 3: 正規化

1. **タグ正規化**: 抽出した各軸のタグ値を `TAG_WHITELIST` と照合する。完全一致しない場合はユーザーに提示する:

   ```
   軸8b_帯域指向 の値 "高 σ" が whitelist にありません。
   候補: "高σ"（スペースなし）

   どうしますか？
   (1) "高σ" に正規化して続行
   (2) "高 σ" のまま新規タグとして投入
   (3) 中止
   ```

2. **スラグ化**: `論文ID` と `手法略称` から `<paper-id>-<slug>` を作る。`slug` は `手法略称` を lowercase + 英数字以外を `-` に置換（例: `KeepLoRA` → `keeplora`、`N-LoRA` → `n-lora`）。手法略称に `/` で複数の名称が含まれる場合は最初のものを使う。

3. **発行年の整数化**: 文字列なら整数に変換する。変換できなければ省略する（Number プロパティに文字列を送るとエラーになるため）。

4. **会議値の照合**: `会議` が select の候補（ICLR / NeurIPS / ...）に無ければ `その他` に寄せる。

### Phase 4: Notion ページ作成

`mcp__claude_ai_Notion__notion-create-pages` を1回だけ呼ぶ。**`content` は送らない**（Notion はプロパティ一覧用。本文は手動でローカルファイルから確認する）。

```
parent:  { type: "data_source_id", data_source_id: "<NOTION_DATA_SOURCE_ID>" }
pages:   [
  {
    properties: {
      "論文ID": "<paper_id>",
      "論文タイトル": "<title>",
      "発行年": <year_int>,
      "会議": "<venue>",
      "手法略称": "<method>",
      "ベンチマーク": "<benchmark>",
      "ベースライン": "<baseline>",
      "ソース": "NotebookLM",
      "date:登録日:start": "<YYYY-MM-DD>",
      "スコープ内": "__YES__",
      "軸1_保護対象": "<JSON配列文字列>",
      ...
      "軸10_評価プロトコル": "<JSON配列文字列>"
    }
  }
]
```

ポイント:
- multi_select プロパティは **JSON 配列を文字列化**して渡す（例: `'["勾配射影", "空間配属"]'`）
- checkbox は `"__YES__"` / `"__NO__"` のリテラル文字列
- date は `date:<列名>:start` の形式で渡す

呼び出し成功後、レスポンスからページ URL を取り出して保持する。

**失敗時**: ローカルリネームは続行し、エラー内容を報告する。

### Phase 5: ローカルリネーム

`<PAPERS_DIR>/notebooklm-tmp.md` を `<PAPERS_DIR>/<paper-id>-<slug>.md` にリネームする。**内容は一切変更しない。**

```bash
mv <PAPERS_DIR>/notebooklm-tmp.md <PAPERS_DIR>/<paper-id>-<slug>.md
```

リネーム先が既に存在する場合（同じ論文の二重登録）は、上書きするか確認する。

OOS の場合も同様にリネームのみ行う（`<paper-id>-<slug>` は判読可能な範囲で命名）。

### Phase 6: 完了報告

```
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  NotebookLM → Notion 完了
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

✅ Notion ページ: [<論文ID>](<notion_page_url>)
✅ ローカル保存: papers/<paper-id>-<slug>.md

登録内容:
- 手法: <手法略称> (<会議> <年>)
- 12軸: <軸1〜軸10 のタグ値を / 区切りで1行サマリ>

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
```

OOS の場合は Notion 行を省き `⚠️ OOS のため Notion 投入をスキップしました` と明示する。

---

## エラーハンドリング

| 症状 | 対応 |
|---|---|
| Notion MCP が応答しない | Phase 5 のリネームは続行、Notion 失敗を報告 |
| whitelist 外タグが Notion に自動作成された | ユーザーに不要タグの削除を案内する |
| リネーム先が既に存在 | 上書きするか確認する |

---

## 参考

- 設計書: [.company/secretary/notes/brainstorm/2026-04-22-notebooklm-to-notion-skill-plan.md](../../../.company/secretary/notes/brainstorm/2026-04-22-notebooklm-to-notion-skill-plan.md)
- プロンプト本体: [.company/lab/projects/lora-dynamic-separation/references/notebooklm-prompt-axis-analysis.md](../../../.company/lab/projects/lora-dynamic-separation/references/notebooklm-prompt-axis-analysis.md)
