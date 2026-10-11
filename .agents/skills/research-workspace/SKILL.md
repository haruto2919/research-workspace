---
name: research-workspace
description: 研究活動用Markdownワークスペースを作成・管理するスキル。初回セットアップ、研究プロジェクト作成、README/frontmatter管理、TODO管理、ダッシュボード表示、研究状況確認、新規研究テーマ追加に使う。「研究ワークスペースを作って」「研究プロジェクトを追加して」「TODOに追加して」「今日のTODO」「ダッシュボード」「研究状況を見せて」「この研究テーマを管理したい」と言われたときに使う。
---

# Research Workspace

研究文脈を `.research/` にMarkdownとして保存し、AIエージェントが継続的に研究活動を支援できる状態を作る。

## 原則

- `.research/` を研究文脈のSingle Source of Truthにする。
- 初回セットアップの質問は最小限にする。
- ファイル操作前に今日の日付を確認する。
- ダッシュボードや研究状況を出す前に必ずファイルを読み直す。
- TODOはユーザーが明示したもの、またはユーザーが採用した候補だけ追加する。
- Notion、Google Calendar、外部DB連携を前提にしない。
- `references/` は外部資料・文献・エクスポート資料の置き場として扱い、原則として読み取り専用にする。
- MTG前資料は `materials/` に保存し、MTG後の議事録だけを `meetings/` に保存する。
- MTG説明用の概念図・再構成図は `materials/figures/` に保存し、実験結果そのものの正本図は `experiments/figures/` に保存する。
- 議事録、MTG前資料、壁打ちの詳細処理は専用スキルに委ねる。

## 管理する構造

```text
.research/
├── secretary/
│   ├── AGENTS.md
│   ├── todos/
│   └── notes/
│       └── brainstorm/
└── lab/
    ├── AGENTS.md
    └── projects/
        └── <project-slug>/
            ├── README.md
            ├── meetings/
            ├── specs/
            ├── experiments/
            │   └── figures/
            ├── materials/
            │   └── figures/
            ├── code-analysis/
            ├── references/
            └── papers/
```

## 初回セットアップ

`.research/` が存在しない、または `.research/lab/projects/` にプロジェクトがない場合に実行する。

質問は2つだけ:

1. 研究テーマ名
2. 研究概要

回答後、研究テーマ名からASCII kebab-caseの `<project-slug>` を作る。日本語タイトルの場合は意味を短い英語に要約してslug化する。うまく作れない場合のみ、`research-project-YYYYMMDD` を使う。

作成するもの:

```text
.research/secretary/todos/YYYY-MM-DD.md
.research/secretary/AGENTS.md
.research/secretary/notes/brainstorm/
.research/lab/AGENTS.md
.research/lab/projects/<project-slug>/README.md
.research/lab/projects/<project-slug>/meetings/
.research/lab/projects/<project-slug>/specs/
.research/lab/projects/<project-slug>/experiments/
.research/lab/projects/<project-slug>/experiments/figures/
.research/lab/projects/<project-slug>/materials/
.research/lab/projects/<project-slug>/materials/figures/
.research/lab/projects/<project-slug>/references/
.research/lab/projects/<project-slug>/papers/
.research/lab/projects/<project-slug>/code-analysis/
```

既存ファイルは上書きしない。既に同じslugのプロジェクトがある場合は、ユーザーに別名を確認する。

`.research/secretary/AGENTS.md` と `.research/lab/AGENTS.md` はテンプレートに同梱されている局所ルールとして扱う。存在する場合は上書きしない。研究プロジェクト追加時に、各プロジェクト配下へ個別の `AGENTS.md` は作らない。

## Project READMEテンプレート

```markdown
---
project: <project-slug>
status: active
summary: <研究概要を1行で要約>
created: YYYY-MM-DD
last_updated: YYYY-MM-DD
---

# <研究テーマ名>

## 概要

<研究概要>

## 現在の状況

研究テーマを作成した初期状態。

## マイルストーン

- [ ] 研究方針を整理する

## 更新履歴

| 日付 | 内容 |
|------|--------|
| YYYY-MM-DD | プロジェクト作成 |
```

`status` は以下だけを使う。

- `active`: 現在進行中
- `paused`: 一時停止中
- `completed`: 完了・一区切り
- `archived`: 参照用に保管

`summary` はダッシュボードの1行表示に使う。研究進捗を更新したら、必要に応じて `summary` と `last_updated` も更新する。

## ファイル命名規則

日付ベースのファイルを作成・更新する前に、必ず今日の日付を確認する。

### 日次ファイル

日次管理ファイルは同日1ファイルにする。

```text
.research/secretary/todos/YYYY-MM-DD.md
```

同じ日付のTODOファイルが既にある場合は、新規作成せず追記・更新する。

### トピックファイル

研究記録・相談・実験・資料は、日付とトピックを含むファイル名にする。

```text
YYYY-MM-DD-kebab-case-topic.md
```

用途別の例:

```text
.research/secretary/notes/brainstorm/YYYY-MM-DD-topic.md
.research/lab/projects/<project>/meetings/YYYY-MM-DD-mtg.md
.research/lab/projects/<project>/specs/YYYY-MM-DD-topic-spec.md
.research/lab/projects/<project>/experiments/YYYY-MM-DD-topic.md
.research/lab/projects/<project>/materials/YYYY-MM-DD-topic-mtg-materials.md
.research/lab/projects/<project>/papers/YYYY-MM-DD-paper-title.md
```

`brainstorm` は必ず `YYYY-MM-DD-topic.md` にする。`YYYY-MM-DD.md` のような日次まとめファイルにはしない。

避ける名前:

```text
memo.md
new-memo.md
final.md
final-v2.md
YYYY-MM-DD-2.md
```

同じ日付・同じ目的のファイルが既にある場合は、重複ファイルを作らず追記・更新する。別トピックなら、内容が分かるkebab-caseトピック名を付ける。

## 研究プロジェクト追加

ユーザーが新しい研究テーマを管理したいと言ったら、初回セットアップと同じ2問を聞く。

1. 研究テーマ名
2. 研究概要

`.research/lab/projects/<project-slug>/` を追加し、同じREADMEテンプレートとサブフォルダを作る。

プロジェクト追加時は、`.research/secretary/AGENTS.md` と `.research/lab/AGENTS.md` を重複作成しない。各プロジェクト配下にも個別の `AGENTS.md` は作らない。

## TODO管理

今日のTODOファイル:

```text
.research/secretary/todos/YYYY-MM-DD.md
```

ファイルがなければ作成する。

```markdown
# YYYY-MM-DD TODO

## 通常
```

TODO形式:

```markdown
- [ ] タスク内容 | 優先度: 高/通常/低
- [ ] タスク内容 | 優先度: 通常 | 期限: YYYY-MM-DD
- [x] 完了タスク | 完了: YYYY-MM-DD
```

ルール:

- 優先度が未指定なら `通常`。
- 期限はユーザーが明示した場合だけ付ける。
- 他スキルが抽出したTODO候補は、ユーザーが採用したものだけ追加する。
- 今日のTODO表示では、未完了を先に出し、完了済みは後にまとめる。

## ダッシュボード

「ダッシュボード」「今日の状況」「研究状況」「TODO見せて」と言われたら、記憶に頼らず以下を読む。

1. `.research/secretary/todos/YYYY-MM-DD.md`
2. `.research/secretary/notes/brainstorm/` のファイル一覧
3. `.research/lab/projects/*/README.md` のfrontmatter

`Brainstorm Notes` の件数では、`.gitkeep` などの管理用プレースホルダファイルを数えない。

出力形式:

```markdown
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  Research Workspace Dashboard (YYYY-MM-DD)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

### 秘書室
TODO: X件 未完了 / Y件 完了
- [ ] ...
- [x] ...

Brainstorm Notes: Z件

### Lab
- [ ] (active) project-name（summary）
- [x] (completed) project-name（summary）

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
```

ステータス表示:

- `active`, `paused`: `- [ ]`
- `completed`, `archived`: `- [x]`

## プロジェクト詳細表示

「<project>のタスク」「<project>の詳細」「<project>の状況」と言われたら、対象プロジェクトの `README.md` を読む。

読むもの:

1. frontmatterの `status`, `summary`, `last_updated`
2. `## 現在の状況`
3. `## マイルストーン`
4. 必要なら `## 更新履歴`

出力では、未完了タスクを優先して表示する。

```markdown
### <project> の状況

- ステータス: active
- サマリー: ...
- 最終更新: YYYY-MM-DD

### 未完了マイルストーン
- [ ] ...

### 直近の更新履歴
- YYYY-MM-DD: ...
```

## README/frontmatter更新

研究進捗を更新したときは、対象プロジェクトの `README.md` を更新する。

更新対象:

- `summary`: 現在状況を1行で表す。
- `last_updated`: 今日の日付。
- `更新履歴`: 重要な更新のみ追記する。
- `マイルストーン`: 明確なタスクや段階が出た場合だけ追記する。

既存内容は削除せず、必要最小限の更新にする。

## References

`references/` は、外部資料、文献メモ、エクスポート資料、配布されたMarkdownなどを置く場所として扱う。

原則:

- AIは `references/` 内の既存ファイルを編集しない。
- 参照・要約・リンクはしてよい。
- ユーザーが明示的に依頼した場合のみ、新規ファイル作成や編集を行う。
- AI自身が作成する分析・計画・実験メモ・MTG前資料は `references/` ではなく、`specs/`, `experiments/`, `materials/`, `papers/`, `brainstorm/` の適切な場所へ保存する。

## Materials

`materials/` は、MTG前の相談資料、進捗報告資料、実験レビュー資料、方針相談資料を置く場所として扱う。

原則:

- MTG前資料の正本は `materials/YYYY-MM-DD-topic-mtg-materials.md` に保存する。
- `meetings/` はMTG後の議事録専用にする。
- Notionに貼った資料や後日exportしたMarkdownが `references/` に入っても、`materials/` の正本は残す。
- MTG説明用に再構成した図、概念図、方針比較図は `materials/figures/YYYY-MM-DD-topic/` に保存する。
- 実験データから生成した正本図や再利用可能な結果図は `experiments/figures/YYYY-MM-DD-topic/` に保存する。

## 役割分担

このスキルで行う:

- 初回セットアップ
- 研究プロジェクト追加
- TODO追加・表示・完了更新
- ダッシュボード表示
- README/frontmatterの軽い更新

このスキルで行わない:

- MTG前資料作成: `meeting-materials`
- MTG後の議事録作成: `meeting-minutes`
- 壁打ち深掘り: `brainstorm`
- 文献DB登録、Notion同期、Google Calendar連携
- コード実装記録

## 完了報告

セットアップやプロジェクト追加後は、作成した主要ファイルを簡潔に報告する。

```markdown
Research Workspaceを作成しました。

- Project: <project-slug>
- README: .research/lab/projects/<project-slug>/README.md
- TODO: .research/secretary/todos/YYYY-MM-DD.md

次は `meeting-materials`, `meeting-minutes`, `brainstorm` を使って研究文脈を蓄積できます。
```
