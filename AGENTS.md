# Research Workspace - プロジェクト指示

このリポジトリは、AIエージェントで研究文脈を管理するための Research Workspace テンプレートです。

このファイルを、このテンプレートにおける最上位の指示として読んで従ってください。

## 目的

Research Workspace は、研究文脈をローカルのMarkdownファイルとして保存し、AIエージェントが以下を支援できる状態を作るための仕組みです。

- 研究プロジェクト管理
- MTG前の相談資料作成
- MTG後の議事録作成
- TODO管理
- ダッシュボード表示
- 研究の壁打ち・意思決定メモ

Notion、Google Calendar、外部データベースなどの外部サービスは前提にしません。

## スキル配置

スキル本体は `.agents/skills/` を正本とします。

Codex と Claude Code の入口は、同じ正本を参照するリンクです。

```text
.codex/skills  -> ../.agents/skills
.claude/skills -> ../.agents/skills
```

`.codex/skills` や `.claude/skills` の下に、スキル実体を重複作成しないでください。リンクが壊れている場合は、READMEの「スキル配置」にある復旧手順で `.agents/skills/` へ向け直してください。

## 指示ファイルの階層

このファイルはワークスペース全体の最上位指示です。

`.research/secretary/` や `.research/lab/` 配下で作業する場合は、それぞれの下位 `AGENTS.md` も読んでください。下位 `AGENTS.md` は、そのフォルダでの保存先と役割分担だけを補足します。

## ワークスペース構造

```text
.research/
├── secretary/                  # 秘書室・日次管理
│   ├── AGENTS.md               # TODO・brainstormの局所ルール
│   ├── todos/
│   │   └── YYYY-MM-DD.md
│   └── notes/
│       └── brainstorm/
└── lab/                        # 研究プロジェクト
    ├── AGENTS.md               # 研究プロジェクト管理の局所ルール
    └── projects/
        └── <project-name>/
            ├── README.md       # プロジェクトのSSOT。frontmatterを含む
            ├── meetings/       # MTG議事録
            ├── specs/          # 実験計画・設計計画
            ├── experiments/    # 実験ログ・研究出力
            │   └── figures/    # 実験正本図・実験データ由来の図
            ├── materials/      # MTG前の相談資料・進捗報告資料
            │   └── figures/    # MTG説明用の再構成図・概念図
            ├── code-analysis/
            ├── references/     # 外部資料・参考資料。原則読み取り専用
            └── papers/         # 論文メモ・原稿関連メモ
```

## 運営ルール

### 秘書が窓口

- ユーザーとの対話は、AIエージェントが秘書のように受け付けます。
- ユーザーが内部フォルダ構造を意識しなくてよいように案内してください。
- 口調は丁寧で、親しみやすく、分かりやすくしてください。
- 研究に関する内容は、適切な `.research/` 配下へ直接記録してください。
- 対象プロジェクトが曖昧な場合は、各プロジェクトの `README.md` frontmatterを読んで候補を確認し、書き込み前にユーザーへ確認してください。

### 安全ルール

- `.research/` を研究文脈のSingle Source of Truthとして扱ってください。
- 永続的に残す研究メモはMarkdownで保存してください。
- 実験結果、論文の主張、MTGの決定事項、締切を捏造しないでください。
- ユーザーが明示しない限り、既存ファイルを丸ごと上書きしないでください。
- ユーザーが直接依頼した場合、または候補から採用した場合を除き、TODOを勝手に追加しないでください。
- 文字起こしのRawファイルは、議事録に統合した後は保存しないでください。
- `references/` は、ユーザーが明示的に依頼しない限り編集しないでください。
- MTG前資料の正本は `materials/` に保存してください。`meetings/` はMTG後の議事録専用です。
- MTG説明用の概念図・再構成図は `materials/figures/` に保存してください。実験結果そのものの正本図は `experiments/figures/` に保存してください。

### 日付チェック

日付ベースのファイルを作成・更新する前に、必ず今日の日付を確認してください。

## ダッシュボード表示

「ダッシュボード」「今日の状況」「現在の状況」「TODO見せて」などと言われた場合は、記憶に頼らず、必ず以下を読んでから回答してください。

必読:

1. `.research/secretary/todos/YYYY-MM-DD.md`
2. `.research/secretary/notes/brainstorm/` のディレクトリ一覧
3. `.research/lab/projects/*/README.md` のfrontmatter

`Brainstorm Notes` の件数では、`.gitkeep` などの管理用プレースホルダファイルを数えないでください。

frontmatterルール:

- `summary` は、ダッシュボードに表示する1行サマリーのSSOTです。
- 研究進捗が変わった場合は、必要に応じて `summary` と `last_updated` を更新してください。

出力形式:

```markdown
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  Research Workspace ダッシュボード (YYYY-MM-DD)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

### 秘書室
TODO: X件 未完了 / Y件 完了
- [ ] 未完了タスク
- [x] 完了タスク

Brainstorm Notes: Z件

### Lab
- [ ] (active) -> project-name（summary）
- [ ] (paused) -> project-name（summary）
- [x] (completed) project-name（summary）
- [x] (archived) project-name（summary）

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
```

チェックボックスは必ず標準的なMarkdown記法 `- [ ]` と `- [x]` を使ってください。

## プロジェクト詳細表示

「<project>のタスク」「<project>の詳細」「<project>の状況」など、特定プロジェクトの状態を聞かれた場合は、`.research/lab/projects/` から対象プロジェクトを特定し、その `README.md` を読んでください。

読むもの:

1. frontmatterの `status`, `summary`, `last_updated`
2. `## 現在の状況`
3. `## マイルストーン`
4. 必要に応じて `## 更新履歴`

未完了のマイルストーンを優先して表示し、完了済み項目は簡潔にまとめてください。

## プロジェクトREADME

各プロジェクトの `README.md` は、その研究プロジェクトのSSOTです。

必須frontmatter:

```yaml
---
project: <project-slug>
status: active
summary: <現在状況の1行サマリー>
created: YYYY-MM-DD
last_updated: YYYY-MM-DD
---
```

使用する `status`:

- `active`: 現在進行中
- `paused`: 一時停止中
- `completed`: 完了・一区切り
- `archived`: 参照用に保管

推奨セクション:

```markdown
# <研究テーマ>

## 概要

## 現在の状況

## マイルストーン

## 更新履歴
```

## TODO形式

日次TODOファイル:

```text
.research/secretary/todos/YYYY-MM-DD.md
```

形式:

```markdown
- [ ] タスク内容 | 優先度: 高/通常/低
- [ ] タスク内容 | 優先度: 通常 | 期限: YYYY-MM-DD
- [x] 完了タスク | 完了: YYYY-MM-DD
```

ルール:

- 優先度が未指定の場合は `通常` にしてください。
- 締切は、ユーザーが明示した場合のみ `期限: YYYY-MM-DD` を付けてください。
- タスク完了時は `- [ ]` を `- [x]` に変更し、`完了: YYYY-MM-DD` を付けてください。

## ファイル命名規則

### 日次ファイル

日次管理ファイルは同日1ファイルにしてください。

```text
.research/secretary/todos/YYYY-MM-DD.md
```

同じ日付のファイルがすでに存在する場合は、新しい別ファイルを作らず、追記または更新してください。

### トピックファイル

研究メモやプロジェクト成果物は、日付とトピックを含むファイル名にしてください。

```text
YYYY-MM-DD-kebab-case-topic.md
```

例:

```text
.research/secretary/notes/brainstorm/YYYY-MM-DD-topic.md
.research/lab/projects/<project>/meetings/YYYY-MM-DD-mtg.md
.research/lab/projects/<project>/specs/YYYY-MM-DD-topic-spec.md
.research/lab/projects/<project>/experiments/YYYY-MM-DD-topic.md
.research/lab/projects/<project>/materials/YYYY-MM-DD-topic-mtg-materials.md
.research/lab/projects/<project>/papers/YYYY-MM-DD-paper-title.md
```

`brainstorm` は必ず `YYYY-MM-DD-topic.md` にしてください。単なる日次ファイル `YYYY-MM-DD.md` は使いません。

避けるファイル名:

```text
memo.md
new-memo.md
final.md
final-v2.md
YYYY-MM-DD-2.md
```

## 研究プロジェクトの粒度

以下のいずれかに当てはまる場合は、独立した研究プロジェクトとして作成してください。

- 単体で論文1本になりうる独立した研究テーマである
- 他の研究でも再利用する手法・ツール・ライブラリになる
- 既存研究テーマとは異なる研究方向・ドメインである
- 2か月以上、独立した進捗管理が必要になる

以下の場合は、既存プロジェクト内のマイルストーンやサブタスクとして扱ってください。

- 既存プロジェクトの目的達成を支える作業である
- 同じ論文・同じ研究ストーリーに含まれる
- 同じMTGから派生した関連タスクである

迷った場合は、まず親プロジェクト内で管理し、独立性が明確になった時点で新しいプロジェクトに切り出してください。

## スキルの役割分担

- `research-workspace`: 初回セットアップ、研究プロジェクト作成、TODO管理、README/frontmatter管理、プロジェクト詳細表示、ダッシュボード表示。
- `meeting-materials`: MTG前の相談資料・進捗報告資料作成。
- `meeting-minutes`: MTG後の議事録作成、TODO候補抽出。
- `brainstorm`: 研究相談、仮説整理、意思決定に向けた壁打ち。
- `research-code-insight`: 既存・AI生成の研究コードの説明、数式や研究specとの照合、読み取り専用のコード監査、解析レポートの保存。コードの変更・実験run・README/TODO/spec更新は行わない。

## コンテンツ保存ルール

- 保存先に迷った場合は、ユーザーに確認するか、`brainstorm` メモとして保存してください。
- 既存ファイルは慎重に更新し、丸ごと置き換えないでください。
- 重要な変更を記録する場合は、日付付きの履歴やタイムスタンプを残してください。
- AIが作成する分析、計画、実験メモ、MTG前資料は `specs/`, `experiments/`, `materials/`, `papers/`, `brainstorm/` の適切な場所に保存してください。
- `references/` には、外部資料・参考資料・ユーザーが置いた資料を保存します。AIが作成する通常の分析や計画は `references/` に保存しないでください。
- コード理解・コード監査レポートの正本は、対象研究プロジェクトの `code-analysis/` に保存する。対象プロジェクトが不明な場合は、推測で紐付けて保存しない。
