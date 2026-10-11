# Research Workspace

Research Workspace は、AIエージェント用の研究管理テンプレートです。

研究テーマ、MTG、TODO、壁打ち、実験計画、MTG前資料などの研究文脈をローカルのMarkdownファイルとして保存し、その文脈をAIエージェントに読ませながら研究活動を進めるためのワークスペースを作ります。

Codex、Claude Code、Gemini CLI、Antigravity など、`AGENTS.md` と `.agents/skills/` を読めるAIエージェントで使えます。

## これは何か

このテンプレートは、研究活動を次のように管理するためのものです。

- 研究プロジェクトごとにREADMEを作り、現在の状況を1か所にまとめる
- MTG前に、過去の文脈を読ませて相談資料を作る
- MTG後に、当日メモや文字起こしから議事録を作る
- TODOを日付ごとに管理し、ダッシュボードで確認する
- 研究相談やアイデア出しを壁打ちメモとして残す
- 実験計画や設計方針を `specs/` に保存する

中心になる考え方は、**研究文脈をMarkdownとして残し、AIエージェントがそれを読みながら継続的に支援できる状態を作る** ことです。

## 誰向けか

主な利用者は研究室メンバーです。

- 卒業研究を進めるB4
- 修士研究を進めるM1/M2
- 博士研究や複数テーマを管理したい学生
- 研究MTG、実験ログ、TODO、相談メモをAIエージェントと一緒に整理したい人

研究テーマが1つだけでも使えます。複数テーマがある場合は、テーマごとに `.research/lab/projects/<project>/` を作って管理します。

## できること

Research Workspace では、以下の作業をAIエージェントに依頼できます。

- 初回セットアップ
- 研究プロジェクトの追加
- TODO管理
- ダッシュボード表示
- 研究相談・壁打ち
- MTG前資料の作成
- MTG後議事録の作成
- 実験計画・設計計画の保存
- 必要に応じた追加スキルの作成・改善

外部サービス連携は前提にしていません。基本はローカルMarkdownだけで動きます。

## Quick Start

### 1. リポジトリをcloneする

```bash
git clone git@github.com:YusakuTakama/research-workspace.git
cd research-workspace
```

### 2. AIエージェントで開く

Codex、Claude Code、Antigravity などで、このフォルダを開きます。

最初に、エージェントへ次のように依頼します。

```text
このリポジトリのAGENTS.mdを読んで、Research Workspaceとして扱ってください。
```

Codex や Claude Code では、同梱されている `.codex/skills` または `.claude/skills` からスキルを参照できます。

### 3. 初回セットアップを依頼する

次のように依頼します。

```text
research-workspace スキルで初回セットアップをしてください。
```

エージェントから、最低限以下の2つを聞かれます。

```text
1. 研究テーマ名
2. 研究概要
```

例:

```text
研究テーマ名: 視覚言語モデルにおける説明可能性の分析
研究概要: 視覚言語モデルが画像内のどの情報を根拠に応答しているかを、特徴可視化と出力説明の対応から分析する。
```

### 4. 研究プロジェクトが作成される

初回セットアップが終わると、以下のようなプロジェクトフォルダが作られます。

```text
.research/lab/projects/<project-name>/
├── README.md
├── meetings/
├── specs/
├── experiments/
│   └── figures/
├── materials/
│   └── figures/
├── references/
└── papers/
```

`README.md` が、その研究プロジェクトの現在状況を表す中心ファイルになります。

### 5. ダッシュボードを確認する

次のように依頼します。

```text
ダッシュボードを見せて。
```

エージェントは、今日のTODO、壁打ちメモ、研究プロジェクトの状態を読み直して表示します。

## 対応ツール

### Codex

Codex では、`.codex/skills` から `.agents/skills/` を参照します。

```text
research-workspace スキルで初回セットアップして。
```

### Claude Code

Claude Code では、`.claude/skills` から `.agents/skills/` を参照します。

```text
AGENTS.mdを読んで、このResearch Workspaceのルールに従ってください。
```

### Antigravity

Antigravity でも、基本の入口は `AGENTS.md` と `.agents/skills/` です。

```text
このリポジトリをResearch Workspaceとして扱い、AGENTS.mdのルールに従ってください。
```

## 基本ワークフロー

1. 研究テーマを登録する
2. 研究相談やアイデアを壁打ちする
3. 固まった方針を `specs/` に保存する
4. MTG前に相談資料を作る
5. MTG後に議事録を作る
6. 議事録からTODO候補を抽出する
7. 採用したTODOを今日のTODOへ追加する
8. ダッシュボードで現在状況を確認する

この流れを繰り返すことで、単発チャットではなく、過去の文脈を踏まえた研究支援を受けやすくなります。

## スキルの使い方

### `research-workspace`

初回セットアップ、研究プロジェクト作成、TODO管理、ダッシュボード表示を担当します。

頼み方:

```text
research-workspace スキルで初回セットアップして。
```

```text
新しい研究テーマを追加したいです。
```

```text
今日のTODOを見せて。
```

```text
ダッシュボードを見せて。
```

主な保存先:

```text
.research/lab/projects/<project>/README.md
.research/secretary/todos/YYYY-MM-DD.md
```

### `brainstorm`

研究相談、仮説整理、実験案の検討、spec化前の壁打ちを担当します。

頼み方:

```text
この研究アイデアについて壁打ちしたいです。
```

```text
次の実験案を整理したいです。プロジェクトは <project> です。
```

```text
この内容をspec化できるか見てください。
```

保存先:

```text
.research/secretary/notes/brainstorm/YYYY-MM-DD-topic.md
```

`brainstorm` は、すぐに結論を出すためだけではなく、問い、仮説、選択肢、未解決点、次アクションを整理するために使います。固まった内容だけを、ユーザーの承認後に `specs/` へ昇格します。

### `meeting-materials`

MTG前の相談資料、進捗報告資料、実験レビュー資料を作成します。

頼み方:

```text
次回MTGの資料を作ってください。プロジェクトは <project> です。
```

```text
この実験結果を先生に説明する資料にしてください。
```

```text
さっきのbrainstormをMTG資料化してください。
```

保存先:

```text
.research/lab/projects/<project>/materials/YYYY-MM-DD-topic-mtg-materials.md
.research/lab/projects/<project>/materials/figures/YYYY-MM-DD-topic/
```

MTG前資料は `meetings/` ではなく `materials/` に保存します。`meetings/` はMTG後の議事録専用です。

### `meeting-minutes`

MTG後に、当日メモや文字起こしから議事録を作成します。TODO候補の抽出と、プロジェクトREADMEの更新も担当します。

頼み方:

```text
MTGが終わりました。議事録を作ってください。
```

```text
文字起こしがあります。meeting-minutes で整理してください。
```

保存先:

```text
.research/lab/projects/<project>/meetings/YYYY-MM-DD-mtg.md
.research/secretary/todos/YYYY-MM-DD.md
```

文字起こし全文は、議事録作成の補助入力として使います。議事録には必要な要約だけを残します。

### `skill-creator`

必要に応じて、新しいスキルの作成や既存スキルの改善に使います。

頼み方:

```text
この作業を自動化するスキルを作りたいです。
```

`skill-creator` は発展的な用途なので、最初から使う必要はありません。

## 利用シナリオ

初めて使うときは、以下の流れを一度試すと全体像がつかみやすいです。

### 1. 初回セットアップ

```text
research-workspace スキルで初回セットアップして。
```

研究テーマ名と研究概要を答えると、`.research/lab/projects/<project>/` が作成されます。

### 2. 研究相談の壁打ち

```text
プロジェクト <project> について、次にどの評価実験をするべきか壁打ちしたいです。
```

壁打ち結果は `.research/secretary/notes/brainstorm/` に保存されます。

### 3. MTG前資料作成

```text
さっきの壁打ちをもとに、次回MTGで相談する資料を作ってください。
```

資料は `.research/lab/projects/<project>/materials/` に保存されます。

### 4. MTG後議事録作成

```text
MTGが終わりました。議事録を作ってください。
```

プロジェクト名、日付、参加者を確認したあと、当日メモや文字起こしをもとに議事録を作成します。

### 5. TODO反映

議事録から抽出されたTODO候補を確認し、採用するものだけ今日のTODOに追加します。

```text
1と3を今日のTODOに追加してください。優先度は通常で。
```

### 6. ダッシュボード確認

```text
ダッシュボードを見せて。
```

今日のTODO、壁打ちメモ数、研究プロジェクトの状態が表示されます。

## ディレクトリ構成

```text
.
├── AGENTS.md
├── .agents/
│   └── skills/
│       ├── research-workspace/
│       ├── brainstorm/
        ├── research-spec/
        ├── engineering-task/
│       ├── meeting-materials/
│       ├── meeting-minutes/
        ├── research-code-insight/
│       └── skill-creator/
├── .codex/
│   └── skills -> ../.agents/skills
├── .claude/
│   └── skills -> ../.agents/skills
└── .research/
    ├── secretary/
    │   ├── AGENTS.md
    │   ├── todos/
    │   └── notes/
    │       └── brainstorm/
    └── lab/
        ├── AGENTS.md
        └── projects/
```

初回セットアップ後は、研究プロジェクトごとに以下の構造が作られます。

```text
.research/lab/projects/<project>/
├── README.md
├── meetings/
├── specs/
├── experiments/
│   └── figures/
├── materials/
│   └── figures/
├── references/
└── papers/
```

## 設計方針

- 研究文脈のSingle Source of TruthはローカルMarkdownです。
- 各研究プロジェクトの `README.md` が、現在状況を表す中心ファイルです。
- `README.md` のfrontmatterにある `summary` と `status` をダッシュボード表示に使います。
- TODOは、ユーザーが明示したもの、または候補から採用したものだけ追加します。
- MTG前資料の正本は `materials/` に保存します。
- MTG後の議事録は `meetings/` に保存します。
- MTG説明用の概念図・再構成図は `materials/figures/` に保存します。
- 実験結果そのものの正本図は `experiments/figures/` に保存します。
- `references/` は外部資料や文献メモの置き場です。AIが作成する通常の分析や計画は置きません。
