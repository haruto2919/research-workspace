# Research Workspace - Lab

このフォルダは、研究プロジェクトの文脈、MTG、spec、実験、資料、文献メモを管理する場所です。

## 役割

- 研究プロジェクトREADMEの管理
- MTG後の議事録保存
- 実験計画・設計計画の保存
- 実験ログ・研究出力の保存
- MTG前資料と説明用図の保存
- 文献メモ・外部資料の参照

## プロジェクト構造

各研究プロジェクトは `.research/lab/projects/<project>/` に置きます。

```text
.research/lab/projects/<project>/
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

## 保存先の使い分け

- `README.md`: プロジェクトのSingle Source of Truth。frontmatter、概要、現在の状況、マイルストーン、更新履歴を管理する。
- `meetings/`: MTG後の議事録専用。MTG前資料は置かない。
- `materials/`: MTG前の相談資料、進捗報告資料、実験レビュー資料、方針相談資料。
- `materials/figures/`: MTG説明用に再構成した図、概念図、方針比較図。
- `specs/`: 実験計画、設計計画、brainstorm後に固まった実行プラン。
- `experiments/`: 実験ログ、研究出力、結果メモ。
- `experiments/figures/`: 実験正本図、実験データから生成したグラフ、再利用可能な結果図。
- `references/`: 外部資料、Notion export、配布資料、参考資料。原則読み取り専用。
- `papers/`: 論文メモ、原稿関連メモ。
- `code-analysis/`: コードの教育的解説、静的監査、研究計画・数式・実装の対応を記録する。コードの実行結果そのものは `experiments/` に保存する。

## スキルの使い分け

- MTG前資料は `meeting-materials` スキルに委ねてください。
- MTG後議事録は `meeting-minutes` スキルに委ねてください。
- 研究相談、仮説整理、spec化前の壁打ちは `brainstorm` スキルに委ねてください。
- プロジェクト作成、TODO、ダッシュボード、README/frontmatter管理は `research-workspace` スキルに委ねてください。
- 研究コードの説明・読み取り専用監査は `research-code-insight` スキルに委ねる。

## ルール

- `.research/lab/projects/<project>/README.md` を、その研究プロジェクトのSSOTとして扱ってください。
- 対象プロジェクトが曖昧な場合は、各プロジェクトの `README.md` frontmatterを読んで候補を確認してください。
- `references/` は、ユーザーが明示した場合を除き編集しないでください。
- AIが作成する分析、計画、実験メモ、MTG前資料は `references/` ではなく、`specs/`, `experiments/`, `materials/`, `papers/` の適切な場所に保存してください。
- 日付ベースのファイルを作成・更新する前に、必ず今日の日付を確認してください。
- 既存ファイルは丸ごと上書きせず、必要最小限の追記・更新にしてください。
- `research-code-insight` はコードを編集・実行せず、研究計画やTODOを更新しない。レポートには解析したrepository、commit、対象ファイル、確認できなかった事項を記録する。

## しないこと

- `implementation/` を標準構造として作らない。
- `ai_research_logs/` や `prompts/` を初期版の標準構造として作らない。
- Notion DB連携や外部サービス連携を前提にしない。
- SSHサーバ固有のルールや個人環境のコードパス対応表を入れない。
- 各プロジェクト配下に個別の `AGENTS.md` を自動作成しない。
