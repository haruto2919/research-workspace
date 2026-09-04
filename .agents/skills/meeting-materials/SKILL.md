---
name: meeting-materials
description: MTG前の研究相談資料・進捗報告資料・実験レビュー資料・方針相談資料をMarkdownで作成するスキル。「MTG資料作って」「次回MTGの資料」「先生に見せる資料」「実験結果を説明する資料」「brainstormをMTG資料化」「この結果を相談資料にして」と言われたときは必ず使う。プロジェクト文脈、spec、experiments、brainstorm、直近議事録を読み、先生に相談したいこと、Evidence、Interpretation、Ask、必要な図を整理して materials/ に保存する。
---

# Meeting Materials（MTG前資料スキル）

研究MTG前に、既存の研究文脈・実験結果・壁打ち結果を読み、先生に説明・相談するためのMarkdown資料を作成する。資料作成だけを行い、README、TODO、議事録、referencesは更新しない。

## 参照する考え方

長い理論説明は出力しないが、運用は以下の型に従う。

- Effective agenda: 会議前に目的、期待値、判断論点を明確にする
  - https://hbr.org/2015/03/how-to-design-an-agenda-for-an-effective-meeting
- Pre-read: 参加者が議論に入れるだけの文脈を効率よく渡す
  - https://experience.dropbox.com/virtual-first-toolkit/communication/prepare-with-a-preread
- Advisor meeting: 結果、解釈、詰まり、次に相談したいことを持参する
  - https://www.jackwbaker.com/advice/Meeting_with_your_advisor.html
  - https://www.cs.washington.edu/academics/graduate/phd-program/handbook/advising-guide/building-relationship/
- Data storytelling: 図にはtakeaway、軸、単位、注記、sourceを持たせる
  - https://urbaninstitute.github.io/graphics-styleguide/
  - https://www.storytellingwithdata.com/blog/2021/1/10/lets-improve-this-graph-yt9xj

## 原則

- MTG前資料は「何を相談し、何を判断してほしいか」を明確にするために作る。
- 読み込んだログを時系列に貼るのではなく、先生に説明しやすい順番へ再構成する。
- 冒頭に「今日のMTGで先生に相談したいこと」を置く。
- `Evidence`、`Interpretation`、`Ask` を分ける。
- 主張レベルを意識し、観測済み、支持される仮説、相談用推測、未確認事項を混ぜない。
- 資料の長さは固定しない。説明内容に応じて必要十分な長さにする。
- 図は必要時機能。ただし説明が分かりやすくなる場合は積極的に提案・作成する。
- Notion貼付を想定し、Markdown、数式、表、画像リンクが崩れにくい形にする。
- ファイル操作前に今日の日付を確認する。
- 既存ファイルは丸ごと上書きしない。既に同日同topicの資料がある場合は、追記または別topic名を確認する。

## しないこと

- READMEを更新しない。
- TODOを追加・更新しない。
- 議事録を作らない。
- `meetings/` にMTG前資料を保存しない。
- `references/` にAI生成資料やAI生成図の正本を保存しない。
- `references/` を全文総読みしない。
- Notionへ直接同期しない。

## 対象パス

Company運用の正本パスを使う。

```text
.company/lab/projects/<project>/README.md
.company/lab/projects/<project>/meetings/*.md
.company/lab/projects/<project>/specs/*.md
.company/lab/projects/<project>/experiments/*.md
.company/lab/projects/<project>/materials/*.md
.company/lab/projects/<project>/materials/figures/
.company/lab/projects/<project>/references/*.md
.company/secretary/notes/brainstorm/*.md
.company/secretary/todos/YYYY-MM-DD.md
```

保存先:

```text
.company/lab/projects/<project>/materials/YYYY-MM-DD-topic-mtg-materials.md
.company/lab/projects/<project>/materials/figures/YYYY-MM-DD-topic/
.company/lab/projects/<project>/experiments/figures/YYYY-MM-DD-experiment-topic/
```

`materials/` と `materials/figures/` は標準保存先として扱う。存在しなければ資料作成時に作成する。

## Notionとreferencesの扱い

`materials/` は、AIが作るMTG前資料のローカル正本である。Notionは先生に説明するための共有・提示先であり、このスキルはNotionへ直接同期しない。

後日、NotionからexportされたMarkdownが `references/` に入っても、それは共有後の写しとして扱う。`materials/` の正本を削除・移動しない。

## Workflow

### Phase 1: 相談対象の確認

ユーザーが「MTG資料」「次回MTG」「先生に見せる資料」「実験結果を説明」「brainstormを資料化」などと言ったら、分かっている情報から対象を特定する。

確認するもの:

- プロジェクト名
- MTG日、または資料の日付
- topic
- 資料の起点
- 先生に相談したいこと
- 図が必要そうか

すでに会話から分かる項目は再質問しない。曖昧な項目だけ短く確認する。

### Phase 2: 文脈読み込み

対象プロジェクトが分かる場合は、資料作成前に文脈を読む。

必読:

- `.company/lab/projects/<project>/README.md`
- `.company/lab/projects/<project>/meetings/*.md` の直近1件

必要に応じて読む:

- `specs/` の関連または最新ファイル 最大3件
- `experiments/` の関連または最新ファイル 最大3件
- `.company/secretary/notes/brainstorm/` の関連メモ 最大3件
- `materials/` の関連または最新ファイル 最大3件
- `.company/secretary/todos/YYYY-MM-DD.md` は、前回MTGやTODO起点の資料の場合だけ読む

`references/` は原則としてファイル一覧のみ確認する。本文を読むのは以下の場合だけに限定し、最大1〜3件に絞る。

- ユーザーが明示した
- README、前回議事録、spec、experiment、brainstormから明示リンクされている
- ファイル名が今回topicと強く一致する
- Notion URLとの照合が必要

読み込み後、前提を短く確認する。

```markdown
現在の文脈はこう理解しました。

- プロジェクト: ...
- 資料の起点: ...
- 今回相談したいこと: ...
- 直近MTGからの流れ: ...
- 読む必要がありそうな図・結果: ...

この理解で資料化します。
```

### Phase 3: 入力元の系統判定

資料が何から生まれたかを判定し、読むファイルと構成を変える。

```text
A. spec -> implementation -> experiment/test 結果
B. brainstorm -> 仮説・方針・相談論点
C. 前回MTG / TODO -> 進捗・未解決点
D. 複数ソース混合
```

Aの場合:

- 元specの目的、仮説、実施内容に対して、実装・実験・テスト結果が何を示したかを中心にする。
- 必要セクション: 元の目的、実施したこと、実験条件、結果、解釈、問題、Ask。
- 優先構成:
  ```markdown
  ## 0. 今日のMTGで先生に相談したいこと
  ## 1. 現時点の結論
  ## 2. 元の目的・前回からの流れ
  ## 3. 実施したこと
  ## 4. Evidence: 結果
  ## 5. Interpretation: 解釈
  ## 6. 今回言えること / まだ言えないこと
  ## 7. 方針候補・比較
  ## 8. 次に進む候補
  ## 9. 参照ファイル
  ```

Bの場合:

- 壁打ち内容をそのまま出さず、先生に相談できる仮説、方針候補、判断論点へ蒸留する。
- 必要セクション: 仮説、方針候補、比較、不確実な点、Ask、MTG後に決めたいこと。
- 優先構成:
  ```markdown
  ## 0. 今日のMTGで先生に相談したいこと
  ## 1. 現在の仮説・見立て
  ## 2. 背景
  ## 3. 方針候補
  ## 4. 比較
  ## 5. 不確実な点
  ## 6. 今回言えること / まだ言えないこと
  ## 7. MTG後に決めたいこと
  ## 8. 参照ファイル
  ```

Cの場合:

- 前回決まったこと、対応状況、未解決点、今回相談したいことを中心にする。
- 必要セクション: 前回MTGからの流れ、対応済み、未完了、詰まり、Ask、次に進む候補。
- 優先構成:
  ```markdown
  ## 0. 今日のMTGで先生に相談したいこと
  ## 1. 前回MTGからの流れ
  ## 2. 対応済み
  ## 3. 未完了・詰まり
  ## 4. Evidence: 確認できたこと
  ## 5. Interpretation: 現時点の見立て
  ## 6. 今回言えること / まだ言えないこと
  ## 7. 次に進む候補
  ## 8. 参照ファイル
  ```

Dの場合:

- 重要な起点を混ぜてよいが、資料本文は説明順に整理する。
- 優先構成:
  ```markdown
  ## 0. 今日のMTGで先生に相談したいこと
  ## 1. 現時点の結論・見立て
  ## 2. 背景
  ## 3. Evidence: 根拠・結果
  ## 4. Interpretation: 解釈
  ## 5. 今回言えること / まだ言えないこと
  ## 6. 方針候補・比較
  ## 7. 次に進む候補
  ## 8. 参照ファイル
  ```

入力元別テンプレートは資料の骨格を安定させるための優先構成であり、固定テンプレートではない。内容が重複する場合は統合し、短い資料では一部セクションを省略してよい。ただし、研究資料では過大主張を避けるため、「今回言えること / まだ言えないこと」はできるだけ残す。

### Phase 4: 資料タイプ判定

資料タイプは固定しない。入力元と目的に応じて組み合わせる。

- 進捗報告型
- 方針相談型
- 実験レビュー型
- spec化前相談型
- 複合型

### Phase 5: 根拠抽出

資料に入れる内容を、以下に分けて抽出する。

```text
Evidence: 実験結果、実装結果、観測、既存ファイルから確認できる事実
Interpretation: Evidenceから言えそうな解釈
Ask: 先生に判断・確認してほしいこと
```

主張レベル:

```text
observed: 実験・文脈から観測済み
supported: ある程度支持される
hypothesis: 仮説
speculative: 相談用の推測
unknown: 未確認
```

本文に毎回ラベルを露骨に出す必要はない。ただし、「現時点で言えること」と「まだ言えないこと」は必要に応じて分ける。

### Phase 6: 図の計画

資料を作る前に、図が必要かを判断する。

確認するもの:

- 図があると説明が明確になるか
- 既存図を使うか、新規生成するか
- 実験結果図、MTG用再構成図、概念図、方針比較図のどれか
- 図を見た先生に何を理解してほしいか
- 元データ・元資料は何か

図タイプ:

```text
実験結果図:
  boxplot, heatmap, scatter, line plot, bar plotなど。
  数値・CSV・実験ログに基づく。
  原則 experiments/figures に保存する。

MTG用再構成図:
  既存実験図の選抜、並べ替え、注釈追加。
  materials/figures に保存する。

概念図:
  仮説、手法、更新方向、研究ストーリー、比較軸。
  materials/figures に保存する。

方針比較図:
  候補A/B/C、研究方向マップ、判断フロー。
  materials/figures に保存する。
```

図生成ルール:

- 既存図があるなら、まず再利用・選抜・注釈追加を優先する。
- 図は1つのメッセージに絞る。
- 軸ラベル、単位、凡例、条件、sourceを省略しない。
- 色は強調に使い、色数を増やしすぎない。
- decorativeな図は作らない。
- 図の直前に「この図で見る点」を書く。

`materials/figures/YYYY-MM-DD-topic/` に図を保存する場合は、可能なら `manifest.md` を作る。

```markdown
# Figure Manifest

## Source

- Material: ../../YYYY-MM-DD-topic-mtg-materials.md
- Source files:
  - ...
- Source data:
  - ...

## Figures

| file | type | purpose | source | note |
|---|---|---|---|---|
| 01-concept-map.png | concept | 研究の現在地を説明 | generated | Notion貼付用 |
| 02-result-summary.png | experiment | 主結果を示す | experiments/figures/... | 元図を再利用 |
```

### Phase 7: 説明順への再構成

基本順序:

```text
相談目的
-> 現時点の結論
-> 必要な背景
-> 根拠・結果
-> 解釈
-> 方針候補
-> 先生に確認したいこと
-> 次に進む候補
```

不要な背景説明、作業ログ、未採用案の羅列は削る。必要な場合は「補足」や「参照ファイル」に回す。

### Phase 8: Markdown資料作成

保存先:

```text
.company/lab/projects/<project>/materials/YYYY-MM-DD-topic-mtg-materials.md
```

frontmatter:

```yaml
---
date: YYYY-MM-DD
project: <project>
type: meeting-materials
topic: <topic>
status: draft
target_meeting: YYYY-MM-DD
notion_url: null
notion_export: null
tags: [meeting-materials, research]
---
```

共通構成:

```markdown
# YYYY-MM-DD MTG資料: <topic>

## 0. 今日のMTGで先生に相談したいこと

## 1. 現時点の結論

## 2. 背景

## 3. Evidence: 根拠・結果

## 4. Interpretation: 解釈

## 5. Ask: 確認したいこと

## 6. 今回言えること / まだ言えないこと

## 7. 方針候補・比較

## 8. 次に進む候補

## 9. 図・生成物

## 10. 参照ファイル
```

構成は固定テンプレートではない。内容に応じて不要セクションは省略・統合してよい。ただし、以下は原則として入れる。

- 今日のMTGで先生に相談したいこと
- 現時点の結論、または現時点での見立て
- Evidence / Interpretation / Ask
- 今回言えること / まだ言えないこと
- 参照ファイル

Notion貼付を想定したMarkdown品質:

- 表は横に広げすぎない。
- 数式はNotionで崩れにくい形にする。
- 図パスは資料からの相対パスを基本にする。
- ローカル絶対パスは必要なら参照ファイル欄に回す。
- 長いログや全文引用は入れない。

### Phase 9: セルフレビュー

作成後、完了報告の前に資料を読み返して軽く自己点検する。これは本文に必ず書くものではなく、資料品質を上げるための内部チェックである。

チェック項目:

- 冒頭だけで、相談目的・結論・判断してほしいことが分かるか。
- Evidence と Interpretation が混ざりすぎていないか。
- 観測済みの事実、支持される解釈、未確認事項が区別されているか。
- 「今回言えること / まだ言えないこと」が過大主張を防いでいるか。
- Ask または確認したいことが曖昧すぎないか。
- 表が横に広すぎず、Notionに貼っても読めるか。
- 長いコマンド、生成ファイル一覧、作業ログが本文を圧迫していないか。
- 参照ファイルが資料末尾にまとまっているか。
- このスキルの範囲外である README、TODO、議事録、references を更新していないか。

問題があれば、完了報告前に資料を微修正する。

### Phase 10: 完了報告

作成後は、保存先と相談事項を簡潔に報告する。

```markdown
MTG資料を作成しました。

- 資料: <materials path>
- 図: <figures path or なし>
- 主な相談事項:
  - ...
- Notionに貼る場合:
  - 図リンクが崩れる場合は、figures配下の画像を一緒にアップロードしてください。
```

## エラーハンドリング

- プロジェクト名が曖昧な場合: `.company/lab/projects/*/README.md` を確認して候補を出す。
- READMEがない場合: 対象プロジェクトの確認を優先し、捏造しない。
- 直近議事録がない場合: 「直近議事録なし」と前提に含めて進める。
- 関連spec/experiment/brainstormが見つからない場合: 見つからない前提で資料化し、不足として書く。
- references本文が必要だが対象が曖昧な場合: 全文総読みせず、候補を提示して確認する。
- 同じ日付・同じtopicの資料がある場合: 新規上書きせず、追記または別topic名を確認する。
