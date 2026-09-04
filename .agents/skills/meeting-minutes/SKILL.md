---
name: meeting-minutes
description: MTGの議事録整理・サマリー作成・タスク抽出を自動化するスキル。「MTGが終わった」「打合せ完了」「議事録作って」「文字起こしがある」と言われたとき、または生の文字起こしや当日メモが提供されたときに使う。研究ミーティング（教授との1on1含む）のサマリー作成、Notion参照の照合、TODO候補提示、プロジェクトREADME更新まで一気通貫で行う。会議が終わった文脈では、ユーザーが明示的にスキル名を言わなくても積極的に使用すること。
---

# Meeting Minutes（議事録スキル）

MTG後に、当日メモ・文字起こし・Notion export・プロジェクト文脈を統合し、議事録、TODO候補、プロジェクトREADME更新まで行う。

## 原則

- **当日メモがベース**: 議事録は当日メモの構造を尊重し、文字起こし、Notion export、過去文脈で補強する。
- **Notionは研究室共有の正式記録**: 研究室で共有される議事録・メモとしてNotionを尊重する。
- **ローカルMDはAI運用の正式記録**: ローカルMarkdownは、AIが継続参照する個人ワークスペース上の正式記録として扱う。
- **Notion exportは重要文脈**: 当日メモにNotion URLがある場合、対応するMarkdown export/referenceを `lab/projects/<project>/references/` に揃えてから議事録生成を開始する。
- **Notion URLなしなら進める**: 当日メモにNotion URLがない場合は、Notionなしで当日メモ・文字起こし・ローカル文脈から議事録を生成する。
- **1on1発話者判定**: 文字起こしの話者ラベルが不完全でも、教授との1on1対話を想定して文脈から発話者を推定する。
- **Raw Transcriptは保存しない**: 文字起こし全文は補助入力としてのみ使い、議事録本文には残さない。議事録ファイルの保存成功後にRawファイルを削除する。
- **既存ファイルは壊さない**: READMEやTODOは、既存内容を上書きせず、必要箇所の更新または追記で反映する。

## 対象パス

Company運用の既存パスを使う。

```text
lab/projects/<project>/README.md
lab/projects/<project>/meetings/YYYY-MM-DD-mtg.md
lab/projects/<project>/meetings/YYYY-MM-DD-mtg-raw.md
lab/projects/<project>/references/*.md
secretary/todos/YYYY-MM-DD.md
```

## Workflow

### Phase 1: 初期確認

ユーザーが「MTG終わった」「議事録作って」「文字起こしがある」などと言ったら、次を確認する。

```markdown
お疲れ様でした！議事録の整理を開始します。

まず、今回のMTGについて教えてください。

1. プロジェクト名: どのプロジェクトに関するMTGですか？（例: lora-dynamic-separation）
2. 日付: いつのMTGですか？（デフォルト: 今日 YYYY-MM-DD）
3. 参加者: 誰が参加しましたか？（例: 教授, ユーザー）
```

すでに会話内でプロジェクト名・日付・参加者が分かる場合は、分かる項目を再質問しない。曖昧な項目だけ確認する。

### Phase 2: コンテキスト読み込み

プロジェクト名が確定したら、内部処理として以下を読む。

必須:

- `lab/projects/<project>/README.md`

存在すれば読む:

- `lab/projects/<project>/meetings/*.md` の最新1件
- `lab/projects/<project>/specs/*.md` の最新1件
- `lab/projects/<project>/experiments/*.md` の最新2件
- `lab/projects/<project>/references/*.md` の一覧

読み込んだ情報は、議事録の文脈補完、README更新、TODO候補抽出に使う。

### Phase 3: Rawファイル作成と入力依頼

文字起こし全文はチャットに貼らせず、Rawファイル経由で受け取る。

1. `lab/projects/<project>/meetings/YYYY-MM-DD-mtg-raw.md` を作成する。
2. ユーザーに以下を伝える。

```markdown
コンテキストを読み込みました。

以下のファイルを作成しました。
[lab/projects/<project>/meetings/YYYY-MM-DD-mtg-raw.md](絶対パス)

このファイルに文字起こし全文を貼り付けて保存してください。

保存できたら、このチャットに「完了」と書き、同時に当日メモを貼り付けてください。

当日メモの例:

\```markdown
# やったこと
- ...

# 次やること
- ...

# 話したこと
- ...
\```
```

### Phase 4: 当日メモ受信とNotion URL照合

ユーザーが「完了」+ 当日メモを送ったら、次を行う。

1. Rawファイルを読み、文字起こしが空でないことを確認する。
2. 当日メモからNotion URLを抽出する。
3. Notion URLがなければ、Phase 5へ進む。
4. Notion URLがある場合、`lab/projects/<project>/references/*.md` と照合する。

Notion URL照合ルール:

- URLから32文字のNotionページIDを抽出する。
- `references/` 内のMarkdownファイル名にそのIDが含まれていれば対応済みとみなす。
- すべて対応済みなら、関連するreferenceファイルを読んでPhase 5へ進む。
- 1つでも不足していれば、議事録生成を開始しない。

不足がある場合は、以下の形式でユーザーへ依頼する。

```markdown
当日メモ内にNotion URLがありますが、対応するMarkdown exportが `references/` にありません。

以下のページをMarkdownでエクスポートし、次のフォルダに保存してください。

- 保存先: `lab/projects/<project>/references/`
- ファイル名: Notionのデフォルト名のままでOKです。ただし、ファイル名にNotionページIDが含まれていることを確認してください。

不足しているURL:
- <notion-url-1>
- <notion-url-2>

保存できたら「完了」と教えてください。再確認してから議事録生成を開始します。
```

ユーザーが「完了」と言ったら、`references/` を再確認する。すべて揃うまでPhase 5へ進まない。

### Phase 5: 議事録生成

優先順位:

1. 当日メモ
2. 文字起こし
3. Notion export/reference
4. 既存コンテキスト（README, specs, experiments, previous meetings）

生成方針:

- 当日メモの「やったこと」「次やること」「話したこと」を基本構造にする。
- 文字起こしから、議論の流れ、補足説明、決定事項、次アクションの根拠を補う。
- Notion exportがある場合は、背景・詳細・研究室共有メモとして重要な内容を統合する。
- 既存コンテキストから、前回の決定事項、現在状況、関連するspec/experimentへのリンクを補う。
- 文字起こし全文は貼り付けない。必要な内容だけ要約・引用なしの要旨として反映する。

1on1発話者判定:

- 質問・助言・提案・方向付けは教授側の発言候補として扱う。
- 実装報告・実験報告・作業状況説明はユーザー側の発言候補として扱う。
- 不確実な場合は断定せず、「議論では」「フィードバックとして」のように中立的に書く。

議事録の保存先:

```text
lab/projects/<project>/meetings/YYYY-MM-DD-mtg.md
```

議事録フォーマット:

```markdown
---
date: YYYY-MM-DD
project: <project-name>
participants: [<participant-1>, <participant-2>]
tags: [meeting, <project-name>]
---

# YYYY-MM-DD MTG 議事録

## 会議情報

- 日時: YYYY-MM-DD
- 参加者: <participants>
- プロジェクト: <project-name>

## やったこと

...

## 話したこと

...

## 決定事項

- ...

## 次やること

- [ ] ...

## 関連リソース

### Notion / 研究室共有メモ

- ...

### プロジェクト内ファイル

- ...

## 当日メモ（原文）

<details>
<summary>当日メモ全文</summary>

...

</details>
```

当日メモ原文は、ユーザーが貼った内容を再確認できるように議事録内へ残してよい。ただし、文字起こし全文は残さない。

### Phase 6: Rawファイル削除

議事録ファイルを保存し、内容が空でないことを確認してからRawファイルを削除する。

```bash
rm lab/projects/<project>/meetings/YYYY-MM-DD-mtg-raw.md
```

削除前に確認すべきこと:

- `lab/projects/<project>/meetings/YYYY-MM-DD-mtg.md` が存在する。
- 議事録に会議情報、やったこと、話したこと、次やることが含まれている。
- 文字起こし全文が議事録本文に残っていない。

### Phase 7: TODO候補提示と反映

当日メモ・議事録からNEXT ACTIONSを抽出し、ユーザーへ候補として提示する。

```markdown
## 抽出されたNEXT ACTIONS

以下のタスクを抽出しました。

1. [ ] ...
2. [ ] ...
3. [ ] ...

本日のTODOリスト（`secretary/todos/YYYY-MM-DD.md`）に反映するものを教えてください。

- 優先度があれば指定してください（高/通常/低）。
- 期限がある場合のみ指定してください。
- 追加・削除・表現調整もできます。
```

TODO追加ルール:

- ユーザーが採用したタスクだけ追加する。
- 優先度未指定は `通常` にする。
- 期限は明示された場合のみ付ける。
- 既存TODOを重複追加しない。

TODO形式:

```markdown
- [ ] タスク内容 | 優先度: 高/通常/低
- [ ] タスク内容 | 優先度: 通常 | 期限: YYYY-MM-DD
```

### Phase 8: README更新

`lab/projects/<project>/README.md` はAI判断で更新する。ユーザーに逐一確認しなくてよい。

更新対象:

- frontmatterの `summary`
- frontmatterの `last_updated`
- `## 現在の状況`
- `## マイルストーン`
- `## 更新履歴`

更新ルール:

- `summary` はダッシュボード用の1行要約として、MTG後の現在状況が分かる内容にする。
- `last_updated` はMTG日または作業日の日付に更新する。
- `## 現在の状況` は、今回のMTGで明確になった研究フェーズ・方針・論点を反映する。
- `## マイルストーン` は、必要な項目を追記する。既存の項目を上書き・削除しない。
- `## 更新履歴` には、`YYYY-MM-DD: <MTGで決まった要点>` の形で短く追記する。

frontmatter例:

```yaml
---
project: lora-dynamic-separation
status: in-progress
department: lab
summary: Hessian/HVPによるold-task loss上昇方向の解析方針を検討中
created: YYYY-MM-DD
last_updated: YYYY-MM-DD
---
```

### Phase 9: 完了報告

すべて完了したら、次を簡潔に報告する。

```markdown
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  Meeting Minutes 完了
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

議事録を作成しました:
[lab/projects/<project>/meetings/YYYY-MM-DD-mtg.md](絶対パス)

TODOに追加したタスク:
- ...

READMEを更新しました:
- summary: ...
- 更新履歴: ...

Raw文字起こしは削除しました。
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
```

## エラーハンドリング

- プロジェクト名が曖昧な場合: `lab/projects/*/README.md` を確認して候補を出し、ユーザーに確認する。
- Rawファイルが空の場合: 議事録生成を止め、文字起こしの貼り付けを依頼する。
- Notion URLあり・reference不足の場合: 議事録生成を止め、Markdown export保存を依頼する。
- 議事録保存に失敗した場合: Rawファイルを削除しない。
- TODO候補がない場合: TODO追加は行わず、README更新と完了報告に進む。
