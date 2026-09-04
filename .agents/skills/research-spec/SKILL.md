---
name: research-spec
description: 研究のbrainstorm、MTG決定、実験案、実装方針を、承認可能で検証可能なspecへ変換するスキル。「spec化できるか見て」「spec化して」「実装specを作って」「実験計画を要件定義にして」「この方針を実装可能な形に固めて」と明示されたときは必ず使う。read-onlyのコード・文脈調査、曖昧性解消、scope・互換性・再現性・成功条件のGate、ユーザー承認後のspec保存を担当する。壁打ち、コード変更、学習や長時間実験の実行には使わない。
---

# Research Spec（研究spec化）

探索で得た方向性を、実装者が追加の研究判断をせずに実行・検証できる変更契約へ
変換する。specは研究背景の複製ではなく、承認された現在要求のSingle Source of Truthである。

## 責務境界

担当する:

- brainstorm、MTG、実験結果、既存specから確定事項を抽出する。
- 対象コードをread-onlyで調査し、既存構造と影響範囲を把握する。
- 目的、前提、scope、対象外、互換性、再現性、成功条件を確定する。
- blockingな曖昧さを表面化し、選択肢とトレードオフを示す。
- ユーザー承認後にプロジェクトの `specs/` へ保存する。
- 実装開始時に `engineering-task` が使える引き継ぎを示す。

担当しない:

- 仮説や研究方針の自由な発散。必要なら `brainstorm` へ戻す。
- 研究コード、config、test、launcherの変更。
- 学習、HVP、大規模評価、長時間buildの実行。
- 曖昧な設計を「一般的にはこうする」と独自に決めること。
- README、TODO、実装報告の自動更新。

## Authorityと承認

情報の優先順位は、現在のユーザー指示、明示されたMTG決定、承認済み既存spec、
関連Evidence、探索中brainstormの順とする。内容が衝突する場合は、最新時刻だけで
勝手に選ばずユーザーへ示す。

brainstormメモは根拠と候補の記録であり、実装権限ではない。specは次の状態を使う。

```text
draft -> approved -> implemented
                 \-> superseded
```

- `draft`: 内容を確認中。実装不可。
- `approved`: ユーザーが内容を明示承認。実装可能。
- `implemented`: 承認内容の実装・必須検証が完了。
- `superseded`: 別specに置き換えられ、実装不可。

既存specのstatusは一括移行しない。古い `planning`、`in-progress`、`completed`、独自値、
statusなしのspecを実装に使う場合は、ユーザーが現在の依頼で対象specを権威ある入力として
明示し、内容と現在コードの不一致を確認する。新規specでは上記状態を使う。

「specを作って」はdraft作成の許可であって、コード実装の許可ではない。
「この内容でspec化して実装して」のように、spec内容と実装の両方が明示され、Gateを
満たす場合はapprovedで保存して `engineering-task` へ引き継げる。長時間runは別途許可を要する。

## 原則

### 考えてから固定する

- 前提を明示し、不確実な点を隠さない。
- 複数解釈が研究主張、データ、interface、結果を変えるなら選択肢を提示する。
- より単純な方法で要求を完全に満たせるなら、その方法を第一候補にする。
- blockingな不明点が残る場合はspecをapprovedにしない。

### 現在要求に限定する

- 要求されていない機能、抽象化、設定項目、fallback、migrationを追加しない。
- 単一用途のためだけの汎用層を設計しない。
- Future Enhancementsをspecの必須節にせず、探索候補はbrainstormへ残す。
- ただの応急処置ではなく、現在要求を満たし、近い将来の全面置換を前提としない
  最小の持続可能な設計を選ぶ。

### 既存システムを基準にする

- 既存構造、命名、dependency、libraryの能力をread-onlyで確認する。
- 一般的な機能を再実装する前に、既存dependencyで十分か確認する。
- public interface、config/data format、CLI、既存baselineを壊す変更は、必要性、影響、
  移行方法が承認されるまで採用しない。
- 不要な互換layerやfallbackを推測で追加しない。

## 対象パス

```text
.company/secretary/notes/brainstorm/*.md
.company/lab/projects/<project>/README.md
.company/lab/projects/<project>/meetings/*.md
.company/lab/projects/<project>/experiments/*.md
.company/lab/projects/<project>/specs/*.md
.company/lab/projects/<project>/implementation/*.md
.company/lab/projects/<project>/references/*.md
.company/lab/projects/<project>/specs/YYYY-MM-DD-topic-spec.md
<research-code-repository>  # read-only調査だけ
```

## Workflow

### Phase 1: 入力と対象を特定する

次を特定する。会話から分かることは再質問しない。

- プロジェクトとspec種別: 実装、実験、分析、資料・運用。
- 元になったbrainstorm、TODO、MTG決定、既存spec。
- ユーザーが確定した内容と、まだ候補に留まる内容。
- specをdraftとして作るのか、承認可能な状態まで固めるのか。

探索が不足し、複数案の比較自体が目的なら `brainstorm` へ戻す。

### Phase 2: 文脈と現行実装をread-onlyで調べる

必須:

- プロジェクトREADME。
- 元brainstormまたは決定記録。
- 関連する最新meeting、experiment、spec。
- 対象repositoryの `AGENTS.md` 等の局所指示。

必要に応じて:

- 現行コード、test、config、CLI、dependency定義。
- 基準commit、worktree、既存output契約。
- 過去のimplementation report。

この調査ではコードを編集しない。新しいworktree、config、output、実験artifactも作らない。

### Phase 3: Decision Contractを抽出する

次を採用、対象外、未決へ分ける。

- 目的と解決したい問題。
- 研究上の問い・仮説、または実装上の期待挙動。
- 必須要件。
- 明示的な対象外。
- 維持すべき既存挙動と互換性。
- 許可された変更範囲。
- 採用した設計と、その判断に必要な理由。
- 棄却・保留案。再導入を防ぐために重要なものだけspecへ短く残し、詳細はbrainstormを参照する。

各要件が現在の依頼または承認済み判断へ遡れることを確認する。

### Phase 4: Ambiguity Gate

不確実性を次へ分ける。

- `blocking`: 選択によって外部挙動、研究主張、比較条件、互換性、変更範囲が変わる。
- `non-blocking`: 実装者が既存styleに従えば結果が変わらない局所的詳細。

blocking項目は、何が曖昧か、選択肢、影響、推奨があれば理由を示して確認する。
non-blocking項目は「既存実装に合わせる」などの制約として記録する。approvedにする時点で
blocking項目を残さない。

spec readinessを回答するときは、`blocking` と `non-blocking` を別々に示す。該当項目が
ない、または元議論が提示されずまだ分類できない場合も、その状態を明記して黙って省略しない。

### Phase 5: 最小で持続可能な設計にする

設計を次で監査する。

- 現在要求を完全に満たす最小構成か。
- 一回しか使わない抽象化や不要な設定を増やしていないか。
- 信頼できる既存libraryや既存helperを使えるか。
- 後で全面置換する前提のstopgapになっていないか。
- 既存コードの責務境界を壊さず、必要な責務だけを分離しているか。
- 変更予定の各componentが要件へ直接対応しているか。

単純化で要件を失う場合は削らない。単純さは行数の少なさだけでなく、理解・検証・保守の
総コストで判断する。

### Phase 6: 研究再現性と比較可能性を固定する

該当する項目だけを確定し、値を推測しない。

- repository、基準commit、worktree方針、dirty stateの扱い。
- dataset、split、transform、model、checkpoint lineage。
- seed、config、hyperparameter、baseline、比較時に固定する条件。
- metric、集計、成功・失敗・停止条件。
- output path、artifact、log、resume・overwrite方針。
- GPU、tmux、計算資源、長時間runの実行境界。

コード実装の承認と実験runの承認を分離する。unit testや短いsmokeを超える実行は、
specに書かれていてもユーザーの現在の実行指示または既存の明示承認が必要である。

### Phase 7: 検証可能な成功条件へ変換する

曖昧な要求を観測可能な結果へ変える。

- bug修正: 失敗を再現するtestまたは手順と、修正後の期待結果。
- 新機能: 入力、出力、境界条件、既存挙動の非回帰。
- refactor: 変更前後で維持するtestと外部interface。
- 実験: baseline、metric、比較条件、失敗しても報告すべき結果。

検証項目は「実行する予定」と「このspecでは対象外」を区別する。full runをunit testのように
扱わず、実装成功と科学的成功を別の判定にする。

### Phase 8: Spec Gateとレビュー

approved候補には最低限次が必要である。

- 目的と前提。
- 採用内容と対象外。
- blockingな未決事項がない。
- 互換性・既存baselineへの影響。
- 実装または実験の手順と影響範囲。
- 検証可能な成功条件。
- 必要な再現性・比較条件。
- ユーザーの明示承認。

保存前に、採用内容、対象外、breaking change、検証、未検証予定を短く提示する。
すでに現在の依頼で内容と保存・実装が明示承認されている場合は、同じ確認を繰り返さず進める。

readiness確認だけを求められた場合は、現在の担当が `research-spec` であること、判定、
blocking、non-blocking、次に必要な判断を示し、specを保存しない。

### Phase 9: 保存と引き継ぎ

保存先:

```text
.company/lab/projects/<project>/specs/YYYY-MM-DD-topic-spec.md
```

既存同topicのspecがあれば、重複を作る前に更新・追記・新versionのどれが正しいか判断する。
既存specを丸ごと上書きせず、履歴と参照を維持する。

specは `.company/templates/requirements-spec-template.md` を基準に、該当しない節を
無理に埋めず、現在要求に比例した詳細度で書く。関連brainstormには昇格先リンクだけを
追記し、spec本文を複製しない。

最後に `engineering-task` へ次を渡す。

```markdown
# Implementation Handoff

- approved spec:
- 実装目的:
- 基準repository/commit:
- 変更scope:
- 対象外・維持条件:
- success criteria:
- 許可されている短時間検証:
- 長時間runの許可状態:
- 未検証予定:
```

このhandoffは新しい正本ではなく、approved specの短い索引である。
