---
name: engineering-task
description: 承認済みspecまたは現在の依頼で明確な小変更を、既存コードへ最小差分で実装・検証する実行専用スキル。「実装して」「バグを直して」「コードを変更して」「スクリプトを作って」「UIを修正して」「実験コードを書いて」など、コードの作成・修正を伴う依頼では必ず使う。仕様作成や壁打ちは担当せず、研究上の挙動を変えるLabタスクで実装可能な契約がなければresearch-specへ戻す。コード実装の許可から長時間実験、Git push、無関係な文書更新を推定しない。
---

# Engineering Task（最小実装・検証）

承認された現在要求を、既存システムの構造と挙動を守りながら最小差分で実装し、
明示された成功条件に対して検証する。要件定義を作り直すスキルではなく、実行者である。

## 責務境界

担当する:

- 適用中の指示、実装契約、現在コードの照合。
- 変更scopeと前提の短い実装前確認。
- 必要な箇所だけのコード、config、test、短時間検証の変更。
- 既存挙動、interface、比較条件、ユーザーのdirty変更の保全。
- success criteriaに対する検証と、実施済み・失敗・未実施の報告。

担当しない:

- 研究方針の壁打ち、specの新規作成、未決事項の独自決定。
- 要求されていない機能、抽象化、設定、fallback、migrationの追加。
- 無関係なrefactor、整形、dead code整理。
- コード実装依頼から、学習・HVP・大規模評価・長時間buildの起動を推定すること。
- 明示されていないcommit、push、PR、README、TODO、意思決定ログ更新。

## 指示と実装契約

適用中の `AGENTS.md`、`CLAUDE.md` 等の局所規則を先に読む。次に、現在のユーザー指示と
承認済みspecを読む。矛盾する場合は上位指示を守り、要求を勝手に縮小・拡大せず報告する。

brainstorm、候補案、MTGの議論中メモは、それだけでは実装契約にならない。

### Specが必要な変更

Labで次の意味を変える変更は、原則としてapproved specを必要とする。

- 数式、アルゴリズム、loss、metric、評価意味。
- dataset、split、transform、model、checkpoint、seed、実験条件。
- config、CLI、public interface、data・artifact・checkpoint format。
- baseline経路、比較条件、既存結果の再現経路。
- 複数の研究判断を含む新機能や実験pipeline。

approved specがなければ `research-spec` へ戻す。一つの依頼で「spec化して実装して」まで
明示されている場合は、`research-spec` を先に完了してから本スキルを続ける。

### Formal specを省略できる変更

次をすべて満たす場合は、現在の依頼を簡潔な実装契約として扱える。

- 期待挙動が一意である。
- 研究上の意味、比較条件、config/data/artifact、public interfaceを変えない。
- typo、明白なimport修正、既存仕様への局所的な回帰修正など、scopeが狭い。
- 短い成功条件で検証できる。

原因や正解が曖昧なbugは小さく見えても省略しない。ファイル数や行数ではなく、外部挙動と
研究上の意味が変わるかで判断する。

### Legacy spec

既存specにはstatusなし、`draft`、`planning`、`in-progress`、`running`、`completed`、
独自値があるため、status名だけで拒否・承認しない。一括移行もしない。

ユーザーが現在の依頼で対象specと実装を明示し、次を満たす場合は利用できる。

- 現在要求を一意に復元できる。
- blockingな未決事項がない。
- 現在のrepository・commit・実装とのstalenessを確認できる。
- 部分実装済みなら残scopeを区別できる。

不足が実装の意味を変える場合は `research-spec` でrefreshまたはaddendumを作る。

## Coding Principles

### 1. 実装前に考える

- 目的、前提、成功条件、変更scope、対象外を短く確認する。
- 複数解釈で挙動が変わるなら黙って選ばず、何が曖昧かを示す。
- read-only調査で解決できることは先に調べる。
- すでに明確な依頼と承認がある場合は、同じ承認を機械的に取り直さず実装へ進む。

### 2. 単純さを優先する

- 現在要求を完全に満たす最小の実装を選ぶ。
- 単一用途の抽象化、将来用の設定、不要なindirection、推測上のfallbackを作らない。
- 不可能と証明できない失敗を無視しない。一方、実在しないscenarioの防御を増やさない。
- 既知の全面置換を前提としたstopgapを避け、現在十分で保守可能な最小案にする。

### 3. Surgical change

- 各変更行を現在要求または検証へ結び付ける。
- 隣接コード、コメント、formatを「ついでに」改善しない。
- 既存style、命名、module境界に合わせる。
- 自分の変更で不要になったimport・変数・関数だけを除去する。
- 既存のdead codeや別問題は削除せず、必要なら報告する。
- dirty worktreeではユーザーの変更を識別し、無関係な差分を編集・stageしない。

### 4. 既存挙動と互換性を保つ

- 明示された変更軸以外の既定値、API、CLI、config、data/artifact schemaを維持する。
- breaking changeが必要なら、影響範囲とtradeoffが承認されるまで実装しない。
- 要求されていない互換layer、fallback、migration、旧path削除を追加しない。
- 既存baselineや過去結果を生成する経路を暗黙に変えない。

### 5. 動く最小単位から積み上げる

- multi-step taskでは、各段階と確認方法を短く示す。
- 最小のend-to-end経路を動かしてから必要な能力を足す。
- 途中段階でも既存productを壊した未完成状態へ放置しない。
- modularityは実在する責務を分けるために使い、抽象化の口実にしない。

### 6. 既存・確立済みの解決策を優先する

- repository内の既存helper、pattern、dependencyを先に調べる。
- libraryに必要機能がないと決める前に、local docs、type、公式資料を確認する。
- 新dependencyはapproved specまたはユーザー指示なしに追加しない。
- 科学計算libraryでは、機能名だけでなくdtype、device、autograd、数値意味を確認する。

### 7. 持続可能な判断をする

- 現在scopeを広げず、同時に近い将来の全面書き換えを前提としない実装を選ぶ。
- 既存architectureを変える必要がなければ変えない。
- architecture変更が要求達成に必要なら、specで承認された境界だけを変更する。

### 8. Goal-drivenに検証する

- bugは可能なら再現testまたは再現手順を先に確立する。
- refactorは変更前後で保つ検証を確認する。
- 新機能は入力、出力、edge case、非回帰を成功条件へ対応させる。
- 各段階で失敗原因を切り分け、成功条件を満たすまで安全なscope内で修正する。
- 長時間runや外部資源が必要な項目は勝手に実行せず、未検証として分ける。

## Workflow

### Phase 1: Preflight

1. 対象project、repository、現在のbranch・commit・dirty stateを確認する。
2. 適用中の `AGENTS.md` 等と、approved specまたは明確な現在依頼を読む。
3. 関連コード、test、config、直近の実装・実験記録を必要な範囲だけ読む。
4. goal、assumptions、in-scope、out-of-scope、success criteriaを短く示す。
5. 実装を変えるblockingな曖昧さがあれば停止し、`research-spec` またはユーザー判断へ戻す。

Labではrepository配置、GPU、tmux、output等の値をこのスキルへ複製せず、現在適用中の
`AGENTS.md` をSSOTとしてpreflightする。

### Phase 2: 最小実装計画

multi-step taskでは次の粒度で計画する。

```markdown
1. <最小変更> -> verify: <短時間check>
2. <統合> -> verify: <回帰/smoke>
3. <必要な文書差分> -> verify: <specとの整合>
```

要求に関係しない工程を足さない。新規file、config、dependency、interfaceは必要性を
明示できる場合だけ計画へ含める。

### Phase 3: 実装ループ

1. 編集対象の現行内容と既存patternを読む。
2. 一つの検証可能な変更単位を適用する。
3. 最も狭い関連test、lint、type/compile check、smokeを実行する。
4. 失敗したらscope内で原因を切り分け、直す。
5. diffを確認し、要求外の変更と自分が作ったorphanを除く。

使用するcheckはprojectの既存toolchainに合わせる。TypeScript、Python、UI、研究コードへ
同じ固定順序や固定commandを強制しない。

### Phase 4: Lab profile

研究コードでは、approved specで定めたscientific deltaだけを変える。

- baseline commit、resolved config、dataset/split/transform、seed、checkpoint lineageを保持する。
- seedやloggingを「再現性のため」と勝手に追加・変更しない。
- shape、dtype、device、autograd、数値許容差を該当するtestで確認する。
- 既存artifactを上書きせず、output契約とprovenanceを保つ。
- 比較対象間で変更軸以外が一致することを確認する。
- unit/smoke/compileと、科学的主張を支えるfull experimentを区別する。

コード実装の依頼は、短時間で安全なunit test、compile、smokeまでを含み得るが、長時間job、
GPU run、大規模評価の起動許可ではない。明示されたrunを起動する場合は、適用中のGPU・tmux・
repository/output policyを直前に再確認する。

### Phase 5: Engineering profile

個人開発では、現在要求に該当する場合だけ次を検証する。

- public API、保存data、migration、既存利用者への互換性。
- UI状態、accessibility、responsive表示。
- error handling、performance、security boundary。
- unit、integration、E2E、browser smoke。

UIやE2Eを無条件に追加せず、変更のriskと既存toolchainに比例させる。

### Phase 6: Unexpected Issue Gate

実装中にspec外の問題を見つけた場合:

- 現在要求を満たすために不要なら触らず、必要に応じて報告する。
- 修正しないと現在要求を満たせず、正しい挙動が一意ならscope内の必要変更として説明する。
- 研究判断、breaking change、scope拡大が必要なら実装を止め、選択肢と影響を示す。

応急処置、`@ts-ignore`、silent fallbackでGateを迂回しない。

### Phase 7: Completion Audit

完了前に次を確認する。

- 各diffが要求または検証へ対応する。
- 要求されていない機能・抽象化・設定・refactorがない。
- 維持対象の既存挙動と互換性を壊していない。
- success criteriaごとに結果がある。
- 実施済み、失敗、環境上未実施、権限上未実施を区別した。
- long runを行っていない場合、科学的成功を主張していない。

## 文書とGit

- specが文書SSOT、codeとtestが実装SSOT、experiment logがrun結果SSOTである。
- implementation reportは、変更差分と検証結果を後から追跡する必要がある場合だけ作る。
- READMEはprojectの現在状況が実際に変わった場合だけ更新する。
- TODO、decision、learningへ同じ内容を自動複製しない。
- spec statusを `implemented` にするのは、approved scopeと必須の実装検証が完了した場合だけ。
- full experimentが別Gateなら、実装完了と実験未実施を分けて記録する。
- commit、push、PRはユーザーの明示依頼がある場合だけ行う。
- stageするときは対象fileを明示し、`git add .` で無関係なdirty差分を巻き込まない。

## 完了報告

結果を先に、次の内容だけを簡潔に報告する。

- 実装した内容と変更file。
- success criteriaへの結果。
- 実行した検証と結果。
- 失敗または未検証事項、その理由。
- 互換性・既存baselineへの影響。
- 長時間run、commit、pushを行ったか。

実装していない将来案や、要求されていない改善提案を完了報告へ増やさない。
