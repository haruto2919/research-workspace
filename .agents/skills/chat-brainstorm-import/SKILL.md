---
name: chat-brainstorm-import
description: ChatGPTなど外部AIとの研究相談・壁打ち・議論をResearch Workspaceへ取り込む専用スキル。「このチャットを研究workspaceに保存して」「ChatGPTで話した内容をbrainstormとして取り込んで」「この会話をresearch-workspaceに保存して」などと言われたときに使う。外部チャットの内容を既存研究文脈と照合し、brainstormスキルの保存形式に従って記録する。新しい壁打ち、spec作成、TODO追加、研究コード変更は行わない。
---

# Chat Brainstorm Import

ChatGPTなど外部AIサービスで行われた研究相談・壁打ちを、
Research Workspaceの既存研究文脈へ安全に取り込む。

このスキルは新しい研究相談を行うためのものではない。
すでに行われた会話を、`brainstorm` スキルの規則に従う研究記録へ変換するための入口である。

## 責務

担当する:

- ユーザーが渡した外部チャットの対象範囲を特定する。
- Research Workspaceの既存研究文脈を確認する。
- `.agents/skills/brainstorm/SKILL.md` を読み、保存形式と分類規則を取得する。
- 外部チャット内の問い、仮説、候補、Evidence、判断、保留、未解決事項を整理する。
- ユーザー発言、AI提案、外部情報を可能な範囲で区別する。
- 採用済み、候補、保留、棄却、未決を区別する。
- brainstormメモとしてResearch Workspaceへ保存する。

担当しない:

- 新しい壁打ちを開始する。
- チャットに存在しない仮説や実験案を追加する。
- AI提案をユーザーの決定として扱う。
- TODOを自動追加する。
- READMEを自動更新する。
- `specs/` を作成・更新する。
- 研究コード、config、launcher、testを変更する。
- 実験を実行する。

## Single Source of Truth

Research Workspaceの研究文脈は `.research/` を正本とする。

brainstormメモの構造、保存規則、frontmatter、追記規則については、

`.agents/skills/brainstorm/SKILL.md`

をSingle Source of Truthとして扱う。

このSkill内でbrainstorm形式を独自に再定義しない。

## Workflow

### Phase 1: 外部チャットの対象範囲を特定

ユーザーから渡されたチャット、会話ログ、要約、貼り付けテキストから、
今回保存する研究相談の範囲を特定する。

確認するもの:

- 会話の中心テーマ
- 対象研究プロジェクト
- 会話の開始点
- 会話で検討した問い
- 明示的に決定されたこと
- AIから提案されただけのこと
- 未解決事項

現在の依頼と会話から十分判断できる場合は、ユーザーへ再質問しない。

複数の研究テーマが混在し、1つのbrainstormメモにまとめると意味が失われる場合は、
テーマごとに分ける。

### Phase 2: Workspaceルールを読む

次を読む。

必須:

- `AGENTS.md`
- `.research/secretary/AGENTS.md`
- `.agents/skills/brainstorm/SKILL.md`

対象プロジェクトが分かる場合は、brainstormスキルの文脈読込規則に従って
必要なREADME、TODO、meeting、experiment、spec、referenceを読む。

外部チャットだけを根拠に既存研究状況を推測しない。

### Phase 3: プロジェクトを特定

外部チャット内の研究テーマと、

`.research/lab/projects/*/README.md`

を照合して対象プロジェクトを特定する。

明確に一致する場合はそのプロジェクトを使用する。

特定できない場合でも、無理に既存プロジェクトへ割り当てない。
brainstormスキルが許す場合は `project: null` として保存する。

複数プロジェクトが同程度に該当し、選択によって意味が変わる場合だけ、
保存前にユーザーへ確認する。

### Phase 4: 外部チャットを分類

チャット内の情報を少なくとも次へ分類する。

- ユーザーが提示した事実・前提
- ユーザーが明示的に採用した方向
- AIが提案した候補
- 外部情報・文献由来の情報
- 保留された案
- 棄却された案
- 未解決事項
- 次アクション候補

ユーザーが「なるほど」「分かった」などと返しただけでは、
AI提案を採用済み決定と判断しない。

会話に存在しない決定、Evidence、数値、実験結果を補完しない。

### Phase 5: Workspace文脈と照合

外部チャットの内容を既存Research Workspaceと比較する。

確認するもの:

- 既に決まっている研究方針との矛盾
- 既存brainstormとの重複
- 既存specですでに確定している事項
- experimentで確認済みのEvidence
- 最新meetingでの決定

矛盾が存在する場合は、外部チャット側を自動的に最新・正しいものとして扱わない。

brainstormメモ内で、

- 既存文脈
- 外部チャットで出た内容
- 矛盾・差分
- 要確認事項

を区別する。

### Phase 6: brainstorm形式へ変換

`.agents/skills/brainstorm/SKILL.md` のPhase 7とPhase 8に従って構造化する。

少なくとも次を保持する。

- 相談の出発点
- 中心となる問い
- 読み込んだ既存研究文脈
- 仮説
- 実験案・実装案
- 比較軸
- 現在の方向性
- 採用候補
- 保留
- 棄却
- 未解決事項
- 次アクション候補
- specへ引き継げる事項
- 関連ファイル

内容を短くすることよりも、
後から「なぜこの案が出たか」「何が決まり、何が決まっていないか」
を再現できることを優先する。

### Phase 7: 保存先を決定

brainstormスキルの保存規則に従い、

`.research/secretary/notes/brainstorm/YYYY-MM-DD-topic.md`

へ保存する。

ファイル名は、

`YYYY-MM-DD-kebab-case-topic.md`

とする。

同日・同topicのファイルがすでに存在する場合は、
新しい重複ファイルを作らずbrainstormスキルの規則に従って追記する。

既存内容を丸ごと上書きしない。

### Phase 8: 保存

frontmatterはbrainstormスキルの定義を使用する。

外部チャット由来であることを示す必要がある場合は、
brainstormスキルの必須項目を壊さない範囲で次を追加してよい。

`source: external-chat`

または、由来が明確な場合:

`source: chatgpt`

本文には必要に応じて、

`## Source`

を設け、外部チャット由来であることを記録する。

会話全文をそのまま保存するのではなく、
研究判断を再現できる構造化記録として保存する。

### Phase 9: 完了報告

保存後、次を簡潔に報告する。

保存結果:

- project
- topic
- 保存path
- 新規作成 / 既存ファイルへ追記
- 採用済み事項の件数
- 未解決事項の件数
- TODO追加: なし
- spec更新: なし

必要であれば、次に `brainstorm` を続けるか、
ユーザーが明示的に希望した場合のみ `research-spec` へ引き継ぐ。

## 禁止事項

- 外部AIの回答を研究上の確認済み事実として無条件に扱う。
- AI提案をユーザー承認済みとみなす。
- 外部チャットにない実験結果を作る。
- 既存Research Workspaceと矛盾した内容を黙って上書きする。
- ユーザー承認なしにTODOを追加する。
- ユーザー承認なしにspecを作成する。
- brainstorm保存と同時にREADMEを更新する。
- 外部チャット全文を無加工でresearchメモへ貼り付ける。