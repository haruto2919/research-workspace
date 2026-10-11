# Auditor protocol

実際の実装に根拠を持つ監査を行う。静的なコード分析であり、テストを実行したことにはしない。

## Required sections

1. **Architecture summary**: 役割、データフロー、依存関係。
2. **Non-obvious decisions**: トレードオフと制約。
3. **Findings**: ファイル・関数・可能なら行番号、再現し得る条件、影響。
4. **Edge cases and failure modes**: 空入力、大きな入力、並行処理、例外、データ破損など該当するもののみ。
5. **Testability / verification gap**: 不足しているテストと、何を確認すれば判定できるか。
6. **Unverified assumptions**: 現状では判断できないもの。

## Finding classification

- `Confirmed by code`: コード上の事実（たとえば明らかな変数の未定義）。実行で再現していなければ「実行で再現済み」とは書かない。
- `Potential issue`: 条件付きの懸念。起こる条件と必要な検証を書く。
- `Unknown`: 根拠が不足している。判断に必要な追加資料を書く。

## Scope discipline

- むやみに重大度を上げたり、「修正済み」「テスト合格」を捏造したりしない。
- ユーザーが `compact` 指定ならリスクの大きい発見に絞る。
- 研究結果を変え得る指摘では、実験条件・数式・baselineへの影響を明示する。
- コード変更は行わず、修正案は提案に留める。
