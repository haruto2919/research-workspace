# Research-aware code analysis

研究コードが対象の場合だけ追加で適用する。

## Source alignment

1. プロジェクトREADMEから研究目的と前提を特定する。
2. 対象コードに関係する approved spec、論文メモ、実験記録を必要範囲だけ読む。
3. 研究上の主張・仕様・数式・ソース実装をそれぞれ別の根拠として扱う。
4. 対応する箇所を並べ、実装上の事実と解釈を明確に区別する。

## Checklist for machine-learning code

- 入力・出力 Tensor の shape、dtype、device。
- パラメータ更新、optimizer、scheduler、勾配伝播 / `detach` / `no_grad`。
- loss と評価 metric の定義、集計単位、符号、正規化。
- dataset、split、transform、sampling、checkpoint、seed、config。
- baselineとの変更軸、比較条件、継続学習やresumeの前提。
- 数値安定性、メモリ量、混合精度、分散実行が該当するか。

## Evidence tiers

- `Observed`: 読み取ったコードや設定で直接確認できる。
- `Aligned`: 原典の式とコードが論理的に対応すると確認できる。ただし実験成功を意味しない。
- `Inferred`: もっともらしい目的だが明示の証拠はない。
- `Unverified`: 外部論文・実験結果・実行時挙動など未確認。

## No overclaims

- 論文全文を確認していなければ「論文と完全に一致する」と断定しない。
- コードを実行しないので、再現性は「再現性関連の実装・記録が揃っているか」の静的確認に限る。
- `spec` が draft なら承認済み実装契約として扱わない。
- 実験実施、数値結果、論文の主張を捏造しない。
