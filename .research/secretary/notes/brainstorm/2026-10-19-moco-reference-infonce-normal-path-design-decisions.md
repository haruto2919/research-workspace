---
date: 2026-10-19
project: sequential-video-lora-analysis
source_todo: null
topic: moco-reference-infonce-normal-path-design-decisions
status: exploratory
tags: [brainstorm, research, moco, infonce, reference-parity, backward-compatibility, checkpoint, spec-handoff]
---

# A1一致済みInfoNCEを通常MoCo経路へ安全に導入するspec作成前の論点

## 位置づけ・対象

ユーザーの依頼：「Meta公式と一致したInfoNCE演算を通常MoCoでも使用できるようにするためのspecについて、決めるべきことを整理したい」。本書は2026-10-19時点の**探索メモ**であり、B1〜B12はまだ採用されていない推奨案。実装可能spec、コード変更、GPU実験実行・研究コードcommit/pushの承認ではない。

既存の研究目的：オンライン動画からLoRAがどのような情報を獲得したかを解析する。その前段として単体MoCoの参照との数値整合性を確かめ、Frozen ViT・Sequential Loader・LoRAの統合工程を一因子ずつ進める。

## 確認済み事実（GitHub、Authority・Evidence）

- Research Workspace: `haruto2919/research-workspace@main`。研究コード: `tamaki-lab/2026_09_ishikawa_sequential-video-lora@dev`、今回参照したHEAD `1ce9badcbae3e2df08fa400944e4c385785063f9`。研究サーバ上の現時点のローカルdirty/worktree・GPU状態は直接未確認。
- `experiments/2026-10-19-meta-moco-a1-infonce-16step-gpu-parity-evidence.md`：A1 synthetic、K64、B4×2、2GPU、16step、fixed `rtol=1e-5,atol=1e-6`、M対S1 `PASS` かつ比較エントリ16,256/16,256 bitwise、M/S0/S1各rerun bitwise。負対照S0対Mはdiff蓄積。raw GPUログは研究サーバ内で、記録はユーザー提供実行結果に基づく。
- `self_supervised/moco/ddp_moco.py`の`DistributedMoCo`には`info_nce_impl`が既実装。通常defaultは`self`、別候補`meta_einsum`は公式と同じeinsum正負積・inplace division。選択属性は**state_dictに含まれない**。
- `self_supervised/moco/standalone_moco.py::moco_logits`は従来のsum/matmul形式で、Standalone/CIFARにも使用されている。今回これを変更する必要は確認できない。
- `scripts/parity/moco_imagenet_parity.py::validate_info_nce_diagnostic`は`meta_einsum`をA1 syntheticの1stepまたは16stepだけ許可。`load_fixture`も非syntheticのmeta_einsumを拒否。`SelfAdapter`は既存modeをconstructorへ伝搬できる。
- 同runnerの`config_identity()`は`diagnostic.info_nce_impl`を現状**含まない**。自己checkpointにmode値が入らないことと併せて、S0 checkpointをS1設定で再開する混同への防止策が必要。
- `compare_moco_parity.py`の既存`implementation` preconditionはMeta対Selfの実装種類を確認するが、それだけで通常A2/A3などでSelfが`meta_einsum`を選択したことまでは保証しない。run provenance/reportにはmodeを記録済み。新しいformal comparator/judge契約ではmode検査を検討。
- `conf/moco_imagenet_parity.yaml`は既定`diagnostic.info_nce_impl: self`、`verification.threshold_status: provisional`、run_kind parity/bench、gate A2等。親のSpec A（2026-10-17）は**draft**。個別A1・16step specのみapproved。

## 中心の問い

A1の固定fixtureと制御条件でbitwise一致した`meta_einsum`演算を、通常のResNet50/DDP MoCo実験へ明示選択可能にしつつ、旧実験の意味・resume安全性・比較のAuthorityを破壊しない最小の設計は何か。

## 12の要決定項目と推奨（未採用）

| ID | 決めること | 推奨候補 | 判断根拠・受入条件 |
|---|---|---|---|
| B1 | 通常MoCoの範囲 | **第一段階はResNet50＋2GPU DDP parity/benchmark runnerのみ**。Standalone/CIFAR・既存動画Streaming MoCoはscope外 | A1で検証したクラスと計算経路に留め、動画/ViTで同じコードが使われると推測しない |
| B2 | 演算方式の既定値 | 既定`self`維持、`meta_einsum`は明示opt-in。`DistributedMoCo`のdefault、既存CIFAR baseline、Meta cloneを変えない | 既存の数値trace・checkpoint・結果の意味を保持。将来のdefault移行は別Gate |
| B3 | 正式config selector | **通常実験用modeと既存A1診断modeを明示的に区別**。例：`moco.info_nce_impl`などcanonical selectorを追加し、旧`diagnostic.info_nce_impl`はA1互換で維持。両方指定の優先順位/衝突拒否を事前規定 | `diagnostic`のまま本番解禁は最小diffだが意味が曖昧。移行設計が複雑なら暫定単一selector案との比較が必要 |
| B4 | 使用許可Gate | 当面A2（synthetic/固定ImageNet実画像1step）、A3（K65536＋長期Queue）、A4（checkpoint/resume）、A5（bench）の設計を許可対象候補とする。ただしCPU設定検証とGPU実行承認は別 | A1の限定profileは変更せず、A2/A3/A4/A5を単純に無制限解禁しない。10epoch Spec BとFrozen ViTは別spec |
| B5 | Checkpoint/Resume契約 | modeをresolved config/provenance/checkpoint sidecar/config_identityに保存・照合。mode不一致のresumeは禁止。旧artifactのmode不明は無条件にselfと推定せずlegacy migration明示時のみ | state_dictにmodeが無い。現在のconfig_identityもmodeを除外。既存checkpoint keyの不要な破壊を回避 |
| B6 | 比較・judgeが検証すること | Meta公式Mと自作S1のformal比ではSelf側のeffective modeが`meta_einsum`であることを必須検査。M対S0は負対照として別欄にし、run/rerunでmode一致必須。first mismatchをstep/rank/phase/tensorで記録 | 実装種類がselfであるだけではMeta算術使用の証拠にならない |
| B7 | 誤差基準 | A1 fixed `rtol=1e-5, atol=1e-6`を**他Gateへ自動昇格しない**。A2以降の正式閾値は公式側再演、精度・環境、評価対象を見て正式判定run前に別specで固定。bitwiseを別指標、exact/discreteを区別 | Synthetic小Kのbitwise一致は他解像度・大K・実画像の普遍保証でない |
| B8 | 実画像/実機preflight | ImageNet root read-only inventory、固定2view前処理・rank/sampler・augmentation hash/初期state/Queue/permutation、2GPU/VRAMとbatch・K測定をGateとする。OOMでbatch・Kを自動変更しない | A4500各19GB級でlocal32/K65536/224pxは未実測。fixture.kind=imagenetのmeta_einsum拒否も選択範囲に合わせて変更対象 |
| B9 | 後方互換とschema | 既存A1 1/16step profile、Standalone/CIFAR、state_dict/checkpoint、config default、旧run v1/v2を維持。metadata拡張はschema/version/reader互換・legacy判定を事前定義し旧artifactを改変しない | 細かい設定解禁がA1の信頼性や既存reportを壊さない |
| B10 | Verification tests | mode伝搬、A1旧許可/拒否、A2/A3/A4/A5新許可/拒否、real fixture、wrong-mode resume、旧checkpoint、comparison role・threshold・provenance、CPU/gloo DDP、CIFAR非回帰、201 CPU testsを基準に差分回帰を定義 | 新しいGPUを動かす前に設定とモデル入口の正当性を立証 |
| B11 | 実行順序・成功基準 | 設定/検証器のengineering gate→CPU tests→短いA2 fixed real fixtureで公式比較→A5 VRAM/速度preflight→許可後A3長期／A4 resume。A1結果は既存Evidenceを保護、必要時は限定再演。受入条件をGateごとに分離 | 大規模Queue1024/1025stepや10epochはコストが高く、1stepの結論だけから同じPASSを断定しない |
| B12 | 実行Authorityと停止・保存 | spec作成・実装・commit/push・GPU A2/A3/A4/A5・10epochは別承認。fresh run dirs、hash/commit/dirty、Cometの明示設定、nan/OOM/hang/contract mismatch停止、既存artifact無上書き | Parent Spec Aがdraftであり、今回の設計採用だけで全Gateの実験は許可されない |

## 重要な対立案・棄却寄り

- A：`StandaloneMoCo.moco_logits`を一括で`einsum`へ変更する案。全依存経路を変更し、過去のCIFAR/動画基準が崩れるため**非推奨**。
- B：`DistributedMoCo(info_nce_impl='meta_einsum')`を即座にdefault変更。新規学習に明快だが、無指定configの意味が変わり既存A1負対照・過去run再現性に影響。**今は非推奨**。
- C：既存`diagnostic.info_nce_impl`をA2〜A5へ広げるだけ。変更量最小だが正式な学習で`diagnostic`が誤解を招き、checkpoint identityを修正しないと危険。先にcontract化できる場合の代案。
- D：正式mode selectorの追加＋A1互換を保持した明示opt-in。差分は多少増えるが意味・安全性・将来のViT移行を分離しやすく**最有力候補**。

## 今回の提案で誤解してはいけない境界

- 学習に使う`meta_einsum`と、Meta公式とのparityが証明済みという事実は同じではない。証拠はA1 synthetic K64 16stepの全比較対象についてのみ。
- A2実画像fixture/224px、A3 K65536の1024→1025境界、A4 checkpoint exact resume、A5 VRAM/throughput、Spec B 10epoch/validation/linear probeは**未検証**。
- 通常MoCoと動画Streaming MoCoのクラス/呼び出し経路が同一とは限らない。後者への適用はコード調査と別specを要する。
- A1のFP32 fixed thresholdを、A2以降の正式な許容差として無断転用しない。

## research-specへ引き継ぐ候補

- 目的：A1 bitwise一致を達成した公式InfoNCE算術を、ResNet50 2GPU DDPの通常実験パスでopt-in可能にする。
- SSOT Evidence：2026-10-19 A1 16step GPU parity evidence。Authorityは`2026-10-19`承認済みA1 16step spec、親Spec Aはdraft。
- 保存対象：研究workspaceの対象project `specs/`（ユーザーがspec作成を明示依頼した場合のみ）。
- コード候補：`conf/moco_imagenet_parity.yaml`、`scripts/parity/moco_imagenet_parity.py`、`scripts/parity/compare_moco_parity.py`、checkpoint/config identity、必要なら`ddp_moco.py`の入口。既存公式式・モデル本体の書換えは原則不要。
- 必須受入：default無変更、旧A1/Standalone/CIFAR回帰なし、効果的mode記録とmode不一致resume拒否、実画像/preflightのfail-fast、正式comparisonが本当にMeta式を使ったことを検証。
- 未決：B1〜B12ユーザー採否、config selectorの衝突ルール、他Gateの誤差閾値とGPU実行計画、legacy checkpoint移行の要否。

## 関連

- [直前ロードマップ](./2026-10-19-post-a1-16step-moco-parity-next-priorities.md)
- [A1 16step GPU Evidence](../../../lab/projects/sequential-video-lora-analysis/experiments/2026-10-19-meta-moco-a1-infonce-16step-gpu-parity-evidence.md)
- [A1 approved spec](../../../lab/projects/sequential-video-lora-analysis/specs/2026-10-18-meta-moco-a1-infonce-16step-parity-spec.md)
- [Parent Spec A draft](../../../lab/projects/sequential-video-lora-analysis/specs/2026-10-17-meta-moco-v2-imagenet-ddp-parity-spec.md)
