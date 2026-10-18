---
project: sequential-video-lora-analysis
spec_type: implementation-and-diagnostic
status: draft
title: Meta MoCo v2 A1 InfoNCE arithmetic self/meta_einsum controlled diagnostic
created: 2026-10-18
last_updated: 2026-10-18
authority: current user request plus Spec A and 2026-10-17 A1 evidence, pending explicit approval
parent_spec: 2026-10-17-meta-moco-v2-imagenet-ddp-parity-spec.md
implementation_repository: tamaki-lab/2026_09_ishikawa_sequential-video-lora
implementation_branch: dev
implementation_commit: c50ab79cbce1738a571daf6e018e31bb870cae24
implementation_dirty_at_spec_review: false
reference_repository: facebookresearch/moco
reference_commit: a419ef282fc9885015674e77f58c59447206a27f
---

# A1追加Spec：InfoNCE演算方式 `self` / `meta_einsum` の1step制御診断（draft）

> **Status: draft（実装不可・未承認）**。本specはA1 Backward勾配差の原因を、InfoNCEの演算方式だけを変える制御実験で切り分ける追加契約である。コード変更、単体テスト実行、GPU run、commit/pushはまだ許可しない。親Spec AのA2/A3/A5、通常学習、Spec Bへ進む許可も含まない。

## 1. 目的と診断仮説

Meta公式MoCo v2と自作MoCoのA1では、step 1の初期state、入力、Shuffle-BN、正規化前後のquery/key特徴が一致する一方、InfoNCE logitsとlossはbitwise非一致で、最初の暫定許容差超過が`after_backward.query_grads.encoder_q.conv1.weight`に現れた。公式・自作それぞれの同一実装再実行は16,000/16,000比較でbitwise一致したため、run間の偶発的な揺らぎだけでは説明しにくい。

本診断の問いは次の一点に限定する。

> 自作側のInfoNCE算術を、現行の`sum/matmul + out-of-place division`からMeta固定commitと同じ`einsum + inplace division`へ変えるだけで、step 1のBackward勾配差とSGD後の差が解消または縮小するか。

これは原因切り分けであり、A1全体の数値一致、長期学習の同等性、表現品質を証明する実験ではない。

## 2. 現行Evidenceとコード基準

### 2.1 観測済みEvidence

- Meta vs Meta：16step、full trace step 1/8/9/16、16,000/16,000項目bitwise一致。
- Self vs Self：同条件で16,000/16,000項目bitwise一致。
- Meta vs Self：9,398/16,000 float tensor比較が暫定許容差外。最初の差はstep 1、rank 0、`after_backward/query_grads/encoder_q.conv1.weight`。
- step 1の`conv1.weight.grad`はmax absolute error `1.5497207641601562e-5`、暫定基準超過852/9,408要素。rank 0/1で同じ値。
- 共通q/k/Queueからの局所比較では、logits、loss、`dL/dq`、`dL/dz_q`はbitwise非一致だが、いずれも暫定`rtol=1e-5, atol=1e-6`内。深いBackwardでの伝播・増幅は未確定。

これらは2026-10-17のユーザー提供artifactに基づく。今回のspec作成時にはGPUで再測定していない。

### 2.2 確認した現行コード

- 自作repository：`dev@c50ab79cbce1738a571daf6e018e31bb870cae24`、確認時clean、`origin/dev`と一致。
- 自作`self_supervised/moco/standalone_moco.py::moco_logits`：

```python
l_pos = (q * k).sum(dim=1, keepdim=True)
l_neg = q @ queue
logits = torch.cat([l_pos, l_neg], dim=1) / temperature
```

- Meta固定commit `moco/builder.py`：

```python
l_pos = torch.einsum("nc,nc->n", [q, k]).unsqueeze(-1)
l_neg = torch.einsum("nc,ck->nk", [q, self.queue.clone().detach()])
logits = torch.cat([l_pos, l_neg], dim=1)
logits /= self.T
```

- `DistributedMoCo`は現状、通常の`moco_logits`を直接呼ぶ。A1 runnerの`SelfAdapter`は`DistributedMoCo(ImageNetResNet50(), ...)`を構築する。
- A1 canonical fixture、rank別入力hash、Shuffle-BN permutation、phase trace、DDP後grad、SGD後parameter/momentumを保存する既存harnessと比較器は利用可能。

## 3. Decision Contract

### 3.1 採用する変更

1. A1 self側だけに、診断設定`diagnostic.info_nce_impl`を追加する。許容値は厳密に`self`と`meta_einsum`の2つとし、既定値は`self`とする。
2. `self`は現行`moco_logits`をそのまま呼び、既存演算順を変えない。
3. `meta_einsum`はMeta固定commitの上記4行と同じ演算子、添字、`unsqueeze(-1)`、連結順、in-place temperature divisionを使う。Queueは既存forwardが作るenqueue前の`clone().detach()` snapshotを入力とする。
4. 切替は自作`DistributedMoCo`のInfoNCE logits生成箇所だけに作用する。CrossEntropy、targets、forward順序、EMA、Shuffle-BN、all-gather、Queue enqueue、DDP、Optimizer、trace hookは共通のままとする。
5. 診断結果には選択したmodeをresolved config、run report、provenance、comparison reportへ必ず保存する。
6. 同じcanonical fixtureを使う1step self対self制御比較で、mode以外の比較条件が同一であることを機械検査する。

### 3.2 明示的な対象外

- `standalone_moco.py`の通常`moco_logits`の式、signature、既定挙動を変更しない。
- `conf/moco_standalone.yaml`、CIFAR-10 smoke、ResNet18、通常のsingle-device MoCoを変更しない。
- Meta公式clone、`moco/builder.py`、公式Adapterのforward/lossを変更しない。
- A2/A3/A4/A5、Queue一巡、ImageNet実画像bench、10epoch学習、閾値の正式化を行わない。
- DDP bucket、ResNet backward、`dL/dz_q`注入VJPなど、InfoNCE以外の診断を同時に追加しない。
- 暫定許容差を今回の結果に合わせて変更しない。

## 4. 最小実装設計

### 4.1 model側

変更対象は`self_supervised/moco/ddp_moco.py`の`DistributedMoCo`に限定する。

- `DistributedMoCo`へ非永続の診断modeを渡せるようにする。既定`self`では、現在importしている`standalone_moco.moco_logits`を同じ引数で呼ぶ。
- `meta_einsum`用の小さなpure helperをDDP側へ置き、Meta式をそのまま表現する。通常/CIFAR共通関数へ汎用mode分岐を持ち込まない。
- mode文字列をparameter/bufferとして`state_dict`へ追加しない。既存checkpoint keyとstate mappingを変えない。
- 不正なmodeはmodel構築時に`ValueError`とし、forwardやEMAなどの状態変更前に失敗させる。

この設計では`DistributedMoCo`のdefault pathは従来と同じ関数を通り、CIFARの`StandaloneMoCo`はsource・挙動とも変更しない。

### 4.2 A1 runner/config側

`conf/moco_imagenet_parity.yaml`に次を追加する。

```yaml
diagnostic:
  info_nce_impl: self  # self | meta_einsum; meta_einsum is A1 self-side one-step only
```

`scripts/parity/moco_imagenet_parity.py`はSelfAdapter構築時だけこの値を`DistributedMoCo`へ渡す。`meta_einsum`は次の全条件を満たす場合だけ許可する。

- `implementation=self`
- `run_kind=parity`
- `gate=A1`
- `training.max_steps=1`
- `trace.full_steps=[1]`
- `fixture.kind=synthetic`
- `moco.queue_size=64`
- `training.local_batch_size=4`、`distributed.world_size=2`
- `data.image_size=64`
- `training.resume=null`、`training.checkpoint_steps=[]`
- `logging.comet_enabled=false`

条件外では明示的な理由付きで開始前に拒否する。`self`は既存A1/A2/A3/A4/A5の既定挙動を維持し、従来の16step A1も妨げない。

modeはdiagnostic用なので、既存checkpointの`config_identity_sha256`には追加しない。代わりにcheckpoint/resume自体を`meta_einsum`で禁止する。run report、provenance、verification summaryには明示的に記録する。

### 4.3 比較器側

既存のMeta vs Self比較は`pair_kind=implementation`をそのまま使い、次の2報告を別々に作る。

- Meta vs Self(`self`)
- Meta vs Self(`meta_einsum`)

加えて、Self(`self`) vs Self(`meta_einsum`)の制御比較用pair kind `a1_info_nce`を追加する。このpair kindは「数値一致を期待するrerun」ではなく、「比較契約が守られ、mode変更でどの値から差が出るか」を記録する診断である。

- 両runが`implementation=self`、A1、1step、full step 1であること。
- fixture SHA、初期state hash、入力hash、shuffle index/seed、環境、backend flags、GPU/rank対応、model/MoCo/Optimizer設定が一致すること。
- resolved configの意味的な差が`diagnostic.info_nce_impl: self -> meta_einsum`だけであること。
- `before_forward`からq/k/queue snapshotまでがbitwise一致すること。差の初出はlogits生成以降でなければ診断をINVALIDとする。
- 診断pair自体はfloat差の有無を理由に通常parityのPASS/FAILを付けない。`contract_valid: true/false`と差分統計を出し、因果判定は第7節の3比較で行う。

## 5. 固定する比較条件

Self(`self`)とSelf(`meta_einsum`)はfresh process・fresh output directoryで実行し、次を完全に固定する。

| 項目 | 固定条件 |
|---|---|
| repository | 同一の実装commit、開始時clean。実装後commit SHAを全runに記録 |
| reference | `facebookresearch/moco@a419ef282fc9885015674e77f58c59447206a27f`、clone clean |
| fixture | 既存A1 synthetic fixtureを読み取り専用で共有。同じfixture SHA/state SHA |
| Model | random-init ResNet50、MLP 2048→2048→128、同じ全parameter/BN buffer/Queue/ptr |
| Input | step 1のrank別`im_q`/`im_k` tensor hashが同じ |
| MoCo | K64、m0.999、T0.2、enqueue前Queue、global batch8 |
| DDP | world size 2、NCCL、1 process/GPU、同じrank/GPU割当、`broadcast_buffers=true`、plain BN、Shuffle-BN permutation完全一致 |
| Optimizer | fresh SGD、LR0.0075、momentum0.9、weight decay1e-4、nesterov=false、同じparameter順。開始時momentum bufferなし |
| Precision | FP32、AMPなし、TF32 off、cudnn deterministic true、benchmark false、同一torch/CUDA/cuDNN/NCCL |
| Execution | epoch 0、1 optimizer step、resumeなし、checkpointなし、Comet OFF、full trace step 1 |
| 唯一の差 | `diagnostic.info_nce_impl`のみ |

Metaとの比較では、親Spec Aどおり公式Optimizerがgradientを持たないKey parameterもparam groupに含む点を維持する。公式をSelfへ合わせる変更はしない。今回「Optimizer固定」と呼ぶのは、各実装内の既存Optimizer契約を変えず、2つのSelf診断run間で完全に同一にすることを指す。

## 6. 実装後に必須の単体・CPUテスト

GPU実行前に、少なくとも次をすべてPASSさせる。

1. **式の単体テスト**
   - 固定float32 q/k/Queue/Tに対し、`meta_einsum`の`l_pos`、`l_neg`、logitsがMeta固定commitのliteral式とbitwise一致する。
   - 同じliteral式から得るCrossEntropyと`autograd.grad(loss, q)`にも一致する。
   - shape、dtype、device、positive列位置、temperature適用後の戻り値契約が`self`と同じ。
2. **default非回帰**
   - mode省略時と`info_nce_impl=self`明示時で、固定state/inputのlogits、loss、全Query grad、EMA後Key、Queue、SGD後Queryがbitwise一致する。
   - `StandaloneMoCo`とCIFARのpublic API/state dict/configに新しい必須引数やkeyを追加しない。
3. **DDP数学テスト**
   - 既存gloo 2-rank・float64参照テストを`self`で維持する。
   - `meta_einsum`でも、公式式を使う独立参照に対してq/k/logits/loss、DDP平均grad、EMA、Queue/ptr、SGD更新が既存tight tolerance内。
4. **入力・設定拒否テスト**
   - 不正modeをmodel構築時に拒否する。
   - `meta_einsum`をMeta側、bench、A2以降、2step以上、resume/checkpoint、Comet ON、A1固定K/B/image size以外で使うとrun開始前に拒否する。
5. **artifact/comparatorテスト**
   - modeがresolved config、provenance、run report、summaryへ残る。
   - `a1_info_nce`がmode以外の差を1つでも検出すれば`contract_valid=false`とする。
   - modeだけが異なる場合は比較を実行し、差の初出phase/tensorと勾配統計を保存する。
6. **既存test非回帰**
   - `test/model/test_moco_standalone.py`
   - `test/model/test_moco_cifar10_reporting.py`
   - `test/model/test_moco_ddp.py`
   - `test/model/test_moco_parity_harness.py`

既存testの失敗を新規変更と無関係として扱う場合は、同一環境・変更前commitでの再現Evidenceを必要とする。今回のscopeではLoRA、ViT、dataset系の既知環境依存failを修正しない。

## 7. 実装後の1step GPU検証

GPU runはコード実装・CPUテストとは別に明示許可を得てから行う。各runは2×GPU、NCCL、A1 synthetic fixture、1step、Comet OFF、別output directoryとし、既存artifactを上書きしない。

### 7.1 必須run

| ID | 実装 | mode | 用途 |
|---|---|---|---|
| M | Meta reference | n/a | 不変の参照。既存run再利用はfixture SHA・環境・backend flags・Meta commitが一致する場合のみ |
| S0 | Self | `self` | 新commit上の現行算術baseline |
| S1 | Self | `meta_einsum` | 算術だけを変えたtreatment |

比較は`M vs S0`、`M vs S1`、`S0 vs S1`の3つを生成する。S0/S1は同じfixture、同じ実装commit、同じconfigでmodeだけを変える。

### 7.2 再現性確認

InfoNCEが原因という結論を出す前に、S0とS1をそれぞれ同じ条件・別directoryでもう1回実行し、同mode内のstep 1 full traceがbitwise一致することを確認する。非一致なら原因判定を`INCONCLUSIVE`とし、CUDA/DDP非決定性を先に調べる。Meta Mは既存のMeta vs Meta 16,000/16,000 bitwise Evidenceを利用できるが、環境またはfixtureが異なる場合はMも再実行する。

### 7.3 run有効性

次のいずれかがあれば勾配差を判定せずINVALID/BLOCKEDとする。

- run失敗、NaN/Inf、within-run必須check不合格。
- fixture/state/input/shuffle/rank/環境/backend/Optimizer条件の不一致。
- S0/S1でmode以外のresolved condition差。
- S0/S1で`before_forward`、`after_query`のz_q/q、`after_key_forward`のz_k/k、enqueue前Queueがbitwise不一致。
- Meta cloneのdirty、固定commit不一致、またはSelf repositoryのrun間commit/dirty不一致。

## 8. 勾配差の測定値と判定条件

### 8.1 必須測定

step 1、rank 0/1の全canonical Query parameterについて、Metaをreferenceとして次を保存する。

- `max_abs_error`
- 暫定規則 `|self - meta| <= 1e-6 + 1e-5 * |meta|` の超過scalar数と全要素数
- `relative_l2 = ||g_self - g_meta||_2 / ||g_meta||_2`。分母0では両者0なら0、それ以外はinfinityとする
- tensorごとのgrad norm
- 全Query gradを論理的に連結したglobal relative L2、総超過scalar数、最大absolute error
- `after_sgd`のQuery parameterとmomentum bufferについて同じ比較統計
- 最初に暫定閾値を超えるphase/group/tensor

暫定`rtol/atol`は診断用の同じ物差しとして使い、親Spec Aの正式Gateへ昇格させない。

### 8.2 仮説判定

1. **SUPPORTED AS SUFFICIENT AT STEP 1**
   - S0はMetaに対してQuery gradに1件以上の暫定閾値超過があり、既知の問題を再現する。
   - S1はMetaに対して、全rank・全Query grad、SGD後Query parameter、momentum bufferの暫定閾値超過が0。
   - S1のexact/discrete比較、within-run checksは全PASSで、最初のfailureがない。
   - S0/S1の同mode再実行がbitwise一致し、制御比較契約がvalid。

   この場合、「A1 step 1ではInfoNCE算術の違いだけで観測された勾配差を許容差内まで解消できた」と主張できる。16step以降や一般的なMoCo実装の原因まで一般化しない。

2. **SUPPORTED AS A PARTIAL CONTRIBUTOR**
   - S1にも閾値超過が残るが、S0よりも、全Query gradの総超過scalar数、global relative L2、最大absolute errorの3指標がすべて小さい。
   - rank 0/1で同じ方向の改善があり、同mode再実行がbitwise一致する。

   この場合、InfoNCE算術は寄与候補だが十分原因ではない。残差の初出を報告し、DDP前後またはencoder backward診断は別spec/承認へ回す。事後的な倍率閾値を作って「ほぼ解決」とはしない。

3. **NOT SUPPORTED**
   - S1とS0のQuery gradがbitwise同一、または上記3指標のいずれも改善しない／悪化する。

   この場合、今回のInfoNCE演算方式を主要原因とする仮説は支持されない。通常経路の式を変更しない。

4. **INCONCLUSIVE**
   - 3指標が混在する、rank間で改善方向が異なる、run再現性がない、または第7.3節の有効性条件を満たさない。

いずれの結論でも、`rtol/atol`の正式固定やA1 PASSへの変更は行わない。S1が良好でも、本番defaultを`meta_einsum`へ変更するには結果レビューと別の明示承認が必要である。

## 9. Artifactと報告

- 既存A1 run artifact一式を各runで保存し、上書きしない。
- comparison reportにreference/candidateのmode、fixture SHA、Self/Meta commit、dirty、環境、3つの集約勾配指標、tensor別上位差、判定カテゴリを記録する。
- 最低限、`encoder_q.fc.2.weight`、最初の閾値超過tensor、`encoder_q.layer1.0.conv1.weight`、`encoder_q.conv1.weight`を表に含める。ただし特定layerだけで全体判定しない。
- 実行後の観測値とEvidence境界はResearch Workspaceの`experiments/`へ別途保存する。推測値を結果として記録しない。
- 巨大trace、checkpoint、Queue dumpはResearch Workspaceへコピーまたはpushしない。相対path、hash、要約だけを保存する。

## 10. 互換性と停止条件

- `self`既定時の既存A1/A2/A3/A4/A5、通常MoCo、CIFAR-10、Meta公式実装の挙動を変えない。
- `StandaloneMoCo`のstate dict、既存checkpoint load、CIFAR config/CLIにmigrationを要求しない。
- Meta公式sourceを変更した場合、そのrunは本診断の参照として無効。
- S1が一致しても自動でdefaultを変更せず、S0を削除せず、長時間runへ進まない。
- INVALID/BLOCKED、NaN/Inf、NCCL hang、unexpected state mutation、mode以外の条件差を検出したら停止し、原因と最後に確認できたphaseを残す。

## 11. Ambiguity Gateとreadiness

### Blocking

- **なし**。診断mode名、式、適用範囲、固定条件、必要test、1step run、判定カテゴリは現在依頼と現行コードから実装可能な粒度まで定義した。

### Non-blocking

- helper/constantの局所名、test function名、JSON内の配置は既存styleに合わせてよい。ただし外部に記録するmode値`self` / `meta_einsum`は変更しない。
- Mの既存runを再利用できるかは実行時provenanceで機械判定する。一致しなければfresh Mを使うため、研究判断は不要。

### Spec readiness

内容上は承認レビュー可能。現在は`draft`であり、明示承認されるまで実装不可。承認後もGPU runは別許可とする。

## 12. 関連文書

- 親Spec A：[2026-10-17-meta-moco-v2-imagenet-ddp-parity-spec.md](2026-10-17-meta-moco-v2-imagenet-ddp-parity-spec.md)
- A1勾配差解析：[../code-analysis/2026-10-17-meta-moco-a1-gradient-divergence-explain.md](../code-analysis/2026-10-17-meta-moco-a1-gradient-divergence-explain.md)
- Spec A実装・CPU検証：[../experiments/2026-10-17-meta-moco-v2-imagenet-ddp-parity-implementation-verification.md](../experiments/2026-10-17-meta-moco-v2-imagenet-ddp-parity-implementation-verification.md)
- Meta固定builder：`facebookresearch/moco@a419ef282fc9885015674e77f58c59447206a27f:moco/builder.py`
- 自作基準：`tamaki-lab/2026_09_ishikawa_sequential-video-lora@c50ab79cbce1738a571daf6e018e31bb870cae24`

## 13. Implementation Handoff（未承認・実行不可）

- approved spec: 未承認。本ファイルを承認後に使用
- 実装目的: A1 self側のInfoNCE算術だけを`self` / `meta_einsum`で切替え、step 1 Backward差への因果寄与を診断
- 基準repository/commit: `tamaki-lab/2026_09_ishikawa_sequential-video-lora@c50ab79cbce1738a571daf6e018e31bb870cae24`、実装開始時にHEAD/dirtyを再確認
- 変更scope: `ddp_moco.py`、A1 parity config/runner/comparator、対応testのみ
- 対象外・維持条件: Standalone/CIFAR/Meta公式、A2以降、閾値変更、通常default変更、長時間run
- success criteria: 第6節のtest PASS、第7節の有効な1step比較、第8節のいずれかの判定をEvidence付きで報告
- 許可されている短時間検証: 現時点ではなし。実装承認後にCPU unit/non-regression test、GPUはさらに別許可
- 長時間runの許可状態: 不許可
- 未検証予定: A2/A3/A5、16step治療効果、正式閾値、表現品質、10epoch
