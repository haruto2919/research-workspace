---
date: 2026-10-20
project: sequential-video-lora-analysis
source_todo: null
topic: a2-imagenet-one-step-parity-next-steps
status: exploratory
tags: [brainstorm, research, moco, meta, imagenet, parity, gate-a2]
---

# A2 ImageNet実画像1step Meta公式parityへの次の進め方（探索）

## 現在地・根拠

- 研究コード `tamaki-lab/2026_09_ishikawa_sequential-video-lora@dev` の最新確認 `dc502e416aa26ed4e0791767fa5a72f7843e399d` にて、通常MoCoで`meta_einsum`の明示opt-in、config identity、Checkpoint mode照合、軽量monitor、formal comparatorを実装。
- ユーザー実行のCPU限定59/59・MoCo非回帰279/279はPASS。詳細は[実装・CPU検証Evidence](../../../lab/projects/sequential-video-lora-analysis/experiments/2026-10-20-moco-normal-infonce-opt-in-cpu-verification.md)。比較器のCPU formal mockは**実際のMeta公式GPU比較ではない**。
- 先行A1 synthetic、2GPU、K64、16stepのM対S1 bitwise PASSは[2026-10-19 Evidence](../../../lab/projects/sequential-video-lora-analysis/experiments/2026-10-19-meta-moco-a1-infonce-16step-gpu-parity-evidence.md)に限定。
- 正式な通常mode導入spec [2026-10-19](../../../lab/projects/sequential-video-lora-analysis/specs/2026-10-19-meta-infonce-normal-moco-opt-in-spec.md)は`status: implemented`・CPU PASS・GPU NOT_RUN。
- 親 [2026-10-17 Spec A](../../../lab/projects/sequential-video-lora-analysis/specs/2026-10-17-meta-moco-v2-imagenet-ddp-parity-spec.md)はまだ`draft`。A2の公式実画像1step・閾値・trace点は別specで判定run前に固定する必要がある。
- 現行runnerの `conf/moco_imagenet_parity.yaml`：2GPU/NCCL、ResNet50＋MLP、ImageNet `/mnt/NAS-TVS872XT/dataset/ImageNet`、image224、localB32/globalB64、K65536、SGD LR0.0075・momentum0.9・wd1e-4、EMA0.999、T0.2、FP32/AMP off/TF32 off、seed0、`training.max_steps=1`が暫定初期値。`verification.threshold_status=provisional`のままではformal PASS/FAIL不可。
- `scripts/parity/build_moco_parity_fixture.py`は `fixture.kind=imagenet fixture.imagenet_steps=1`の固定2-view（epoch0 DistributedSampler順＋augmentation RNG）と公式初期state、Queue、tensor/hashのartifact生成に対応。fixture生成はCPU／Read-only ImageNetで、生成先は新規directory。
- 実機NAS/GPU/VRAMとMeta cloneのdirtyはGitHubから直接確認できない。実行前のpreflightが必要。

## 研究上の問い

ResNet50＋DDPの通常`meta_einsum`経路は、実ImageNet 224px・K65536・localB32の1stepでも、同一fixtureによってMeta公式とq/k/features、InfoNCE、grad、post-SGD Query、SGD momentum、Key EMA、Queue、BN/DDP状態が正式な事前固定基準内で一致するか。

## 推奨する順序（探索候補・未承認）

1. **読み取りだけのA0環境preflight**：両repo commit/clean、公式clone import provenance、ImageNet train/val配置・class inventory/1サンプルdecode、2GPU index/VRAM free・既存GPUプロセス、Python/torch/CUDA環境を確認。データ構造・batch/K/precisionを自動変更しない。
2. **A2専用spec作成**（ユーザーから依頼があったとき）：A2のfixed実画像fixture、最小1stepの2rank、2view/sample/rank/broadcast、MとS1 independent fresh run、Reference rerun要否、比較tensor集合、full trace step [1]候補、閾値をfixする手続き、preflight VRAM、失敗時停止、fresh output/Comet設定、判定とEvidenceを詳細化。親Spec Aの未承認箇所のApproval Gateを尊重。
3. **評価前閾値固定**：診断/同一Meta fixture rerunにより環境由来の変動を確認し、正式比較の**前**にFP32閾値（A1の1e-5/1e-6は候補に留める）、exact/discrete判定をspecへ固定。診断runを事後的なformal PASSへ転用しない。
4. **短時間fixture生成**：`python -m scripts.parity.build_moco_parity_fixture fixture.kind=imagenet fixture.imagenet_steps=1 --output-dir <fresh-dir>`は既存機能。必ず現行configのlocalB32/K65536/image224/2GPUと同一にする。dataset read-only、生成物はローカル新規path。
5. **別途GPU実行許可後だけ独立A2 run**：Meta MとSelf S1（通常selector `moco.info_nce_impl=meta_einsum`, `diagnostic.info_nce_impl=null`）を同一fixtureから2GPUで別ディレクトリへ起動。必要ならMeta rerunを同一fixtureで行う。1step traceは全full [1]候補。OOM、rank不一致、NaN/NCCL等で停止。
6. **offline comparator**：run終了後に`compare_moco_parity.py --pair-kind formal_implementation`を固定rtol/atol/statusで実行。first mismatch/step/rank/phase/tensorとfloat/exact/bitwiseを別集計。v3 artifactが必要。Meta公式cloneは変更しない。
7. **Evidence保存・次のGate判断**：run report/provenance/fixture manifest/compare reportの結果をResearch Workspace experimentsへ要約記録。raw fixture/trace/checkpointはSSH上、GitHubへ巨大データはpushしない。A2 PASSならA5のmonitor off/light 100〜300step feasibility、次いで別承認のA3/A4とFrozen ViT接続へ。A2 FAILならfirst mismatchを局在化して修正Gateへ戻る。

## 選択肢とtradeoff

- **推奨**：A2固定1step＋小さい実機preflightから。メリットは短い実験でも実画像・大K・BN/DDPまで公式と対照できること。欠点はメモリ/IO負荷がA1より高い。
- **代案**：localB4/K64のImageNet 224pxでまず1stepのfeasibility診断後に正式localB32/K65536へ。トラブルシュートはしやすいが、正式A2のB32/K65536をPASSしたことにはならない。条件変更は別記録・必要な承認。
- **非推奨**：閾値暫定のままMとS1を本番比較して事後に閾値を緩める、公式側だけ異なるaugmentation、OOM時の自動batch/K低減、GPU比較なしで10epoch/ViT移行に数値一致を主張する。

## A2専用specへ引き継ぐ未決事項

- ImageNet NAS構造と1サンプルdecode、2GPU RTX A4500の現在の空きVRAMと利用可否、公式cloneのdirty・import shim。
- 原案local32/K65536/224pxの実機可否。OOMなら変更値、LR/Queue一巡境界も独立条件として再審議。
- A2の正式閾値固定手順、同一Meta参照rerun回数、CPU/FP32 backend flags、比較するfull-step tensor群、作成artifact容量と停止条件。
- Comet OFFを最初の一致比較で推奨するが、記録要件によって追加で判断。独立runのfresh output、追試の独立性。
- 旧A1 v2の再判定を新judgeへ求める必要性（現行v3では古いv2をINVALIDと扱う既知制約）をA2から分離。

本メモは探索記録であり、**A2 specの承認・GPU runの実施許可ではない**。ユーザーがA2のspec作成を明示依頼したらresearch-specへhandoffする。
