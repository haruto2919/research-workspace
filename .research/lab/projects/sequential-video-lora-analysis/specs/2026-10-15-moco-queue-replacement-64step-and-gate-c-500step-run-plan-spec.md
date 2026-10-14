---
project: sequential-video-lora-analysis
spec_type: experiment
status: approved
created: 2026-10-15
last_updated: 2026-10-15
title: CIFAR-10 standalone MoCo Queue完全置換前後64step診断とGate C 500step
authority: user-requested staged experiment execution; existing approved Stage1 contract unchanged
implementation_repository: tamaki-lab/2026_09_ishikawa_sequential-video-lora
implementation_branch: dev
reviewed_code_commit: 8939d7fca7d1cc3367b437c5f6adb48dd8e86f4a
depends_on: 2026-10-15-standalone-moco-v2-cifar10-mechanics-verification-spec.md
execution_status: not-run-by-ChatGPT
---

# Standalone MoCo：Queue置換64step診断 → Gate C 500step実験計画（approved）

## 1. Authority・目的・現状

ユーザーは、2026-10-15のStandalone MoCo実CIFAR-10 20step Gate A/B/D PASSを受け、①Queue置換後までの40–64step診断、②Gate C 500step実験、③その後ViT＋MoCoへの移行、の順に進むことを明示。Stage1の[approved spec](./2026-10-15-standalone-moco-v2-cifar10-mechanics-verification-spec.md)の構造・データ・Random Queue・Key EMA・enqueue方式は変更しない。

- **今回のexperiment planの対象は①と②。** 40–64の範囲から、Queue置換step32の前後を十分残す**64step（fresh）**を選ぶ。これは独立したcustom診断であり、Gate B/Cの改訂・達成ではない。
- **Gate Cは、すでに承認済みの`max_steps=500,scheduler=cosine`を採用して、別のfresh runで実施する。** 64stepのcheckpointからresumeしない（固定LRからcosineへ途中切替しない）。
- **③Frozen ViTへの移行は本specで実装しない**。別の[ViTブリッジdraft](./2026-10-15-frozen-vit-moco-cifar10-stage2-integration-spec.md)で設計・承認・実装Gateを管理。
- 直近実Evidence：[20step Gate A/B/D検証](../experiments/2026-10-15-standalone-moco-cifar10-gate-abd-console-evidence.md)。`dev` commit `8939d7f...`, dirty=false, CIFAR10 train、CPU、20step B PASS、MoCo pytest42 PASS、LoRA回帰32 PASS。Loss2.527→6.388で上昇、Negative mean cosine -0.004→0.523、Query gradientは発散していない。
- **実行境界**：ChatGPTからSSH serverへはアクセスできず、現時点で研究サーバのrunは未実施。このspecの承認と実際の実験完了を混同しない。ユーザーは実験実施を望んでおり、提示する既存entrypointコマンドは研究サーバ側で実行する必要がある。GPU使用は実サーバでhealth check後に明示選択。GitHubへのresearch code変更・pushは行わない。

## 2. 基準と固定条件

実行直前に`git branch --show-current`,`git rev-parse HEAD`,`git status --porcelain --untracked-files=all`を確認。研究結果の比較基準は`dev@8939d7fca7d1cc3367b437c5f6adb48dd8e86f4a`のclean状態。違っていれば改変せず差分を監査する。実行runの`report.json.provenance.git`で追跡。

| 条件 | 20step既存run | 64step追加診断 | Gate C 500step |
| --- | --- | --- | --- |
| Dataset | CIFAR10 train、50,000、labels未使用 | **同じroot/train** | **同じ** |
| Backbone | ResNet18-CIFAR、random init | **同じ** | **同じ** |
| Two-view aug | MoCo v2 32×32適応 | **同じ** | **同じ** |
| Queue | K1024、randn L2 init、全Queue negatives | **同じ** | **同じ** |
| Momentum/T | 0.999/0.2 | **同じ** | **同じ** |
| SGD | LR.00375、momentum.9、WD.0001 | **同じ** | **同じ初期値** |
| Batch/seed | 32 / 0 | **32 / 0** | **32 / 0** |
| Fresh / resume | fresh | **fresh、resume null** | **fresh、resume null** |
| Step | 20 | **64** | **500** |
| Scheduler | none（固定LR） | **none（固定LR）** | **cosine（承認済みGate C）** |
| device | CPU | CPU推奨（20stepと揃える） | CPUまたはhealth確認済みCUDA、deviceをreport記載 |
| run kind | `gate_b_smoke20` | **`custom`が正常** | **`gate_c_pilot500`** |
| Gate | B PASS | B/CともNOT_RUNが正常 | Cのengineering checks判定が得られる |

`64*32=2048`画像、`500*32=16000`画像でCIFAR train 50,000以内。バッチ間のsample indexは同じseedのshuffle順の先頭から再使用なく消費。異なる新規runなので**同一sample/初期seedの計算を比較**できるが、500stepは学習率履歴が違い途中からパラメータ値が変わり得る。step1～20のCPU floating差は同じライブラリ・deviceなら比較可能だが、GPUとのbitwise一致を要求しない。

## 3. 実行コマンド（リモート研究サーバ）

```bash
cd /mnt/HDD12TB-1/ishikawa/2026_09_ishikawa_sequential-video-lora
git branch --show-current
git rev-parse HEAD
git status --short

# 64step: Queueが全置換されるstep32を跨いで観測する。CPU/SGD固定LR。
CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=4 \
python -m scripts.smoke.smoke_moco_cifar10 \
  data.root=/mnt/NAS-TVS872XT/dataset/CIFAR10 \
  data.download=false training.device=cpu \
  training.max_steps=64 training.scheduler=none

# 上記の結果・プロセス終了状態・reportを確認してから、別fresh runとしてGate C。
# CPUで実行するとき（GPU未確認の環境でも実行可能）:
CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=4 \
python -m scripts.smoke.smoke_moco_cifar10 \
  data.root=/mnt/NAS-TVS872XT/dataset/CIFAR10 \
  data.download=false training.device=cpu \
  training.max_steps=500 training.scheduler=cosine

# GPUで実行したい場合は上記CPUコマンドの代わりに、
# nvidia-smi等で指定deviceの健全性確認後、
# CUDA_VISIBLE_DEVICES=<実際に使うGPU> と training.device=cuda を明示する。
```

出力は既存Hydra規則によりそれぞれ一意の`log/moco_standalone/YYYY-MM-DD/HH-MM-SS-ffffff/`。既存20step成果物は上書きしない。64stepが終わる前に500stepを同時起動しない。CPUやGPUのrun完了はこの文書の記載だけでは証明しない。

## 4. 成功・異常判定

### 4.1 64step custom診断（主要Gate）

- `report.status=completed`, `report.run_kind=custom`, `optimizer.completed_steps=64`, 各`checks`がPASS、checkpoint roundtrip=true。run内`gates.B/C=NOT_RUN`はcustom runとして正しい。
- **step32直後**：Queue `real_key_count=1024`, `random_remaining_count=0`, `queue_ptr=0`。
- **step64直後**：Queue `real_key_count=1024`, `random_remaining_count=0`, `queue_ptr=0`。
- 1–32 stepを「Random key置換期」、**33–64** stepを「Queue全件が実画像keyになった時期」と区別して、Loss、Positive、mean Negative、hardest Negative、query backbone/projector grad/update、正規化前projector norm、batch varianceを比較する。
- **単なるLoss減少を必須PASSとしない。** ただし step33以降もLossが上昇し続ける／hardest negativeがPositiveより高い／varianceが小さいなら研究上の要診断としてGate C前に解釈する。
- 注意：Queueが完全に実keyに置換してもencoder/keyの重みは学習で変化しており、`33–64stepのLoss横ばい`だけで表現品質は証明されない。

### 4.2 Gate C 500step（既存承認spec）

- `report.run_kind=gate_c_pilot500`、`status=completed`、`gates.C.status=PASS`、`completed_steps=500`、全`checks`PASS、checkpoint roundtrip=true。
- 初期Queueは32stepで完全置換、`500*32=16000`枚消費、500step終了後`real_key_count=1024`,`random_remaining_count=0`,`queue_ptr=(500*32)%1024=128`。
- Cosine LRのスケジュールを確認。**Gate C のengineering PASSは「表現が識別的になった」のPASSではない**。Loss増加、Negative類似度、hard-negativeとpositiveのmargin、pre-normalization norms、variance、gradient/updateを診断し、研究的な解釈は別扱い。
- NaN/Inf、EMA誤差、Keyへのgradient、Queue不整合、checkpoint復元失敗はFAIL。実験失敗時もreport/history・failure stage/stepを保存して原因を検討。
- 500stepの成績によらず「MoCoが有益な表現を獲得した」と結論しない。必要なら後日固定評価セット/Linear Probe等の別評価specを設計。

## 5. 保存・Research Workspace Evidence

- 既存artifact契約：`resolved_config.yaml`,`report.json`,`loss.png`,`similarity.png`,`grad_update.png`,`queue.png`,`verification_summary.md`,`checkpoint.pt`。
- 既存20stepと64step（両者固定LR）および500step（cosine LR）を**同じ条件／異なる条件を明記して**並べる。特に初期32stepの負例分布変化はLoss比較上の重要交絡因子。
- 実測が得られたら`.research/lab/projects/sequential-video-lora-analysis/experiments/`に、**run固有のprovenanceと合否、測定値、研究的な限界**を追記。ChatGPTは未実施の値を記録しない。
- 成果物が研究サーバのみに存在する場合、GitHubへ自動同期されたとは言わず、チャットに共有された測定結果に基づいて記録する。

## 6. Implementation / run boundary

既存の`scripts/smoke/smoke_moco_cifar10.py`と`conf/moco_standalone.yaml`だけで両方のrunを構成できる。**新しい研究コードの実装・commit・pushは不要**。研究サーバの実行はChatGPTのGitHub connectorでは実行できない。GPU使用可否・NAS dataset root・runtimeは実サーバでpreflightする。正式なGate Cの実施はユーザーの今回の明示希望に含まれるが、実際の実行結果はユーザー側サーバでしか観測できない。

## 7. Stage2 handoff

64→500の診断結果から、MoCo mechanicsの問題と表現学習上の懸念を分離したうえで[ViT導入spec draft](./2026-10-15-frozen-vit-moco-cifar10-stage2-integration-spec.md)の未決項目を確定する。**Stage2の新コードはこの実験planだけでは承認されない。**
