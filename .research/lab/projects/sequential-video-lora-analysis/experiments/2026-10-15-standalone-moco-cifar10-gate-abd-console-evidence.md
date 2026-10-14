---
project: sequential-video-lora-analysis
record_type: experiment-evidence-review
status: gate-abd-passed-gate-c-not-run
created: 2026-10-15
last_updated: 2026-10-15
spec: ../specs/2026-10-15-standalone-moco-v2-cifar10-mechanics-verification-spec.md
related_analysis: ../code-analysis/2026-10-15-standalone-moco-gate-abd-audit.md
implementation_repository: tamaki-lab/2026_09_ishikawa_sequential-video-lora
implementation_branch: dev
implementation_commit: 8939d7fca7d1cc3367b437c5f6adb48dd8e86f4a
github_reviewed_dev_commit: 8939d7fca7d1cc3367b437c5f6adb48dd8e86f4a
evidence_source: user-provided research-server console log and generated verification_summary.md text
dataset: CIFAR-10 train
gate_c: not-run
---

# Standalone MoCo v2・CIFAR-10 20step Gate A/B/D：ユーザー提示console logのEvidence整理

> **Evidence level = ユーザーから提示された研究サーバのconsole log。** ChatGPTはユーザー環境でコードを実行していない。GitHub上の`dev`の対象実装を静的に調査済みだが、**研究サーバの現在のcommit・dirty状態、作成された`report.json`/`verification_summary.md`/`checkpoint.pt`の内容は未確認**。`Gate B=PASS`というreport内の自動判定はこの記録からは未検証。Gate Cはユーザー明示で未実施。

## 1. 実行・出所

- 実行環境：`ishikawa@150-740GP-TNRT`、作業ディレクトリ `/mnt/HDD12TB-1/ishikawa/2026_09_ishikawa_sequential-video-lora`、`.venv`、CPU指定`CUDA_VISIBLE_DEVICES=''`、`OMP_NUM_THREADS=4`。
- CIFAR-10 root：`/mnt/NAS-TVS872XT/dataset/CIFAR10`、`cifar-10-batches-py`に`batches.meta`, `data_batch_1..5`, `test_batch`等が存在することを`ls`で確認。
- Gate A/D対応のMoCo 2テストファイル：`python -m pytest -q -o addopts='' test/model/test_moco_standalone.py test/model/test_moco_cifar10_reporting.py` → **42 passed in 42.01s**。
- Gate B実データ：`python -m scripts.smoke.smoke_moco_cifar10 data.root=/mnt/NAS-TVS872XT/dataset/CIFAR10 data.download=false training.device=cpu`。コンソールに**step1～20まで各JSON出力**され、最後に8 artifacts保存の成功メッセージがある。保存先はリモート研究サーバの
  `/mnt/HDD12TB-1/ishikawa/2026_09_ishikawa_sequential-video-lora/log/moco_standalone/2026-10-15/02-11-51-664374/`。
- 既存LoRA regression：`python -m pytest -q -o addopts='' test/model/test_lora_linear.py test/model/test_vit_lora_frame_encoder.py test/model/test_vit_lora_checkpoint.py test/model/test_vit_lora_reporting.py` → **32 passed in 8.84s**。
- 提示されたconsoleには`git rev-parse HEAD`や`git status --short`の出力がなく、コードprovenanceは未検証。GitHub側で調査された最新`dev` commit `8939d7fca7d1cc3367b437c5f6adb48dd8e86f4a`はローカル実行が同じcommitだった証拠ではない。

## 2. 20step consoleの観測値と解釈

| Metric | Step 1 | Step 20 | 観察・限定された意味 |
| --- | ---: | ---: | --- |
| `loss` | 2.52748084 | 6.38750744 | **19/19遷移で増加**。学習性能の成功ではない。初期random queueが実Keyへ置換される非定常なnegative分布の影響を分離要 |
| `positive_cosine_mean` | 0.91238189 | 0.89264524 | 同一画像2viewは高類似度だが、初期から高いためそれ単独で表現学習成功を主張できない |
| `negative_cosine_mean` | -0.00435864 | 0.52251947 | 初期random Negativeのcos≈0が、画像由来Key（近接した特徴が多い）への置換で増加する自然な要因 |
| `hardest_negative_cosine_mean` | 0.27546352 | 0.94437557 | **step2～20の19/20 stepでPositive平均よりhardest Negative平均が高い**。他画像の特徴と十分に識別できていない可能性・全体的anisotropyを要診断 |
| `query_backbone_grad_norm` | 2.92110538 | 1.86896229 | 観測範囲1.6564～2.9212、急増せず有限 |
| `query_projector_grad_norm` | 2.35019279 | 0.78874952 | 観測範囲0.7340～2.3502、急増せず有限 |
| `query_backbone_update_norm` | 0.01095427 | 0.01645864 | 各step正で有限、更新を確認 |
| `query_projector_update_norm` | 0.00881323 | 0.00844823 | 各step正で有限、更新を確認 |
| `z_q_norm_min` | 1.60734737 | 2.13650799 | 全20step最小約1.4876、正規化前normがゼロ付近に縮小する兆候は今回見られない |
| `z_k_norm_min` | 1.48957896 | 1.49323404 | 全20step最小約1.4497、同上 |
| `q_variance_mean` | 0.00080356 | 0.00057481 | 小さく一部低下。collapseは証明されないが多様性の診断継続 |
| `k_variance_mean` | 0.00084776 | 0.00093288 | 同程度・やや増加。Q/Kで変化方向が異なる |
| `ema_max_abs_error` | 2.980e-08 | 7.486e-08 | 全stepの最大は約7.617e-08。各step`ema_within_tolerance=true` |
| `key_grad_tensors` | 0 | 0 | 全20stepで0 |
| `queue_ptr` | 32 | 640 | 32key/stepの期待値と一致 |
| `real_key_count` | 32 | 640 | 32key/stepの期待値と一致 |
| `random_remaining_count` | 992 | 384 | 1024 - 20×32 =384、完全な実画像由来への置換はstep32 |

**Lossの増加の有力な説明（推論・未確証）**：CIFAR10 random-init ResNet18とMLPの未学習画像特徴が非常に似ており、Queueへ実画像由来Keyが追加されるほど高類似度Negativeが増える。最初のQueueはrandom normalizedで平均類似度ほぼ0。このようにNegative分布が毎step変化しており、別mini-batch上のlossをそのまま単調学習曲線として解釈できない。Step2以降、hardest Negative平均約0.94–0.96がPositive平均約0.89–0.92を上回るため、単純に「正例を負例より識別できた」と主張できない。

**反例・未解決**：同一画像と他画像の混同はラベル不使用でも起こり得る。Hardest negative>positiveは観測バッチ平均に関する事実であり、全画像がそうだったという事実ではない。20stepではcollapse、安定性、LR適否を判定できない。後半のLoss上昇が全てQueue置換のせいとは断定できず、encoder feature geometry／data augmentation／BN／Optimizationの影響も残る。

## 3. Gate判定（証拠レベル厳守）

| Gate | 現在確認できる状態 | 不足 |
| --- | --- | --- |
| A（MoCo mechanics） | **42テストPASS**。実装内の数学・Queue・EMA・BN・異常入力テストが含まれる | コンソールだけでは各assert数値は読めない。結果一覧・test revisionの照合が必要 |
| B（実CIFAR10 20step） | **20step完走、8 artifact保存のコンソールメッセージ、Query更新・Key勾配0・EMA tolerance・Queue算術の整合** | **未受領の`report.json`の`status`, `run_kind`, `gates.B.status`, 全`checks`**を確認して正式PASSを確定 |
| C（500step） | **NOT_RUN（ユーザー明示）** | 学習安定性、Queue全置換後の継続、崩壊/分離性能の実証 |
| D（checkpoint/repro/regression） | checkpointの保存メッセージ、MoCo suiteにresume/roundtrip含む；既存LoRA regression **32PASS** | **実CIFAR10 runのroundtrip結果自体**は未受領。GitHub HEAD/local dirtyの記録も必要 |

## 4. 次に確認すべきEvidenceと作業の推奨

1. **Gate B自動判定の確認（必須・低コスト）**：上記runディレクトリの`report.json`で`status=completed`、`run_kind=gate_b_smoke20`、`gates.B.status=PASS`、全`checks` PASS、`verification.checkpoint_roundtrip.passed=true` を確認。`verification_summary.md`と一致していることも検証。reportはGitHubから読めないローカル成果物。
2. **監視値の解釈**：`loss.png`, `similarity.png`, `grad_update.png`, `queue.png`を確認して、データと図が一致すること、loss増加とreal-key増加の時間的重なり、Key grad0・EMA・Query更新の安定を示す。
3. **短期の追加診断候補**：Queueがrandomから実画像由来に完全置換されたstep32以降を確認できるよう、fresh 40～64stepのcustom実験を**別run**で設計する（spec既定の20step Gate Bを変更しない）。またはGate Cの承認済み500step pilotへ進む。最終的な学習品質評価は、Queueが完全に実Key化した後の値や固定hold-out特徴比較を要する。
4. **識別能力を追加測定**：定義の明確な同一画像2view Positiveに対する、異なる画像Negativeとの差（positive minus hardest/mean negative）を同じ入力条件で追跡する。固定batch・固定Queueの診断を別検証として設ければ「Queueが置換されたせいか」「Encoder更新のせいか」を切り分けやすい。再現性を保つため元のStage1 specと分ける。
5. **静的監査の差分**：現GitHubコードの`comet_enabled=true`未サポート（specはopt-inを想定）と、Meta固定参照のライセンスが本当はMITであるのにソースdocstringがCC-BY-NCと記載する点は別途修正対象。このログのPASSで解消したとはみなさない。

## 5. まだ主張しないこと

- コンソール出力だけからGate Bの**公式report check PASS**を断定しない。
- 20stepのLoss増加を「MoCo実装が壊れている」または「学習成功」のどちらとも即断しない。
- 自己教師あり特徴の表現品質の改善、崩壊なしの保証、100step/500step安定性、動画時間情報獲得を主張しない。
- 以前のPEFT動画MoCoのStage5/6Bと同条件だと扱わない（Dataset/Encoder/Queue初期化/温度/EMA/enqueue等が異なる）。

## 6. 関連資料

- [Approved Stage1 spec](../specs/2026-10-15-standalone-moco-v2-cifar10-mechanics-verification-spec.md)
- [Gate A/B/DのGitHub静的監査](../code-analysis/2026-10-15-standalone-moco-gate-abd-audit.md)
- [2026-10-08 MTG](../meetings/2026-10-08-mtg.md)
- [段階的統合の探索案](../../../../secretary/notes/brainstorm/2026-10-15-moco-loader-lora-staged-causal-debugging.md)


## 7. 2026-10-15追加Evidence：verification_summary.mdによるGate B正式PASS

**新たなユーザー提供物は、同じrun `report.json` から生成された `verification_summary.md` の貼付本文**であり、研究サーバ上のファイルをChatGPTが直接開いたわけではない。コンソール先行記録の「Gate B正式PASS未確認」「Git dirty/head未確認」は**この追記により更新された情報**である。元の観測の経緯を保存するため先行節は改変しない。

### 確定したrun provenance

| 項目 | 提示されたverification summaryの値 |
| --- | --- |
| `run status` | **completed** |
| `run kind` | **gate_b_smoke20** |
| local branch/HEAD | **dev / 8939d7fca7d1cc3367b437c5f6adb48dd8e86f4a** |
| Git dirty | **false** |
| Dataset | CIFAR10 train、50,000枚、640枚使用、download false、seed 0 |
| Model | torchvision ResNet18 weights=None、CIFAR stem、Query/Key projector512→512→128 |
| MoCo | Queue K1024 random unit-key初期、m0.999、T0.2、全Queue Negative |
| Optimization | CPU、SGD lr0.00375、momentum0.9、WD1e-4、scheduler none、batch32、20step |
| Environment | Python3.12.3、PyTorch2.14.0+cu130、torchvision0.29.0+cu130、CUDA13.0（**実行はCPU**） |

### Gateの照合と実測PASS（提供されたファイル本文）

| Gate | 確認結果 | Evidence boundary |
| --- | --- | --- |
| A | **42 pytest passed / 42.01s**（別途提供済console） | 自動生成summaryでは`A=NOT_RUN`（単体テストをrun内で実行していないため正常）。両証拠を合わせてA合格 |
| B | **reportに基づく`Gate B=PASS`**、`status=completed`, `run_kind=gate_b_smoke20` | 提供されたsummaryの「検証結果」は**全14項目PASS**。実ローカル`report.json`のバイト列・checkpointファイルは未受領 |
| C | **NOT_RUN** | 500stepの安定性・表現品質の主張不可 |
| D | **report内`Gate D=PASS`**（checkpoint別instance復元）、MoCo suiteがresumeテスト含む42PASS、既存LoRA 32PASS | 実data runの保存復元報告に基づく。研究サーバcheckpointバイナリ自体を外部検査したわけではない |

**全14のPASS項目：** 初期Q/K state一致・独立storage；Key requires_grad=false；optimizer=Queryのみ；20/20step更新；views/Q/K/logits shape一致（`[32,3,32,32]`, `[32,128]`, `[32,1025]`）；history finite；Key grad 0；EMA tolerance；Backbone60/60 tensors changed；Projector4/4 tensors changed；Queue pointer/count一致；Queue unit norm finite；dataset sample order/target=0；checkpoint roundtrip。

**重要な測定値：** `ema_max_abs_error_max=7.61747e-08`（rtol1e-05/atol1e-06）、`queue_ptr=640`、real/random `640/384`、Queue L2 norm max error `4.17233e-07`、checkpoint state/optimizer一致、復元したeval feature max abs diff `0`、Git clean。

### 解釈の更新

- **確定できた主張（工学的）**：固定された`dev`のcleanなcommitにおいて、承認済みspecのGate A/B/Dに対応するCPU実行・テストは**提供された測定Evidence上合格**。MoCo v2型の計算・Query/Key更新境界・FIFO・checkpoint保存復元・既存LoRAの非回帰を支持する。
- **研究的未解決**：Lossは`2.52748→6.38751`へ19連続増、mean Negative cosineは約`-0.00436→0.52252`、positiveは約`0.91238→0.89265`。初期random Negativeが実画像keyへ置換中であり、損失の単調増だけで欠陥と断定しない。しかしstep2以降hardest-negative平均`~0.94–0.96`がpositive平均`~0.89–0.92`を上回るため、この時点で識別的な特徴学習の成功とは言えない。samplewise判定・Queue置換後の追跡を後段の診断に引き継ぐ。
- **Gate C無しで次段を検討する場合の境界**：MoCo mechanicsの独立controlとして使うことは可能。ただし500step安定性・collapse無し・CIFAR線形評価精度・動画の時間情報獲得は依然**未検証**。Stage2以降に進む場合も、未解決のMoCo学習挙動を隠さず比較軸を固定する。
- **specとの残差**：Comet opt-inがコード上`ValueError`で未実装、公式MITライセンスがソースdocstringではCC-BY-NC4.0と誤表記（2026-10-15 audit F01/F02）。A/B/DのPASSはこれらを解消した証拠ではない。

### 次の検証候補

1. **最小・低リスク**：本runの`report.json`、`resolved_config.yaml`、4図の値一致とSHAsをチェックし、検証証拠として保存。今回提供されたMarkdownからGate Bの判定とprovenanceは確認済み。
2. **学習挙動の原因切り分け**：Queue実画像置換が完成するstep32以後も同じ構成でloss・Positive/Negative cosine・projector varianceを追う**別fresh 40–64step custom pilot**。これを公式Gate C達成とは言わない。
3. **承認済みGate C（まだ未実施）**：500step fresh cosine runnerと勾配・feature norm・varianceの継続観測。実行するならユーザーの実験運用方針に従う。
4. **仕様差分**：Comet opt-in実装またはspec上の範囲変更、ライセンス出典表記の修正を別engineering taskで扱う。

本追記は**ユーザーから提示された測定Evidenceの正確な記録**であり、ChatGPTが独自に実験run、code編集、checkpoint取得、GPU/SSH環境検査を行ったわけではない。
