---
date: 2026-10-15
project: sequential-video-lora-analysis
source_todo: null
topic: standalone-moco-spec-decision-points
status: exploratory
tags: [brainstorm, MoCo, CIFAR-10, standalone, spec-design, evidence]
---

# Stage 1：Standalone MoCo検証specを作るための決定事項（探索）

## 背景・問い・Authority

2026-10-08の教授MTGでは、統合ActivityNet学習でMoCo lossが振動しQuery LoRAの勾配が急増したため、LoRA・MoCo・Sequential Loaderを分離して検証する方針となった。ユーザーは2026-10-15に「まずMoCo単体を実施したい。specで何を決める必要があるか」と依頼。これは設計相談であり、仕様・実装の承認ではない。

研究の次段階は「MoCo単体 → ViT＋MoCo → 固定動画chunk＋MoCo → Sequential Loader＋MoCo → 自作LoRA追加 → Strict Online」とする探索案（[前メモ](./2026-10-15-moco-loader-lora-staged-causal-debugging.md)）。Stage 1では**LoRAもSequential Loaderも動画も使わず**、MoCo coreの数学・Query/Key勾配境界・EMA・FIFO・InfoNCE・2-view・学習安定性を独立検証する。長時間の動画表現性能を主張しない。

GitHub `tamaki-lab/2026_09_ishikawa_sequential-video-lora@dev` HEAD `7dcd194f439c7e28d796e2986857ab0df1982cbe`では、既存の`model/vit/`・CIFAR10 ordinary DataLoader・HydraがあるがMoCo modulesは現行treeにない。歴史的Stage5 MoCoはPEFT・動画を含む未コミット実装の記録であり、そのまま「現在devのMoCo」と扱わない。既存SSH worktree dirty/未pushは確認不能。

Metaの公式MoCoリポジトリ `facebookresearch/moco` は2026-02-03 commit `8976944da9c6b94cbd9158d7ebe50912aef807ef` の`main`でファイルが空になっていることを確認。過去commit `a419ef282fc9885015674e77f58c59447206a27f` のREADME、`moco/builder.py`、`main_moco.py`、MIT LICENSEはGitHub connectorから参照可能。この古い公式版ではImageNet/マルチGPU DDP、ResNet50、queue K=65536、m=.999、v1 T=.07、v2として`--mlp --moco-t 0.2 --aug-plus --cos`を指定。CIFAR10・単一GPUへの改変は「公式論文の完全再現」とは呼ばない。

## 候補の発散と収束：主要なspec決定事項

| 決定点 | 最有力提案 | 理由・残る論点 |
| --- | --- | --- |
| Stage1の責務 | MoCo v2-styleのengine mechanics + 小規模画像pilot、性能ベンチマークは別段階 | 次のViT置換時の独立基準を作る。full ImageNet精度の再現は範囲外 |
| 参照コード | Meta公式の過去commitを正本、式/更新順を明記。研究コードに新規small coreを分離して実装 | 公式現行mainが空。外部コード丸ごとコピーは避けMITの条件に従い必要な参照のみ |
| Dataset | CIFAR10 train画像（ラベル未使用）、subsetや順序・seedを記録、事前ダウンロード済みデータのみ | 既存のStandalone LoRAを検証した場所と一致。入力32x32用augが要る |
| Encoder | unitはtiny trainable encoder、実画像pilotはResNet18-CIFAR仕様（3x3 stride1, maxpoolなし） | 計算量・勾配が見やすい。公式ResNet50 ImageNet再現ではない。BN／single GPU shuffle／running statsの扱いをspecで固定 |
| 2-view | 同じCIFAR画像から独立にRandomResizedCrop/Flip/ColorJitter/Grayなど。GaussianBlur含むか確認し、seed設定・正規化を固定 | 参照MoCo v2の強いaugmentationが核だが224用設定を32に盲適用しない |
| projector | 2層MLP（ResNet18 feature512→512→128、ReLU）＋L2正規化 | 非線形projectorを説明可能にし、正規化前normを記録 |
| Queue K | 1024 または 4096。第一候補1024（小さなrunで早く充填）、過去4096も切替可能 | Queue占有とlossの関係を見やすくする。過去Stage5との数値横比較は不可 |
| Queue初期 | 空＋明示的key-only warmup（候補）、または公式random-normalized初期化（参照用）を決める | warmup方式は過去実装と整合しpast-onlyを証明しやすいが公式との差を明記。比較時に両者を混ぜない |
| Negative | Queue内の過去の*別sample*キー、現在positive/futureを不使用、valid countとsample idを監査 | same-classは真のラベルを見ないためfalse negativeになり得る。CIFAR dataset labelを使用しない |
| EMA | m=.999の初期候補、Keyはrequires_grad false、Optimizer除外。更新対象・BN buffer処理と更新タイミングを明記 | Query/Key初期同一、storage独立。EMA式のunit比較を必須 |
| Temperature | T=.2（公式v2フラグ）を初期候補。旧動画統合T=.07と区別 | 旧実験とのloss値単純比較をしない |
| Optimizer | 標準に合わせSGD momentum=.9とcosine LRを第一候補。ただし初期pilotは固定LRで短時間、optimizer/lr/batchをspecで合わせて固定 | ImageNet・batch256の公式LR=.03をbs32に無根拠に適用しない。初期lr候補=.03×32/256=.00375は探索値、最適値保証なし |
| Batch・steps | batch32、seed0。Gate A synthetic数step、Gate B CIFAR 20step smoke、Gate C pilot500step（初期候補、GPU負荷で調整） | 20stepのみではlearning qualityを判断しない。精度の実証を強制しない |
| 実行環境 | CPUでunit/synthetic、GPUの短時間pilotは環境確認後。単一device・DDPなし | GPU不調の履歴を踏まえ自動でGPUへ飛ばない。既存コードや依存関係を壊さない |
| 保存物 | report.json、resolved_config.yaml、loss/similarity/gradient/queue graphs、verification_summary.md、Query/Key/Queue/optimizer/step/RNG checkpoint | 研究室向けLoRA結果フォーマットに合わせる。復元後の決定的比較・実行provenanceを残す |
| Comet | 第一specはopt-in（既定OFF）、タグと各metricの意味・global_stepを固定 | ローカル成果物を正本、可視化は後から追加可能。旧stage6bとタグ・指標を混ぜない |
| Scope境界 | LoRA/Sequential Loader/ActivityNet/strict online/LinearProbe/full trainingを除外 | 異常時にMoCoへの帰属を明確にする |

## 推奨する検証Gate

- **Gate A / 数学・メカニクス**：人工Q/Key/QueueでInfoNCEが手計算と一致、positive位置正しい、FIFO/wrap-overflow、empty queue/warmup、過去Keyのみ、Keyはbackward勾配0、Query/Projectorのgrad nonzero+finite、EMA式と更新順序一致。異常入力（zero negatives、NaN、容量、誤shape）をfail-fast。BatchNormバッファの独立性/更新ルールもtest。
- **Gate B / 実画像smoke**：同じCIFAR画像から2view、label不使用、実ResNet18+MLPでshape/finite、20stepでQuery更新、Key EMA、Queue挙動、checkpoint保存/復元。Lossの単調減少を合格条件にしない。
- **Gate C / 統制した学習pilot**：固定seed/設定で500step程度。全stepのLoss、positive/negative similarity（hardest含む）、Queue有効数、正規化前projector norm、feature variance、Query grad/actual update、Key EMA/finiteを確認。Collapse兆候や勾配急増を検出した場合はgate保留して診断する。lossやsimilarityの絶対閾値は現時点で定められず、合格条件は工学的基準とモニタリング可能性が中心。意味のある表現品質・汎化改善を主張する場合は後でheld-out linear evaluation等が必要。
- **Gate D / 再現性・非回帰**：同seedでtiny syntheticは数値一致、保存復元直後の同一forward・queueポインタ・optimizer一致、既存ViT/LoRA smokeに回帰なし。大規模GPU完全決定性は主張しない。

## Spec化に必要なblocking decisionsと推奨初期値

1. **MoCo v2-style mechanicsを目的とし、ImageNet公式精度再現を目指さないか**：推奨YES。
2. **参照実装を過去公式commitへ固定して、研究コードには独立small MoCo coreを持つか**：推奨YES。
3. **dataset/encoder**：CIFAR10 train、ResNet18-CIFAR仕様、LoRAなし・Loaderなしを推奨。公式ImageNetの結果は再現目標としない。
4. **Queue初期化・warmup / Negative定義 / 更新順序**：empty queue + key-only warmup、past-only valid negatives、pre-EMA positiveのenqueuing（古い研究運用と整合）を提案。ただしこれは公式MoCo v2のrandom initial queueとは異なる。Stage1で公式式に寄せるならrandom normalized 初期化を選び、差をtestで明示する。
5. **trainingの値と予算**：batch32、seed0、T0.2、m0.999、projector512→512→128、K1024、SGD・LRは画像pilot基準、20step smoke→500step診断。GPU可否とpilot許可は分離。
6. **check/report/pass conditions**：数学のexact test、finite/grad/EMA/Queue、一定stepsの正常終了、collapseの診断ログ、checkpoint reloadの一致。Loss下降単独を成功としない。

未決：BatchNormのKey側buffer扱い、実行タイミング（EMA before key-forward vs optimizer後）、DDPなし時のbatch shuffle、公式augとの差分、empty queueかrandom queueかの最終採用、実際のGPU計算予算。これらはspecで曖昧にできない。

## Research-specへのHandoff候補

目的：Sequential Loader/LoRA不在のMoCo v2-style mechanicsを説明可能にし、短時間CIFAR10で妥当な更新・再現性・数値監査を証拠付きで残す。実装範囲はMoCo core、tiny/real-image runner、Hydra preset、tests、reportingのみ。以前のPEFT依存MoCoコードは安全な参照として扱い、現在GitHub devにあるとは仮定しない。次の段階でfrozen ViTへ差し替えられるQuery/Key encoder interfaceを明示する。現在は探索記録、spec作成・実装・runの許可ではない。


## 2026-10-15追記：公式commit・ResNet18・Queue初期化・EMA/Queue順序の解説と推奨

ユーザーは、Stage1 spec検討の前提として「Meta公式MoCo過去commitの意味」「ResNet18をViTより先に使う理由」「Queue初期のrandom-normalized vs empty+warmup」「EMA・Queueの更新順序」を質問。これは設計の理解・壁打ちであり、正式specの承認ではない。

### 事実と出典

- 公式MoCo `facebookresearch/moco` の2026-02-03 commit `8976944da9c6b94cbd9158d7ebe50912aef807ef` は現mainのファイルが空になった状態。実装とREADMEは過去commit `a419ef282fc9885015674e77f58c59447206a27f` で取得・確認できた。commit pinとはGit履歴中の同一コードスナップショットを指定すること。閲覧するだけならcloneやcheckout操作を強制しない。
- `moco/builder.py` ではQueue bufferを`torch.randn(dim, K)`で作り列単位でL2 normalizeし、最初からK個のrandom keyをNegativeとして使用。real sample IDとは対応しない。`MoCo.forward` の`with torch.no_grad()`内で `_momentum_update_key_encoder()`→`encoder_k(im_k)`、次に古いQueueでlogitsを計算、`_dequeue_and_enqueue(k)`。train loopのoptimizer.stepはforward/loss後なので、Queueのenqueuingはoptimizerより前。BN DDP batch_shuffleはマルチGPU前提。
- 2026-09-24歴史的Stage5 specでは`key-only warmup`で実データkeyをFIFOへ投入後、training stepは `query/key forward→旧Queueでloss→backward→Query optimizer→Key EMA→事前計算済みkey enqueue`。これは公式の更新順序とは異なる。過去の動画runはPEFT LoRA＋動画MoCoを含み、現devにMoCo実装が存在するとは扱わない。

### 初学者向け整理

- ResNet18案：MoCo自身のInfoNCE/Queue/EMAが動くかを、比較的軽いCNNで先に検査。Meta公式の主対象は**ResNet50**であって、ResNet18+CIFAR10は公式のImageNet精度再現ではない。後段Frozen ViT+Projectorはencoder全面学習の通常MoCoとも異なる。ResNet18が必須という意味ではない。BNによるextra complexityはある。
- 初期Queue案A：正規化したrandom Kベクトルを最初から配置。初回からLossとQuery optimizer stepが可能だが初期Negativeは画像から取得したkeyではない。初期損失はwarmup案と数値比較不可。
- 初期Queue案B：空Queue（占有0）→key-only warmupで少なくとも1個の過去実画像Keyを確保→別sampleでInfoNCE update。warmupではQueryのBackward/optimizerを行わず、Queue要素数と更新回数を別カウント。sample IDによるcurrent positiveとnegative混入防止、同一CIFAR画像の別augmentがqueueに存在してもsampleIDで除外するならofficialとの差を明記。valid negative=0ではfail-fastまたは明示warmup、cross entropy single positive onlyを成功Lossとして使わない。
- EMA `theta_k <- m theta_k + (1-m) theta_q`。例m=.9、古いkey=1, 更新前query=2, 更新後query=3なら official pre-optimizer EMA=1.1、研究方式 post-optimizer EMA=1.2。m=.999では単step差は小さくても同一とは限らない。
- 公式タイムライン：Query特徴forward → (KeyパラメータEMA → Key特徴forward) → previous QueueでInfoNCE logits/loss → current key enqueue → backward → Query optimizer step。enqueueはloss計算のあと・optimizerの前。元のQueueでlossを作っているので当step positive keyはNegativeに含まれない。
- 研究タイムライン：Query/Key forward（その時点でEMAなし）→previous QueueでInfoNCE→backward→Query optimizer→Key EMA（updated Query重みに追従）→**事前に算出したpre-EMA key**をenqueue。後からkeyを再ForwardしてQueueに入れたわけではない。future keyをNegativeに使わないことは両方式に共通。
- EMAはKey Encoder/Projectorの重みを追従させる処理。FIFO Queueへのenqueueは特徴ベクトルを記憶する別処理であり、EMAはQueueの値そのものには適用しない。BN buffers（running mean/var）をどのように保持・コピー・更新するかは両方式と別に明示決定が必要。

### 現時点の推奨と未決

Stage1で最も重要なのは「公式との差分の一覧」と「最終動画MoCoで使う順序をStage2以降で変更しない設計」。参照テスト/数式検証では過去公式commitをbaselineとしつつ、**ユーザー研究の主要実装はempty Queue＋warmup、old Queue→Query optimizer→Key EMA→cached pre-EMA key enqueue**を第一候補とする。参照コードを全量copyしない。公式式との違いを単体テストで可視化し、これを公式MoCo v2の完全再現と呼ばない。逆に完全な公式再現をStage1の主張にするならrandom Queue＋EMA-before-key-forward＋forward時enqueueを採用し、Stage2以降の変更は別の独立Gateで検証する。最終採用は未承認。

新たなspec・コード・学習run・TODOの作成はこの説明の範囲外。
