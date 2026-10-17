---
date: 2026-10-17
project: sequential-video-lora-analysis
source_todo: "2026-10-15 MTG: Meta MoCo v2参照実装の再現・比較と学習条件の検証"
topic: meta-moco-v2-reference-parity-validation
status: exploratory
tags: [brainstorm, moco, reference-implementation, parity, baseline, evaluation]
---

# Meta MoCo v2公式実装との比較によるStandalone MoCo検証の方針相談

## 出発点・中心となる問い
ユーザー提案：「Meta MoCo v2をcloneして実行した結果と、自分のMoCo単体実験の結果が同じになることを目標に、MoCo単体実装がうまくいっていることを確認したい。この方針はよいか」。

中心の問いは、Meta公式実装を独立基準として自作MoCoの **(1) 数学・状態更新の同等性** と **(2) 学習挙動・特徴品質の再現性** をどのように検証するか。「同じ」を無条件に全stepのloss完全一致とは定義しない。

## 既存文脈と確認できた根拠
- 2026-10-15 MTG決定：Meta MoCo v2／研究室内実装の標準条件調査、参照実装clone、なるべく挙動を変えずにComet保存追加、基準曲線作成後にLR・学習期間見直し、その後ResNet→ViT→ViT+LoRA→Sequential Loader。
- Standalone CIFAR-10 MoCoは64-step／500-stepとも工学GateでPASS。EMA、Queue、checkpointの機構検証であり、識別的特徴の獲得は未検証。500step cosineで終盤LRほぼゼロ、step500 Loss約6.64。
- 2026-10-17時点にGitHubの `tamaki-lab/2026_09_ishikawa_sequential-video-lora@dev` から直接確認した `self_supervised/moco/standalone_moco.py` はCIFAR仕様ResNet18、別Projector（512→512→128）、Random unit queue、EMA（Key-forward直前）、old queueでlogits、enqueue、backward/SGD、単一device。 `scripts/smoke/smoke_moco_cifar10.py` と `conf/moco_standalone.yaml` はCIFAR-10、batch32、K1024、m0.999、T0.2、SGD LR0.00375、20step固定 / 500step cosine、公式aug-plusの32x32適応。Comet enabled=true は現在 `ValueError` になる。各ソースの内容はGitHub上の確認に限る。
- Meta `facebookresearch/moco` の公式MoCo v2はImageNet、ResNet50、DDP、global batch256、K65536、200epoch、SGD LR0.03などを基準とし、公式READMEでv2には `--mlp --moco-t 0.2 --aug-plus --cos` を明記。v1の既定引数のままではv2再現にならない。参考の固定commitは現行Standaloneコード内 `a419ef282fc9885015674e77f58c59447206a27f` と記録。
- 関連資料： 
  - https://github.com/facebookresearch/moco
  - https://github.com/tamaki-lab/2026_09_ishikawa_sequential-video-lora/blob/dev/self_supervised/moco/standalone_moco.py
  - https://github.com/tamaki-lab/2026_09_ishikawa_sequential-video-lora/blob/dev/scripts/smoke/smoke_moco_cifar10.py
  - https://github.com/haruto2919/research-workspace/blob/main/.research/lab/projects/sequential-video-lora-analysis/meetings/2026-10-15-mtg.md
  - https://github.com/haruto2919/research-workspace/blob/main/.research/lab/projects/sequential-video-lora-analysis/experiments/2026-10-15-standalone-moco-gpu-64step-and-gate-c-500step-evidence.md

## 仮説、評価軸、反例
### 最有力方針：検証を二層に分離
1. **実装数値同等性（mechanics parity）**：公式 `moco/builder.py` を参照し、共通条件・固定された二つのview、初期Query/Key/projector/Queue・BN buffers、RNG、Optimizerを揃えて、1stepおよび複数stepのQuery/Key特徴、positive/negative logits、InfoNCE loss、勾配、Optimizer更新、EMA、Queue内容とpointer、BN buffersを許容誤差で比較する。公式ロジックは原則無変更。バックボーンを揃えるための外部adapter／state_dict変換、単一rank-DDPなど、同等性に影響し得る層を切り分ける。
2. **学習・表現同等性（behavioral reproduction）**：同じdataset split、Backbone、Augmentation、data order、seed、batch、Queue、EMA、T、optimizerとLR scheduleの全期間を一致させ、固定eval setでのloss／positive rank／margin／retrieval、特徴variance/effective rank、可能ならheld-out linear probeを比較。複数seedで差を確認。全stepのtrain lossがビット単位で同じことではなく、学習挙動・評価性能が測定誤差とrun間変動の範囲で同等かをみる。

### 参照実験の運用
- **A: 公式忠実run**：指定commitの公式実装を独立cloneし、標準条件（MoCo v2 flagsを明示）で参考曲線を得る。正式再現の追跡性確保のため学習意味論の変更を避け、Comet計測はloss／設定／stepの非侵襲loggingとして導入。公式フルImageNet 200epochは計算資源・データ availabilityに依存するため、最初から必須にはしない。
- **B: 比較制御run**：公式実装のMoCoロジックを保持しつつ、研究比較用の外部harness／データアダプタで自作版と同じCIFAR-10・CIFAR-ResNet18・一律batch/Kに揃える。これを「公式ImageNet再現」と混同しない。
- **C: 自作run**：Bの完全同一条件で学習・評価する。
- 公式忠実Aと自作Cのloss絶対値をそのまま比較することは不適切。

## 比較方式の候補と判断
| 候補 | メリット | 弱点・評価 |
|---|---|---|
| 公式/ImageNet/ResNet50 と自作/CIFAR/ResNet18 のloss曲線をそのまま一致判定 | 追加作業小 | dataset/model/optimizer schedule/BN/DDP/Queueが異なり、**棄却寄り** |
| 固定入力・同一初期状態のstep-wise parity | 数学・更新差分を最も直接的に絞る | DDP shuffle-BN、numercial precision、state_dict変換が必要；**強く推奨** |
| 同条件の複数epoch・複数seed、固定評価集合の曲線と性能比較 | 学習再現性の検証に強い | 計算費用。epoch設計と評価条件を確定する必要；**推奨** |
| 公式ImageNet 200epochのみを先に丸ごと再現 | 公式性能と接続しやすい | 計算資源・データ費用、CIFAR自作版の正しさは直接判定できない；**保留／別目的** |

## 成功条件（spec化前の提案）
- Gate 0: 公式version/commit、実装diff、データと入力条件、MoCo v2 flags、GPU/世界サイズ、環境、Checkpoint、dataset indices、Comet logging有無が追跡可能。
- Gate 1: 固定view・共通初期stateに対する1-step forward/logits/loss、backward gradients、EMA・Queue・SGD・BN stateの整合を `torch.testing.assert_close` 等で確認。許容誤差（例えばFP32のrtol/atol）やGPU determinismはspecで測定・固定する。
- Gate 2: 2step以上〜Queue一周の複数step比較で状態遷移とOptimizer momentum bufferを照合し、相違があれば最初の不一致pointと原因を記録。
- Gate 3: 同一条件で短時間、次に複数epochのreference/self学習を比較。固定評価集合と複数seedの結果で「表現品質」を判定。学習lossの単調減少は必須条件ではない。定量的な同等許容幅はpilot結果前に決める。
- Gate 4 (任意): 公式ImageNet実験の既報Linear Probe Top-1（200epoch 67.5±0.1%など）を再現するのは別の、計算費用の大きなGate。

## 不確実性／反例
- BNのshuffle、distributed all_gather、global/per-rank batch、RNGの消費順序、torchvision preprocessing、DDP optimizer/scheduler step timingの差が厳密一致を妨げ得る。公式に準拠した差を不用意に改変しない。
- 初期seed一致でも重み・Queue・augmentとsample順が一致するとは限らない。値を明示的に移植する。
- referenceと自作のLossが偶然近くても、誤ったQueueやfeature collapseの可能性を除けない。fixed eval＋内部状態照合が必須。
- 学習Lossが違ってもBN/DDP/別GPU/AMPなどの差に起因し得るので、即バグと断定しない。
- 公式の値と自作値が同等でも、ActivityNet逐次LoaderやLoRAの動作の正しさ・時間情報獲得は未検証。

## 結論・次アクション候補
**採用候補：Meta公式を独立基準としてクローンし、A) 固定状態の機構数値一致 → B) 同一条件での学習・特徴評価 → C) ViT→LoRA→逐次Loaderへ進む。** 公式忠実runと自作同条件比較runを別物として運用する。

次アクション候補（TODO未反映）：
1. 公式commitの確定、DDP single rank／使用可能GPUの確認、Comet計測の非侵襲性の方針確定。
2. Official/CIFAR parity harness のscopeとBackbone/state_dict/BN compatibility方針を決める。
3. 差分表・同条件のデータ/augmentation/LR horizon等の比較契約、Gate/判定許容差、artifact/log designをspec化する（ユーザーから明示依頼された場合のみresearch-specへ）。
4. 制御された短期parity後に必要epoch・sample数と固定評価方法を決める。

## Research Spec Handoff候補（未承認）
- 目的：自作Standalone MoCoの公式MoCo v2に対する機構同等性と学習再現性を分離して検証。
- Scope：公式clone baseline / adaptation harness / comparison reports、Comet及びローカルraw artifacts。
- Out of scope：Video/Sequential Loader、ViT、LoRA統合、実証前の表現獲得主張。
- 未決：公式runの忠実度とハードウェア条件、CIFARの比較adapter方式、評価指標、数値許容誤差とseed数、予算／学習期間。
- Authority：本記録はexploratoryな提案であり、MTG採用済みspec・コード実装許可ではない。

---

## 追記 2026-10-17T13:29:58+09:00：ImageNet-1K／ResNet／DDP＋Shuffle-BN／10epoch／Queue縮小案

### ユーザーからの追加方針（暫定）
- ImageNet-1Kを公式と自作の共通データセットとする。
- BackboneはResNet系にする（実装同等性検証では同一アーキテクチャ、ResNet50推奨）。
- Queue容量65,536では研究環境に乗らない可能性を考慮し、縮小を検討する。
- 学習期間は両者とも10epochに揃える。
- DDPおよびShuffle-BNを両者で有効化する。
- 意図する成功は公式論文の200epoch性能再現ではなく、**同一変更を入れた公式MoCo v2と自作MoCoの数値同等性・短期学習挙動の再現**である。

### 根拠と解釈
- 公式MoCo v2は `--moco-k` でQueue変更可能。Kを同じ値に減らした比較はできるが、負例数と保持期間が変わり、65,536の場合とLoss絶対値や表現品質は一致しない。
- FP32 dim128のQueue本体はK=65,536でも128×65,536×4=33,554,432 bytes=32MiB/GPU。QueueのみがVRAMの主因とは限らない。ResNet50 224x224のactivation、local minibatch、DDP buffers/optimizerと実測を確認して縮小要否を判断する。K=16,384（8MiB）、K=4,096（2MiB）が暫定縮小候補。
- MoCo公式builderではglobal/all-gather batchでenqueueし `K % gathered_batch == 0` を要求する。例：global batch256ならK=16384は64step、K=4096は16step、K=65536は256stepで一巡。グローバルbatchとGPU数、rankごとのbatchを同一にする。
- 10epochは初期の機構・短期学習比較に適切だが、公式ImageNet-1K ResNet50の200epoch Linear Probe結果を再現したと主張しない。10epochのcosine完走は200epochの先頭10epochとは別のLR曲線である。比較には `--epochs 10 --cos` を両者に使うか、200epoch scheduleを保って10epochで中断するかを実験契約として明確化。
- 公式DDP Shuffle-BNを参考に、self実装は現状のsingle-device/no Shuffle-BNから拡張が必要（コード確認：`self_supervised/moco/standalone_moco.py`, `scripts/smoke/smoke_moco_cifar10.py` on tamaki-lab@dev）。`SyncBatchNorm`と`Shuffle-BN`を混同しない。all_gather、batch shuffle/unshuffle、Queue同期、BN buffers、DDP broadcast設定、sampler set_epoch、view RNG、init weight/queue、fp precisionを照合する。
- 公式・自作を両方同じImageNet/ResNet50/K/epoch/DDP/Shuffle-BN/augmentation/optimizer/total batch/seedで試すのは妥当。**ただし、独立ランダムrunのloss値を厳密一致させることは数値同等性を証明しない**。固定state/fixed augmented views/Queueのone-step＆multistep parityと、10epochの傾向・固定評価指標の比較をGate分離する。
- 試験例：公式忠実版のK65,536・200epochは参考条件として記録、比較用の公式変更版K16384・10epochと自作変更版K16384・10epochの2本を揃える。具体KはGPUの実測を待ち、未決定の候補に留める。

### 次の判断事項・反証条件
1. 選択するResNetのvariant（公平比較のためResNet50推奨）、global batch/GPU数と使用可能VRAM、ImageNet storage/preprocessingの確認。
2. Queueを削減する理由がQueue本体のVRAMかlogit計算かGPU総VRAMかを測定し、K=65,536維持の可能性を再確認する。
3. 10epochでCosineを完了させる短期pilotと、200epoch schedule最初10epochの比較のどちらかを決定。
4. DDP下のfixed batch／BN shuffle permutationを含めた一段階・Queue一巡の比較の許容誤差と、10epoch時の学習評価指標の定義。
5. 10epochで両者が似た挙動でも特征品質は未保証。損失差があれば先にdata order, Aug, DDP ranks, BN, queue parityを調査。
6. 次のspec作成・コード変更・GPU runは別依頼とGateを経る。本文はexploratoryであり実装許可ではない。

### 参照先
- 公式：https://github.com/facebookresearch/moco
- v2論文：https://arxiv.org/abs/2003.04297
- 現行コード：https://github.com/tamaki-lab/2026_09_ishikawa_sequential-video-lora/blob/dev/self_supervised/moco/standalone_moco.py

---

## 追記 2026-10-17 13:33 JST：実測クラスタ資源からのQueue・実行時間評価

### ユーザーから提示された現環境・証拠
- 2026-10-17 13:29 JSTの`cluster-smi`画像：node `150-740GP-TNRT`、RTX A4500×2、各GPU表示19GB、GPU0使用約0.23GB・GPU1約0.01GB、両者GPU Util 0%、driver615.71.09、CUDA13.4。計測時点はアイドルで学習peakではない。
- `cluster-top`画像：64 CPU cores、物理RAM251GiB、使用5GiB、CPU待ち時間低い。データセット保存先やSSD/NAS I/Oは未確認。
- DDPではVRAMは各GPU19GBの独立空間。19+19GBを単一rankモデル用38GBとして足し合わせてはならない。

### Queue容量についての判断
- MoCo v2におけるFP32 dim128 Queue本体のメモリ占有はK=65536で32MiB／rank、K32768で16MiB、K16384で8MiB、K4096で2MiB。加えてlogitsや演算一時領域は必要だが、ResNet50 224^2のactivations、局所batch・DDP/BN等がより大きい制約になり得る。
- 現環境なら**K65536維持が第一候補**。Kを縮める前に2GPUでbatch32／rankを試し、`torch.cuda.max_memory_reserved()`/`max_memory_allocated()`、`nvidia-smi`、OOM、処理速度を確認。必要なら局所batch縮小や、比較上の影響を明示したAMPの検討。公式と自作は同じdtype/AMPを揃える。
- Kを変えると負例の数・特徴の保持時間・lossが変わり、元の公式200ep性能との一致は主張できない。両実装で同Kなら機構比較は可能。global batch64の場合K65536一周1024steps、K16384なら256steps。

### Batch／LR／DDPの暫定方針
- 推奨候補は同一ResNet50/ImageNet-1K/224px/2GPU-DDP/ShuffleBN、**local batch32×2=global64**、SGD initial lr=`0.03*64/256=0.0075`、K65536、m0.999、T0.2、MLP・aug+、weight_decay1e-4。これは測定前の候補でありbatch32が19GBに収まることはまだ実行確認されていない。
- global64ならImageNet-1K train約128万枚で1epoch約20,018step、10epoch約200,180step。global128（local64/rank）なら約10,009step／epoch、ただし局所VRAMを実測して採否判断する。
- DDP shuffle/unshuffle、all_gather、sampler/set_epoch、global/local batch、drop_last、BN buffer、seed/transform/Queue state/optimizer/evalの照合が必要。

### 学習期間と推定実行時間：測定値ではなくモデル仮定
- Meta公式READMEでImageNet ResNet50 MoCoの200epochがV100×8で約53hという公開基準。これをGPU台数のみ単純換算すると、10epoch・2GPUで約10.6h（**V100とA4500が同等速度と仮定**）。実測データではなく機種・CPU/I/O/precisionの差を考慮できない。
- 2×A4500・local batch32(global64)、約20,018step/epochに対し、**定常1step 0.20 / 0.35 / 0.60秒**という明示的な仮定を置くと、1epoch=約1.1 / 1.95 / 3.34h、10epoch=約11.1 / 19.5 / 33.4h、30epoch=約33.4 / 58.4 / 100.1h、50epoch=約55.6 / 97.3 / 166.8h／1 run。epoch起動、I/O、checkpoint、eval、2 runを考えれば上振れ。過去のCIFAR 500stepからImageNet ResNet50の速度を直接推定しない。
- **推奨**：100〜300step実画像warmup/bench（特にI/O込）→1epoch実測→両実装の10epoch比較→必要なら30〜50epoch。10epochのLoss一致だけでは表現品質の証明にならない。固定validation positive rank/similarity/features/effective rank、可能ならLinear Probe等を併用。
- Cosineの総epoch10と50は別学習設定。resumeで30〜50epochへ延長したい場合、**初めから長期total epochsを同一に設定してepoch10でcheckpoint停止→再開**するか、10epoch独立pilotと30〜50epochは別fresh-runとして区別。実装のsave/resumeがschedulerとRNG、DDP data orderを保つか別途検証する。

### 未確認・次の判断
- A4500×2でResNet50 MoCo DDP local batch32 FP32がVRAM19GBに収まるか未確認。
- ImageNet-1Kが研究環境で適法に入手・読み出し可能であるか、I/Oパス・デコード速度未確認。
- 学習期間のprimary objectiveが「自作vs公式のparity」か「特徴品質の充分な向上」かで10epoch成功Gateが違う。前者は10epochで有意義、後者は最終Linear Probe込みで別Gate。
- 次回は100〜300 stepの`images_per_second`、`step_time`、`max_memory_reserved`、`nvidia-smi`（both rank）、local batch、AMP有無から実行時間を再推定する。現在は設計探索であり実験run未実施。

---

## 追記 2026-10-17T14:03:05+09:00：spec準備：公式MoCo v2と自作MoCoの比較実験に必要な判断契約

### 相談の位置付け
ユーザーは、実験仕様の確定に着手し、spec作成に先立って決めるべきことの洗い出しを依頼。今回の依頼は**spec本文の作成や実装・GPU runの承認ではない**。以下は探索段階のDecision Contract候補。

### 確認済み（Authority区別）
- 現在のユーザー指示：共通データImageNet-1K、ResNet系、2-GPU DDP＋Shuffle-BN、10epochを第一目標に検討。Queueを削減する当初案は、10/17提示のA4500×2のidleメモリから65,536を維持する案に更新されつつある（**未実測の推奨候補**）。
- 2026-10-15 MTG：公式実装を参照し、標準条件との比較でMoCo mechanicsと学習の成立を区別する。完成後、ResNet→ViT→ViT＋LoRA→Sequential Loaderへ進む。
- 現在コードGitHub `tamaki-lab/2026_09_ishikawa_sequential-video-lora@dev` の `self_supervised/moco/standalone_moco.py` はCIFAR用ResNet18、single deviceで`all_gather`は恒等、DDP Shuffle-BNなし。`scripts/smoke/smoke_moco_cifar10.py` はCIFAR専用、single epoch内max_step、Comet ONを未実装のため拒否。`conf/moco_standalone.yaml`もCIFAR条件。既存CIFAR仕様とテストを非回帰で残す必要。
- 既存2026-10-15承認済みCIFAR standalone specはImageNet/DDPをscope外とするため、今回用の**新規specが必要**。
- 既存承認済みspecには、`facebookresearch/moco`の現在`main`で学習コードが取得できず、歴史的参照`a419ef282fc9885015674e77f58c59447206a27f`を使用すると記載。今回のspecでは`git checkout`するcommitと固定対象ファイルおよびSHAを記録する。公式v2用flagsは `--mlp --moco-t 0.2 --aug-plus --cos`。
- 既存EvidenceはCIFAR 64/500step工学PASSであり表現品質やDDPの動作についての証拠ではない。
- 実サーバのGPU training throughput、ピークVRAM、ImageNet-1Kのデータ所在・I/Oは未確認。

### spec作成前に必要なDecision Contract（推薦・未採用）
| 優先 | 項目 | 推奨・選択肢 | 影響 |
|---|---|---|---|
| blocking | specのscopeと実行境界 | まずimplementation+parityと10epoch training experimentを別specとして扱い、長時間runは別実行許可 | 既存baseline変更、実装権限と長時間runを混同しない |
| blocking | 「再現」の定義 | 固定入力・初期状態・2GPU通信までのnumerical parityと、10epoch behavioral/representation evaluationを別Gate | Loss値だけでの誤判定回避 |
| blocking | 参照実装のauthority | Meta固定commit、公式ロジックは変えず可観測性のみ最小追加。単体公式ベースラインと共通条件版を明記 | provenanceと再現比較 |
| blocking | model/data/augmentation | ImageNet1K train(SSLラベル不使用)・val、ResNet50 random-init、同じ224px/aug+/sampler、実際のImageNet rootとmanifestをpreflightで確認 | 直接比較可能性 |
| blocking | GPU/optimizer/AMP | 2×A4500、world_size2、local B32/global64を第一試験、SGD lr0.0075 momentum0.9 wd1e-4。最初のparityはFP32; AMPが必要なら共通変更版として別Gate | 数値差と性能/VRAM |
| blocking | Queue・MoCo | K65536を第一候補、m0.999、T0.2、128-d、MLP、初期random keys・enqueue順序・全rank同期の契約 | 負例・Loss・数値一致 |
| blocking | Scheduler / epoch | 10epochでcosine終える独立pilot **または** 30/50epochのcosineを定義し10epで途中観測するrun。後者の途中変更は別実験になる | 学習結果と延長可能性 |
| blocking | DDP / Shuffle-BN semantics | original同様のshuffle/unshuffle、all_gather、rank別batch、queue同一性、BN buffers、sampler.set_epoch、step計数 | 通信・モデル状態 |
| blocking | Parity tolerances | 同じview・weight・Queue・BN・Optimizer・shuffle順で1stepおよびQueue full cycle以上（K/globalB=1024）のforward/logits/gradient/EMA/SGD/buffersを明示rtol/atolで比較。FP32暫定1e-5/1e-6は事前試験で検証 | 工学合格の客観化 |
| blocking | Training evaluation | 同条件2 runs、10epochのepoch-mean loss, pos/neg separation、fixed eval, feature statistics; label-free probeかheld-out linear probeの扱いを選択。1seed pilotと追加seedの必要性を分離 | 表現品質を別Gate判定 |
| blocking | Artifacts, checkpoint/resume | Both Comet + local loss/similarity/grad/update/LR/Queue/throughput/maxVRAM/DDP rank logs, config, code SHA, dataset manifest, weights, resume integrity, failure reports | 証拠と比較・再実行 |
| non-blocking unless it affects results | logging cadence/worker count, visual style/path | officialに合わせ比較契約で具体化。worker数はI/Oと乱数再現性を事前確認 | 計測負荷と性能、可用性 |

### 推奨Gateと成功条件（未承認）
- Gate0: pinned reference checkout・Dataset preflight・CIFAR既存テスト非回帰・DDP起動／rankログ／VRAM確認。
- Gate1: deterministic first-step parity（固定view／init state／rank割当）、q,k, logits, CE loss, grads, Optimizer momentum, EMA, BN buffers, Queueの誤差保存と閾値判定。
- Gate2: 複数step・Queue循環・checkpoint resume。global batch64・K65536なら1024stepで一巡。
- Gate3: 実ImageNet 100〜300step benchmark、最初のwarmupを除いたstep time・I/O・max VRAMで10epoch時間を再計算。Queue縮小を最初の解として選ばない。
- Gate4: 参照版＋自作版を同一条件で10epochの独立run、固定評価で表現学習の比較。Loss単調減少や相互のbitwise一致は必要条件ではない。
- Gate5（将来）：必要に応じて30〜50epoch、Linear Probeや追加seeds。学習結果の外挿ではない。

### 対案・反例
- 公式ImageNet200epochの精度と自作10epochを照合する案は評価目的が異なり棄却。
- Kを先に縮小するとメモリ節約は僅少、負例とLoss条件が変わるため今回の目的には劣後。ただし実測OOMや計算負担が発生した場合は修正可能。
- seedだけ揃えて独立runのLossが似ても実装数値一致の証拠にならない。共有した重み、サンプル順、augmented views、DDP permutation、Queueを固定する。
- 10epochでは表現品質が十分とは言えない可能性。特徴品質の定量基準とparity工学基準を混同しない。

### 次に確認すべき最小の未決事項
1. **specを二分するか**：先に公式+自作のDDP/parity implementation spec、次に10epoch experiment spec（推奨）。
2. **10epochの意味**：独立した10epoch cosine run／30〜50epochの途中観測（推奨は初回の独立10epoch、追加runは別fresh）。
3. **学習評価**：10epochは固定validation similarity+feature統計まで必須にするか、軽量Linear Probeまで必須にするか（推奨は前者必須・後者別Gate）。
4. **数値一致の対象と許容差**：FP32の固定1step+Queue1周を原則にするか、最初からAMPを含めるか（推奨FP32）。
5. **実行予算・GPU test**：local batch32のVRAM/速度を計測して採否、ImageNet rootは実サーバ側で確認。

**Authority:** 本追記はbrainstorm exploratoryであり、正式spec、実装承認、GPU長時間run承認ではない。
