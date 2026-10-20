---
date: 2026-10-20
project: sequential-video-lora-analysis
source_todo: null
topic: meta-moco-comet-learning-curve-and-reproduction
status: exploratory
tags: [brainstorm, research, moco, meta-reference, comet, learning-curves, representation-evaluation]
---

# Meta公式MoCoと自作MoCoをCometで比較し、数値再現と表現学習成立を分けて検証する（探索）

## 出発点とAuthority

ユーザーの希望：「Lossなどの比較をCometを使って行いたい。MoCoで正しく学習ができていて、それを自分のやつで再現できていることを確認するため」。

本記録は**相談段階の検討メモ**であり、研究コード変更、学習run、Cometへの実データ送信、既存spec変更、実装Gate通過の承認を意味しない。

Authorityの参照先：
- [2026-10-15 MTG決定](../../../lab/projects/sequential-video-lora-analysis/meetings/2026-10-15-mtg.md)：公式MoCoに対してLoss/類似度/更新量を比較し、単体MoCoで標準的な学習を成立させてからViT/LoRA/逐次Loaderへ進む。CIFAR500stepの工程PASSだけでは表現学習成功としない。
- [最新Project README](../../../lab/projects/sequential-video-lora-analysis/README.md)、[A3正式Evidence](../../../lab/projects/sequential-video-lora-analysis/experiments/2026-10-20-meta-moco-a3-synthetic-two-gpu-1025step-formal-implementation-evidence.md)。
- [A5監視overhead spec](../../../lab/projects/sequential-video-lora-analysis/specs/2026-10-20-meta-moco-a5-imagenet-gpu-bench-monitor-overhead-experiment-spec.md)。A5-pairは自作のMonitor OFF/Lightの処理時間差を計測するもの、CometはOFF、Meta公式とのLoss比較・表現学習の成否判定ではない。
- 現在コードは`tamaki-lab/2026_09_ishikawa_sequential-video-lora@dev`の`scripts/parity/moco_imagenet_parity.py`と`conf/moco_imagenet_parity.yaml`、`scripts/parity/run_moco_gates.py`をread-only確認。GitHub上のdevはA3 2GPUが実行されたself commit `914e736c2699e2e129211a888d1f19013dd1dd1a`相当としてユーザー提出ログがある。作業時点のSSH localの未コミット変更は未確認。

## 確認済みの基盤

- A2のImageNet実画像2GPU・1step Meta公式M対自作S1では3,926/3,926比較エントリbitwise一致。
- A3の固定synthetic入力2GPU・1025step×2 runでは58,790/58,790比較エントリbitwise一致、full traceはstep1/1024/1025。**数値的再現の強いEvidence**だが、実ImageNetで有用特徴が学習された証明ではない。
- A5-pilotはImageNet実画像で自作`meta_einsum`通常bench20stepの実行成立と測定窓を確認。表現の有用性は未評価。
- 現在の`moco_imagenet_parity.py`は`CometRun`で**両adapterのrank0**から`loss`、`positive_cosine_mean`、`negative_cosine_mean`、`queue_ptr`、`lr`および条件付きnormを`log_metrics(..., step=step)`記録できる。Comet自動ログは停止、`run_kind=parity`は毎step記録、`run_kind=bench`は`monitor.enabled`で記録間隔が決まる。
- A5-pair統合ランチャーは`S1_off100`と`S1_light100`の**自作MoCoだけ**を実行し、`logging.comet_enabled=false`を強制する。今回の主目的と別。
- **重大な比較境界**：現状`bench_loop`は`adapter.build(None)`でMeta/selfを独立初期化、`sampler.set_epoch(0)`、`adapter.set_epoch_lr(0)`、`next(iter(loader))`を1回のepoch範囲で消費する短期benchmark形式。両モデルの同一初期パラメータ・2-viewランダム変換のbitwise対応を保証しない。**max_steps増加だけで正式10epoch比較へ転用しない**。`parity_loop`は同一fixture stateとviewsを両adapterへ読ませ、毎steptrace/historyを書き出す。

## ユーザーが確認したい問いを分離

1. **Engineering reproduction**：同じ入力・初期重みで公式と自作のLossや特徴量、grad、SGD、EMA、Queueが一致するか。→ A3正式PASSの強いEvidenceが既にある。
2. **Training behavior**：ImageNet実画像でMoCoが有限/安定に学習し、正例と負例の関係、学習率と更新等が期待どおり推移するか。→ 新規longer trainingが必要。単一ミニバッチInfoNCE Lossの単調低下は必要条件ではない。
3. **Representation quality**：ラベルなしMoCo事前学習の特徴が下流評価に有用か。→ 固定held-out評価（画像retrieval、ImageNet val kNN、凍結encoder linear probe）を初期/学習後/公式/自作で比較しなければ証明できない。Meta公式との学習曲線一致だけでは両者が同じ失敗を再現する可能性を排除できない。

## 比較案とトレードオフ

### 推奨候補A：既存A3の履歴をCometへ事後登録（追加GPU学習不要）

`A3/runs/M/run_report.json`と`A3/runs/S1/run_report.json`の各rank0 `history[].{global_step,metrics}`からLoss/positive_cosine/negative_cosine/queue_ptr/LR/query_grad_norm/query_update_normを読み出し、同一Comet projectで独立experimentにstep軸を維持して表示。Provenanceと原artifact hashを明記。メトリクス名は両者共通。uploadは入力をread-onlyにして二重登録防止、送信失敗時の再試行を設計。**同一synthetic fixture上の数値一致を見やすく示すだけ**で、表現学習成功の新証拠にはならない。

### 推奨候補B：実ImageNetでMeta公式と自作を揃えた学習runをComet比較

2GPU / ImageNet train / ResNet50+MLP / local32 global64 / K65536 / FP32 / same seed and fixed initial state / same augmentation definition, sample IDs and epoch0+ data order / same optimizer/scheduler+EMA+Queue/InfoNCE arithmetic / common metric definitions。参照cloneは改変せず、両adapterを利用する薄いrunnerまたは別プロセスのreport→Comet uploaderとする。

まず短いpaired canary（同一入力・初期state、固定subset/1epoch未満）でLoss, positive and negative cosine, hardest negative, LR, Query grad/update, Queue occupancy/pointer, feature variance, finite + DDP/checkpointを確認。数値の一致を主張するなら固定入力/transformとstateのhash及び各step comparatorをGate化する。**Independent trainingではモデル初期化・乱数入力・異なるaugmentation・dataset orderが完全一致を保証しない**ため、Lossの近似形状は再現傾向の可視化、bitwiseはA3の正式Gateで判定する。

条件整備後、採用された学習期間で2runを実行し、Cometではstepとepochの双方でoverlap表示し、同一条件の曲線・変化を比較する。別seed/条件は異なるexperiment groupに分ける。10epochは従来の研究構想に存在する**短期の学習挙動比較候補**であり、公式MoCo v2の標準的な完全収束epoch数を保証するものではない。**現在のbenchはmulti-epochを正しく回さないため、設計と実装の拡張Gateが必要**。

### 推奨候補C：凍結backboneの評価

同じ分割と同じ計測protocolで**学習前 / Meta学習後 / 自作学習後**のImageNet val retrieval/kNNを最低限確認し、必要なら線形probe（trainでlinear head、val評価）。選択した特徴位置（backbone pool / projector）、正規化、Augなし評価、checkpoint/seed/ラベル使用範囲を固定。これはContrastive pretextのLossや正例コサインだけでは見抜けないcollapse/無意味特徴を排除する補助Evidence。

### 保留または棄却寄り

- 現A5-pairのmonitor OFF/Light 100stepをそのまま「Meta vs 自作Loss再現」と呼ぶ：**不適切**。比較条件も目的も異なる。
- Cometへ送れたことだけで学習成立と判定：不適切。Cometは可視化/管理基盤で科学的検証条件ではない。
- Loss単調低下/最後のLossの絶対値だけでPASS判定：不適切。MoCo InfoNCEはQueue交換・augmentation・negative難易度・LRに左右される。
- 実ImageNet full trainingを`bench.max_steps`だけ増やして「10epoch」扱い：不適切。sampler / LRのepoch進行未実装。
- Meta参照clone内にComet挿入のため学習ロジックを直接改変：非推奨。外付けlogger/uploaderによる実験ロジック保全を推奨。

## 推奨の判断軸・Gate

| 段階 | 目的 | 合格/観測 |
|---|---|---|
| A: 既存A3 Comet可視化 | 再現性のエビデンスを人が読める曲線に | 同じstep indexの公式/自作メトリクス差0、各run/report SHAと成功ログ、二重送信なし |
| B: ImageNet matched training | 実データでの学習経路の推移比較 | 各設定・provenance、同一初期state/input契約、finite、Queue管理、Loss/pos/neg/grad/LR履歴、full run report; 独立run曲線の一致条件とexact tensor判定を混同しない |
| C: 固定表現評価 | 有用な特徴獲得の確認 | same split、backbone features、val kNN/linear probeやretrievalで学習前との比較。必要なら複数seed |

「公式と自作の曲線が一致しない」場合の代替説明：初期値とaugmentation不一致、Sampler/worker RNG、Queue初期状態、gradient scalingやscheduler、run length、Comet自体のmonitorタイミング、順序/キャッシュ変動。

## 未決事項（research-specへの引き継ぎ候補）

- まずA（A3既存履歴Comet可視化）のみか、A＋B＋Cを1つの実験計画として設計するか。推奨は**Aの短期の可視化とB/Cの学習・表現Evidenceを別spec/Gateに分ける**。
- Bの実験予算（100〜数百step canary、1epoch試験、従来候補10epochの長期run）。目的・予算に応じて異なる主張になる。
- 履歴のCometアップロードは`run_report.json`由来の事後登録（GPU run不変）を第一候補。オンライン送信失敗でGPU学習が失敗することを避ける。project/name/tags/metric keys + source provenanceを正式固定する必要がある。
- Bで**どこまでLossをbitwise比較するか**。同一fixtureを用いたA3は既にbitwise確認済み。実ImageNet long-runの matched model init/input/augmentation and linear probeを優先する。
- 学習成立の判断（loss以外にどのpositive/negative separation・variance・feature rank/held-out metricを最低条件とするか）と参照baselineの品質。
- GPU空き/データI/O、multi-epoch runtime、Comet upload failureとlocal artifacts retention、環境資格情報の扱い。

## 現時点の方向性

**最有力（未承認）**：既存A3のstep-Loss等をCometで2系列に重ね、次に同一初期stateとDataLoader/augmentation/optimizer契約を守る実ImageNet matched MoCo pretraining比較を別Gateで設計し、特徴表現評価を別Gateで追加する。Cometの記録の成功と、モデルが有用な表現を学習したかは明確に分けて判断する。

研究コード・spec・README・TODO・Comet実験はこのbrainstorm段階では変更しない。ユーザーが明示的にspec化を求めた場合に`research-spec`へhandoffする。

---

## 2026-10-20 追記：Meta公式v2レシピ・10epochのユーザー希望

ユーザーから新たな明示方針：

> 実ImageNetでのStep B（公式Metaと自作の学習挙動をComet比較）を、公式の設定に合わせ、**10epoch**実行したい。

**確定**：対象はMoCo v2、Meta公式と自作の両run、ImageNet-1K実画像、10epoch、Comet Loss等の比較。**未確定**：2GPUでのglobal batch256を維持するか、公式推奨のlinear LR scalingによりglobal64へ調整するか。ユーザー指示は方針であり、実験runや研究コード変更の許可、spec approvedを意味しない。

### 固定公式ソースによる確認

Meta公式clone正本commit：`facebookresearch/moco@a419ef282fc9885015674e77f58c59447206a27f`。

- `README.md`に記載の公式8GPU ImageNet pretrainコマンド：ResNet50、`--lr 0.03 --batch-size 256`。MoCo v2へ切り替えるため`--mlp --moco-t 0.2 --aug-plus --cos`を追加。READMEには4GPUの場合`--lr 0.015 --batch-size 128`というlinear LR scaling推奨あり。
- `main_moco.py`：デフォルトepoch 200、SGD momentum0.9/weight decay 1e-4、moco dim128/K65536/momentum0.999、MoCo v2 T0.2はopt-in、RandomResizedCrop224(scale=.2-1), ColorJitter(.4,.4,.4,.1) p=.8, grayscale .2, GaussianBlur .5, horizontal flip, ToTensor, ImageNet normalization。公式はepochごとに`train_sampler.set_epoch(epoch)`、`adjust_learning_rate(optimizer,epoch,args)`、一epochのloader全体をtrainし、epoch checkpointを保存。`cos` schedulerは`lr0×(1+cos(pi×epoch/args.epochs))/2`。
- **200epochでのMoCo v2 Linear evaluationの公式性能と、10epoch短縮学習の性能を同一視しない**。10epochをschedulerの全期間とする候補を推奨。200epoch schedulerの最初の10epoch切り出しなら別experimentとして区別。

### 公式Global Batchと2GPU調整のblocking choice

| 案 | Global Batch | 2GPU local batch | 初期LR | 適合とリスク |
|---|---:|---:|---:|---|
| **strict official example** | 256 | 128 | 0.03 | 8GPU向け公式global batch/LRをそのまま維持。RTX A4500 20GiB×2でlocal128のVRAMは未測定、OOMの可能性あり。事前の2GPU pilotで実測し、変更には再決定が必要。 |
| **official scaling recipe（運用第一候補）** | 64 | 32 | 0.0075 | 公式READMEがglobal256→128でLRを半減する方針を2GPU/global64へ拡張。既存A2/A3/A5-pilotの検証済みbatch/K/FP32に整合。ただし公式例global256そのものではなく2GPU向け縮小構成と明記。 |

単なるgradient accumulationでlocal32×4をglobal256に見せても、BN統計、DDP Shuffle-BN、Queue更新タイミング、Key EMAの頻度などが変わるため、**同じ公式レシピと断定しない**。global256の実機feasibilityが確定するまで速度・VRAMを推測で保証しない。

### 10epochフルImageNetコストの概算

ImageNet train = **1,281,167画像**（先行ユーザー実測）、2GPU global64でDistributedSampler + per-rank batch32/drop_lastなら`floor(ceil(1281167/2)/32)=20018` update/epoch、**200,180 update/run、Meta＋自作で400,360更新**。A5-pilot測定は**約80.96 images/s（2GPU合計）**で、10epochあたり約`1281167×10/80.96≈158,246 sec≈44h/run`、順次2runなら**約88h**となる。ただし20step pilotの一window値なので、長時間でのNAS/cache/worker/Comet/epoch保存・公式M実装速度に外挿できない。正式budgetは100step→1epoch pilotの実測で再推定。global256を選ぶならupdate数が変わり、A5-pilotのglobal64 throughputをそのまま使用しない。

### 現行コードの拡張境界

`tamaki-lab/2026_09_ishikawa_sequential-video-lora@dev` `scripts/parity/moco_imagenet_parity.py`はMetaReferenceAdapterとSelfAdapterの共通runner、`CometRun`のrank0送信（Loss/positive-negative cosine/Queue pointer/LR、条件付きgrad+update norm）を実装している。しかし`bench_loop`は**両adapter `build(None)`で初期状態共有なし／`set_epoch(0)`と`set_epoch_lr(0)`で学習率更新なし／`run_bench_steps(max_steps)`で1epoch iteratorを消費して終わり／`bench`にはresume/checkpoint禁止**。単純に`training.max_steps`を増やして10epochにしてはならない。`run_kind=parity`は同一canonical fixtureで始まるものの、全step traceを書くA3専用で、10epoch full ImageNetの実用的なデータ管理とログ容量に向かない。`run_moco_gates.py`のA5-pairはComet OFF/自作のみのmonitor負荷比較なので、今回の実ImageNet公式vs自作の目的とは分離する。

新規10epoch専用experiment contractでは、epoch-aware DataLoader/sampler, LR scheduler,同じ初期stateの厳密なマッピング,同じ変換系列とデータ順序,独立fresh run+multi-epoch resume, epoch checkpoints/sidecars, run manifest/hash, local artifactが正本でCometは可視化層、異なるComet run名/タグ・共通metric keys、学習時毎10step Loss/pos-neg cosine/LR、毎100step grad norm、epoch集約の安定性/finiteを推奨。実行順序は短いcanary→1epoch確認→承認した10epochのM/S1。Comet実験keyが異なるものを誤比較しない。

### 検証の主張を区別

1. **数値の正しさ**：A3 synthetic 1025step 58,790/58,790エントリ bitwise PASSは既存の証拠。
2. **実画像10epochの学習挙動再現**：Loss/positive-negative cosine/LR/grad/Queue等の曲線・epoch指標が整合するか。初期state/入力streamが完全一致しなければbitwise一致を判定基準にしない。
3. **有用特徴獲得**：同じ画像・分割を使う学習前→後、Meta後→自作後のImageNet val kNN／後段Linear Probeなどが必須。Loss一致だけでは「MoCoの特徴学習成功」の主張は不可。

### 現在の判断・handoff

**採用（ユーザー）**：Meta公式MoCo v2のrecipeを基準に実ImageNet 10epoch学習を両実装で比較し、CometでLoss等を見たい。

**推奨候補（未承認）**：global64/local32 + LR0.0075（公式linear scaling）、cos horizon10epochs、同一初期state/data order/aug、10step scalar記録、100step grad、0step update clone、epoch checkpoint、事後のComet比較、固定held-out特徴品質Gate。短期pilotと1epochで必要資源を評価。

**blocking decision**：strict公式例global256/LR0.03（2GPU local128の実機VRAM pilot必須）と2GPUスケーリングglobal64/LR0.0075のどちらを「公式に合わせた」と呼ぶか。

**研究コード/既存spec/README/TODO/Comet/研究GPU runは今回変更なし**。ユーザーがspec作成を明示した段階で`research-spec`を使い、正式10epoch実験を実装可能な契約にする。長時間GPU runは別途明示承認。

### 2026-10-20 research-spec handoff（現在のユーザー採用範囲）

新規[Meta公式MoCo v2・Global Batch256・2GPU/local128・LR0.03・ImageNet 10epoch・Comet再現比較のdraft spec](../../../lab/projects/sequential-video-lora-analysis/specs/2026-10-20-meta-moco-v2-official-global256-imagenet-10epoch-comet-reproduction-spec.md)に昇格候補を整理。ユーザーが採用した設定は固定したが、比較入力/公式cuDNN運用/Cometアップロード/held-out表現評価・運用詳細の推奨は承認待ちであり、**specはdraft**。コード実装・VRAM pilot・10epoch・Comet送信はまだ実行していない。本文の探索的候補は引き続き履歴として保持する。
