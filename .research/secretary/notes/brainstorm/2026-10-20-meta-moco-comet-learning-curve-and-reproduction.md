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
