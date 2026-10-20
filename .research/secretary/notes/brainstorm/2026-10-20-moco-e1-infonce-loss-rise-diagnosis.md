---
date: 2026-10-20
project: sequential-video-lora-analysis
source_todo: null
topic: "ImageNet E1 100stepでMeta公式と自作のInfoNCE lossが一致しつつ上昇する理由"
status: exploratory
tags: [brainstorm, moco-v2, imagenet, loss, queue, self-supervised-learning]
authority: "User asked for causes; no new experiment changes approved."
---

# E1（100step）InfoNCE loss上昇の原因候補と診断

## 出発点・参照した現在の文脈

ユーザー質問：「100ステップではlossがどちらも上昇しているんだけど原因として何が考えられますか．その際，結果は両方とも一致しています．」

- [E1実測Evidence](../../../lab/projects/sequential-video-lora-analysis/experiments/2026-10-20-meta-moco-v2-e1-imagenet-100step-comet-parity-evidence.md)：ImageNet train、Meta公式/自作各100step、`contract_valid`・`both_completed`・`metrics_bitwise_equal` PASS。サンプリング入力hash 2/2、一致指標Loss 11/11、query grad norm 2/2。100stepはpartial epoch、`epoch_states_identical=None`。E2/E3は未許可。Comet project `haruto2919/moco-imagenet-v2-reproduction`。
- [採用spec](../../../lab/projects/sequential-video-lora-analysis/specs/2026-10-20-meta-moco-v2-official-global256-imagenet-10epoch-comet-reproduction-spec.md)：Meta MoCo v2、2GPU global256、K65536、T0.2、m0.999、ResNet50+MLP128、SGD LR0.03、10epoch Cosine horizon、FP32、同じ初期fixture。V1/E1工学確認と10epoch/held-out表現評価を分離。
- [2026-10-15 MTG](../../../lab/projects/sequential-video-lora-analysis/meetings/2026-10-15-mtg.md)：初期random Queue置換中のloss上昇はあり得る、しかしloss/positive cosineの挙動から特徴獲得の成立は未検証、公式基準を取得して判定する。
- [先行brainstorm](2026-10-20-meta-moco-comet-learning-curve-and-reproduction.md)：数値再現、学習曲線、表現の有用性は別のclaim。Lossが単調に下がることは100stepのE1 PASS条件ではない。
- 公式コード：`facebookresearch/moco@a419ef282fc9885015674e77f58c59447206a27f:moco/builder.py`：初期Queueは`torch.randn(dim,K)`後に各列正規化、`forward`は`l_pos=q·k`と`l_neg=q·queue`を連結して`/T`、当stepのlogits算出**後**にglobal batchのKeyをenqueue。CE target=0。
- 自作runner `tamaki-lab/2026_09_ishikawa_sequential-video-lora@d6f7acc:scripts/parity/moco_imagenet_10epoch.py`：同じMetaAdapter/SelfAdapterを使う、loss、positive cosine、negative cosine、hardest-negative、Queue pointerを10step間隔（step1/100含む）で記録。平均negative cosineは`logits[:,1:].mean()*T`、positiveは`logits[:,0].mean()*T`（`scripts/parity/moco_imagenet_parity.py:775-779`）。対応したQuery/Key特徴の単純cosineではなくprojector後の正規化特徴とInfoNCE Queueの比較である。

## 確認済み事実（添付E1出力・現在の公式/研究コード）

| 指標 | step1 | step100 |
|---|---:|---:|
| InfoNCE Loss | 6.441793441772461 | 9.806382179260254 |
| positive cosine | 0.9494454383850098 | 0.7111494064331055 |
| mean negative cosine | -0.00013514384627342225（同初期fixture V1のstep1観測値） | 0.24109346866607667 |
| queue pointer（after enqueue） | 256 | 25600 |
| query grad norm | 10.994661...（同初期fixture V1のstep1） | 0.12952577874793422 |

E1のstep1 lossはE1 Evidence記載。step1 positive/negative/gradは同レシピ同固定seedで取得されたV1 step1を併記したもの。厳密なE1固有のstep1メトリクスは現時点で同E1 evidence上で直接列記されていないため、**独立runの数値を同一runの直接観測と混同しないこと**（V1/E1が別campaignだが同初期状態を再生成した前提）。E1のより確かな比較にはE1 `metrics.jsonl`を現物で読む。E1比較器が示すMeta vs自作 sampled scalar は全項目 bitwise equal。

K65536を256 keys/stepで更新するので、100step終了時点の実画像Key置換率は25600/65536=39.0625%。100stepのlogitsを計算する**直前**のQueue置換率は99*256/65536=38.671875%。**同時に約61%が初期ランダムKeyのまま残っている**。256step enqueueで全列が実画像Keyに置き換わる。

InfoNCE式：`L = -log(exp(s_pos/T) / (exp(s_pos/T) + sum_{j=1}^{K} exp(s_neg_j/T)))`、T=0.2。仮に全候補が同じlogitなら一様予測のcross-entropyは`log(65537) ≈ 11.09`。この値はcollapseの判定閾値ではなく理解用ベースラインであり、loss単体で特徴分布を確定しない。

## 解釈・原因候補（優先度順）

1. **Queue warm-upによる負例分布の非定常性（最有力、コードで機序確認）**：初期random単位ベクトルQueueは特徴に対してほぼ無相関で区別しやすい。各stepで実画像由来、同種のencoderからのKeyと置換され、正例との区別が相対的に難しくなる。`mean negative cosine`が上昇した点も整合。固定された負例集合に対するlossではなく「毎step異なる負例集合」であるため単調減少は期待できない。
2. **Positive Cosine低下（観測）**：step100で0.711となり、step1に比べて正例のlogitが下がる。QueryがSGDで変化する一方、Key encoderはm=0.999のEMAで追随が遅れること、2-view augmentationが強いこと、初期の表現が変化することなどを**原因仮説**として区別。各要因の寄与は未分解。
3. **実画像Key同士の高い類似（潜在的表現集中/難負例）**：単にQueueが埋まるだけでなく、real-key間で高いcorrelationがあればInfoNCEのlog-sum-expは増大。max/hardest negative、random部分vsreal部分の分布、特徴variance/effective-rankを分析してcollapseの兆候か、自然なQueue warm-upかを分ける。高い平均負例類似だけではcollapseは断定不能。
4. **最適化条件/augmentation/EMAなど**：LR0.03、強いMoCo v2 augmentation、温度T0.2、projector構造、BN、EMA追随性によって初期学習曲線は振れる。公式と自作に共通の結果なので自作固有バグより共通設定の影響が第一候補。ただし両runの合致は有用特徴の学習成立を証明しない。
5. **実装/ロギング異常の代替説明**：別々の公式/自作runが一致したことは強い工学的Evidenceだが、学習不適切な共通条件や全テンソルstate未比較の可能性を排除しない。ログ上`finite`とQuery gradを確認し、原因未確定を維持。

Logitの平均差は`positive-negative`のcosine marginが約0.95→約0.47に狭まる方向で、T0.2によりlogit差は約2.4小さくなる。実Lossの上昇量は約3.36。これは説明の手掛かりであり、CEは`logsumexp`を含み**平均類似度だけから正確なloss増加を再構成できない**。

## 推奨する低コスト診断（研究案であり未承認）

- **今あるE1ログ（GPU追加不要）**：11件の記録stepについて `loss`、`positive_cosine_mean`、`negative_cosine_mean`、`hardest_negative_cosine_mean`、`queue_ptr/K`、LR、grad normを同じstep軸で可視化する。Step30→100でqueue占有が増すときlossと正例/負例cosine差が連動するかを見る。
- **消極/積極判断**：Queueが完全実Keyへ置換された後、例えばE2で256step以降のloss・cosineの傾向を見る（E2の別GPU許可が必要）。数百〜数千stepでlossが高止まり・positiveが低下し続けるなら、feature variance、embedding平均ペアcosine、norm/effective rank、batch diversity、LR/augmentation強度、EMA driftを診断する。
- **情報を切り分ける案（将来の追加計測候補）**：Queue列を実画像Key由来と初期ランダム由来で分け、それぞれのnegative log-sum-exp / 平均/上位分位を計測。固定negative bankでの同条件診断は実際のMoCo学習の契約を変えるため、独立診断としてのみ扱う。
- E2の1epoch checkpoint/stateの比較とE4のheld-out診断は別Gate。E1ログ一致を下流識別性の証拠と混同しない。Queue一巡で必ずlossが下降すると予言しない。

## 現時点の方向性・未解決事項

- **最有力**：Queue warm-upと負例難度変化。
- **同時に重要**：Positive Cosineの低下（Query/Key差、Augmentationやoptimizer起因は仮説）。
- **反例**：Queueが実画像Keyで一巡した後も、正例と負例の分離が改善せず高いlossが継続するなら、初期非定常性だけでは説明不足。
- **判断保留**：E2のGPU実行、追加instrumentation、LR等の変更。ユーザーの現在の依頼は原因説明であり、GPU実行承認やspec変更ではない。
- **次**：E1の手元11点のCosine/Hardest/Queue推移を確認。必要なら別承認を得てE2などの長時間runに進む。研究コード・spec・README・TODOは更新しない。

---

## 2026-10-20 17:55 JST追記：query_grad_norm低下はLoss停滞の原因か

ユーザーは「途中からquery_grad_normが急激に下がってしまう（x:100）。これのせいでlossが下がらなくなるのでは」と懸念。E1実測Evidence、現在`dev@d6f7acc`の`conf/moco_imagenet_10epoch.yaml`と`scripts/parity/moco_imagenet_10epoch.py:479-499`、`moco_imagenet_parity.py:559-560,1116-1132`、公式MetaのSGD/Queue実装を再照合。

### 事実

- E1の100stepは`loss 9.806382179260254`、`query_grad_norm 0.12952577874793422`、`positive_cosine_mean 0.7111494064331055`、`negative_cosine_mean 0.24109346866607667`、`queue_ptr 25600`。`lr=0.03`はepoch0の間固定。
- V1（同レシピ/seed、別run）のstep1`query_grad_norm=10.994661049828359`。E1のstep1実測ログをGitHubからは直接取得できないため、この2点の約85倍差は「V1 step1 vs E1 step100の**参考値**」として扱い、まずE1自身の`metrics.jsonl`からstep1と100を確認すべき。
- `monitor.grad_norm_every_steps=100`、`monitor.log_first_step=true`、`monitor.log_last_step=true`。`BenchMonitor.is_due('grad_norm')`によりE1は**step1/100だけ**を測る。**100step目で急激に下落したという時系列証拠はない**。
- `query_grad_norm`は`train_step`内で`loss.backward(); optimizer.step()`後に`p.grad`を対象Query parameters（Meta:encoder_q、self:backbone_q+projector_q）から収集し、tensorごとのdouble L2 normを合成した一つの`L2`。SGD stepは`p.grad`自体を消去しないため、更新直前の勾配を参照する。`optimizer.zero_grad()`は次step前に呼ばれる。
- Query SGDは`learning_rate=0.03,momentum=0.9,weight_decay=0.0001,nesterov=False`。実際のparam deltaは現stepの生gradientだけでなく、運動量bufferとdecayに依存。`monitor.update_norm_every_steps=0`なので、E1は**実パラメータ更新normを記録していない**。低gradient_norm＝無更新は**未実証**。
- MoCo InfoNCEは可変の負例Queueであり、Lossのstep単調減少は期待しない。負例1巡はstep256。E1 step100は初期ランダムQueue置換39%に過ぎない。ある時点でLossが高く勾配が小さいことは、計算グラフ上の勾配キャンセル/正規化/Query-Key不一致/特徴集中の仮説を含むが単独の証拠ではない。
- 公式Metaと自作がE1のsampled scalarをbitwise一致（grad 2/2）。よって自作固有の異常は考えにくいが、共通のレシピで学習が成立した保証はない。
- 2026-10-20に更新された採用specとREADMEは**E2 1epoch Meta/self+Cometの個別実行が承認済み・未実行**と記録。前日の探索メモの「E2未承認」等は最新の採用履歴へ従って読み替える。今回の議論はE2の自動実行指示・runner変更許可ではない。

### 原因仮説・重要度（確定ではない）

1. **疎なサンプリングと初期高gradientの見かけ**：100stepだけを見て「ここで急落」と結論づけられない。Step1初期値が異常に大きく、早期に下がっただけかもしれない。検証にはstep10/20/...の測定が必要。
2. **Query/Key/Queueの学習初期過渡現象**：正例cosineは低下、実画像key投入でnegativeが難化。対照タスクの難度上昇はLoss上昇を説明するが**grad norm低下まで必然的に説明するわけではない**。EMA keyのlagなどを追加検証。
3. **高Lossでもgradient消失・キャンセル/表現集中の可能性**：正規化後query embeddingでの`∂L/∂q=(Σ_i p_i k_i−k_pos)/T`（softmax確率を含む。query normalization前・ネットワークparam勾配ではない）。query embeddingのweighted negative keyとpositive keyが近い場合やネットワークJacobiansが小さい場合、large CEでもparam gradient normが下がり得る。単純なsoftmaxの「lossが大きいから全勾配がvanish」は誤り。
4. **更新はまだ動いているかもしれない**：SGD momentum/weight decayによりstep100のraw gradientとは異なるparam deltaが続く。E1`query_update_norm`はOFFなので`0.03×0.1295`を実際の重み変化量として報告しない。
5. **feature collapseや学習停滞が本当か**：rank0 global normの単一値ではlayerwise variation/feature varianceを見落とす。数百step以降のloss/positive-negative margin、feature variance、backbone/projector別grad・update、momentum buffer、EMA driftを観測。Step100 positive .711 > mean neg .241であり、完全collapseは示されていないが、部分的特徴集中は否定できない。

### 低コストの切り分け候補（未採用の案）

1. GPU追加なし：実機E1 Self run`metrics.jsonl`でstep1のgradを直接確認、step1→100の11点のLoss/positive/negative/hardest/QueuePtrを表・グラフで照合。`query_grad_norm`の2点間は`not recorded`として補間しない。
2. 計測密度を上げる場合は、**別途合意のうえで**`monitor.grad_norm_every_steps=10`、`monitor.update_norm_every_steps=10`等の観測だけを変更した比較runを検討。更新normは監視overheadが増え得る。実行条件・比較契約・元E1を上書きしない。
3. 現在E2の1epoch実行自体は最新specで承認済みだが、既定のgrad計測は1/100/200/...しか得られない。E2を走らせる場合はstep256 Queue一巡前後、100以降のgradientの持続低下やLoss頭打ちを見る。**E2の既定計測条件は黙って変えない**。異常が深刻なら診断用canaryを優先するかは次の判断。
4. 詳細機序の診断案：backbone/projector別grad norm vs param norm、SGD optimizer momentum buffer norm、query param update norm、`||q-k||`とcosine margin、実Key/ランダムKey別難例分布、固定heldout特徴分散。独立計測でなければ学習の正本設定は変更しない。

### 現段階の結論

**「勾配低下のため学習停滞」は検証可能だが未確認**。最重要の誤読はstep100での**急落タイミングを観測したと扱うこと**。現時点ではQuery生勾配2点のみで、optimizer delta未測定かつQueue入替えによる非定常Loss。同じ値がMeta/Selfに出たことはmechanics再現の工学Evidenceである一方、モデル品質の保証ではない。E2の承認状態と最新specを尊重し、追加runやmetric設定変更を勝手に行わない。
