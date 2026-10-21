---
date: 2026-10-22
project: sequential-video-lora-analysis
source_todo: null
topic: vit-moco-g3-collapse-diagnostic-roadmap
status: exploratory
tags: [brainstorm, vit, moco, collapse, lora, full-ft, optimizer, ddp]
---

# G3 10step PASS後の最優先課題：Query特徴量のCollapse切り分け

## 相談の出発点

ユーザーが2026-10-21 ImageNet real image・2GPU×local128（global256）・FP32でViT＋MoCoのFull FTとFrozen ViT＋Q/V LoRA各10stepの検証を実施し、翌2026-10-22に正式GPU Gate集計を提示した。今回の問いは「次にすべきこと」。研究上の狙いはFull FT / LoRAの公正な表現比較を経てActivityNet単一フレーム、さらにSequential Loaderへ進むこと。現行Gateの資源・数値有限性の確認と表現学習の成否を混同しない。

## 確認済みEvidenceと実装

- [G3 10step Evidence](../../../../lab/projects/sequential-video-lora-analysis/experiments/2026-10-21-vit-ddp-moco-g3-imagenet-10step-console-evidence.md): 両modeが10/10step完成、全rank VRAM Gate PASS、`failures=[]`、最終state整合PASS、両modeの全10step入力hash一致。Peak reserved/rankはLoRA 12338 MiB、Full FT 18346 MiB。Full FT headroom 1749.875 MiB（Gate閾値1024MiB）。ローカル `run_report.json`原本はChatGPTから直接アクセスしておらず、ユーザー提出のローカル集計に基づく。
- `query_feature_std` は研究コード [runner](https://github.com/tamaki-lab/2026_09_ishikawa_sequential-video-lora/blob/dev/scripts/vit_moco/moco_vit_ddp_smoke.py) `out.q.detach().std(dim=0).mean()`。 `out.q` はProjectorの出力にL2 normalizeを施した128-d Queryであり、ViT CLSの分散ではない。Step1の値は双方0.07224058。Step2でLoRA 0.00634409 / FT 0.00081159、Step10でLoRA 0.01095892 / FT 0.00000606。FTで極端な正規化Query均一化の疑い。
- `train_step` はforward → backward → AdamW optimizer.stepの順。 `step2` のforwardはstep1の更新後であり、「学習更新が原因」の仮説に整合。ただしstep2の入力はstep1と異なるため、同じ固定評価ミニバッチのstep0/step1更新前後を測らなければ厳密に切り分けられない。
- [承認済みspec](../../../../lab/projects/sequential-video-lora-analysis/specs/2026-10-21-vit-ddp-moco-full-ft-lora-switch-smoke-spec.md)のGPU固定条件：ViT-B/16 pretrained、AdamW LR0.03 WD1e-4、warmupなし、Cosine horizon5epoch、2GPU local128、global256、FP32、K65536/m0.999/T0.2。GPU runnerはmax_steps 1/10/20のみ。GPU recipeを無断変更しない。
- 現在 `checkpoint_steps: []`（[Hydra config](https://github.com/tamaki-lab/2026_09_ishikawa_sequential-video-lora/blob/dev/conf/moco_vit_ddp_smoke.yaml)）。既存10stepの学習後重みは保存されておらず、オフラインでそのforwardを再演して任意の層を観察することは現時点ではできない。
- 2026-10-15 [MTG](../../../../lab/projects/sequential-video-lora-analysis/meetings/2026-10-15-mtg.md)では複数要素を一度に変更せず、LoRA・MoCo・Sequential Loaderを段階的に接続する方向。2GPU G3の工学PASSは研究性能の成功条件ではない。

## 中心となる問い・原因仮説

**問い**: 更新1回後に何がほぼ定数特徴を出し、Full FTでより顕著なのはなぜか。先にどこまでdiagnosticを行えば、次の20step / ActivityNet / Full FT対LoRA表現比較の判断ができるか。

- H1 **AdamW LR=0.03が事前学習済みViT/Projectorに高すぎる**（最有力、未確認）。SGDでのLR数値をAdamWへそのまま当てはめたことの再検討が必要。同一LRは再現性の比較には有効でも、各方式に適した探索をした場合の公正な性能順位と同一視できない。
- H2 **Projectorのみが崩壊**し、CLS空間の識別性は残っている。両mode共通の学習対象がProjectorであることから有力。
- H3 Full FTではCLSも崩壊し、LoRAにはその制約がある程度抑制的に働く。これを直接示すCLS feature実測はまだない。
- H4 Query vs EMA Keyの非定常性、初期ランダムQueueとoptimizer/InfoNCEの相互作用。
- H5 入力／拡張の異常。全10stepのFull FT/LoRA入力hash一致は**両run間比較**であり、入力内容が学習上望ましい多様性を持つこと自体は保証しない。

## 発散：選択肢

1. **A（推奨）：固定評価バッチでpre/post updateを分けた多段特徴量診断（新規診断spec候補）**。Fresh seed0の元レシピを基準とし、同一probe minibatchでstep0（optimizer前）、step1 optimizer後、step2/10付近を評価。Query ViTのCLS出力 `[B,768]`、Projector raw `[B,128]`、L2-normalized `q[B,128]` を**別指標**で比較。rank別と集計、batch内 pairwise cosine off-diagonal mean・p50・p95、feature per-dim std、embedding normの中央値・min、非有限値、CLS単位ベクトルcosine、同じprobeの距離、query/key類似を記録。Projector BatchNorm等のモード依存を固定し、学習時の測定/推論時固定probeを分ける。GPU peak/headroom guard保持。
2. **B（A後推奨）：LR切り分け**。元の0.03を基準に、別specで候補の低LR（例 3e-4、1e-4等、未採用の仮値）を少数step・同一seed/入力/augmentation/precision/global batchで比較。採用候補を最初から別の正式本学習recipeと混同しない。必要ならprojector-only/frozen-backbone ablationを設けてCLS vs Projector原因を区別。各runのinput hashesとoptimizer step deltaも測る。
3. **C：現行固定レシピの20stepを先に実行**。G3工程の完了は早いが、FT step10 Query feature std約6e-6であり、追加10stepでcollapse原因は特定できない。資源余裕もFTは閾値に近い。最優先にはしない。20stepは「元レシピの工学Gate完了」を重視する場合に限り、停止境界と研究的非成功を明記して選ぶ。
4. **D：いきなりActivityNet/Sequential Loaderまたは5epoch比較**。非逐次画像の崩壊原因が不明なまま変数を増やすため現時点で保留。
5. **E：現在のloss低下や正負類似度だけを根拠に学習成立とみなす**。特徴分散の異常を説明できないため棄却寄り。

## 収束：提案順序と事前Gate（未承認）

### Gate D0：観測設計の決定・spec化候補

- 原因が **(i) CLS（backbone）、(ii) Projector raw、(iii) normalized q、(iv) Key/Queue** のどこかを、同一入力に対する時点比較で分離できる計測をまず仕様化。
- 同じ前処理、同じ事前学習重み、同じseed、同じprobe minibatchを使い、学習中のランダムview分布との差も保持。
- `optimizer.step()` の前後の各trainable parameter groupのupdate norm/parameter norm、ViT attention Q/V/MLP とProjectorを分けた傾向、勾配の有限値を測る。AdamWでgrad normだけではupdate量を推定しない。
- 保存済み10stepにはcheckpointがなく、**既存ログから後知恵でCLS/Projectorを計算することはできない**。再現runが必要。追加instrumentationがGPUメモリを圧迫しないよう、detach、計測区間の整理、必要最小限のrecording、DDP collectiveを考慮。
- 工学合格と表現非collapseを別gateに定義し、数値閾値は診断指標/基準分布から決定する。raw stdが減少しただけで事前に固定した絶対閾値なしに「正式collapse」とは判定しない。

### Gate D1：短時間diagnostic（実装・GPU実行は別許可）

- まず従来LR 0.03・Full FT / LoRA・seed0・各1〜2更新を持つfresh runで発生局所を確認。元G3の20stepは未実行のまま保留。
- Projector前CLSが保たれ、後で落ちるならProjector優先でupdate/LRを調査。CLSから低下するならpretrained ViTの更新ステップ幅、正規化、optimizerのパラメータ群、LRを優先。norm後だけ異常ならraw feature normやnormalizationの数値安定性を調べる。
- 同じ入力hashとRankごとのmetricsを保存し、vram_gate PASS・normal DDP final checksを必須とする。

### Gate D2：低LRなどの因果的比較

- ユーザーが採用/承認した場合に限り、LR低下候補を別recipeとして試す。pretrained ViT CLSとnormalized qがfixed probeで変動を保ち、finite/VRAMが成立するかを比較する。
- 少数stepでのcollapse回避だけでは性能良好とは言えず、正式な非逐次Full FT / LoRA表現比較（線形プローブ、必要ならkNN、共通評価分割、複数seed、コスト評価）はさらに別Gate。

### その後のロードマップ

問題切り分け・レシピ改訂判断 → 短時間20step sanity → ActivityNet単一フレーム（抽出間隔のフレーム類似度診断も別課題） → 画像の非逐次Full FT/LoRA下流特徴評価 → Sequential Loader統合 → 時間情報評価・LoRA特異値解析。

## 未解決点・反証可能性・引き継ぎ

- 正規化Queryのstd低下は本当にCLR/Projectorのcollapseか、入力viewサンプリング/BNモードの影響か。固定probeとpre/post updateの直接比較が未検証。
- LayerNorm/BatchNorm（Projector内）のtrain/evalモード切替で統計が変わる可能性。比較は同一モードで行い、学習本体のstateを評価probesで変更しない。
- Full FTとLoRAの性能公平性：LRを揃えるcontrol比較と、各方式の探索予算を揃えたtuned比較は別物。
- 20step工学Gateを先に通す選択自体は不可能ではないが、現状では表現崩壊の疑いを解決しないため優先度を下げる。
- このbrainstormは**探索文書**であり、承認済みspecの書換え、GPU学習許可、研究コード編集、TODO確定ではない。ユーザーがspec作成を明示依頼した場合は現在のresearch-spec skillでGateを判定する。

## 関連文書

- [G3 10step Evidence](../../../../lab/projects/sequential-video-lora-analysis/experiments/2026-10-21-vit-ddp-moco-g3-imagenet-10step-console-evidence.md)
- [研究project README](../../../../lab/projects/sequential-video-lora-analysis/README.md)
- [採用済み2026-10-21 ViT DDP spec](../../../../lab/projects/sequential-video-lora-analysis/specs/2026-10-21-vit-ddp-moco-full-ft-lora-switch-smoke-spec.md)
- [過去のG3へのロードマップ](2026-10-21-vit-moco-post-implementation-g3-roadmap.md)
- [Code runner](https://github.com/tamaki-lab/2026_09_ishikawa_sequential-video-lora/blob/dev/scripts/vit_moco/moco_vit_ddp_smoke.py)


## 2026-10-22 01:39 JST：research-specへの引き継ぎ（draft保存）

ユーザーの「Collapse診断用のspecを作成したい」という明示依頼に基づき、[ViT MoCo Query Collapse診断spec（draft）](../../../lab/projects/sequential-video-lora-analysis/specs/2026-10-22-vit-moco-query-collapse-diagnostic-spec.md)をresearch-specにより作成した。

- 探索案から、固定64枚probe・決定的前処理、ViT CLS / Projector raw / L2 normalized Queryの3段、step0/1/2/10の時点比較、元G3 recipe保持、CPU GateとGPU Gate分離、元runnerへのoptional instrumentation案を**承認候補**として移した。
- Spec statusは`draft`。B1〜B6の選択肢は**ユーザー未承認**、採用済み条件を上書きしない。
- 研究コード変更・GPU run・Git code commit/pushは実行も許可取得もしていない。
- 本brainstormは探索史料のまま維持する。判断・承認後の実装契約は上記specを参照する。


## 2026-10-22 03:55 JST：D0〜D3報告後のD4移行前Gateとremote状態

### 入力と確認済み根拠

ユーザーは、Collapse診断を承認済みspecどおり実装しCPU D0〜D3 PASS、D4 GPUは未実施、コード・Research Workspaceのcommit/push未実施と報告し、「次にすべきこと」を質問した。以下、**研究サーバ作業ツリーの状況はユーザー報告のみ**であり、GitHubの公開済みbranch状態とは分ける。

2026-10-22のGitHub再確認では、**ユーザーの報告作成時点よりも新しいか同時期のremote更新が観測**された。

- 研究コード `tamaki-lab/2026_09_ishikawa_sequential-video-lora@dev` HEAD `312c121975d68e860d5f89bf38ed26aab23b0981`。前HEAD `2c269915f0c5036fa2a460959fdf401d54456026` から1 commit進み、**新規3ファイル＋runner1ファイルの同じ4ファイル**が変更された。 [commit](https://github.com/tamaki-lab/2026_09_ishikawa_sequential-video-lora/commit/312c121975d68e860d5f89bf38ed26aab23b0981)
- Research Workspace `haruto2919/research-workspace@main` は `5f8cb803aacc10724b9f116d1e8ca6b22e68ee9a`。1 commitで`README.md`、diagnostic spec、CPU Evidenceが変更・作成済み。 [commit](https://github.com/haruto2919/research-workspace/commit/5f8cb803aacc10724b9f116d1e8ca6b22e68ee9a)
- したがって、**既にGitHubに存在する変更を重複commit/pushしない**。ただしGitHubが進んだことを**研究サーバのローカルcloneへ反映済みとは解釈しない**。`git fetch origin; git status --short --branch; git rev-parse HEAD; git rev-parse origin/dev; git diff --stat origin/dev` で、実行するローカルtreeがremoteと一致しcleanか確認する。dirtyなら勝手にreset/pull/stash/上書きしない。
- 現在remoteの`conf/moco_vit_query_collapse.yaml`と`scripts/vit_moco/query_collapse_diagnostic.py`を確認。2GPUのGPU runは `--config-name moco_vit_query_collapse model.train_mode=lora|full_ft training.max_steps=10`。比較CLIは `python -m scripts.vit_moco.query_collapse_diagnostic <lora_dir> <full_ft_dir> [--output new.json]`。

### 次のGate・判断候補

1. **最優先（採用を推奨、追加のGPU許可を意味しない）**：研究サーバ上のコードの`HEAD`・`origin/dev`・dirtyを照合。既存remote commitに差分が入っているなら重複commitしない。Research Workspace cloneも同様にremoteとの差分をチェックし、衝突を保護。
2. **B6個別許可を待つ**：D4は研究サーバのGPU 0,1の空き、NAS ImageNet read access、clean commit、現在の2GPU構成を直前確認。旧Full FT headroom `1749.875MiB` は1024MiB閾値より726MiB程度しか大きくなく、追加診断でOOM/VRAM Gate BLOCKEDの可能性。自動AMP/batch/K/解像度変更は不可。
3. **承認後のD4候補**：LoRA fresh10step→その結果の`run_report.json`と`diagnostic_summary.json`が工学PASSならFull FT fresh10step。事前学習・transform・seed・optimizer LR0.03・queue・global batch・FP32は承認spec固定。両方のrunで `engineering_status=PASS`、Rank state、probe0/1/2/10のCLS/z_q/q、入力hash、`probe_memory`を確認。
4. **比較の分離**：比較CLIの`comparable=true`は訓練view SHA、probe入力、step0特徴一致の判定であり、**両runの`engineering_status=PASS`を意味しない**（比較コードで独立）。両方の確認が必要。
5. **D5 interpretation**：固定64枚の`mean_dim_std`、offdiag cosine p50/p95、normとstep0からのdrift、update_metricsを層別に整理して、CLS / Projector / normalizationのどこで収縮したかを確認。警告閾値は科学的collapseの正式証明ではない。AdamW 0.03原因説、低LR comparison、Full FT vs LoRA性能優劣はD4だけでは未検証。
6. GPU runがBLOCKEDなら、その結果もEvidenceとして保存し、レシピを勝手に変更せず後続のspec/承認を待つ。ActivityNet/Sequential Loader/20stepはここでは優先しない。

### 対象外・未確認

- 今回は相談のみ。CPU/GPU run、code編集、研究コードcommit/push、既存spec/README/実験結果のstatus更新は行わない。GitHub上に現れたcommitの作成主体・同期対象local worktree・未commit差分は現在未確認。
- 次のユーザー判断：**D4 GPU実験を個別に許可するか**。これは現行approved specや過去G3許可から推定しない。今回のNoteはexploratoryであり、GPUの実行契約を変更しない。
