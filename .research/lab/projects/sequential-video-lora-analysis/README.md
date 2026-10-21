---
project: sequential-video-lora-analysis
status: active
summary: 2026-10-22 ViT＋MoCo Query Collapse診断：CPU D0〜D3とImageNet 2GPU D4両mode工学PASS、全学習入力と固定probeのstep0特徴が一致。Full FTはstep1でCLS cosine中央値0.032→0.985、LoRAのCLSは維持。Projector出力は両modeでstep1に方向集中（cos p50 LoRA0.993/FT0.99994）、両modeのProjector初回相対update約1.44。LR0.03原因は未検証、次は対照spec検討。
implementation_root: /mnt/HDD12TB-1/ishikawa/2026_09_ishikawa_sequential-video-lora
created: 2026-09-04
last_updated: 2026-10-22
---

# 動画の逐次学習によるLoRAの獲得情報の解析と活用

## 概要

画像で事前学習されたモデルに動画を逐次的に入力し、LoRAを用いて追加学習することで、画像情報だけでは捉えにくい時間的・動的な情報を獲得することを目指す。学習されたLoRAの内部を解析し、背景や物体などの静的情報と、動きや時間変化に関する動的情報がどのように保持されているかを明らかにする。さらに、得られたLoRAをVLMや動画生成などの下流タスクへ活用する方法について検討する。

## 実装環境

- メイン実装フォルダ: [`2026_09_ishikawa_sequential-video-lora`](../../../../../2026_09_ishikawa_sequential-video-lora/)
- 用途: このプロジェクトのコード編集、動作確認、学習・評価の実行
- 研究文脈の正本: このREADMEと、同じプロジェクト配下の `specs/`、`experiments/`、`materials/`、`meetings/`

コード変更はメイン実装フォルダで行い、方針、実験条件、結果、意思決定など継続的に参照する研究文脈は、このResearch Workspace側へ記録する。

## 現在の状況

第一目標として、再利用可能なシーケンシャルデータローダを独立リポジトリに整備し、逐次入力に対してLoRAのみを学習するファインチューニングを動作させる。林さんが分離したloaderを利用できること、およびloaderの出力をMeMViTの入力形式へ変換できることを確認した。別の最小構成では、全16 blockのattention q/v（計32 module）へのLoRA注入と、pretrained baseを固定した1 step更新も確認済みである。

一方、loaderからMeMViT、LoRA更新までの本実装への統合と、frame index・target・supervision mask・annotationの対応、chunk境界の状態保持、sequence切り替え時のresetは引き続き検証が必要である。最小構成の動作確認だけから、LoRAが時間情報・動作情報を獲得したという結論は出さない。

2026-09-23には別経路のStage 2として、50Salads `train1` の先頭16 frameを `sequential_loader` から取得し、frozen ViTのCLS feature列 `[16,768]` へ変換するsmokeを確認した。valid featureの有限性とbackboneの学習可能parameter数0を実データで、padding行のゼロ埋めとtimestamp逆行時の順序保持を人工データで確認した。ActivityNet対応とViT-LoRA学習はこの段階では未実施。

同日、その後の承認に基づき、独立loaderリポジトリの`ActivityNet` branchへActivityNet Adapterを追加した。既存Coreを変更せず、全183テスト、実データのsplit件数`10024 / 4926 / 5044`、`.mp4/.mkv/.webm`各1本の先頭16frame読み出しを確認した。詳細は[実装・短時間検証記録](experiments/2026-09-23-activitynet-adapter-verification.md)を参照。この段階ではActivityNet→ViT接続とLoRA学習統合は未実施。

同日、Stage 3としてActivityNet `training`の先頭16 frameから、frozen ViTとmasked meanで有限な`clip_feature [768]`を生成する経路を実装した。実データCPU smoke、padding除外・順序不変性・gradient保持の新規25テスト、50Salads 1件・Hydra 38件の回帰テストが成功。既存ViTテストの期待引数不一致1件は変更前後で同一。詳細は[Stage 3実装・検証記録](experiments/2026-09-23-activitynet-clip-feature-verification.md)を参照。ViTの学習可能parameter数は0で、LoRA・MoCo学習と時間情報獲得の評価は未実施。

2026-09-24、承認済みStage 4としてViT Q/V計24 moduleへのPEFT LoRAを実装し、ActivityNet先頭16 frameのCPU smokeで1-step更新を確認した。学習対象は294,912 parameters / 48 tensors、base変更0、LoRA変更24 tensors、勾配・更新後parameterは有限。新規22件・既存65件のテストが成功した。既存のpooler無効化と対応テスト修正も保持して検証済み。詳細は[Stage 4実装・検証記録](experiments/2026-09-24-vit-lora-one-step-verification.md)を参照。engineering-only lossでの経路確認までであり、MoCo・複数step学習・時間情報獲得の評価は未実施。

同日、Stage 5としてViT-LoRAへMoCo v2-styleのQuery / Key / Projector / FIFO Queue / InfoNCE / EMAを追加した。実ActivityNetの異なる2動画を使うCPU 1-step smokeで、Query LoRA・Projector更新、Key EMA、両base不変を確認。same-sequence negative除外を含む新規38件・既存87件のテストも成功した。詳細は[Stage 5実装・検証記録](experiments/2026-09-24-stage5-moco-one-step-verification.md)を参照。複数step学習、性能、時間情報獲得の評価はこの段階では未実施。

同日、Stage 6Aのconsumer側4-stream round-robinとKey-only warm-up、multi-step MoCo更新・監査を実装した。新規42件・既存125件のテストが成功し、人工データで10/100-stepと実ViT/PEFT接続を確認。MTG時点では実ActivityNetの100-step結果は未確認だったが、その後、AIを用いた実行で10-stepと別fresh processの100-stepがCPUでPASSした。Queueは4→14 / 4→104、Baseは不変、Query LoRAとProjectorは更新された。これらはAIを用いて実行した範囲の観測結果であり、実装全体の正しさには現段階でも確証がない。表現性能・動的情報獲得の証拠でもない。詳細は[Stage 6A実装・短時間検証記録](experiments/2026-09-24-stage6a-moco-multistep-verification.md)を参照。実装commitは`8304d033`。

2026-09-24のMTGでは、MoCoの過去Keyを逐次入力でnegativeにする妥当性、画像MAE重みをVideoMAEへ移してearly fusionを試す案、学習済みLoRAの下流タスク・linear probeおよび特異値・層別変化の解析を議論した。現行のActivityNet + late fusion経路はAIを用いて確認を進めているが、実装の正しさは未確証。次週に確認結果と残る不確実性、LoRA学習の根拠を報告する。詳細は[9月24日の議事録](meetings/2026-09-24-mtg.md)を参照。

2026-09-28、承認済みspecに従い、Stage 6AとStage 6Bを共通Streaming MoCo engineと3軸protocolへ統合した。Stage 6Bは1動画のstrict-single、valid frameのGBR→horizontal flip、FIFO内のsame-video past negativesを使用する。既存167件・新規75件のCPUテストが成功し、Stage 6A基準loopとの人工10-step数値一致、Stage 6Bの実ViT/PEFT短時間integrationを確認した。実装は未コミット。Stage 6Bの実ActivityNet runと表現性能評価は未実施で、これらのテストは実装全体の正しさや時間情報獲得を保証するものではない。詳細は[共通Streaming MoCo実装・検証記録](experiments/2026-09-28-shared-streaming-moco-verification.md)を参照。

2026-10-08のMTGでは、MoCo lossの振動とQuery LoRA勾配の途中からの急増を確認し、原因が未特定のためLinear Probeの解釈を保留した。Sequential Loader、LoRA、MoCoを一度に統合した状態を見直し、LoRA単体・MoCo単体を小さなステップで検証してから統合する方針とした。現在の出所と挙動を十分に説明できないLoRA実装は基盤として使い続けず、Hugging Face PEFTや研究室内で実績のある簡潔な実装を調査する。研究室紹介では、オフライン学習を前提とした通常のAIからオンライン学習の制約と研究上の必要性へ接続する構成を検討する。詳細は[10月8日の議事録](meetings/2026-10-08-mtg.md)を参照。

2026-10-15、独立Standalone MoCo v2（CIFAR-10＋ResNet18、LoRA/Sequential Loaderなし）の基本動作を確認した。Meta公式に近いランダム初期Queue／Key-forward直前EMA／logits後・Backward前enqueue方式で、MoCo関連42件・既存LoRA回帰32件のpytestが成功。実CIFAR-10のCPU 20stepはcleanな`dev` commit `8939d7fca7d1cc3367b437c5f6adb48dd8e86f4a`で**Gate B PASS**、全14検証項目PASS、checkpoint復元PASS。**Gate C 500stepは未実施**。Lossが2.53→6.39に増加し、Negative類似度も増えているため、特徴の識別性・長時間安定性は未確認。詳細は[Stage1 Standalone MoCo Gate A/B/D実測確認](experiments/2026-10-15-standalone-moco-cifar10-gate-abd-console-evidence.md)を参照。

2026-10-15、GPUでStandalone MoCoのQueue置換64step診断とGate Cの500step実験を実行したことを、ユーザー提出の自動生成`verification_summary.md`2件から確認した。両runとも実CIFAR-10（ResNet18、LoRA/Sequential Loaderなし）で**全14工学チェックPASS**、checkpoint別インスタンス復元PASS、Git clean/同一dev commit。64stepではstep32でランダムQueue完全置換、500stepは**Gate C（工学）PASS**、EMA/Queue/Query更新は500stepを通じて整合。Lossはstep42に約7.00、step450に6.26、step500に6.64で、PositiveとHardest Negativeの識別性や学習表現品質は未確定。研究的な表現評価は別Gateとして残る。詳細は[GPU 64stepと500stepの実測Evidence](experiments/2026-10-15-standalone-moco-gpu-64step-and-gate-c-500step-evidence.md)を参照。

2026-10-20、MoCo単体のMeta公式parity検証を進め、ImageNet-1Kの固定実画像64枚・224px・ResNet50・2GPU NCCL・localB32/K65536・FP32の**A2正式1step数値一致検証がPASS**した。Meta公式M1と自作S1（通常`meta_einsum` opt-in）の独立fresh実行が正常終了し、事前固定`rtol=1e-5, atol=1e-6`で比較した3,926エントリは**3,926/3,926 bitwise一致**、exact/float failures 0。先立つMeta公式M0/M0rの同一fixture再演も3,926/3,926 bitwise一致。Evidenceは[A2正式GPU結果](experiments/2026-10-20-meta-moco-a2-imagenet-real-one-step-formal-gpu-parity-evidence.md)と[承認済みA2 spec](specs/2026-10-20-meta-moco-a2-imagenet-one-step-parity-experiment-spec.md)。この結果は**指定条件の1stepの比較範囲だけ**を証明し、Queue一巡/長時間安定性、A5通常学習速度、特徴識別性、ViT/LoRA/動画統合の正しさを保証しない。次のA5資源・軽量監視検証は別spec・別承認とする。

同日、承認済みの[A3/A4/A5統合実行spec](specs/2026-10-20-meta-moco-a3-a4-a5-integrated-execution-spec.md)に従い、既存のfixture生成・runner・comparatorを呼び出すだけの**段階ランチャー**（`scripts/parity/run_moco_gates.py`）とCPUテスト23件を追加した（実装当初未commit、既存ファイル無変更。その後devの914e736へcommit）。A5-pilot→A4→A3→A5-pairを1 stageずつ明示実行し、出力衝突・dirty・前段非PASS・GPU不足/使用中では何も書かずに停止する。CPUテストとミューテーション確認はPASSした（この段落は実装当初の状態）。GPU Gateの更新後状況は次段落に示す。詳細は[実装・CPU検証記録](experiments/2026-10-20-meta-moco-a3-a4-a5-gate-launcher-implementation-verification.md)。

同日11:46頃、統合ランチャーの実ImageNet-1K・RTX A4500×2・FP32・ResNet50・localB32/K65536・20step通常bench **A5-pilotがPASS**（`a345_20261020_114641`、`S1_off20`）。ユーザーが提示した端末ログでrun前preflight clean（research dev `914e736c...`、Meta `a419ef2`）、GPU2枚利用可能、ImageNet trainの存在、`issues=[]`と、run後`A5-pilot PASS: every step completed and every check holds`を確認。CPU軽量検証は新規23件＋既存代表6件の**29 passed / EXIT_CODE=0**。追加でユーザーがrun reportとstage summaryの表示ログを提示し、GPU peak reservedはrank0=3788MiB/rank1=3886MiB、ウォームアップ後10stepの平均step時間は約0.7905秒、global throughputは約80.96枚/秒、data waitはrank0約0.5992秒/step・rank1約0.5857秒/stepと確認。**A4は後続の別GPU実行で正式PASS**、A3・A5-pairはNOT_RUN。詳細は次段落に記録。詳しくは[A5-pilot GPU Evidence](experiments/2026-10-20-meta-moco-a5-pilot-real-imagenet-two-gpu-20step-evidence.md)を参照。

同日、**A4（2GPU synthetic / step2 native+rank sidecar checkpoint → step3 resume）のFormal検証がPASS**した。A4 stageはfixture、S1_uninterrupted、S1_resumed、compareの4工程が全てexit0、issuesなし。比較器`formal_resume`の固定閾値`rtol=1e-5, atol=1e-6`でstep3/full-trace step3の**3,608 / 3,608比較エントリがbitwise一致**、exact/close failures0、first difference/failureなし、contract_valid True。これは指定synthetic 3stepのCheckpoint/Resume再現性であり、Meta公式とのA3長時間parityや表現品質は未検証。[A4正式GPU Evidence](experiments/2026-10-20-meta-moco-a4-synthetic-two-gpu-step2-resume-formal-parity-evidence.md)参照。

同日、A3のMeta公式MoCo M対自作MoCo S1、**synthetic 1025step×2run／Queue一巡を含むFormal implementation比較が、統合ランチャー最終判定でPASS**。ユーザーが提示した終端ログは`A3 PASS: every step completed and every check holds`。固定`rtol=1e-5/atol=1e-6`の比較器まで進んでいる。追送された正式比較reportの集計値では**58,790/58,790比較エントリbitwise一致**、exact/close failures 0、contract_valid True、steps1〜1025の全比較、full traceは[1,1024,1025]。A3の各工程合計は実測約25分34秒、保存容量は`du -sh`で9.9G。Queue pointer個別実測値は未受領のため断定しない。**A5-pairはNOT_RUN**。[A3 GPU Evidence](experiments/2026-10-20-meta-moco-a3-synthetic-two-gpu-1025step-formal-implementation-evidence.md)。

2026-10-21、[ViT DDP MoCo Full FT / LoRA切替spec](specs/2026-10-21-vit-ddp-moco-full-ft-lora-switch-smoke-spec.md)をユーザー承認のもとで確定し、実装した（研究コードは新規ファイルのみ。`9348efe` feat・`2c26991` testとして`origin/dev`へpush済み）。pretrained ViT-B/16を自作DistributedMoCoに接続し、`model.train_mode=full_ft|lora`で切り替える。データはImageNet-1KとActivityNet単一フレーム（10秒間隔）。CPUでG0（実checkpointでQ/V 24箇所への注入、両モードの初期CLSがbitwise一致）、G1/G2（2-rank DDP同期、Frozen Baseのbitwise不変、EMA）、G4（step checkpointからのresumeがuninterrupted runとbitwise一致、別モードからのresumeを拒否）がPASS。既存testの失敗集合は変更前後で同一。ActivityNet manifestは10,024本・117,522枚で生成し、実decodeは64/64で成功した。**G3（2GPU pilot）は未実施**で、ViT Full FT local128 FP32のVRAM適合は未確認。[実装・CPU検証記録](experiments/2026-10-21-vit-ddp-moco-full-ft-lora-implementation-cpu-verification.md)

実装の正しさと基盤の成立を検証した後、学習済みLoRAに時間情報・動作情報が保持されているかを動画生成やVLMなどで評価する。その後、静的・動的情報の分離、直交化、LoRA空間での変換・組み合わせを検討する。LoRAの最終的な活用方法は探索段階にある。

2026-10-21、[G3 ImageNet 2GPU・1step実験Evidence](experiments/2026-10-21-vit-ddp-moco-g3-imagenet-1step-console-evidence.md)にユーザー提示の実行ログおよびrun_report集計を追記した。LoRA・Full FTともに1step completed、smoke_completed=true、VRAM Gate PASS、failures=[]、rank間state一致・frozen checks=true、入力sha一致。GPU rankあたりpeak reservedはLoRA 12038 MiB、Full FT 16862 MiBで、headroomはそれぞれ8057.875 / 3233.875 MiB。これはG3 **1stepのみの合格**であり、10/20stepや長期学習の安定性・表現品質は未検証。研究サーバのローカルJSON実体はGitHubから直接確認していない。


2026-10-21、[ViT DDP G3 10stepコンソールEvidence](experiments/2026-10-21-vit-ddp-moco-g3-imagenet-10step-console-evidence.md)を追加。ImageNet 2GPUでLoRA/Full FTの両runが10stepのメトリクスを出力し保存処理まで達した。全step有限だが、正規化Projector Query `query_feature_std` がstep1の0.07224からstep10にLoRA 0.01096、Full FT 0.00000606へ急落し、表現collapseの疑いあり。学習率AdamW 0.03は原因候補であって未検証。**run_reportのVRAM/最終Gate確認前に10step正式PASSや20step推奨へ自動昇格しない**。実験ログ・コードの追加変更・長時間学習は今回のResearch Workspace記録には含まれない。


2026-10-22、ユーザー提供のrun_reportおよびmetrics集計により、[G3 10step実験Evidence](experiments/2026-10-21-vit-ddp-moco-g3-imagenet-10step-console-evidence.md)を正式補足。LoRA・Full FTとも `status=completed`、`smoke_completed=true`、`vram_gate=PASS`、最終2rank一致、`failures=[]`、全10stepの入力ハッシュ一致で **G3 10step工学Gate PASS** と確定。ピークVRAM予約はrankあたりLoRA 12338 MiB、FT 18346 MiB。Full FTのheadroom 1749.875 MiBは閾値1024 MiBに対して余裕が狭い。normalized Query feature stdがFull FT step10で約6.06e-6へ落ち込む懸念は残り、representation qualityは未確立。20stepへ直行せずCLS/Projector/Queryを分けたdiagnostic検討を推奨。研究コード・GPUをChatGPT側で実行または変更していない。

2026-10-22、承認済みの[Query Collapse診断spec](specs/2026-10-22-vit-moco-query-collapse-diagnostic-spec.md)に従い、診断を研究コードへ実装した（`dev@2c26991`上の未commit差分。新規の診断module・診断用preset・CPUテスト33件と、runnerへの最小hook）。ImageNet trainから決定的に選んだ64枚の固定probeについて、ViT CLS、Projector出力、正規化Queryを、optimizer更新0/1/2/10回の時点でそれぞれ測る。あわせて、step1のgrad・parameter・update量をBackbone（LoRAはA/B）とProjectorに分けて記録する。診断は専用presetを指定したときだけ動き、出力先も別になる。**CPU Gate D0〜D3はPASS**：診断の有無で学習traceがbitwise一致し、診断OFFは変更前のコードとcheckpointのバイト列まで一致し、全テストの失敗ID集合は変更前後で同一だった。実ImageNet・実ViT-BのCPU preflight（学習なし）では、両modeのstep0特徴がbitwise一致した。**GPU診断run（D4）は未許可のため未実施**で、実行前に研究コードのcommitが必要（runnerはdirtyなrepositoryからのGPU runを拒否する）。Collapseの原因や表現品質はまだ判断していない。[実装・CPU検証記録](experiments/2026-10-22-vit-moco-query-collapse-diagnostic-implementation-cpu-verification.md)


2026-10-22、[Query Collapse診断D4 ImageNet GPU Evidence](experiments/2026-10-22-vit-moco-query-collapse-d4-imagenet-gpu-diagnostic-evidence.md)にユーザー提示のLoRA 10step `run_report.json`と`diagnostic_summary.json`集計を追記。**LoRA単独のD4工学GateはPASS**（両GPU peak reserved 12338MiB、headroom 7757.875MiB、frozen base不変、DDP rank state一致、probe ranks一致、`engineering_status=PASS`）。固定64画像はstep0→1でCLSのcosine中央値が0.03194→0.03192とほぼ維持される一方、**Projector raw出力の画像間cosine中央値が0.31879→0.99317**へ急上昇。正規化前`z_q`のrow norm中央値も2.266→153.144へ増大し、方向の集中はL2正規化前から発生。`q`のstdは0.07263→0.00715へ低下。Projectorのstep1相対update norm 1.4399、LoRA Bの初期norm0に対する巨大比率は無意味。AdamW lr0.03原因の可能性は高まったが未証明。**Full FT診断・両mode比較・D5解釈は未完了**。GitHubから研究サーバのraw JSON原本は直接確認していない。


2026-10-22、[D4 Query Collapse ImageNet 2GPU Evidence](experiments/2026-10-22-vit-moco-query-collapse-d4-imagenet-gpu-diagnostic-evidence.md)に**Full FT診断runの10stepコンソール完走**を追記。run出力先は`log/moco_vit_query_collapse/full_ft/imagenet/2026-10-22_04-20-32-327570`、全step`finite=true`で、診断なしG3とloss/gradient/Query feature stdの数値は同じ表示値。step10のtrain-batch `query_feature_std=6.05681771e-6` はLoRAの`0.01095892396`と比べ大幅に小さい。ただし**Full FTの正式D4 PASSは未判定**：保存された`run_report.json`のVRAM Gateと`diagnostic_summary.json`の固定64枚のCLS/z_q/q、step1更新量を未受領。LoRA D4は正式工学PASS、両mode比較・研究的解釈D5は引き続き保留。GPU実験はユーザーが研究サーバで実行しており、GitHubからローカルraw JSONを直接確認したものではない。

2026-10-22、ユーザーがLoRAとFull FTの保存済み`run_report.json`、`diagnostic_summary.json`とcross-mode比較結果を提出。**[D4の両方式2GPU・各10stepの工学GateはPASS](experiments/2026-10-22-vit-moco-query-collapse-d4-imagenet-gpu-diagnostic-evidence.md)**（LoRA/FT peak reserved各GPU 12,338/18,346MiB、headroom 7,757.875/1,749.875MiB、両rank一致、probe全rank一致、入力一致、step0 CLS/z_q/q bitwise一致、comparable true）。D5記述的解析では、固定64画像のFull FT CLS画像間cos p50が更新前0.032から**最初の更新後0.985**へ急増（step10では0.999997）、対してLoRA CLSはstep1/2で約0.032を維持。しかしProjector raw `z_q`は両方式のstep1で強く同方向化（LoRA cos p50 0.99317／Full FT 0.99994）し、Projector相対update normは両方式約1.44。Full FT Backbone相対updateは約0.36。これは**Collapse発生箇所の局在化**であり、AdamW LR0.03を原因とする因果検証・特徴品質の優劣は未成立。次は現行レシピを保存したまま、低LR/群別LR切り分けの別specを検討。GPU runはChatGPTが実行しておらず、ローカルJSONはユーザーの集計を根拠とする。

## マイルストーン

- [ ] 研究方針を整理する
- [x] シーケンシャルデータローダを独立リポジトリに分離し、再利用可能にする
- [ ] 逐次入力に対するLoRAのみのファインチューニングを動作させる
- [ ] 学習済みLoRAが保持する動画情報の評価方法を定める
- [ ] 動画生成またはVLMを用いてLoRAを評価する
- [ ] 現行ActivityNet + late fusion + MoCo経路のAIによる確認結果と実装の未確証点を次回MTGで報告する
- [ ] 画像MAE/ViTの重みをVideoMAEへ移す方法を調査・検証する
- [ ] 下流タスク・linear probeとLoRAの大きさ・特異値・層別変化による評価条件を定める
- [ ] 静的・動的情報の分離やLoRAの直交化を検討する
- [ ] LoRAを単体検証し、採用する実装・ライブラリとQ/V更新の挙動を確認する
- [ ] MoCoを単体検証し、loss・勾配・positive similarity・Queueの基準挙動を確認する
- [x] ImageNet実画像・2GPUのMeta公式MoCoと自作`meta_einsum`のA2 1step正式数値一致を検証（3,926/3,926 bitwise、2026-10-20）
- [x] MoCo A5-pilot：ImageNet real DataLoaderで2GPU・通常bench 20stepを完了（2026-10-20、ランチャーPASS）
- [x] MoCo A4：2GPU syntheticのstep2 Checkpoint/step3 Resumeが正式比較3,608/3,608 bitwise一致（2026-10-20）
- [x] MoCo A3：Meta公式M対自作S1の2GPU synthetic 1025step×2runで正式比較58,790/58,790エントリbitwise一致（2026-10-20）
- [ ] MoCo loss振動とQuery LoRA勾配急増の原因を切り分け、統合実験の再開条件を定める
- [ ] 研究室紹介用にオンライン学習の難しさと研究上の必要性を説明するスライドを作成する

## 更新履歴

| 日付 | 内容 |
|------|------|
| 2026-10-21 | ViT Full FT / LoRAのG3 ImageNet 2GPU・1stepは両方式PASS。rank0/1のVRAM Gate、final state、入力hashの一致をユーザー提出ローカルレポート集計で確認。GPU peak reservedは12038/16862 MiB。10/20stepは未実施。[G3 Evidence](experiments/2026-10-21-vit-ddp-moco-g3-imagenet-1step-console-evidence.md) |
| 2026-09-03 | シーケンシャルデータローダとLoRA逐次学習を第一目標とし、その後に動画情報の評価・分離へ進む方針を確認 |
| 2026-09-04 | プロジェクト作成 |
| 2026-09-04 | `2026_04_ishikawa_simple-MeMViT` をメイン実装フォルダに設定 |
| 2026-09-10 | sequential_loaderからMeMViTへの入力変換と、全16 blockのattention q/vへのLoRA注入・最小更新を確認。本実装統合とloader出力の挙動検証を次段階とした |
| 2026-09-13 | `2026_09_ishikawa_sequential-video-lora` をメイン実装フォルダに変更 |
| 2026-09-23 | 50Saladsの1 chunkからfrozen ViT frame feature `[16,768]` へのStage 2接続を実データで確認。paddingと順序保持は人工データで検証 |
| 2026-09-23 | ActivityNet Adapter specを承認し、指定branchへ実装。Core無変更で全183テスト・実データinventory・3形式の先頭chunk smokeが成功 |
| 2026-09-23 | Stage 3 specを承認・実装し、ActivityNet→frozen ViT→masked meanのclip feature `[768]`を実データ1 chunkで確認。新規25件・既存50Salads 1件・Hydra 38件成功 |
| 2026-09-24 | Stage 4のQ/V LoRA encoderと1-step smokeを実装。新規22件・既存65件成功、実ActivityNet CPU smokeでbase不変・LoRA 24 tensors更新を確認 |
| 2026-09-24 | Stage 5 MoCo v2-style mechanicsを実装。新規38件・既存87件と実ActivityNet 2動画のCPU 1-step smokeが成功。Query更新・Key EMA・base不変・queue更新を確認 |
| 2026-09-24 | MTGでオンライン学習に合う動画LoRA獲得を研究目的として再確認。現行MoCo経路と画像MAE重みのVideoMAE移植を検討し、下流評価・LoRA解析を課題とした |
| 2026-09-24 | MTG後、AIを用いたStage 6Aの実ActivityNet 10-stepとfresh 100-stepがPASS。新規42・既存125テストも成功。表現性能・時間情報獲得は未評価 |
| 2026-09-25 | ユーザー補足を反映。動作・テストのPASSはAIを用いた確認結果であり、実装が意図どおり正しいかは現段階でも未確証と明記 |
| 2026-09-28 | Stage 6A/6Bを共通Streaming MoCoへ統合。既存167・新規75テスト、Stage 6A人工10-step数値一致、Stage 6B実ViT/PEFT接続が成功。実データStage 6B・表現性能は未検証 |
| 2026-10-06 | Full MoCo / Linear Probeの設定をHydraのversion付きpresetへ一元化（未コミット）。記録値と実行値の二重管理を解消し、manifest / feature schemaを更新。全テスト527件成功、失敗36件は既存の環境依存。詳細は[検証記録](experiments/2026-10-06-full-pipeline-hydra-config-verification.md) |
| 2026-10-08 | MoCo loss振動とQuery LoRA勾配急増のためLinear Probeの解釈を保留。LoRA単体・MoCo単体を小さなステップで検証し、信頼できるLoRA実装を調査してから統合する方針を確認 |
| 2026-10-15 | Standalone MoCoの実CIFAR-10 GPU Queue置換64stepとGate C 500stepが工学的PASS。特徴表現の識別性・時間情報の学習は未検証。結果はexperimentsのEvidence記録を参照 |
| 2026-10-20 | Meta公式M0/M0r再演の3,926/3,926 bitwise一致後、事前固定FP32閾値によるImageNet実画像・ResNet50・2GPU×1stepのMeta公式M1対自作S1でFormal A2 PASS（3,926/3,926 bitwise）。A5/A3/A4・表現品質は別検証 |
| 2026-10-20 | A3/A4/A5統合実行specに従い段階ランチャーとCPUテスト23件を実装（未commit、既存ファイル無変更）。CPU検証PASS、GPU Gateは全てNOT_RUN。詳細は[実装・CPU検証記録](experiments/2026-10-20-meta-moco-a3-a4-a5-gate-launcher-implementation-verification.md) |
| 2026-10-20 | A3/A4/A5ランチャーをdev@`914e736c`へcommit済みと確認。CPU軽量29件PASS、plan全4stage正常。ImageNet実画像2GPUのA5-pilot 20stepはランチャー判定PASS。A4/A3/A5-pairは未実施。追加表示ログでGPU peak reserved 3788/3886MiB、平均0.7905秒/step、global 80.96枚/秒を確認。A4 planは前段PASS・issuesなし。[GPU Evidence](experiments/2026-10-20-meta-moco-a5-pilot-real-imagenet-two-gpu-20step-evidence.md) |
| 2026-10-20 | A4のstep2 Checkpoint→step3 Resumeを2GPU syntheticで正式検証。4工程exit0、stage/formal_resumeともPASS、固定閾値で3,608/3,608比較エントリbitwise一致。A3・A5-pairはNOT_RUN。[A4 Evidence](experiments/2026-10-20-meta-moco-a4-synthetic-two-gpu-step2-resume-formal-parity-evidence.md) |
| 2026-10-20 | A3のMeta公式M対自作S1をsynthetic 1025step×2run、Queue一巡後まで実行。Stage/formalともPASS、比較58,790/58,790エントリbitwise一致、failures0、正式契約valid。所要時間約25分34秒・保存容量9.9G。A5-pair NOT_RUN。[A3正式Gate Evidence](experiments/2026-10-20-meta-moco-a3-synthetic-two-gpu-1025step-formal-implementation-evidence.md) |
| 2026-10-20 | MoCo v2 global256/2GPU×local128の実ImageNet V1 30stepをMeta公式と自作の両実装で完了、全rank VRAM Gate PASS。ピークreservedはMeta13026MiB、自作13038MiB、headroom7069.875/7057.875MiB、Comet双方OFF。表示されたLoss/Cosine/Queue等は一致するが全パラメータbitwiseは未検証。MetaはDataLoader待ちが長く速度比較は保留。次はV1 offline input-hash比較を確認後、別許可でE1へ。[V1 Evidence](experiments/2026-10-20-meta-moco-v2-v1-global256-gpu-feasibility-partial.md) |
| 2026-10-20 | V1 30stepのMeta公式/自作でオフライン比較が完了。比較契約valid・両run completed・記録scalarはLoss等4/4、grad norm2/2完全一致（max_abs_diff=0）、入力ハッシュ1/1一致。epoch-state hashは未比較（None）。V1のresource feasibilityと記録指標parityのみPASS、E1/E2/E3は未実施・別許可。[V1 Evidence](experiments/2026-10-20-meta-moco-v2-v1-global256-gpu-feasibility-partial.md) |
| 2026-10-20 | V1 30step×2の実ImageNet結果と比較器出力がPASS。ユーザーが次のE1（Meta公式・自作各100step、Comet `moco-imagenet-v2-reproduction`へライブ送信）のみ実行承認。研究コード・GPU実行・Comet通信はこのResearch Workspace更新時には未実施。E2/E3/E4は保留。[E1実行契約](specs/2026-10-20-meta-moco-v2-official-global256-imagenet-10epoch-comet-reproduction-spec.md) |
| 2026-10-20 | E1（Meta公式と自作の実ImageNet各100step、global256、2GPU、FP32、Cometライブ）はユーザー提出ログで両run completed、契約valid、入力hash2/2一致、Loss等の全11地点・grad norm2地点でbitwise一致（max_abs_diff 0）。Comet実験リンク2件あり。Partial epochのためepoch state未評価、特徴品質未確認。次のE2（1epoch）は別承認待ち。[E1実験Evidence](experiments/2026-10-20-meta-moco-v2-e1-imagenet-100step-comet-parity-evidence.md) |
| 2026-10-20 | ユーザーがE1再現性比較PASSを踏まえ、E2（Meta公式と自作各1epoch、ImageNet-1K/2GPU×local128/FP32、Cosine10epoch schedule・1epochで停止、Cometライブ、checkpoint/canonical state比較）のGPU実行のみ承認。仕様を更新済み、実験実行・結果は未確認。E3 10epochは別承認が必要。[E2採用実行契約](specs/2026-10-20-meta-moco-v2-official-global256-imagenet-10epoch-comet-reproduction-spec.md) |
| 2026-10-20 | E2：Meta公式・自作ともImageNet1epoch 5004step完了。入力hash51/51、Loss等の記録指標502/502、query_grad_norm52/52がbitwise一致、epoch mean loss差0。比較器の`curve_agreement=bitwise`を確認。ただし`canonical_state_equal`行が貼付断片から欠けているため**E2正式PASSは保留**。Checkpoint/Comet実験の整合確認後に確定し、E3は別承認。[E2暫定Evidence](experiments/2026-10-20-meta-moco-v2-e2-imagenet-one-epoch-pair-partial-evidence.md) |
| 2026-10-20 | E2 1epoch 5004stepのMeta/自作ペアについて、ユーザーが比較reportから全Verdict PASS（contract_valid、both_completed、inputs_identical、metrics_bitwise_equal、**epoch_states_identical**）を追加確認。epoch0 canonical state SHA一致、mean loss差0、51/51 input hashes、502/502指標等がbitwise一致。**E2比較PASS**。Checkpoint/rank sidecarsの存在、Comet deliveryはこの表示から未確認。E3 10epochは別承認。[E2 Evidence](experiments/2026-10-20-meta-moco-v2-e2-imagenet-one-epoch-pair-partial-evidence.md) |
| 2026-10-20 | E2最終確認：ユーザーがMeta公式/自作それぞれのrun_reportとディスク実体を照合し、両方5004step完了・canonical state SHA `2dc5e599...`一致・checkpoint_epoch001.pt・rank0/1 sidecarの実在・Comet URL記録・errors=[]・failures=[]を確認。既存の51/51 input hash・502/502 logged metric・52/52 grad norm一致と合わせ、**E2比較・保存Gate PASS**。Checkpoint内容・Optimizer/RNG sidecarの全bitwise・Resume・画像特徴品質は未検証。E3 10epochは別承認。[E2 Evidence](experiments/2026-10-20-meta-moco-v2-e2-imagenet-one-epoch-pair-partial-evidence.md) |
| 2026-10-20 | **研究方針更新**：ユーザーが長時間の10epoch比較を中止し、Meta公式5epoch→自作5epochの自動逐次学習へ変更。Cometは同一Projectの独立した2 Experimentとして記録。5epoch Cosine horizon、新規共通fixture、旧E2 Checkpointからresumeしない、公式が成功しなければ自作を起動しないGateを採用。[5epoch逐次比較spec](specs/2026-10-20-meta-moco-v2-imagenet-5epoch-sequential-comet-comparison-spec.md)を作成し、旧10epoch E3実行契約はsuperseded。**実装・Git push・5epoch GPU実験は未実行**。 |
| 2026-10-20 | 5epoch specに従い、runnerに`gate=E5`（GPU runで5epochを受理する唯一のGate。V1〜E3は10epochのまま）と、Meta公式→自作を1本ずつ実行して各runの完了・checkpoint/sidecar・Comet記録を確認してから次へ進む逐次ランチャー`run_moco_5epoch_pair.py`（既定は計画表示のみ）を実装。CPUテストは関連3ファイルで108件PASS（新規33件）。計画モードは研究サーバで何も書き込まないことを確認。**研究コード未commit、`--execute`・5epoch GPU学習・Comet実送信は未実施**。[実装・CPU検証記録](experiments/2026-10-20-meta-moco-v2-5epoch-pair-launcher-implementation-verification.md) |
| 2026-10-20 | ユーザー許可により5epochランチャー実装を`dev`へcommit（`dd0d8b3` feat、`7fc1f56` test）し、`origin/dev`へpush。clean commitで計画モードを再実行し、issuesなし（研究repo・公式clone clean、config契約・ImageNet train確認PASS）。GPU空き確認・Comet preflightは`--execute`時のみで未確認。**5epoch GPU学習とComet実送信は未実施・未承認**。[実装・CPU検証記録](experiments/2026-10-20-meta-moco-v2-5epoch-pair-launcher-implementation-verification.md) |
| 2026-10-21 | **E5 5epoch比較PASS**（ユーザー提示の研究サーバpair_summary）：campaign `moco_v2_e5_20261020_220032`、Meta公式→自作の逐次学習が各25,020/25,020step・epoch0..4完走。各run独立Comet Experiment、errors=[]、全4処理 exit0。共通指標2503 logged stepsでbitwise、入力hash251/251、各epoch canonical model state SHA一致、比較契約全Verdict True。epoch平均Lossは9.7178→7.6079、所要40,288秒（11時間11分28秒）、VRAM peak reserved約13GiB/20GiB/rank。**MoCo数値実装parityが5epochに拡張された**が特徴品質・動画時間情報は未検証。[E5実験結果](experiments/2026-10-21-meta-moco-v2-e5-imagenet-5epoch-pair-parity-evidence.md)・[5epoch spec](specs/2026-10-20-meta-moco-v2-imagenet-5epoch-sequential-comet-comparison-spec.md) |
| 2026-10-21 | ViT DDP MoCo Full FT / LoRA切替specのBlockingを候補どおり承認（正規化はaug-plus＋ViT 0.5、ActivityNet単一フレームは10秒間隔・t=5+10n・上限なし・同一フレーム2-view・同一動画negativeは除外しない）。新規ファイルのみで実装（未commit）。CPUでG0/G1/G2/G4がPASS、新規54 test PASS、既存testの失敗集合は変更前後で同一（GPU不可視・NAS依存）。ActivityNet manifestは10,024本・117,522枚で生成（607秒）、実decodeは64/64。**G3 GPU pilot・commit/pushは未実施**。[実装・CPU検証記録](experiments/2026-10-21-vit-ddp-moco-full-ft-lora-implementation-cpu-verification.md)・[spec](specs/2026-10-21-vit-ddp-moco-full-ft-lora-switch-smoke-spec.md) |
| 2026-10-21 | ユーザーの指示により、ViT DDP MoCo Full FT / LoRA切替の研究コードを`dev`へcommit（`9348efe` feat、`2c26991` test）し、`origin/dev`へpush（HEAD＝origin/dev、clean）。ActivityNet manifestは同一内容のコードで生成済み（provenanceの記録は`7fc1f56`＋dirty）。**G3 GPU pilotは未実施・未承認**。[実装・CPU検証記録](experiments/2026-10-21-vit-ddp-moco-full-ft-lora-implementation-cpu-verification.md) |
| 2026-10-21 | ViT DDP G3 10stepでLoRA/Full FTともconsole上完走・finite、Queue pointer 2560。ただしnormalized Query feature stdはstep10で0.01096 / 0.00000606へ低下（Collapse疑い）。run_reportのVRAM/final checksと原因診断が未了、20stepは慎重に判断。[G3 10step Evidence](experiments/2026-10-21-vit-ddp-moco-g3-imagenet-10step-console-evidence.md) |
| 2026-10-22 | ViT DDP G3 10stepのLoRAとFTについてrun_report集計でVRAM/最終チェックもPASS、全10step同一入力を確認。Peak reservedは12338/18346 MiB/rank。FT feature std≈6e-6のcollapse疑いは未解消、表現評価は保留。[10step Evidence](experiments/2026-10-21-vit-ddp-moco-g3-imagenet-10step-console-evidence.md) |
| 2026-10-22 | Query Collapse診断spec（B1〜B5）を実装（未commit）。固定probe 64枚でCLS/z_q/qを更新0/1/2/10回時に測定し、step1の更新量も記録。CPU D0〜D3 PASS（新規33件、診断の有無で学習traceがbitwise一致、全テストの失敗ID集合は変更前後で同一）。実データCPU preflightで両modeのstep0特徴がbitwise一致。specは`implemented`。GPU D4は未許可・未実施。[実装・CPU検証記録](experiments/2026-10-22-vit-moco-query-collapse-diagnostic-implementation-cpu-verification.md) |
| 2026-10-22 | Query Collapse診断D4のLoRA 10stepはユーザー提示run_reportとdiagnostic_summary集計で工学PASS。fixed64画像のstep1でCLSはほぼ維持、Projector raw出力の画像間cos中央値は0.319→0.993、raw normは2.266→153と増大。Full FT D4・compare CLI・因果判定は保留。[D4 GPU Evidence](experiments/2026-10-22-vit-moco-query-collapse-d4-imagenet-gpu-diagnostic-evidence.md) |
| 2026-10-22 | D4 Full FTもGPU 10stepコンソール上完走・finite。学習batch Query stdはstep10で約6.06e-6、LoRAより約1809倍小さい。Full FTのrun_report/diagnostic_summary正式照合、固定probeのCLS/z_q/q解析、両mode比較は未完了。[D4 Evidence](experiments/2026-10-22-vit-moco-query-collapse-d4-imagenet-gpu-diagnostic-evidence.md) |
| 2026-10-22 | Query Collapse D4はLoRA・Full FT両方式のGPU 10stepで工学PASS、全10step入力・固定64画像・step0特徴は一致。D5ではFull FT CLSがstep1でcos p50 0.032→0.985、LoRA CLSは初期保持。一方両modeのProjector出力はstep1に強く方向集中（cos p50 0.993/0.99994）、Projector相対更新約1.44。LR0.03の因果仮説は未検証。[D4/D5 Evidence](experiments/2026-10-22-vit-moco-query-collapse-d4-imagenet-gpu-diagnostic-evidence.md) |
