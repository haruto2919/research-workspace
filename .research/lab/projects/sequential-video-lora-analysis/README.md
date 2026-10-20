---
project: sequential-video-lora-analysis
status: active
summary: MoCo単体Meta公式との数値parity（A2実ImageNet1step、A3 synthetic1025step）がbitwise PASS。10epoch用runnerはdev@d6f7accでCPU52/52 PASS。ImageNet Global Batch256/2GPU FP32のV1（各30step）は両run VRAM PASS、記録指標一致。E1（Meta/自作各100step）は双方completed、入力hash2/2一致、Loss等11/11・grad norm2/2のbitwise一致（最大絶対誤差0）、両方Comet実験URLあり。epoch stateは未評価。E2（Meta/自作各1epoch+Comet+checkpoint/state比較）は2026-10-20に個別実行承認済み・GPU未実行。E3（10epoch）は別承認待ち。
implementation_root: /mnt/HDD12TB-1/ishikawa/2026_09_ishikawa_sequential-video-lora
created: 2026-09-04
last_updated: 2026-10-20
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

実装の正しさと基盤の成立を検証した後、学習済みLoRAに時間情報・動作情報が保持されているかを動画生成やVLMなどで評価する。その後、静的・動的情報の分離、直交化、LoRA空間での変換・組み合わせを検討する。LoRAの最終的な活用方法は探索段階にある。

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
