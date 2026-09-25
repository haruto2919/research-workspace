---
project: sequential-video-lora-analysis
status: active
summary: AIを用いたActivityNet MoCo 100-step検証はPASS。実装の正しさは未確証で、LoRA評価とVideoMAE移植を検討中。
implementation_root: /mnt/HDD12TB-1/ishikawa/2026_09_ishikawa_sequential-video-lora
created: 2026-09-04
last_updated: 2026-09-25
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
