---
project: sequential-video-lora-analysis
status: active
summary: Stage 3のActivityNet→frozen ViT→masked mean clip feature [768]を実装・実データ1 chunkで検証済み。LoRA学習統合は未実施。
implementation_root: /mnt/HDD12TB-1/ishikawa/2026_09_ishikawa_sequential-video-lora
created: 2026-09-04
last_updated: 2026-09-23
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

基盤の成立後、学習済みLoRAに時間情報・動作情報が保持されているかを動画生成やVLMなどで評価する。その後、静的・動的情報の分離、直交化、LoRA空間での変換・組み合わせを検討する。LoRAの最終的な活用方法は探索段階にある。

## マイルストーン

- [ ] 研究方針を整理する
- [x] シーケンシャルデータローダを独立リポジトリに分離し、再利用可能にする
- [ ] 逐次入力に対するLoRAのみのファインチューニングを動作させる
- [ ] 学習済みLoRAが保持する動画情報の評価方法を定める
- [ ] 動画生成またはVLMを用いてLoRAを評価する
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
