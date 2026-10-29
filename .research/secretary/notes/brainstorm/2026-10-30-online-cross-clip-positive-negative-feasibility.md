---
date: 2026-10-30
project: sequential-video-lora-analysis
source_todo: null
topic: online-cross-clip-positive-negative-feasibility
status: exploratory
tags: [brainstorm, online-learning, contrastive-learning, moco, positive-negative, temporal-information]
---

# オンライン条件における「同一動画の異時刻clip正例・別動画負例」の成立性（探索）

## 相談の出発点

ユーザーの問い：「正例：同じ動画の時刻の違うクリップ／負例：別動画」の定義を、自分の研究のオンライン学習で実現できるか。

## 参照した文脈・確認済み事実（2026-10-30）

- 研究目的：画像事前学習ViTを凍結し、時間的・動的情報がLoRAに獲得されるか検証する。プロジェクトREADMEと2026-10-27提案手法実験設定に記載。
- 採用済み旧Stage 6B（共通Streaming MoCo）：current clip raw vs x_transを正例とし、同じ動画の過去Keyを含む旧Queueの全履歴を負例にする。strict_singleでは過去以外の動画が未観測の場面がある。 `specs/2026-09-28-shared-streaming-moco-protocol-spec.md`。
- 2026-10-27提案手法メモの追記では、異時刻の同一動画clipを無条件に正例とせず、同一clipの2 viewを基本とし、時間方向の追加信号を分けることを推奨（exploratory）。
- GitHubで確認できた研究コード `tamaki-lab/2026_09_ishikawa_sequential-video-lora@dev` のcommit `46ad9833f764ed7d9bb8df45f2687345b7610f2a`：`scripts/vit_moco/activitynet_frames.py` では同一フレーム2 viewが正例、同一動画別フレームは負例から除外しない。`self_supervised/moco/ddp_moco.py` のQueueは過去Keyを含むが、現在のコードに同動画異時刻のpositive選択や別動画だけのnegative filterが実装済みとは確認できない。

## 結論と必要なオンライン条件（推論）

原理的に実現可能。ただし学習時刻tのQueryに使えるのはtまでに観測し終えたクリップだけ。例えばA2の更新時、既知の過去A1をcross-clip positive、既知の別動画B1をnegativeにできるが、未来A3/B3は使用不可。A1/B1の埋め込みを計算済みKey Queueまたは履歴バッファに保持すると、動画の再読込なしに使える（過去埋め込みの保持を許容するオンライン定義が前提）。

1本ずつA全編→B全編のstrict_singleなら、最初の動画A処理中にはB由来negativeがまだない。別動画negativeを得るには (a) 異動画を交互に時間順処理する複数stream、(b) 先に処理済みの動画からQueueを継承し、未準備の初回動画はwarm-upにする、等の設計が必要。いずれも各動画内で未来を使用しないことを監査する。

## 候補・比較

| 方針 | 正例 | 負例 | 長所 | 注意 |
|---|---|---|---|---|
| A: 質問どおりのcross-clip | 同一動画の既観測の異時刻clip | 既観測の別動画 | onlineで構成可能、場面をまたぐ不変性 | 同一動画内で動作が変わるfalse positive、動画固有の静的情報だけで成立、時刻差選択の難しさ |
| B: 同一clip 2-view（対照） | 現在までに観測した同一clipの2 augmentation | 既観測の別動画 | 正例の意味の一致、実装しやすさ | 静的特徴だけで解ける可能性 |
| C: 時間判別を足す（探索） | Bと同じ | 別動画に加え同じframe集合のreverse/shuffle等 | 時間の手がかりを使う方向へ誘導できる可能性 | 時間位置表現／early fusionが必要、逆再生が意味的に同じケースでfalse negative |

重要：別動画のclipも同じ行動クラスを含むことがあり、別動画＝別クラスではない。映像識別のshortcut、異動作のfalse positiveとfalse negativeへの影響を評価する必要がある。

## 採用候補・保留・棄却

- **採用候補（未決定）**：研究本条件はBをベースに、時間判別用の追加信号Cを別扱いとして評価し、Aをablationに置く。実装許可・spec確定ではない。
- **保留**：同動画異時刻の正例を本実験の主要方式に採用するか。正例を何秒・何clip離すか、同じ行動である保証をどう確保するか。
- **棄却寄り**：未来の異時刻clipを先読みしてpositiveにする方式（strict onlineに反する）。最初の動画から別動画negativeがあると仮定する方式。
- **反例**：ActivityNetの1動画に複数の異なる動作がある場合、A1とA2が異動作でもpositiveになり得る。動画内で背景だけが同じ場合、動的情報を学ばなくても損失を下げられる。

## 実験案・判定軸

- 各updateで使った`query_video_id, positive_video_id, negative_video_ids, timestamp, max_observed_timestamp, queue_age`を監査し、未来先読みなし・正例は過去・負例は別動画を機械的に検査する。
- 比較：A対B対Cで同じ学習clip集合・更新予算・モデルを使い、MoCo lossではなくLoRA有無の差分、reverse/shuffle感度、動きが重要な行動認識等を評価する。
- `strict_single`と複数streamの違いはサンプル順序とQueue構成に交絡するため、同じ「オンライン」として混同しない。

## 未解決事項・次の判断

1. 1本ずつのstrict_singleを必須とするか、各動画の未来を使わないmulti-streamを許容するか。
2. クロスクリップ正例を主条件にするか、対照実験に位置付けるか。
3. 同じ動作かどうかの手がかりをannotation等で使うか（自己教師あり性・online性の境界も確認）。
4. 過去の映像本体のreplayを禁じつつ、過去Key特徴量のQueueを許容するか。

## 関連資料

- [プロジェクトREADME](../../../lab/projects/sequential-video-lora-analysis/README.md)
- [提案手法実験設定（2026-10-27）](2026-10-27-proposed-method-experimental-setup.md)
- [オンライン/オフライン（2026-10-27）](2026-10-27-online-vs-offline-dynamic-information-lora.md)
- [共通Streaming MoCo spec](../../../lab/projects/sequential-video-lora-analysis/specs/2026-09-28-shared-streaming-moco-protocol-spec.md)
- [研究コード current dev](https://github.com/tamaki-lab/2026_09_ishikawa_sequential-video-lora/tree/dev)
