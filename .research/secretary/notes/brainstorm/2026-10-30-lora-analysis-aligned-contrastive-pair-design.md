---
date: 2026-10-30
project: sequential-video-lora-analysis
source_todo: null
topic: lora-analysis-aligned-contrastive-pair-design
status: exploratory
tags: [brainstorm, contrastive-learning, online-learning, moco, lora, dynamic-information, evaluation]
---

# 中間発表のLoRA解析4項目と整合する正例・負例設計（探索）

## 出発点・Authority

- ユーザーは2026-10-30中間発表スライドに記載した学習（正例＝同じ動画の時刻の違うclip、負例＝別動画）が、後段のLoRA解析と研究の問いに適切か比較検討を依頼。
- 添付 `20261030_中間発表_石川.pptx`：p.5のFrozen image ViT／ΔW動的情報の仮説とsingle-pass online、p.6のearly fusion・Q/V LoRAとhead、p.7の正例・負例および学習時入力比較、p.8逆再生、p.9外観スタイル介入、p.10特異値分解と静的基準の部分空間への射影、p.11 Kinetics Temporal-32 vs Static-class動作認識評価。
- Research Workspace `README.md` （最終更新10/22）は動画LoRAで動的情報獲得を**未実証**とする。Meta MoCo parity・ViT＋MoCo短時間工学PASSと、動的情報獲得は別Evidence。
- `2026-10-29-case1-case2-verification-design-kinetics.md` には、**ユーザーの当時の学習設定案**「正例＝同動画異時刻clip、負例＝別動画」を記録。逆再生を学習に使わず評価の循環を避ける狙い。同時に背景shortcutを懸念。10/29の追記では、p.7の「順序を崩したLoRA」条件は保留し、frame-wise LoRAを主な静的比較基準とする方針。添付p.7と後日の探索判断は不一致あり。
- `2026-10-27-proposed-method-experimental-setup.md` 19:55追記は同動画異時刻clipを無条件にpositiveにしないと補正。スライド及びメモの各案は実装・採用specではなくexploratoryである。

## 質問の中心

「オンラインsingle-pass、画像事前学習ViT Base凍結＋early fusion Q/V LoRAのみ更新という制約下で、ΔWの時間感度・見た目頑健性・静的基準との差・動きが必要なタスクの4種類の事後解析の識別力を損ねず、どの正例・負例の定義が妥当か。」

## 先行研究の裏づけ（公開Webを2026-10-30確認）

- Feichtenhofer et al., CVPR 2021 `A Large-Scale Study on Unsupervised Spatiotemporal Representation Learning`：同動画の時間的に離れたclipをpositiveとする方向で時系列持続性が学習され得る。https://openaccess.thecvf.com/content/CVPR2021/html/Feichtenhofer_A_Large-Scale_Study_on_Unsupervised_Spatiotemporal_Representation_Learning_CVPR_2021_paper.html
- Qian et al., *Controllable augmentations for video representation learning*, 2024：同動画別clip positivesは背景が類似して動作が異なるため、静的背景バイアスを生むと指摘。https://link.springer.com/article/10.1007/s44267-023-00034-7
- Wei et al., CVPR 2018 `Learning and Using the Arrow of Time`：時間方向判別の自己教師あり、人工的な手掛かりへの注意。https://openaccess.thecvf.com/content_cvpr_2018/html/Wei_Learning_and_Using_CVPR_2018_paper.html
- Kuang et al., ICCV Workshop 2021 `Video Contrastive Learning With Global Context`：video-level positiveだけでなく時間順序正則化を別途導入。https://openaccess.thecvf.com/content/ICCV2021W/CVEU/html/Kuang_Video_Contrastive_Learning_With_Global_Context_ICCVW_2021_paper.html

## 正例・負例候補と収束

| 候補 | 正例 | 負例 | 目的・利点 | 限界/判定 |
|---|---|---|---|---|
| O1基準（主たる統制群として推奨） | 現在clipの2種類の、時間順序を壊さないaugmentation | **既観測の別動画**のclip key | 無条件のfalse positiveが少なく、逆再生やスタイル交換を学習時に使わず後段評価の独立性を守る | 静的特徴だけでもlossを下げられる。O1だけでΔWの動的学習は未保証 |
| O-cross（10/29・スライド設定、比較群として保持） | 同じ動画の異時刻・過去clip | 既観測の別動画clip key | 時間的な持続性・video-level表現には合理性 | 長いActivityNetでは別動作のfalse positive、同背景shortcut。主設定に無条件採用は弱い。どうしても使う場合は過去・時間間隔を制限し、motion一貫性／静的背景shortcutを測定 |
| O-temporal（研究目的に対応する探索条件） | O1と同じ | 既観測の別動画clipに加え、**十分に区別できる**同一動画過去clipをmotion-confusable hard negativeとして選択、またはMoCo本体を保ち時間順序の補助課題を分離 | 同一背景で動きの違いが必要な学習信号を強める可能性があり、通常MoCoとの比較で追加信号の寄与を検証できる | 同動画の同じ動作ならfalse negative。動きが違うという無ラベル判定の安全性は要検証。時間順序の補助課題はp.8評価との独立性の問題が残る |
| O-reverse（補助ablationに限定） | O1 | O1負例＋現在clipのreverse/shuffle | フレーム集合と外観を揃えて順序識別を促す | p.8順逆判定との循環が強く、対称動作を誤負例にし得る。主解析に使うなら別の未学習動作課題で確証が必要 |

**有力方向**：O1を先に学習／評価し、O-crossと、時間情報を狙うO-temporalを分離して比較する。p.8の「順逆判定を初見評価」にしたいならO-reverseを主条件にしない。O1だけでは失敗し得るので、失敗結果もH3（学習目標に時間情報が不足）として扱う。

## 4解析との実験上の整合

1. **向きの判定（スライドp.8）**：O1 vs O-cross vs O-temporal（reverse-negativeを使わない主評価）でBase+ΔWの時間順向き判定を同一評価セット・学習し直す線形分類器で比較。reverseを学習した条件の向き判定は予期された挙動の確認にとどめ、未知の動作認識で汎化検証。
2. **見た目の変更（p.9）**：同一clipの時間方向と背景／style方向の変化を同一特徴空間で測る。training augmentationでスタイル変換を使った場合は評価の独立性が弱くなるので、評価時に未知styleへ外挿する、同一スタイル条件等を設ける。両viewのaugmentationは全時刻で一貫させ、途中フレームだけの不自然なちらつきに依存しない。
3. **ΔWの部分空間（p.10）**：Frame-wise LoRAとVideo LoRAを比較するなら、同じQ/V LoRA注入先・rank・alpha・初期化・更新量／入力動画量と、まずは**共通のO1**で比較する。動画のみ追加 temporal loss と frame-wise O1 を比べるとArchitectureとObjectiveの2要因が交絡。frame-wiseとのsubspace共通方向を機械的に静的と断定せず候補とし、射影成分／直交成分を除去する機能介入（ノルム・rankを考慮した対照）で裏付ける。構造が違う場合は評価用特徴の比較を揃える。SVDは各層の実効 `ΔW=(α/r)BA` を対象。
4. **動きが必要な行動認識（p.11）**：事前学習はActivityNet trainingのlabelを使わない。下流Linear ProbeではKinetics400のtrain/valでTemporal-32と他368クラスの増分を比較。異動画にも同一クラスがあるためfalse negativeの確率を監査。Projectorではなくencoder特徴、LoRA on/offを比較し、逆再生だけ上がるshortcutと区別する。

## 公平な比較契約（提案、未承認）

- `Base + frame-wise LoRA + O1`（静的参照） vs `Base + early-fusion temporal LoRA + O1`（構造効果） vs `Base + early-fusion temporal LoRA + O-temporal`（目的関数効果）を中心に置き、`Base + early-fusion + O-cross`を補助群にする。Frame-wiseではreverse-hard-negativeを無理に適用しない。
- exact同じ学習clip集合、time order、frame rate、T、視覚augment分布、Queue予算、EMA、optimizer、Projector容量、update予算、評価器を可能な限り一致。違い（frame-wise vs early fusionによるtoken数／Compute）を明示する。
- 2026-10-29後続メモでは「順序を崩したLoRA」は保留なので、添付p.7の比較列を現行確定事項と扱わない。
- O-crossを実施する場合、異時刻clipはtまでの観測済み過去だけで、Queue metadata `video_id, clip_index, timestamp, augmentation, key_age` とsource stream cursorを保存。同じ動画過去をO1において無条件negativeにする必要はなく、負例定義に応じて除外する。最初の各動画ではpositive/warm-upの定義を別途必要とする。
- O-temporal同動画hard negativeは、同じ動作を誤って押し離す危険がある。annotationやラベルを使って監督すればSSLとは異なる契約となるため、別条件として扱う。
- 注意：複数動画round-robinは各動画内の因果順を守れるが、動画間の厳密な実時間到着を再現するとは限らない。過去の特徴量保存を許容するオンラインの定義を明記。
- LoRAの成功条件はLoss低下でなく、学習済みencoder特徴でのp.8〜p.11評価とLoRAの介入による増分。Frozen Base = 完全な静的成分ではない。

## 確定・未確定

- 既存の明示的な学習案：同動画異時刻正例・別動画負例（10/29）。本相談では、その設計がp.8〜11の解析に強いかを**再評価**した。これを採用spec変更やコード変更と見なさない。
- **第一推奨（提案）**：O1を共通の比較土台に据える。動的目的を増やす場合はO1との差分を検証し、逆再生を評価に使うならreverse-negativeの直接使用は主条件で避ける。
- **保留**：O-crossを主条件として残すか、temporal追加信号をどの無ラベル基準で作るか、スライドp.7の順序を崩したLoRA条件を正式に除外するか。
- **未実証**：O1/O-cross/O-temporalのどれが最終的な動作認識とΔWの機能的分離に優れるか、長時間オンラインの性能、GPU収束。新規spec/実装/実験runは行っていない。

## 次の判断とspecへの引き継ぎ候補

- 必要な決定：主条件の正例を現状の同動画異時刻にするか、同一clipの2 viewに変更するか／時間目的の追加を採用するか／評価への漏洩をどう避けるか。
- 次アクション（候補）：O1で統一したframe-wise vs early-fusion比較を最小構成として確認し、その後O-cross／O-temporalを独立追加。特定の成功条件とmetadata監査を定義した上でresearch-specのGateへ渡す。ユーザーが明示的に求めるまでspecには昇格しない。

## 関連

- `2026-10-30-online-cross-clip-positive-negative-feasibility.md`
- `2026-10-29-case1-case2-verification-design-kinetics.md`
- `2026-10-27-proposed-method-experimental-setup.md`
- `2026-10-28-dynamic-information-verification-proposals.md`
- `2026-09-28-shared-streaming-moco-protocol-spec.md`
