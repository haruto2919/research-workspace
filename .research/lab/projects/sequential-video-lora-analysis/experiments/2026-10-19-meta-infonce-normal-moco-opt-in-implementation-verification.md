---
project: sequential-video-lora-analysis
record_type: implementation-verification
status: completed
created: 2026-10-19
last_updated: 2026-10-19
spec: ../specs/2026-10-19-meta-infonce-normal-moco-opt-in-spec.md
implementation_branch: dev
implementation_base_commit: 1ce9badcbae3e2df08fa400944e4c385785063f9
implementation_committed: false
implementation_diff_sha256: 13bf95635d84eb441d9ad0e48fe950e0600f48990feb92b0511cb48eff3cca76
engineering_cpu_gate: PASS
gpu_gates: NOT_RUN
reference: facebookresearch/moco@a419ef282fc9885015674e77f58c59447206a27f
reference_clone: /mnt/HDD12TB-1/ishikawa/facebookresearch-moco
---

# Meta InfoNCE演算の通常経路opt-in：実装とCPU検証

2026-10-19のユーザー依頼「このスペックに従って実装してください」に基づき、[通常経路opt-in spec](../specs/2026-10-19-meta-infonce-normal-moco-opt-in-spec.md)の第10節の範囲を実装し、第11節のCPU検証を実行した。spec §12.1のEngineering/CPU Gateは**PASS**（判定の詳細と注意点は下記）。

**GPUは使っていない。** A2〜A5のGPU Gate（§13のG2、G5、G3/G4）は別承認が必要なため未実施である。研究コードの変更は**未コミット**で、commit・push・Comet送信は行っていない。本記録は、ユーザーの承認後（同日）に作成した。

## 環境と基準

- 実装root: `/mnt/HDD12TB-1/ishikawa/2026_09_ishikawa_sequential-video-lora`、branch `dev`
- 開始時HEAD: `1ce9badcbae3e2df08fa400944e4c385785063f9`（specの基準commitと一致、clean）。現在もHEADは同じで、変更6ファイルが未コミット
- 変更後の`git diff`のsha256: `13bf95635d84eb441d9ad0e48fe950e0600f48990feb92b0511cb48eff3cca76`（`git diff --stat`: 6 files, +1323 / −114）
- Python 3.12.3、torch 2.14.0+cu130、pytest 9.1.1（`.venv`）。テストは`CUDA_VISIBLE_DEVICES=""`でCPU（gloo 2-process、一部はin-process 1 rank）のみ
- 公式clone: `a419ef2`のまま、`git status`差分なし

## 変更したファイル

spec §10.1の6ファイルだけを変更した。§10.2の変更しないファイル（`standalone_moco.py`、`conf/moco_standalone.yaml`、A1 judge 2種、`utils/checkpoint.py`）は`git diff`で無変更を確認した。

| ファイル | 内容 | 最終sha256（先頭12桁） |
|---|---|---|
| `conf/moco_imagenet_parity.yaml` | `moco.info_nce_impl: self`（通常selector）、`diagnostic.info_nce_impl: null`（A1 selector、既定を`self`から変更）、`training.legacy_resume_info_nce_impl: null`、`monitor.*`（enabled、scalar 10、grad norm 100、update norm 0、first/last step）、`bench.measurement_window_steps: 10` | `003a475c201c` |
| `scripts/parity/moco_imagenet_parity.py` | resolver `resolve_info_nce_selection`、SelfAdapterへの伝搬、全rank照合`check_info_nce_ranks`、config identity v2、checkpoint/sidecar v2、collective-safeなResume検査、legacy Resume、bench監視（`BenchMonitor`）と窓計測（`WindowTimer`）、run終了時の有限性検査、run schema v3、provenance・Comet・summary | `d70757afb0cc` |
| `scripts/parity/compare_moco_parity.py` | effective mode reader（v3優先、旧schemaはfallback）、`formal_implementation`・`formal_resume`、formal前提条件、comparison schema v3 | `09198ce517ec` |
| `self_supervised/moco/ddp_moco.py` | docstringのみ（A1診断専用の記述を通常opt-inの説明へ更新）。算術・既定値・`INFO_NCE_IMPLS`・state_dict keyは無変更 | `cbd5ab52b621` |
| `test/model/test_moco_ddp.py` | 両modeのgloo 2rank照合と、rank差での全rank停止（4件） | `aa8883286bfa` |
| `test/model/test_moco_parity_harness.py` | selector表、拒否、identity、Checkpoint/Resume、legacy、監視・窓計測、比較器、formal pair（74件）。既存1件のassertionを更新（下記） | `d119d8de0c74` |

## 実装した契約（spec §5〜§9）

### Selector（§5）

単一のresolverが`configured_normal`・`configured_diagnostic`・`selector_source`・`effective`・`legacy_compat`を返す。validation、adapter、run report、Comet、checkpoint、fixture検査はすべてこの戻り値を使う。

| implementation / Gate | `moco` | `diagnostic` | 結果 |
|---|---|---|---|
| self、A2〜A5 | self / meta_einsum | null | normal値（source `moco`） |
| self、A2〜A5 | self | self | `self`、source `legacy_diagnostic_self`、`legacy_compat=true` |
| self、A1 | self | null / self / meta_einsum | nullなら`self`（source `moco`）、それ以外はdiagnostic値（source `diagnostic`）。meta_einsumは既存2 profileのみ |
| self、A1 | meta_einsum | 任意 | 拒否 |
| self、A2〜A5 | meta_einsum | self | 拒否（混在） |
| self、A2〜A5 | 任意 | meta_einsum | 拒否（既存profileの理由文） |
| meta_reference | self | null / self | `meta_reference`（selfなら`legacy_compat=true`） |
| meta_reference | meta_einsum、またはdiagnostic meta_einsum | — | 拒否 |
| 任意 | 未知の値 | 任意 | 拒否 |

拒否は`resolved_config.yaml`の保存後、spawn前に起こる。errorには両selector、implementation、run_kind、gate、許可された組合せを含める。既存A1テストが要求する「末尾が`differing (configured values): {...}`」の形式も維持した。

### mode identityとCheckpoint（§6、§7）

- `check_info_nce_ranks`: model構築後、最初のforward・checkpoint load・Comet metricより前に、各rankが（effective、model属性、selector source）を`all_gather_object`で集め、1つでも違えば全rankで同じ一覧を付けて停止する。
- config identity v2: `schema_version`と`info_nce`（resolver出力）を追加した。v1部分は従来の規則（`moco.info_nce_impl`を除く）で、旧sidecarのhashを再現できる。
- Self native checkpoint v2はtop-level `metadata`を持ち、sidecar v2と同じ4項目（effective mode、selector source、identity schema、identity sha256）を記録する。Meta公式native形式は変更せず、sidecarだけに記録する。`checkpoint_manifest.json`にも同じ4項目を書き、`legacy_format: false`とする。
- Resume: 各rankが自分のsidecar（Selfはnative metadataも）を検査し、結果を`all_gather_object`で集約する。どのrankに問題があっても全rankが同じ理由で停止し、`load_state_dict`・optimizer・RNG復元へ進まない。
- legacy（v1、mode未記録）: 既定（`null`）では拒否する。Selfは、明示値＝現在のeffective、旧`resolved_config.yaml`が存在し`diagnostic.info_nce_impl`が明示値と一致、旧sidecarと現在configと旧configのv1 identityが一致、nativeもv1、のすべてを満たす場合だけ再開する。許可時はrun report・provenance・summaryの`resume.legacy_mode_assertion`に明示値と旧configのsha256を記録する。Meta公式のv1 sidecarはv1 identityだけを検査し、`meta_reference`として扱う。

### 通常run・監視・負荷（§8）

- bench loopから、毎step・全rankのQuery clone、毎stepのgrad/update norm、毎stepの`torch.cuda.synchronize`を除いた。
- rank 0だけが、scalar（loss、positive/negative cosine mean、queue_ptr、LR）を10stepごと、grad normを100stepごと、update normを正の間隔の時だけ（既定0＝無効）記録する。first/last stepは正の間隔の指標にだけ追加する。未計測の指標はfieldごと省略し、0で埋めない。学習step数は`completed_steps`として別に数える。
- 時間は、warmup後の`measurement_window_steps`ごとの窓で、窓の境界だけでdevice同期してwall timeを測る。窓のstep time、global images/sec、data waitを集計し、peak allocated/reservedはrun単位で記録する。窓の時間はdata waitと監視のコストを含む。
- `monitor.enabled=false`は指標とComet metric送信を止めるが、mode/provenance・終了status・failure artifact・有限性検査は止めない。run終了時にQuery parameter、Queue、momentumの有限性を検査する。
- `PhaseTracer`はparityだけで使い、benchには入れない。

### 比較器（§9）

- `effective_info_nce(run)`: v3の`info_nce.effective`を優先する。旧run（v1/v2）は`diagnostic.info_nce_impl`、Metaはimplementationから求め、`mode_source=legacy_fallback`と記録する。どちらもない旧Self runは`unknown`。
- 既存のcomparison field（`reference/candidate.info_nce_impl`、A1 control field）は意味を変えずに残した。`comparable_conditions`からは`moco.info_nce_impl`を除いた（modeはpair kindごとに検査する）。
- `formal_implementation`（Meta対Self `meta_einsum`）と`formal_resume`（同じSelf `meta_einsum`の無中断run対resumed run）を追加した。値を読む前に、次のどれかに当たれば`INVALID`とする。
  - modeが期待と違う。
  - v3の`info_nce`がない、またはrank間で不一致。
  - config・provenance・run reportのmodeが食い違う。
  - dirty、commit差、clone不整合、max_steps差、比較範囲のfull trace step差。
  - `formal_resume`で、resume証跡がない、checkpoint identityが合わない、referenceがresumeしている。
- 値の比較後は、traceの欠損、判定値の欠損、NaN/Infを`INVALID`とする。閾値が`fixed`でない、または両runの記録と一致しない場合は`BLOCKED`。bitwise一致の件数・割合・最初の差分は、許容差判定とは別のfield（`formal.bitwise`）に出す。

## spec §11との対応と結果

| spec | 確認内容 | 結果 |
|---|---|---|
| 11.1 selector | §5.3表の受理14行（effective、source、legacy）と拒否11行（理由と、errorに必要な項目）。未知値・混在・A1でのnormal meta・Meta runへのmeta selectorをspawn前に拒否（`resolved_config.yaml`以外のartifactなし）。legacy指定値と監視設定の不正値を拒否。既定はeffective `self` | PASS |
| 11.1 伝搬 | in-process 1 rankでSelfAdapterを構築し、`self`／normal meta／A1 S1で`DistributedMoCo.info_nce_impl`がresolver値と一致。provenanceへ記録 | PASS |
| 11.1 rank照合 | gloo 2rankで両modeが一致すれば通過。片rankだけ別mode、または全rankがselectionと違うmodeの場合は、全rankが同じ一覧で停止 | PASS |
| 11.1 fixture | ImageNet fixtureでA2/A3/A4のnormal `meta_einsum`を受理。A1診断metaはsynthetic限定のまま（既存test） | PASS |
| 11.2 identity | effective modeまたはselector sourceが変わるとhashが変わる（A1でsourceだけ違う場合を含む）。gate・steps・trace・loggingの変更では変わらない。v1は従来の規則 | PASS |
| 11.2 v2 Resume | 同modeで再開し、resume証跡を記録。gloo 2rankの実runで、uninterrupted対resumedを`resume`と`formal_resume`で比較し、fixed閾値でPASS | PASS |
| 11.2 不一致の拒否 | 現在config・native metadata・sidecarのmode、selector source、identity schemaの不一致（6条件）を、`load_checkpoint`呼出し0回で拒否 | PASS |
| 11.2 rank間不一致 | gloo 2rankの実runで、rank 1のsidecarだけを改変。deadlockせず両rankが`resume`段階で同じ理由で停止し、run reportは`failed` | PASS |
| 11.2 legacy | 既定null、明示値とeffectiveの不一致、旧configの`self`対明示`meta_einsum`、旧configの欠損・未知値・ファイル不在、旧configや現在configのv1 identity不一致をすべて拒否。条件を満たす場合だけ許可して記録。旧形式へ書き換えたcheckpointを実runでassertion付きで再開し、無中断runとの`resume`比較がPASS | PASS |
| 11.2 state_dict | normal metaとselfのcheckpointで`model` keyの列が同一。既存のkey一致test | PASS |
| 11.3 監視 | disabledではclone・norm・scalar・Comet metricの呼出しが0回で、`completed_steps`は全step。light既定（205step）では、scalarがstep 1・10の倍数・最終、grad normがstep 1・100・200・最終、update normなし、未計測fieldは省略。update間隔3ではrank 0だけがstep 1・3・6・7でclone。rank 1は記録しない | PASS |
| 11.3 窓計測 | 窓の境界（warmup末・窓末・最終step）でのみ同期し、窓ごとのstep数・elapsed・step time・data wait・images/secと、run単位のmedian/p90/meanを手計算値と一致させた。warmup 0では開始前に窓を開く | PASS |
| 11.3 停止 | monitor disabledでも、loss非有限（step 4）、OOM相当の例外、最終stepのQueue非有限で停止し、停止段階と完了step数を記録 | PASS |
| 11.4 比較器 | v3はinfo_nce、v2はfallback、v1はunknown。formal_implementationはMeta対Self metaでPASS（全judged値bitwise）、provisional閾値ではBLOCKED。Self `self`、mode不明、rank不一致、provenance不一致、config不一致、dirty、full step差、NaN、trace欠損はINVALID（Self `self`は1値も読まない）。1ulp差は許容差内PASSでbitwise割合<1・最初の差分を記録、1e-3差はFAIL。通常の`implementation` pairでS0を比較できる。formal_resumeは、resumeしていないcandidateとcheckpoint identityの改変をINVALID | PASS |
| 11.4 A1非回帰 | 既存A1 1step/16step comparator・judgeのPASS/FAIL/BLOCKED/INVALID/NOT_RUNの合成test、比較reportの上書き拒否 | PASS |
| 11.5 非回帰 | 下記 | PASS（注意点1） |

formal比較のCPU testでは、Meta runはCUDA専用のため、Self `meta_einsum` runのreport・config・provenanceをMetaとして付け替えた代役を使った。traceの中身は同一である。これは比較器の契約検査であり、Meta公式との数値一致のEvidenceではない。

### 実行したpytest

```text
# 変更前: 1ce9bad（clean）、同じ.venv・同じ4ファイル
CUDA_VISIBLE_DEVICES="" .venv/bin/python -m pytest -q -p no:cacheprovider -o addopts="" \
  test/model/test_moco_standalone.py test/model/test_moco_cifar10_reporting.py \
  test/model/test_moco_ddp.py test/model/test_moco_parity_harness.py
201 passed in 697.80s (0:11:37)

# 変更後: 実装root（未コミット）、同じコマンド
279 passed in 911.70s (0:15:11)   # 旧201 + 新規78、failed・skip・xfail・xpass 0
```

内訳（変更後のcollect数）は、standalone 23、cifar10_reporting 19、ddp 22（旧18＋4）、parity_harness 215（旧141＋74）。最終実行は変更6ファイルの最終更新（11:54:19 JST）より後に開始し（11:54台開始、12:09:56終了）、以後ファイルは変更していない。所要時間の増加は、主にgloo 2rankのResNet50 CPU run（SMALL寸法・3step）6本と、それを使う比較による。6本のうち2本は、resume段階で停止することを確かめる拒否確認のrunである。

`flake8`（`.flake8`、行長120）では、変更前からある命名規約プラグイン（FNE）の指摘と、既存行の行長2件を除き、新規・変更行に指摘はない。`git diff --check`も問題なし。LoRA・ViT・dataset系のテストはscope外のため実行していない。

## 注意点（Gate判定に関わる解釈）

1. **既存テスト1件のassertionを更新した。** `test_info_nce_impl_defaults_to_self_and_self_keeps_every_gate_available`の`cfg.diagnostic.info_nce_impl == "self"`は、spec §5.1の既定`null`と直接矛盾する。このため「`null`であり、resolverのeffectiveが`self`」の確認へ置き換えた。同じtestの他の検査（各Gateのvalidate、identityに`diagnostic`がないこと）は変えていない。§11.5の「旧201件すべてPASS」は、この1 assertionの更新を含めた結果である。
2. **A1 16step judgeは新しいschemaだけを受け付けるようになる。** judge16（無変更）はrunnerの`RUN_SCHEMA`とcomparatorの`SCHEMA_VERSION`をimportしており、両方がv3になったため、v3のrunとcomparisonを要求する。既存のA1 artifact（run v2 / comparison v2）は、新commitのjudge16では`INVALID`になり再判定できない。旧commitで出た判定（[A1 16step GPU Evidence](2026-10-19-meta-moco-a1-infonce-16step-gpu-parity-evidence.md)）は変わらない。合成reportのjudge testはPASS。
3. **`formal_resume`でも、resume後最初のforward前にあるrank>0のBN bufferは判定しない（informational）。** native checkpointはrank 0だけが保存し、DDPは次のforwardでrank 0の値を再broadcastするためで、既存`resume` pairと同じ扱いである。この規則がないと、ResNet50のrank 1 BN running mean/var 212件でFAILになることをCPU実runで確認した。
4. **benchではlossの有限性検査を毎step残した。** Meta公式モデルは内部に有限性検査を持たないためで、毎step lossの計算完了を待つ小さなhost同期が残る（Selfは`DistributedMoCo`内で元々同期する）。除いたのは毎stepの全device同期（`_sync`）、Query clone、normである。
5. **A1で`diagnostic.info_nce_impl=self`を指定した場合は、既存2 profile以外でも受理する。** 変更前と同じ挙動で、§5.3表の「exact A1 profile」はmeta_einsumの制限と解釈した（既存A1契約の維持、B4/B9）。
6. その他の実装判断:
   - `training.legacy_resume_info_nce_impl`の非null値はimplementation=selfに限る（Metaは演算が固定で、指定は意味を持たないため拒否）。v2 checkpointの再開時には、この値は使わない。
   - formal pairは両runにv3の`info_nce`を要求するため、v2以前のrunは`INVALID`になる。

## 未実施と次の段階

- **研究コードのcommit**: 未実施。GPU Gateの正式runはclean commitが前提になるため、commitが先に必要である（別権限）。
- **G2（A2 short parity）**: ImageNet inventory、実画像2-view fixture、初期state、GPU/VRAM、Gate固有の閾値・full trace step・比較対象を別specで事前固定してから、GPU実行の別承認を得る。
- **G5（A5 overhead/VRAM）**: 同じcommit・hardware・data order・seed・modeで、`monitor.enabled=false`とlight既定のfreshなpairを実行し、`overhead_percent = 100 * (light_median_step_time / off_median_step_time - 1)`をreportする。許容overhead値は実測前に決めない。
- **G3/G4（A3・A4）**: A2/A5のEvidenceの後に別承認。
- 10epoch、表現品質、Linear Probe、Frozen ViT、動画Loader、LoRAは対象外（別spec）。

GPU実行が承認された場合の起動例（Gate固有の閾値・fixtureは別specで固定してから使う）:

```text
# A5: 通常経路でMeta算術を明示opt-in（off/lightのoverhead pair）
CUDA_VISIBLE_DEVICES=0,1 python -m scripts.parity.moco_imagenet_parity run_kind=bench gate=A5 \
    moco.info_nce_impl=meta_einsum monitor.enabled=false
CUDA_VISIBLE_DEVICES=0,1 python -m scripts.parity.moco_imagenet_parity run_kind=bench gate=A5 \
    moco.info_nce_impl=meta_einsum
# A2: Meta MとSelf S1を別directoryで独立実行し、オフラインで比較
CUDA_VISIBLE_DEVICES=0,1 python -m scripts.parity.moco_imagenet_parity implementation=meta_reference gate=A2 \
    fixture.path=<fixture> verification.threshold_status=fixed verification.rtol=<A2> verification.atol=<A2>
CUDA_VISIBLE_DEVICES=0,1 python -m scripts.parity.moco_imagenet_parity implementation=self gate=A2 \
    moco.info_nce_impl=meta_einsum fixture.path=<fixture> verification.threshold_status=fixed \
    verification.rtol=<A2> verification.atol=<A2>
python -m scripts.parity.compare_moco_parity --pair-kind formal_implementation --reference <M> --candidate <S1> \
    --threshold-status fixed --rtol <A2> --atol <A2>
```

## 更新履歴

| 日付 | 内容 |
|---|---|
| 2026-10-19 | 実装とCPU検証（変更前201 passed → 変更後279 passed）。Engineering/CPU Gate PASS。GPU Gateは未実施、研究コードは未コミット |
