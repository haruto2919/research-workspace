# `no-image.md` 資料保管庫コピー dry run

## 結果

- 判定: PASS
- 実行日: 2026-07-19 JST
- 対象: 隔離Company rootへ複製した `.company/no-image.md` 1件
- title: `画像なし検証資料`
- 画像数: 0件（local 0件、remote 0件）
- source SHA-256: `82a1c02064f306562d1c009d79431ce55841fab482e4e3de72a172061f89170e`
- source size: 209 bytes
- helper警告: なし
- Notion書き込み: 0件
- HTTP server起動: なし
- SSH tunnel作成: なし

fixtureそのものは編集していない。隔離コピーは`prepare`前後ともfixtureと同じSHA-256であり、本文は変更されていない。本実行では209 bytesの元Markdown全文を先頭から末尾までcontentとして使用し、frontmatter、H1 `# 画像なし検証資料`、表、コードブロックを保持する。ページtitleにH1を使用しても、本文中のH1は削除しない。画像がないため、renderや画像markup置換、一時HTTPS転送は行わない。

## 重複確認

このdry runではNotionを呼び出していないため、実在する重複候補の有無は未確認である。本実行時は、固定parent page ID `3a284249-7e4a-81ef-8944-e5f6efb048a5`をfetchし、parent titleが`資料保管庫`であることを確認した後、既存child page title一覧と`画像なし検証資料`をexact matchで比較する。

- exact title matchあり: 上書きも複製もせずskipし、既存page URLを報告する。
- exact title matchなし: 固定parent直下に1 pageだけ作成する。
- 固定parentを取得できない、またはtitle不一致: 代替parentを作らず停止する。

## 本実行後の完了検証

作成したpageをfetchし、次を確認する。

1. ancestorが固定parent page ID `3a284249-7e4a-81ef-8944-e5f6efb048a5`である。
2. page titleがmanifest title `画像なし検証資料`と一致する。
3. 元Markdownの先頭と末尾の非空内容が存在し、frontmatter、H1、表、`print("keep exactly")`を含むコードブロックの可視内容と順序が維持されている。
4. Notion image block数がlocal image数の0件と一致する。
5. source MarkdownのSHA-256がpreflight値 `82a1c02064f306562d1c009d79431ce55841fab482e4e3de72a172061f89170e`から変化していない。
6. source image検証は画像0件のため該当なし。

検証失敗時もpageを自動削除・上書きせず、作成済みpage URLと不一致内容を報告する。

## コマンド証跡

日付確認:

```console
$ date '+%Y-%m-%d %H:%M:%S %Z'
2026-07-19 15:38:00 JST
```

fixtureを隔離Company rootへコピーし、コピー前の同一性を確認:

```console
$ mkdir -p /mnt/HDD8TB-2/takama/Takama-Systems-Group/.agents/skills/notion-material-archive-workspace/iteration-1/eval-1-no-image/with_skill/outputs/test-workspace/.company && cp /mnt/HDD8TB-2/takama/Takama-Systems-Group/.agents/skills/notion-material-archive/evals/files/no-image.md /mnt/HDD8TB-2/takama/Takama-Systems-Group/.agents/skills/notion-material-archive-workspace/iteration-1/eval-1-no-image/with_skill/outputs/test-workspace/.company/no-image.md && sha256sum /mnt/HDD8TB-2/takama/Takama-Systems-Group/.agents/skills/notion-material-archive/evals/files/no-image.md /mnt/HDD8TB-2/takama/Takama-Systems-Group/.agents/skills/notion-material-archive-workspace/iteration-1/eval-1-no-image/with_skill/outputs/test-workspace/.company/no-image.md
82a1c02064f306562d1c009d79431ce55841fab482e4e3de72a172061f89170e  /mnt/HDD8TB-2/takama/Takama-Systems-Group/.agents/skills/notion-material-archive/evals/files/no-image.md
82a1c02064f306562d1c009d79431ce55841fab482e4e3de72a172061f89170e  /mnt/HDD8TB-2/takama/Takama-Systems-Group/.agents/skills/notion-material-archive-workspace/iteration-1/eval-1-no-image/with_skill/outputs/test-workspace/.company/no-image.md
```

スキルhelperの`prepare`のみを実行:

```console
$ python3 /mnt/HDD8TB-2/takama/Takama-Systems-Group/.agents/skills/notion-material-archive/scripts/material_archive.py prepare /mnt/HDD8TB-2/takama/Takama-Systems-Group/.agents/skills/notion-material-archive-workspace/iteration-1/eval-1-no-image/with_skill/outputs/test-workspace/.company/no-image.md --workspace-root /mnt/HDD8TB-2/takama/Takama-Systems-Group/.agents/skills/notion-material-archive-workspace/iteration-1/eval-1-no-image/with_skill/outputs/test-workspace
{"manifest": "/tmp/notion-material-archive-o7rsw9y1/manifest.json", "documents": 1, "local_images": 0, "remote_images": 0, "stage_dir": "/tmp/notion-material-archive-o7rsw9y1", "assets_dir": "/tmp/notion-material-archive-o7rsw9y1/assets"}
```

`prepare`後にfixtureと隔離コピーの同一性を再確認:

```console
$ sha256sum /mnt/HDD8TB-2/takama/Takama-Systems-Group/.agents/skills/notion-material-archive/evals/files/no-image.md /mnt/HDD8TB-2/takama/Takama-Systems-Group/.agents/skills/notion-material-archive-workspace/iteration-1/eval-1-no-image/with_skill/outputs/test-workspace/.company/no-image.md
82a1c02064f306562d1c009d79431ce55841fab482e4e3de72a172061f89170e  /mnt/HDD8TB-2/takama/Takama-Systems-Group/.agents/skills/notion-material-archive/evals/files/no-image.md
82a1c02064f306562d1c009d79431ce55841fab482e4e3de72a172061f89170e  /mnt/HDD8TB-2/takama/Takama-Systems-Group/.agents/skills/notion-material-archive-workspace/iteration-1/eval-1-no-image/with_skill/outputs/test-workspace/.company/no-image.md
```

## cleanup結果

helperが作成した限定的な一時stagingをcleanupした。

```console
$ python3 /mnt/HDD8TB-2/takama/Takama-Systems-Group/.agents/skills/notion-material-archive/scripts/material_archive.py cleanup --manifest /tmp/notion-material-archive-o7rsw9y1/manifest.json
{"removed": "/tmp/notion-material-archive-o7rsw9y1", "exists": false}
```

隔離テストrootも削除した。

```console
$ rm -r /mnt/HDD8TB-2/takama/Takama-Systems-Group/.agents/skills/notion-material-archive-workspace/iteration-1/eval-1-no-image/with_skill/outputs/test-workspace
$ find /mnt/HDD8TB-2/takama/Takama-Systems-Group/.agents/skills/notion-material-archive-workspace/iteration-1/eval-1-no-image/with_skill/outputs -maxdepth 1 -name test-workspace -print
```

最後の`find`出力は空であり、隔離テストrootは残っていない。HTTP serverとSSH tunnelは一度も起動していないため、停止対象のsession、listen port、関連processはない。
