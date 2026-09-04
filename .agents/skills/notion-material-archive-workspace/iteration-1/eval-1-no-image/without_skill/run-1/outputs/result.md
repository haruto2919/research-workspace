# 資料保管庫コピー dry run 結果

- 判定: PASS
- モード: dry run（Notion への読み書きなし）
- 対象 source: `/mnt/HDD8TB-2/takama/Takama-Systems-Group/.agents/skills/notion-material-archive/evals/files/no-image.md`
- 対象 title: `画像なし検証資料`
- 画像数: `0`
- source SHA-256: `82a1c02064f306562d1c009d79431ce55841fab482e4e3de72a172061f89170e`
- source サイズ: `209 bytes`
- source 行数: `19`

## 本文の取り扱い

本文は変更しない。実行時も source の内容をそのままコピー対象とし、要約、言い換え、見出し変更、表の再構成、コードブロックの整形、frontmatter の変更を行わない。送信 payload を作成した直後に、その本文から SHA-256 を再計算し、上記 source SHA-256 と一致しない場合は Notion 書き込み前に停止する。

## 実行時に行う重複確認

1. 資料保管庫を title の完全一致 `画像なし検証資料` で検索する。
2. 一致候補ごとに page ID、title、既存の source SHA-256（保存されている場合）、本文を取得する。
3. 同一 title かつ同一 source SHA-256、または本文が source と一致するページがあれば、同一資料として新規作成をスキップする。
4. 同一 title だが SHA-256 または本文が異なる場合は上書きせず、衝突として停止して報告する。
5. 完全一致候補がなければ、新規ページを 1 件だけ作成する。

## 実行時に行う完了検証

1. 作成または既存判定した page を page ID で再取得する。
2. title が `画像なし検証資料` と完全一致することを確認する。
3. 本文の全ブロックを順番どおり取得し、source と照合する。見出し、段落、表、コードブロックの内容と順序に差分がないことを確認する。
4. source SHA-256 が記録対象の場合は、値が `82a1c02064f306562d1c009d79431ce55841fab482e4e3de72a172061f89170e` と一致することを確認する。
5. title 完全一致の検索を再実行し、この処理により重複ページが増えていないことを確認する。
6. いずれかが不一致なら完了扱いにせず、page ID と差分を報告する。

## Evidence

- `sha256sum` の観測値: `82a1c02064f306562d1c009d79431ce55841fab482e4e3de72a172061f89170e`
- `wc` の観測値: `19 lines`, `209 bytes`
- Markdown 画像構文 `![...](...)` および HTML `<img>` の検索結果: `0件`
- title の根拠: source 内の最初の H1 `# 画像なし検証資料`
- fixture の確認時 inode: `219491656`
- fixture の確認時 mtime: `2026-07-19 15:34:52.126306440 +0900`

## Cleanup status

- Notion 呼び出し: `0`
- Notion 作成・更新・削除: `0`
- HTTPS server 起動: `0`
- SSH tunnel 作成: `0`
- 一時ファイル・一時 URL: `0`
- fixture および実 Company content の変更: `なし`
- cleanup: `不要（外部 resource・一時 resource を作成していない）`
