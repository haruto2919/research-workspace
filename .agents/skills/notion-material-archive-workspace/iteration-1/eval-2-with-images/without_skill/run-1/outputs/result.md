# 資料保管庫コピー dry run 結果

- 判定: PASS
- モード: dry run（Notionへの読み書き、画像のHTTPS公開、HTTP/HTTPS server、SSH tunnelはいずれも未実行）
- source: `/mnt/HDD8TB-2/takama/Takama-Systems-Group/.agents/skills/notion-material-archive/evals/files/with-images.md`
- title: `複数画像付き検証資料`
- source SHA-256: `7e4272ce6b52530705ee1758f95558e40fd8b1a89b2f39556b1295711d6b308e`
- source size: `218 bytes`
- source lines: `16`
- 参照画像: `2件`（ローカル相対参照2件、remote参照0件）

## 参照画像manifest

参照先はsource Markdownの親ディレクトリを基準に解決した。どちらも通常ファイルとして存在する。

| 順序 | source line | 構文 | 記載参照 | 解決済み絶対path | alt | size | SHA-256 |
|---:|---:|---|---|---|---|---:|---|
| 1 | 9 | Markdown image | `images/first.svg` | `/mnt/HDD8TB-2/takama/Takama-Systems-Group/.agents/skills/notion-material-archive/evals/files/images/first.svg` | `first alt` | 211 bytes | `726f79d7ca5b0d1741961b8cb4ae16d7beb636250c45e3ae121a72a9d8760066` |
| 2 | 13 | HTML `img` | `images/second.svg` | `/mnt/HDD8TB-2/takama/Takama-Systems-Group/.agents/skills/notion-material-archive/evals/files/images/second.svg` | `second alt` | 212 bytes | `21c63dd761894ef36dff873cafee86f4f829e557d2e26e513d0541aafe1db30d` |

## 原文で置換を許可するmarkup

live runで一時HTTPS URLを用意できた場合に限り、以下の2箇所にあるローカル参照文字列だけを、対応する一時HTTPS URLへ置換する。見出し、frontmatter、段落、空行、alt text、引用符、画像順序、およびその他の本文は変更しない。

1. line 9
   - 原文: `![first alt](images/first.svg)`
   - 許可する変換: `![first alt](<FIRST_HTTPS_URL>)`
   - 実際の置換token: `images/first.svg` のみ
2. line 13
   - 原文: `<img src="images/second.svg" alt="second alt">`
   - 許可する変換: `<img src="<SECOND_HTTPS_URL>" alt="second alt">`
   - 実際の置換token: `src` 属性値の `images/second.svg` のみ

`<FIRST_HTTPS_URL>` と `<SECOND_HTTPS_URL>` は説明用placeholderであり、本dry runでは実URLを発行していない。置換後のpayloadを作る場合は、上記2 token以外をsourceとbyte単位で比較し、差分があればNotion書き込み前に停止する。

## live run時の検証計画

1. title完全一致で資料保管庫を検索し、同一資料の重複作成を防ぐ。同名で内容が異なる場合は上書きせず停止する。
2. 各公開URLから取得したbyte列のSHA-256がmanifestの対応値と一致することをNotion書き込み前に確認する。
3. 作成後のpageを再取得し、title、本文順序、alt text、画像数2件、および画像対応順が一致することを確認する。
4. title完全一致検索を再実行し、この処理によって重複pageが増えていないことを確認する。

## 必須cleanupと現在のstatus

live runでは成功・失敗を問わず、検証後に次を必ず行う。

- 一時HTTP/HTTPS serverを停止し、processが残っていないことを確認する。
- SSH tunnelを使用した場合は終了し、listener/processが残っていないことを確認する。
- stagingした画像copy、一時設定、一時log、一時directoryを削除する。
- 外部hostへ画像を配置した場合はremote copyと公開URLを削除または失効させ、取得不能になったことを確認する。
- cleanup後も元fixture 3ファイルを削除・変更しない。

本dry runのcleanup statusは `clean`。Notion呼び出し0件、Notion書き込み0件、HTTP/HTTPS server起動0件、SSH tunnel作成0件、公開URL発行0件、一時resource作成0件のため、削除対象はない。

## Evidence

- `sha256sum with-images.md`: `7e4272ce6b52530705ee1758f95558e40fd8b1a89b2f39556b1295711d6b308e`
- `sha256sum first.svg`: `726f79d7ca5b0d1741961b8cb4ae16d7beb636250c45e3ae121a72a9d8760066`
- `sha256sum second.svg`: `21c63dd761894ef36dff873cafee86f4f829e557d2e26e513d0541aafe1db30d`
- `stat`: source `218 bytes`、first.svg `211 bytes`、second.svg `212 bytes`
- `wc -l with-images.md`: `16`
- title根拠: source内の最初のH1 `# 複数画像付き検証資料`
- source inode / mtime: `219491657` / `2026-07-19 15:34:52.127492436 +0900`
- first.svg inode / mtime: `219491659` / `2026-07-19 15:34:52.128202706 +0900`
- second.svg inode / mtime: `219491660` / `2026-07-19 15:34:52.128604239 +0900`

