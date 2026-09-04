# with-images.md dry-run result

## 結果

- 判定: 成功
- 実行日: 2026-07-19 (JST)
- 実行範囲: `prepare` と必須 `cleanup` のみ
- Notion read/write: 未実行
- HTTP/HTTPS server: 未起動
- SSH tunnel: 未起動
- render: 未実行
- fixture / real Company content: 未編集

隔離した Company test root:

`/mnt/HDD8TB-2/takama/Takama-Systems-Group/.agents/skills/notion-material-archive-workspace/iteration-1/eval-2-with-images/with_skill/outputs/isolated-company-root`

対象:

`isolated-company-root/.company/with-images.md`

title は最初の H1 から `複数画像付き検証資料`、source SHA-256 は
`7e4272ce6b52530705ee1758f95558e40fd8b1a89b2f39556b1295711d6b308e`
と判定された。警告はなく、local image は2件、remote image は0件だった。
Notionを呼ばないdry runのため、既存child pageとの重複候補確認は未実施である。

## Exact helper evidence

実行した helper command:

```bash
python3 .agents/skills/notion-material-archive/scripts/material_archive.py prepare \
  "/mnt/HDD8TB-2/takama/Takama-Systems-Group/.agents/skills/notion-material-archive-workspace/iteration-1/eval-2-with-images/with_skill/outputs/isolated-company-root/.company/with-images.md" \
  --workspace-root "/mnt/HDD8TB-2/takama/Takama-Systems-Group/.agents/skills/notion-material-archive-workspace/iteration-1/eval-2-with-images/with_skill/outputs/isolated-company-root" \
  --output "/mnt/HDD8TB-2/takama/Takama-Systems-Group/.agents/skills/notion-material-archive-workspace/iteration-1/eval-2-with-images/with_skill/outputs/prepare-manifest.json"
```

helper stdout（原文）:

```json
{"manifest": "/mnt/HDD8TB-2/takama/Takama-Systems-Group/.agents/skills/notion-material-archive-workspace/iteration-1/eval-2-with-images/with_skill/outputs/prepare-manifest.json", "documents": 1, "local_images": 2, "remote_images": 0, "stage_dir": "/tmp/notion-material-archive-wli1ez1x", "assets_dir": "/tmp/notion-material-archive-wli1ez1x/assets"}
```

manifest は `prepare-manifest.json` に保存した。抽出された参照画像manifestは次のとおり。

| asset_id | kind | source target | original markup | offsets | staged filename | bytes | content type |
| --- | --- | --- | --- | --- | --- | ---: | --- |
| `asset-0001` | `markdown-inline` | `images/first.svg` | `![first alt](images/first.svg)` | `[50, 80)` | `0001-726f79d7ca5b-first.svg` | 211 | `image/svg+xml` |
| `asset-0002` | `html` | `images/second.svg` | `<img src="images/second.svg" alt="second alt">` | `[92, 138)` | `0002-21c63dd76189-second.svg` | 212 | `image/svg+xml` |

staging assets directoryには上記2ファイルだけが存在した。

## Image checksums

| image | SHA-256 | fixture / isolated copy / staged copy |
| --- | --- | --- |
| `first.svg` | `726f79d7ca5b0d1741961b8cb4ae16d7beb636250c45e3ae121a72a9d8760066` | 3者一致、byte-identical |
| `second.svg` | `21c63dd761894ef36dff873cafee86f4f829e557d2e26e513d0541aafe1db30d` | 3者一致、byte-identical |

隔離コピーのMarkdownもfixtureとbyte-identicalだった。

## Replacement markup plan

dry runではuploadもrenderも行っていないため、置換はまだ適用していない。本実行時に置換するのは、以下のローカル画像markupだけである。

1. source文字offset `[50, 80)` の
   `![first alt](images/first.svg)` を、attachment responseが返す
   `upload_map.assets.asset-0001` の `suggested_markdown` で置換する。
2. source文字offset `[92, 138)` の
   `<img src="images/second.svg" alt="second alt">` を、attachment responseが返す
   `upload_map.assets.asset-0002` の `suggested_markdown` で置換する。

frontmatter、H1、各文章、改行、その他のmarkupは置換対象にしない。render時はhelperが元markupとoffsetを再照合し、後方のoffsetから置換する。

## 必須cleanup

実行した helper command:

```bash
python3 .agents/skills/notion-material-archive/scripts/material_archive.py cleanup \
  --manifest "/mnt/HDD8TB-2/takama/Takama-Systems-Group/.agents/skills/notion-material-archive-workspace/iteration-1/eval-2-with-images/with_skill/outputs/prepare-manifest.json"
```

helper stdout（原文）:

```json
{"removed": "/tmp/notion-material-archive-wli1ez1x", "exists": false}
```

- `/tmp/notion-material-archive-wli1ez1x`: 削除確認済み
- SSH tunnel session: 未起動のため停止対象なし
- local HTTP server session: 未起動のため停止対象なし
- listen port: 割当なし
- HTTPS公開: なし
- 隔離Company test root: 評価入力の証跡としてoutputs配下に保持

