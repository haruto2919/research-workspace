---
name: notion-material-archive
description: Company内のMarkdown資料を、本文を要約・改変せず、参照画像も原本のままNotionの「資料保管庫」へコピーするスキル。「Notionにコピーして」「資料保管庫へ保存して」「このMarkdownをそのままNotionへ」「このディレクトリの資料をNotionに入れて」など、`.company/` 配下のMarkdownをNotionへ保管・転記・アーカイブする依頼では必ず使う。単一ファイル、ディレクトリ直下の一括処理、明示された再帰処理、画像の一時HTTPS転送、重複防止、転送後検証、確実な後始末まで扱う。
---

# Notion Material Archive

Company内のMarkdownを、元ファイルを編集せずNotionの固定親ページ「資料保管庫」へ保存する。Notionへ渡す生成物はEnhanced Markdownへ決定論的に変換してよいが、本文の可視テキスト、順序、H1、frontmatter、表、数式、コードを削除・要約・言い換えしない。

## 固定設定

| 項目 | 値 |
| --- | --- |
| Company root | `/mnt/HDD8TB-2/takama/Takama-Systems-Group/.company` |
| Notion workspace | `髙間勇作のNotion` |
| Parent title | `資料保管庫` |
| Parent page ID | `3a284249-7e4a-81ef-8944-e5f6efb048a5` |
| Parent URL | `https://app.notion.com/p/3a2842497e4a81ef8944e5f6efb048a5` |
| Supported source | Markdown (`.md`) |
| Local image types | PNG, JPEG, WebP, SVG |

親IDは同名ページの検索結果で置き換えない。固定親を取得できない、またはtitleが一致しない場合は、新しい親ページを作らず停止する。

## 忠実度の定義

- 元Markdownと元画像を編集しない。source SHA-256をprepare時と完了時に照合する。
- Markdown本文を先頭から末尾まで使用し、H1やfrontmatterも残す。
- ページtitleには最初のH1を使い、H1がなければfilename stemを使う。
- 要約、注釈、コピー日時、source path、説明文を本文へ追加しない。
- Notion用の一時生成物に限り、可視内容を保つため次の構文正規化を許可する。
  - コード外のインライン数式 `\(...\)` を `$...$` に変換する。
  - コード外の独立数式 `\[...\]` を `$$...$$` に変換する。
  - Markdown表の区切りにある `:---`、`---:`、`:---:` を `---` に正規化する。
  - ローカル画像markupを、Notionが返した`file-upload://...`のimage blockへ置換する。
- HTTPS画像などのremote image markupはそのまま残す。
- 通常リンク、ローカルファイルリンク、画像以外のattachmentは初版では変更しない。
- Markdown構文・空白のbyte-for-byte保存用attachmentは追加しない。
- fenced code、Mermaid、inline codeの内部は数式・表の正規化対象にせず、元の文字列を保持する。

NotionはMarkdownをnative blockへ正規化するため、block表現や空白が元ファイルのbyte列と同一になることは保証しない。ここでの「そのまま」は、元ファイルを不変に保ち、Notion上の可視内容、順序、数式本体、表構造、コードを維持する意味である。

## 対象の決め方

1. ユーザーが指定したpathを使う。分かっているpathを聞き直さない。
2. 単一ファイルなら、その`.md`だけを処理する。
3. ディレクトリなら、標準では直下の`*.md`だけをfilename順で処理する。
4. 「再帰的に」「配下すべて」など明示された場合だけ、subdirectoryも再帰処理する。
5. `.company/`全体を無指定で一括同期しない。
6. source Markdownが`.company/`外なら停止する。参照画像も標準では`.company/`内だけを許可する。

## Workflow

### 1. Preflight

1. `.company/CLAUDE.md`を読む。
2. 今日の日付を確認する。
3. Notion toolsを確認し、Notion workspace identityをfetchする。
4. 固定parent pageをfetchし、titleが`資料保管庫`であることを確認する。
5. Notion enhanced Markdown resourceが利用可能なら読む。resourceが利用できない場合は、attachment toolが返す`suggested_markdown`を画像markupの正本として使う。
6. 次のhelperでsource一覧、title、SHA-256、画像参照、Notionコンパイルpreview、staging directoryを作る。

```bash
python3 .agents/skills/notion-material-archive/scripts/material_archive.py prepare \
  "<source-path>" \
  --workspace-root "/mnt/HDD8TB-2/takama/Takama-Systems-Group"
```

再帰指定時だけ`--recursive`を付ける。helperはコード外の数式delimiterと表区切りをpreflightし、manifestの`notion_compilation_preview`へ変換数、数式hash、表構造、コードブロック数を記録する。missing image、unsupported local image、scope外path、不均衡な数式delimiter、不正な表構造を報告した場合はNotionへ書き込まず停止する。

### 2. 重複確認

parent pageのfetch結果にある既存child page titleと、manifest内のtitleを比較する。

- exact title matchがあれば、そのMarkdownは上書きも複製もせずskipする。
- skipした既存page URLを完了報告に含める。
- directory batchでは全titleを先に確認し、作成対象だけを確定する。
- source同士で同じtitleが衝突する場合も作成せず、該当fileを報告する。

### 3. 画像なし資料

画像がないMarkdownは一時serverを使わない。ただしraw sourceを直接Notionへ渡してはいけない。必ずhelperの`render`を実行し、数式と表をNotion Enhanced Markdownへ正規化した`rendered_path`をcontentとして使う。page titleを本文から削除しない。

```bash
python3 .agents/skills/notion-material-archive/scripts/material_archive.py render \
  --manifest "<temporary-manifest.json>" \
  --output-dir "<temporary-rendered-dir>"
```

### 4. ローカル画像の一時HTTPS転送

manifestにlocal imageがある場合だけ行う。この一時転送はユーザーから恒常的に許可されているため、呼び出しごとの確認は不要。

1. manifestの`assets_dir`には参照画像だけがbyte-for-byte copyされていることを確認する。
2. bundled serverをPTY sessionで起動する。directory listingは無効である。

```bash
python3 .agents/skills/notion-material-archive/scripts/serve_staged.py \
  --directory "<assets_dir>" \
  --port 0
```

3. 出力された`PORT=<port>`を使い、別PTY sessionでSSH reverse tunnelを起動する。

```bash
ssh -o StrictHostKeyChecking=accept-new -o ServerAliveInterval=30 \
  -R 80:localhost:<port> nokey@localhost.run
```

4. tunnel出力のHTTPS originを取得する。
5. 各staged filenameへ`curl -fsSIL --max-time 15`を実行し、200 responseを確認する。
6. Notion attachment toolへ`filename`と`source_url`を渡し、全画像が`uploaded`になるまでpageを作らない。
7. attachment responseの`suggested_markdown`を、asset IDごとのupload map JSONへ保存する。
8. fallbackとして別のpublic file hostingへuploadしない。tunnelが使えない場合は停止して報告する。

### 5. Notion用content生成

全資料をhelperの`render`へ通す。local imageがある場合はupload mapを使って画像markupを置換し、その後に数式と表をNotion向けへ正規化する。

```bash
python3 .agents/skills/notion-material-archive/scripts/material_archive.py render \
  --manifest "<temporary-manifest.json>" \
  --upload-map "<temporary-upload-map.json>" \
  --output-dir "<temporary-rendered-dir>"
```

`render-manifest.json`の各`rendered_path`を読み、対応するtitleで固定parent配下へpageを作る。directory batchでも、1 pageにつき1 Markdownを対応させる。

`render-manifest.json`の`notion_compilation.profile`を登録後検証の正本にする。profileには次が含まれる。

- inline/display数式の件数と数式本体SHA-256
- 表数、および各表のrow数、column数、header有無、header/content SHA-256
- fenced code block数、Mermaid block数
- 正規化した数式delimiter数と表separator数

### 6. 検証

作成した各pageをfetchし、次を確認する。

- ancestorが固定parentである。
- page titleがmanifest titleと一致する。
- 元Markdownの先頭と末尾の非空textが存在する。
- Notion image block数がlocal image数と一致する。
- inline/display数式の件数と本体SHA-256が`notion_compilation.profile`と一致する。
- 表数、各表のrow数・column数・header/content SHA-256がprofileと一致し、全表がheader rowを持つ。
- `---`、`---:`、`:---`、`:---:`だけからなるseparatorがdata rowとして残っていない。
- fenced code block数とMermaid block数がprofileと一致する。
- source Markdownとsource imageのSHA-256がpreflight時から変化していない。
- 可能ならfetchで得たNotion image signed URLをdownloadし、SHA-256を原画像と比較する。Notion側の変換でbyte比較できない場合はfilenameとcontent lengthを最低限確認し、その旨を報告する。

fetchで取得したpageの`<content>`をstage directory内の一時UTF-8 fileへ保存できる場合は、次のauditを実行する。

```bash
python3 .agents/skills/notion-material-archive/scripts/material_archive.py audit \
  --render-manifest "<render-manifest.json>" \
  --page-content "<fetched-page-content.md>" \
  --document-index 1
```

directory batchでは`--document-index`を1始まりで対応させる。auditが`verified: false`またはexit code 3を返した場合は検証失敗である。tool surface上、一時fileへ保存できない場合だけ、同じprofile項目をfetch結果から直接比較する。

検証に失敗したpageを自動削除・上書きしない。「成功」と報告せず、作成済みpage URLと不一致内容を報告する。

### 7. 必須cleanup

成功・失敗にかかわらず、作業終了前に必ず行う。

1. SSH tunnel sessionへCtrl-Cを送り、終了を確認する。
2. local HTTP server sessionへCtrl-Cを送り、終了を確認する。
3. listen portと関連processが残っていないことを確認する。
4. helperでtemporary stagingとrendered filesを削除する。

```bash
python3 .agents/skills/notion-material-archive/scripts/material_archive.py cleanup \
  --manifest "<temporary-manifest.json>"
```

cleanup対象はhelperが作った`/tmp/notion-material-archive-*`だけに限定する。広いpathへ`rm -rf`を実行しない。

## Upload map format

`render`へ渡すJSONは次の形にする。

```json
{
  "assets": {
    "asset-0001": "<image src=\"file-upload://...\"></image>"
  }
}
```

keyはprepare manifestの`asset_id`と一致させ、valueにはNotion attachment responseの`suggested_markdown`をそのまま使う。

## Dry run

ユーザーが「確認だけ」「dry run」と指定した場合は、`prepare`まで実行し、Notion write、HTTP server、SSH tunnelを行わない。対象Markdown、title、画像数、source SHA-256に加え、`notion_compilation_preview`の数式数、表数、正規化数、コードブロック数、警告、重複候補を報告してcleanupする。

## 完了報告

簡潔に次を返す。

```markdown
コピー完了:
- 作成: N件
- 画像: N件
- スキップ: N件
- 検証: 成功 / 要確認
- 一時HTTPS転送: 停止確認済み

- [ページタイトル](Notion URL)
```

失敗時は、作成済みpage、未作成file、cleanup結果を分けて報告する。

## 禁止事項

- sourceの要約・再構成・校正。
- source Markdown、画像、Company README、TODOの更新。
- 画像なし資料でraw sourceを直接Notion contentへ渡すこと。
- fenced code、Mermaid、inline code内の数式・表らしき文字列を正規化すること。
- 数式・表のround-trip検証を行わず「成功」と報告すること。
- 固定parentが見つからない場合の代替parent作成。
- 同名pageの無断更新・複製。
- `.company/`外Markdownの処理。
- image transfer以外のfileを一時公開すること。
- Git commit/push。
