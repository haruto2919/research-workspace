# Notion保存先の設定

## 設定ファイル

通常利用するNotion保存先は、Skillディレクトリ内の次のファイルに記録する。

```text
config/notion-destination.local.json
```

同じフォルダの`notion-destination.example.json`は未設定状態の見本である。認証トークンはこの設定ファイルへ書かない。

## このSkillに設定済みの保存先

この配布物の`notion-destination.local.json`には、次のデータソースが既定保存先として設定済みである。

```json
{
  "schema_version": 2,
  "enabled": true,
  "destination_type": "data_source",
  "workspace_name": "nitech_notion",
  "data_source_title": "研究ノート_DB",
  "data_source_id": "e8cad5f0-b4be-83a1-9c58-0724c9251ad1",
  "property_mapping": {
    "title": "名前",
    "created_date": "作成日",
    "keywords": "キーワード"
  }
}
```

そのため、通常の利用時はNotion URLやIDを毎回指定しなくてよい。ただし保存の意思は毎回明示する。

## データソース設定の項目

- `schema_version`: `2`固定。
- `enabled`: 既定保存先として利用する場合は`true`。
- `destination_type`: データソースへ追加する場合は`data_source`。
- `workspace_name`: Notionで接続中のワークスペース名。
- `data_source_title`: データソースの正確なタイトル。
- `data_source_id`: NotionデータソースID。
- `property_mapping.title`: タイトル型プロパティの名前。
- `property_mapping.created_date`: 作成日を入れる日付型プロパティの名前。
- `property_mapping.keywords`: キーワードを入れる複数選択型プロパティの名前。

保存のたびにデータソースをIDで取得し、タイトルとワークスペースに加えて、上記3プロパティの存在と型を現在のNotionスキーマに照合する。設定が有効でも、実体と一致しなければ書き込まない。

## 通常ページへ保存する場合

通常ページの子ページとして保存したい場合は、次の形式へ変更する。

```json
{
  "schema_version": 2,
  "enabled": true,
  "destination_type": "page",
  "workspace_name": "研究用Notion",
  "parent_page_title": "壁打ちアーカイブ",
  "parent_page_url": "https://www.notion.so/壁打ちアーカイブ-0123456789abcdef0123456789abcdef",
  "parent_page_id": "01234567-89ab-cdef-0123-456789abcdef"
}
```

`parent_page_url`または`parent_page_id`のどちらか一方は必須である。両方を書く場合は同じページを指す必要がある。

## ローカル形式の検証

Research Workspaceのルートから次を実行する。

```bash
python3 .agents/skills/brainstorm-to-notion/scripts/validate_destination.py \
  .agents/skills/brainstorm-to-notion/config/notion-destination.local.json
```

`"valid": true`と正規化された`data_source_id`または`parent_page_id`が表示されれば、設定ファイルの形式は有効である。この検証だけでは、Notion上の存在、権限、最新スキーマは確認できない。それらは実際の保存直前に検証する。

## 保存先の優先順位

1. 現在の依頼で明示されたNotion URL／ID
2. `config/notion-destination.local.json`

現在の依頼で別URL／IDを指定しても、設定ファイルは変更しない。既定保存先を書き換える場合は、ユーザーが設定変更を明示したときだけ編集する。

## 秘密情報を置かない

このファイルへNotion APIトークン、OAuthトークン、パスワード、Cookieを記入しない。URLとIDは認証情報ではないが、公開リポジトリではワークスペース構成を推測する手掛かりになる。このため`notion-destination.local.json`は同梱の`.gitignore`でGit管理から除外する。
