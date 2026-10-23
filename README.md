# 🔍 WikiLens — Wikipedia画像連想ゲーム

Wikipediaの象徴的・特徴的な「画像」から対象記事を推理するWebブラウザゲームです。  
入力した単語と正解記事との「意味的な近さ（ランキング 1位〜20,000位）」を手がかりに、1位（正解記事名）を目指します。

## ✨ 特徴

- **完全0円運用（静的ホスティング＋事前バッチ計算）**
  - 推論APIやデータベースサーバーを一切使わず、ローカルで事前計算した静的JSONとHTML/CSS/Vanilla JSのみで動作します。
- **ネタバレ防止＆Wikimedia規約準拠**
  - Wikimedia Commonsのサムネイル画像をバッチ処理時に日付ID名（`./data/images/YYYY-MM-DD.jpg`）で保存し、画像ファイル名からの正解ネタバレを防止しています。
  - Creative Commonsライセンス（CC BY-SA等）の帰属表示（著作者名・ライセンスリンク・元ファイルへのリンク）を画像下部に明記しています。
- **ハイブリッド意味類似度ランキング**
  - 多言語埋め込みモデル `intfloat/multilingual-e5-small` によるベクトル類似度（タイトル・カテゴリ・導入部概要）と、Wikipedia記事の内部リンク・本文共起ボーナスを組み合わせ、直感的で納得感の高い約20,000語のランキングを実現しています。
- **表記ゆれ・別名（リダイレクト）対応**
  - Wikipediaのリダイレクトや読み仮名（例：「金閣寺」と「鹿苑寺」）を別名として登録し、どちらを入力しても1位（正解）として判定します。

## 🚀 ローカルでのテストプレイ方法

静的ファイルのみで動作するため、任意のローカルHTTPサーバーで起動できます。

```bash
python3 -m http.server 8000
```

ブラウザで `http://localhost:8000` にアクセスしてください。

## 🛠️ 新しい問題の生成方法（バッチ処理）

Python 3.10+ 環境（または `uv`）を使用して、指定したWikipedia記事から新しい問題データJSONと画像を生成できます。

```bash
# 初期プリセット問題の一括生成
uv run --python 3.11 --with-requirements requirements.txt python3 scripts/generate_problem.py --preset

# 日付とWikipedia記事名を指定して1問追加生成
uv run --python 3.11 --with-requirements requirements.txt python3 scripts/generate_problem.py \
  --date 2026-10-24 \
  --title "厳島神社" \
  --aliases "いつくしまじんじゃ,宮島"
```

生成されたファイルは `data/YYYY-MM-DD.json` と `data/images/YYYY-MM-DD.jpg` に出力され、`data/manifest.json` が自動更新されます。

## 📜 ライセンス・クレジット

- 出題画像および記事テキストは [Wikipedia](https://ja.wikipedia.org/) および [Wikimedia Commons](https://commons.wikimedia.org/) の各著作者に帰属し、各画像に付随する Creative Commons ライセンス（CC BY-SA 3.0 / 4.0、CC BY、Public Domain 等）のもとで提供されています。
- ユーザーのプレイ履歴はブラウザ内の `LocalStorage` のみに保存され、外部サーバーへの送信やトラッキングは一切行いません。
