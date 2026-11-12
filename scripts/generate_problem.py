#!/usr/bin/env python3
"""
WikiLens バッチ問題生成スクリプト
- MediaWiki API から記事情報・代表画像（画像版）または冒頭文（文字版）を取得
- 画像がある記事は「画像版」、ない記事（孤独・記憶・意識・愛等）は「文字版」として生成
- intfloat/multilingual-e5-small を用いて日本語頻出語彙（約100,000語）とのセマンティック類似度を計算
- 静的配信用の JSON データ (/data/YYYY-MM-DD.json) と一覧 (/data/manifest.json) を出力
"""

from __future__ import annotations

import argparse
import html
import json
import re
import time
import unicodedata
from pathlib import Path
from typing import Any
from urllib.parse import quote, urlparse

import numpy as np
import requests
from sentence_transformers import SentenceTransformer
from wordfreq import get_frequency_dict, top_n_list

_JA_FREQ_DICT: dict[str, float] | None = None


def get_ja_freq(word: str) -> float:
    global _JA_FREQ_DICT
    if _JA_FREQ_DICT is None:
        _JA_FREQ_DICT = get_frequency_dict("ja")
    return _JA_FREQ_DICT.get(word, 0.0)


# Wikimedia Foundation User-Agent Policy 準拠ヘッダー
USER_AGENT = "WikiLensBot/1.0 (https://github.com/takedahika/wikipedia_image_guess; educational browser game)"
WIKI_API_URL = "https://ja.wikipedia.org/w/api.php"

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
IMAGES_DIR = DATA_DIR / "images"
QUEUE_DIR = BASE_DIR / "queue"
CACHE_DIR = Path(__file__).resolve().parent / ".cache"

# 除外する一般的な機能語・ストップワード・ノイズ語
STOPWORDS = {
    "する", "いる", "ある", "なる", "できる", "れる", "られる", "せる", "させる",
    "ない", "たい", "らしい", "ようだ", "そうだ", "こと", "もの", "ため", "とき",
    "ところ", "よう", "うち", "ほか", "これ", "それ", "あれ", "どれ", "ここ",
    "そこ", "あそこ", "どこ", "わたし", "あなた", "かれ", "かのじょ", "われわれ",
    "それぞれ", "ならびに", "および", "または", "しかし", "また", "さらに", "そして",
    "ただし", "なお", "つまり", "すなわち", "たとえば", "とくに", "すでに", "まだ",
    "もっとも", "かなり", "非常に", "とても", "あまり", "ほとんど", "すべて",
    "一覧", "曖昧さ回避", "ウィキペディア", "出典", "脚注", "参考文献", "外部リンク",
    "関連項目", "概要", "特徴", "種類", "分類", "名称", "定義", "意味",
    "場合", "結果", "状態", "方法", "利用", "使用", "現在", "当時", "以降", "以前",
    "自分", "自体", "自身", "ほか", "ため", "ゆえ", "もと", "うえ", "した",
}

# メンテナンス系・メタ系カテゴリの除外パターン
HIDDEN_CAT_PATTERNS = [
    r"スタブ", r"ウィキデータ", r"良質な記事", r"秀逸な記事", r"出典", r"検証",
    r"独自研究", r"リンク切れ", r"曖昧さ回避", r"一覧", r"記事", r"ページ",
    r"テンプレート", r"カテゴリ", r"画像", r"座標", r"識別子", r"典拠管理",
    r"ISBN", r"ISSN", r"LCCN", r"NDL", r"VIAF", r"Commons", r"コモンズ",
    r"和製漢語", r"名数", r"語句", r"用語", r"言葉",
]


def normalize_word(word: str) -> str:
    """全角半角正規化（NFKC）とトリム"""
    return unicodedata.normalize("NFKC", word.strip())


def is_valid_japanese_vocab(word: str, strict_freq: bool = False) -> bool:
    """ゲーム用の語彙として適切かどうかを判定する"""
    if not word or len(word) > 12:
        return False
    if len(word) == 1:
        # 1文字語は「心・脳・法・悪・秒・愛・夢・光・音・死・生・神・美」などの漢字1字の名詞を収録
        return bool(re.fullmatch(r"[一-龥]", word)) and word not in STOPWORDS and get_ja_freq(word) >= 1e-6
    if word in STOPWORDS:
        return False
    if re.search(r"[0-9０-９!！?？\(\)（）\[\]「」『』【】/:：;；,、.。・\s　\"'~〜\-_＝=＋+＆&]", word):
        return False
    if re.fullmatch(r"[a-zA-Z]+", word):
        return False
    if re.fullmatch(r"[ぁ-ん]{2}", word):
        if get_ja_freq(word) < 3e-6:
            return False
    if re.fullmatch(r"[ァ-ヴー]{2}", word):
        if get_ja_freq(word) < 3e-6:
            return False
    if re.search(
        r"(ました|ません|ている|ていた|ており|された|される|として|について|による|において|における|から|まで|だけ|など|くらい|ばかり|ながら|けれど|なかった|なければ|といった|という)$",
        word,
    ):
        return False
    if re.fullmatch(r"[ぁ-ん]{3,}", word) and word.endswith(
        ("う", "く", "ぐ", "す", "つ", "ぬ", "ぶ", "む", "る", "い", "た", "だ", "て", "で", "に", "と", "ば", "よ", "ね", "わ", "ん", "し", "り")
    ):
        return False
    if strict_freq and len(word) >= 8 and get_ja_freq(word) == 0.0:
        return False
    return True


def build_base_vocabulary(target_size: int = 100000) -> list[str]:
    """wordfreq の日本語頻出語からクリーンな語彙リスト（デフォルト100,000語）を構築してキャッシュする"""
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    vocab_cache_file = CACHE_DIR / f"base_vocab_v4_{target_size}.json"
    if vocab_cache_file.exists():
        return json.loads(vocab_cache_file.read_text(encoding="utf-8"))

    print(f"ベース語彙プール ({target_size} 語) を構築中...")
    raw_words = top_n_list("ja", 300000)
    clean_words: list[str] = []
    seen: set[str] = set()

    # 第1パス: 厳密な日本語語彙フィルタ
    for w in raw_words:
        nw = normalize_word(w)
        if nw in seen:
            continue
        if is_valid_japanese_vocab(nw):
            seen.add(nw)
            clean_words.append(nw)
            if len(clean_words) >= target_size:
                break

    # 第2パス: target_size に満たない場合、日本語文字（漢字・かな）を含む一般語彙で100,000語まで補完
    if len(clean_words) < target_size:
        for w in raw_words:
            nw = normalize_word(w)
            if nw in seen or nw in STOPWORDS or not nw or len(nw) > 15:
                continue
            if re.search(r"[一-龥ぁ-んァ-ヴー]", nw) and not re.search(r"[!！?？\(\)（）\[\]「」『』【】/:：;；,、.。\s　\"']", nw):
                seen.add(nw)
                clean_words.append(nw)
                if len(clean_words) >= target_size:
                    break

    vocab_cache_file.write_text(json.dumps(clean_words, ensure_ascii=False), encoding="utf-8")
    print(f"ベース語彙プール構築完了: {len(clean_words)} 語")
    return clean_words


def strip_html(text: str) -> str:
    """HTMLタグを除去してプレーンテキストにする"""
    if not text:
        return ""
    cleaned = re.sub(r"<[^>]+>", "", text)
    cleaned = html.unescape(cleaned)
    cleaned = re.sub(r"\s+", " ", cleaned).strip()
    return cleaned


def mask_sentence_text(sentence: str, target: str, aliases: list[str], hard_nouns: list[str] | None = None) -> str:
    """文中のターゲット名・別名・読み仮名を伏せ字（██）に変換する"""
    masked = sentence
    # 正解・別名の伏せ字化
    terms = sorted(set([target] + aliases), key=len, reverse=True)
    for t in terms:
        if len(t) >= 2 or t == target:  # 「運」のような1文字の答えも伏せる
            masked = re.sub(re.escape(t), "█" * len(t), masked, flags=re.IGNORECASE)
    # 冒頭の読み仮名カッコなどを伏せ字化
    masked = re.sub(r"（[^）]{1,45}）", "（…）", masked, count=1)
    # 高難度用：キー名詞を伏せる
    if hard_nouns:
        for hn in hard_nouns:
            if len(hn) >= 2 and hn in masked:
                masked = masked.replace(hn, "░" * len(hn))
    return masked


def fetch_wikipedia_article(title: str, force_mode: str = "auto") -> dict[str, Any]:
    """MediaWiki API から記事テキスト、画像（あれば）、カテゴリ、セクション、リンクを取得する"""
    session = requests.Session()
    session.headers.update({"User-Agent": USER_AGENT})

    params = {
        "action": "query",
        "format": "json",
        "titles": title,
        "redirects": 1,
        "prop": "extracts|pageimages|categories|redirects|links|info",
        "exintro": 1,
        "explaintext": 1,
        "piprop": "name|thumbnail|original",
        "pithumbsize": 800,
        "clshow": "!hidden",
        "cllimit": 50,
        "rdlimit": 50,
        "plnamespace": 0,
        "pllimit": 500,
    }
    resp = session.get(WIKI_API_URL, params=params, timeout=20)
    resp.raise_for_status()
    data = resp.json()

    pages = data.get("query", {}).get("pages", {})
    page = next(iter(pages.values()), None)
    if not page or "missing" in page:
        raise ValueError(f"Wikipedia記事が見つかりません: {title}")

    canonical_title = normalize_word(page.get("title", title))
    intro_extract = page.get("extract", "").strip()
    page_length = page.get("length", len(intro_extract))

    # 本文全体の先頭4000文字も取得
    body_params = {
        "action": "query",
        "format": "json",
        "titles": canonical_title,
        "prop": "extracts",
        "explaintext": 1,
        "exchars": 4000,
    }
    body_resp = session.get(WIKI_API_URL, params=body_params, timeout=20)
    body_extract = intro_extract
    if body_resp.ok:
        bpages = body_resp.json().get("query", {}).get("pages", {})
        bpage = next(iter(bpages.values()), None)
        if bpage and bpage.get("extract"):
            body_extract = bpage["extract"]

    # カテゴリ抽出
    raw_cats = [c["title"].replace("Category:", "").replace("カテゴリ:", "") for c in page.get("categories", [])]
    clean_cats = []
    mask_targets = {canonical_title, normalize_word(title)}
    for c in raw_cats:
        if any(re.search(pat, c) for pat in HIDDEN_CAT_PATTERNS):
            continue
        masked_c = c
        for mt in mask_targets:
            if mt and mt in masked_c:
                masked_c = masked_c.replace(mt, "〇〇")
        if masked_c != "〇〇" and "行" not in masked_c:
            clean_cats.append(masked_c)

    # リダイレクト（別名）抽出
    redirects = []
    if normalize_word(title) != canonical_title:
        redirects.append(normalize_word(title))
        redirects.append(canonical_title)
    for r in page.get("redirects", []):
        rt = normalize_word(r["title"])
        if rt != canonical_title and len(rt) <= 20 and "(" not in rt:
            redirects.append(rt)

    # 内部リンク語抽出
    linked_words = []
    for lk in page.get("links", []):
        lt = normalize_word(lk["title"])
        lt_clean = re.sub(r"\s*\([^)]+\)$", "", lt)
        if is_valid_japanese_vocab(lt_clean, strict_freq=True) and lt_clean not in mask_targets:
            linked_words.append(lt_clean)

    # セクション（目次）取得
    sec_params = {
        "action": "parse",
        "format": "json",
        "page": canonical_title,
        "prop": "sections",
    }
    sec_resp = session.get(WIKI_API_URL, params=sec_params, timeout=20)
    sections = []
    if sec_resp.ok:
        sec_data = sec_resp.json().get("parse", {}).get("sections", [])
        exclude_secs = {"脚注", "出典", "参考文献", "関連項目", "外部リンク", "注釈", "案内"}
        for s in sec_data:
            line = strip_html(s.get("line", ""))
            if line and line not in exclude_secs:
                for mt in mask_targets:
                    if mt and mt in line:
                        line = line.replace(mt, "〇〇")
                sections.append(line)

    display_target = normalize_word(title)
    aliases_list = list(dict.fromkeys(redirects))[:20]

    # 文の分割とテキストパズルの構築
    raw_sentences = [
        s.strip() + "。"
        for s in intro_extract.replace("\r", "").split("。")
        if s.strip()
    ]
    if not raw_sentences:
        raw_sentences = [f"{display_target}に関する記事。"]

    # 文字版パズルの各文伏せ字
    text_puzzle = {
        "sentence_1": mask_sentence_text(raw_sentences[0], display_target, aliases_list),
        "sentence_2": mask_sentence_text(raw_sentences[1], display_target, aliases_list) if len(raw_sentences) > 1 else "",
        "sentence_3": mask_sentence_text(raw_sentences[2], display_target, aliases_list) if len(raw_sentences) > 2 else "",
        "sentence_4": mask_sentence_text(raw_sentences[3], display_target, aliases_list) if len(raw_sentences) > 3 else "",
        "raw_sentences": raw_sentences[:4],
    }

    # 画像の取得判定
    pageimage_name = page.get("pageimage")
    mode = "text"
    image_meta = None

    if force_mode != "text" and pageimage_name:
        # SVGやロゴ・アイコン系は除外
        low_name = pageimage_name.lower()
        if not low_name.endswith(".svg") and not any(k in low_name for k in ["logo", "icon", "flag", "symbol"]):
            try:
                img_params = {
                    "action": "query",
                    "format": "json",
                    "titles": f"File:{pageimage_name}",
                    "prop": "imageinfo",
                    "iiprop": "url|extmetadata|user",
                    "iiurlwidth": 800,
                }
                img_resp = session.get(WIKI_API_URL, params=img_params, timeout=20)
                if img_resp.ok:
                    img_pages = img_resp.json().get("query", {}).get("pages", {})
                    img_page = next(iter(img_pages.values()), {})
                    imageinfo_list = img_page.get("imageinfo", [])
                    if imageinfo_list:
                        ii = imageinfo_list[0]
                        thumb_url = ii.get("thumburl") or ii.get("url")
                        commons_url = ii.get("descriptionurl", f"https://commons.wikimedia.org/wiki/File:{quote(pageimage_name)}")
                        ext = ii.get("extmetadata", {})

                        author = (
                            strip_html(ext.get("Artist", {}).get("value", ""))
                            or strip_html(ext.get("Credit", {}).get("value", ""))
                            or "Wikimedia Commons Contributor"
                        )
                        if len(author) > 80:
                            author = author[:77] + "..."
                        if re.search(r"(投稿者自身|自身による著作物|Own work|own work)", author) or author == "Wikimedia Commons Contributor":
                            author = ii.get("user") or author

                        license_short = strip_html(ext.get("LicenseShortName", {}).get("value", "CC BY-SA"))
                        license_url = ext.get("LicenseUrl", {}).get("value", "https://creativecommons.org/licenses/by-sa/4.0/")
                        first_sentence = raw_sentences[0] if raw_sentences else canonical_title
                        caption = first_sentence[:140]

                        image_meta = {
                            "filename": pageimage_name,
                            "thumb_url": thumb_url,
                            "commons_url": commons_url,
                            "author": author,
                            "license": license_short,
                            "license_url": license_url,
                            "caption": caption,
                        }
                        mode = "image"
            except Exception as e:
                print(f"画像取得スキップ ({pageimage_name}): {e}")
                mode = "text"

    if force_mode == "image" and mode != "image":
        print(f"⚠️ 指定された記事「{title}」に適切な画像が見つからないため、文字版として処理します。")
        mode = "text"

    return {
        "mode": mode,
        "title": display_target,
        "canonical_title": canonical_title,
        "extract": intro_extract,
        "body_extract": body_extract,
        "page_length": page_length,
        "categories": clean_cats[:8],
        "sections": sections[:8],
        "redirects": aliases_list,
        "linked_words": list(dict.fromkeys(linked_words)),
        "article_url": f"https://ja.wikipedia.org/wiki/{quote(canonical_title)}",
        "text_puzzle": text_puzzle,
        "image_meta": image_meta,
    }


def download_and_anonymize_image(thumb_url: str, problem_id: str, images_dir: Path) -> str:
    """
    サムネイル画像をダウンロードし、ファイル名から正解がネタバレしないよう
    images/<problem_id>.jpg として保存する
    """
    images_dir.mkdir(parents=True, exist_ok=True)
    local_filename = f"{problem_id}.jpg"
    local_path = images_dir / local_filename

    session = requests.Session()
    session.headers.update({"User-Agent": USER_AGENT})
    resp = session.get(thumb_url, timeout=30)
    resp.raise_for_status()
    local_path.write_bytes(resp.content)
    return f"./data/images/{local_filename}"


def filter_aliases(target: str, aliases: list[str]) -> list[str]:
    """
    正解扱いにする別名を絞り込む。
    答えを含む別の言葉（運→不運・幸運、記憶→長期記憶、正義→正義感）は意味が違うので正解にしない。
    """
    kept = []
    for a in aliases:
        if not a or a == target:
            continue
        if target in a and len(a) > len(target):
            continue
        kept.append(a)
    return kept


def create_masked_summary(extract: str, target: str, aliases: list[str]) -> str:
    """導入部の最初の2文から正解ワードと別名を伏せ字（███）にしたヒント文を生成する"""
    sentences = [s.strip() for s in extract.replace("\n", "").split("。") if s.strip()]
    snippet = "。".join(sentences[:2]) + "。" if sentences else ""
    masked = snippet
    terms = sorted(set([target] + aliases), key=len, reverse=True)
    for t in terms:
        if len(t) >= 2 or t == target:  # 「運」のような1文字の答えも伏せる
            masked = re.sub(re.escape(t), "█" * len(t), masked, flags=re.IGNORECASE)
    masked = re.sub(r"（[^）]{1,45}）", "（…）", masked, count=1)
    if len(masked) > 180:
        masked = masked[:180] + "…"
    return masked


def compute_rankings(
    model: SentenceTransformer,
    base_vocab: list[str],
    base_embeddings: np.ndarray,
    article_info: dict[str, Any],
    max_words: int = 100000,
) -> list[str]:
    """
    セマンティック距離推測のランキング計算:
    1. ターゲット単語埋め込みとのコサイン類似度 (0.40)
    2. ターゲット概念（カテゴリ群）埋め込みとの類似度 (0.35)
    3. 記事導入部テキスト埋め込みとの類似度 (0.25)
    4. 内部リンク語・本文登場語の自然な関連度ボーナス
    （※写っている物への過剰な人工ブーストは廃止し、本来の意味的近さで競う）
    """
    target = article_info["title"]
    canonical = article_info["canonical_title"]
    extract = article_info["extract"]
    body_extract = article_info["body_extract"]
    categories = article_info["categories"]
    linked_set = set(article_info["linked_words"])

    base_set = set(base_vocab)
    must_include = ["心", "脳", "法", "悪", "秒", "一人", "愛", "夢", "光", "音", "死", "生", "神", "美", "くじ", "天秤", "クローバー"]
    extra_words = [
        w for w in (must_include + article_info["linked_words"])
        if w not in base_set and w != target and w != canonical
    ][:600]

    all_words = list(base_vocab)
    if extra_words:
        extra_prompts = [f"query: {w}" for w in extra_words]
        extra_embs = model.encode(extra_prompts, normalize_embeddings=True, show_progress_bar=False, batch_size=128)
        all_words.extend(extra_words)
        all_embs = np.vstack([base_embeddings, extra_embs])
    else:
        all_embs = base_embeddings

    # ベクトル生成
    title_emb = model.encode(f"query: {target}", normalize_embeddings=True, show_progress_bar=False)
    concept_text = f"query: {target} {' '.join(categories[:5])}"
    concept_emb = model.encode(concept_text, normalize_embeddings=True, show_progress_bar=False)
    summary_text = f"passage: {target}。{extract[:600]}"
    summary_emb = model.encode(summary_text, normalize_embeddings=True, show_progress_bar=False)

    sim_title = all_embs @ title_emb
    sim_concept = all_embs @ concept_emb
    sim_summary = all_embs @ summary_emb

    scores = 0.40 * sim_title + 0.35 * sim_concept + 0.25 * sim_summary

    for idx, w in enumerate(all_words):
        if w == target or w == canonical:
            continue

        in_link = w in linked_set
        in_intro = len(w) >= 2 and (w in extract)
        in_body = len(w) >= 2 and (w in body_extract)

        if in_link and in_intro:
            scores[idx] += 0.055
        elif in_link and in_body:
            scores[idx] += 0.040
        elif in_intro:
            scores[idx] += 0.035
        elif in_link:
            scores[idx] += 0.022

        # ターゲット文字列を含む複合語（例: 記憶→長期記憶・記憶障害）が上位を独占しすぎないよう調整
        if len(w) > len(target) and target in w:
            scores[idx] -= 0.035

        # ターゲット文字列の一部を含むだけの表層断片ノイズを抑制
        if len(w) <= 3 and w in target and not in_link and not in_intro:
            scores[idx] -= 0.06

    sorted_indices = np.argsort(-scores)
    ranked_words = [target]
    seen = {target}

    for idx in sorted_indices:
        w = all_words[int(idx)]
        if w not in seen:
            seen.add(w)
            ranked_words.append(w)
            if len(ranked_words) >= max_words:
                break

    return ranked_words


def extract_related_hint_words(
    ranked_words: list[str],
    target: str,
    aliases: list[str],
    linked_words: list[str] | None = None,
    count: int = 4,
) -> list[str]:
    """
    ヒント表示用に、答えの文字（漢字・かな）を1文字も含まない上位の連想ワード（名詞）を抽出する。
    記事内部リンク語（linked_words）に含まれる関連概念を優先し、動詞・副詞・メタ語を除外する。
    """
    alias_set = {a.lower() for a in aliases}
    target_chars = set(target)
    linked_set = set(linked_words or [])
    weak_meta = {
        "概念", "テーマ", "分野", "意味", "研究", "理論", "学問", "対象", "性質", "現象",
        "関係", "構造", "体系", "一般", "基礎", "基本", "ものの", "人々", "感じ", "状況",
        "公的", "公約", "公社", "公言", "理学", "科学", "考察", "機能", "物体", "動物",
    }

    def is_good_hint_noun(w: str) -> bool:
        if w.lower() in alias_set or w in STOPWORDS or w in weak_meta:
            return False
        if target in w or w in target:
            return False
        if any(ch in w for ch in target_chars):
            return False
        if len(w) < 1 or len(w) > 6:
            return False
        # ひらがなのみで終わる動詞・形容詞・副詞っぽい語を除外
        if re.search(r"[ぁ-ん]$", w) and not re.search(r"(さ|み|き|り|め|ち|て)$", w):
            return False
        if w.endswith(("く", "ぐ", "す", "つ", "ぬ", "ぶ", "む", "る", "う", "い", "な", "に", "と", "で")):
            return False
        return True

    result: list[str] = []
    # 第1優先: 上位500語のうち、記事内リンクにも存在する確かな関連名詞
    for w in ranked_words[1:500]:
        if w in linked_set and is_good_hint_noun(w):
            result.append(w)
            if len(result) >= count:
                return result

    # 第2優先: 上位300語から条件を満たす名詞を補完
    for w in ranked_words[1:300]:
        if w not in result and is_good_hint_noun(w):
            result.append(w)
            if len(result) >= count:
                break

    return result


def get_base_embeddings(model: SentenceTransformer, base_vocab: list[str]) -> np.ndarray:
    """ベース語彙の埋め込みベクトルを計算またはキャッシュから読み込む"""
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    emb_cache_file = CACHE_DIR / f"base_embs_v3_e5_small_{len(base_vocab)}.npy"
    if emb_cache_file.exists():
        print("キャッシュ済み語彙ベクトルを読み込み中...")
        return np.load(emb_cache_file)

    print(f"ベース語彙 ({len(base_vocab)} 語) の埋め込みベクトルを計算中...")
    prompts = [f"query: {w}" for w in base_vocab]
    embs = model.encode(
        prompts,
        normalize_embeddings=True,
        show_progress_bar=True,
        batch_size=256,
    )
    embs = np.asarray(embs, dtype=np.float32)
    np.save(emb_cache_file, embs)
    return embs


def generate_problem(
    problem_id: str,
    target_title: str,
    model: SentenceTransformer,
    base_vocab: list[str],
    base_embeddings: np.ndarray,
    extra_aliases: list[str] | None = None,
    force_mode: str = "auto",
    custom_caption: str | None = None,
    out_dir: Path | None = None,
    max_words: int = 100000,
) -> dict[str, Any]:
    """1問分の問題データJSON（および画像版なら画像）を生成する"""
    dest_dir = out_dir or DATA_DIR
    images_dir = dest_dir / "images" if dest_dir.name == "data" else dest_dir.parent / "data" / "images"
    dest_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n[{problem_id}] 記事「{target_title}」を取得中 (mode: {force_mode})...")
    info = fetch_wikipedia_article(target_title, force_mode=force_mode)

    mode = info["mode"]
    image_obj = None

    if mode == "image" and info["image_meta"]:
        local_img_url = download_and_anonymize_image(info["image_meta"]["thumb_url"], problem_id, images_dir)
        caption = custom_caption or info["image_meta"]["caption"]
        image_obj = {
            "url": local_img_url,
            "commons_url": info["image_meta"]["commons_url"],
            "caption": caption,
            "author": info["image_meta"]["author"],
            "license": info["image_meta"]["license"],
            "license_url": info["image_meta"]["license_url"],
        }

    aliases = filter_aliases(info["title"], list(dict.fromkeys((extra_aliases or []) + info["redirects"])))
    ranked_words = compute_rankings(
        model,
        base_vocab,
        base_embeddings,
        info,
        max_words=max_words,
    )

    masked_summary = create_masked_summary(info["extract"], info["title"], aliases)
    clean_cats = [
        c for c in info["categories"]
        if c != "〇〇" and "行" not in c and not any(re.search(pat, c) for pat in HIDDEN_CAT_PATTERNS)
    ][:5]
    if not clean_cats:
        clean_cats = ["哲学・科学の概念"]

    related_words = extract_related_hint_words(ranked_words, info["title"], aliases, info.get("linked_words"), count=4)

    problem_data = {
        "id": problem_id,
        "mode": mode,
        "target": info["title"],
        "aliases": aliases,
        "total_words": len(ranked_words),
        "article_url": info["article_url"],
        "image": image_obj,
        "text_puzzle": info["text_puzzle"],
        "hints": {
            "category": clean_cats,
            "sections": info["sections"][:6],
            "summary_length": info["page_length"],
            "first_char": info["title"][0] if info["title"] else "",
            "char_length": len(info["title"]),
            "masked_summary": masked_summary,
            "related_words": related_words,
        },
        "words": ranked_words,
    }

    out_path = dest_dir / f"{problem_id}.json"
    out_path.write_text(json.dumps(problem_data, ensure_ascii=False, separators=(",", ":")), encoding="utf-8")
    size_kb = out_path.stat().st_size / 1024
    print(f"[{problem_id}] 生成完了 [{mode}版] -> {out_path.name} ({size_kb:.1f} KB), 上位10語: {ranked_words[:10]}, ヒント関連語: {related_words}")
    return problem_data


# 自動補充用の厳選・抽象概念＆象徴モチーフ記事プール
REFILL_CANDIDATES: list[tuple[str, list[str], str]] = [
    ("自由", ["じゆう", "フリーダム", "リバティ"], "auto"),
    ("幸福", ["こうふく", "しあわせ", "ハピネス"], "auto"),
    ("愛", ["あい", "ラブ"], "auto"),
    ("夢", ["ゆめ", "ドリーム"], "auto"),
    ("美", ["び", "うつくしさ", "ビューティー"], "auto"),
    ("真理", ["しんり", "トゥルース"], "auto"),
    ("知識", ["ちしき", "ナレッジ"], "auto"),
    ("生命", ["せいめい", "いのち", "ライフ"], "auto"),
    ("文明", ["ぶんめい", "シヴィライゼーション"], "auto"),
    ("平和", ["へいわ", "ピース"], "auto"),
    ("希望", ["きぼう", "ホープ"], "auto"),
    ("恐怖", ["きょうふ", "おそれ"], "auto"),
    ("信頼", ["しんらい", "トラスト"], "auto"),
    ("言語", ["げんご", "ことば"], "auto"),
    ("音楽", ["おんがく", "ミュージック"], "auto"),
    ("芸術", ["げいじゅつ", "アート"], "auto"),
    ("科学", ["かがく", "サイエンス"], "auto"),
    ("歴史", ["れきし", "ヒストリー"], "auto"),
    ("未来", ["みらい", "フューチャー"], "auto"),
    ("空間", ["くうかん", "スペース"], "auto"),
    ("光", ["ひかり", "ライト"], "auto"),
    ("影", ["かげ", "シャドウ"], "auto"),
    ("熱", ["ねつ", "ヒート"], "auto"),
    ("音", ["おと", "サウンド"], "auto"),
    ("色", ["いろ", "色彩", "カラー"], "auto"),
    ("無限", ["むげん", "インフィニティ"], "auto"),
    ("秩序", ["ちつじょ", "オーダー"], "auto"),
    ("混沌", ["こんとん", "カオス"], "auto"),
    ("運命", ["うんめい", "宿命"], "auto"),
    ("偶然", ["ぐうぜん", "チャンス"], "auto"),
    ("直観", ["ちょっかん", "インスピレーション"], "auto"),
    ("想像力", ["そうぞうりょく", "イマジネーション"], "auto"),
    ("共感", ["きょうかん", "エンパシー"], "auto"),
    ("勇気", ["ゆうき", "カレッジ"], "auto"),
    ("欲望", ["よくぼう", "デザイア"], "auto"),
    ("道徳", ["どうとく", "モラル"], "auto"),
    ("権力", ["けんりょく", "パワー"], "auto"),
    ("貨幣", ["かへい", "おかね", "マネー"], "auto"),
    ("労働", ["ろうどう", "ワーク"], "auto"),
    ("遊び", ["あそび", "遊戯", "プレイ"], "auto"),
    ("沈黙", ["ちんもく", "サイレンス"], "auto"),
    ("秘密", ["ひみつ", "シークレット"], "auto"),
]


CURATED_RELATED_HINTS: dict[str, list[str]] = {
    "運": ["占い", "努力", "偶然", "神社"],
    "情報": ["知識", "通信", "データ", "メディア"],
    "感情": ["気持ち", "喜び", "心理", "悲しみ"],
    "意識": ["認知", "心理学", "覚醒", "脳"],
    "正義": ["道徳", "倫理", "法律", "権利"],
    "思考": ["哲学", "推理", "判断", "知性"],
    "孤独": ["寂しさ", "心情", "人間関係", "社会"],
    "重力": ["宇宙", "地球", "質量", "引力"],
    "記憶": ["学習", "思い出", "忘却", "脳"],
    "時間": ["時計", "空間", "世紀", "秒"],
}

# 画像版で「画像に写っている具体的なモチーフ（中間ステップ：距離 120〜320 ✨）」として扱う単語
CURATED_VISUAL_MOTIFS: dict[str, list[str]] = {
    "運": ["クローバー", "四つ葉", "葉", "植物", "草", "緑", "花"],
    "感情": ["顔", "表情", "版画", "絵画", "肖像", "人間", "笑う", "泣く", "目", "眉"],
    "正義": ["女神", "天秤", "剣", "銅像", "彫刻", "神話", "目隠し", "秤", "像"],
    "思考": ["壁画", "絵画", "女性", "ペン", "書く", "文字", "ローマ", "古代", "本", "紙"],
    "重力": ["銀河", "星", "天体", "渦", "光", "望遠鏡", "夜空", "リンゴ", "ニュートン"],
    "時間": ["懐中時計", "時計", "針", "文字盤", "歯車", "砂時計", "カレンダー"],
}

# 概念的な近傍語（距離 1〜90 🔥 に確実に配置する橋渡し語）
CURATED_CONCEPT_BRIDGES: dict[str, list[str]] = {
    "運": ["占い", "運命", "偶然", "努力", "奇跡", "縁起", "お守り", "ジンクス", "くじ", "勝負", "成功", "チャンス", "幸せ", "幸福", "神", "神社"],
    "情報": ["知識", "通信", "データ", "メディア", "記号", "信号", "伝達", "ニュース", "コンピュータ", "インターネット", "放送", "報道", "秘密", "知恵"],
    "感情": ["気持ち", "喜び", "悲しみ", "怒り", "恐怖", "驚き", "心理", "心", "気分", "情動", "情緒", "愛情", "涙", "笑顔", "共感", "不安"],
    "意識": ["認知", "覚醒", "睡眠", "脳", "心", "心理学", "無意識", "自我", "注意", "感覚", "知覚", "麻酔", "夢", "哲学", "主観"],
    "正義": ["道徳", "倫理", "法律", "法", "権利", "裁判", "裁判所", "公正", "公平", "平等", "善", "悪", "罪", "罰", "秩序", "真実", "良心"],
    "思考": ["考え", "哲学", "推理", "判断", "論理", "知性", "理性", "発想", "想像", "知恵", "知識", "脳", "頭", "精神", "アイデア", "概念", "仮説"],
    "孤独": ["一人", "寂しさ", "孤立", "心情", "人間関係", "社会", "群衆", "自由", "自立", "静寂", "沈黙", "疎外", "友人", "家族", "心"],
    "重力": ["引力", "宇宙", "地球", "質量", "重さ", "落下", "軌道", "惑星", "太陽", "月", "ブラックホール", "相対性理論", "物理", "物理学", "力", "加速度", "無重力"],
    "記憶": ["学習", "思い出", "脳", "忘却", "認知", "過去", "経験", "記録", "歴史", "懐かしさ", "暗記", "睡眠", "心", "知識", "印象"],
    "時間": ["時刻", "空間", "秒", "分", "過去", "現在", "未来", "永遠", "瞬間", "歴史", "時代", "時空", "変化", "流れ", "寿命", "歳月", "世紀", "時計"],
}

# 文字版で1手目から辞書定義（〇〇とは〜）で即バレしないための3段階推理文
CURATED_TEXT_PUZZLES: dict[str, list[str]] = {
    "情報": [
        "日本語の言葉としては明治時代に軍事用語（敵情の報告）の翻訳から定着したが、現代では自然科学から社会科学まであらゆる領域の基礎概念となっている。",
        "クロード・シャノンはこれを「不確実性を減らすもの」として数理的に定義し、生物の遺伝や神経伝達においても不可欠な指令・信号として機能している。",
        "文字・数値・記号などの媒体を通じて伝達され、受け手に新たな知識や判断の手がかりをもたらす「知らせ」や「内容」の総称。",
    ],
    "意識": [
        "物質である神経細胞の活動から、なぜ「赤色の感じ」や「痛み」といった主観的な体験が生まれるのかという問いは、科学における最大の難問（ハード・プロブレム）と呼ばれる。",
        "睡眠や麻酔によって一時的に失われる「目覚めている状態」を指す場合と、自分が今何を感じ・考えているかを自ら把握している働きを指す場合がある。",
        "外界の刺激や自分自身の内面状態を、主観的に体験し気づいている心の働き・状態を指す言葉。",
    ],
    "孤独": [
        "どれほど大勢の群衆の中に囲まれていたとしても、周囲と内面が通じ合っていないと感じるとき、人はこの状態に置かれる。",
        "近代以降の都市化や個人主義の広がりとともに社会問題として議論される一方、宗教や哲学・芸術においては自己と向き合う創造的な境地としても捉えられてきた。",
        "精神的なよりどころとなる人や心の通う相手を欠き、ひとり取り残されたような感覚を伴う状態を指す言葉。",
    ],
    "記憶": [
        "古代ギリシアでは「心の蝋板（ろうばん）に刻印を残すこと」に例えられ、哲学や文学において個人の同一性を支える根幹として扱われてきた。",
        "反復によって行動が変化する「学習」と表裏一体の関係にあり、保持される時間の長さや内容の性質によって複数の段階に分類される。",
        "過去の経験を符号化して蓄え、必要なときに再び引き出すという、生物の脳や神経系が持つ基本的な働き。",
    ],
}

# ヒント3（ラストチャンス）用の、辞書丸写しではない本質ヒント文
CURATED_FINAL_HINTS: dict[str, str] = {
    "運": "【答えは漢字1文字】自分の意思や努力だけではコントロールできない、人生の巡り合わせや勝負の流れ。",
    "情報": "【答えは漢字2文字】記号や信号を通じて伝えられ、受け手の「不確実性」を減らして判断を導く知らせ。",
    "感情": "【答えは漢字2文字】外からの出来事や対象に触れたとき、心の中に自然と湧き起こる「喜・怒・哀・楽」などの動き。",
    "意識": "【答えは漢字2文字】眠りから覚めている状態であり、「いま自分が何を見聞きし、何を感じているか」に気づいている心の働き。",
    "正義": "【答えは漢字2文字】偏りのない天秤と剣で象徴される、社会や法・道徳において「何が正しいか」を定める理念。",
    "思考": "【答えは漢字2文字】問いに対して筋道を立て、概念や判断をつなぎ合わせて結論へとたどり着こうとする知的な営み。",
    "孤独": "【答えは漢字2文字】たとえ雑踏の中にいても、誰とも心が通じ合っていないと感じるときに生まれる静かな隔たりの感覚。",
    "重力": "【答えは漢字2文字】リンゴが木から落ちる現象から、惑星や銀河が互いを引き止め合う現象までを貫く、質量が生み出す普遍的な働き。",
    "記憶": "【答えは漢字2文字】過ぎ去った出来事や学んだ知識を脳内に刻み込み、時を超えて再び呼び起こす心の働き。",
    "時間": "【答えは漢字2文字】過去から現在、未来へと絶えず流れ続け、あらゆる変化や出来事の順序を成り立たせる見えない軸。",
}


def apply_curated_ranking_and_motifs(target: str, ranked_words: list[str], max_words: int = 100000) -> list[str]:
    """
    1. 概念的な近傍語（CURATED_CONCEPT_BRIDGES）が確実に上位（距離 1〜90 🔥）に入るように調整
    2. 画像に写っている具体的なモチーフ（CURATED_VISUAL_MOTIFS）が遠すぎず近すぎない
       「画像の手がかりゾーン（距離 140〜260 ✨）」に入るように配置
    """
    bridges = [w for w in CURATED_CONCEPT_BRIDGES.get(target, []) if w != target]
    motifs = [w for w in CURATED_VISUAL_MOTIFS.get(target, []) if w != target and w not in bridges]
    if not bridges and not motifs:
        return ranked_words[:max_words]

    pos_map = {w: i for i, w in enumerate(ranked_words)}

    # すでに上位80位以内にあるbridgeはそのまま、80位より下にあるbridgeは15〜75位付近に自然挿入
    late_bridges = [w for w in bridges if pos_map.get(w, 999999) > 80]
    # モチーフ語で300位より下（遠すぎる）にあるものは 150〜240位（✨かなり近い）に配置
    late_motifs = [w for w in motifs if pos_map.get(w, 999999) > 300 or pos_map.get(w, 999999) < 80]

    remove_set = set(late_bridges) | set(late_motifs)
    base_list = [w for w in ranked_words if w not in remove_set]

    # bridgeを上位（index 12 から 3語おき）に挿入
    insert_pos = 12
    for bw in late_bridges:
        idx = min(insert_pos, len(base_list))
        base_list.insert(idx, bw)
        insert_pos += 3

    # motifを中間ステップ（index 155 から 7語おき：距離155〜230 ✨）に挿入
    motif_pos = 155
    for mw in late_motifs:
        idx = min(motif_pos, len(base_list))
        base_list.insert(idx, mw)
        motif_pos += 7

    return base_list[:max_words]


def rerank_existing_problems(
    model: SentenceTransformer,
    base_vocab: list[str],
    base_embeddings: np.ndarray,
    max_words: int = 100000,
) -> list[dict[str, Any]]:
    """data/ 内の既存問題JSONを100,000語辞書＆新ヒント仕様で再計算し、最新日付（本日まで）に更新する"""
    import datetime
    import shutil

    jst = datetime.timezone(datetime.timedelta(hours=9))
    today_date = datetime.datetime.now(jst).date()
    old_files = sorted(DATA_DIR.glob("2026-*.json"))
    start_date = today_date - datetime.timedelta(days=len(old_files) - 1)
    images_dir = DATA_DIR / "images"

    updated = []
    loaded_items = []
    for idx, jf in enumerate(old_files):
        d = json.loads(jf.read_text(encoding="utf-8"))
        new_id = (start_date + datetime.timedelta(days=idx)).isoformat()
        loaded_items.append((jf, d, new_id))

    for jf, d, new_id in loaded_items:
        old_id = d["id"]
        target = d["target"]
        raw_sents = d.get("text_puzzle", {}).get("raw_sentences", [])
        try:
            wiki_info = fetch_wikipedia_article(target, force_mode=d.get("mode", "auto"))
            extract = wiki_info["extract"]
            body_extract = wiki_info["body_extract"]
            linked_words = wiki_info["linked_words"]
            raw_cats = wiki_info["categories"]
        except Exception:
            extract = "".join(raw_sents) or d.get("image", {}).get("caption", "") or target
            body_extract = extract
            linked_words = [w for w in d.get("words", [])[1:600] if w != target]
            raw_cats = d.get("hints", {}).get("category", [])

        aliases = filter_aliases(target, d.get("aliases", []))
        clean_cats = [
            c for c in raw_cats
            if c != "〇〇" and "行" not in c and not any(re.search(pat, c) for pat in HIDDEN_CAT_PATTERNS)
        ][:5]
        if not clean_cats:
            clean_cats = ["哲学・科学の概念"]

        curated_hints = CURATED_RELATED_HINTS.get(target, [])
        info = {
            "title": target,
            "canonical_title": target,
            "extract": extract,
            "body_extract": body_extract,
            "categories": clean_cats,
            "linked_words": list(dict.fromkeys(curated_hints + linked_words)),
        }
        ranked_words = compute_rankings(model, base_vocab, base_embeddings, info, max_words=max_words)
        ranked_words = apply_curated_ranking_and_motifs(target, ranked_words, max_words=max_words)
        related_words = curated_hints or extract_related_hint_words(ranked_words, target, aliases, linked_words, count=4)

        if target in CURATED_TEXT_PUZZLES:
            cs = CURATED_TEXT_PUZZLES[target]
            d["text_puzzle"] = {
                "sentence_1": cs[0],
                "sentence_2": cs[1],
                "sentence_3": cs[2],
                "sentence_4": "",
                "raw_sentences": raw_sents[:4] if raw_sents else cs,
            }
        elif raw_sents:
            # 1文目（辞書定義）を最後に回し、周辺文脈から出す
            ordered_sents = (raw_sents[1:3] + [raw_sents[0]]) if len(raw_sents) >= 3 else list(reversed(raw_sents))
            d["text_puzzle"] = {
                "sentence_1": mask_sentence_text(ordered_sents[0], target, aliases),
                "sentence_2": mask_sentence_text(ordered_sents[1], target, aliases) if len(ordered_sents) > 1 else "",
                "sentence_3": mask_sentence_text(ordered_sents[2], target, aliases) if len(ordered_sents) > 2 else "",
                "sentence_4": "",
                "raw_sentences": raw_sents[:4],
            }

        if d.get("image") and d["image"].get("url"):
            old_img = images_dir / f"{old_id}.jpg"
            new_img = images_dir / f"{new_id}.jpg"
            if old_img.exists() and old_img != new_img:
                shutil.move(str(old_img), str(new_img))
            d["image"]["url"] = f"./data/images/{new_id}.jpg"

        masked_summary = CURATED_FINAL_HINTS.get(target) or create_masked_summary(extract, target, aliases)
        d["id"] = new_id
        d["aliases"] = aliases
        d["visual_motifs"] = CURATED_VISUAL_MOTIFS.get(target, [])
        d["total_words"] = len(ranked_words)
        d["hints"]["category"] = clean_cats
        d["hints"]["char_length"] = len(target)
        d["hints"]["first_char"] = target[0] if target else ""
        d["hints"]["masked_summary"] = masked_summary
        d["hints"]["related_words"] = related_words
        d["words"] = ranked_words

        new_jf = DATA_DIR / f"{new_id}.json"
        new_jf.write_text(json.dumps(d, ensure_ascii=False, separators=(",", ":")), encoding="utf-8")
        if jf.exists() and jf != new_jf:
            jf.unlink()
        size_kb = new_jf.stat().st_size / 1024
        print(f"[再計算完了] {old_id} -> {new_id} ({target}): {len(ranked_words)} 語 ({size_kb:.1f} KB) | 関連ヒント語: {related_words}")
        updated.append(d)

    # manifest.json を新しい日付一覧でクリーンに書き直す
    manifest_path = DATA_DIR / "manifest.json"
    if manifest_path.exists():
        manifest_path.unlink()
    return updated


def update_manifest(problems_meta: list[dict[str, Any]], manifest_dir: Path | None = None) -> None:
    """問題一覧マニフェスト (/data/manifest.json) を更新する（ネタバレ情報は含めない）"""
    target_dir = manifest_dir or DATA_DIR
    manifest_path = target_dir / "manifest.json"
    existing: dict[str, dict[str, Any]] = {}
    if manifest_path.exists():
        try:
            old_data = json.loads(manifest_path.read_text(encoding="utf-8"))
            for item in old_data.get("problems", []):
                existing[item["id"]] = item
        except Exception:
            pass

    for p in problems_meta:
        existing[p["id"]] = {
            "id": p["id"],
            "mode": p.get("mode", "image"),
            "total_words": p["total_words"],
        }

    sorted_list = sorted(existing.values(), key=lambda x: x["id"], reverse=True)
    manifest = {
        "latest": sorted_list[0]["id"] if sorted_list else "",
        "updated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "problems": sorted_list,
    }
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"マニフェスト更新完了: {len(sorted_list)} 問収録")


def refill_problems(
    model: SentenceTransformer,
    base_vocab: list[str],
    base_embeddings: np.ndarray,
    days: int = 10,
    max_words: int = 100000,
) -> list[dict[str, Any]]:
    """
    data/ と queue/ にまだ存在しない候補概念を選び、
    最新日付の翌日から days 日分の問題を queue/ に自動補充する。
    """
    import datetime

    QUEUE_DIR.mkdir(parents=True, exist_ok=True)
    used_targets: set[str] = set()
    all_dates: list[str] = []

    for d_dir in [DATA_DIR, QUEUE_DIR]:
        if not d_dir.exists():
            continue
        for jf in d_dir.glob("2026-*.json"):
            all_dates.append(jf.stem)
            try:
                jd = json.loads(jf.read_text(encoding="utf-8"))
                if jd.get("target"):
                    used_targets.add(jd["target"])
            except Exception:
                pass

    jst = datetime.timezone(datetime.timedelta(hours=9))
    today_date = datetime.datetime.now(jst).date()

    if all_dates:
        latest_date = datetime.date.fromisoformat(max(all_dates))
        start_date = max(latest_date + datetime.timedelta(days=1), today_date)
    else:
        start_date = today_date

    available = [c for c in REFILL_CANDIDATES if c[0] not in used_targets]
    created: list[dict[str, Any]] = []

    for i in range(min(days, len(available))):
        target_date = (start_date + datetime.timedelta(days=i)).isoformat()
        title, aliases, mode_req = available[i]
        dest = DATA_DIR if target_date <= today_date.isoformat() else QUEUE_DIR
        try:
            p = generate_problem(
                target_date,
                title,
                model,
                base_vocab,
                base_embeddings,
                extra_aliases=aliases,
                force_mode=mode_req,
                out_dir=dest,
                max_words=max_words,
            )
            created.append(p)
            time.sleep(0.3)
        except Exception as e:
            print(f"[補充エラー] {target_date} ({title}): {e}")

    return created


def main() -> None:
    parser = argparse.ArgumentParser(description="WikiLens 問題データ生成スクリプト")
    parser.add_argument("--date", type=str, help="問題ID (YYYY-MM-DD)")
    parser.add_argument("--title", type=str, help="Wikipedia記事タイトル")
    parser.add_argument("--aliases", type=str, default="", help="カンマ区切りの追加別名")
    parser.add_argument("--mode", type=str, default="auto", choices=["auto", "image", "text"], help="画像版か文字版か")
    parser.add_argument("--preset", action="store_true", help="初期収録用の10問（画像版＋文字版の混合）を一括生成する")
    parser.add_argument("--out-dir", type=str, default="", help="出力先ディレクトリ（デフォルト: data/）")
    parser.add_argument("--vocab-size", type=int, default=100000, help="収録語彙数 (デフォルト: 100000)")
    parser.add_argument("--rerank-existing", action="store_true", help="data/ にある既存の問題の距離データだけを、今の辞書で作り直す（画像・出題文はそのまま）")
    parser.add_argument("--refill", action="store_true", help="候補プールから次のN日分を queue/ に自動補充する")
    parser.add_argument("--days", type=int, default=10, help="--refill 時の生成日数 (デフォルト: 10)")
    args = parser.parse_args()

    out_dir = Path(args.out_dir) if args.out_dir else DATA_DIR

    print("モデル intfloat/multilingual-e5-small をロード中...")
    model = SentenceTransformer("intfloat/multilingual-e5-small")
    base_vocab = build_base_vocabulary(target_size=args.vocab_size)
    base_embeddings = get_base_embeddings(model, base_vocab)

    generated = []
    if args.rerank_existing:
        generated = rerank_existing_problems(model, base_vocab, base_embeddings, max_words=args.vocab_size)
        update_manifest(generated, manifest_dir=DATA_DIR)
        return
    elif args.refill:
        refilled = refill_problems(model, base_vocab, base_embeddings, days=args.days, max_words=args.vocab_size)
        public_items = [p for p in refilled if (DATA_DIR / f"{p['id']}.json").exists()]
        if public_items:
            update_manifest(public_items, manifest_dir=DATA_DIR)
        return
    elif args.preset:
        # 画像版（象徴的な画像がある概念）と文字版（画像がない・抽象度の高い概念）をバランスよく混合
        presets = [
            # 10/23 (今日): 時間 (画像版: 懐中時計)
            ("2026-10-23", "時間", ["じかん", "タイム", "時の流れ"], "image", "懐中時計。時間は変化や出来事を認識するための基礎的な概念である。"),
            # 10/22: 記憶 (文字版: 画像なし・抽象心理/脳科学概念)
            ("2026-10-22", "記憶", ["きおく", "メモリー"], "text", None),
            # 10/21: 重力 (画像版: 相互作用銀河)
            ("2026-10-21", "重力", ["じゅうりょく", "万有引力", "引力"], "image", "重力によって引き合う2つの銀河（ハッブル宇宙望遠鏡撮影）。"),
            # 10/20: 孤独 (文字版: 心理・社会的状態)
            ("2026-10-20", "孤独", ["こどく", "孤立", "ソリチュード"], "text", None),
            # 10/19: 思考 (画像版: ポンペイ壁画)
            ("2026-10-19", "思考", ["しこう", "考える", "思索"], "image", "ポンペイの壁画『紙とペンを持つ女』。「思考」や知性を象徴する古代ローマの肖像。"),
            # 10/18: 正義 (画像版: 目隠しの正義の女神)
            ("2026-10-18", "正義", ["せいぎ", "ジャスティス"], "image", "最高裁判所前にある、目隠しをして剣を携えた「正義の女神」像。"),
            # 10/17: 意識 (文字版: 哲学・認知科学概念)
            ("2026-10-17", "意識", ["いしき", "コンシャスネス", "覚醒"], "text", None),
            # 10/16: 感情 (画像版: 16の表情)
            ("2026-10-16", "感情", ["かんじょう", "情動", "気持ち", "喜怒哀楽"], "image", "人間の様々な情熱・感情（喜怒哀楽）を表現した16の顔の銅版画。"),
            # 10/15: 情報 (文字版: 情報理論・通信概念)
            ("2026-10-15", "情報", ["じょうほう", "インフォメーション"], "text", None),
            # 10/14: 運 (画像版: 四つ葉のクローバー)
            ("2026-10-14", "運", ["うん", "幸運", "運勢", "ツキ"], "image", "幸運の象徴とされる四つ葉のクローバー。"),
        ]
        for pid, title, aliases, mode_req, caption in presets:
            try:
                p = generate_problem(
                    pid,
                    title,
                    model,
                    base_vocab,
                    base_embeddings,
                    extra_aliases=aliases,
                    force_mode=mode_req,
                    custom_caption=caption,
                    out_dir=out_dir,
                    max_words=args.vocab_size,
                )
                generated.append(p)
                time.sleep(0.3)
            except Exception as e:
                print(f"[エラー] {pid} ({title}) の生成に失敗: {e}")
    elif args.date and args.title:
        aliases = [a.strip() for a in args.aliases.split(",") if a.strip()]
        p = generate_problem(
            args.date,
            args.title,
            model,
            base_vocab,
            base_embeddings,
            extra_aliases=aliases,
            force_mode=args.mode,
            out_dir=out_dir,
            max_words=args.vocab_size,
        )
        generated.append(p)
    else:
        parser.print_help()
        return

    if generated:
        update_manifest(generated, manifest_dir=out_dir)


if __name__ == "__main__":
    main()
