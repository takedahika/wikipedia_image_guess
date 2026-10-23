#!/usr/bin/env python3
"""
WikiLens バッチ問題生成スクリプト
- MediaWiki API から記事情報・代表画像（画像版）または冒頭文（文字版）を取得
- 画像がある記事は「画像版」、ない記事（孤独・記憶・意識・愛等）は「文字版」として生成
- intfloat/multilingual-e5-small を用いて日本語頻出語彙（約20,000語）とのセマンティック類似度を計算
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

# メンテナンス系カテゴリの除外パターン
HIDDEN_CAT_PATTERNS = [
    r"スタブ", r"ウィキデータ", r"良質な記事", r"秀逸な記事", r"出典", r"検証",
    r"独自研究", r"リンク切れ", r"曖昧さ回避", r"一覧", r"記事", r"ページ",
    r"テンプレート", r"カテゴリ", r"画像", r"座標", r"識別子", r"典拠管理",
    r"ISBN", r"ISSN", r"LCCN", r"NDL", r"VIAF", r"Commons", r"コモンズ",
]


def normalize_word(word: str) -> str:
    """全角半角正規化（NFKC）とトリム"""
    return unicodedata.normalize("NFKC", word.strip())


def is_valid_japanese_vocab(word: str, strict_freq: bool = False) -> bool:
    """ゲーム用の語彙として適切かどうかを判定する"""
    if not word or len(word) < 2 or len(word) > 10:
        return False
    if word in STOPWORDS:
        return False
    if re.search(r"[0-9０-９!！?？\(\)（）\[\]「」『』【】/:：;；,、.。・\s　\"'~〜\-_＝=＋+＆&]", word):
        return False
    if re.fullmatch(r"[a-zA-Z]+", word):
        return False
    if re.fullmatch(r"[ぁ-ん]{2}", word):
        return False
    if re.fullmatch(r"[ァ-ヴー]{2}", word):
        if get_ja_freq(word) < 1.2e-5:
            return False
    if re.search(
        r"(ました|ません|ている|ていた|ており|された|される|として|について|による|において|における|から|まで|だけ|など|くらい|ばかり|ながら|けれど|なかった|なければ|といった|という|といった)$",
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


def build_base_vocabulary(target_size: int = 20000) -> list[str]:
    """wordfreq の日本語頻出語からクリーンな語彙リストを構築してキャッシュする"""
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    vocab_cache_file = CACHE_DIR / f"base_vocab_v2_{target_size}.json"
    if vocab_cache_file.exists():
        return json.loads(vocab_cache_file.read_text(encoding="utf-8"))

    print("ベース語彙プールを構築中...")
    raw_words = top_n_list("ja", 150000)
    clean_words: list[str] = []
    seen: set[str] = set()

    for w in raw_words:
        nw = normalize_word(w)
        if nw in seen:
            continue
        if is_valid_japanese_vocab(nw):
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
        if len(t) >= 2:
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


def create_masked_summary(extract: str, target: str, aliases: list[str]) -> str:
    """導入部の最初の2文から正解ワードと別名を伏せ字（███）にしたヒント文を生成する"""
    sentences = [s.strip() for s in extract.replace("\n", "").split("。") if s.strip()]
    snippet = "。".join(sentences[:2]) + "。" if sentences else ""
    masked = snippet
    terms = sorted(set([target] + aliases), key=len, reverse=True)
    for t in terms:
        if len(t) >= 2:
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
    max_words: int = 20000,
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
    extra_words = [
        w for w in article_info["linked_words"]
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


def get_base_embeddings(model: SentenceTransformer, base_vocab: list[str]) -> np.ndarray:
    """ベース語彙の埋め込みベクトルを計算またはキャッシュから読み込む"""
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    emb_cache_file = CACHE_DIR / f"base_embs_v2_e5_small_{len(base_vocab)}.npy"
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
    max_words: int = 20000,
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

    aliases = list(dict.fromkeys((extra_aliases or []) + info["redirects"]))
    ranked_words = compute_rankings(
        model,
        base_vocab,
        base_embeddings,
        info,
        max_words=max_words,
    )

    masked_summary = create_masked_summary(info["extract"], info["title"], aliases)
    clean_cats = [c for c in info["categories"] if c != "〇〇" and "行" not in c][:5]
    if not clean_cats:
        clean_cats = ["哲学・科学の概念"]

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
        },
        "words": ranked_words,
    }

    out_path = dest_dir / f"{problem_id}.json"
    out_path.write_text(json.dumps(problem_data, ensure_ascii=False, separators=(",", ":")), encoding="utf-8")
    size_kb = out_path.stat().st_size / 1024
    print(f"[{problem_id}] 生成完了 [{mode}版] -> {out_path.name} ({size_kb:.1f} KB), 上位10語: {ranked_words[:10]}")
    return problem_data


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


def main() -> None:
    parser = argparse.ArgumentParser(description="WikiLens 問題データ生成スクリプト")
    parser.add_argument("--date", type=str, help="問題ID (YYYY-MM-DD)")
    parser.add_argument("--title", type=str, help="Wikipedia記事タイトル")
    parser.add_argument("--aliases", type=str, default="", help="カンマ区切りの追加別名")
    parser.add_argument("--mode", type=str, default="auto", choices=["auto", "image", "text"], help="画像版か文字版か")
    parser.add_argument("--preset", action="store_true", help="初期収録用の10問（画像版＋文字版の混合）を一括生成する")
    parser.add_argument("--out-dir", type=str, default="", help="出力先ディレクトリ（デフォルト: data/）")
    parser.add_argument("--vocab-size", type=int, default=20000, help="収録語彙数 (デフォルト: 20000)")
    args = parser.parse_args()

    out_dir = Path(args.out_dir) if args.out_dir else DATA_DIR

    print("モデル intfloat/multilingual-e5-small をロード中...")
    model = SentenceTransformer("intfloat/multilingual-e5-small")
    base_vocab = build_base_vocabulary(target_size=args.vocab_size)
    base_embeddings = get_base_embeddings(model, base_vocab)

    generated = []
    if args.preset:
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
