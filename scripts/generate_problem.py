#!/usr/bin/env python3
"""
WikiLens バッチ問題生成スクリプト
- MediaWiki API から記事情報・代表画像・CCライセンスメタデータ・内部リンク・リダイレクトを取得
- Wikimedia Commons のサムネイル画像を日付ID名で保存（ファイル名からの正解ネタバレ防止）
- intfloat/multilingual-e5-small を用いて日本語頻出語彙（約20,000語）とのハイブリッド類似度を計算
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
    # 数値・英字・記号のみ、または数字や記号を含む語は除外
    if re.search(r"[0-9０-９!！?？\(\)（）\[\]「」『』【】/:：;；,、.。・\s　\"'~〜\-_＝=＋+＆&]", word):
        return False
    # ASCII文字のみは除外
    if re.fullmatch(r"[a-zA-Z]+", word):
        return False
    # ひらがな2文字（助詞・副詞・活用語尾が多い）は除外
    if re.fullmatch(r"[ぁ-ん]{2}", word):
        return False
    # カタカナ2文字の断片ノイズを抑制（一般的な外来語のみ許容）
    if re.fullmatch(r"[ァ-ヴー]{2}", word):
        if get_ja_freq(word) < 1.2e-5:
            return False
    # 動詞・形容詞の活用形や助詞結合のノイズを除外
    if re.search(
        r"(ました|ません|ている|ていた|ており|された|される|として|について|による|において|における|から|まで|だけ|など|くらい|ばかり|ながら|けれど|なかった|なければ|といった|という|といった)$",
        word,
    ):
        return False
    # ひらがなのみの3文字以上で動詞・形容詞・助詞語尾のものを除外
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


def fetch_wikipedia_article(title: str) -> dict[str, Any]:
    """MediaWiki API から記事テキスト、画像、ライセンス情報、カテゴリ、セクション、リンクを取得する"""
    session = requests.Session()
    session.headers.update({"User-Agent": USER_AGENT})

    # 1. 導入部テキスト・カテゴリ・リダイレクト・代表画像ファイル名・リンクを取得
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

    # 本文全体の先頭4000文字も取得（関連語の出現判定用）
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

    # カテゴリ抽出（メンテナンス用カテゴリを除外）
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

    # 内部リンク語抽出（自然な一般名詞・固有名詞のみ厳選）
    linked_words = []
    for lk in page.get("links", []):
        lt = normalize_word(lk["title"])
        lt_clean = re.sub(r"\s*\([^)]+\)$", "", lt)
        if is_valid_japanese_vocab(lt_clean, strict_freq=True) and lt_clean not in mask_targets:
            linked_words.append(lt_clean)

    # 2. セクション（目次）取得
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

    # 3. 画像とWikimedia Commonsライセンス詳細の取得
    pageimage_name = page.get("pageimage")
    if not pageimage_name:
        raise ValueError(f"記事「{canonical_title}」に代表画像 (pageimage) が設定されていません。")

    img_params = {
        "action": "query",
        "format": "json",
        "titles": f"File:{pageimage_name}",
        "prop": "imageinfo",
        "iiprop": "url|extmetadata|user",
        "iiurlwidth": 800,
    }
    img_resp = session.get(WIKI_API_URL, params=img_params, timeout=20)
    img_resp.raise_for_status()
    img_pages = img_resp.json().get("query", {}).get("pages", {})
    img_page = next(iter(img_pages.values()), {})
    imageinfo_list = img_page.get("imageinfo", [])
    if not imageinfo_list:
        raise ValueError(f"画像メタデータが取得できません: File:{pageimage_name}")

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
    # 「自身の作品」等の定型文しかない場合はアップロード者名で補う
    if re.search(r"(投稿者自身|自身による著作物|Own work|own work)", author) or author == "Wikimedia Commons Contributor":
        author = ii.get("user") or author

    license_short = strip_html(ext.get("LicenseShortName", {}).get("value", "CC BY-SA"))
    license_url = ext.get("LicenseUrl", {}).get("value", "https://creativecommons.org/licenses/by-sa/4.0/")

    # キャプションは日本語記事の導入部1文目を優先（英語等のメタデータ混入防止）
    first_sentence = intro_extract.split("。")[0] + "。" if intro_extract else canonical_title
    caption = first_sentence[:140]

    # プレイヤーにとって最も馴染みのある表記（指定タイトル）をターゲット名として採用
    display_target = normalize_word(title)

    return {
        "title": display_target,
        "canonical_title": canonical_title,
        "extract": intro_extract,
        "body_extract": body_extract,
        "page_length": page_length,
        "categories": clean_cats[:8],
        "sections": sections[:8],
        "redirects": list(dict.fromkeys(redirects))[:20],
        "linked_words": list(dict.fromkeys(linked_words)),
        "article_url": f"https://ja.wikipedia.org/wiki/{quote(canonical_title)}",
        "image_meta": {
            "filename": pageimage_name,
            "thumb_url": thumb_url,
            "commons_url": commons_url,
            "author": author,
            "license": license_short,
            "license_url": license_url,
            "caption": caption,
        },
    }


def download_and_anonymize_image(thumb_url: str, problem_id: str) -> str:
    """
    サムネイル画像をダウンロードし、ファイル名から正解がネタバレしないよう
    ./data/images/<problem_id>.<ext> として保存する
    """
    IMAGES_DIR.mkdir(parents=True, exist_ok=True)
    parsed_path = urlparse(thumb_url).path.lower()
    ext = ".jpg"
    if parsed_path.endswith(".png"):
        ext = ".png"
    elif parsed_path.endswith(".webp"):
        ext = ".webp"

    local_filename = f"{problem_id}{ext}"
    local_path = IMAGES_DIR / local_filename

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
    visual_clues: list[str] | None = None,
    max_words: int = 20000,
) -> list[str]:
    """
    人間にとって直感的で納得感の高いハイブリッド類似度計算:
    1. ターゲット概念（タイトル＋主要カテゴリ＋視覚モチーフ）との意味類似度
    2. 記事導入部テキストとの意味類似度
    3. 記事本文・内部リンク・視覚モチーフ語での共起ボーナス
    """
    target = article_info["title"]
    canonical = article_info["canonical_title"]
    extract = article_info["extract"]
    body_extract = article_info["body_extract"]
    categories = article_info["categories"]
    linked_set = set(article_info["linked_words"])
    visual_set = set(visual_clues or [])

    # ベース語彙にない主要内部リンク語・視覚モチーフ語を追加してエンコード
    base_set = set(base_vocab)
    candidates_extra = list(visual_clues or []) + article_info["linked_words"]
    extra_words = []
    seen_extra = set()
    for w in candidates_extra:
        if w not in base_set and w != target and w != canonical and w not in seen_extra:
            seen_extra.add(w)
            extra_words.append(w)
            if len(extra_words) >= 800:
                break

    all_words = list(base_vocab)
    if extra_words:
        extra_prompts = [f"query: {w}" for w in extra_words]
        extra_embs = model.encode(extra_prompts, normalize_embeddings=True, show_progress_bar=False, batch_size=128)
        all_words.extend(extra_words)
        all_embs = np.vstack([base_embeddings, extra_embs])
    else:
        all_embs = base_embeddings

    # ターゲット概念ベクトル
    title_emb = model.encode(f"query: {target}", normalize_embeddings=True, show_progress_bar=False)
    visual_str = " ".join(visual_clues[:6]) if visual_clues else ""
    concept_text = f"query: {target} {' '.join(categories[:5])} {visual_str}"
    concept_emb = model.encode(concept_text, normalize_embeddings=True, show_progress_bar=False)
    summary_text = f"passage: {target}。{extract[:600]}"
    summary_emb = model.encode(summary_text, normalize_embeddings=True, show_progress_bar=False)

    sim_title = all_embs @ title_emb
    sim_concept = all_embs @ concept_emb
    sim_summary = all_embs @ summary_emb

    scores = 0.35 * sim_title + 0.35 * sim_concept + 0.30 * sim_summary

    # 共起・内部リンク・視覚モチーフボーナスと表層文字一致ノイズの補正
    for idx, w in enumerate(all_words):
        if w == target or w == canonical:
            continue

        in_visual = w in visual_set
        in_link = w in linked_set
        in_intro = len(w) >= 2 and (w in extract)
        in_body = len(w) >= 2 and (w in body_extract)

        if in_visual:
            scores[idx] += 0.085
        if in_link and in_intro:
            scores[idx] += 0.070
        elif in_link and in_body:
            scores[idx] += 0.050
        elif in_intro:
            scores[idx] += 0.042
        elif in_link:
            scores[idx] += 0.032
        elif in_body:
            scores[idx] += 0.020

        # カタカナ2〜3文字がターゲット文字列の一部と偶然一致しただけの表層ノイズを抑制
        if len(w) <= 3 and w in target and not in_link and not in_intro and not in_visual:
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
    visual_clues: list[str] | None = None,
    custom_caption: str | None = None,
    max_words: int = 20000,
) -> dict[str, Any]:
    """1問分の問題データJSONと画像を生成する"""
    print(f"\n[{problem_id}] 記事「{target_title}」を取得中...")
    info = fetch_wikipedia_article(target_title)
    local_img_url = download_and_anonymize_image(info["image_meta"]["thumb_url"], problem_id)

    aliases = list(dict.fromkeys((extra_aliases or []) + info["redirects"]))
    ranked_words = compute_rankings(
        model,
        base_vocab,
        base_embeddings,
        info,
        visual_clues=visual_clues,
        max_words=max_words,
    )

    masked_summary = create_masked_summary(info["extract"], info["title"], aliases)
    caption = custom_caption or info["image_meta"]["caption"]

    # 不適切なメタカテゴリを除外
    clean_cats = [c for c in info["categories"] if c != "〇〇" and "行" not in c][:5]
    if not clean_cats:
        clean_cats = ["哲学・科学の概念"]

    problem_data = {
        "id": problem_id,
        "target": info["title"],
        "aliases": aliases,
        "total_words": len(ranked_words),
        "article_url": info["article_url"],
        "image": {
            "url": local_img_url,
            "commons_url": info["image_meta"]["commons_url"],
            "caption": caption,
            "author": info["image_meta"]["author"],
            "license": info["image_meta"]["license"],
            "license_url": info["image_meta"]["license_url"],
        },
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

    DATA_DIR.mkdir(parents=True, exist_ok=True)
    out_path = DATA_DIR / f"{problem_id}.json"
    out_path.write_text(json.dumps(problem_data, ensure_ascii=False, separators=(",", ":")), encoding="utf-8")
    size_kb = out_path.stat().st_size / 1024
    print(f"[{problem_id}] 生成完了 -> {out_path.name} ({size_kb:.1f} KB), 上位12語: {ranked_words[:12]}")
    return problem_data


def update_manifest(problems_meta: list[dict[str, Any]]) -> None:
    """問題一覧マニフェスト (/data/manifest.json) を更新する（ネタバレ情報は含めない）"""
    manifest_path = DATA_DIR / "manifest.json"
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
    parser.add_argument("--visuals", type=str, default="", help="カンマ区切りの画像モチーフ語")
    parser.add_argument("--preset", action="store_true", help="初期収録用の抽象概念問題セット（10問）を一括生成する")
    parser.add_argument("--vocab-size", type=int, default=20000, help="収録語彙数 (デフォルト: 20000)")
    args = parser.parse_args()

    print("モデル intfloat/multilingual-e5-small をロード中...")
    model = SentenceTransformer("intfloat/multilingual-e5-small")
    base_vocab = build_base_vocabulary(target_size=args.vocab_size)
    base_embeddings = get_base_embeddings(model, base_vocab)

    generated = []
    if args.preset:
        # 象徴的な画像から抽象概念・事象を推理する厳選10問
        presets = [
            (
                "2026-10-23",
                "時間",
                ["じかん", "タイム", "時の流れ"],
                ["時計", "懐中時計", "時刻", "過去", "未来", "現在", "秒", "針", "暦", "永遠", "空間", "歴史"],
                "懐中時計。時計は時間を計測・視覚化するための代表的な象徴である。",
            ),
            (
                "2026-10-22",
                "知識",
                ["ちしき", "ナレッジ"],
                ["図書館", "遺跡", "書物", "本", "学問", "学習", "知恵", "情報", "科学", "哲学", "古代ローマ", "認識"],
                "エフェソスのケルスス図書館跡。図書館は人類の知識の集積を象徴する。",
            ),
            (
                "2026-10-21",
                "重力",
                ["じゅうりょく", "万有引力", "引力"],
                ["銀河", "宇宙", "天体", "惑星", "星", "軌道", "相対性理論", "質量", "物理", "地球", "ブラックホール"],
                "重力によって相互作用し引き合う2つの銀河（ハッブル宇宙望遠鏡撮影）。",
            ),
            (
                "2026-10-20",
                "美",
                ["び", "美しさ", "美的"],
                ["ステンドグラス", "バラ窓", "大聖堂", "教会", "芸術", "美学", "調和", "対称性", "色彩", "光", "装飾"],
                "パリ・ノートルダム大聖堂のバラ窓。光の調和と対称性は美の象徴とされる。",
            ),
            (
                "2026-10-19",
                "思考",
                ["しこう", "考える", "思索"],
                ["壁画", "絵画", "ペン", "女性", "ポンペイ", "哲学", "知性", "論理", "心", "意識", "推論", "発想"],
                "ポンペイの壁画『紙とペンを持つ女』。「思考」や知性を象徴する古代ローマの肖像。",
            ),
            (
                "2026-10-18",
                "正義",
                ["せいぎ", "ジャスティス"],
                ["女神", "彫刻", "石像", "目隠し", "裁判", "裁判所", "法律", "法", "天秤", "剣", "倫理", "道徳", "公平"],
                "ブラジル連邦最高裁判所前にある、目隠しをして剣を携えた「正義の女神」像。",
            ),
            (
                "2026-10-17",
                "生命",
                ["せいめい", "いのち", "命"],
                ["サンゴ礁", "サンゴ", "海", "海洋", "魚", "生物", "生態系", "進化", "自然", "細胞", "地球", "誕生"],
                "多様な海洋生物が共生するサンゴ礁。地球上の豊かな生命の象徴。",
            ),
            (
                "2026-10-16",
                "感情",
                ["かんじょう", "情動", "気持ち", "喜怒哀楽"],
                ["顔", "表情", "人間", "心理", "心理学", "喜び", "怒り", "悲しみ", "恐怖", "笑い", "心", "情熱"],
                "人間の様々な情熱・感情（喜怒哀楽）を表現した16の顔の銅版画（1821年）。",
            ),
            (
                "2026-10-15",
                "文明",
                ["ぶんめい", "シヴィライゼーション"],
                ["スフィンクス", "ピラミッド", "エジプト", "遺跡", "古代", "歴史", "都市", "文化", "人類", "社会", "建築"],
                "ギザの大スフィンクスとピラミッド。古代エジプトは人類最古の文明のひとつである。",
            ),
            (
                "2026-10-14",
                "運",
                ["うん", "幸運", "運勢", "ツキ"],
                ["クローバー", "四つ葉", "植物", "葉", "幸福", "偶然", "確率", "運命", "縁起", "奇跡", "チャンス"],
                "幸運の象徴とされる四つ葉のクローバー。",
            ),
        ]
        for pid, title, aliases, visuals, caption in presets:
            try:
                p = generate_problem(
                    pid,
                    title,
                    model,
                    base_vocab,
                    base_embeddings,
                    extra_aliases=aliases,
                    visual_clues=visuals,
                    custom_caption=caption,
                    max_words=args.vocab_size,
                )
                generated.append(p)
                time.sleep(0.3)
            except Exception as e:
                print(f"[エラー] {pid} ({title}) の生成に失敗: {e}")
    elif args.date and args.title:
        aliases = [a.strip() for a in args.aliases.split(",") if a.strip()]
        visuals = [v.strip() for v in args.visuals.split(",") if v.strip()]
        p = generate_problem(
            args.date,
            args.title,
            model,
            base_vocab,
            base_embeddings,
            extra_aliases=aliases,
            visual_clues=visuals,
            max_words=args.vocab_size,
        )
        generated.append(p)
    else:
        parser.print_help()
        return

    if generated:
        update_manifest(generated)


if __name__ == "__main__":
    main()
