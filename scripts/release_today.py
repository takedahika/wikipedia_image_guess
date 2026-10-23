#!/usr/bin/env python3
"""
WikiLens 毎日リリーススクリプト
- 日本時間 (JST: UTC+9) の現在日付、または指定日付を取得
- queue/ に保管されている問題データのうち、公開日を迎えたものを data/ に移動
- data/manifest.json を再構築（未来の問題は絶対に掲載しない）
"""

from __future__ import annotations

import argparse
import datetime
import json
import shutil
from pathlib import Path
from typing import Any

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
QUEUE_DIR = BASE_DIR / "queue"


def get_jst_today() -> str:
    """日本時間 (JST: UTC+9) の YYYY-MM-DD を取得"""
    jst = datetime.timezone(datetime.timedelta(hours=9))
    now = datetime.datetime.now(jst)
    return now.strftime("%Y-%m-%d")


def release(target_date: str | None = None) -> None:
    today_str = target_date or get_jst_today()
    print(f"=== WikiLens 公開処理 (基準日: {today_str} JST) ===")

    DATA_DIR.mkdir(parents=True, exist_ok=True)
    images_dir = DATA_DIR / "images"
    images_dir.mkdir(parents=True, exist_ok=True)

    # 1. queue/ から公開日を迎えたファイル (<= today_str) を data/ に移動
    released_count = 0
    if QUEUE_DIR.exists():
        for qf in sorted(QUEUE_DIR.glob("*.json")):
            pid = qf.stem
            if pid <= today_str:
                dest = DATA_DIR / qf.name
                shutil.move(str(qf), str(dest))
                print(f"  [リリース] {qf.name} -> data/{dest.name}")
                released_count += 1

                # 付属画像があれば移動
                q_img = QUEUE_DIR / "images" / f"{pid}.jpg"
                if q_img.exists():
                    shutil.move(str(q_img), str(images_dir / f"{pid}.jpg"))
                    print(f"  [画像移動] {q_img.name} -> data/images/")

    # 2. data/ 内の全問題をスキャンし、manifest.json を作成 (<= today_str のみ)
    problems_meta: list[dict[str, Any]] = []
    for jf in sorted(DATA_DIR.glob("2026-*.json"), reverse=True):
        pid = jf.stem
        if pid > today_str:
            # 未来の問題が紛れ込んでいれば queue/ に退避
            QUEUE_DIR.mkdir(parents=True, exist_ok=True)
            shutil.move(str(jf), str(QUEUE_DIR / jf.name))
            print(f"  [未来分退避] {jf.name} -> queue/{jf.name}")
            continue

        try:
            d = json.loads(jf.read_text(encoding="utf-8"))
            problems_meta.append({
                "id": d["id"],
                "mode": d.get("mode", "image"),
                "total_words": d.get("total_words", 20000),
            })
        except Exception as e:
            print(f"  [警告] {jf.name} 読み込み失敗: {e}")

    # 日付降順ソート
    problems_meta.sort(key=lambda x: x["id"], reverse=True)
    latest_id = problems_meta[0]["id"] if problems_meta else ""

    manifest = {
        "latest": latest_id,
        "updated_at": datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "problems": problems_meta,
    }

    manifest_file = DATA_DIR / "manifest.json"
    manifest_file.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"マニフェスト更新完了: {len(problems_meta)} 問公開中 (最新: {latest_id})")


def main() -> None:
    parser = argparse.ArgumentParser(description="WikiLens 本日分の公開スクリプト")
    parser.add_argument("--date", type=str, help="基準日 (YYYY-MM-DD)。省略時はJSTの本日")
    args = parser.parse_args()
    release(args.date)


if __name__ == "__main__":
    main()
