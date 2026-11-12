/**
 * WikiLens - Wikipedia連想ゲーム（6回勝負）
 * クライアントサイドロジック
 * - 最大6回の制限付き推測
 * - 不正解ごとに自動開放される5段階のヒント
 * - 画像版 ＆ 文字版（文章の段階的開放）両対応
 * - 日本時間 (JST) 基準の日替わり切り替え
 * - O(1) ハッシュマップ検索 ＆ LocalStorage永続化
 */

(() => {
  "use strict";

  const STORAGE_PREFIX = "wikilens_state_v5_";
  const MAX_TRIES = 6;

  // アプリケーション状態
  const state = {
    manifest: null,
    currentProblemId: null,
    problem: null,
    rankMap: new Map(),      // 正規化単語 -> 順位
    aliasSet: new Set(),     // 別名（1位扱い）
    guesses: [],             // [{ word, rank, attemptNum, isWin, tier, emoji }]
    cleared: false,
    failed: false,
    sortBy: "time",          // "time" | "rank"
    lastGuessWord: null,
  };

  // DOM要素参照
  const el = {
    problemSelect: document.getElementById("problem-select"),
    problemBadge: document.getElementById("problem-badge"),
    modeBadge: document.getElementById("mode-badge"),
    charLenBadge: document.getElementById("char-len-badge"),
    vocabCountBadge: document.getElementById("vocab-count-badge"),
    imageFrame: document.getElementById("image-frame"),
    imageSkeleton: document.getElementById("image-skeleton"),
    targetImage: document.getElementById("target-image"),
    btnZoom: document.getElementById("btn-zoom"),
    textPuzzleCard: document.getElementById("text-puzzle-card"),
    textSentence1: document.getElementById("text-sentence-1"),
    textSentence2: document.getElementById("text-sentence-2"),
    textSentence3: document.getElementById("text-sentence-3"),
    revealedCaptionBox: document.getElementById("revealed-caption-box"),
    revealedTargetName: document.getElementById("revealed-target-name"),
    revealedWikiLink: document.getElementById("revealed-wiki-link"),
    revealedCaptionText: document.getElementById("revealed-caption-text"),
    imageCreditFooter: document.getElementById("image-credit-footer"),
    creditAuthor: document.getElementById("credit-author"),
    creditLicense: document.getElementById("credit-license"),
    btnToggleSource: document.getElementById("btn-toggle-source"),
    creditSourceDrawer: document.getElementById("credit-source-drawer"),
    creditCommonsUrl: document.getElementById("credit-commons-url"),
    triesTracker: document.getElementById("tries-tracker"),
    triesRemainingLabel: document.getElementById("tries-remaining-label"),
    btnGiveup: document.getElementById("btn-giveup"),
    btnResultModal: document.getElementById("btn-result-modal"),
    btnNextPill: document.getElementById("btn-next-pill"),
    tabModeImage: document.getElementById("tab-mode-image"),
    tabModeText: document.getElementById("tab-mode-text"),
    tabCountImage: document.getElementById("tab-count-image"),
    tabCountText: document.getElementById("tab-count-text"),
    btnNextImageInline: document.getElementById("btn-next-image-inline"),
    btnNextTextInline: document.getElementById("btn-next-text-inline"),
    btnNextImageModal: document.getElementById("btn-next-image-modal"),
    btnNextTextModal: document.getElementById("btn-next-text-modal"),
    guessForm: document.getElementById("guess-form"),
    guessInput: document.getElementById("guess-input"),
    btnSubmit: document.getElementById("btn-submit"),
    latestGuessContainer: document.getElementById("latest-guess-container"),
    latestGuessCard: document.getElementById("latest-guess-card"),
    unlockedHintsCount: document.getElementById("unlocked-hints-count"),
    hintEmptyState: document.getElementById("hint-empty-state"),
    hintItem1: document.getElementById("hint-item-1"),
    hintItem1Body: document.getElementById("hint-item-1-body"),
    hintItem2: document.getElementById("hint-item-2"),
    hintItem2Body: document.getElementById("hint-item-2-body"),
    hintItem3: document.getElementById("hint-item-3"),
    hintItem3Body: document.getElementById("hint-item-3-body"),
    hintItem4: document.getElementById("hint-item-4"),
    hintItem4Body: document.getElementById("hint-item-4-body"),
    hintItem5: document.getElementById("hint-item-5"),
    hintItem5Body: document.getElementById("hint-item-5-body"),
    historyCount: document.getElementById("history-count"),
    sortByTime: document.getElementById("sort-by-time"),
    sortByRank: document.getElementById("sort-by-rank"),
    emptyHistory: document.getElementById("empty-history"),
    historyList: document.getElementById("history-list"),
    resultModal: document.getElementById("result-modal"),
    btnCloseResult: document.getElementById("btn-close-result"),
    modalBadgeIcon: document.getElementById("modal-badge-icon"),
    modalResultTitle: document.getElementById("modal-result-title"),
    modalResultSubtitle: document.getElementById("modal-result-subtitle"),
    modalAnswerWord: document.getElementById("modal-answer-word"),
    modalAnswerCaption: document.getElementById("modal-answer-caption"),
    modalWikiUrl: document.getElementById("modal-wiki-url"),
    modalStatAttempts: document.getElementById("modal-stat-attempts"),
    modalStatBest: document.getElementById("modal-stat-best"),
    modalStatMode: document.getElementById("modal-stat-mode"),
    shareTextarea: document.getElementById("share-textarea"),
    btnCopyShare: document.getElementById("btn-copy-share"),
    btnXShare: document.getElementById("btn-x-share"),
    helpModal: document.getElementById("help-modal"),
    btnHelp: document.getElementById("btn-help"),
    btnCloseHelp: document.getElementById("btn-close-help"),
    zoomModal: document.getElementById("zoom-modal"),
    zoomImage: document.getElementById("zoom-image"),
    btnCloseZoom: document.getElementById("btn-close-zoom"),
    toastContainer: document.getElementById("toast-container"),
  };

  /**
   * 日本時間 (JST: UTC+9) の YYYY-MM-DD を取得
   */
  function getJSTToday() {
    const now = new Date();
    const utc = now.getTime() + now.getTimezoneOffset() * 60000;
    const jst = new Date(utc + 9 * 3600000);
    const y = jst.getFullYear();
    const m = String(jst.getMonth() + 1).padStart(2, "0");
    const d = String(jst.getDate()).padStart(2, "0");
    return `${y}-${m}-${d}`;
  }

  function normalizeInput(raw) {
    if (!raw) return "";
    return raw
      .normalize("NFKC")
      .trim()
      .replace(/[\s　]+/g, "");
  }

  function toHiragana(str) {
    return str.replace(/[\u30a1-\u30f6]/g, (ch) =>
      String.fromCharCode(ch.charCodeAt(0) - 0x60)
    );
  }

  function toKatakana(str) {
    return str.replace(/[\u3041-\u3096]/g, (ch) =>
      String.fromCharCode(ch.charCodeAt(0) + 0x60)
    );
  }

  function escapeHtml(str) {
    return String(str)
      .replace(/&/g, "&amp;")
      .replace(/</g, "&lt;")
      .replace(/>/g, "&gt;")
      .replace(/"/g, "&quot;");
  }

  function showToast(message, type = "info") {
    const toast = document.createElement("div");
    toast.className = `toast ${type}`;
    toast.textContent = message;
    el.toastContainer.appendChild(toast);
    setTimeout(() => {
      toast.style.opacity = "0";
      toast.style.transition = "opacity 0.25s";
      setTimeout(() => toast.remove(), 260);
    }, 2800);
  }

  /**
   * 正解との距離とビジュアルスタイル（絵文字・幅・色帯・ラベル）を算出
   * rank 1 (正解) -> distance 0
   * rank 2 -> distance 1
   */
  function getRankVisuals(rank, totalWords) {
    const total = totalWords || 100000;
    const distance = Math.max(0, rank - 1);

    if (distance === 0) {
      return {
        distance: 0,
        tier: "tier-top",
        widthPct: 100,
        emoji: "🎯",
        colorEmoji: "🟩",
        label: "距離 0 🎯 正解！",
        shortLabel: "距離 0",
      };
    }
    if (distance <= 200) {
      const pct = Math.max(82, 99 - (distance / 200) * 17);
      return {
        distance,
        tier: "tier-hot",
        widthPct: pct,
        emoji: "🔥",
        colorEmoji: "🟩",
        label: `距離 ${distance.toLocaleString()} 🔥`,
        shortLabel: `距離 ${distance.toLocaleString()}`,
      };
    }
    if (distance <= 1500) {
      const pct = Math.max(62, 82 - ((distance - 200) / 1300) * 20);
      return {
        distance,
        tier: "tier-warm",
        widthPct: pct,
        emoji: "✨",
        colorEmoji: "🟦",
        label: `距離 ${distance.toLocaleString()} ✨`,
        shortLabel: `距離 ${distance.toLocaleString()}`,
      };
    }
    if (distance <= 6000) {
      const pct = Math.max(38, 62 - ((distance - 1500) / 4500) * 24);
      return {
        distance,
        tier: "tier-mid",
        widthPct: pct,
        emoji: "💡",
        colorEmoji: "🟨",
        label: `距離 ${distance.toLocaleString()} 💡`,
        shortLabel: `距離 ${distance.toLocaleString()}`,
      };
    }
    if (distance <= 20000) {
      const pct = Math.max(18, 38 - ((distance - 6000) / 14000) * 20);
      return {
        distance,
        tier: "tier-cool",
        widthPct: pct,
        emoji: "🟠",
        colorEmoji: "🟧",
        label: `距離 ${distance.toLocaleString()} 🟠`,
        shortLabel: `距離 ${distance.toLocaleString()}`,
      };
    }
    const pct = Math.max(5, 18 - ((distance - 20000) / Math.max(1, total - 20000)) * 13);
    return {
      distance,
      tier: "tier-cold",
      widthPct: pct,
      emoji: "❄️",
      colorEmoji: "⬜",
      label: `距離 ${distance.toLocaleString()} ❄️`,
      shortLabel: `距離 ${distance.toLocaleString()}`,
    };
  }

  /**
   * LocalStorageへの保存と復元
   */
  function saveProgress() {
    if (!state.currentProblemId) return;
    const payload = {
      id: state.currentProblemId,
      guesses: state.guesses,
      cleared: state.cleared,
      failed: state.failed,
      lastGuessWord: state.lastGuessWord,
      savedAt: Date.now(),
    };
    try {
      localStorage.setItem(STORAGE_PREFIX + state.currentProblemId, JSON.stringify(payload));
      updateProblemSelectBadges();
    } catch (e) {
      console.warn("LocalStorage save failed:", e);
    }
  }

  function loadProgress(problemId) {
    try {
      const raw = localStorage.getItem(STORAGE_PREFIX + problemId);
      if (!raw) return false;
      const saved = JSON.parse(raw);
      if (saved && saved.id === problemId) {
        state.guesses = Array.isArray(saved.guesses) ? saved.guesses : [];
        state.cleared = Boolean(saved.cleared);
        state.failed = Boolean(saved.failed);
        state.lastGuessWord = saved.lastGuessWord || null;
        return true;
      }
    } catch (e) {
      console.warn("LocalStorage load failed:", e);
    }
    return false;
  }

  function getSavedStatusIcon(problemId) {
    try {
      const raw = localStorage.getItem(STORAGE_PREFIX + problemId);
      if (!raw) return "";
      const saved = JSON.parse(raw);
      if (saved.cleared) return "✅ ";
      if (saved.failed) return "🏳️ ";
      if (saved.guesses && saved.guesses.length > 0) return "⏳ ";
    } catch (_) {}
    return "";
  }

  function isProblemFinished(problemId) {
    try {
      const raw = localStorage.getItem(STORAGE_PREFIX + problemId);
      if (!raw) return false;
      const saved = JSON.parse(raw);
      return Boolean(saved && (saved.cleared || saved.failed));
    } catch (_) {
      return false;
    }
  }

  /**
   * 指定モード（"image" | "text" | null）の次に遊ぶべき問題を取得
   */
  function getNextPlayableProblem(targetMode = null) {
    const allProblems = state.manifest?.problems;
    if (!Array.isArray(allProblems) || allProblems.length === 0) return null;

    const filtered = targetMode
      ? allProblems.filter((p) => (p.mode || "image") === targetMode)
      : allProblems;
    if (filtered.length === 0) return null;

    const total = filtered.length;
    const currentIdx = filtered.findIndex((p) => p.id === state.currentProblemId);

    // 現在の問題がこのモードに含まれない場合は、先頭（最新）から未終了を探す
    if (currentIdx === -1) {
      for (let i = 0; i < total; i++) {
        if (!isProblemFinished(filtered[i].id)) {
          return { problem: filtered[i], num: total - i, isUnfinished: true };
        }
      }
      return { problem: filtered[0], num: total, isUnfinished: false };
    }

    // 現在の問題から次の未終了問題を探す
    for (let offset = 1; offset < total; offset++) {
      const idx = (currentIdx + offset) % total;
      const p = filtered[idx];
      if (!isProblemFinished(p.id)) {
        return { problem: p, num: total - idx, isUnfinished: true };
      }
    }

    // 他の問題がすべて終了済みで、現在の問題が未終了なら現在の問題を返す
    if (!isProblemFinished(filtered[currentIdx].id)) {
      return { problem: filtered[currentIdx], num: total - currentIdx, isUnfinished: true };
    }

    const nextIdx = (currentIdx + 1) % total;
    return { problem: filtered[nextIdx], num: total - nextIdx, isUnfinished: false };
  }

  function updateModeTabsAndNextButtons() {
    const allProblems = state.manifest?.problems || [];
    const imgProblems = allProblems.filter((p) => (p.mode || "image") === "image");
    const txtProblems = allProblems.filter((p) => p.mode === "text");

    const imgDone = imgProblems.filter((p) => isProblemFinished(p.id)).length;
    const txtDone = txtProblems.filter((p) => isProblemFinished(p.id)).length;

    const currentMode = state.problem?.mode || "image";
    if (el.tabModeImage) {
      el.tabModeImage.classList.toggle("active", currentMode === "image");
    }
    if (el.tabModeText) {
      el.tabModeText.classList.toggle("active", currentMode === "text");
    }
    if (el.tabCountImage) {
      el.tabCountImage.textContent = `${imgDone}/${imgProblems.length}問`;
    }
    if (el.tabCountText) {
      el.tabCountText.textContent = `${txtDone}/${txtProblems.length}問`;
    }

    const nextImg = getNextPlayableProblem("image");
    const nextTxt = getNextPlayableProblem("text");

    const imgBtns = [el.btnNextImageInline, el.btnNextImageModal];
    imgBtns.forEach((b) => {
      if (!b) return;
      if (!nextImg) {
        b.classList.add("hidden");
      } else {
        b.classList.remove("hidden");
        b.textContent = nextImg.isUnfinished
          ? `🖼️ 次の画像版を遊ぶ (#${nextImg.num})`
          : `🖼️ 画像版 (#${nextImg.num}・全問済)`;
      }
    });

    const txtBtns = [el.btnNextTextInline, el.btnNextTextModal];
    txtBtns.forEach((b) => {
      if (!b) return;
      if (!nextTxt) {
        b.classList.add("hidden");
      } else {
        b.classList.remove("hidden");
        b.textContent = nextTxt.isUnfinished
          ? `📝 次の文字版を遊ぶ (#${nextTxt.num})`
          : `📝 文字版 (#${nextTxt.num}・全問済)`;
      }
    });

    const nextSameMode = getNextPlayableProblem(currentMode);
    if (el.btnNextPill && nextSameMode) {
      const icon = currentMode === "text" ? "📝" : "🖼️";
      el.btnNextPill.textContent = `▶️ 次の${currentMode === "text" ? "文字" : "画像"}版 (#${nextSameMode.num}) ${icon}`;
    }
  }

  function handleNextProblem(targetMode = null) {
    const modeToUse = targetMode || state.problem?.mode || null;
    const nextInfo = getNextPlayableProblem(modeToUse);
    if (!nextInfo) return;
    el.resultModal.classList.add("hidden");
    loadProblem(nextInfo.problem.id);
    window.scrollTo({ top: 0, behavior: "smooth" });
  }

  function handleSwitchModeTab(targetMode) {
    if (!state.manifest?.problems) return;
    const currentMode = state.problem?.mode || "image";
    const isFinished = state.cleared || state.failed;

    // 別モードのタブを押した場合、または現在の問題が終了済みの場合は、そのモードの次の未プレイ問題へ移動
    if (currentMode !== targetMode || isFinished) {
      handleNextProblem(targetMode);
    } else {
      // 同じモードでプレイ中に押した場合は、同モードの別問題へ切り替え
      const modeProblems = state.manifest.problems.filter((p) => (p.mode || "image") === targetMode);
      if (modeProblems.length > 1) {
        const curIdx = modeProblems.findIndex((p) => p.id === state.currentProblemId);
        const nextP = modeProblems[(curIdx + 1) % modeProblems.length];
        loadProblem(nextP.id);
      }
    }
  }

  /**
   * 推測結果の1行DOM生成
   */
  function createGuessRowElement(item, highlight = false) {
    const total = state.problem?.total_words || 20000;
    const { tier, widthPct, label } = getRankVisuals(item.rank, total);

    const row = document.createElement("div");
    row.className = `guess-row ${tier}${highlight ? " latest-highlight" : ""}`;
    row.setAttribute("role", "listitem");

    const bar = document.createElement("div");
    bar.className = "guess-row-bar";
    bar.style.width = `${widthPct.toFixed(1)}%`;

    const left = document.createElement("div");
    left.className = "guess-row-left";

    const numSpan = document.createElement("span");
    numSpan.className = "guess-num";
    numSpan.textContent = `#${item.attemptNum}`;

    const wordSpan = document.createElement("span");
    wordSpan.className = "guess-word";
    wordSpan.textContent = item.word;

    left.appendChild(numSpan);
    left.appendChild(wordSpan);

    const right = document.createElement("div");
    right.className = "guess-row-right";

    const rankBadge = document.createElement("span");
    rankBadge.className = "guess-rank-badge";
    rankBadge.textContent = label;

    right.appendChild(rankBadge);

    row.appendChild(bar);
    row.appendChild(left);
    row.appendChild(right);
    return row;
  }

  /**
   * 画面全体の描画
   */
  function renderUI() {
    if (!state.problem) return;

    const isFinished = state.cleared || state.failed;
    const remainingTries = Math.max(0, MAX_TRIES - state.guesses.length);
    const missesCount = state.guesses.filter((g) => !g.isWin).length;

    // 1. スロット・トラッカーの更新 (Wordle風)
    el.triesRemainingLabel.textContent = isFinished
      ? (state.cleared ? `🎉 ${state.guesses.length}/6 でクリア！` : "ゲーム終了")
      : `残り ${remainingTries} 回`;

    const slotEls = el.triesTracker.querySelectorAll(".try-slot");
    slotEls.forEach((slot, index) => {
      slot.className = "try-slot";
      const slotNum = index + 1;
      const guessItem = state.guesses[index];

      if (guessItem) {
        const { tier, distance } = getRankVisuals(guessItem.rank, state.problem.total_words);
        slot.classList.add(tier);
        slot.innerHTML = guessItem.rank === 1 ? "🎯" : `<span class="slot-rank-mini">${distance}</span>`;
        slot.title = `#${slotNum}: ${guessItem.word} (${guessItem.rank === 1 ? "正解！" : "正解との距離: " + distance})`;
      } else if (slotNum === state.guesses.length + 1 && !isFinished) {
        slot.classList.add("active");
        slot.innerHTML = `<span class="slot-num">${slotNum}</span>`;
        slot.title = `現在の挑戦 (#${slotNum})`;
      } else {
        slot.innerHTML = `<span class="slot-num">${slotNum}</span>`;
        slot.title = `未挑戦`;
      }
    });

    // 2. ボタン・入力フォームの制御
    el.guessInput.disabled = isFinished;
    el.btnSubmit.disabled = isFinished;
    el.btnGiveup.classList.toggle("hidden", isFinished);
    el.btnResultModal.classList.toggle("hidden", !isFinished);
    if (el.btnNextPill) {
      el.btnNextPill.classList.toggle("hidden", !isFinished);
    }
    updateModeTabsAndNextButtons();

    if (isFinished) {
      el.guessInput.placeholder = state.cleared
        ? `🎉 正解「${state.problem.target}」を的中させました！`
        : `答え: 「${state.problem.target}」`;
      el.revealedCaptionBox.classList.remove("hidden");
      el.revealedTargetName.textContent = state.problem.target;
      el.revealedWikiLink.href =
        state.problem.article_url ||
        `https://ja.wikipedia.org/wiki/${encodeURIComponent(state.problem.target)}`;
      el.revealedCaptionText.textContent =
        state.problem.image?.caption ||
        state.problem.text_puzzle?.raw_sentences?.join(" ") ||
        "";
    } else {
      el.guessInput.placeholder = state.problem.mode === "text"
        ? "文脈から連想する言葉を入力（距離0で正解！）"
        : "写っている物や象徴する概念を入力（距離0で正解！）";
      el.revealedCaptionBox.classList.add("hidden");
    }

    // 3. 文字版パズルの段階的開放
    if (state.problem.mode === "text" && state.problem.text_puzzle) {
      const tp = state.problem.text_puzzle;
      el.textSentence1.innerHTML = isFinished
        ? escapeHtml(tp.raw_sentences?.[0] || tp.sentence_1)
        : escapeHtml(tp.sentence_1);

      if (missesCount >= 3 || isFinished) {
        el.textSentence2.className = "text-sentence active";
        el.textSentence2.innerHTML = isFinished
          ? escapeHtml(tp.raw_sentences?.[1] || tp.sentence_2)
          : escapeHtml(tp.sentence_2 || "（2文目なし）");
      } else {
        el.textSentence2.className = "text-sentence locked";
        el.textSentence2.innerHTML = `<span class="sentence-lock-tag">🔒 3回外すと開放（第2文）</span>`;
      }

      if (missesCount >= 5 || isFinished) {
        el.textSentence3.className = "text-sentence active";
        el.textSentence3.innerHTML = isFinished
          ? escapeHtml(tp.raw_sentences?.[2] || tp.sentence_3)
          : escapeHtml(tp.sentence_3 || "（3文目なし）");
      } else {
        el.textSentence3.className = "text-sentence locked";
        el.textSentence3.innerHTML = `<span class="sentence-lock-tag">🔒 5回外すと開放（第3文・ラストチャンス）</span>`;
      }
    }

    // 4. 自動開放ヒントタイムライン
    renderHintsTimeline(missesCount, isFinished);

    // 5. 直近の入力ハイライト表示
    if (state.lastGuessWord) {
      const latestItem = state.guesses.find((g) => g.word === state.lastGuessWord);
      if (latestItem) {
        el.latestGuessContainer.classList.remove("hidden");
        el.latestGuessCard.replaceWith(createGuessRowElement(latestItem, true));
        el.latestGuessCard = el.latestGuessContainer.querySelector(".guess-row");
      } else {
        el.latestGuessContainer.classList.add("hidden");
      }
    } else {
      el.latestGuessContainer.classList.add("hidden");
    }

    // 6. 履歴リスト描画
    el.historyCount.textContent = String(state.guesses.length);
    el.emptyHistory.classList.toggle("hidden", state.guesses.length > 0);
    el.historyList.innerHTML = "";

    const sorted = [...state.guesses];
    if (state.sortBy === "rank") {
      sorted.sort((a, b) => a.rank - b.rank || a.attemptNum - b.attemptNum);
    } else {
      sorted.sort((a, b) => a.attemptNum - b.attemptNum);
    }

    const fragment = document.createDocumentFragment();
    for (const item of sorted) {
      const isLatest = item.word === state.lastGuessWord;
      fragment.appendChild(createGuessRowElement(item, isLatest));
    }
    el.historyList.appendChild(fragment);

    el.sortByTime.classList.toggle("active", state.sortBy === "time");
    el.sortByRank.classList.toggle("active", state.sortBy === "rank");
  }

  /**
   * 段階的ヒントタイムラインの描画（後半3回で開放される全3段階）
   * 挑戦回数 1〜3回目 (0〜2回ミス): ヒントなし（自力連想ゾーン）
   * 挑戦回数 4回目 (3回ミス): ヒント 1 (🏷️ 分野カテゴリ ＆ 📑 目次テーマ)
   * 挑戦回数 5回目 (4回ミス): ヒント 2 (🔥 答えに非常に近い関連ワード＋距離)
   * 挑戦回数 6回目 (5回ミス): ヒント 3 (💡 最初の1文字 ＆ 📝 導入部伏せ字抜粋)
   */
  function renderHintsTimeline(missesCount, isFinished) {
    const h = state.problem.hints || {};
    let unlockedCount = 0;
    if (isFinished || missesCount >= 5) {
      unlockedCount = 3;
    } else if (missesCount >= 4) {
      unlockedCount = 2;
    } else if (missesCount >= 3) {
      unlockedCount = 1;
    } else {
      unlockedCount = 0;
    }

    el.unlockedHintsCount.textContent = String(unlockedCount);
    el.hintEmptyState.classList.toggle("hidden", unlockedCount > 0);

    // ヒント1: 分野カテゴリ & 目次テーマ (4回目挑戦時 / 3回ミス)
    if (missesCount >= 3 || isFinished) {
      el.hintItem1.classList.remove("hidden");
      const cats = Array.isArray(h.category) && h.category.length > 0
        ? h.category.map((c) => `<span class="hint-tag">${escapeHtml(c)}</span>`).join("")
        : `<span class="hint-tag">一般概念</span>`;
      const secs = Array.isArray(h.sections) && h.sections.length > 0
        ? h.sections.slice(0, 4).map((s) => `<span class="hint-tag">§ ${escapeHtml(s)}</span>`).join("")
        : "";
      el.hintItem1Body.innerHTML = `
        <div class="hint-tag-list" style="margin-bottom: 6px;">${cats}</div>
        ${secs ? `<div class="hint-tag-list">${secs}</div>` : ""}
      `;
    } else {
      el.hintItem1.classList.add("hidden");
    }

    // ヒント2: 答えに非常に近い関連ワード (5回目挑戦時 / 4回ミス)
    if (missesCount >= 4 || isFinished) {
      el.hintItem2.classList.remove("hidden");
      let relWords = Array.isArray(h.related_words) ? h.related_words : [];
      if (relWords.length === 0 && Array.isArray(state.problem.words)) {
        const target = state.problem.target || "";
        const targetChars = new Set(target.split(""));
        for (const w of state.problem.words.slice(1, 250)) {
          if (w && !w.includes(target) && ![...targetChars].some((c) => w.includes(c)) && !state.aliasSet.has(w)) {
            relWords.push(w);
            if (relWords.length >= 4) break;
          }
        }
      }
      const relHtml = relWords.slice(0, 4).map((w) => {
        const r = state.rankMap.get(w);
        const distStr = r ? ` (距離 ${r - 1})` : "";
        return `<span class="hint-tag">🔥 ${escapeHtml(w)}${distStr}</span>`;
      }).join("");
      el.hintItem2Body.innerHTML = `
        <div style="margin-bottom: 4px; font-size: 0.82rem; color: #cbd5e1;">正解のすぐ近くにある概念・言葉：</div>
        <div class="hint-tag-list">${relHtml || "<span class='hint-tag'>データなし</span>"}</div>
      `;
    } else {
      el.hintItem2.classList.add("hidden");
    }

    // ヒント3: 冒頭伏せ字抜粋 (6回目挑戦時 / 5回ミス・ラストチャンス)
    if (missesCount >= 5 || isFinished) {
      el.hintItem3.classList.remove("hidden");
      const masked =
        state.problem.mode === "text"
          ? (state.problem.text_puzzle?.sentence_3 || h.masked_summary || "（上部の第3文が開放されました）")
          : (h.masked_summary || "（データなし）");
      el.hintItem3Body.innerHTML = `<blockquote>${escapeHtml(masked)}</blockquote>`;
    } else {
      el.hintItem3.classList.add("hidden");
    }

    if (el.hintItem4) {
      el.hintItem4.classList.add("hidden");
    }
    if (el.hintItem5) {
      el.hintItem5.classList.add("hidden");
    }
  }

  /**
   * 単語順位のルックアップ (O(1))
   */
  function lookupWordRank(rawWord) {
    const norm = normalizeInput(rawWord);
    if (!norm) return null;

    // 1. 正解または別名なら1位
    if (
      norm.toLowerCase() === state.problem.target.toLowerCase() ||
      state.aliasSet.has(norm.toLowerCase()) ||
      state.aliasSet.has(toHiragana(norm)) ||
      state.aliasSet.has(toKatakana(norm))
    ) {
      return { canonicalWord: state.problem.target, rank: 1, isWin: true };
    }

    // 2. 辞書マップから直接検索
    if (state.rankMap.has(norm)) {
      return { canonicalWord: norm, rank: state.rankMap.get(norm), isWin: false };
    }

    // 3. かな変換検索
    const hira = toHiragana(norm);
    if (state.rankMap.has(hira)) {
      return { canonicalWord: hira, rank: state.rankMap.get(hira), isWin: false };
    }
    const kata = toKatakana(norm);
    if (state.rankMap.has(kata)) {
      return { canonicalWord: kata, rank: state.rankMap.get(kata), isWin: false };
    }

    return null;
  }

  /**
   * ユーザーの推測送信処理 (6回制限)
   */
  function handleGuessSubmit(rawInput) {
    if (!state.problem || state.cleared || state.failed) return;

    const clean = normalizeInput(rawInput);
    if (!clean) return;

    const result = lookupWordRank(clean);
    if (!result) {
      showToast(`「${clean}」は辞書にありません（一般的な名詞でお試しください）`, "warn");
      return;
    }

    // すでに入力済みか
    const existing = state.guesses.find((g) => g.word === result.canonicalWord);
    if (existing) {
      state.lastGuessWord = existing.word;
      renderUI();
      const existingDist = Math.max(0, existing.rank - 1);
      showToast(`「${existing.word}」はすでに入力済みです（#${existing.attemptNum}回目 / 距離: ${existingDist.toLocaleString()}）`, "info");
      return;
    }

    const attemptNum = state.guesses.length + 1;
    const { tier, emoji } = getRankVisuals(result.rank, state.problem.total_words);
    const newEntry = {
      word: result.canonicalWord,
      rank: result.rank,
      attemptNum,
      isWin: result.isWin,
      tier,
      emoji,
    };

    state.guesses.push(newEntry);
    state.lastGuessWord = newEntry.word;

    if (result.isWin) {
      state.cleared = true;
      saveProgress();
      renderUI();
      showToast(`🎉 正解！ ${attemptNum}/${MAX_TRIES} でクリア！`, "success");
      setTimeout(() => openResultModal(), 450);
    } else if (state.guesses.length >= MAX_TRIES) {
      state.failed = true;
      saveProgress();
      renderUI();
      showToast(`6回の挑戦終了！本日の正解は「${state.problem.target}」でした。`, "warn");
      setTimeout(() => openResultModal(), 500);
    } else {
      saveProgress();
      renderUI();
    }
  }

  /**
   * 降参処理
   */
  function handleGiveUp() {
    if (!state.problem || state.cleared || state.failed) return;
    const ok = window.confirm("降参して本日の正解を表示しますか？（本日の挑戦は終了となります）");
    if (!ok) return;

    state.failed = true;
    saveProgress();
    renderUI();
    openResultModal();
  }

  /**
   * 結果シェアモーダル
   */
  function openResultModal() {
    if (!state.problem) return;

    const attemptsCount = state.guesses.length;
    const minDistance = state.guesses.length > 0
      ? Math.min(...state.guesses.map((g) => Math.max(0, g.rank - 1)))
      : null;

    if (state.cleared) {
      el.modalBadgeIcon.textContent = "🎉";
      el.modalResultTitle.textContent = "正解！クリアおめでとうございます！";
      el.modalResultSubtitle.textContent = `${attemptsCount}/6 回の挑戦で見事特定しました！`;
      el.modalStatAttempts.textContent = `${attemptsCount}/6`;
    } else {
      el.modalBadgeIcon.textContent = "🏳️";
      el.modalResultTitle.textContent = "挑戦終了 — 本日の正解";
      el.modalResultSubtitle.textContent = "次回また挑戦してみましょう！";
      el.modalStatAttempts.textContent = `X/6`;
    }

    el.modalAnswerWord.textContent = state.problem.target;
    el.modalAnswerCaption.textContent =
      state.problem.image?.caption ||
      state.problem.text_puzzle?.raw_sentences?.join(" ") ||
      "";
    el.modalWikiUrl.href =
      state.problem.article_url ||
      `https://ja.wikipedia.org/wiki/${encodeURIComponent(state.problem.target)}`;

    el.modalStatBest.textContent = minDistance !== null ? `距離 ${minDistance.toLocaleString()}` : "-";
    el.modalStatMode.textContent = state.problem.mode === "text" ? "📝 文字版" : "🖼️ 画像版";

    const shareText = buildShareText();
    el.shareTextarea.value = shareText;
    el.btnXShare.href = `https://twitter.com/intent/tweet?text=${encodeURIComponent(shareText)}`;

    el.resultModal.classList.remove("hidden");
  }

  /**
   * シェア用テキストの生成 (6回制 Wordleスタイル)
   */
  function buildShareText() {
    const pid = state.problem.id;
    const modeStr = state.problem.mode === "text" ? "📝文字版" : "🖼️画像版";
    const statusLine = state.cleared
      ? `🎯 ${state.guesses.length}/${MAX_TRIES} でクリア！ (${modeStr})`
      : `🏳️ X/${MAX_TRIES}（ゲームオーバー） (${modeStr})`;

    const rows = state.guesses.map((g) => {
      const { colorEmoji, distance } = getRankVisuals(g.rank, state.problem.total_words);
      const slotNum = ["1️⃣", "2️⃣", "3️⃣", "4️⃣", "5️⃣", "6️⃣"][g.attemptNum - 1] || "▪️";
      if (g.isWin) {
        return `${slotNum} 🎯 距離 0（正解！）`;
      }
      return `${slotNum} ${colorEmoji} 距離 ${distance.toLocaleString()}`;
    });

    const pageUrl = window.location.origin + window.location.pathname;
    return [
      `🔍 WikiLens #${pid}`,
      statusLine,
      "",
      ...rows,
      "",
      `#WikiLens #Wikipedia連想ゲーム`,
      pageUrl,
    ].join("\n");
  }

  /**
   * 問題データのロード
   */
  async function loadProblem(problemId) {
    try {
      el.imageSkeleton.classList.remove("hidden");
      el.imageSkeleton.textContent = "問題を読み込み中...";
      el.targetImage.classList.add("hidden");

      const resp = await fetch(`./data/${problemId}.json`);
      if (!resp.ok) {
        throw new Error(`問題データの取得に失敗しました (${resp.status})`);
      }
      const data = await resp.json();

      state.currentProblemId = data.id || problemId;
      state.problem = data;
      state.rankMap.clear();
      state.aliasSet.clear();

      // 1. words配列から O(1) ハッシュマップ構築
      if (Array.isArray(data.words)) {
        for (let i = 0; i < data.words.length; i++) {
          const w = normalizeInput(data.words[i]);
          if (w && !state.rankMap.has(w)) {
            state.rankMap.set(w, i + 1);
          }
        }
      }

      // 2. 別名の登録
      if (Array.isArray(data.aliases)) {
        for (const a of data.aliases) {
          const na = normalizeInput(a);
          if (na) {
            state.aliasSet.add(na.toLowerCase());
            state.aliasSet.add(toHiragana(na));
            state.aliasSet.add(toKatakana(na));
          }
        }
      }

      // 3. 状態の初期化と復元
      state.guesses = [];
      state.cleared = false;
      state.failed = false;
      state.lastGuessWord = null;

      loadProgress(state.currentProblemId);

      // 4. モード別表示切り替え（画像版 vs 文字版）
      const mode = data.mode || (data.image ? "image" : "text");
      state.problem.mode = mode;

      el.problemBadge.textContent = `📅 問題: ${state.currentProblemId}`;
      el.modeBadge.textContent = mode === "text" ? "📝 文字版" : "🖼️ 画像版";
      if (el.charLenBadge) {
        const cLen = data.hints?.char_length || data.target?.length || "-";
        el.charLenBadge.textContent = `🔤 答え: ${cLen}文字`;
      }
      el.vocabCountBadge.textContent = `辞書: ${(data.total_words || state.rankMap.size).toLocaleString()}語`;

      if (mode === "image" && data.image) {
        el.imageFrame.classList.remove("hidden");
        el.textPuzzleCard.classList.add("hidden");
        el.imageCreditFooter.classList.remove("hidden");

        el.targetImage.onload = () => {
          el.imageSkeleton.classList.add("hidden");
          el.targetImage.classList.remove("hidden");
        };
        el.targetImage.onerror = () => {
          el.imageSkeleton.textContent = "画像の読み込みに失敗しました。";
        };
        el.targetImage.src = data.image.url || "";
        el.zoomImage.src = data.image.url || "";

        el.creditAuthor.textContent = data.image.author || "Wikimedia Commons Contributor";
        el.creditLicense.textContent = data.image.license || "CC BY-SA";
        el.creditLicense.href = data.image.license_url || "https://creativecommons.org/licenses/by-sa/4.0/";
        el.creditCommonsUrl.href = data.image.commons_url || "https://commons.wikimedia.org/";
        el.creditSourceDrawer.classList.add("hidden");
      } else {
        // 文字版
        el.imageFrame.classList.add("hidden");
        el.textPuzzleCard.classList.remove("hidden");
        el.imageCreditFooter.classList.add("hidden");
      }

      if (el.problemSelect.value !== state.currentProblemId) {
        el.problemSelect.value = state.currentProblemId;
      }

      renderUI();
    } catch (err) {
      console.error(err);
      el.imageSkeleton.textContent = "問題データの読み込みエラーが発生しました。";
      showToast("問題データの読み込みに失敗しました", "warn");
    }
  }

  /**
   * プルダウン更新（画像版・文字版のグループ別に整理）
   */
  function updateProblemSelectBadges() {
    if (!state.manifest || !Array.isArray(state.manifest.problems)) return;
    const currentVal = el.problemSelect.value || state.currentProblemId;
    el.problemSelect.innerHTML = "";

    const imgProblems = state.manifest.problems.filter((p) => (p.mode || "image") === "image");
    const txtProblems = state.manifest.problems.filter((p) => p.mode === "text");

    const appendGroup = (label, list, modeIcon) => {
      if (list.length === 0) return;
      const group = document.createElement("optgroup");
      group.label = label;
      const total = list.length;
      list.forEach((p, idx) => {
        const opt = document.createElement("option");
        opt.value = p.id;
        const icon = getSavedStatusIcon(p.id);
        const num = total - idx;
        opt.textContent = `${icon}${modeIcon} #${num} (${p.id})`;
        group.appendChild(opt);
      });
      el.problemSelect.appendChild(group);
    };

    appendGroup("🖼️ 画像版コース", imgProblems, "🖼️画像");
    appendGroup("📝 文字版コース", txtProblems, "📝文字");

    if (currentVal) {
      el.problemSelect.value = currentVal;
    }
    updateModeTabsAndNextButtons();
  }

  /**
   * 初期化
   */
  async function init() {
    el.guessForm.addEventListener("submit", (e) => {
      e.preventDefault();
      const val = el.guessInput.value;
      el.guessInput.value = "";
      handleGuessSubmit(val);
    });

    document.querySelectorAll(".quick-chip").forEach((btn) => {
      btn.addEventListener("click", () => {
        const w = btn.getAttribute("data-word");
        if (w) handleGuessSubmit(w);
      });
    });

    el.sortByTime.addEventListener("click", () => {
      state.sortBy = "time";
      renderUI();
    });
    el.sortByRank.addEventListener("click", () => {
      state.sortBy = "rank";
      renderUI();
    });

    el.btnGiveup.addEventListener("click", handleGiveUp);
    el.btnResultModal.addEventListener("click", openResultModal);
    if (el.btnNextPill) el.btnNextPill.addEventListener("click", () => handleNextProblem(null));
    if (el.tabModeImage) el.tabModeImage.addEventListener("click", () => handleSwitchModeTab("image"));
    if (el.tabModeText) el.tabModeText.addEventListener("click", () => handleSwitchModeTab("text"));
    if (el.btnNextImageInline) el.btnNextImageInline.addEventListener("click", () => handleNextProblem("image"));
    if (el.btnNextTextInline) el.btnNextTextInline.addEventListener("click", () => handleNextProblem("text"));
    if (el.btnNextImageModal) el.btnNextImageModal.addEventListener("click", () => handleNextProblem("image"));
    if (el.btnNextTextModal) el.btnNextTextModal.addEventListener("click", () => handleNextProblem("text"));

    el.btnToggleSource.addEventListener("click", () => {
      el.creditSourceDrawer.classList.toggle("hidden");
    });

    el.btnCloseResult.addEventListener("click", () => {
      el.resultModal.classList.add("hidden");
    });
    el.resultModal.addEventListener("click", (e) => {
      if (e.target === el.resultModal) el.resultModal.classList.add("hidden");
    });

    el.btnHelp.addEventListener("click", () => {
      el.helpModal.classList.remove("hidden");
    });
    el.btnCloseHelp.addEventListener("click", () => {
      el.helpModal.classList.add("hidden");
    });
    el.helpModal.addEventListener("click", (e) => {
      if (e.target === el.helpModal) el.helpModal.classList.add("hidden");
    });

    const openZoom = () => {
      if (state.problem?.image?.url) {
        el.zoomModal.classList.remove("hidden");
      }
    };
    el.targetImage.addEventListener("click", openZoom);
    el.btnZoom.addEventListener("click", openZoom);
    el.btnCloseZoom.addEventListener("click", () => {
      el.zoomModal.classList.add("hidden");
    });
    el.zoomModal.addEventListener("click", (e) => {
      if (e.target === el.zoomModal) el.zoomModal.classList.add("hidden");
    });

    el.btnCopyShare.addEventListener("click", async () => {
      try {
        await navigator.clipboard.writeText(el.shareTextarea.value);
        showToast("📋 結果をコピーしました！", "success");
      } catch (_) {
        el.shareTextarea.select();
        document.execCommand("copy");
        showToast("📋 コピーしました！", "success");
      }
    });

    el.problemSelect.addEventListener("change", (e) => {
      const pid = e.target.value;
      if (pid) loadProblem(pid);
    });

    // マニフェスト読み込み
    try {
      const resp = await fetch("./data/manifest.json");
      if (!resp.ok) throw new Error("manifest not found");
      const manifest = await resp.json();
      state.manifest = manifest;
      updateProblemSelectBadges();

      const jstToday = getJSTToday();
      const hasToday = manifest.problems?.some((p) => p.id === jstToday);
      const initialId = hasToday ? jstToday : manifest.latest || manifest.problems?.[0]?.id;
      if (initialId) {
        await loadProblem(initialId);
      }
    } catch (e) {
      console.error("Manifest load failed:", e);
      await loadProblem("2026-10-23");
    }
  }

  document.addEventListener("DOMContentLoaded", init);
})();
