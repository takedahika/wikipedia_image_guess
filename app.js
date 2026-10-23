/**
 * WikiLens - Wikipedia画像連想ゲーム
 * クライアントサイドロジック (Vanilla JS / O(1) ハッシュマップ検索 / LocalStorage永続化)
 */

(() => {
  "use strict";

  const STORAGE_PREFIX = "wikilens_state_v1_";

  // アプリケーション状態
  const state = {
    manifest: null,
    currentProblemId: null,
    problem: null,
    rankMap: new Map(),      // 正規化単語 -> 順位 (1〜N)
    aliasSet: new Set(),     // 別名（1位扱い）のセット
    guesses: [],             // [{ word, rank, attemptNum, isHint }]
    hintsUnlocked: {
      1: false,
      2: false,
      3: false,
    },
    wordHintsUsed: 0,
    cleared: false,
    gaveUp: false,
    sortBy: "rank",          // "rank" | "time"
    lastGuessWord: null,
  };

  // DOM要素参照
  const el = {
    problemSelect: document.getElementById("problem-select"),
    problemBadge: document.getElementById("problem-badge"),
    vocabCountBadge: document.getElementById("vocab-count-badge"),
    imageSkeleton: document.getElementById("image-skeleton"),
    targetImage: document.getElementById("target-image"),
    btnZoom: document.getElementById("btn-zoom"),
    revealedCaptionBox: document.getElementById("revealed-caption-box"),
    revealedTargetName: document.getElementById("revealed-target-name"),
    revealedWikiLink: document.getElementById("revealed-wiki-link"),
    revealedCaptionText: document.getElementById("revealed-caption-text"),
    creditAuthor: document.getElementById("credit-author"),
    creditLicense: document.getElementById("credit-license"),
    btnToggleSource: document.getElementById("btn-toggle-source"),
    creditSourceDrawer: document.getElementById("credit-source-drawer"),
    creditCommonsUrl: document.getElementById("credit-commons-url"),
    statAttempts: document.getElementById("stat-attempts"),
    statBestRank: document.getElementById("stat-best-rank"),
    statHints: document.getElementById("stat-hints"),
    btnOpenHints: document.getElementById("btn-open-hints"),
    btnResultModal: document.getElementById("btn-result-modal"),
    btnGiveup: document.getElementById("btn-giveup"),
    guessForm: document.getElementById("guess-form"),
    guessInput: document.getElementById("guess-input"),
    btnSubmit: document.getElementById("btn-submit"),
    latestGuessContainer: document.getElementById("latest-guess-container"),
    latestGuessCard: document.getElementById("latest-guess-card"),
    hintsPanel: document.getElementById("hints-panel"),
    btnCloseHints: document.getElementById("btn-close-hints"),
    btnUnlockHint1: document.getElementById("btn-unlock-hint1"),
    btnUnlockHint2: document.getElementById("btn-unlock-hint2"),
    btnUnlockHint3: document.getElementById("btn-unlock-hint3"),
    hint1Content: document.getElementById("hint1-content"),
    hint2Content: document.getElementById("hint2-content"),
    hint3Content: document.getElementById("hint3-content"),
    btnWordHint: document.getElementById("btn-word-hint"),
    historyCount: document.getElementById("history-count"),
    sortByRank: document.getElementById("sort-by-rank"),
    sortByTime: document.getElementById("sort-by-time"),
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
    modalStatHints: document.getElementById("modal-stat-hints"),
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
   * 入力文字列のサニタイズ・正規化 (F-02)
   */
  function normalizeInput(raw) {
    if (!raw) return "";
    return raw
      .normalize("NFKC")
      .trim()
      .replace(/[\s　]+/g, "");
  }

  /**
   * カタカナ ⇔ ひらがな変換（表記ゆれ吸収用）
   */
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

  /**
   * トースト通知を表示
   */
  function showToast(message, type = "info") {
    const toast = document.createElement("div");
    toast.className = `toast ${type}`;
    toast.textContent = message;
    el.toastContainer.appendChild(toast);
    setTimeout(() => {
      toast.style.opacity = "0";
      toast.style.transition = "opacity 0.25s";
      setTimeout(() => toast.remove(), 260);
    }, 2600);
  }

  /**
   * JST（またはローカルタイムゾーン）の YYYY-MM-DD を取得
   */
  function getTodayDateString() {
    const now = new Date();
    const y = now.getFullYear();
    const m = String(now.getMonth() + 1).padStart(2, "0");
    const d = String(now.getDate()).padStart(2, "0");
    return `${y}-${m}-${d}`;
  }

  /**
   * LocalStorageへの状態保存と復元 (F-06)
   */
  function saveProgress() {
    if (!state.currentProblemId) return;
    const payload = {
      id: state.currentProblemId,
      guesses: state.guesses,
      hintsUnlocked: state.hintsUnlocked,
      wordHintsUsed: state.wordHintsUsed,
      cleared: state.cleared,
      gaveUp: state.gaveUp,
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
        state.hintsUnlocked = saved.hintsUnlocked || { 1: false, 2: false, 3: false };
        state.wordHintsUsed = Number(saved.wordHintsUsed || 0);
        state.cleared = Boolean(saved.cleared);
        state.gaveUp = Boolean(saved.gaveUp);
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
      if (saved.gaveUp) return "🏳️ ";
      if (saved.guesses && saved.guesses.length > 0) return "⏳ ";
    } catch (_) {}
    return "";
  }

  /**
   * 順位に応じた色ティアとプログレスバー幅を算出
   */
  function getRankVisuals(rank, totalWords) {
    const total = totalWords || 20000;
    if (rank === 1) {
      return { tier: "tier-top", widthPct: 100, emoji: "🎯" };
    }
    if (rank <= 100) {
      const pct = Math.max(82, 99 - (rank / 100) * 17);
      return { tier: "tier-hot", widthPct: pct, emoji: "🔥" };
    }
    if (rank <= 500) {
      const pct = Math.max(62, 82 - ((rank - 100) / 400) * 20);
      return { tier: "tier-warm", widthPct: pct, emoji: "✨" };
    }
    if (rank <= 2000) {
      const pct = Math.max(38, 62 - ((rank - 500) / 1500) * 24);
      return { tier: "tier-mid", widthPct: pct, emoji: "💡" };
    }
    if (rank <= 5000) {
      const pct = Math.max(18, 38 - ((rank - 2000) / 3000) * 20);
      return { tier: "tier-cool", widthPct: pct, emoji: "🟠" };
    }
    const pct = Math.max(5, 18 - ((rank - 5000) / Math.max(1, total - 5000)) * 13);
    return { tier: "tier-cold", widthPct: pct, emoji: "❄️" };
  }

  /**
   * 推測結果の1行DOM要素を生成
   */
  function createGuessRowElement(item, highlight = false) {
    const total = state.problem?.total_words || 20000;
    const { tier, widthPct, emoji } = getRankVisuals(item.rank, total);

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

    if (item.isHint) {
      const tag = document.createElement("span");
      tag.className = "guess-hint-tag";
      tag.textContent = "ヒント";
      left.appendChild(tag);
    }

    const right = document.createElement("div");
    right.className = "guess-row-right";

    const rankBadge = document.createElement("span");
    rankBadge.className = "guess-rank-badge";
    rankBadge.textContent = item.rank === 1 ? `1位 ${emoji} 正解！` : `${item.rank.toLocaleString()}位 ${emoji}`;

    right.appendChild(rankBadge);

    row.appendChild(bar);
    row.appendChild(left);
    row.appendChild(right);
    return row;
  }

  /**
   * 画面全体の再描画
   */
  function renderUI() {
    if (!state.problem) return;

    const totalGuesses = state.guesses.filter((g) => !g.isHint).length;
    const bestRank = state.guesses.length > 0
      ? Math.min(...state.guesses.map((g) => g.rank))
      : null;
    const totalHintsUsed =
      Object.values(state.hintsUnlocked).filter(Boolean).length + state.wordHintsUsed;

    el.statAttempts.textContent = String(totalGuesses);
    el.statBestRank.textContent = bestRank ? `${bestRank.toLocaleString()}位` : "-";
    el.statHints.textContent = `${totalHintsUsed}回`;

    // クリアまたはギブアップ状態の制御
    const isFinished = state.cleared || state.gaveUp;
    el.guessInput.disabled = isFinished;
    el.btnSubmit.disabled = isFinished;
    el.btnGiveup.classList.toggle("hidden", isFinished);
    el.btnResultModal.classList.toggle("hidden", !isFinished);

    if (isFinished) {
      el.guessInput.placeholder = state.cleared
        ? `🎉 正解「${state.problem.target}」をクリア済みです！`
        : `答え: 「${state.problem.target}」`;
      el.revealedCaptionBox.classList.remove("hidden");
      el.revealedTargetName.textContent = state.problem.target;
      el.revealedWikiLink.href = state.problem.article_url || `https://ja.wikipedia.org/wiki/${encodeURIComponent(state.problem.target)}`;
      el.revealedCaptionText.textContent = state.problem.image?.caption || "";
      // クリア後は出典ドロワーのネタバレ注意も不要なので開いても安全
    } else {
      el.guessInput.placeholder = "画像から連想する単語を入力（例: 建物、歴史、自然…）";
      el.revealedCaptionBox.classList.add("hidden");
    }

    // 直近の入力ハイライト表示
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

    // 履歴一覧の描画
    el.historyCount.textContent = String(state.guesses.length);
    el.emptyHistory.classList.toggle("hidden", state.guesses.length > 0);
    el.historyList.innerHTML = "";

    const sorted = [...state.guesses];
    if (state.sortBy === "rank") {
      sorted.sort((a, b) => a.rank - b.rank || b.attemptNum - a.attemptNum);
    } else {
      sorted.sort((a, b) => b.attemptNum - a.attemptNum);
    }

    const fragment = document.createDocumentFragment();
    for (const item of sorted) {
      const isLatest = item.word === state.lastGuessWord;
      fragment.appendChild(createGuessRowElement(item, isLatest));
    }
    el.historyList.appendChild(fragment);

    // ソートボタンのアクティブ状態
    el.sortByRank.classList.toggle("active", state.sortBy === "rank");
    el.sortByTime.classList.toggle("active", state.sortBy === "time");

    // ヒント描画
    renderHints();
  }

  /**
   * 段階的ヒントパネルの描画 (F-04)
   */
  function renderHints() {
    if (!state.problem) return;
    const h = state.problem.hints || {};

    // ヒント1: カテゴリ・記事文字数
    if (state.hintsUnlocked[1]) {
      el.btnUnlockHint1.textContent = "開放済み";
      el.btnUnlockHint1.disabled = true;
      el.hint1Content.classList.remove("hidden");
      const cats = Array.isArray(h.category) && h.category.length > 0
        ? h.category.map((c) => `<span class="hint-tag">${escapeHtml(c)}</span>`).join("")
        : `<span class="hint-tag">一般記事</span>`;
      const lenStr = h.summary_length ? `約 ${Number(h.summary_length).toLocaleString()} バイト` : "標準規模";
      el.hint1Content.innerHTML = `
        <div>📊 <strong>記事規模:</strong> ${lenStr}</div>
        <div style="margin-top:4px;">🏷️ <strong>所属カテゴリ:</strong></div>
        <div class="hint-tag-list">${cats}</div>
      `;
    } else {
      el.btnUnlockHint1.textContent = "開放する";
      el.btnUnlockHint1.disabled = false;
      el.hint1Content.classList.add("hidden");
    }

    // ヒント2: セクション目次・文字数・頭文字
    if (state.hintsUnlocked[2]) {
      el.btnUnlockHint2.textContent = "開放済み";
      el.btnUnlockHint2.disabled = true;
      el.hint2Content.classList.remove("hidden");
      const secs = Array.isArray(h.sections) && h.sections.length > 0
        ? h.sections.map((s) => `<span class="hint-tag">§ ${escapeHtml(s)}</span>`).join("")
        : `<span class="hint-tag">§ 概要・歴史</span>`;
      const charLen = h.char_length || state.problem.target.length;
      const firstChar = h.first_char || state.problem.target[0];
      el.hint2Content.innerHTML = `
        <div>🔤 <strong>記事名の文字数:</strong> ${charLen} 文字（頭文字: 「<strong>${escapeHtml(firstChar)}</strong>」）</div>
        <div style="margin-top:4px;">📑 <strong>記事の目次セクション:</strong></div>
        <div class="hint-tag-list">${secs}</div>
      `;
    } else {
      el.btnUnlockHint2.textContent = "開放する";
      el.btnUnlockHint2.disabled = false;
      el.hint2Content.classList.add("hidden");
    }

    // ヒント3: 冒頭概要の伏せ字
    if (state.hintsUnlocked[3]) {
      el.btnUnlockHint3.textContent = "開放済み";
      el.btnUnlockHint3.disabled = true;
      el.hint3Content.classList.remove("hidden");
      const masked = h.masked_summary || "（伏せ字概要データなし）";
      el.hint3Content.innerHTML = `
        <div>📝 <strong>導入部抜粋:</strong><br/>${escapeHtml(masked)}</div>
      `;
    } else {
      el.btnUnlockHint3.textContent = "開放する";
      el.btnUnlockHint3.disabled = false;
      el.hint3Content.classList.add("hidden");
    }

    // 単語ヒントボタンの制御
    el.btnWordHint.disabled = state.cleared || state.gaveUp;
  }

  function escapeHtml(str) {
    return String(str)
      .replace(/&/g, "&amp;")
      .replace(/</g, "&lt;")
      .replace(/>/g, "&gt;")
      .replace(/"/g, "&quot;");
  }

  /**
   * 単語の順位を検索する（表記ゆれ・別名対応）
   */
  function lookupWordRank(rawWord) {
    const norm = normalizeInput(rawWord);
    if (!norm) return null;

    // 1. 正解または別名（aliases）に一致すれば1位
    if (
      norm.toLowerCase() === state.problem.target.toLowerCase() ||
      state.aliasSet.has(norm.toLowerCase()) ||
      state.aliasSet.has(toHiragana(norm)) ||
      state.aliasSet.has(toKatakana(norm))
    ) {
      return { canonicalWord: state.problem.target, rank: 1 };
    }

    // 2. 辞書マップから直接検索 O(1)
    if (state.rankMap.has(norm)) {
      return { canonicalWord: norm, rank: state.rankMap.get(norm) };
    }

    // 3. カタカナ・ひらがな変換で検索
    const hira = toHiragana(norm);
    if (state.rankMap.has(hira)) {
      return { canonicalWord: hira, rank: state.rankMap.get(hira) };
    }
    const kata = toKatakana(norm);
    if (state.rankMap.has(kata)) {
      return { canonicalWord: kata, rank: state.rankMap.get(kata) };
    }

    return null;
  }

  /**
   * ユーザーの単語入力送信処理 (F-02)
   */
  function handleGuessSubmit(rawInput, isHintWord = false) {
    if (!state.problem || state.cleared || state.gaveUp) return;

    const clean = normalizeInput(rawInput);
    if (!clean) return;

    const result = lookupWordRank(clean);
    if (!result) {
      showToast(`「${clean}」は辞書にありません（一般的な名詞でお試しください）`, "warn");
      return;
    }

    // 既に入力済みかチェック
    const existing = state.guesses.find((g) => g.word === result.canonicalWord);
    if (existing) {
      state.lastGuessWord = existing.word;
      renderUI();
      showToast(`「${existing.word}」はすでに入力済みです（${existing.rank.toLocaleString()}位）`, "info");
      return;
    }

    const attemptNum = state.guesses.length + 1;
    const newEntry = {
      word: result.canonicalWord,
      rank: result.rank,
      attemptNum,
      isHint: isHintWord,
    };

    state.guesses.push(newEntry);
    state.lastGuessWord = newEntry.word;

    if (result.rank === 1) {
      state.cleared = true;
      saveProgress();
      renderUI();
      showToast("🎉 正解！1位を的中させました！", "success");
      setTimeout(() => openResultModal(), 450);
    } else {
      saveProgress();
      renderUI();
    }
  }

  /**
   * お助け単語ヒント（現在の最高順位より近い単語を1つ開示）
   */
  function handleWordHint() {
    if (!state.problem || !Array.isArray(state.problem.words) || state.cleared || state.gaveUp) {
      return;
    }

    const guessedWords = new Set(state.guesses.map((g) => g.word));
    const bestRank = state.guesses.length > 0
      ? Math.min(...state.guesses.map((g) => g.rank))
      : 20000;

    // 目標とするヒント順位を決定
    let targetRank = 300;
    if (bestRank <= 15) {
      targetRank = Math.max(2, bestRank - 2);
    } else if (bestRank <= 40) {
      targetRank = 12;
    } else if (bestRank <= 120) {
      targetRank = 35;
    } else if (bestRank <= 350) {
      targetRank = 100;
    }

    // targetRank 付近でまだ入力されていない単語を探す（1位は除外）
    const words = state.problem.words;
    let chosenWord = null;
    for (let r = targetRank; r >= 2; r--) {
      const candidate = words[r - 1];
      if (candidate && !guessedWords.has(candidate)) {
        chosenWord = candidate;
        break;
      }
    }
    if (!chosenWord) {
      for (let r = targetRank + 1; r <= Math.min(words.length, 500); r++) {
        const candidate = words[r - 1];
        if (candidate && !guessedWords.has(candidate)) {
          chosenWord = candidate;
          break;
        }
      }
    }

    if (!chosenWord) {
      showToast("これ以上開示できるヒント単語がありません", "warn");
      return;
    }

    state.wordHintsUsed += 1;
    handleGuessSubmit(chosenWord, true);
    showToast(`💡 ヒント単語「${chosenWord}」を追加しました`, "info");
  }

  /**
   * ギブアップ（降参）処理
   */
  function handleGiveUp() {
    if (!state.problem || state.cleared || state.gaveUp) return;
    const ok = window.confirm("本当に降参して正解を表示しますか？");
    if (!ok) return;

    state.gaveUp = true;
    saveProgress();
    renderUI();
    openResultModal();
  }

  /**
   * クリア・結果シェアモーダルを開く (F-05)
   */
  function openResultModal() {
    if (!state.problem) return;

    const normalAttempts = state.guesses.filter((g) => !g.isHint).length;
    const bestRank = state.guesses.length > 0
      ? Math.min(...state.guesses.map((g) => g.rank))
      : null;
    const totalHints =
      Object.values(state.hintsUnlocked).filter(Boolean).length + state.wordHintsUsed;

    if (state.cleared) {
      el.modalBadgeIcon.textContent = "🎉";
      el.modalResultTitle.textContent = "正解！クリアおめでとうございます！";
      el.modalResultSubtitle.textContent = `${normalAttempts}回の推測でWikipedia記事を特定しました`;
    } else {
      el.modalBadgeIcon.textContent = "🏳️";
      el.modalResultTitle.textContent = "ギブアップ — 本日の正解";
      el.modalResultSubtitle.textContent = "次回また挑戦してみましょう！";
    }

    el.modalAnswerWord.textContent = state.problem.target;
    el.modalAnswerCaption.textContent = state.problem.image?.caption || "";
    el.modalWikiUrl.href =
      state.problem.article_url ||
      `https://ja.wikipedia.org/wiki/${encodeURIComponent(state.problem.target)}`;

    el.modalStatAttempts.textContent = `${normalAttempts}回`;
    el.modalStatBest.textContent = bestRank ? `${bestRank.toLocaleString()}位` : "-";
    el.modalStatHints.textContent = `${totalHints}回`;

    const shareText = buildShareText(normalAttempts, totalHints);
    el.shareTextarea.value = shareText;
    el.btnXShare.href = `https://twitter.com/intent/tweet?text=${encodeURIComponent(shareText)}`;

    el.resultModal.classList.remove("hidden");
  }

  /**
   * SNSシェア用テキストの生成 (F-05)
   */
  function buildShareText(attempts, hintsUsed) {
    const pid = state.problem.id;
    const statusLine = state.cleared
      ? `🎯 正解！（試行: ${attempts}回 / ヒント: ${hintsUsed}回）`
      : `🏳️ ギブアップ（試行: ${attempts}回 / ヒント: ${hintsUsed}回）`;

    // 推測履歴の絵文字推移（最大10個に要約）
    const emojis = state.guesses.map((g) => {
      if (g.rank === 1) return "🎯";
      if (g.rank <= 100) return "🟩";
      if (g.rank <= 500) return "🟦";
      if (g.rank <= 2000) return "🟨";
      if (g.rank <= 5000) return "🟧";
      return "⬜";
    });
    const trail = emojis.length <= 12
      ? emojis.join("")
      : emojis.slice(0, 5).join("") + "…" + emojis.slice(-6).join("");

    const pageUrl = window.location.origin + window.location.pathname;
    return [
      `🔍 WikiLens #${pid}`,
      statusLine,
      trail || "⬜",
      `#WikiLens #Wikipedia画像連想ゲーム`,
      pageUrl,
    ].join("\n");
  }

  /**
   * 問題データの読み込みと初期化
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

      // 1. words配列 または ranksオブジェクト から O(1) ハッシュマップを構築
      if (Array.isArray(data.words)) {
        for (let i = 0; i < data.words.length; i++) {
          const w = normalizeInput(data.words[i]);
          if (w && !state.rankMap.has(w)) {
            state.rankMap.set(w, i + 1);
          }
        }
      } else if (data.ranks && typeof data.ranks === "object") {
        for (const [w, r] of Object.entries(data.ranks)) {
          state.rankMap.set(normalizeInput(w), Number(r));
        }
      }

      // 2. 別名（aliases）の登録
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

      // 3. 状態のリセット＆LocalStorage復元
      state.guesses = [];
      state.hintsUnlocked = { 1: false, 2: false, 3: false };
      state.wordHintsUsed = 0;
      state.cleared = false;
      state.gaveUp = false;
      state.lastGuessWord = null;

      loadProgress(state.currentProblemId);

      // 4. 画像とクレジットの反映 (F-01 / 5.3)
      const totalWords = data.total_words || state.rankMap.size;
      el.problemBadge.textContent = `📅 問題: ${state.currentProblemId}`;
      el.vocabCountBadge.textContent = `辞書: ${totalWords.toLocaleString()}語`;

      const imgObj = data.image || {};
      el.targetImage.onload = () => {
        el.imageSkeleton.classList.add("hidden");
        el.targetImage.classList.remove("hidden");
      };
      el.targetImage.onerror = () => {
        el.imageSkeleton.textContent = "画像の読み込みに失敗しました。";
      };
      el.targetImage.src = imgObj.url || "";
      el.zoomImage.src = imgObj.url || "";

      el.creditAuthor.textContent = imgObj.author || "Wikimedia Commons Contributor";
      el.creditLicense.textContent = imgObj.license || "CC BY-SA";
      el.creditLicense.href = imgObj.license_url || "https://creativecommons.org/licenses/by-sa/4.0/";
      el.creditCommonsUrl.href = imgObj.commons_url || "https://commons.wikimedia.org/";
      el.creditSourceDrawer.classList.add("hidden");

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
   * 問題選択プルダウンのバッジ更新
   */
  function updateProblemSelectBadges() {
    if (!state.manifest || !Array.isArray(state.manifest.problems)) return;
    const currentVal = el.problemSelect.value || state.currentProblemId;
    el.problemSelect.innerHTML = "";
    const total = state.manifest.problems.length;
    state.manifest.problems.forEach((p, idx) => {
      const opt = document.createElement("option");
      opt.value = p.id;
      const icon = getSavedStatusIcon(p.id);
      const num = total - idx;
      opt.textContent = `${icon}#${num} (${p.id})`;
      el.problemSelect.appendChild(opt);
    });
    if (currentVal) {
      el.problemSelect.value = currentVal;
    }
  }

  /**
   * 初期化処理
   */
  async function init() {
    // イベントリスナー登録
    el.guessForm.addEventListener("submit", (e) => {
      e.preventDefault();
      const val = el.guessInput.value;
      el.guessInput.value = "";
      handleGuessSubmit(val, false);
    });

    // クイックお試しチップ
    document.querySelectorAll(".quick-chip").forEach((btn) => {
      btn.addEventListener("click", () => {
        const w = btn.getAttribute("data-word");
        if (w) handleGuessSubmit(w, false);
      });
    });

    // ソート切り替え (F-03)
    el.sortByRank.addEventListener("click", () => {
      state.sortBy = "rank";
      renderUI();
    });
    el.sortByTime.addEventListener("click", () => {
      state.sortBy = "time";
      renderUI();
    });

    // ヒントパネル開閉と段階開放 (F-04)
    el.btnOpenHints.addEventListener("click", () => {
      el.hintsPanel.classList.toggle("hidden");
    });
    el.btnCloseHints.addEventListener("click", () => {
      el.hintsPanel.classList.add("hidden");
    });

    [el.btnUnlockHint1, el.btnUnlockHint2, el.btnUnlockHint3].forEach((btn) => {
      btn.addEventListener("click", () => {
        const step = Number(btn.getAttribute("data-hint"));
        if (step && !state.hintsUnlocked[step]) {
          state.hintsUnlocked[step] = true;
          saveProgress();
          renderUI();
        }
      });
    });

    el.btnWordHint.addEventListener("click", handleWordHint);
    el.btnGiveup.addEventListener("click", handleGiveUp);
    el.btnResultModal.addEventListener("click", openResultModal);

    // 出典詳細トグル
    el.btnToggleSource.addEventListener("click", () => {
      el.creditSourceDrawer.classList.toggle("hidden");
    });

    // モーダル制御
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

    // 画像拡大ズーム
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

    // シェアテキストのクリップボードコピー
    el.btnCopyShare.addEventListener("click", async () => {
      try {
        await navigator.clipboard.writeText(el.shareTextarea.value);
        showToast("📋 クリップボードにコピーしました！", "success");
      } catch (_) {
        el.shareTextarea.select();
        document.execCommand("copy");
        showToast("📋 コピーしました！", "success");
      }
    });

    // 問題切り替えプルダウン
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

      const today = getTodayDateString();
      const hasToday = manifest.problems?.some((p) => p.id === today);
      const initialId = hasToday ? today : manifest.latest || manifest.problems?.[0]?.id;
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
