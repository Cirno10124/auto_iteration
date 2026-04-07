(() => {
  const state = {
    tasks: [],
    currentIndex: 0,
    originalPseudoLabel: "",
  };

  const statusFilterEl = document.getElementById("statusFilter");
  const sortModeEl = document.getElementById("sortMode");
  const taskListEl = document.getElementById("taskList");
  const progressTextEl = document.getElementById("progressText");
  const progressBarInnerEl = document.getElementById("progressBarInner");
  const currentTaskInfoEl = document.getElementById("currentTaskInfo");
  const currentMetaInfoEl = document.getElementById("currentMetaInfo");
  const metaDetailsEl = document.getElementById("metaDetails");
  const statsListEl = document.getElementById("statsList");
  const audioEl = document.getElementById("audioPlayer");
  const playPauseBtn = document.getElementById("playPauseBtn");
  const backwardBtn = document.getElementById("backwardBtn");
  const forwardBtn = document.getElementById("forwardBtn");
  const playbackRateSelect = document.getElementById("playbackRateSelect");
  const volumeSlider = document.getElementById("volumeSlider");
  const volumeValueEl = document.getElementById("volumeValue");
  const timeInfoEl = document.getElementById("timeInfo");
  const labelEditorEl = document.getElementById("labelEditor");
  const resetTextBtn = document.getElementById("resetTextBtn");
  const acceptBtn = document.getElementById("acceptBtn");
  const submitNextBtn = document.getElementById("submitNextBtn");
  const skipBtn = document.getElementById("skipBtn");
  const discardBtn = document.getElementById("discardBtn");
  const shortcutHelpBtn = document.getElementById("shortcutHelpBtn");
  const exportBackfillBtn = document.getElementById("exportBackfillBtn");
  const shortcutDialog = document.getElementById("shortcutDialog");
  const closeShortcutDialog = document.getElementById("closeShortcutDialog");
  const issueTagSelect = document.getElementById("issueTagSelect");
  const labelLevelInputs = document.querySelectorAll("input[name='labelLevel']");

  const jumpSeconds = 3;

  let audioCtx = null;
  let gainNode = null;

  function initAudioGain() {
    if (gainNode) return;
    try {
      audioCtx = new (window.AudioContext || window.webkitAudioContext)();
      const source = audioCtx.createMediaElementSource(audioEl);
      gainNode = audioCtx.createGain();
      source.connect(gainNode);
      gainNode.connect(audioCtx.destination);
      applyVolume();
    } catch (e) {
      console.warn("Web Audio 不可用，使用原生音量", e);
    }
  }

  function applyVolume() {
    const pct = volumeSlider ? Math.min(500, Math.max(50, Number(volumeSlider.value) || 100)) : 100;
    if (volumeValueEl) volumeValueEl.textContent = pct + "%";
    if (gainNode) gainNode.gain.value = pct / 100;
    else if (audioEl) audioEl.volume = Math.min(1, pct / 100);
  }

  function fmtTime(sec) {
    if (!isFinite(sec)) return "00:00";
    const m = Math.floor(sec / 60);
    const s = Math.floor(sec % 60);
    return `${String(m).padStart(2, "0")}:${String(s).padStart(2, "0")}`;
  }

  function updateTimeInfo() {
    timeInfoEl.textContent = `${fmtTime(audioEl.currentTime)} / ${fmtTime(
      audioEl.duration
    )}`;
  }

  function getSelectedLabelLevel() {
    for (const input of labelLevelInputs) {
      if (input.checked) return input.value;
    }
    return "B";
  }

  function setSelectedLabelLevel(level) {
    labelLevelInputs.forEach((input) => {
      input.checked = input.value === level;
    });
  }

  function computeStats(tasks) {
    const total = tasks.length;
    let labeled = 0;
    let skipped = 0;
    let removed = 0;
    tasks.forEach((t) => {
      if (t.status === "labeled") labeled += 1;
      else if (t.status === "skipped") skipped += 1;
      else if (t.status === "removed") removed += 1;
    });
    const pending = total - labeled - skipped - removed;
    return { total, labeled, skipped, removed, pending };
  }

  function applySort(list) {
    const mode = sortModeEl.value;
    if (mode === "priority") {
      return [...list].sort((a, b) => (b.priority || 0) - (a.priority || 0));
    }
    if (mode === "confidenceLow") {
      return [...list].sort(
        (a, b) => (a.confidence ?? 1) - (b.confidence ?? 1)
      );
    }
    if (mode === "confidenceHigh") {
      return [...list].sort(
        (a, b) => (b.confidence ?? 0) - (a.confidence ?? 0)
      );
    }
    return list;
  }

  function renderTasks() {
    const { tasks, currentIndex } = state;
    const stats = computeStats(tasks);
    progressTextEl.textContent = `第 ${tasks.length ? currentIndex + 1 : 0} / ${
      stats.total
    } 条 · 已标注 ${stats.labeled} · 跳过 ${stats.skipped} · 已剔除 ${stats.removed}`;
    const ratio = stats.total
      ? (stats.labeled + stats.skipped + stats.removed) / stats.total
      : 0;
    progressBarInnerEl.style.width = `${Math.round(ratio * 100)}%`;

    statsListEl.innerHTML = `
      <li><strong>总数：</strong>${stats.total}</li>
      <li><strong>已标注：</strong>${stats.labeled}</li>
      <li><strong>跳过：</strong>${stats.skipped}</li>
      <li><strong>已剔除：</strong>${stats.removed}</li>
      <li><strong>待标注：</strong>${stats.pending}</li>
    `;

    const statusFilter = statusFilterEl.value;
    const filtered = tasks.filter((t) =>
      statusFilter ? t.status === statusFilter : true
    );
    const sorted = applySort(filtered);

    taskListEl.innerHTML = "";
    sorted.forEach((task) => {
      const li = document.createElement("li");
      li.className = "task-item";
      if (tasks[currentIndex] && task.id === tasks[currentIndex].id) {
        li.classList.add("active");
      }
      li.dataset.taskId = task.id;
      li.innerHTML = `
        <div class="task-main-line">
          <span class="task-id">${task.id}</span>
          <span class="task-status ${task.status}">${
        task.status === "labeled"
          ? "已标注"
          : task.status === "skipped"
          ? "已跳过"
          : task.status === "removed"
          ? "已剔除"
          : "待标注"
      }</span>
        </div>
        <div class="task-secondary">
          <span>${task.pseudoLabel ? task.pseudoLabel.slice(0, 14) + "…" : ""}</span>
          <span>${typeof task.confidence === "number" ? `置信度 ${(
        task.confidence * 100
      ).toFixed(1)}%` : ""}</span>
        </div>
      `;
      li.addEventListener("click", () => {
        const index = tasks.findIndex((t) => t.id === task.id);
        if (index >= 0) {
          loadTask(index);
        }
      });
      taskListEl.appendChild(li);
    });
  }

  function loadTask(index) {
    const { tasks } = state;
    if (!tasks.length) return;
    const clamped = Math.max(0, Math.min(index, tasks.length - 1));
    state.currentIndex = clamped;
    const task = tasks[clamped];

    state.originalPseudoLabel = task.pseudoLabel || "";
    labelEditorEl.value =
      (task.annotation && task.annotation.finalText) || task.pseudoLabel || "";

    currentTaskInfoEl.textContent = `${task.id} · ${
      task.status === "labeled"
        ? "已标注"
        : task.status === "skipped"
        ? "已跳过"
        : task.status === "removed"
        ? "已剔除"
        : "待标注"
    }`;
    currentMetaInfoEl.textContent = [
      task.audioPath,
      typeof task.confidence === "number"
        ? `置信度 ${(task.confidence * 100).toFixed(1)}%`
        : null,
    ]
      .filter(Boolean)
      .join(" · ");

    metaDetailsEl.innerHTML = `
      <div>批次：${task.batchId || "—"}</div>
      <div>优先级：${task.priority ?? "—"}</div>
      <div>时长：${
        task.meta && task.meta.durationSec
          ? `${task.meta.durationSec.toFixed(1)}s`
          : "—"
      }</div>
    `;

    const audioUrl = task.id ? `/api/audio/${encodeURIComponent(task.id)}` : "";
    if (audioEl.src !== audioUrl) {
      audioEl.src = audioUrl;
      audioEl.currentTime = 0;
      audioEl.pause();
    }

    if (task.annotation && task.annotation.labelLevel) {
      setSelectedLabelLevel(task.annotation.labelLevel);
    } else {
      setSelectedLabelLevel("B");
    }
    issueTagSelect.value = (task.annotation && task.annotation.issueTag) || "";

    if (discardBtn) {
      discardBtn.textContent =
        task.status === "removed" ? "撤销剔除" : "剔除当前（置空text）";
    }

    renderTasks();
  }

  async function fetchTasks() {
    const res = await fetch("/api/tasks");
    const data = await res.json();
    state.tasks = data.items || [];
    if (!state.tasks.length) {
      progressTextEl.textContent = "暂无任务，请导入任务数据。";
      return;
    }
    renderTasks();
    loadTask(state.currentIndex || 0);
  }

  async function submitAnnotation({ skipOnly, acceptOriginal, removedOnly }) {
    const { tasks, currentIndex, originalPseudoLabel } = state;
    const task = tasks[currentIndex];
    if (!task) return;

    const isRemoved = Boolean(removedOnly);
    const labelLevel = getSelectedLabelLevel();
    const issueTag = issueTagSelect.value || null;

    const finalText = isRemoved
      ? ""
      : skipOnly
      ? ""
      : acceptOriginal
      ? originalPseudoLabel
      : labelEditorEl.value || "";

    const payload = {
      taskId: task.id,
      finalText,
      labelLevel,
      skipped: Boolean(skipOnly || isRemoved),
      issueTag,
      removed: isRemoved,
    };

    const res = await fetch("/api/annotations", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(payload),
    });

    if (!res.ok) {
      console.error("提交失败", await res.text());
      return;
    }

    const data = await res.json();
    const newTasks = state.tasks.slice();
    newTasks[currentIndex] = {
      ...task,
      status: isRemoved ? "removed" : skipOnly ? "skipped" : "labeled",
      annotation: data.annotation,
    };
    state.tasks = newTasks;

    const nextPendingIndex = newTasks.findIndex((t, idx) => {
      if (idx <= currentIndex) return false;
      return t.status === "pending";
    });
    const nextIndex =
      nextPendingIndex >= 0
        ? nextPendingIndex
        : Math.min(currentIndex + 1, newTasks.length - 1);
    loadTask(nextIndex);
  }

  function setupEvents() {
    statusFilterEl.addEventListener("change", renderTasks);
    sortModeEl.addEventListener("change", renderTasks);

    playPauseBtn.addEventListener("click", () => {
      if (audioEl.paused) audioEl.play();
      else audioEl.pause();
    });
    backwardBtn.addEventListener("click", () => {
      audioEl.currentTime = Math.max(0, audioEl.currentTime - jumpSeconds);
    });
    forwardBtn.addEventListener("click", () => {
      audioEl.currentTime = Math.min(
        isFinite(audioEl.duration) ? audioEl.duration : audioEl.currentTime + jumpSeconds,
        audioEl.currentTime + jumpSeconds
      );
    });
    playbackRateSelect.addEventListener("change", () => {
      audioEl.playbackRate = Number(playbackRateSelect.value) || 1;
    });
    if (volumeSlider) {
      volumeSlider.addEventListener("input", () => {
        initAudioGain();
        applyVolume();
      });
    }
    audioEl.addEventListener("timeupdate", updateTimeInfo);
    audioEl.addEventListener("loadedmetadata", updateTimeInfo);
    audioEl.addEventListener("play", () => {
      initAudioGain();
      if (audioCtx && audioCtx.state === "suspended") audioCtx.resume();
    });

    resetTextBtn.addEventListener("click", () => {
      labelEditorEl.value = state.originalPseudoLabel || "";
    });

    acceptBtn.addEventListener("click", () =>
      submitAnnotation({ skipOnly: false, acceptOriginal: true })
    );
    submitNextBtn.addEventListener("click", () =>
      submitAnnotation({ skipOnly: false, acceptOriginal: false })
    );
    skipBtn.addEventListener("click", () =>
      submitAnnotation({ skipOnly: true, acceptOriginal: false })
    );
    if (discardBtn) {
      discardBtn.addEventListener("click", async () => {
        const task = state.tasks[state.currentIndex];
        if (!task) return;
        if (task.status === "removed") {
          const res = await fetch(`/api/annotations/${encodeURIComponent(task.id)}`, {
            method: "DELETE",
          });
          if (!res.ok) {
            console.error("撤销剔除失败", await res.text());
            return;
          }
          // 刷新任务列表并停留在当前 index
          await fetchTasks();
          loadTask(Math.min(state.currentIndex, state.tasks.length - 1));
          return;
        }
        submitAnnotation({
          skipOnly: false,
          acceptOriginal: false,
          removedOnly: true,
        });
      });
    }

    if (exportBackfillBtn) {
      exportBackfillBtn.addEventListener("click", () => {
        // 直接打开导出链接，让浏览器下载生成的 CSV
        window.open("/api/export/backfill-two-col", "_blank");
      });
    }

    shortcutHelpBtn.addEventListener("click", () => {
      shortcutDialog.classList.remove("hidden");
    });
    closeShortcutDialog.addEventListener("click", () => {
      shortcutDialog.classList.add("hidden");
    });
    shortcutDialog.addEventListener("click", (e) => {
      if (e.target === shortcutDialog) {
        shortcutDialog.classList.add("hidden");
      }
    });

    window.addEventListener("keydown", (e) => {
      const active = document.activeElement;
      const inEditor = active === labelEditorEl;

      if (shortcutDialog && !shortcutDialog.classList.contains("hidden")) {
        if (e.key === "Escape") {
          shortcutDialog.classList.add("hidden");
          e.preventDefault();
        }
        return;
      }

      if (e.key === " " && !inEditor) {
        e.preventDefault();
        if (audioEl.paused) audioEl.play();
        else audioEl.pause();
      } else if (e.key === "Enter" && !inEditor) {
        e.preventDefault();
        submitNextBtn.click();
      } else if (e.key.toLowerCase() === "s" && !inEditor) {
        e.preventDefault();
        skipBtn.click();
      } else if (e.key === "ArrowLeft" || e.key.toLowerCase() === "j") {
        if (!inEditor) {
          e.preventDefault();
        }
        audioEl.currentTime = Math.max(0, audioEl.currentTime - jumpSeconds);
      } else if (e.key === "ArrowRight" || e.key.toLowerCase() === "l") {
        if (!inEditor) {
          e.preventDefault();
        }
        audioEl.currentTime = Math.min(
          isFinite(audioEl.duration) ? audioEl.duration : audioEl.currentTime + jumpSeconds,
          audioEl.currentTime + jumpSeconds
        );
      } else if (e.key === "Escape") {
        if (inEditor) {
          labelEditorEl.blur();
        }
      } else if (e.key === "1") {
        setSelectedLabelLevel("A");
      } else if (e.key === "2") {
        setSelectedLabelLevel("B");
      } else if (e.key === "3") {
        setSelectedLabelLevel("C");
      }
    });
  }

  document.addEventListener("DOMContentLoaded", () => {
    setupEvents();
    initAudioGain();
    applyVolume();
    fetchTasks().catch((err) =>
      console.error("加载任务列表失败", err)
    );
  });
})();
