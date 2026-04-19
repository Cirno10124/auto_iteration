const path = require("path");
const fs = require("fs");
const express = require("express");
const cors = require("cors");

const app = express();
app.use(cors());
app.use(express.json());

const rootDir = path.resolve(__dirname, "..");
const configPath = path.join(rootDir, "annotation_tool.config.json");
const config = JSON.parse(fs.readFileSync(configPath, "utf8"));

const dataDir = path.resolve(rootDir, "data");
const tasksFile = path.resolve(rootDir, config.tasksFile || "./data/tasks.json");
const annotationsFile = path.resolve(rootDir, config.annotationsFile || "./data/annotations.json");

function loadJson(filePath, fallback) {
  try {
    if (!fs.existsSync(filePath)) {
      return fallback;
    }
    const raw = fs.readFileSync(filePath, "utf8");
    return raw.trim() ? JSON.parse(raw) : fallback;
  } catch (e) {
    console.error("Failed to load JSON:", filePath, e);
    return fallback;
  }
}

function saveJson(filePath, data) {
  fs.mkdirSync(path.dirname(filePath), { recursive: true });
  fs.writeFileSync(filePath, JSON.stringify(data, null, 2), "utf8");
}

function getTasks() {
  return loadJson(tasksFile, []);
}

function getAnnotations() {
  return loadJson(annotationsFile, []);
}

function upsertAnnotation(annotation) {
  const annotations = getAnnotations();
  const idx = annotations.findIndex((a) => a.taskId === annotation.taskId);
  if (idx >= 0) {
    annotations[idx] = { ...annotations[idx], ...annotation };
  } else {
    annotations.push(annotation);
  }
  saveJson(annotationsFile, annotations);
  return annotations;
}

function deleteAnnotation(taskId) {
  const annotations = getAnnotations();
  const next = annotations.filter((a) => a.taskId !== taskId);
  saveJson(annotationsFile, next);
  return next;
}

function enrichTasksWithStatus(tasks, annotations) {
  const map = new Map();
  annotations.forEach((a) => {
    map.set(a.taskId, a);
  });
  return tasks.map((t, index) => {
    const ann = map.get(t.id);
    let status = "pending";
    if (ann) {
      if (ann.removed) status = "removed";
      else if (ann.skipped) status = "skipped";
      else status = "labeled";
    }
    return {
      ...t,
      index,
      status,
      annotation: ann || null,
    };
  });
}

app.use(
  "/audio",
  express.static(path.resolve(rootDir, config.audioRoot || "./data/audio"), {
    fallthrough: true,
  })
);

// 通过 taskId 获取音频（支持 tasks.json 中记录 absolute originalPath）
app.get("/api/audio/:taskId", (req, res) => {
  const tasks = getTasks();
  const task = tasks.find((t) => t.id === req.params.taskId);
  if (!task) {
    return res.status(404).json({ error: "Task not found" });
  }

  const candidatePaths = [];
  if (typeof task.originalPath === "string" && task.originalPath.trim()) {
    candidatePaths.push(task.originalPath.trim());
  }
  if (typeof task.audioPath === "string" && task.audioPath.trim()) {
    const audioRoot = path.resolve(rootDir, config.audioRoot || "./data/audio");
    candidatePaths.push(path.join(audioRoot, task.audioPath.trim()));
  }

  const existing = candidatePaths.find((p) => {
    try {
      return fs.existsSync(p) && fs.statSync(p).isFile();
    } catch {
      return false;
    }
  });

  if (!existing) {
    return res.status(404).json({
      error: "Audio file not found",
      tried: candidatePaths,
    });
  }

  return res.sendFile(existing);
});

app.get("/api/tasks", (req, res) => {
  const tasks = getTasks();
  const annotations = getAnnotations();
  const enriched = enrichTasksWithStatus(tasks, annotations);

  const { status } = req.query;
  let filtered = enriched;
  if (
    status === "pending" ||
    status === "labeled" ||
    status === "skipped" ||
    status === "removed"
  ) {
    filtered = enriched.filter((t) => t.status === status);
  }

  res.json({
    total: enriched.length,
    items: filtered,
  });
});

app.get("/api/tasks/:id", (req, res) => {
  const tasks = getTasks();
  const annotations = getAnnotations();
  const enriched = enrichTasksWithStatus(tasks, annotations);
  const task = enriched.find((t) => t.id === req.params.id);
  if (!task) {
    return res.status(404).json({ error: "Task not found" });
  }
  res.json(task);
});

app.post("/api/annotations", (req, res) => {
  const {
    taskId,
    finalText,
    labelLevel,
    skipped,
    issueTag,
    removed,
  } = req.body || {};
  if (!taskId) {
    return res.status(400).json({ error: "taskId is required" });
  }
  const tasks = getTasks();
  if (!tasks.find((t) => t.id === taskId)) {
    return res.status(404).json({ error: "Task not found" });
  }

  const annotation = {
    taskId,
    finalText: typeof finalText === "string" ? finalText : "",
    labelLevel: labelLevel || "B",
    skipped: Boolean(skipped),
    issueTag: issueTag || null,
    removed: Boolean(removed),
    updatedAt: new Date().toISOString(),
  };

  upsertAnnotation(annotation);
  res.json({ ok: true, annotation });
});

// 撤销剔除/撤销标注：删除标注记录，使任务回到 pending
app.delete("/api/annotations/:taskId", (req, res) => {
  const taskId = req.params.taskId;
  if (!taskId) {
    return res.status(400).json({ error: "taskId is required" });
  }
  deleteAnnotation(taskId);
  return res.json({ ok: true });
});

app.get("/api/export", (req, res) => {
  const format = (req.query.format || "json").toString().toLowerCase();
  const tasks = getTasks();
  const annotations = getAnnotations();
  const enriched = enrichTasksWithStatus(tasks, annotations);

  if (format === "csv") {
    const headers = [
      "audio_id",
      "audio_path",
      "final_text",
      "label_level",
      "skipped",
      "issue_tag",
      "confidence",
      "batch_id",
    ];
    const rows = enriched.map((t) => {
      const ann = t.annotation || {};
      const removedFlag = Boolean(ann.removed);
      const skippedFlag = removedFlag ? true : Boolean(ann.skipped);
      const values = [
        t.id,
        t.audioPath || "",
        removedFlag ? "" : ann.finalText || "",
        removedFlag ? "" : ann.labelLevel || "",
        skippedFlag ? "1" : "0",
        removedFlag ? "" : ann.issueTag || "",
        typeof t.confidence === "number" ? t.confidence.toString() : "",
        t.batchId || "",
      ];
      return values
        .map((v) => {
          const s = String(v ?? "");
          if (s.includes(",") || s.includes('"') || s.includes("\n")) {
            return `"${s.replace(/"/g, '""')}"`;
          }
          return s;
        })
        .join(",");
    });
    const csv = [headers.join(","), ...rows].join("\n");
    res.header("Content-Type", "text/csv; charset=utf-8");
    res.attachment("annotations.csv");
    return res.send(csv);
  }

  res.json(
    enriched.map((t) => ({
      audio_id: t.id,
      audio_path: t.audioPath,
      final_text: (() => {
        const ann = t.annotation || {};
        return ann.removed ? "" : ann.finalText || "";
      })(),
      label_level: (() => {
        const ann = t.annotation || {};
        return ann.removed ? "" : ann.labelLevel || "";
      })(),
      skipped: (() => {
        const ann = t.annotation || {};
        return ann.removed ? true : Boolean(ann.skipped);
      })(),
      issue_tag: (() => {
        const ann = t.annotation || {};
        return ann.removed ? null : ann.issueTag || null;
      })(),
      confidence: t.confidence,
      batch_id: t.batchId || null,
    }))
  );
});

// 基于最初的 CSV 回填标注，导出「原 CSV + 标注列」的新 CSV。
// 依赖 scripts/csv_to_tasks.py 写入的 data/source_csv_path.txt 与 tasks.json 中的 originalPath 字段。
app.get("/api/export/backfill", (req, res) => {
  try {
    const notePath = path.join(dataDir, "source_csv_path.txt");
    if (!fs.existsSync(notePath)) {
      return res.status(400).json({
        error:
          "找不到源 CSV 路径记录。请先使用 csv_to_tasks.py 从 CSV 生成 tasks.json。",
      });
    }
    const sourceCsvPath = fs.readFileSync(notePath, "utf8").trim();
    if (!sourceCsvPath) {
      return res
        .status(400)
        .json({ error: "源 CSV 路径记录为空，请重新生成 tasks.json。" });
    }

    if (!fs.existsSync(sourceCsvPath)) {
      return res.status(404).json({
        error: `源 CSV 文件不存在: ${sourceCsvPath}`,
      });
    }

    const tasks = getTasks();
    const annotations = getAnnotations();
    const annMap = new Map();
    annotations.forEach((a) => annMap.set(a.taskId, a));
    const taskByOriginalPath = new Map();
    tasks.forEach((t) => {
      if (t.originalPath) {
        taskByOriginalPath.set(t.originalPath, t);
      }
    });

    const csvContent = fs.readFileSync(sourceCsvPath, "utf8");
    const lines = csvContent.split(/\r?\n/).filter((l) => l.length > 0);
    if (lines.length === 0) {
      return res
        .status(400)
        .json({ error: "源 CSV 文件为空，无法回填。" });
    }

    const headerLine = lines[0];
    const headers = headerLine.split(",");
    const hasHeader =
      headers.includes("audio_filepath") && headers.includes("text");

    const rows = [];
    let dataLines = lines;
    if (hasHeader) {
      dataLines = lines.slice(1);
    }

    const outHeaders = [
      ...headers,
      "final_text",
      "label_level",
      "skipped",
      "issue_tag",
    ];

    dataLines.forEach((line) => {
      const cols = [];
      let current = "";
      let inQuotes = false;
      for (let i = 0; i < line.length; i++) {
        const ch = line[i];
        if (ch === '"' && (i === 0 || line[i - 1] !== "\\")) {
          inQuotes = !inQuotes;
          continue;
        }
        if (ch === "," && !inQuotes) {
          cols.push(current);
          current = "";
        } else {
          current += ch;
        }
      }
      cols.push(current);

      const rowObj = {};
      headers.forEach((h, idx) => {
        rowObj[h] = cols[idx] ?? "";
      });

      const originalPath = rowObj["audio_filepath"] || "";
      const task = taskByOriginalPath.get(originalPath);
      const ann = task ? annMap.get(task.id) : null;

      const removedFlag = ann && ann.removed ? true : false;
      const skippedFlag = ann && ann.skipped ? true : false;

      // 训练/回填侧通常只依赖原始的两列：audio_filepath,text。
      // - removed：text 置空（不训练）
      // - labeled：text 替换为 finalText
      // - skipped：保持原始 text 不变
      if (typeof rowObj["text"] !== "undefined") {
        if (removedFlag) rowObj["text"] = "";
        else if (ann && !skippedFlag) rowObj["text"] = ann.finalText || "";
      }

      rowObj["final_text"] = removedFlag ? "" : ann && ann.finalText ? ann.finalText : "";
      rowObj["label_level"] = removedFlag
        ? ""
        : ann && ann.labelLevel
        ? ann.labelLevel
        : "";
      rowObj["skipped"] = removedFlag || skippedFlag ? "1" : "0";
      rowObj["issue_tag"] = removedFlag
        ? ""
        : ann && ann.issueTag
        ? ann.issueTag
        : "";

      const outRow = outHeaders
        .map((h) => {
          const v = rowObj[h] ?? "";
          const s = String(v);
          if (s.includes(",") || s.includes('"') || s.includes("\n")) {
            return `"${s.replace(/"/g, '""')}"`;
          }
          return s;
        })
        .join(",");

      rows.push(outRow);
    });

    const outCsv = [outHeaders.join(","), ...rows].join("\n");
    res.header("Content-Type", "text/csv; charset=utf-8");
    res.attachment("annotated_backfill.csv");
    return res.send(outCsv);
  } catch (err) {
    console.error("Backfill export failed:", err);
    return res.status(500).json({ error: "回填导出失败", detail: String(err) });
  }
});

// 只输出原始两列：audio_filepath,text，并根据标注结果回填/置空。
// - labeled：将 text 替换为 finalText
// - skipped：保持原始 text 不变
// - removed：将 text 置空（从训练侧移除）
app.get("/api/export/backfill-two-col", (req, res) => {
  try {
    const notePath = path.join(dataDir, "source_csv_path.txt");
    if (!fs.existsSync(notePath)) {
      return res.status(400).json({
        error:
          "找不到源 CSV 路径记录。请先使用 csv_to_tasks.py 从 CSV 生成 tasks.json。",
      });
    }
    const sourceCsvPath = fs.readFileSync(notePath, "utf8").trim();
    if (!sourceCsvPath || !fs.existsSync(sourceCsvPath)) {
      return res.status(400).json({
        error: `源 CSV 文件不存在: ${sourceCsvPath}`,
      });
    }

    const tasks = getTasks();
    const annotations = getAnnotations();
    const annMap = new Map();
    annotations.forEach((a) => annMap.set(a.taskId, a));
    const taskByOriginalPath = new Map();
    tasks.forEach((t) => {
      if (t.originalPath) {
        taskByOriginalPath.set(t.originalPath, t);
      }
    });

    const csvContent = fs.readFileSync(sourceCsvPath, "utf8");
    const lines = csvContent.split(/\r?\n/).filter((l) => l.length > 0);
    if (lines.length === 0) {
      return res.status(400).json({ error: "源 CSV 文件为空，无法回填。" });
    }

    function escapeCsvValue(v) {
      const s = String(v ?? "");
      if (s.includes(",") || s.includes('"') || s.includes("\n")) {
        return `"${s.replace(/"/g, '""')}"`;
      }
      return s;
    }

    function parseCsvLine(line) {
      const cols = [];
      let current = "";
      let inQuotes = false;
      for (let i = 0; i < line.length; i++) {
        const ch = line[i];
        if (ch === '"' && (i === 0 || line[i - 1] !== "\\")) {
          inQuotes = !inQuotes;
          continue;
        }
        if (ch === "," && !inQuotes) {
          cols.push(current);
          current = "";
        } else {
          current += ch;
        }
      }
      cols.push(current);
      return cols;
    }

    const headerLine = lines[0];
    const headers = headerLine.split(",");
    const hasHeader =
      headers.includes("audio_filepath") && headers.includes("text");

    const audioIdx = hasHeader ? headers.indexOf("audio_filepath") : 0;
    const textIdx = hasHeader ? headers.indexOf("text") : 1;

    const outLines = [];
    if (hasHeader) outLines.push(headerLine);

    const dataLines = hasHeader ? lines.slice(1) : lines;
    dataLines.forEach((line) => {
      const cols = parseCsvLine(line);
      const originalPath = cols[audioIdx] ?? "";
      const originalText = cols[textIdx] ?? "";

      const task = taskByOriginalPath.get(originalPath);
      const ann = task ? annMap.get(task.id) : null;

      const removedFlag = ann && ann.removed ? true : false;
      const skippedFlag = ann && ann.skipped ? true : false;

      let outText = originalText;
      if (removedFlag) outText = "";
      else if (ann && !skippedFlag) outText = ann.finalText || "";

      outLines.push(
        `${escapeCsvValue(originalPath)},${escapeCsvValue(outText)}`
      );
    });

    res.header("Content-Type", "text/csv; charset=utf-8");
    res.attachment("annotated_backfill_two_col.csv");
    return res.send(outLines.join("\n"));
  } catch (err) {
    console.error("Backfill two-col export failed:", err);
    return res.status(500).json({ error: "回填导出失败", detail: String(err) });
  }
});

app.use(
  "/",
  express.static(path.resolve(rootDir, "web"), {
    index: "index.html",
  })
);

const port = config.serverPort || 3100;
const host = config.serverHost ?? "0.0.0.0";

let serverInstance = null;
if (require.main === module) {
  serverInstance = app.listen(port, host, () => {
    console.log(
      `Annotation tool server listening on http://${host}:${port}`
    );
  });
}

module.exports = {
  app,
  listen: (p = port, cb) => {
    if (serverInstance) return serverInstance;
    serverInstance = app.listen(p, host, cb);
    return serverInstance;
  },
};
