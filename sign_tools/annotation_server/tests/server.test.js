const path = require("path");
const fs = require("fs");
const request = require("supertest");

const rootDir = path.resolve(__dirname, "..");
const dataDir = path.join(rootDir, "data");
const tasksFile = path.join(dataDir, "tasks.json");
const annotationsFile = path.join(dataDir, "annotations.json");

const { app } = require(path.join(rootDir, "server", "index.js"));

function resetTestData() {
  const sampleTasks = [
    {
      id: "sample-001",
      audioPath: "sample-001.wav",
      pseudoLabel: "这是一个示例伪标签，用来测试标注工具的整体流程。",
      confidence: 0.82,
      batchId: "demo-batch-1",
      priority: 1,
      meta: {
        source: "demo",
        durationSec: 5.2
      }
    }
  ];
  fs.mkdirSync(dataDir, { recursive: true });
  fs.writeFileSync(tasksFile, JSON.stringify(sampleTasks, null, 2), "utf8");
  fs.writeFileSync(annotationsFile, "[]", "utf8");
}

describe("Annotation tool server APIs", () => {
  test("GET /api/tasks 应返回任务列表和状态字段", async () => {
    resetTestData();
    const res = await request(app).get("/api/tasks");
    expect(res.status).toBe(200);
    expect(res.body).toHaveProperty("total");
    expect(res.body).toHaveProperty("items");
    expect(Array.isArray(res.body.items)).toBe(true);
    const first = res.body.items[0];
    expect(first).toHaveProperty("id", "sample-001");
    expect(first).toHaveProperty("status", "pending");
    expect(first).toHaveProperty("annotation", null);
  });

  test("GET /api/tasks/:id 应返回单条任务详情", async () => {
    resetTestData();
    const res = await request(app).get("/api/tasks/sample-001");
    expect(res.status).toBe(200);
    expect(res.body).toHaveProperty("id", "sample-001");
    expect(res.body).toHaveProperty("pseudoLabel");
  });

  test("POST /api/annotations 应能写入标注并更新状态", async () => {
    resetTestData();
    const payload = {
      taskId: "sample-001",
      finalText: "修正后的文本",
      labelLevel: "A",
      skipped: false,
      issueTag: null
    };
    const res = await request(app).post("/api/annotations").send(payload);
    expect(res.status).toBe(200);
    expect(res.body).toHaveProperty("ok", true);
    expect(res.body).toHaveProperty("annotation");
    expect(res.body.annotation).toMatchObject({
      taskId: "sample-001",
      finalText: "修正后的文本",
      labelLevel: "A",
      skipped: false
    });

    const listRes = await request(app).get("/api/tasks");
    const first = listRes.body.items[0];
    expect(first.status).toBe("labeled");
    expect(first.annotation).not.toBeNull();
    expect(first.annotation.finalText).toBe("修正后的文本");
  });

  test("GET /api/export?format=json 应返回 JSON 导出结构", async () => {
    resetTestData();
    const res = await request(app).get("/api/export?format=json");
    expect(res.status).toBe(200);
    expect(Array.isArray(res.body)).toBe(true);
    const first = res.body[0];
    expect(first).toHaveProperty("audio_id", "sample-001");
    expect(first).toHaveProperty("final_text");
    expect(first).toHaveProperty("label_level");
    expect(first).toHaveProperty("skipped");
  });

  test("GET /api/export?format=csv 应返回 CSV 内容", async () => {
    resetTestData();
    const res = await request(app).get("/api/export?format=csv");
    expect(res.status).toBe(200);
    expect(res.text).toContain("audio_id,audio_path,final_text,label_level,skipped,issue_tag,confidence,batch_id");
    expect(res.text).toContain("sample-001");
  });
});
