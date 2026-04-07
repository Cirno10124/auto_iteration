# 音频标注工具（Node 后端 + 静态前端）

用于「模型预生成伪标签 + 人工校对」的标注服务，与 [annotation_tool_requirements](../../docs/annotation_tool_requirements.md) 对应。

## 目录结构

- `server/` — Express API（任务列表、标注提交、导出）
- `web/` — 静态前端（HTML/CSS/JS）
- `data/` — 任务与标注数据（tasks.json、annotations.json、audio/）
- `tests/` — API 测试（Jest + Supertest）

## 运行方式

在 **本目录** 下执行（依赖与根目录或 sign_tools 的 node_modules 分离，需单独安装）：

```bash
cd auto_iteration/sign_tools/annotation_server
npm install
npm run dev
```

浏览器访问：`http://localhost:3100`（端口可在 `annotation_tool.config.json` 中修改）。

## 测试

```bash
npm test
```

## 打包与移动

整个 `annotation_server` 目录可单独复制或打包；在新环境中执行 `npm install` 后即可 `npm run dev` 或 `npm test`。无需依赖项目根目录或 sign_tools 的 node_modules。
