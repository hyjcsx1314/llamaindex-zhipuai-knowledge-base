# AI Knowledge Base

基于 **LlamaIndex + 智谱 GLM + LlamaDeploy UI** 的 PDF / 文本文档知识库问答系统。

系统会先把文档解析、切分并建立向量索引；用户在网页界面提问时，Agent 调用知识库检索工具，智谱模型根据检索结果生成回答，并展示引用来源。

## 功能

- PDF / 文本文档索引
- 智谱 `embedding-3` 向量化
- 智谱 GLM 回答生成
- `AgentWorkflow` 工具调用
- 流式回答
- 文档引用与来源展示
- 本地 Web UI
- 工具调用异常兜底

## 工作流程

```text
PDF / 文本
  ↓
文档解析与切分
  ↓
embedding-3 生成向量
  ↓
保存到 src/storage
  ↓
用户提问
  ↓
Agent 调用 query_index
  ↓
检索相关文档片段
  ↓
智谱 GLM 生成回答
  ↓
UI 展示答案和来源
```

## 项目结构

```text
ai-knowledge-base/
├─ src/
│  ├─ settings.py
│  ├─ workflow.py
│  ├─ query.py
│  ├─ index.py
│  ├─ generate.py
│  ├─ citation.py
│  └─ .env
├─ ui/
│  └─ data/
├─ llama_deploy.yml
├─ pyproject.toml
└─ README.md
```

## 环境

当前项目主要在以下环境中验证：

- Windows
- Python 3.11
- `uv`
- Node.js
- pnpm 9
- LlamaDeploy 0.9.2
- LlamaIndex Core 0.13.6
- LlamaIndex Workflows 1.3.0
- `llama-index-llms-zhipuai` 0.4.1

> 本项目依赖一组已验证的兼容版本。不要在未测试的情况下全面升级 LlamaIndex 相关依赖。

## 安装

```powershell
git clone <YOUR_REPOSITORY_URL>
cd ai-knowledge-base
uv sync
```

## 配置

```powershell
Copy-Item .\src\.env.example .\src\.env
```

编辑 `src/.env`：

```dotenv
ZHIPU_API_KEY=your_api_key
MODEL=glm-4-flash
EMBEDDING_MODEL=embedding-3
EMBEDDING_DIMENSIONS=1024
TEMPERATURE=0.2
MAX_TOKENS=4096
EMBEDDING_TIMEOUT=60
```

不要提交真实 API Key。

## 添加知识库文档

将有权使用的 PDF 或文本放入：

```text
ui/data/
```

生成索引：

```powershell
uv run generate
```

## 启动

窗口 1：

```powershell
uv run -m llama_deploy.apiserver
```

窗口 2：

```powershell
uv run llamactl deploy .\llama_deploy.yml
```

打开：

```text
http://127.0.0.1:4501/deployments/chat/ui
```

## 已处理的兼容问题

### 智谱流式工具调用

`llama-index-llms-zhipuai==0.4.1` 的原始流式实现可能在 `delta.content is None` 时跳过包含 `tool_calls` 的分片。项目中的 `SafeZhipuAI` 会保留并合并工具名称和 JSON 参数，同时对工具调用失败返回明确错误。

### Windows pnpm

LlamaDeploy 0.9.2 在 Windows 上可能需要把启动命令从 `pnpm` 改为 `pnpm.cmd`。该修改位于 `.venv`，重建虚拟环境后可能需要重新应用。

## 安全说明

提交前确认：

- `src/.env` 未被 Git 跟踪
- API Key 未出现在源码、日志或提交历史中
- `.venv`、`node_modules`、临时部署目录未提交
- 仅提交你有权公开的 PDF 和数据
- 日志中没有敏感业务信息

```powershell
git status
git diff --cached
git ls-files | Select-String "\.env$"
```

如果真实密钥曾经提交，应立即在智谱平台撤销并重新生成。

## 后续计划

- [ ] 多文档批量导入
- [ ] 文档增量索引
- [ ] 页码与原文定位
- [ ] OCR 和扫描 PDF 支持
- [ ] 混合检索与重排序
- [ ] 强制知识库问题先检索
- [ ] 用户与文档权限控制
- [ ] 自动化测试
- [ ] Docker 部署
- [ ] 迁移到仍在维护的工作流服务方案
- [ ] 工具调用监控与日志

## License

请根据项目用途选择许可证。尚未确定前，不要默认添加许可证。
