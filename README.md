# 智服AI客服工作台

一个基于 `FastAPI + LangChain + Milvus + PostgreSQL + Redis` 的智能客服Agent项目，支持多轮对话、知识库检索增强、文档上传入库、Skill动态加载，以及面向业务人员的Web工作台。

项目同时提供后端API和内置前端页面，适合用作以下场景的原型或内部系统基础：

- 企业知识库问答
- 售前/售后客服辅助
- SOP与文档检索增强问答
- 可扩展的Skill化业务能力注入 

## 项目特性

- 多轮对话
  - 支持按用户维度保存会话、查看历史会话、删除会话
  - 支持普通返回和流式返回
- RAG 检索增强
  - 基于 Milvus 做向量检索
  - 结合本地 dense embedding 和 BM25 sparse embedding
  - 支持混合召回、RRF 融合、Rerank、查询重写、Step-back、HyDE
  - 返回可视化 `rag_trace`，便于调试检索过程
- 文档知识库
  - 自动三层分块
  - 父级分块落 PostgreSQL，叶子分块落 Milvus
  - 支持异步上传、异步删除和前端进度展示
- Skill 能力扩展
  - 支持上传 Skill zip 包
  - 自动扫描 `SKILL.md` 和关联资源
  - Agent 可按需发现、加载、复用 Skill 上下文
- 权限体系
  - 支持用户注册、登录、JWT 鉴权
  - 支持普通用户 / 管理员角色
  - 管理员可管理知识库与 Skill
- 内置前端
  - 提供客服工作台 UI
  - 支持登录、聊天、会话中心、知识库管理、Skill 管理

## 技术栈

- 后端：`FastAPI`、`SQLAlchemy`、`Uvicorn`
- Agent：`LangChain`、`LangGraph`
- 向量库：`Milvus`
- 关系型存储：`PostgreSQL`
- 缓存：`Redis`
- Embedding：本地 `bge-m3` 模型
- 前端：原生 `HTML + CSS + JavaScript`，通过 `Vue 3 CDN` 增强交互

## 系统架构

```text
Frontend Workbench
        |
        v
     FastAPI
        |
        +-- Auth / Session / Chat API
        +-- Document Upload / Delete API
        +-- Skill Management API
        |
        v
   LangChain Agent
        |
        +-- search_knowledge_base
        +-- list_skills / load_skill / load_skill_resource
        |
        v
   RAG Pipeline
        |
        +-- Dense Embedding (local bge-m3)
        +-- Sparse Embedding (BM25)
        +-- Hybrid Retrieve / Rerank / Auto-merge
        |
        +-- Milvus        -> leaf chunks
        +-- PostgreSQL    -> users, sessions, parent chunks
        +-- Redis         -> conversation cache
```

## 目录结构

```text
agent/
├─ app.py                 # FastAPI 应用入口
├─ api.py                 # 业务 API 路由
├─ agent.py               # LangChain Agent 与会话存储逻辑
├─ auth.py                # 登录、注册、JWT、角色权限
├─ rag_pipeline.py        # RAG 编排流程
├─ rag_utils.py           # 检索、重排、查询扩展等工具
├─ embedding.py           # 本地向量模型与 BM25 稀疏向量
├─ milvus_client.py       # Milvus 连接与检索封装
├─ milvus_writer.py       # 文档向量写入
├─ document_loader.py     # PDF/Word/Excel 文档解析与三级分块
├─ parent_chunk_store.py  # 父级分块持久化
├─ database.py            # PostgreSQL 连接
├─ cache.py               # Redis 缓存封装
├─ skill_loader.py        # Skill 扫描、解析、资源加载
├─ tools.py               # 暴露给 Agent 的工具
├─ upload_jobs.py         # 异步上传/删除任务状态管理
├─ frontend/              # 前端工作台
├─ skills/                # 外部 Skill 目录
└─ docker-compose.yml     # Milvus 相关服务编排
```

## 核心能力说明

### 1. 对话能力

- 用户登录后可创建和维护自己的会话
- 对话消息存储在 PostgreSQL 中
- Redis 用于缓存消息与会话元数据
- 支持 `/chat` 普通响应和 `/chat/stream` SSE 流式响应

### 2. 知识库能力

- 上传业务文档后，系统会自动执行三级分块
- 叶子分块写入 Milvus，用于检索
- 父级分块写入 PostgreSQL，用于 Auto-merging 场景
- 检索链路支持：
  - Hybrid Retrieval
  - RRF 融合
  - DashScope Rerank
  - 查询重写
  - Step-back 问题扩展
  - HyDE 假设文档扩展

### 3. Skill 能力

Skill 目录约定如下：

```text
skills/
└─ your-skill/
   ├─ SKILL.md
   ├─ references/
   ├─ assets/
   └─ scripts/
```

其中：

- `SKILL.md` 是入口文件
- 支持在 frontmatter 中定义 `name`、`description`、`references`
- Agent 可以通过工具动态读取 Skill 内容和资源文件

## API 概览

### 认证

- `POST /auth/register`：注册账号
- `POST /auth/login`：登录
- `GET /auth/me`：获取当前用户信息

### 会话

- `GET /sessions`：获取当前用户会话列表
- `GET /sessions/{session_id}`：获取指定会话消息
- `DELETE /sessions/{session_id}`：删除会话

### 聊天

- `POST /chat`：普通聊天
- `POST /chat/stream`：流式聊天

### 文档知识库

- `GET /documents`：获取文档列表
- `POST /documents/upload`：同步上传文档
- `POST /documents/upload/async`：异步上传文档
- `GET /documents/upload/jobs`：获取上传任务列表
- `GET /documents/upload/jobs/{job_id}`：获取上传任务详情
- `DELETE /documents/{filename}`：删除文档向量数据
- `DELETE /documents/delete/async/{filename}`：异步删除文档
- `GET /documents/delete/jobs/{job_id}`：获取删除任务详情

### Skill 管理

- `GET /admin/skills`：获取 Skill 列表
- `POST /admin/skills/upload`：上传 Skill zip 包

## 前端说明

前端资源位于 `frontend/` 目录，由 FastAPI 直接托管，包含：

- 登录 / 注册面板
- 聊天工作台
- 历史会话中心
- 管理员知识库管理面板
- 管理员 Skill 管理面板
- 流式回答和检索过程可视化

因此这个项目默认是“单体交付”风格：后端 API 和前端页面一起运行，不需要再单独起一个前端开发服务器。

## 开发建议

这个项目已经具备一个完整 AI 客服系统原型的核心骨架，后续可以继续往下面扩展：

- 接入更多文档格式
- 增加工单系统或CRM对接
- 增加更细粒度权限控制
- 增加测试、日志和监控
- 支持多模型切换与配置中心
- 支持对象存储和分布式任务队列
