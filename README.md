# 智服AI客服工作台

一个基于 `FastAPI + LangChain + Milvus + PostgreSQL + Redis` 的智能客服Agent项目，支持多轮对话、知识库检索增强、文档上传入库、Skill动态加载，以及面向业务人员的Web工作台。
<img width="1621" height="845" alt="屏幕截图 2026-06-15 162558" src="https://github.com/user-attachments/assets/ab6c1979-9eb2-4d8a-a3fb-a8e809d03d53" />
<img width="1622" height="844" alt="屏幕截图 2026-06-15 162848" src="https://github.com/user-attachments/assets/5edf6934-820f-40f4-975b-ffaa1fa15218" />



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
- Embedding：本地 `bge-m3` 模型 + BM25 sparse embedding
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
├─ app.py                         # 兼容启动入口，转发到 backend.app.main
├─ backend/
│  └─ app/
│     ├─ main.py                  # FastAPI 应用组装与静态资源挂载
│     ├─ schemas.py               # 全部 API 的 Pydantic 请求/响应模型
│     ├─ tools.py                 # 暴露给 Agent 的 LangChain 工具集合
│     ├─ api/
│     │  └─ router.py             # HTTP 路由入口，聚合认证/会话/聊天/知识库/Skill/MCP 接口
│     ├─ core/
│     │  ├─ config.py             # 环境变量读取与全局配置
│     │  ├─ security.py           # JWT、密码哈希、角色权限、依赖注入
│     │  └─ cache.py              # Redis 缓存封装
│     ├─ db/
│     │  ├─ session.py            # SQLAlchemy Engine、SessionLocal、Base、init_db
│     │  └─ models.py             # 用户、会话、消息、父级分块、MCP 配置等 ORM 模型
│     ├─ services/
│     │  └─ agent_service.py      # LangChain Agent、MCP 工具装配、会话存储逻辑
│     ├─ rag/
│     │  ├─ document_loader.py    # PDF/Word/Excel 文档解析与三级分块
│     │  ├─ rag_pipeline.py       # LangGraph RAG 编排流程
│     │  ├─ rag_utils.py          # 检索、重排、查询扩展、Auto-merging 等工具
│     │  └─ parent_chunk_store.py # 父级分块在 PostgreSQL + Redis 中的持久化封装
│     ├─ integrations/
│     │  ├─ embedding.py          # 本地 dense embedding + BM25 sparse embedding
│     │  ├─ milvus_client.py      # Milvus 连接、建表、查询、删除封装
│     │  └─ milvus_writer.py      # 文档向量写入 Milvus
│     ├─ jobs/
│     │  └─ upload_jobs.py        # 上传/删除任务状态管理与进度跟踪
│     └─ skills/
│        └─ skill_loader.py       # Skill 扫描、frontmatter 解析、资源加载
├─ frontend/
│  ├─ index.html                  # 工作台页面骨架
│  ├─ script.js                   # 前端交互逻辑、接口调用、状态管理
│  ├─ style.css                   # 页面样式
│  └─ favicon.ico                 # 站点图标
├─ skills/                        # 外部 Skill 根目录，每个子目录一个技能包
├─ .env                           # 本地开发环境变量
├─ .env.example                   # 环境变量示例
├─ docker-compose.yml             # 本地依赖服务编排
└─ README.md                      # 项目说明文档
```

### 文件作用说明

- `app.py`：保留原有启动方式，内部转发到 `backend.app.main`，避免重构后本地启动命令失效。
- `backend/app/main.py`：创建 FastAPI 应用、注册路由、初始化数据库、挂载 `frontend/` 静态页面。
- `backend/app/api/router.py`：集中定义所有 HTTP 接口，是前端调用后端能力的统一入口。
- `backend/app/core/config.py`：读取 `.env` 中的数据库、Redis、Milvus、模型与服务端口等配置。
- `backend/app/core/security.py`：封装登录认证、JWT 生成校验、管理员校验和数据库依赖。
- `backend/app/core/cache.py`：封装 Redis 读写，给会话缓存和父级分块缓存复用。
- `backend/app/db/session.py`：维护 SQLAlchemy 连接、会话工厂和 `init_db()` 初始化逻辑。
- `backend/app/db/models.py`：定义 `User`、`ChatSession`、`ChatMessage`、`ParentChunk`、`MCPServerConfig` 等表结构。
- `backend/app/schemas.py`：定义认证、会话、聊天、文档、Skill、MCP 等接口的数据结构。
- `backend/app/services/agent_service.py`：负责 Agent 构建、工具来源标记、会话存储、聊天与流式聊天执行。
- `backend/app/tools.py`：定义 `search_knowledge_base`、`list_skills`、`load_skill` 等可供 Agent 调用的工具。
- `backend/app/rag/document_loader.py`：负责业务文档加载、清洗和三级分块。
- `backend/app/rag/rag_pipeline.py`：编排检索、评估、查询重写、生成答案等 RAG 工作流节点。
- `backend/app/rag/rag_utils.py`：实现检索召回、Rerank、Step-back、HyDE、Auto-merging 等核心算法逻辑。
- `backend/app/rag/parent_chunk_store.py`：将父级分块写入 PostgreSQL，并通过 Redis 做热点缓存。
- `backend/app/integrations/embedding.py`：加载本地 `bge-m3` 模型，维护 dense/sparse embedding 与 BM25 状态。
- `backend/app/integrations/milvus_client.py`：封装 Milvus 的集合初始化、向量插入、混合检索与删除操作。
- `backend/app/integrations/milvus_writer.py`：把文档批量向量化后写入 Milvus。
- `backend/app/jobs/upload_jobs.py`：维护上传与删除任务的进度、状态与步骤信息，供前端轮询展示。
- `backend/app/skills/skill_loader.py`：扫描 `skills/` 目录、解析 `SKILL.md` frontmatter，并按需读取资源文件。
- `frontend/index.html`：管理后台页面结构，承载登录、聊天、知识库和 Skill 管理视图。
- `frontend/script.js`：负责前端事件绑定、SSE 流式处理、会话切换与后台接口调用。
- `frontend/style.css`：负责工作台布局、组件样式与交互状态样式。
- `frontend/favicon.ico`：浏览器页签图标资源。
- `skills/`：外部技能包目录，每个技能由 `SKILL.md` 和若干 `references/`、`assets/`、`scripts/` 资源组成。
- `.env`：本地实际运行配置，通常不提交到仓库。
- `.env.example`：给新环境初始化用的配置模板。
- `docker-compose.yml`：用于本地拉起 Milvus 等依赖服务。
- `README.md`：项目使用说明、架构说明和目录说明。
- `backend/`、`backend/app/` 及其子目录下的 `__init__.py`：Python 包标记文件，用于支持包内导入与模块组织。

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

## 致谢
本项目基于icey1287/SuperMew二次开发，感谢原作者的开源贡献。
在原有基础上新增/优化了以下特性：
1. 修改了BM25的Tokenizer方法，先用正则把中英文片段切出来，中文交给jieba分词，英文/数字片段整体保留，最终用于BM25稀疏向量构建
2. Skill模块化扩展：支持将客服话术、业务规则、处理流程和参考资料封装为独立Skill。Agent可根据用户任务动态发现并加载对应Skill，在不改动主对话流程的情况下扩展新能力，提升回答的专业性和可维护性
3. 使用ContextVar管理请求级上下文，替代简单全局变量，避免并发场景下的上下文污染，提升异步对话链路的隔离性和稳定性

## 许可证
MIT License

