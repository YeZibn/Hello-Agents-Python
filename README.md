# Hello Agents Python

一个基于 Python 的智能代理（Agent）应用开发框架，支持 ReAct 推理模式、记忆管理和 RAG（检索增强生成）能力。

## 项目结构

```
Hello-Agents-Python/
├── hello_agents/              # 主包
│   ├── __init__.py            # 包初始化
│   ├── main.py                # 主入口
│   ├── agent/                 # Agent 相关
│   │   ├── agent.py           # Agent 基类
│   │   └── react_agent.py     # ReAct Agent 实现
│   ├── config/                # 配置相关
│   │   ├── config.py          # 全局配置
│   │   ├── memory_config.py   # 记忆配置
│   │   └── rag_config.py      # RAG 配置
│   ├── context/               # 上下文管理
│   │   ├── context_builder.py # 上下文构建器
│   │   └── context_packet.py  # 上下文数据包
│   ├── embedder/              # 嵌入模型
│   │   └── qwen_embedder.py   # Qwen 嵌入模型
│   ├── llm/                   # LLM 相关
│   │   └── HelloAgentsLLM.py # HelloAgents LLM
│   ├── memory/                # 记忆系统
│   │   ├── memory.py          # 记忆基类
│   │   ├── memory_manager.py # 记忆管理器
│   │   ├── memory_tool.py    # 记忆工具
│   │   ├── memory_list/      # 记忆实现
│   │   │   ├── episodic_memory.py    # 情景记忆
│   │   │   ├── semantic_memory.py   # 语义记忆
│   │   │   ├── working_memory.py    # 工作记忆
│   │   │   └── perceptual_memory.py  # 感知记忆
│   │   ├── neo4j/            # Neo4j 图数据库
│   │   │   └── neo4j_graph_store.py
│   │   ├── qdrant/           # Qdrant 向量数据库
│   │   │   └── qdrant_vector_store.py
│   │   └── sqlite/           # SQLite 文档存储
│   │       └── sqlite_document_store.py
│   ├── message/              # 消息处理
│   │   └── message.py
│   ├── rag/                  # RAG 系统
│   │   ├── rag_manager.py    # RAG 管理器
│   │   └── rag_tool.py      # RAG 工具
│   ├── tool/                 # 工具系统
│   │   ├── tool.py           # 工具基类
│   │   ├── tool_chain.py     # 工具链
│   │   ├── tool_registry.py  # 工具注册表
│   │   └── tool_list/       # 内置工具
│   │       ├── bash_tool.py  # Bash 命令工具
│   │       ├── read_tool.py  # 读取文件工具
│   │       └── write_tool.py # 写入文件工具
│   └── utils/                # 工具函数
│       ├── __init__.py
│       └── helpers.py
├── tests/                    # 测试目录
├── requirements.txt          # 依赖
└── setup.py                  # 包安装配置
```

## 核心功能

### 1. Agent 系统

基于 ReAct（Reasoning + Acting）模式的智能代理，支持：
- 推理与行动结合
- 工具调用（tool_calls）
- 多轮对话
- 上下文管理

```python
from hello_agents.agent.react_agent import ReActAgent
from hello_agents.llm.HelloAgentsLLM import HelloAgentsLLM
from hello_agents.tool.tool_registry import ToolRegistry

# 初始化
agent = ReActAgent(
    name="助手",
    llm=llm,
    tool_registry=tool_registry
)

# 运行
result = agent.run("你的问题")
```

### 2. 记忆系统

三层记忆架构，支持用户数据隔离：

| 记忆类型 | 描述 | 存储 |
|---------|------|------|
| 工作记忆 (Working Memory) | 短期记忆，有容量限制 | SQLite |
| 情景记忆 (Episodic Memory) | 按会话存储的经历 | SQLite + Qdrant |
| 语义记忆 (Semantic Memory) | 知识图谱，实体关系 | Neo4j + Qdrant |

```python
from hello_agents.memory.memory_tool import MemoryTool

memory_tool = MemoryTool(user_id="user_001")

# 存储记忆
memory_tool.run({
    "action": "store",
    "content": "用户喜欢川菜",
    "memory_type": "semantic",
    "importance": 0.8
})

# 检索记忆
results = memory_tool.run({
    "action": "retrieve",
    "query": "用户口味偏好"
})
```

### 3. RAG 系统

检索增强生成系统，支持多格式文档处理：

```python
from hello_agents.rag.rag_tool import RAGTool

rag_tool = RAGTool(user_id="user_001")

# 检索文档
results = rag_tool.run({
    "action": "search",
    "query": "退换货政策",
    "top_k": 5
})
```

#### 支持的文档格式
- PDF、Word、Excel、PPT
- TXT、Markdown、JSON、CSV
- XML、HTML

#### RAG 管理器

```python
from hello_agents.rag.rag_manager import RAGManager

rag_manager = RAGManager(user_id="user_001")

# 索引文档
rag_manager.index_document("path/to/document.pdf")

# 添加文本
rag_manager.add_text("要添加的文本内容", source="手动输入")

# 列出文档
rag_manager.list_documents()
```

### 4. 工具系统

可扩展的工具注册与调用机制：

```python
from hello_agents.tool.tool_registry import ToolRegistry
from hello_agents.tool.tool_list.bash_tool import BashTool

registry = ToolRegistry()
registry.register_tool(BashTool())

# 执行工具
result = registry.execute_tool("bash", "command=ls -la")
```

#### 内置工具
- **bash**: 执行 Shell 命令
- **read**: 读取文件内容
- **write**: 写入文件内容
- **memory**: 记忆操作
- **rag**: 文档检索

## 安装

### 环境要求
- Python 3.10+
- Neo4j 图数据库（用于语义记忆）
- Qdrant 向量数据库（用于向量检索）

### 安装依赖

```bash
pip install -r requirements.txt
```

### 数据库配置

确保以下服务运行：
1. **Neo4j**: http://localhost:7474
2. **Qdrant**: http://localhost:6333

可在 `hello_agents/config/` 目录下修改数据库连接配置。

## 使用示例

### 1. 创建 Agent 并使用工具

```python
from hello_agents.agent.react_agent import ReActAgent
from hello_agents.llm.HelloAgentsLLM import HelloAgentsLLM
from hello_agents.tool.tool_registry import ToolRegistry
from hello_agents.rag.rag_tool import RAGTool
from hello_agents.memory.memory_tool import MemoryTool

# 初始化 LLM
llm = HelloAgentsLLM(model="your-model")

# 初始化工具注册表
registry = ToolRegistry()
registry.register_tool(RAGTool(user_id="user_001"))
registry.register_tool(MemoryTool(user_id="user_001"))

# 创建 Agent
agent = ReActAgent(
    name="智能助手",
    llm=llm,
    tool_registry=registry,
    max_steps=50
)

# 运行
response = agent.run("你好，我想了解一下退换货政策")
print(response)
```

### 2. 使用记忆系统

```python
from hello_agents.memory.memory_tool import MemoryTool

memory = MemoryTool(user_id="user_xiaoming")

# 存储用户偏好
memory.run({
    "action": "store",
    "content": "用户喜欢川菜和海鲜，偏好辣味",
    "memory_type": "semantic",
    "importance": 0.9,
    "metadata": {
        "tags": ["口味偏好", "川菜", "海鲜"],
        "user_name": "小明"
    }
})

# 按会话检索
memory.run({
    "action": "retrieve_by_session",
    "session_id": "session_123"
})

# 获取统计信息
memory.run({"action": "stats"})
```

### 3. RAG 文档管理

```python
from hello_agents.rag.rag_manager import RAGManager

rag = RAGManager(user_id="user_001")

# 索引 PDF 文档
result = rag.index_document("path/to/manual.pdf")
print(result)  # ✅ 文档索引成功，共生成 XX 个分块

# 搜索相关文档
result = rag.query("退换货期限")
print(result)
```

## 配置说明

### 记忆配置 (memory_config.py)

```python
from hello_agents.config.memory_config import MemoryConfig

config = MemoryConfig(
    working_memory_capacity=100,      # 工作记忆容量
    episodic_ttl_days=30,             # 情景记忆保留天数
    neo4j_uri="bolt://localhost:7687", # Neo4j 连接地址
    neo4j_username="neo4j",            # Neo4j 用户名
    neo4j_password="your_password",    # Neo4j 密码
    qdrant_url="http://localhost:6333", # Qdrant 地址
    qdrant_api_key="your_api_key"      # Qdrant API Key
)
```

### RAG 配置 (rag_config.py)

```python
from hello_agents.config.rag_config import RAGConfig

config = RAGConfig(
    knowledge_base_path="./knowledge_base", # 知识库路径
    rag_namespace="rag",                    # RAG 命名空间
    chunk_size=500,                         # 分块大小
    chunk_overlap=50,                        # 分块重叠
    top_k=5,                                # 默认检索数量
    collection_name="hello_agents"          # Qdrant 集合名
)
```

## 目录结构

索引后的文档会保存在以下目录：

```
knowledge_base/
└── user_{user_id}/
    ├── documents/    # 原始文档
    ├── chunks/       # 分块数据
    └── markdown/      # 转换后的 Markdown
```

## 依赖

主要依赖：
- `sentence-transformers`: 文本嵌入模型
- `qdrant-client`: Qdrant 向量数据库客户端
- `neo4j`: Neo4j 图数据库驱动
- `markitdown`: 文档格式转换
- `pydantic`: 数据验证

详见 `requirements.txt`

## 运行测试

```bash
# 运行所有测试
pytest

# 运行特定测试
pytest tests/test_memory.py -v
```

## 许可证

MIT License

## 联系方式

如有问题或建议，请提交 Issue 或 Pull Request。
