# 智能法律文书生成 Agent 平台

基于 RAG 技术和多模态解析的智能法律文书自动生成系统。

## 系统概述

本系统通过 Agent 架构结合 RAG 技术，打造一个多模态输入的法律文书自动生成引擎。系统能够解析用户提供的多元证据材料（文本、图像、音视频），将其结构化为案件事实，并基于本地化的法律法规知识库进行精准检索，最终自动生成符合法定格式要求的法律文书。

## 核心功能

### 1. 多模态证据解析
- **文本解析**：支持 txt、md、doc、docx、pdf 等格式
- **图像解析**：OCR 提取文字 + 场景描述
- **音频解析**：语音转文字 + 时间戳提取
- **视频解析**：抽帧分析 + 音频提取

### 2. 案件事实结构化
- 使用 Pydantic 定义强类型数据结构
- LLM Function Calling 强制输出 JSON
- 提取时间、地点、起因、经过、结果等关键信息

### 3. 法律知识库与 RAG 检索
- **智能切分**：按"编-章-节-条-款-项-目"层级切分
- **Metadata 注入**：包含法条名称、编号、适用范围、生效时间等
- **混合检索**：向量检索 + Metadata 过滤 + 重排序
- **时间过滤**：根据事件时间过滤已废止法条

### 4. Agent 推理与文书生成
- **ReAct 范式**：思考-行动-观察-生成
- **工具调用**：RAG 检索、模板检索、事实提取
- **文书类型**：起诉书、答辩状、上诉状、申请书、代理词

### 5. 评估体系
- **专家评审**：5 维度评分（格式、完整性、逻辑、准确性、法条适用性）
- **LLM 评判**：一致性检验 + 逻辑连贯性检验

## 技术栈

- **LLM/VLM**：Qwen-Max / Qwen-VL
- **ASR**：Whisper / SenseVoice
- **Embedding**：BGE-m3
- **Reranker**：BGE-Reranker-v2
- **向量数据库**：Milvus / Qdrant
- **开发框架**：Python + PyTorch + LangChain

## 安装

### 环境要求
- Python 3.8+
- CUDA 11.0+（如使用 GPU）

### 安装依赖

```bash
pip install -r requirements.txt
```

### 配置

1. 复制配置文件模板：
```bash
cp bootstrap.yaml.example bootstrap.yaml
```

2. 编辑 `bootstrap.yaml`，配置各项参数

3. 设置环境变量：
```bash
export QWEN_API_KEY="your_api_key_here"
```

## 使用方法

### 1. 构建向量索引

首次使用需要构建法律法规向量索引：

```bash
python main.py build --limit 100  # 限制处理 100 部法律（测试用）
```

不使用 limit 参数将处理所有法律：

```bash
python main.py build
```

### 2. 生成法律文书

```bash
python main.py generate \
    --evidence "evidence1.txt,evidence2.jpg,audio.mp3" \
    --type "起诉书" \
    --request "要求被告偿还借款本金及利息" \
    --output "complaint_20240101.txt"
```

参数说明：
- `--evidence`：证据文件路径，多个文件用逗号分隔
- `--type`：文书类型（起诉书/答辩状/上诉状/申请书/代理词）
- `--request`：用户需求（可选）
- `--output`：输出文件名（可选）
- `--no-save`：不保存文件，直接输出到控制台

### 3. 检索法律法规

```bash
python main.py search --query "民间借贷纠纷" --top-k 5
```

### 4. 评估文书质量

```bash
python main.py evaluate \
    --document "generated_complaint.txt" \
    --evidence "original_evidence.txt" \
    --type "起诉书" \
    --reference "reference_complaint.txt"
```

## 项目结构

```
LawRAG/
├── bootstrap.yaml              # 配置文件
├── config.py                   # 配置管理
├── main.py                     # 主入口
├── requirements.txt            # 依赖包
├── src/
│   ├── models/                 # 数据模型
│   │   ├── legal_event.py     # 法律事件模型
│   │   └── __init__.py
│   ├── parsers/                # 多模态解析
│   │   ├── multimodal_parser.py
│   │   └── __init__.py
│   ├── rag/                    # RAG 检索
│   │   ├── chunker.py         # 文本切分
│   │   ├── vector_db.py       # 向量数据库
│   │   ├── retriever.py       # 检索器
│   │   └── __init__.py
│   ├── agent/                  # Agent 推理
│   │   ├── legal_agent.py
│   │   └── __init__.py
│   ├── data/                   # 数据加载
│   │   ├── dataset_loader.py
│   │   └── __init__.py
│   ├── generation/             # 文书生成
│   │   ├── document_generator.py
│   │   └── __init__.py
│   └── evaluation/            # 评估模块
│       ├── evaluator.py
│       └── __init__.py
├── dataset/
│   └── Laws-master/            # 法律法规数据集
├── templates/                  # 文书模板
├── output/                     # 输出目录
│   └── documents/
├── logs/                       # 日志目录
└── docs/                       # 文档
    └── 项目整体的架构.md
```

## 配置说明

### LLM 配置
```yaml
llm:
  primary_model: "qwen-max"
  vision_model: "qwen-vl"
  api_key: "${QWEN_API_KEY}"
  base_url: "https://dashscope.aliyuncs.com/compatible-mode/v1"
```

### RAG 配置
```yaml
rag:
  embedding_model: "BAAI/bge-m3"
  reranker_model: "BAAI/bge-reranker-v2-m3"
  vector_db:
    type: "milvus"  # 或 "qdrant"
    host: "localhost"
    port: 19530
```

### 数据集配置
```yaml
dataset:
  base_path: "./dataset/Laws-master"
  sqlite_path: "./dataset/Laws-master/DLC/db.sqlite3"
```

## 开发指南

### 添加新的文书类型

1. 在 `templates/` 目录下添加对应的模板文件
2. 在 `bootstrap.yaml` 的 `document.supported_types` 中添加类型
3. 在 `DocumentGenerator` 中添加生成逻辑

### 自定义解析器

继承 `MultimodalParser` 类并重写相应方法：

```python
from src.parsers import MultimodalParser

class CustomParser(MultimodalParser):
    def parse_text(self, file_path: str):
        # 自定义文本解析逻辑
        pass
```

### 扩展 Agent 工具

创建新的 Tool 类：

```python
from src.agent import Tool

class CustomTool(Tool):
    def __init__(self):
        super().__init__(
            name="custom_tool",
            description="工具描述",
            func=self._custom_func
        )
    
    def _custom_func(self, **kwargs):
        # 工具实现
        pass
```

## 评估体系

### 专家评审标准

| 维度 | 评分标准 | 法学考量点 |
|------|----------|------------|
| 格式正确性 | 0-5分 | 首部、正文、尾部结构是否符合规范 |
| 内容完整性 | 0-5分 | 诉讼请求、事实依据、法律依据是否齐备 |
| 逻辑条理性 | 0-5分 | 是否遵循"三段论"逻辑 |
| 内容准确性 | 0-5分 | 证据转化的客观性，无主观臆断 |
| 法条适用性 | 0-5分 | 法条是否切中要害，是否存在适用冲突 |

### LLM 评判

- **一致性检验**：对比生成事实与原始证据的偏差
- **逻辑连贯性检验**：检验法律逻辑的连贯性

## 性能优化

1. **缓存机制**：启用 Redis 缓存减少重复计算
2. **并发处理**：调整 `max_concurrent_requests` 提升吞吐量
3. **批量索引**：使用 `--limit` 参数分批构建索引
4. **向量检索**：调整 `top_k_initial` 和 `top_k_final` 平衡精度和速度

## 常见问题

### Q: 向量数据库连接失败？
A: 检查 Milvus/Qdrant 服务是否启动，配置中的 host 和 port 是否正确。

### Q: LLM 调用失败？
A: 确认 API Key 已正确设置，检查网络连接和 API 配额。

### Q: 索引构建很慢？
A: 使用 `--limit` 参数限制处理数量，或使用 GPU 加速嵌入计算。

### Q: 生成的文书质量不理想？
A: 
1. 检查证据材料是否完整清晰
2. 调整 LLM 的 temperature 参数
3. 使用评估系统分析问题所在

## 贡献指南

欢迎贡献代码、报告问题或提出建议！

1. Fork 本仓库
2. 创建特性分支 (`git checkout -b feature/AmazingFeature`)
3. 提交更改 (`git commit -m 'Add some AmazingFeature'`)
4. 推送到分支 (`git push origin feature/AmazingFeature`)
5. 开启 Pull Request

## 许可证

本项目采用 MIT 许可证。详见 LICENSE 文件。

## 联系方式

- 项目主页：[GitHub Repository]
- 问题反馈：[Issues]
- 邮箱：[Email]

## 致谢

感谢以下开源项目的支持：
- Qwen 系列 LLM
- BGE 系列嵌入模型
- Milvus / Qdrant 向量数据库
- LangChain 框架

---

**注意**：本系统生成的法律文书仅供参考，实际使用前请务必由专业律师审核。