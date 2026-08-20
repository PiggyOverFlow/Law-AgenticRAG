# LawRAG

一个面向中国法律场景的 Agentic-RAG 项目，支持法条检索、法律问答和法律文书生成。

## 项目简介

LawRAG 以法律法规知识库为基础，将法律文本切分、向量检索、检索重排、问答生成和文书生成串联起来，提供一套从法律知识检索到生成式应用的完整流程。

项目当前主要包含三部分能力：

- 法条检索：基于向量召回、规则重排和多轮 query 规划，返回更贴近法律适用语境的法条结果
- 法律问答：基于检索证据生成带引用的法律回答，并保留检索轨迹
- 文书生成：基于案件证据、法条结果和模板生成起诉书、答辩状等法律文书

## 核心思路

### 1. 法律知识库构建

- 将法律文本按“编-章-节-条”等层级结构切分
- 以“条”为核心检索单元，并保留父级路径、定位信息和上下文
- 为每个 chunk 生成可检索文本和结构化元数据

### 2. Agentic-RAG 检索

- 对用户问题先做 query rewrite、关键词抽取、法律要素抽取和争点拆解
- 基于多条 query 进行多轮召回
- 对候选结果按关键词命中、结构路径、优先级规则和 contrastive 信号重排

### 3. 问答与文书生成

- 问答场景：基于检索证据生成带法条引用的回答
- 文书场景：使用 StateGraph 驱动事实抽取、法条检索、证据筛选、模板检索和文书生成

### 4. 知识库更新

- 支持基于法律正文变化的增量索引更新
- 支持按案件时间过滤历史有效法条版本

## 项目结构

```text
LawRAG/
├── main.py                  # CLI 入口
├── bootstrap.yaml           # 配置文件
├── src/
│   ├── agent/               # 法律 Agent
│   ├── data/                # 数据加载与索引构建
│   ├── generation/          # 文书生成
│   ├── llm/                 # LLM 后端
│   ├── parsers/             # 证据解析
│   ├── rag/                 # chunk、检索、向量库
│   └── evaluation/          # 评估模块
├── docs/                    # 技术文档
```

## 环境准备

### 1. 安装依赖

```bash
pip install -r requirements.txt
```

### 2. 配置文件

项目默认使用根目录下的 `bootstrap.yaml`。

你至少需要确认以下配置：

- `dataset.base_path`：法律数据目录
- `dataset.sqlite_path`：法律数据库路径
- `rag.vector_db`：Milvus-lite 配置
- `llm`：远程模型或本地模型配置

如果你使用本地模型，需要正确配置：

- `llm.local_model_path`
- `llm.use_local_model`
- `llm.lora_adapter_path`（可选）

## 如何运行

### 1. 构建法律索引

增量构建：

```bash
python main.py build
```

只处理部分法律：

```bash
python main.py build --limit 100
```

只同步指定法律：

```bash
python main.py build --law-names "中华人民共和国民法典,中华人民共和国民事诉讼法"
```

全量重建：

```bash
python main.py build --full-refresh
```

### 2. 检索法条

```bash
python main.py search --query "债权转让后受让人能否直接起诉债务人" --top-k 5
```

按案件时间过滤法条版本：

```bash
python main.py search --query "民间借贷诉讼时效" --case-date 2021-06-01
```

### 3. 法律问答

```bash
python main.py ask --query "债权转让后受让人是否可以直接向债务人主张权利"
```

带案件时间的问答：

```bash
python main.py ask --query "2019年发生的民间借贷适用哪些法条" --case-date 2019-08-01
```

### 4. 生成法律文书

```bash
python main.py generate \
  --evidence "dataset/evident/sample1.txt,dataset/evident/sample2.txt" \
  --type "起诉书" \
  --request "要求被告返还借款本金及利息"
```

不保存文件，直接输出到控制台：

```bash
python main.py generate \
  --evidence "dataset/evident/sample1.txt" \
  --type "起诉书" \
  --no-save
```

支持的文书类型：

- `起诉书`
- `答辩状`
- `上诉状`
- `申请书`
- `代理词`

### 5. 评估生成结果

```bash
python main.py evaluate \
  --document output/generated.txt \
  --evidence dataset/evident/sample1.txt \
  --type 起诉书
```

## 说明

- 检索主链路当前以 dense retrieval 为主
- 文书生成依赖本地或远程 LLM 配置
- 多模态解析能力已预留，但当前最稳定的输入仍然是文本证据


## 注意

本项目生成的法律回答与法律文书仅用于研究、学习和系统验证，不构成正式法律意见。
