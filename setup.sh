#!/bin/bash

# LawRAG 智能法律文书生成系统 - 快速启动脚本

set -e

echo "=========================================="
echo "LawRAG 智能法律文书生成系统"
echo "=========================================="
echo ""

# 检查 Python 版本
echo "检查 Python 版本..."
python_version=$(python3 --version 2>&1 | awk '{print $2}')
echo "Python 版本: $python_version"
echo ""

# 检查虚拟环境
if [ ! -d "venv" ]; then
    echo "创建虚拟环境..."
    python3 -m venv venv
fi

# 激活虚拟环境
echo "激活虚拟环境..."
source venv/bin/activate
echo ""

# 安装依赖
echo "安装依赖包..."
pip install --upgrade pip
pip install -r requirements.txt
echo ""

# 检查配置文件
if [ ! -f "bootstrap.yaml" ]; then
    echo "警告: bootstrap.yaml 不存在"
    echo "请创建配置文件并设置必要的环境变量"
    echo ""
    echo "示例配置:"
    echo "  export QWEN_API_KEY='your_api_key_here'"
    echo ""
fi

# 创建必要的目录
echo "创建必要的目录..."
mkdir -p output/documents
mkdir -p logs
mkdir -p templates
mkdir -p data
echo ""

# 检查向量数据库
echo "检查向量数据库服务..."
if ! command -v milvus &> /dev/null; then
    echo "提示: Milvus 未安装"
    echo "请参考文档安装 Milvus 或使用 Qdrant"
    echo ""
fi

echo "=========================================="
echo "初始化完成！"
echo "=========================================="
echo ""
echo "下一步操作:"
echo "1. 设置环境变量: export QWEN_API_KEY='your_api_key'"
echo "2. 构建向量索引: python main.py build --limit 100"
echo "3. 运行示例: python example.py"
echo "4. 查看帮助: python main.py --help"
echo ""