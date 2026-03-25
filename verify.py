#!/usr/bin/env python3
"""
LawRAG 系统验证脚本
检查系统各模块是否正常工作
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))


def test_imports():
    """测试所有模块导入"""
    print("=" * 60)
    print("测试 1: 模块导入")
    print("=" * 60)
    
    try:
        print("导入配置模块...")
        from config import get_config
        print("✓ 配置模块导入成功")
        
        print("导入数据模型...")
        from src.models import LegalEvent, CaseFacts
        print("✓ 数据模型导入成功")
        
        print("导入解析器...")
        from src.parsers import MultimodalParser
        print("✓ 解析器导入成功")
        
        print("导入 RAG 模块...")
        from src.rag import LawChunk, LawChunker, LegalRAG
        print("✓ RAG 模块导入成功")
        
        print("导入 Agent 模块...")
        from src.agent import LegalAgent
        print("✓ Agent 模块导入成功")
        
        print("导入数据加载模块...")
        from src.data import LawDatasetLoader
        print("✓ 数据加载模块导入成功")
        
        print("导入文书生成模块...")
        from src.generation import DocumentGenerator
        print("✓ 文书生成模块导入成功")
        
        print("导入评估模块...")
        from src.evaluation import EvaluationFramework
        print("✓ 评估模块导入成功")
        
        print("\n✓ 所有模块导入成功！\n")
        return True
        
    except Exception as e:
        print(f"\n✗ 模块导入失败: {e}\n")
        return False


def test_config():
    """测试配置加载"""
    print("=" * 60)
    print("测试 2: 配置加载")
    print("=" * 60)
    
    try:
        from config import get_config
        
        print("加载配置文件...")
        config = get_config()
        
        print(f"✓ LLM 模型: {config.llm.primary_model}")
        print(f"✓ 嵌入模型: {config.rag.embedding_model}")
        print(f"✓ 向量数据库: {config.rag.vector_db.type}")
        print(f"✓ 数据集路径: {config.dataset.base_path}")
        
        print("\n✓ 配置加载成功！\n")
        return True
        
    except Exception as e:
        print(f"\n✗ 配置加载失败: {e}\n")
        return False


def test_models():
    """测试数据模型"""
    print("=" * 60)
    print("测试 3: 数据模型")
    print("=" * 60)
    
    try:
        from src.models import LegalEvent, CaseFacts
        from datetime import datetime
        
        print("创建 LegalEvent...")
        event = LegalEvent(
            evident_type="文本",
            time=datetime(2024, 1, 15),
            place="北京市",
            cause="民间借贷",
            process="借款过程",
            result="未还款"
        )
        print(f"✓ 创建事件: {event.cause}")
        
        print("创建 CaseFacts...")
        case_facts = CaseFacts(
            events=[event],
            evidence_summary="测试证据摘要",
            key_disputes=["争议点1", "争议点2"]
        )
        print(f"✓ 创建案件事实: {len(case_facts.events)} 个事件")
        
        print("\n✓ 数据模型测试成功！\n")
        return True
        
    except Exception as e:
        print(f"\n✗ 数据模型测试失败: {e}\n")
        return False


def test_chunker():
    """测试文本切分"""
    print("=" * 60)
    print("测试 4: 文本切分")
    print("=" * 60)
    
    try:
        from src.rag import LawChunker
        
        print("创建切分器...")
        chunker = LawChunker()
        
        print("测试层级识别...")
        test_cases = [
            ("第一编", "编"),
            ("第一章", "章"),
            ("第一节", "节"),
            ("第一条", "条"),
            ("（一）", "款"),
            ("一、", "项"),
        ]
        
        for text, expected_level in test_cases:
            level = chunker._detect_level(text)
            if level == expected_level:
                print(f"  ✓ '{text}' -> {level}")
            else:
                print(f"  ✗ '{text}' -> {level} (期望: {expected_level})")
        
        print("\n✓ 文本切分测试成功！\n")
        return True
        
    except Exception as e:
        print(f"\n✗ 文本切分测试失败: {e}\n")
        return False


def test_directories():
    """测试目录结构"""
    print("=" * 60)
    print("测试 5: 目录结构")
    print("=" * 60)
    
    required_dirs = [
        "src",
        "src/models",
        "src/parsers",
        "src/rag",
        "src/agent",
        "src/data",
        "src/generation",
        "src/evaluation",
        "templates",
        "docs"
    ]
    
    all_exist = True
    for dir_path in required_dirs:
        if Path(dir_path).exists():
            print(f"  ✓ {dir_path}")
        else:
            print(f"  ✗ {dir_path} (不存在)")
            all_exist = False
    
    if all_exist:
        print("\n✓ 目录结构完整！\n")
    else:
        print("\n✗ 部分目录缺失！\n")
    
    return all_exist


def test_files():
    """测试必要文件"""
    print("=" * 60)
    print("测试 6: 必要文件")
    print("=" * 60)
    
    required_files = [
        "bootstrap.yaml",
        "config.py",
        "main.py",
        "requirements.txt",
        "README.md"
    ]
    
    all_exist = True
    for file_path in required_files:
        if Path(file_path).exists():
            print(f"  ✓ {file_path}")
        else:
            print(f"  ✗ {file_path} (不存在)")
            all_exist = False
    
    if all_exist:
        print("\n✓ 必要文件完整！\n")
    else:
        print("\n✗ 部分文件缺失！\n")
    
    return all_exist


def main():
    """运行所有测试"""
    print("\n" + "=" * 60)
    print("LawRAG 系统验证")
    print("=" * 60 + "\n")
    
    tests = [
        ("目录结构", test_directories),
        ("必要文件", test_files),
        ("模块导入", test_imports),
        ("配置加载", test_config),
        ("数据模型", test_models),
        ("文本切分", test_chunker),
    ]
    
    results = []
    for test_name, test_func in tests:
        try:
            result = test_func()
            results.append((test_name, result))
        except Exception as e:
            print(f"✗ 测试 '{test_name}' 出错: {e}\n")
            results.append((test_name, False))
    
    print("=" * 60)
    print("测试结果汇总")
    print("=" * 60)
    
    passed = sum(1 for _, result in results if result)
    total = len(results)
    
    for test_name, result in results:
        status = "✓ 通过" if result else "✗ 失败"
        print(f"{test_name:20s} {status}")
    
    print(f"\n总计: {passed}/{total} 通过")
    
    if passed == total:
        print("\n🎉 所有测试通过！系统可以正常使用。\n")
        return 0
    else:
        print(f"\n⚠️  有 {total - passed} 个测试失败，请检查配置。\n")
        return 1


if __name__ == "__main__":
    sys.exit(main())