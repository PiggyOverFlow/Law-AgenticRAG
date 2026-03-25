#!/usr/bin/env python3
"""
LawRAG 系统使用示例
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from main import LawRAG


def example_build_index():
    """示例：构建向量索引"""
    print("=" * 60)
    print("示例 1: 构建向量索引")
    print("=" * 60)
    
    lawrag = LawRAG()
    
    print("开始构建索引（限制处理 10 部法律用于测试）...")
    lawrag.build_index(limit=10)
    
    stats = lawrag.get_index_stats()
    print(f"\n索引统计: {stats}")
    print()


def example_search_laws():
    """示例：检索法律法规"""
    print("=" * 60)
    print("示例 2: 检索法律法规")
    print("=" * 60)
    
    lawrag = LawRAG()
    
    query = "民间借贷纠纷"
    print(f"检索查询: {query}")
    
    results = lawrag.search_laws(query, top_k=3)
    
    print(f"\n找到 {len(results)} 条相关法条:")
    for idx, result in enumerate(results, 1):
        print(f"\n{idx}. {result['law_name']} {result['article_num']}")
        print(f"   相关度: {result['score']:.4f}")
        print(f"   内容: {result['content'][:150]}...")
    print()


def example_generate_document():
    """示例：生成法律文书"""
    print("=" * 60)
    print("示例 3: 生成法律文书")
    print("=" * 60)
    
    lawrag = LawRAG()
    
    from src.models import CaseFacts, LegalEvent
    from datetime import datetime
    
    case_facts = CaseFacts(
        events=[
            LegalEvent(
                evident_type="文本",
                time=datetime(2024, 1, 15),
                place="北京市朝阳区",
                cause="民间借贷",
                process="被告于2024年1月15日向原告借款人民币10万元，约定月利率2%，借款期限为3个月",
                result="借款到期后，被告未按约定归还借款本金及利息",
                source_file="evidence.txt"
            )
        ],
        evidence_summary="原告提供了借款合同、转账记录等证据材料",
        key_disputes=["借款金额是否属实", "利息约定是否合法"]
    )
    
    print("案件事实:")
    print(f"  事件数量: {len(case_facts.events)}")
    print(f"  证据摘要: {case_facts.evidence_summary}")
    print(f"  核心争议: {', '.join(case_facts.key_disputes)}")
    print()
    
    print("开始生成起诉书...")
    
    try:
        content = lawrag.generate_document(
            case_facts=case_facts,
            document_type="起诉书",
            user_request="要求被告偿还借款本金10万元及利息",
            save=False
        )
        
        print("\n生成的文书内容:")
        print("-" * 60)
        print(content)
        print("-" * 60)
    except Exception as e:
        print(f"生成文书时出错: {e}")
        print("请确保向量索引已构建完成")
    
    print()


def example_evaluate_document():
    """示例：评估文书质量"""
    print("=" * 60)
    print("示例 4: 评估文书质量")
    print("=" * 60)
    
    lawrag = LawRAG()
    
    generated_doc = """
# 起诉书

原告：张三
被告：李四

## 诉讼请求
1. 判令被告偿还借款本金10万元
2. 判令被告支付利息

## 事实与理由
被告于2024年1月15日向原告借款10万元，约定月利率2%，借款期限为3个月。借款到期后，被告未按约定归还借款本金及利息。
"""
    
    original_evidence = "借款合同、转账记录"
    
    print("生成的文书:")
    print(generated_doc)
    print()
    
    print("原始证据:")
    print(original_evidence)
    print()
    
    print("开始评估文书...")
    
    result = lawrag.evaluate_document(
        generated_document=generated_doc,
        original_evidence=original_evidence,
        document_type="起诉书"
    )
    
    print("\n评估结果:")
    print(f"  文书类型: {result['document_type']}")
    print(f"  评估时间: {result['evaluated_at']}")
    
    if 'llm_judge' in result:
        llm_judge = result['llm_judge']
        print(f"\n  LLM 评判:")
        print(f"    总体评分: {llm_judge['overall_score']:.2f}")
        print(f"    一致性评分: {llm_judge['consistency_score']:.2f}")
        print(f"    逻辑评分: {llm_judge['logic_score']:.2f}")
    
    print()


def main():
    """运行所有示例"""
    print("\n" + "=" * 60)
    print("LawRAG 智能法律文书生成系统 - 使用示例")
    print("=" * 60 + "\n")
    
    examples = [
        ("构建向量索引", example_build_index),
        ("检索法律法规", example_search_laws),
        ("生成法律文书", example_generate_document),
        ("评估文书质量", example_evaluate_document),
    ]
    
    print("可用示例:")
    for idx, (name, _) in enumerate(examples, 1):
        print(f"  {idx}. {name}")
    print(f"  0. 运行所有示例")
    print()
    
    choice = input("请选择要运行的示例 (0-4): ").strip()
    
    try:
        choice_num = int(choice)
        
        if choice_num == 0:
            for _, func in examples:
                func()
        elif 1 <= choice_num <= len(examples):
            examples[choice_num - 1][1]()
        else:
            print("无效的选择")
    except ValueError:
        print("请输入有效的数字")
    
    print("\n" + "=" * 60)
    print("示例运行完成")
    print("=" * 60 + "\n")


if __name__ == "__main__":
    main()