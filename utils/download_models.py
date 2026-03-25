#!/usr/bin/env python3
"""
模型下载脚本
将 BGE 模型从 Hugging Face 下载并保存到本地目录
使用国内镜像源加速下载
"""

import os
import sys
from pathlib import Path
import argparse


def setup_mirror():
    """设置国内镜像源"""
    mirror_url = "https://hf-mirror.com"
    os.environ["HF_ENDPOINT"] = mirror_url
    print(f"使用国内镜像源: {mirror_url}")
    print()


def download_model(model_name: str, local_path: str):
    """下载模型到本地"""
    print(f"正在下载模型: {model_name}")
    print(f"保存到: {local_path}")
    
    try:
        from transformers import AutoTokenizer, AutoModel, AutoModelForSequenceClassification
        
        local_path_obj = Path(local_path)
        local_path_obj.mkdir(parents=True, exist_ok=True)
        
        print("下载 tokenizer...")
        tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
        tokenizer.save_pretrained(local_path)
        print(f"✓ Tokenizer 保存成功")
        
        print("下载模型...")
        
        if "reranker" in model_name.lower() or "cross-encoder" in model_name.lower():
            model = AutoModelForSequenceClassification.from_pretrained(model_name, trust_remote_code=True)
        else:
            model = AutoModel.from_pretrained(model_name, trust_remote_code=True)
        
        model.save_pretrained(local_path)
        print(f"✓ 模型保存成功")
        
        print(f"\n✓ 模型 {model_name} 下载完成！")
        print(f"  保存位置: {local_path}")
        print(f"  文件数量: {len(list(local_path_obj.iterdir()))}")
        
        return True
        
    except Exception as e:
        print(f"\n✗ 下载失败: {e}")
        import traceback
        traceback.print_exc()
        return False


def download_embedding_model(local_path: str = "/app/model/bge-m3"):
    """下载嵌入模型"""
    model_name = "BAAI/bge-m3"
    print("=" * 60)
    print("下载 BGE-M3 嵌入模型")
    print("=" * 60)
    return download_model(model_name, local_path)


def download_reranker_model(local_path: str = "/app/model/bge-reranker-v2-m3"):
    """下载重排序模型"""
    model_name = "BAAI/bge-reranker-v2-m3"
    print("=" * 60)
    print("下载 BGE-Reranker-v2-m3 重排序模型")
    print("=" * 60)
    return download_model(model_name, local_path)


def download_all_models(base_path: str = "/app/model"):
    """下载所有模型"""
    print("\n" + "=" * 60)
    print("下载所有 BGE 模型")
    print("=" * 60 + "\n")
    
    results = []
    
    results.append(("嵌入模型", download_embedding_model(f"{base_path}/bge-m3")))
    print()
    results.append(("重排序模型", download_reranker_model(f"{base_path}/bge-reranker-v2-m3")))
    
    print("\n" + "=" * 60)
    print("下载结果汇总")
    print("=" * 60)
    
    for name, success in results:
        status = "✓ 成功" if success else "✗ 失败"
        print(f"{name:20s} {status}")
    
    all_success = all(success for _, success in results)
    
    if all_success:
        print("\n🎉 所有模型下载完成！")
        print(f"模型保存位置: {base_path}")
        return 0
    else:
        print("\n⚠️  部分模型下载失败，请检查错误信息")
        return 1


def main():
    parser = argparse.ArgumentParser(description="下载 BGE 模型到本地")
    
    parser.add_argument(
        "--model",
        choices=["embedding", "reranker", "all"],
        default="all",
        help="要下载的模型类型"
    )
    
    parser.add_argument(
        "--path",
        default="/app/model",
        help="模型保存路径"
    )
    
    parser.add_argument(
        "--embedding-path",
        default=None,
        help="嵌入模型保存路径（覆盖默认）"
    )
    
    parser.add_argument(
        "--reranker-path",
        default=None,
        help="重排序模型保存路径（覆盖默认）"
    )
    
    parser.add_argument(
        "--no-mirror",
        action="store_true",
        help="不使用国内镜像源"
    )
    
    args = parser.parse_args()
    
    if not args.no_mirror:
        setup_mirror()
    
    if args.model == "embedding":
        path = args.embedding_path or f"{args.path}/bge-m3"
        success = download_embedding_model(path)
        return 0 if success else 1
    
    elif args.model == "reranker":
        path = args.reranker_path or f"{args.path}/bge-reranker-v2-m3"
        success = download_reranker_model(path)
        return 0 if success else 1
    
    else:
        return download_all_models(args.path)


if __name__ == "__main__":
    sys.exit(main())