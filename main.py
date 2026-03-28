import logging
from pathlib import Path
from typing import List, Optional

from config import get_config
from src.models import CaseFacts
from src.parsers import MultimodalParser
from src.rag import LegalRAG
from src.agent import LegalAgent
from src.generation import DocumentGenerator
from src.data import LawIndexBuilder
from src.evaluation import EvaluationFramework


class LawRAG:
    def __init__(self, config_path: str = "bootstrap.yaml"):
        self.config = get_config(config_path)
        self._setup_logging()
        
        self.parser = MultimodalParser()
        self.rag = LegalRAG()
        self.agent = LegalAgent()
        self.agent.set_rag(self.rag)
        self.document_generator = DocumentGenerator()
        self.document_generator.agent.set_rag(self.rag)
        self.evaluator = EvaluationFramework()
        
        logger = logging.getLogger(__name__)
        logger.info("LawRAG 系统初始化完成")

    def _setup_logging(self):
        log_config = self.config.logging
        log_file = Path(log_config.file)
        log_file.parent.mkdir(parents=True, exist_ok=True)
        
        logging.basicConfig(
            level=getattr(logging, log_config.level),
            format=log_config.format,
            handlers=[
                logging.FileHandler(log_file, encoding='utf-8'),
                logging.StreamHandler()
            ]
        )

    def build_index(self, limit: Optional[int] = None):
        index_builder = LawIndexBuilder()
        index_builder.build_full_index(limit=limit)

    def parse_evidence(self, file_paths: List[str]) -> CaseFacts:
        return self.parser.parse_case_facts(file_paths)

    def generate_document(
        self,
        case_facts: CaseFacts,
        document_type: str,
        user_request: Optional[str] = None,
        save: bool = True,
        filename: Optional[str] = None
    ) -> str:
        if save:
            return self.document_generator.generate_and_save(
                case_facts,
                document_type,
                user_request,
                filename
            )
        else:
            result = self.document_generator.generate_document(
                case_facts,
                document_type,
                user_request
            )
            return result["content"]

    def search_laws(self, query: str, top_k: int = 5) -> List[dict]:
        results = self.rag.search(query)
        return [
            {
                "law_name": r.chunk.law_name,
                "article_num": r.chunk.article_num,
                "content": r.chunk.content,
                "score": r.score
            }
            for r in results[:top_k]
        ]

    def evaluate_document(
        self,
        generated_document: str,
        original_evidence: str,
        document_type: str,
        reference_document: Optional[str] = None
    ) -> dict:
        return self.evaluator.evaluate_document(
            generated_document,
            original_evidence,
            document_type,
            reference_document
        )

    def get_index_stats(self) -> dict:
        return self.rag.get_collection_stats()

    def reset_index(self):
        self.rag.reset_index()


def main():
    import argparse
    
    parser = argparse.ArgumentParser(description="智能法律文书生成系统")
    parser.add_argument("--config", default="bootstrap.yaml", help="配置文件路径")
    
    subparsers = parser.add_subparsers(dest="command", help="可用命令")
    
    build_parser = subparsers.add_parser("build", help="构建向量索引")
    build_parser.add_argument("--limit", type=int, help="限制处理的法律数量")
    
    generate_parser = subparsers.add_parser("generate", help="生成法律文书")
    generate_parser.add_argument("--evidence", required=True, help="证据文件路径（多个文件用逗号分隔）")
    generate_parser.add_argument("--type", required=True, choices=["起诉书", "答辩状", "上诉状", "申请书", "代理词"], help="文书类型")
    generate_parser.add_argument("--request", help="用户需求")
    generate_parser.add_argument("--output", help="输出文件名")
    generate_parser.add_argument("--no-save", action="store_true", help="不保存文件")
    
    search_parser = subparsers.add_parser("search", help="检索法律法规")
    search_parser.add_argument("--query", required=True, help="检索查询")
    search_parser.add_argument("--top-k", type=int, default=5, help="返回结果数量")
    
    evaluate_parser = subparsers.add_parser("evaluate", help="评估文书质量")
    evaluate_parser.add_argument("--document", required=True, help="生成的文书路径")
    evaluate_parser.add_argument("--evidence", required=True, help="原始证据路径")
    evaluate_parser.add_argument("--type", required=True, help="文书类型")
    evaluate_parser.add_argument("--reference", help="参考文书路径")
    
    args = parser.parse_args()
    
    if not args.command:
        parser.print_help()
        return
    
    lawrag = LawRAG(args.config)
    
    if args.command == "build":
        lawrag.build_index(limit=args.limit)
    
    elif args.command == "generate":
        evidence_files = []
        for raw_path in args.evidence.split(","):
            path_str = raw_path.strip()
            if not path_str:
                continue

            input_path = Path(path_str)
            if input_path.exists():
                evidence_files.append(str(input_path))
                continue

            fallback_path = Path("dataset/evident") / path_str
            if fallback_path.exists():
                evidence_files.append(str(fallback_path))
                continue

            raise FileNotFoundError(
                f"证据文件不存在: {path_str}。请使用相对项目根目录的路径，"
                f"例如 dataset/evident/{path_str}"
            )

        if not evidence_files:
            raise ValueError("--evidence 不能为空，请至少提供一个证据文件路径")

        case_facts = lawrag.parse_evidence(evidence_files)
        
        content = lawrag.generate_document(
            case_facts,
            args.type,
            args.request,
            save=not args.no_save,
            filename=args.output
        )
        
        if args.no_save:
            print(content)
    
    elif args.command == "search":
        results = lawrag.search_laws(args.query, args.top_k)
        
        for idx, result in enumerate(results, 1):
            print(f"\n{idx}. {result['law_name']} {result['article_num']}")
            print(f"   相关度: {result['score']:.4f}")
            print(f"   内容: {result['content'][:200]}...")
    
    elif args.command == "evaluate":
        with open(args.document, "r", encoding="utf-8") as f:
            generated_doc = f.read()
        
        with open(args.evidence, "r", encoding="utf-8") as f:
            original_evidence = f.read()
        
        reference_doc = None
        if args.reference:
            with open(args.reference, "r", encoding="utf-8") as f:
                reference_doc = f.read()
        
        result = lawrag.evaluate_document(
            generated_doc,
            original_evidence,
            args.type,
            reference_doc
        )
        
        print("评估结果:")
        print(result)


if __name__ == "__main__":
    main()