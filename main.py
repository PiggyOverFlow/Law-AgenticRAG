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
    """智能法律文书生成系统主类，整合各功能模块"""
    
    def __init__(self, config_path: str = "bootstrap.yaml"):
        """初始化 LawRAG 系统
        
        Args:
            config_path: 配置文件路径
        """
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
        """配置日志系统"""
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

    def build_index(
        self,
        limit: Optional[int] = None,
        incremental: bool = True,
        law_names: Optional[List[str]] = None,
        full_refresh: bool = False,
    ):
        """构建法律法规向量索引
        
        Args:
            limit: 限制处理的法律数量，None 表示处理全部
        """
        index_builder = LawIndexBuilder()
        if full_refresh:
            return index_builder.rebuild_index(limit=limit)
        if incremental:
            return index_builder.build_incremental_index(law_names=law_names, limit=limit)
        return index_builder.build_full_index(limit=limit)

    def parse_evidence(self, file_paths: List[str]) -> CaseFacts:
        """解析证据文件，提取案件事实
        
        Args:
            file_paths: 证据文件路径列表
            
        Returns:
            CaseFacts: 案件事实对象
        """
        return self.parser.parse_case_facts(file_paths)

    def generate_document(
        self,
        case_facts: CaseFacts,
        document_type: str,
        user_request: Optional[str] = None,
        save: bool = True,
        filename: Optional[str] = None
    ) -> str:
        """生成法律文书
        
        Args:
            case_facts: 案件事实
            document_type: 文书类型（起诉书、答辩状、上诉状、申请书、代理词）
            user_request: 用户需求
            save: 是否保存文件
            filename: 输出文件名
            
        Returns:
            str: 生成的文书内容或文件路径
        """
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

    def search_laws(
        self,
        query: str,
        top_k: int = 5,
        case_date: Optional[str] = None,
    ) -> List[dict]:
        """检索相关法律法规
        
        Args:
            query: 检索查询
            top_k: 返回结果数量
            
        Returns:
            List[dict]: 检索结果列表
        """
        filters = {"case_date": case_date} if case_date else None
        results = self.rag.search(query, filters=filters)
        return [
            {
                "law_name": r.chunk.law_name,
                "article_num": r.chunk.article_num,
                "content": r.chunk.content,
                "score": r.score,
                "distance": r.score,
                "rerank_score": r.chunk.metadata.get("rerank_score", r.score),
                "path_match_score": r.chunk.metadata.get("path_match_score", 0.0),
                "legal_priority_score": r.chunk.metadata.get("legal_priority_score", 0.0),
                "legal_priority_reasons": r.chunk.metadata.get("legal_priority_reasons", []),
                "locator": r.chunk.metadata.get("structure_locator", ""),
                "hierarchy_path": r.chunk.metadata.get("structure_hierarchy_path", ""),
                "keyword_hits": r.chunk.metadata.get("keyword_hits", []),
                "path_hits": r.chunk.metadata.get("path_hits", []),
                "path_focus_hits": r.chunk.metadata.get("path_focus_hits", []),
                "effective_keywords": r.chunk.metadata.get("effective_keywords", []),
                "keyword_idf": r.chunk.metadata.get("keyword_idf", {}),
                "effective_date": r.chunk.metadata.get("effective_date", ""),
                "repeal_date": r.chunk.metadata.get("repeal_date", ""),
                "version_id": r.chunk.metadata.get("version_id", ""),
                "is_current": r.chunk.metadata.get("is_current", True),
            }
            for r in results[:top_k]
        ]

    def answer_query(self, query: str, case_date: Optional[str] = None) -> dict:
        """基于法条证据回答用户问题并给出引用
        
        Args:
            query: 用户问题
            
        Returns:
            dict: 包含回答、引用和检索轨迹的结果
        """
        filters = {"case_date": case_date} if case_date else None
        return self.rag.answer_with_citations(query, filters=filters)

    def evaluate_document(
        self,
        generated_document: str,
        original_evidence: str,
        document_type: str,
        reference_document: Optional[str] = None
    ) -> dict:
        """评估文书质量
        
        Args:
            generated_document: 生成的文书内容
            original_evidence: 原始证据内容
            document_type: 文书类型
            reference_document: 参考文书内容（可选）
            
        Returns:
            dict: 评估结果
        """
        return self.evaluator.evaluate_document(
            generated_document,
            original_evidence,
            document_type,
            reference_document
        )

    def get_index_stats(self) -> dict:
        """获取索引统计信息
        
        Returns:
            dict: 索引统计数据
        """
        return self.rag.get_collection_stats()

    def reset_index(self):
        """重置向量索引"""
        self.rag.reset_index()


def main():
    """命令行入口函数"""
    import argparse
    
    parser = argparse.ArgumentParser(description="智能法律文书生成系统")
    parser.add_argument("--config", default="bootstrap.yaml", help="配置文件路径")
    
    subparsers = parser.add_subparsers(dest="command", help="可用命令")
    
    build_parser = subparsers.add_parser("build", help="构建向量索引")
    build_parser.add_argument("--limit", type=int, help="限制处理的法律数量")
    build_parser.add_argument("--law-names", help="仅同步指定法律名称，多个名称用英文逗号分隔")
    build_parser.add_argument("--full-refresh", action="store_true", help="删除现有索引后全量重建")
    build_parser.add_argument("--full", action="store_true", help="执行全量构建但不先清空索引")
    
    generate_parser = subparsers.add_parser("generate", help="生成法律文书")
    generate_parser.add_argument("--evidence", required=True, help="证据文件路径（多个文件用逗号分隔）")
    generate_parser.add_argument("--type", required=True, choices=["起诉书", "答辩状", "上诉状", "申请书", "代理词"], help="文书类型")
    generate_parser.add_argument("--request", help="用户需求")
    generate_parser.add_argument("--output", help="输出文件名")
    generate_parser.add_argument("--no-save", action="store_true", help="不保存文件")
    
    search_parser = subparsers.add_parser("search", help="检索法律法规")
    search_parser.add_argument("--query", required=True, help="检索查询")
    search_parser.add_argument("--top-k", type=int, default=5, help="返回结果数量")
    search_parser.add_argument("--case-date", help="案件日期，按该日期过滤有效法条，格式 YYYY-MM-DD")

    ask_parser = subparsers.add_parser("ask", help="基于法条证据回答并给出引用")
    ask_parser.add_argument("--query", required=True, help="用户问题")
    ask_parser.add_argument("--case-date", help="案件日期，按该日期过滤有效法条，格式 YYYY-MM-DD")

    finetune_parser = subparsers.add_parser("finetune", help="对本地 Qwen3-8B 执行 LoRA 微调")
    finetune_parser.add_argument("--data-path", default="./dataset/lora_data/最核心9k测试题_5k.json", help="LoRA 训练数据路径")
    finetune_parser.add_argument("--model-path", default="/app/model/models/Qwen/Qwen3-8B", help="本地基础模型路径")
    finetune_parser.add_argument("--output-dir", default="./output/qwen3_8b_lora", help="LoRA 输出目录")
    finetune_parser.add_argument("--epochs", type=int, default=2, help="训练轮数")
    finetune_parser.add_argument("--batch-size", type=int, default=2, help="单卡 batch size")
    finetune_parser.add_argument("--grad-accum", type=int, default=4 ,help="梯度累积步数")
    finetune_parser.add_argument("--learning-rate", type=float, default=1e-5, help="学习率")
    finetune_parser.add_argument("--max-seq-length", type=int, default=4096, help="最大序列长度")
    finetune_parser.add_argument("--lora-r", type=int, default=32, help="LoRA rank")
    finetune_parser.add_argument("--lora-alpha", type=int, default=64, help="LoRA alpha")
    finetune_parser.add_argument("--lora-dropout", type=float, default=0.05, help="LoRA dropout")
    finetune_parser.add_argument("--lora-scope", default="qv", choices=["q", "qv", "qkv", "attention", "full", "custom"], help="LoRA 注入范围，默认只调 Q/V")
    finetune_parser.add_argument("--report-to", default="wandb", help="训练日志上报方式，默认 wandb；多个值用英文逗号分隔，none 表示关闭")
    finetune_parser.add_argument("--wandb-project", default="lawrag-qwen3-8b-lora", help="wandb 项目名")
    finetune_parser.add_argument("--wandb-run-name", default="", help="wandb 运行名，默认使用输出目录名")
    finetune_parser.add_argument("--merge", action="store_true", help="训练完成后合并 LoRA 权重")
    
    evaluate_parser = subparsers.add_parser("evaluate", help="评估文书质量")
    evaluate_parser.add_argument("--document", required=True, help="生成的文书路径")
    evaluate_parser.add_argument("--evidence", required=True, help="原始证据路径")
    evaluate_parser.add_argument("--type", required=True, help="文书类型")
    evaluate_parser.add_argument("--reference", help="参考文书路径")
    
    args = parser.parse_args()
    
    if not args.command:
        parser.print_help()
        return
    
    lawrag = None
    if args.command != "finetune":
        lawrag = LawRAG(args.config)
    
    if args.command == "build":
        law_names = None
        if getattr(args, "law_names", None):
            law_names = [item.strip() for item in args.law_names.split(",") if item.strip()]

        result = lawrag.build_index(
            limit=args.limit,
            incremental=not bool(args.full or args.full_refresh),
            law_names=law_names,
            full_refresh=bool(args.full_refresh),
        )

        if result:
            mode = result.get("mode", "unknown")
            print(f"索引模式: {mode}")
            if mode == "incremental":
                print(f"扫描法律数量: {result.get('scanned_laws', 0)}")
                print(f"新增法律: {len(result.get('new_laws', []))}")
                print(f"更新法律: {len(result.get('updated_laws', []))}")
                print(f"未变更法律: {len(result.get('unchanged_laws', []))}")
                print(f"失败法律: {len(result.get('failed_laws', []))}")
            print(f"写入 chunks: {result.get('chunks_indexed', 0)}")
            stats = result.get("stats", {})
            if stats:
                print(
                    f"索引统计: collection={stats.get('collection')} "
                    f"vectors={stats.get('vectors_count')} indexed_sources={stats.get('indexed_sources', 0)}"
                )
    
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
        results = lawrag.search_laws(args.query, args.top_k, case_date=getattr(args, "case_date", None))
        
        for idx, result in enumerate(results, 1):
            print(f"\n{idx}. {result['law_name']} {result['article_num']}")
            print(f"   原始距离: {float(result['distance']):.4f} (向量召回，越小越相关)")
            print(f"   最终排序分: {float(result.get('rerank_score', result['distance'])):.4f} (重排后，越小越靠前)")
            if result.get("effective_date") or result.get("repeal_date"):
                print(
                    f"   时效区间: {result.get('effective_date') or '未知生效'}"
                    f" ~ {result.get('repeal_date') or '现行有效'}"
                )
            if result.get("version_id"):
                print(f"   版本标识: {result['version_id']}")
            if result.get("hierarchy_path"):
                print(f"   父级路径: {result['hierarchy_path']}")
            if result.get("locator"):
                print(f"   体系定位: {result['locator']}")
            if result.get("keyword_hits"):
                print(f"   关键词命中: {', '.join(result['keyword_hits'])}")
            if result.get("path_hits"):
                print(f"   路径命中: {', '.join(result['path_hits'])}")
            if result.get("path_focus_hits"):
                print(f"   路径重点命中: {', '.join(result['path_focus_hits'])}")
            if float(result.get("path_match_score", 0.0)) > 0:
                print(f"   路径匹配加权: {float(result['path_match_score']):.4f}")
            if float(result.get("legal_priority_score", 0.0)) != 0:
                print(f"   法律优先级加权: {float(result['legal_priority_score']):.4f}")
            if result.get("legal_priority_reasons"):
                print(f"   优先级原因: {'；'.join(result['legal_priority_reasons'])}")
            if result.get("effective_keywords"):
                print(f"   重排有效词: {', '.join(result['effective_keywords'])}")
            if result.get("keyword_idf"):
                idf_text = ", ".join([f"{k}:{v:.2f}" for k, v in result["keyword_idf"].items()])
                print(f"   命中词区分度(IDF): {idf_text}")
            print(f"   内容: {result['content'][:200]}...")

    elif args.command == "ask":
        result = lawrag.answer_query(args.query, case_date=getattr(args, "case_date", None))

        issue_outline = result.get("issue_outline", [])
        if issue_outline:
            print("争点提纲:")
            for item in issue_outline:
                print(f"- {item.get('title', '')}: {item.get('focus', '')}")
            print("")

        print("回答:")
        print(result.get("answer", ""))

        citations = result.get("citations", [])
        if citations:
            print("\n引用:")
            for item in citations:
                print(
                    f"[{item['id']}] {item['law_name']} {item['article_num']} "
                    f"(原始距离={float(item['distance']):.4f}, 最终排序分={float(item.get('rerank_score', item['distance'])):.4f})"
                )
                if item.get("effective_date") or item.get("repeal_date"):
                    print(
                        f"    时效区间: {item.get('effective_date') or '未知生效'}"
                        f" ~ {item.get('repeal_date') or '现行有效'}"
                    )
                if item.get("locator"):
                    print(f"    体系定位: {item['locator']}")

        trace = result.get("trace", [])
        if trace:
            print("\n检索轨迹:")
            for step in trace:
                print(
                    f"- 第{step['round_index']}轮 | query={step['query_used']} "
                    f"| keywords={step['keywords']} | hits={step['vector_hits']}->{step['kept_hits']}"
                )
    
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

    elif args.command == "finetune":
        from src.finetuning.LoraQwenModel import QLoraTrainingConfig, QLoraQwenTrainer

        report_to = [item.strip() for item in args.report_to.split(",") if item.strip()]
        if [item.lower() for item in report_to] == ["none"]:
            report_to = []

        train_config = QLoraTrainingConfig(
            model_name_or_path=args.model_path,
            output_dir=args.output_dir,
            data_path=args.data_path,
            num_train_epochs=args.epochs,
            per_device_train_batch_size=args.batch_size,
            gradient_accumulation_steps=args.grad_accum,
            learning_rate=args.learning_rate,
            max_seq_length=args.max_seq_length,
            lora_r=args.lora_r,
            lora_alpha=args.lora_alpha,
            lora_dropout=args.lora_dropout,
            lora_scope=args.lora_scope,
            report_to=report_to,
            wandb_project=args.wandb_project,
            wandb_run_name=args.wandb_run_name,
        )

        trainer = QLoraQwenTrainer(train_config)
        trainer.load_model_and_tokenizer()
        trainer.setup_lora()
        train_dataset = trainer.load_training_data()
        trainer.setup_trainer(train_dataset)
        trainer.train()
        trainer.save_model()

        if args.merge:
            trainer.merge_and_save()

        print(f"LoRA 微调完成，输出目录: {args.output_dir}")


if __name__ == "__main__":
    main()
