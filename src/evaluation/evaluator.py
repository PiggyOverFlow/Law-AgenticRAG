from typing import Dict, Any, List, Optional
from dataclasses import dataclass
from enum import Enum
import logging
from datetime import datetime
import re

from config import get_config


logger = logging.getLogger(__name__)


class EvaluationCriterion(Enum):
    FORMAT_CORRECTNESS = "格式正确性"
    CONTENT_COMPLETENESS = "内容完整性"
    LOGICAL_COHERENCE = "逻辑条理性"
    CONTENT_ACCURACY = "内容准确性"
    LAW_APPLICABILITY = "法条适用性"


@dataclass
class ExpertReview:
    reviewer_name: str
    reviewer_id: str
    document_type: str
    scores: Dict[EvaluationCriterion, int]
    comments: str
    reviewed_at: datetime

    def get_total_score(self) -> int:
        return sum(self.scores.values())

    def get_average_score(self) -> float:
        return self.get_total_score() / len(self.scores)


@dataclass
class LLMJudgeResult:
    consistency_check: Dict[str, Any]
    logic_coherence_check: Dict[str, Any]
    overall_score: float
    detailed_feedback: str
    judged_at: datetime


class ExpertReviewSystem:
    def __init__(self):
        self.config = get_config()
        self.reviews: List[ExpertReview] = []

    def create_review_form(self, document_type: str) -> str:
        criteria = self.config.evaluation.expert_review.criteria
        
        form = f"""
# 法律文书专家评审表

文书类型：{document_type}
评审人姓名：__________
评审日期：__________

## 评分标准（每项 0-5 分）

"""
        
        for idx, criterion in enumerate(criteria, 1):
            form += f"""
### {idx}. {criterion}

评分说明：
- 0分：完全不符合要求
- 1分：严重不符合要求
- 2分：部分不符合要求
- 3分：基本符合要求
- 4分：符合要求
- 5分：完全符合要求

评分：[0-5] _______

"""
        
        form += """
## 综合评价

请对该文书进行总体评价，指出优点和不足：

优点：
___________________________________________________________________________
___________________________________________________________________________

不足：
___________________________________________________________________________
___________________________________________________________________________

改进建议：
___________________________________________________________________________
___________________________________________________________________________

"""
        
        return form

    def submit_review(
        self,
        reviewer_name: str,
        reviewer_id: str,
        document_type: str,
        scores: Dict[str, int],
        comments: str
    ) -> ExpertReview:
        score_dict = {}
        for criterion_name, score in scores.items():
            try:
                criterion = EvaluationCriterion(criterion_name)
                if 0 <= score <= 5:
                    score_dict[criterion] = score
                else:
                    logger.warning(f"评分超出范围: {criterion_name} = {score}")
            except ValueError:
                logger.warning(f"未知的评价标准: {criterion_name}")
        
        review = ExpertReview(
            reviewer_name=reviewer_name,
            reviewer_id=reviewer_id,
            document_type=document_type,
            scores=score_dict,
            comments=comments,
            reviewed_at=datetime.now()
        )
        
        self.reviews.append(review)
        logger.info(f"收到评审: {reviewer_name} - {document_type}")
        
        return review

    def get_review_statistics(self) -> Dict[str, Any]:
        if not self.reviews:
            return {"message": "暂无评审数据"}
        
        total_reviews = len(self.reviews)
        total_scores = [review.get_total_score() for review in self.reviews]
        avg_scores = [review.get_average_score() for review in self.reviews]
        
        criterion_stats = {}
        for criterion in EvaluationCriterion:
            scores = [review.scores.get(criterion, 0) for review in self.reviews]
            criterion_stats[criterion.value] = {
                "average": sum(scores) / len(scores) if scores else 0,
                "min": min(scores) if scores else 0,
                "max": max(scores) if scores else 0
            }
        
        return {
            "total_reviews": total_reviews,
            "total_score_stats": {
                "average": sum(total_scores) / len(total_scores),
                "min": min(total_scores),
                "max": max(total_scores)
            },
            "average_score_stats": {
                "average": sum(avg_scores) / len(avg_scores),
                "min": min(avg_scores),
                "max": max(avg_scores)
            },
            "criterion_stats": criterion_stats
        }

    def export_reviews(self, output_path: str):
        import json
        from pathlib import Path
        
        output_file = Path(output_path)
        output_file.parent.mkdir(parents=True, exist_ok=True)
        
        reviews_data = []
        for review in self.reviews:
            reviews_data.append({
                "reviewer_name": review.reviewer_name,
                "reviewer_id": review.reviewer_id,
                "document_type": review.document_type,
                "scores": {c.value: score for c, score in review.scores.items()},
                "total_score": review.get_total_score(),
                "average_score": review.get_average_score(),
                "comments": review.comments,
                "reviewed_at": review.reviewed_at.isoformat()
            })
        
        with open(output_file, "w", encoding="utf-8") as f:
            json.dump(reviews_data, f, ensure_ascii=False, indent=2)
        
        logger.info(f"评审数据已导出: {output_file}")


class LLMJudgeAgent:
    def __init__(self):
        self.config = get_config()
        self._init_judge_model()

    def _init_judge_model(self):
        model_name = self.config.evaluation.llm_judge.model
        api_key = self.config.evaluation.llm_judge.api_key
        base_url = self.config.evaluation.llm_judge.base_url
        self.temperature = getattr(self.config.evaluation.llm_judge, 'temperature', 0.3)
        self.max_tokens = getattr(self.config.evaluation.llm_judge, 'max_tokens', 4096)
        logger.info(f"初始化评判模型: {model_name} (Temp: {self.temperature}, Max Tokens: {self.max_tokens})")
        
        # 实际使用中可以初始化 openai 客户端或其他底层调用
        # self.client = OpenAI(api_key=api_key, base_url=base_url)
        self.judge_model = model_name

    def judge_document(
        self,
        generated_document: str,
        original_evidence: str,
        reference_document: Optional[str] = None
    ) -> LLMJudgeResult:
        logger.info("开始 LLM 评判")
        
        consistency_result = self._perform_consistency_check(
            generated_document,
            original_evidence
        )
        
        logic_result = self._perform_logic_coherence_check(
            generated_document,
            reference_document
        )
        
        overall_score = self._calculate_overall_score(consistency_result, logic_result)
        
        detailed_feedback = self._generate_detailed_feedback(
            consistency_result,
            logic_result,
            overall_score
        )
        
        result = LLMJudgeResult(
            consistency_check=consistency_result,
            logic_coherence_check=logic_result,
            overall_score=overall_score,
            detailed_feedback=detailed_feedback,
            judged_at=datetime.now()
        )
        
        logger.info(f"LLM 评判完成，总分: {overall_score:.2f}")
        return result

    def _perform_consistency_check(
        self,
        generated_document: str,
        original_evidence: str
    ) -> Dict[str, Any]:
        generated_facts = self._extract_atomic_facts(generated_document)
        evidence_facts = self._extract_atomic_facts(original_evidence)

        generated_claims = self._flatten_facts(generated_facts, include_law_refs=False)
        evidence_claims = self._flatten_facts(evidence_facts, include_law_refs=False)

        supported = sorted(generated_claims & evidence_claims)
        unsupported = sorted(generated_claims - evidence_claims)

        support_rate = 0.0
        coverage_rate = 0.0
        hallucination_rate = 0.0

        if generated_claims:
            support_rate = self._safe_div(len(supported), len(generated_claims))
            coverage_rate = self._safe_div(
                len(supported),
                min(len(evidence_claims), len(generated_claims) + 2),
            )
            hallucination_rate = self._safe_div(len(unsupported), len(generated_claims))
            consistency_score = max(
                0.0,
                min(1.0, 0.75 * (1.0 - hallucination_rate) + 0.25 * coverage_rate),
            )
        else:
            lexical_overlap = self._jaccard_similarity(
                self._tokenize_text(generated_document),
                self._tokenize_text(original_evidence),
            )
            hallucination_rate = max(0.0, 1.0 - lexical_overlap)
            consistency_score = max(0.0, min(1.0, 0.45 + 0.40 * lexical_overlap))

        discrepancies = [
            f"未在证据中找到支撑事实: {item}"
            for item in unsupported[:12]
        ]

        if hallucination_rate <= 0.15:
            analysis = "事实与证据对齐良好，未发现明显虚构事实。"
        elif hallucination_rate <= 0.35:
            analysis = "存在少量未被证据支撑的事实，建议逐条核验金额、日期和利率。"
        else:
            analysis = "未被证据支撑的事实较多，存在较高幻觉风险。"

        return {
            "score": round(consistency_score, 4),
            "discrepancies": discrepancies,
            "analysis": analysis,
            "hallucination_rate": round(hallucination_rate, 4),
            "support_rate": round(support_rate, 4),
            "generated_facts_count": len(generated_claims),
            "supported_facts_count": len(supported),
            "unsupported_facts": unsupported[:20],
            "supported_facts": supported[:20],
        }

    def _perform_logic_coherence_check(
        self,
        generated_document: str,
        reference_document: Optional[str]
    ) -> Dict[str, Any]:
        content = generated_document or ""
        logical_issues: List[str] = []

        has_facts_section = "事实与理由" in content
        has_request_section = any(
            marker in content
            for marker in ("诉讼请求", "申请事项", "答辩请求", "上诉请求")
        )
        has_closing = "此致" in content

        if not has_facts_section:
            logical_issues.append("缺少“事实与理由”部分")
        if not has_request_section:
            logical_issues.append("缺少请求事项部分（诉讼请求/申请事项/答辩请求/上诉请求）")
        if not has_closing:
            logical_issues.append("缺少“此致”结尾格式")

        sentence_count = len([s for s in re.split(r"[。！？!?；;\n]+", content) if s.strip()])
        paragraph_count = len([line for line in content.splitlines() if line.strip()])
        citation_count = len(re.findall(r"第[一二三四五六七八九十百千万零\d]+条", content))
        causal_markers = sum(
            1 for marker in ("因此", "据此", "综上", "故请求", "故应") if marker in content
        )

        if sentence_count < 4:
            logical_issues.append("有效句子数量偏少，论证链条可能不完整")
        if citation_count == 0:
            logical_issues.append("未检测到明确法条引用（如“第XX条”）")
        if causal_markers == 0:
            logical_issues.append("缺少结论性连接词，推理闭环不清晰")

        reference_similarity = None
        if reference_document:
            reference_similarity = self._jaccard_similarity(
                self._tokenize_text(content),
                self._tokenize_text(reference_document),
            )
            if reference_similarity < 0.05:
                logical_issues.append("与参考文书文本重合度较低，建议复核文书体例")

        coherence_score = 1.0
        coherence_score -= 0.18 if not has_facts_section else 0.0
        coherence_score -= 0.18 if not has_request_section else 0.0
        coherence_score -= 0.10 if not has_closing else 0.0
        coherence_score -= 0.15 if citation_count == 0 else 0.0
        coherence_score -= 0.12 if sentence_count < 4 else 0.0
        coherence_score -= 0.08 if causal_markers == 0 else 0.0
        if reference_similarity is not None and reference_similarity < 0.05:
            coherence_score -= 0.07
        coherence_score = max(0.0, min(1.0, coherence_score))

        if not logical_issues:
            analysis = "文书结构完整，法律论证链条较清晰。"
        elif len(logical_issues) <= 2:
            analysis = "文书逻辑基本可用，但仍有局部结构或论证缺口。"
        else:
            analysis = "文书存在较明显结构/论证问题，建议先修复再用于对外输出。"

        return {
            "score": round(coherence_score, 4),
            "logical_issues": logical_issues,
            "analysis": analysis,
            "metrics": {
                "sentence_count": sentence_count,
                "paragraph_count": paragraph_count,
                "citation_count": citation_count,
                "causal_markers": causal_markers,
                "reference_similarity": round(reference_similarity, 4) if reference_similarity is not None else None,
            },
        }

    def _extract_atomic_facts(self, text: str) -> Dict[str, List[str]]:
        normalized = re.sub(r"\s+", " ", text or "").strip()
        if not normalized:
            return {"dates": [], "amounts": [], "rates": [], "keywords": [], "law_refs": []}

        dates = self._dedup_list(re.findall(r"20\d{2}年\d{1,2}月\d{1,2}日", normalized))
        amounts = self._dedup_list(re.findall(r"\d+(?:\.\d+)?(?:万|千|百)?元", normalized))
        rates = self._dedup_list(
            re.findall(r"(?:月利率|年利率|利率)\s*\d+(?:\.\d+)?%|\d+(?:\.\d+)?%", normalized)
        )
        law_refs = self._dedup_list(re.findall(r"第[一二三四五六七八九十百千万零\d]+条", normalized))
        keywords = self._extract_domain_keywords(normalized)

        return {
            "dates": dates,
            "amounts": amounts,
            "rates": rates,
            "keywords": keywords,
            "law_refs": law_refs,
        }

    def _extract_domain_keywords(self, text: str) -> List[str]:
        keyword_bank = [
            "借款", "出借", "还款", "逾期", "违约", "合同", "借条", "转账", "利息", "本金",
            "侵权", "赔偿", "医疗费", "误工费", "劳动合同", "解除", "工资", "举证", "时效",
            "交通事故", "责任", "损害", "拒不履行", "管辖",
        ]
        return [k for k in keyword_bank if k in text]

    def _flatten_facts(self, facts: Dict[str, List[str]], include_law_refs: bool = False) -> set[str]:
        categories = ["dates", "amounts", "rates", "keywords"]
        if include_law_refs:
            categories.append("law_refs")

        pool: set[str] = set()
        for category in categories:
            for item in facts.get(category, []):
                token = re.sub(r"\s+", "", str(item or "")).strip()
                if token:
                    pool.add(token)
        return pool

    def _tokenize_text(self, text: str) -> List[str]:
        tokens = re.findall(r"[\u4e00-\u9fa5A-Za-z0-9]{2,}", text or "")
        low_info = {
            "根据", "相关", "问题", "情况", "认为", "需要", "应当", "可以", "进行", "内容", "部分",
            "法律", "法规", "条文", "文书", "事实", "理由", "请求", "法院", "当事人",
        }

        cleaned = []
        for token in tokens:
            if token in low_info:
                continue
            if token.isdigit():
                continue
            cleaned.append(token)
        return cleaned

    def _jaccard_similarity(self, left: List[str], right: List[str]) -> float:
        left_set = set(left)
        right_set = set(right)
        if not left_set or not right_set:
            return 0.0
        return len(left_set & right_set) / len(left_set | right_set)

    def _safe_div(self, numerator: float, denominator: float) -> float:
        if denominator <= 0:
            return 0.0
        return numerator / denominator

    def _dedup_list(self, values: List[str]) -> List[str]:
        dedup: List[str] = []
        seen = set()
        for value in values:
            normalized = re.sub(r"\s+", "", str(value or "")).strip()
            if not normalized or normalized in seen:
                continue
            seen.add(normalized)
            dedup.append(normalized)
        return dedup

    def _calculate_overall_score(
        self,
        consistency_result: Dict[str, Any],
        logic_result: Dict[str, Any]
    ) -> float:
        consistency_score = consistency_result.get("score", 0.0)
        logic_score = logic_result.get("score", 0.0)
        
        overall_score = (consistency_score * 0.5) + (logic_score * 0.5)
        return overall_score

    def _generate_detailed_feedback(
        self,
        consistency_result: Dict[str, Any],
        logic_result: Dict[str, Any],
        overall_score: float
    ) -> str:
        hallucination_rate = float(consistency_result.get("hallucination_rate", 0.0))
        logical_issues = logic_result.get("logical_issues", []) or []

        suggestions: List[str] = []
        if hallucination_rate > 0.30:
            suggestions.append("优先修复证据对齐：逐条核验金额、日期、利率和关键行为描述。")
        elif hallucination_rate > 0.15:
            suggestions.append("降低生成温度并收紧法条上下文，减少未证据支撑的扩写。")

        if logical_issues:
            suggestions.append("补齐文书结构要素（请求事项、事实与理由、结尾格式）并增强法条引用。")

        if not suggestions:
            suggestions.append("当前质量稳定，可继续扩大样本评测并固化评估口径。")

        suggestion_lines = "\n".join([f"{idx}. {text}" for idx, text in enumerate(suggestions, 1)])

        feedback = f"""
# LLM 评判报告

## 总体评分
{overall_score:.2f} / 1.00

## 一致性检验
评分：{consistency_result['score']:.2f}
分析：{consistency_result['analysis']}
幻觉率：{hallucination_rate:.2%}

## 逻辑连贯性检验
评分：{logic_result['score']:.2f}
分析：{logic_result['analysis']}
逻辑问题数：{len(logical_issues)}

## 改进建议
{suggestion_lines}
"""
        
        return feedback


class EvaluationFramework:
    def __init__(self):
        self.config = get_config()
        self.expert_system = ExpertReviewSystem()
        self.llm_judge = LLMJudgeAgent()

    def evaluate_document(
        self,
        generated_document: str,
        original_evidence: str,
        document_type: str,
        reference_document: Optional[str] = None
    ) -> Dict[str, Any]:
        logger.info(f"开始评估文书: {document_type}")
        
        evaluation_result = {
            "document_type": document_type,
            "evaluated_at": datetime.now().isoformat()
        }
        
        if self.config.evaluation.llm_judge.enabled:
            llm_result = self.llm_judge.judge_document(
                generated_document,
                original_evidence,
                reference_document
            )
            evaluation_result["llm_judge"] = {
                "overall_score": llm_result.overall_score,
                "consistency_score": llm_result.consistency_check["score"],
                "logic_score": llm_result.logic_coherence_check["score"],
                "hallucination_rate": llm_result.consistency_check.get("hallucination_rate", 0.0),
                "discrepancies": llm_result.consistency_check.get("discrepancies", []),
                "logical_issues": llm_result.logic_coherence_check.get("logical_issues", []),
                "detailed_feedback": llm_result.detailed_feedback
            }
        
        if self.config.evaluation.expert_review.enabled:
            evaluation_result["expert_review_form"] = self.expert_system.create_review_form(document_type)
        
        return evaluation_result

    def generate_evaluation_report(self, evaluation_results: List[Dict[str, Any]]) -> str:
        report = "# 法律文书生成系统评估报告\n\n"
        report += f"评估时间：{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n"
        report += f"评估文书数量：{len(evaluation_results)}\n\n"
        
        llm_scores = []
        for result in evaluation_results:
            if "llm_judge" in result:
                llm_scores.append(result["llm_judge"]["overall_score"])
        
        if llm_scores:
            report += "## LLM 评判统计\n\n"
            report += f"- 平均分：{sum(llm_scores) / len(llm_scores):.2f}\n"
            report += f"- 最高分：{max(llm_scores):.2f}\n"
            report += f"- 最低分：{min(llm_scores):.2f}\n\n"

            halluc_rates = [
                float(result["llm_judge"].get("hallucination_rate", 0.0))
                for result in evaluation_results
                if "llm_judge" in result
            ]
            if halluc_rates:
                report += "## 幻觉率统计\n\n"
                report += f"- 平均幻觉率：{sum(halluc_rates) / len(halluc_rates):.2%}\n"
                report += f"- 最高幻觉率：{max(halluc_rates):.2%}\n"
                report += f"- 最低幻觉率：{min(halluc_rates):.2%}\n\n"
        
        expert_stats = self.expert_system.get_review_statistics()
        if "total_reviews" in expert_stats and expert_stats["total_reviews"] > 0:
            report += "## 专家评审统计\n\n"
            report += f"- 评审数量：{expert_stats['total_reviews']}\n"
            report += f"- 平均总分：{expert_stats['total_score_stats']['average']:.2f}\n"
            report += f"- 平均均分：{expert_stats['average_score_stats']['average']:.2f}\n\n"
        
        return report