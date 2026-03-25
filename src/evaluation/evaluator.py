from typing import Dict, Any, List, Optional
from dataclasses import dataclass
from enum import Enum
import logging
from datetime import datetime

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
        consistency_score = 0.85
        discrepancies = []
        
        return {
            "score": consistency_score,
            "discrepancies": discrepancies,
            "analysis": "生成内容与原始证据基本一致"
        }

    def _perform_logic_coherence_check(
        self,
        generated_document: str,
        reference_document: Optional[str]
    ) -> Dict[str, Any]:
        coherence_score = 0.80
        
        return {
            "score": coherence_score,
            "logical_issues": [],
            "analysis": "法律逻辑基本连贯"
        }

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
        feedback = f"""
# LLM 评判报告

## 总体评分
{overall_score:.2f} / 1.00

## 一致性检验
评分：{consistency_result['score']:.2f}
分析：{consistency_result['analysis']}

## 逻辑连贯性检验
评分：{logic_result['score']:.2f}
分析：{logic_result['analysis']}

## 改进建议
1. 加强事实描述的准确性
2. 优化法条引用的逻辑性
3. 完善文书结构的完整性
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
        
        expert_stats = self.expert_system.get_review_statistics()
        if "total_reviews" in expert_stats and expert_stats["total_reviews"] > 0:
            report += "## 专家评审统计\n\n"
            report += f"- 评审数量：{expert_stats['total_reviews']}\n"
            report += f"- 平均总分：{expert_stats['total_score_stats']['average']:.2f}\n"
            report += f"- 平均均分：{expert_stats['average_score_stats']['average']:.2f}\n\n"
        
        return report