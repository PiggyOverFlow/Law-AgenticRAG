from typing import Dict, Any, Optional
from pathlib import Path
import logging
from datetime import datetime

from config import get_config
from src.models.legal_event import CaseFacts
from src.agent.legal_agent import LegalAgent


logger = logging.getLogger(__name__)


class DocumentGenerator:
    """法律文书生成器，基于案件事实和用户需求生成各类法律文书"""
    
    def __init__(self):
        """初始化文书生成器"""
        self.config = get_config()
        self.agent = LegalAgent()

    def generate_document(
        self,
        case_facts: CaseFacts,
        document_type: str,
        user_request: Optional[str] = None,
        additional_info: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """生成法律文书
        
        Args:
            case_facts: 案件事实
            document_type: 文书类型
            user_request: 用户需求
            additional_info: 额外信息
            
        Returns:
            Dict[str, Any]: 包含文书内容和元数据的字典
        """
        logger.info(f"开始生成文书: {document_type}")
        
        if document_type not in self.config.document.supported_types:
            raise ValueError(f"不支持的文书类型: {document_type}")
        
        context = {
            "task": f"生成{document_type}",
            "case_facts": case_facts,
            "document_type": document_type,
            "user_request": user_request or "",
            "additional_info": additional_info or {}
        }
        
        document_content = self.agent.run(context)
        
        result = {
            "document_type": document_type,
            "content": document_content,
            "generated_at": datetime.now().isoformat(),
            "case_facts_summary": case_facts.evidence_summary,
            "key_disputes": case_facts.key_disputes
        }
        
        logger.info(f"文书生成完成: {document_type}")
        return result

    def save_document(self, document: Dict[str, Any], filename: Optional[str] = None) -> str:
        """保存文书到文件
        
        Args:
            document: 文书数据字典
            filename: 文件名，如不指定则自动生成
            
        Returns:
            str: 保存的文件路径
        """
        output_dir = Path(self.config.document.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        if not filename:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"{document['document_type']}_{timestamp}.txt"
        
        file_path = output_dir / filename
        
        with open(file_path, "w", encoding="utf-8") as f:
            f.write(document["content"])
        
        logger.info(f"文书已保存: {file_path}")
        return str(file_path)

    def generate_and_save(
        self,
        case_facts: CaseFacts,
        document_type: str,
        user_request: Optional[str] = None,
        filename: Optional[str] = None
    ) -> str:
        """生成并保存文书
        
        Args:
            case_facts: 案件事实
            document_type: 文书类型
            user_request: 用户需求
            filename: 文件名
            
        Returns:
            str: 保存的文件路径
        """
        document = self.generate_document(case_facts, document_type, user_request)
        return self.save_document(document, filename)


class DocumentFormatter:
    """文书格式化工具，用于填充文书模板中的占位符"""
    
    @staticmethod
    def format_complaint(document: Dict[str, Any], plaintiff_info: Dict[str, Any], defendant_info: Dict[str, Any]) -> str:
        """格式化起诉书
        
        Args:
            document: 文书数据字典
            plaintiff_info: 原告信息
            defendant_info: 被告信息
            
        Returns:
            str: 格式化后的起诉书内容
        """
        content = document["content"]
        
        replacements = {
            "{plaintiff_info}": DocumentFormatter._format_party_info(plaintiff_info, "原告"),
            "{defendant_info}": DocumentFormatter._format_party_info(defendant_info, "被告"),
            "{plaintiff}": plaintiff_info.get("name", ""),
            "{defendant}": defendant_info.get("name", ""),
            "{date}": datetime.now().strftime("%Y年%m月%d日")
        }
        
        for placeholder, value in replacements.items():
            content = content.replace(placeholder, value)
        
        return content

    @staticmethod
    def format_defense(document: Dict[str, Any], defendant_info: Dict[str, Any], plaintiff_info: Dict[str, Any]) -> str:
        """格式化答辩状
        
        Args:
            document: 文书数据字典
            defendant_info: 答辩人信息
            plaintiff_info: 被答辩人信息
            
        Returns:
            str: 格式化后的答辩状内容
        """
        content = document["content"]
        
        replacements = {
            "{defendant_info}": DocumentFormatter._format_party_info(defendant_info, "答辩人"),
            "{plaintiff_info}": DocumentFormatter._format_party_info(plaintiff_info, "被答辩人"),
            "{defendant}": defendant_info.get("name", ""),
            "{plaintiff}": plaintiff_info.get("name", ""),
            "{date}": datetime.now().strftime("%Y年%m月%d日")
        }
        
        for placeholder, value in replacements.items():
            content = content.replace(placeholder, value)
        
        return content

    @staticmethod
    def _format_party_info(info: Dict[str, Any], role: str) -> str:
        """格式化当事人信息
        
        Args:
            info: 当事人信息字典
            role: 当事人角色（原告、被告等）
            
        Returns:
            str: 格式化后的当事人信息字符串
        """
        parts = [f"{role}：{info.get('name', '')}"]
        
        if info.get("gender"):
            parts.append(f"性别：{info['gender']}")
        if info.get("age"):
            parts.append(f"年龄：{info['age']}")
        if info.get("id_number"):
            parts.append(f"身份证号：{info['id_number']}")
        if info.get("address"):
            parts.append(f"住址：{info['address']}")
        if info.get("phone"):
            parts.append(f"联系电话：{info['phone']}")
        
        if info.get("is_legal_person", False):
            parts.append(f"法定代表人：{info.get('legal_representative', '')}")
            parts.append(f"统一社会信用代码：{info.get('credit_code', '')}")
        
        return "，".join(parts)


class DocumentValidator:
    """文书验证工具，检查文书格式和内容的完整性"""
    
    @staticmethod
    def validate_document(document: Dict[str, Any]) -> Dict[str, Any]:
        """验证文书的基本完整性
        
        Args:
            document: 文书数据字典
            
        Returns:
            Dict[str, Any]: 验证结果，包含错误和警告信息
        """
        validation_result = {
            "is_valid": True,
            "errors": [],
            "warnings": []
        }
        
        content = document["content"]
        
        if not content or len(content) < 100:
            validation_result["is_valid"] = False
            validation_result["errors"].append("文书内容过短")
        
        if "原告" not in content and "申请人" not in content:
            validation_result["warnings"].append("缺少原告或申请人信息")
        
        if "被告" not in content and "被申请人" not in content:
            validation_result["warnings"].append("缺少被告或被申请人信息")
        
        if "诉讼请求" not in content and "申请事项" not in content:
            validation_result["warnings"].append("缺少诉讼请求或申请事项")
        
        if "事实与理由" not in content:
            validation_result["warnings"].append("缺少事实与理由部分")
        
        return validation_result

    @staticmethod
    def validate_complaint(document: Dict[str, Any]) -> Dict[str, Any]:
        """验证起诉书的完整性
        
        Args:
            document: 文书数据字典
            
        Returns:
            Dict[str, Any]: 验证结果
        """
        validation_result = DocumentValidator.validate_document(document)
        
        content = document["content"]
        
        if "此致" not in content:
            validation_result["warnings"].append("缺少致送法院信息")
        
        if "具状人" not in content:
            validation_result["warnings"].append("缺少具状人签名")
        
        return validation_result