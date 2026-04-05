from typing import List, Dict, Any, Optional
from datetime import datetime
from dataclasses import dataclass, field
import logging
import re
import hashlib

from config import get_config


logger = logging.getLogger(__name__)


@dataclass
class LawTreeNode:
    """法律结构树节点，表示法律文本的层级结构"""
    node_id: str
    law_name: str
    level: str
    title: str
    label: str
    parent_id: Optional[str] = None
    children: List[str] = field(default_factory=list)
    article_num: str = ""
    order: int = 0

    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return {
            "node_id": self.node_id,
            "law_name": self.law_name,
            "level": self.level,
            "title": self.title,
            "label": self.label,
            "parent_id": self.parent_id,
            "children": list(self.children),
            "article_num": self.article_num,
            "order": self.order,
        }


@dataclass
class LawStructureTree:
    """法律结构树，表示法律的完整层级结构"""
    law_name: str
    root_id: str
    nodes: Dict[str, LawTreeNode] = field(default_factory=dict)
    article_index: Dict[str, str] = field(default_factory=dict)

    @classmethod
    def create(cls, law_name: str) -> "LawStructureTree":
        """创建法律结构树
        
        Args:
            law_name: 法律名称
            
        Returns:
            LawStructureTree: 法律结构树对象
        """
        root_id = f"{law_name}::root"
        tree = cls(law_name=law_name, root_id=root_id)
        tree.nodes[root_id] = LawTreeNode(
            node_id=root_id,
            law_name=law_name,
            level="法",
            title=law_name,
            label=law_name,
            parent_id=None,
            order=0,
        )
        return tree

    def add_node(
        self,
        level: str,
        title: str,
        label: str,
        parent_id: Optional[str],
        order: int,
        article_num: str = "",
    ) -> LawTreeNode:
        """添加节点到结构树
        
        Args:
            level: 层级
            title: 标题
            label: 标签
            parent_id: 父节点ID
            order: 顺序
            article_num: 条号
            
        Returns:
            LawTreeNode: 新创建的节点
        """
        safe_title = str(title or "").strip()
        node_id = f"{self.law_name}::{level}::{order}::{safe_title}"
        node = LawTreeNode(
            node_id=node_id,
            law_name=self.law_name,
            level=level,
            title=safe_title,
            label=str(label or safe_title).strip() or safe_title,
            parent_id=parent_id or self.root_id,
            order=order,
            article_num=str(article_num or "").strip(),
        )
        self.nodes[node_id] = node
        parent = self.nodes.get(node.parent_id)
        if parent and node_id not in parent.children:
            parent.children.append(node_id)
        if node.article_num and node.article_num not in self.article_index:
            self.article_index[node.article_num] = node_id
        return node

    def get_path(self, node_id: Optional[str]) -> List[LawTreeNode]:
        """获取节点路径
        
        Args:
            node_id: 节点ID
            
        Returns:
            List[LawTreeNode]: 从根节点到目标节点的路径
        """
        path: List[LawTreeNode] = []
        current_id = node_id
        seen = set()
        while current_id and current_id not in seen:
            seen.add(current_id)
            node = self.nodes.get(current_id)
            if node is None:
                break
            path.append(node)
            current_id = node.parent_id
        return list(reversed(path))

    def get_children(self, node_id: Optional[str]) -> List[LawTreeNode]:
        """获取子节点
        
        Args:
            node_id: 节点ID
            
        Returns:
            List[LawTreeNode]: 子节点列表
        """
        node = self.nodes.get(node_id or "")
        if not node:
            return []
        return [self.nodes[child_id] for child_id in node.children if child_id in self.nodes]


@dataclass
class LawChunk:
    """法律文本块，表示法律文本的一个可检索单元"""
    chunk_id: str
    law_name: str
    article_num: str
    content: str
    level: str
    metadata: Dict[str, Any]
    
    def __repr__(self):
        return f"<LawChunk {self.law_name} {self.article_num}>"


@dataclass
class RetrievalResult:
    """检索结果"""
    chunk: LawChunk
    score: float
    rank: int


class LawChunker:
    """法律文本分块器，将法律文本分割为可检索的文本块"""
    
    def __init__(self):
        """初始化分块器"""
        self.config = get_config()
        self.level_order = ["法", "编", "章", "节", "条", "款", "项", "目"]
        
        self.level_patterns = {
            '编': r'^第[一二三四五六七八九十百千万零]+编',
            '章': r'^第[一二三四五六七八九十百千万零]+章',
            '节': r'^第[一二三四五六七八九十百千万零]+节',
            '条': r'^第[一二三四五六七八九十百千万零]+条',
            '款': r'^\([一二三四五六七八九十百千万零]+\)',
            '项': r'^[一二三四五六七八九十百千万零]+[、.]',
            '目': r'^\([1-9]\)'
        }

    def chunk_law_text(self, law_name: str, law_content: str, metadata: Dict[str, Any]) -> List[LawChunk]:
        """将法律文本分割为文本块
        
        Args:
            law_name: 法律名称
            law_content: 法律文本内容
            metadata: 元数据
            
        Returns:
            List[LawChunk]: 文本块列表
        """
        chunks: List[LawChunk] = []
        lines = law_content.split('\n')
        structure_tree = LawStructureTree.create(law_name)
        current_nodes: Dict[str, LawTreeNode] = {"法": structure_tree.nodes[structure_tree.root_id]}
        order_counter = 0
        current_chunk: Optional[LawChunk] = None
        current_chunk_node_id: Optional[str] = None
        chunk_lines: List[str] = []

        for line in lines:
            line = line.strip()
            if not line:
                continue
            
            detected_level = self._detect_level(line)

            if detected_level in {'编', '章', '节'}:
                if current_chunk:
                    self._finalize_chunk(
                        current_chunk,
                        chunk_lines,
                        current_chunk_node_id,
                        structure_tree,
                    )
                    chunks.append(current_chunk)
                    current_chunk = None
                    current_chunk_node_id = None
                    chunk_lines = []

                order_counter += 1
                node = self._create_structure_node(
                    law_name=law_name,
                    line=line,
                    level=detected_level,
                    tree=structure_tree,
                    current_nodes=current_nodes,
                    order=order_counter,
                )
                self._set_current_node(current_nodes, node)
                continue

            if detected_level == '条':
                if current_chunk:
                    self._finalize_chunk(
                        current_chunk,
                        chunk_lines,
                        current_chunk_node_id,
                        structure_tree,
                    )
                    chunks.append(current_chunk)

                article_num = self._extract_article_num(line)
                order_counter += 1
                node = self._create_structure_node(
                    law_name=law_name,
                    line=line,
                    level='条',
                    tree=structure_tree,
                    current_nodes=current_nodes,
                    order=order_counter,
                    article_num=article_num,
                )
                self._set_current_node(current_nodes, node)
                current_chunk = LawChunk(
                    chunk_id=f"{law_name}_{article_num}",
                    law_name=law_name,
                    article_num=article_num,
                    content="",
                    level="条",
                    metadata=metadata.copy()
                )
                current_chunk_node_id = node.node_id
                chunk_lines = [line]
                continue

            if detected_level in {'款', '项', '目'} and current_chunk:
                order_counter += 1
                node = self._create_structure_node(
                    law_name=law_name,
                    line=line,
                    level=detected_level,
                    tree=structure_tree,
                    current_nodes=current_nodes,
                    order=order_counter,
                )
                self._set_current_node(current_nodes, node)
                chunk_lines.append(line)
                continue

            else:
                if current_chunk:
                    chunk_lines.append(line)
        
        if current_chunk:
            self._finalize_chunk(
                current_chunk,
                chunk_lines,
                current_chunk_node_id,
                structure_tree,
            )
            chunks.append(current_chunk)
        
        return chunks

    def _create_structure_node(
        self,
        law_name: str,
        line: str,
        level: str,
        tree: LawStructureTree,
        current_nodes: Dict[str, LawTreeNode],
        order: int,
        article_num: str = "",
    ) -> LawTreeNode:
        """创建结构节点
        
        Args:
            law_name: 法律名称
            line: 文本行
            level: 层级
            tree: 结构树
            current_nodes: 当前节点字典
            order: 顺序
            article_num: 条号
            
        Returns:
            LawTreeNode: 新创建的节点
        """
        label = self._extract_label(line, level)
        parent_id = self._find_parent_id(current_nodes, level)
        return tree.add_node(
            level=level,
            title=line,
            label=label,
            parent_id=parent_id,
            order=order,
            article_num=article_num,
        )

    def _find_parent_id(self, current_nodes: Dict[str, LawTreeNode], level: str) -> Optional[str]:
        """查找父节点ID
        
        Args:
            current_nodes: 当前节点字典
            level: 层级
            
        Returns:
            Optional[str]: 父节点ID
        """
        current_index = self.level_order.index(level)
        for parent_level in reversed(self.level_order[:current_index]):
            node = current_nodes.get(parent_level)
            if node:
                return node.node_id
        return None

    def _set_current_node(self, current_nodes: Dict[str, LawTreeNode], node: LawTreeNode) -> None:
        """设置当前节点
        
        Args:
            current_nodes: 当前节点字典
            node: 节点
        """
        current_index = self.level_order.index(node.level)
        for level in self.level_order[current_index:]:
            current_nodes.pop(level, None)
        current_nodes[node.level] = node

    def _finalize_chunk(
        self,
        chunk: LawChunk,
        chunk_lines: List[str],
        node_id: Optional[str],
        tree: LawStructureTree,
    ) -> None:
        """完成文本块的构建
        
        Args:
            chunk: 文本块
            chunk_lines: 文本行列表
            node_id: 节点ID
            tree: 结构树
        """
        raw_content = '\n'.join(chunk_lines).strip()
        path_nodes = tree.get_path(node_id)
        hierarchy_nodes = [node for node in path_nodes if node.level in {'编', '章', '节'}]
        context_lines = [tree.law_name] + [node.title for node in hierarchy_nodes]
        contextual_content = raw_content
        if context_lines:
            contextual_content = '\n'.join(context_lines + [raw_content]).strip()

        path_snapshot = [self._serialize_tree_node(node) for node in path_nodes]
        child_snapshot = [self._serialize_tree_node(node) for node in tree.get_children(node_id)]
        hierarchy_path = " > ".join([node.title for node in hierarchy_nodes]) or "无上位结构"
        retrieval_text = "\n".join(
            [
                f"法律名称：{tree.law_name}",
                f"体系路径：{hierarchy_path}",
                f"条号：{chunk.article_num or '未知条号'}",
                f"正文：{raw_content}",
            ]
        ).strip()
        source_law_id = str(chunk.metadata.get("source_law_id") or tree.law_name).strip()
        version_id = str(
            chunk.metadata.get("version_id")
            or chunk.metadata.get("source_hash", "")[:12]
            or "v1"
        ).strip()
        article_key = chunk.article_num or str(node_id or "unknown")
        article_id = self._build_stable_id(source_law_id, article_key)
        chunk.chunk_id = self._build_stable_id(article_id, version_id)

        chunk.content = contextual_content
        chunk.metadata.update(
            {
                "source_law_id": source_law_id,
                "article_id": article_id,
                "version_id": version_id,
                "raw_content": raw_content,
                "context_text": '\n'.join(context_lines),
                "retrieval_text": retrieval_text,
                "structure_root_id": tree.root_id,
                "structure_node_id": node_id,
                "structure_parent_id": path_nodes[-1].parent_id if path_nodes else tree.root_id,
                "structure_path": path_snapshot,
                "structure_path_ids": [node["node_id"] for node in path_snapshot],
                "structure_path_levels": [node["level"] for node in path_snapshot],
                "structure_path_titles": [node["title"] for node in path_snapshot],
                "structure_path_text": " > ".join(
                    [node["title"] for node in path_snapshot if node["level"] != "法"]
                ),
                "structure_hierarchy_path": hierarchy_path,
                "structure_locator": " / ".join([node["title"] for node in path_snapshot]),
                "structure_children": child_snapshot,
                "structure_depth": max(0, len(path_snapshot) - 1),
            }
        )

    def _build_stable_id(self, *parts: str) -> str:
        raw = "::".join([str(part or "").strip() for part in parts if str(part or "").strip()])
        if not raw:
            return ""
        safe = re.sub(r"[^0-9A-Za-z\u4e00-\u9fff:_-]+", "_", raw)
        if len(safe) <= 96:
            return safe
        digest = hashlib.sha1(raw.encode("utf-8")).hexdigest()[:12]
        return f"{safe[:80]}::{digest}"

    def _serialize_tree_node(self, node: LawTreeNode) -> Dict[str, Any]:
        """序列化树节点
        
        Args:
            node: 树节点
            
        Returns:
            Dict[str, Any]: 序列化后的节点
        """
        return {
            "node_id": node.node_id,
            "level": node.level,
            "label": node.label,
            "title": node.title,
            "parent_id": node.parent_id,
            "article_num": node.article_num,
        }

    def _detect_level(self, line: str) -> Optional[str]:
        """检测文本行的层级
        
        Args:
            line: 文本行
            
        Returns:
            Optional[str]: 层级
        """
        for level, pattern in self.level_patterns.items():
            if re.match(pattern, line):
                return level
        return None

    def _extract_label(self, line: str, level: str) -> str:
        """提取标签
        
        Args:
            line: 文本行
            level: 层级
            
        Returns:
            str: 标签
        """
        if level == '条':
            return self._extract_article_num(line)

        label_patterns = {
            '编': r'^(第[一二三四五六七八九十百千万零]+编)',
            '章': r'^(第[一二三四五六七八九十百千万零]+章)',
            '节': r'^(第[一二三四五六七八九十百千万零]+节)',
            '款': r'^(\([一二三四五六七八九十百千万零]+\))',
            '项': r'^([一二三四五六七八九十百千万零]+[、.])',
            '目': r'^(\([1-9]\))',
        }
        pattern = label_patterns.get(level)
        if not pattern:
            return line
        match = re.search(pattern, line)
        if match:
            return match.group(1)
        return line

    def _extract_article_num(self, line: str) -> str:
        """提取条号
        
        Args:
            line: 文本行
            
        Returns:
            str: 条号
        """
        match = re.search(r'第[一二三四五六七八九十百千万零]+条', line)
        if match:
            return match.group()
        return ""


class MetadataEnricher:
    """元数据增强器，为文本块添加丰富的元数据"""
    
    def __init__(self):
        """初始化元数据增强器"""
        self.config = get_config()

    def enrich_metadata(self, chunk: LawChunk, law_info: Dict[str, Any]) -> LawChunk:
        """增强文本块的元数据
        
        Args:
            chunk: 法律文本块
            law_info: 法律信息
            
        Returns:
            LawChunk: 增强后的文本块
        """
        chunk.metadata.update({
            'law_name': law_info.get('law_name', law_info.get('name', '')),
            'level': law_info.get('level', ''),
            'publish_date': law_info.get('publish_date', ''),
            'effective_date': law_info.get('effective_date', ''),
            'repeal_date': law_info.get('repeal_date', ''),
            'source_law_id': law_info.get('source_law_id', ''),
            'source_hash': law_info.get('source_hash', ''),
            'version_id': law_info.get('version_id', ''),
            'version_anchor': law_info.get('version_anchor', ''),
            'is_current': law_info.get('is_current', True),
            'source_path': law_info.get('source_path', ''),
            'source_filename': law_info.get('source_filename', ''),
            'applicability_scope': law_info.get('applicability_scope', ''),
            'applicability_object': law_info.get('applicability_object', ''),
            'tags': law_info.get('tags', []),
            'article_num': chunk.article_num,
            'chunk_level': chunk.level,
        })
        
        return chunk

    def filter_by_metadata(self, chunks: List[LawChunk], filters: Dict[str, Any]) -> List[LawChunk]:
        """根据元数据过滤文本块
        
        Args:
            chunks: 文本块列表
            filters: 过滤条件
            
        Returns:
            List[LawChunk]: 过滤后的文本块列表
        """
        filtered_chunks = []
        
        for chunk in chunks:
            match = True
            
            for key, value in filters.items():
                if key not in chunk.metadata:
                    match = False
                    break
                
                if isinstance(value, list):
                    if chunk.metadata[key] not in value:
                        match = False
                        break
                else:
                    if chunk.metadata[key] != value:
                        match = False
                        break
            
            if match:
                filtered_chunks.append(chunk)
        
        return filtered_chunks

    def filter_by_time(self, chunks: List[LawChunk], event_time: datetime) -> List[LawChunk]:
        """根据时间过滤文本块
        
        Args:
            chunks: 文本块列表
            event_time: 事件时间
            
        Returns:
            List[LawChunk]: 过滤后的文本块列表
        """
        filtered_chunks = []
        
        for chunk in chunks:
            effective_date = chunk.metadata.get('effective_date')
            repeal_date = chunk.metadata.get('repeal_date')
            
            if effective_date:
                try:
                    effective_dt = datetime.strptime(effective_date, '%Y-%m-%d')
                    if event_time < effective_dt:
                        continue
                except:
                    pass
            
            if repeal_date:
                try:
                    repeal_dt = datetime.strptime(repeal_date, '%Y-%m-%d')
                    if event_time > repeal_dt:
                        continue
                except:
                    pass
            
            filtered_chunks.append(chunk)
        
        return filtered_chunks
