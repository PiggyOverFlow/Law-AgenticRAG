from typing import List, Dict, Any, Optional
from datetime import datetime
from dataclasses import dataclass
import logging
import re

from config import get_config


logger = logging.getLogger(__name__)


@dataclass
class LawChunk:
    chunk_id: str
    law_name: str
    article_num: str
    content: str
    level: str  # 编、章、节、条、款、项、目
    metadata: Dict[str, Any]
    
    def __repr__(self):
        return f"<LawChunk {self.law_name} {self.article_num}>"


@dataclass
class RetrievalResult:
    chunk: LawChunk
    score: float
    rank: int


class LawChunker:
    def __init__(self):
        self.config = get_config()
        
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
        chunks = []
        lines = law_content.split('\n')
        
        current_chunk = None
        current_level = None
        current_article_num = None
        chunk_lines = []
        
        for line in lines:
            line = line.strip()
            if not line:
                continue
            
            detected_level = self._detect_level(line)
            
            if detected_level == '条':
                if current_chunk:
                    current_chunk.content = '\n'.join(chunk_lines).strip()
                    chunks.append(current_chunk)
                
                article_num = self._extract_article_num(line)
                current_article_num = article_num
                current_chunk = LawChunk(
                    chunk_id=f"{law_name}_{article_num}",
                    law_name=law_name,
                    article_num=article_num,
                    content="",
                    level="条",
                    metadata=metadata.copy()
                )
                chunk_lines = [line]
                current_level = "条"
            
            elif detected_level and detected_level in ['编', '章', '节']:
                if current_chunk:
                    current_chunk.content = '\n'.join(chunk_lines).strip()
                    chunks.append(current_chunk)
                
                current_chunk = None
                current_article_num = None
                chunk_lines = []
                current_level = detected_level
            
            else:
                if current_chunk:
                    chunk_lines.append(line)
        
        if current_chunk:
            current_chunk.content = '\n'.join(chunk_lines).strip()
            chunks.append(current_chunk)
        
        return chunks

    def _detect_level(self, line: str) -> Optional[str]:
        for level, pattern in self.level_patterns.items():
            if re.match(pattern, line):
                return level
        return None

    def _extract_article_num(self, line: str) -> str:
        match = re.search(r'第[一二三四五六七八九十百千万零]+条', line)
        if match:
            return match.group()
        return ""


class MetadataEnricher:
    def __init__(self):
        self.config = get_config()

    def enrich_metadata(self, chunk: LawChunk, law_info: Dict[str, Any]) -> LawChunk:
        chunk.metadata.update({
            'law_name': law_info.get('name', ''),
            'level': law_info.get('level', ''),
            'publish_date': law_info.get('publish_date', ''),
            'effective_date': law_info.get('effective_date', ''),
            'repeal_date': law_info.get('repeal_date', ''),
            'applicability_scope': law_info.get('applicability_scope', ''),
            'applicability_object': law_info.get('applicability_object', ''),
            'tags': law_info.get('tags', []),
            'article_num': chunk.article_num,
            'chunk_level': chunk.level
        })
        
        return chunk

    def filter_by_metadata(self, chunks: List[LawChunk], filters: Dict[str, Any]) -> List[LawChunk]:
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