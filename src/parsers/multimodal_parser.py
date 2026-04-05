import os
import re
from pathlib import Path
from typing import List, Optional, Union
from datetime import datetime
import logging

from src.models.legal_event import (
    TextEvidence,
    ImageEvidence,
    AudioEvidence,
    AudioSegment,
    VideoEvidence,
    VideoFrame,
    LegalEvent,
    CaseFacts
)
from config import get_config


logger = logging.getLogger(__name__)


class MultimodalParser:
    """多模态证据解析器，支持文本、图片、音频、视频等多种格式"""
    
    def __init__(self):
        """初始化多模态解析器"""
        self.config = get_config()
        self._init_models()

    def _init_models(self):
        """初始化所需的模型（OCR、语音识别等）"""
        pass

    def parse_text(self, file_path: str) -> TextEvidence:
        """解析文本文件
        
        Args:
            file_path: 文本文件路径
            
        Returns:
            TextEvidence: 文本证据对象
        """
        logger.info(f"解析文本文件: {file_path}")
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        return TextEvidence(
            content=content,
            file_path=file_path
        )

    def parse_image(self, file_path: str) -> ImageEvidence:
        """解析图像文件，提取 OCR 文字和场景描述
        
        Args:
            file_path: 图像文件路径
            
        Returns:
            ImageEvidence: 图像证据对象
        """
        logger.info(f"解析图像文件: {file_path}")
        
        ocr_text = self._extract_text_from_image(file_path)
        scene_description = self._describe_image_scene(file_path)
        
        return ImageEvidence(
            ocr_text=ocr_text,
            scene_description=scene_description,
            file_path=file_path
        )

    def _extract_text_from_image(self, file_path: str) -> str:
        """从图像中提取文字（OCR）
        
        Args:
            file_path: 图像文件路径
            
        Returns:
            str: 提取的文字内容
        """
        return ""

    def _describe_image_scene(self, file_path: str) -> str:
        """描述图像场景
        
        Args:
            file_path: 图像文件路径
            
        Returns:
            str: 场景描述
        """
        return ""

    def parse_audio(self, file_path: str) -> AudioEvidence:
        """解析音频文件，进行语音识别
        
        Args:
            file_path: 音频文件路径
            
        Returns:
            AudioEvidence: 音频证据对象
        """
        logger.info(f"解析音频文件: {file_path}")
        
        segments = self._transcribe_audio(file_path)
        duration = self._get_audio_duration(file_path)
        
        return AudioEvidence(
            segments=segments,
            file_path=file_path,
            duration=duration
        )

    def _transcribe_audio(self, file_path: str) -> List[AudioSegment]:
        """转写音频内容
        
        Args:
            file_path: 音频文件路径
            
        Returns:
            List[AudioSegment]: 音频片段列表
        """
        return []

    def _get_audio_duration(self, file_path: str) -> float:
        """获取音频时长
        
        Args:
            file_path: 音频文件路径
            
        Returns:
            float: 音频时长（秒）
        """
        return 0.0

    def parse_video(self, file_path: str) -> VideoEvidence:
        """解析视频文件，提取帧和音频
        
        Args:
            file_path: 视频文件路径
            
        Returns:
            VideoEvidence: 视频证据对象
        """
        logger.info(f"解析视频文件: {file_path}")
        
        frames = self._extract_video_frames(file_path)
        audio_evidence = self._extract_audio_from_video(file_path)
        
        return VideoEvidence(
            frames=frames,
            audio_evidence=audio_evidence,
            file_path=file_path,
            duration=audio_evidence.duration
        )

    def _extract_video_frames(self, file_path: str) -> List[VideoFrame]:
        """提取视频帧
        
        Args:
            file_path: 视频文件路径
            
        Returns:
            List[VideoFrame]: 视频帧列表
        """
        return []

    def _extract_audio_from_video(self, file_path: str) -> AudioEvidence:
        """从视频中提取音频
        
        Args:
            file_path: 视频文件路径
            
        Returns:
            AudioEvidence: 音频证据对象
        """
        return AudioEvidence(
            segments=[],
            file_path=file_path,
            duration=0.0
        )

    def parse_file(self, file_path: str) -> Union[TextEvidence, ImageEvidence, AudioEvidence, VideoEvidence]:
        """根据文件类型自动解析文件
        
        Args:
            file_path: 文件路径
            
        Returns:
            Union[TextEvidence, ImageEvidence, AudioEvidence, VideoEvidence]: 证据对象
        """
        file_path = Path(file_path)
        suffix = file_path.suffix.lower()
        
        text_extensions = ['.txt', '.md', '.doc', '.docx', '.pdf']
        image_extensions = ['.jpg', '.jpeg', '.png', '.gif', '.bmp', '.webp']
        audio_extensions = ['.mp3', '.wav', '.m4a', '.flac', '.aac', '.ogg']
        video_extensions = ['.mp4', '.avi', '.mov', '.mkv', '.flv', '.wmv']
        
        if suffix in text_extensions:
            return self.parse_text(str(file_path))
        elif suffix in image_extensions:
            return self.parse_image(str(file_path))
        elif suffix in audio_extensions:
            return self.parse_audio(str(file_path))
        elif suffix in video_extensions:
            return self.parse_video(str(file_path))
        else:
            raise ValueError(f"不支持的文件类型: {suffix}")

    def extract_legal_events(self, evidence: Union[TextEvidence, ImageEvidence, AudioEvidence, VideoEvidence]) -> List[LegalEvent]:
        """从证据中提取法律事件
        
        Args:
            evidence: 证据对象
            
        Returns:
            List[LegalEvent]: 法律事件列表
        """
        events = []
        
        if isinstance(evidence, TextEvidence):
            events.extend(self._extract_events_from_text(evidence))
        elif isinstance(evidence, ImageEvidence):
            events.extend(self._extract_events_from_image(evidence))
        elif isinstance(evidence, AudioEvidence):
            events.extend(self._extract_events_from_audio(evidence))
        elif isinstance(evidence, VideoEvidence):
            events.extend(self._extract_events_from_video(evidence))
        
        return events

    def _extract_events_from_text(self, evidence: TextEvidence) -> List[LegalEvent]:
        """从文本证据中提取事件
        
        Args:
            evidence: 文本证据对象
            
        Returns:
            List[LegalEvent]: 法律事件列表
        """
        content = (evidence.content or "").strip()
        if not content:
            return []

        normalized = re.sub(r"\s+", " ", content)
        sentences = [s.strip() for s in re.split(r"[。！？!?\n]+", normalized) if s.strip()]

        main_time = self._extract_datetime(normalized)
        cause = self._infer_cause(normalized)

        event = LegalEvent(
            evident_type="文本",
            time=main_time,
            place=None,
            cause=cause,
            process="；".join(sentences[:20]),
            result=self._infer_result(normalized),
            source_file=evidence.file_path,
            timestamp=None,
        )
        return [event]

    def _extract_events_from_image(self, evidence: ImageEvidence) -> List[LegalEvent]:
        """从图片证据中提取事件
        
        Args:
            evidence: 图片证据对象
            
        Returns:
            List[LegalEvent]: 法律事件列表
        """
        return []

    def _extract_events_from_audio(self, evidence: AudioEvidence) -> List[LegalEvent]:
        """从音频证据中提取事件
        
        Args:
            evidence: 音频证据对象
            
        Returns:
            List[LegalEvent]: 法律事件列表
        """
        events = []
        for segment in evidence.segments:
            event = LegalEvent(
                evident_type="录音",
                time=None,
                place=None,
                cause="",
                process=segment.text,
                result="",
                source_file=evidence.file_path,
                timestamp=f"{segment.start_time:.2f}-{segment.end_time:.2f}"
            )
            events.append(event)
        return events

    def _extract_events_from_video(self, evidence: VideoEvidence) -> List[LegalEvent]:
        """从视频证据中提取事件
        
        Args:
            evidence: 视频证据对象
            
        Returns:
            List[LegalEvent]: 法律事件列表
        """
        events = []
        
        for frame in evidence.frames:
            event = LegalEvent(
                evident_type="视频",
                time=None,
                place=None,
                cause="",
                process=frame.description,
                result="",
                source_file=evidence.file_path,
                timestamp=f"{frame.timestamp:.2f}s"
            )
            events.append(event)
        
        for segment in evidence.audio_evidence.segments:
            event = LegalEvent(
                evident_type="录音",
                time=None,
                place=None,
                cause="",
                process=segment.text,
                result="",
                source_file=evidence.file_path,
                timestamp=f"{segment.start_time:.2f}-{segment.end_time:.2f}"
            )
            events.append(event)
        
        return events

    def parse_case_facts(self, file_paths: List[str]) -> CaseFacts:
        """解析多个证据文件，整合为案件事实
        
        Args:
            file_paths: 证据文件路径列表
            
        Returns:
            CaseFacts: 案件事实对象
        """
        all_events = []
        evidence_summary_parts = []
        
        for file_path in file_paths:
            try:
                evidence = self.parse_file(file_path)
                events = self.extract_legal_events(evidence)
                all_events.extend(events)
                evidence_summary_parts.append(f"文件: {file_path}")
                evidence_summary_parts.append(f"类型: {evidence.__class__.__name__}")
                if isinstance(evidence, TextEvidence):
                    snippet = re.sub(r"\s+", " ", evidence.content).strip()[:800]
                    evidence_summary_parts.append(f"内容摘要: {snippet}")
                elif isinstance(evidence, ImageEvidence):
                    image_summary = " ".join([
                        (evidence.ocr_text or "").strip()[:300],
                        (evidence.scene_description or "").strip()[:200],
                    ]).strip()
                    if image_summary:
                        evidence_summary_parts.append(f"内容摘要: {image_summary}")
                elif isinstance(evidence, AudioEvidence):
                    transcript = " ".join([seg.text for seg in evidence.segments[:20] if seg.text])
                    if transcript:
                        evidence_summary_parts.append(f"内容摘要: {transcript[:800]}")
                elif isinstance(evidence, VideoEvidence):
                    frame_text = " ".join([f.description for f in evidence.frames[:20] if f.description])
                    audio_text = " ".join([seg.text for seg in evidence.audio_evidence.segments[:20] if seg.text])
                    merged = f"{frame_text} {audio_text}".strip()
                    if merged:
                        evidence_summary_parts.append(f"内容摘要: {merged[:800]}")
            except Exception as e:
                logger.error(f"解析文件失败 {file_path}: {e}")
        
        key_disputes = self._extract_key_disputes(all_events)
        
        return CaseFacts(
            events=all_events,
            evidence_summary="\n".join(evidence_summary_parts),
            key_disputes=key_disputes
        )

    def _extract_key_disputes(self, events: List[LegalEvent]) -> List[str]:
        """从事件中提取关键争议点
        
        Args:
            events: 法律事件列表
            
        Returns:
            List[str]: 关键争议点列表
        """
        if not events:
            return []

        merged_text = "\n".join([
            " ".join([
                (event.cause or ""),
                (event.process or ""),
                (event.result or ""),
            ])
            for event in events
        ])

        disputes: List[str] = []
        if any(k in merged_text for k in ["借款", "民间借贷", "出借"]):
            disputes.append("借贷法律关系是否成立")

        principal_match = re.search(r"(\d+(?:\.\d+)?)\s*(万)?\s*元", merged_text)
        if principal_match:
            amount = principal_match.group(1)
            unit = principal_match.group(2) or ""
            disputes.append(f"借款本金金额为{amount}{unit}元")

        monthly_rate = re.search(r"月利率\s*([0-9]+(?:\.[0-9]+)?%)", merged_text)
        annual_rate = re.search(r"年利率\s*([0-9]+(?:\.[0-9]+)?%)", merged_text)
        if monthly_rate:
            disputes.append(f"约定利率为月利率{monthly_rate.group(1)}")
        elif annual_rate:
            disputes.append(f"约定利率为年利率{annual_rate.group(1)}")

        term_match = re.search(r"借款期限\s*为\s*([0-9一二三四五六七八九十两]+个?月)", merged_text)
        if term_match:
            disputes.append(f"借款期限为{term_match.group(1)}")

        if any(k in merged_text for k in ["未按约定归还", "未偿还", "拒不归还", "逾期"]):
            disputes.append("被告是否应承担逾期还款责任")

        return list(dict.fromkeys(disputes))[:8]

    def _extract_datetime(self, text: str) -> Optional[datetime]:
        """从文本中提取日期时间
        
        Args:
            text: 文本内容
            
        Returns:
            Optional[datetime]: 提取的日期时间对象
        """
        match = re.search(r"(20\d{2})年(\d{1,2})月(\d{1,2})日", text)
        if not match:
            return None
        try:
            return datetime(int(match.group(1)), int(match.group(2)), int(match.group(3)))
        except Exception:
            return None

    def _infer_cause(self, text: str) -> str:
        """推断案件起因/法律关系
        
        Args:
            text: 文本内容
            
        Returns:
            str: 案件起因
        """
        if any(k in text for k in ["借款", "出借", "民间借贷"]):
            return "民间借贷"
        if any(k in text for k in ["劳动", "工资", "辞退"]):
            return "劳动争议"
        if any(k in text for k in ["交通事故", "碰撞", "交警"]):
            return "交通事故"
        return "待识别"

    def _infer_result(self, text: str) -> str:
        """推断案件结果
        
        Args:
            text: 文本内容
            
        Returns:
            str: 案件结果
        """
        if any(k in text for k in ["未按约定归还", "未偿还", "逾期"]):
            return "借款到期未清偿"
        return "待补充"