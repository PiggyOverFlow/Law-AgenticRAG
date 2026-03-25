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
    def __init__(self):
        self.config = get_config()
        self._init_models()

    def _init_models(self):
        pass

    def parse_text(self, file_path: str) -> TextEvidence:
        logger.info(f"解析文本文件: {file_path}")
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        return TextEvidence(
            content=content,
            file_path=file_path
        )

    def parse_image(self, file_path: str) -> ImageEvidence:
        logger.info(f"解析图像文件: {file_path}")
        
        ocr_text = self._extract_text_from_image(file_path)
        scene_description = self._describe_image_scene(file_path)
        
        return ImageEvidence(
            ocr_text=ocr_text,
            scene_description=scene_description,
            file_path=file_path
        )

    def _extract_text_from_image(self, file_path: str) -> str:
        return ""

    def _describe_image_scene(self, file_path: str) -> str:
        return ""

    def parse_audio(self, file_path: str) -> AudioEvidence:
        logger.info(f"解析音频文件: {file_path}")
        
        segments = self._transcribe_audio(file_path)
        duration = self._get_audio_duration(file_path)
        
        return AudioEvidence(
            segments=segments,
            file_path=file_path,
            duration=duration
        )

    def _transcribe_audio(self, file_path: str) -> List[AudioSegment]:
        return []

    def _get_audio_duration(self, file_path: str) -> float:
        return 0.0

    def parse_video(self, file_path: str) -> VideoEvidence:
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
        return []

    def _extract_audio_from_video(self, file_path: str) -> AudioEvidence:
        return AudioEvidence(
            segments=[],
            file_path=file_path,
            duration=0.0
        )

    def parse_file(self, file_path: str) -> Union[TextEvidence, ImageEvidence, AudioEvidence, VideoEvidence]:
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
        return []

    def _extract_events_from_image(self, evidence: ImageEvidence) -> List[LegalEvent]:
        return []

    def _extract_events_from_audio(self, evidence: AudioEvidence) -> List[LegalEvent]:
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
        all_events = []
        evidence_summary_parts = []
        
        for file_path in file_paths:
            try:
                evidence = self.parse_file(file_path)
                events = self.extract_legal_events(evidence)
                all_events.extend(events)
                evidence_summary_parts.append(f"文件: {file_path}")
                evidence_summary_parts.append(f"类型: {evidence.__class__.__name__}")
            except Exception as e:
                logger.error(f"解析文件失败 {file_path}: {e}")
        
        key_disputes = self._extract_key_disputes(all_events)
        
        return CaseFacts(
            events=all_events,
            evidence_summary="\n".join(evidence_summary_parts),
            key_disputes=key_disputes
        )

    def _extract_key_disputes(self, events: List[LegalEvent]) -> List[str]:
        return []