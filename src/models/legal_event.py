from datetime import datetime
from typing import Optional, Literal
from pydantic import BaseModel, Field


class LegalEvent(BaseModel):
    evident_type: Literal["文本", "图片", "视频", "录音"]
    time: Optional[datetime] = Field(description="事件发生的具体时间，精确到可用范围")
    place: Optional[str] = Field(description="事件发生地，涉及管辖权判定")
    cause: str = Field(description="事件起因/基础法律关系，如：民间借贷、交通事故")
    process: str = Field(description="事件经过详述")
    result: str = Field(description="当前状态或造成的损害后果")
    source_file: Optional[str] = Field(description="来源文件路径")
    timestamp: Optional[str] = Field(description="时间戳，用于音视频定位")


class TextEvidence(BaseModel):
    content: str
    file_path: str
    extracted_at: datetime = Field(default_factory=datetime.now)


class ImageEvidence(BaseModel):
    ocr_text: str = Field(description="OCR 提取的文字内容")
    scene_description: str = Field(description="场景描述解释")
    file_path: str
    extracted_at: datetime = Field(default_factory=datetime.now)


class AudioSegment(BaseModel):
    start_time: float = Field(description="开始时间（秒）")
    end_time: float = Field(description="结束时间（秒）")
    speaker: Optional[str] = Field(description="说话人标识")
    text: str = Field(description="转写文本")
    confidence: float = Field(description="置信度")


class AudioEvidence(BaseModel):
    segments: list[AudioSegment]
    file_path: str
    duration: float = Field(description="音频总时长（秒）")
    extracted_at: datetime = Field(default_factory=datetime.now)


class VideoFrame(BaseModel):
    timestamp: float = Field(description="时间戳（秒）")
    frame_number: int = Field(description="帧编号")
    description: str = Field(description="画面描述")
    ocr_text: Optional[str] = Field(description="画面中的文字（如果有）")


class VideoEvidence(BaseModel):
    frames: list[VideoFrame]
    audio_evidence: AudioEvidence
    file_path: str
    duration: float = Field(description="视频总时长（秒）")
    extracted_at: datetime = Field(default_factory=datetime.now)


class CaseFacts(BaseModel):
    events: list[LegalEvent]
    evidence_summary: str = Field(description="证据材料汇总")
    key_disputes: list[str] = Field(description="核心争议点")
    extracted_at: datetime = Field(default_factory=datetime.now)