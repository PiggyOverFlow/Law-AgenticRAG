import os
import yaml
from pathlib import Path
from typing import Any, Dict
from dataclasses import dataclass


@dataclass
class LLMConfig:
    primary_model: str
    vision_model: str
    api_key: str
    base_url: str
    temperature: float
    max_tokens: int


@dataclass
class ASRConfig:
    model: str
    enable_timestamps: bool
    word_level_timestamps: bool
    language: str


@dataclass
class VectorDBConfig:
    type: str  # 固定为 "milvus"
    collection_name: str
    dimension: int
    uri: str = "./dataset/milvus.db"
    user: str = ""
    password: str = ""
    secure: bool = False


@dataclass
class OllamaConfig:
    base_url: str
    embedding_model: str
    reranker_model: str


@dataclass
class RetrievalConfig:
    top_k_initial: int
    top_k_final: int
    chunk_size: int
    chunk_overlap: int
    agentic_max_rounds: int = 3
    agentic_min_rounds: int = 1
    min_results_to_stop: int = 5


@dataclass
class RAGConfig:
    use_ollama: bool = True
    ollama: OllamaConfig = None
    embedding_model: str = ""
    embedding_model_path: str = ""
    reranker_model: str = ""
    reranker_model_path: str = ""
    use_local_model: bool = False
    vector_db: VectorDBConfig = None
    retrieval: RetrievalConfig = None


@dataclass
class DatabaseConfig:
    type: str
    path: str


@dataclass
class DatasetConfig:
    base_path: str
    sqlite_path: str
    ignore_folders: list


@dataclass
class VideoConfig:
    frame_rate: int
    max_frames: int


@dataclass
class ImageConfig:
    ocr_enabled: bool
    scene_description: bool


@dataclass
class AudioConfig:
    sample_rate: int
    channels: int


@dataclass
class MultimodalConfig:
    video: VideoConfig
    image: ImageConfig
    audio: AudioConfig


@dataclass
class AgentConfig:
    max_iterations: int
    tools: list
    reasoning_mode: str


@dataclass
class DocumentConfig:
    output_dir: str
    template_dir: str
    supported_types: list


@dataclass
class ExpertReviewConfig:
    enabled: bool
    criteria: list


@dataclass
class LLMJudgeConfig:
    model: str
    enabled: bool
    checks: list
    api_key: str = ""
    base_url: str = ""
    temperature: float = 0.6
    max_tokens: int = 8192
    timeout: int = 60


@dataclass
class EvaluationConfig:
    expert_review: ExpertReviewConfig
    llm_judge: LLMJudgeConfig


@dataclass
class LoggingConfig:
    level: str
    format: str
    file: str
    max_bytes: int
    backup_count: int


@dataclass
class RedisConfig:
    host: str
    port: int
    db: int


@dataclass
class CacheConfig:
    enabled: bool
    ttl: int
    backend: str
    redis: RedisConfig


@dataclass
class PerformanceConfig:
    max_concurrent_requests: int
    request_timeout: int
    retry_attempts: int
    retry_delay: int


class Config:
    def __init__(self, config_path: str = "bootstrap.yaml"):
        self.config_path = Path(config_path)
        self._raw_config = self._load_config()
        self._parse_config()

    def _load_config(self) -> Dict[str, Any]:
        if not self.config_path.exists():
            raise FileNotFoundError(f"配置文件不存在: {self.config_path}")

        with open(self.config_path, "r", encoding="utf-8") as f:
            config = yaml.safe_load(f)

        self._resolve_env_vars(config)
        return config

    def _resolve_env_vars(self, config: Dict[str, Any]):
        def resolve_value(value):
            if isinstance(value, str) and value.startswith("${") and value.endswith("}"):
                env_var = value[2:-1]
                return os.getenv(env_var, value)
            elif isinstance(value, dict):
                return {k: resolve_value(v) for k, v in value.items()}
            elif isinstance(value, list):
                return [resolve_value(item) for item in value]
            return value

        for key, value in config.items():
            config[key] = resolve_value(value)

    def _parse_config(self):
        self.llm = LLMConfig(**self._raw_config["llm"])
        self.asr = ASRConfig(**self._raw_config["asr"])
        
        vector_db_data = self._raw_config["rag"]["vector_db"]
        vector_db = VectorDBConfig(**vector_db_data)
        
        retrieval = RetrievalConfig(**self._raw_config["rag"]["retrieval"])
        
        rag_data = self._raw_config["rag"]
        use_ollama = rag_data.get("use_ollama", False)
        ollama = None
        if "ollama" in rag_data:
            ollama = OllamaConfig(**rag_data["ollama"])

        self.rag = RAGConfig(
            use_ollama=use_ollama,
            ollama=ollama,
            embedding_model=rag_data.get("embedding_model", ""),
            embedding_model_path=rag_data.get("embedding_model_path", ""),
            reranker_model=rag_data.get("reranker_model", ""),
            reranker_model_path=rag_data.get("reranker_model_path", ""),
            use_local_model=rag_data.get("use_local_model", False),
            vector_db=vector_db,
            retrieval=retrieval
        )
        
        self.database = DatabaseConfig(**self._raw_config["database"])
        self.dataset = DatasetConfig(**self._raw_config["dataset"])
        
        video = VideoConfig(**self._raw_config["multimodal"]["video"])
        image = ImageConfig(**self._raw_config["multimodal"]["image"])
        audio = AudioConfig(**self._raw_config["multimodal"]["audio"])
        self.multimodal = MultimodalConfig(video=video, image=image, audio=audio)
        
        self.agent = AgentConfig(**self._raw_config["agent"])
        self.document = DocumentConfig(**self._raw_config["document"])
        
        expert_review = ExpertReviewConfig(**self._raw_config["evaluation"]["expert_review"])
        llm_judge = LLMJudgeConfig(**self._raw_config["evaluation"]["llm_judge"])
        self.evaluation = EvaluationConfig(
            expert_review=expert_review,
            llm_judge=llm_judge
        )
        
        self.logging = LoggingConfig(**self._raw_config["logging"])
        
        redis = RedisConfig(**self._raw_config["cache"]["redis"])
        self.cache = CacheConfig(
            enabled=self._raw_config["cache"]["enabled"],
            ttl=self._raw_config["cache"]["ttl"],
            backend=self._raw_config["cache"]["backend"],
            redis=redis
        )
        
        self.performance = PerformanceConfig(**self._raw_config["performance"])

    def get(self, key: str, default: Any = None) -> Any:
        keys = key.split(".")
        value = self._raw_config
        for k in keys:
            if isinstance(value, dict) and k in value:
                value = value[k]
            else:
                return default
        return value


_global_config = None


def get_config(config_path: str = "bootstrap.yaml") -> Config:
    global _global_config
    if _global_config is None:
        _global_config = Config(config_path)
    return _global_config