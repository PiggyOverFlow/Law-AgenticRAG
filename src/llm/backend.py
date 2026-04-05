from __future__ import annotations

from typing import Dict, List, Optional, Any
import logging

import requests
import torch

from config import get_config


logger = logging.getLogger(__name__)


class LLMBackend:
    """统一的 LLM 后端，支持远程 OpenAI 兼容接口和本地 Qwen + LoRA 推理。"""

    def __init__(self):
        self.config = get_config()
        self._tokenizer = None
        self._model = None

    def is_local_enabled(self) -> bool:
        llm_cfg = self.config.llm
        backend = str(getattr(llm_cfg, "backend", "remote") or "remote").lower()
        local_path = str(getattr(llm_cfg, "local_model_path", "") or "").strip()
        return bool(getattr(llm_cfg, "use_local_model", False) or backend == "local" or local_path)

    def is_available(self) -> bool:
        if self.is_local_enabled():
            return bool(str(getattr(self.config.llm, "local_model_path", "") or "").strip())
        llm_cfg = self.config.llm
        return bool(
            str(llm_cfg.base_url).strip()
            and str(llm_cfg.primary_model).strip()
            and str(llm_cfg.api_key).strip()
            and not str(llm_cfg.api_key).startswith("${")
        )

    def generate(
        self,
        messages: List[Dict[str, str]],
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        timeout: Optional[int] = None,
    ) -> str:
        if self.is_local_enabled():
            return self._generate_local(messages, temperature=temperature, max_tokens=max_tokens)
        return self._generate_remote(messages, temperature=temperature, max_tokens=max_tokens, timeout=timeout)

    def _generate_remote(
        self,
        messages: List[Dict[str, str]],
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        timeout: Optional[int] = None,
    ) -> str:
        llm_cfg = self.config.llm
        if not self.is_available():
            raise RuntimeError("远程 LLM 配置不可用")

        response = requests.post(
            f"{str(llm_cfg.base_url).rstrip('/')}/chat/completions",
            headers={
                "Authorization": f"Bearer {llm_cfg.api_key}",
                "Content-Type": "application/json",
            },
            json={
                "model": llm_cfg.primary_model,
                "temperature": llm_cfg.temperature if temperature is None else temperature,
                "max_tokens": llm_cfg.max_tokens if max_tokens is None else max_tokens,
                "messages": messages,
            },
            timeout=timeout or getattr(self.config.performance, "request_timeout", 120),
        )
        response.raise_for_status()
        return (
            response.json()
            .get("choices", [{}])[0]
            .get("message", {})
            .get("content", "")
            .strip()
        )

    def _load_local_model(self) -> None:
        if self._model is not None and self._tokenizer is not None:
            return

        llm_cfg = self.config.llm
        model_path = str(getattr(llm_cfg, "local_model_path", "") or "").strip()
        if not model_path:
            raise RuntimeError("未配置本地模型路径 local_model_path")

        from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig

        logger.info("加载本地 LLM: %s", model_path)
        self._tokenizer = AutoTokenizer.from_pretrained(
            model_path,
            trust_remote_code=getattr(llm_cfg, "trust_remote_code", True),
        )
        if self._tokenizer.pad_token is None:
            self._tokenizer.pad_token = self._tokenizer.eos_token
            self._tokenizer.pad_token_id = self._tokenizer.eos_token_id

        quantization_config = None
        if bool(getattr(llm_cfg, "load_in_4bit", False)):
            quantization_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_use_double_quant=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=getattr(torch, getattr(llm_cfg, "torch_dtype", "bfloat16")),
            )

        load_kwargs = {
            "trust_remote_code": getattr(llm_cfg, "trust_remote_code", True),
            "device_map": getattr(llm_cfg, "device_map", "auto"),
            "torch_dtype": getattr(torch, getattr(llm_cfg, "torch_dtype", "bfloat16")),
            "quantization_config": quantization_config,
        }
        attn_implementation = str(getattr(llm_cfg, "attn_implementation", "") or "").strip()
        if attn_implementation:
            load_kwargs["attn_implementation"] = attn_implementation
        try:
            self._model = AutoModelForCausalLM.from_pretrained(
                model_path,
                **load_kwargs,
            )
        except Exception as exc:
            logger.warning("本地模型首次加载失败，尝试回退加载参数: %s", exc)
            load_kwargs.pop("attn_implementation", None)
            self._model = AutoModelForCausalLM.from_pretrained(
                model_path,
                **load_kwargs,
            )

        adapter_path = str(getattr(llm_cfg, "lora_adapter_path", "") or "").strip()
        if adapter_path:
            from pathlib import Path
            if Path(adapter_path).exists():
                from peft import PeftModel

                logger.info("加载 LoRA 适配器: %s", adapter_path)
                self._model = PeftModel.from_pretrained(self._model, adapter_path)
            else:
                logger.warning("配置了 lora_adapter_path，但路径不存在: %s", adapter_path)

        self._model.eval()
        logger.info("本地 LLM 加载完成")

    def _generate_local(
        self,
        messages: List[Dict[str, str]],
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
    ) -> str:
        self._load_local_model()
        llm_cfg = self.config.llm
        generate_tokens = llm_cfg.max_tokens if max_tokens is None else max_tokens
        sample_temperature = llm_cfg.temperature if temperature is None else temperature

        prompt_text = self._render_chat_prompt(messages)
        model_inputs = self._tokenizer(
            prompt_text,
            return_tensors="pt",
            truncation=True,
            max_length=4096,
        )
        model_device = getattr(self._model, "device", None)
        if model_device is None:
            model_device = next(self._model.parameters()).device
        model_inputs = {key: value.to(model_device) for key, value in model_inputs.items()}

        with torch.no_grad():
            outputs = self._model.generate(
                **model_inputs,
                max_new_tokens=generate_tokens,
                do_sample=bool(sample_temperature and sample_temperature > 0),
                temperature=max(sample_temperature, 0.01),
                top_p=0.9,
                repetition_penalty=1.08,
                pad_token_id=self._tokenizer.pad_token_id,
                eos_token_id=self._tokenizer.eos_token_id,
            )

        prompt_length = model_inputs["input_ids"].shape[-1]
        generated_ids = outputs[0][prompt_length:]
        text = self._tokenizer.decode(generated_ids, skip_special_tokens=True).strip()
        return text

    def _render_chat_prompt(self, messages: List[Dict[str, str]]) -> str:
        if hasattr(self._tokenizer, "apply_chat_template"):
            return self._tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
            )

        parts: List[str] = []
        for message in messages:
            role = str(message.get("role", "user")).strip()
            content = str(message.get("content", "")).strip()
            parts.append(f"{role.upper()}:\n{content}")
        parts.append("ASSISTANT:\n")
        return "\n\n".join(parts)
