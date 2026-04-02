import os
import json
import torch
from typing import Dict, List, Optional, Any
from pathlib import Path
from dataclasses import dataclass, field
from datetime import datetime
import logging

from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    TrainingArguments,
    Trainer,
    DataCollatorForSeq2Seq,
    BitsAndBytesConfig
)
from peft import (
    LoraConfig,
    get_peft_model,
    TaskType,
    prepare_model_for_kbit_training
)
from datasets import Dataset
from config import get_config

logger = logging.getLogger(__name__)


@dataclass
class QLoraTrainingConfig:
    model_name_or_path: str = "Qwen/Qwen3-8B-Instruct"
    output_dir: str = "./output/qlora_finetuned"
    data_path: str = "./dataset/finetuning_data.json"
    
    lora_r: int = 64
    lora_alpha: int = 16
    lora_dropout: float = 0.05
    lora_target_modules: List[str] = field(default_factory=lambda: ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"])
    
    per_device_train_batch_size: int = 4
    gradient_accumulation_steps: int = 2
    num_train_epochs: int = 3
    learning_rate: float = 2e-4
    warmup_ratio: float = 0.03
    logging_steps: int = 10
    save_steps: int = 100
    eval_steps: int = 100
    
    max_seq_length: int = 2048
    
    bnb_4bit_compute_dtype: str = "bfloat16"
    bnb_4bit_quant_type: str = "nf4"
    bnb_4bit_use_double_quant: bool = True
    
    resume_from_checkpoint: Optional[str] = None
    deepspeed: Optional[str] = None


class QLoraQwenTrainer:
    def __init__(self, config: Optional[QLoraTrainingConfig] = None):
        self.config = config or QLoraTrainingConfig()
        self.project_config = get_config()
        self._setup_directories()
        self._setup_logging()
        
        self.tokenizer = None
        self.model = None
        self.trainer = None

    def _setup_directories(self):
        Path(self.config.output_dir).mkdir(parents=True, exist_ok=True)
        Path(self.config.data_path).parent.mkdir(parents=True, exist_ok=True)

    def _setup_logging(self):
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )

    def load_model_and_tokenizer(self):
        logger.info(f"加载模型: {self.config.model_name_or_path}")
        
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.config.model_name_or_path,
            trust_remote_code=True,
            padding_side="right"
        )
        
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
            self.tokenizer.pad_token_id = self.tokenizer.eos_token_id
        
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type=self.config.bnb_4bit_quant_type,
            bnb_4bit_compute_dtype=getattr(torch, self.config.bnb_4bit_compute_dtype),
            bnb_4bit_use_double_quant=self.config.bnb_4bit_use_double_quant
        )
        
        self.model = AutoModelForCausalLM.from_pretrained(
            self.config.model_name_or_path,
            quantization_config=bnb_config,
            device_map="auto",
            trust_remote_code=True,
            attn_implementation="flash_attention_2"
        )
        
        self.model = prepare_model_for_kbit_training(self.model)
        
        logger.info("模型和分词器加载完成")

    def setup_lora(self):
        logger.info("配置QLoRA参数")
        
        lora_config = LoraConfig(
            r=self.config.lora_r,
            lora_alpha=self.config.lora_alpha,
            lora_dropout=self.config.lora_dropout,
            target_modules=self.config.lora_target_modules,
            bias="none",
            task_type=TaskType.CAUSAL_LM,
            inference_mode=False
        )
        
        self.model = get_peft_model(self.model, lora_config)
        
        self.model.print_trainable_parameters()
        
        logger.info("QLoRA配置完成")

    def load_training_data(self, data_path: Optional[str] = None) -> Dataset:
        data_path = data_path or self.config.data_path
        
        if not os.path.exists(data_path):
            logger.warning(f"训练数据文件不存在: {data_path}")
            logger.info("创建示例训练数据")
            self._create_sample_data(data_path)
        
        with open(data_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        logger.info(f"加载了 {len(data)} 条训练数据")
        
        dataset = Dataset.from_list(data)
        
        processed_dataset = dataset.map(
            self._preprocess_function,
            batched=True,
            remove_columns=dataset.column_names,
            desc="处理训练数据"
        )
        
        return processed_dataset

    def _create_sample_data(self, data_path: str):
        sample_data = [
            {
                "instruction": "根据以下法律条文，回答问题。",
                "input": "《中华人民共和国民法典》第一千零七十七条：自婚姻登记机关收到离婚登记申请之日起三十日内，任何一方不愿意离婚的，可以向婚姻登记机关撤回离婚登记申请。",
                "output": "根据《民法典》第1077条的规定，离婚登记申请后有30天的冷静期。在这30天内，任何一方都可以撤回离婚申请。"
            },
            {
                "instruction": "请解释以下法律概念。",
                "input": "什么是诉讼时效？",
                "output": "诉讼时效是指权利人在法定期间内不行使权利，该期间届满后，义务人可以拒绝履行义务的法律制度。根据《民法典》规定，普通诉讼时效期间为3年，自权利人知道或者应当知道权利受到损害以及义务人之日起计算。"
            },
            {
                "instruction": "根据法律条文判断以下说法是否正确。",
                "input": "用人单位可以随意解除劳动合同。",
                "output": "不正确。根据《劳动合同法》的规定，用人单位解除劳动合同必须符合法定情形，如劳动者严重违反规章制度、不能胜任工作等，且需要遵循法定程序，不能随意解除。"
            },
            {
                "instruction": "请根据案情分析法律责任。",
                "input": "张某驾驶机动车闯红灯撞伤行人李某，造成李某轻伤。交警认定张某负全责。",
                "output": "根据《道路交通安全法》和《侵权责任法》，张某驾驶机动车闯红灯违反交通法规，负全部责任，应当承担民事赔偿责任，赔偿李某的医疗费、误工费等损失。如果情节严重，还可能面临行政处罚。"
            },
            {
                "instruction": "解释法律术语。",
                "input": "什么是不可抗力？",
                "output": "不可抗力是指不能预见、不能避免并不能克服的客观情况。根据《民法典》第180条，因不可抗力不能履行民事义务的，不承担民事责任，法律另有规定的除外。常见的不可抗力包括自然灾害（如地震、台风）、社会异常事件（如战争、罢工）等。"
            }
        ]
        
        with open(data_path, 'w', encoding='utf-8') as f:
            json.dump(sample_data, f, ensure_ascii=False, indent=2)
        
        logger.info(f"已创建示例训练数据: {data_path}")

    def _preprocess_function(self, examples):
        instructions = examples["instruction"]
        inputs = examples.get("input", [""] * len(instructions))
        outputs = examples["output"]
        
        prompts = []
        for instruction, input_text in zip(instructions, inputs):
            if input_text:
                prompt = f"<|im_start|>system\n你是一个专业的法律助手，请根据法律条文和案例提供准确的法律建议。<|im_end|>\n<|im_start|>user\n{instruction}\n{input_text}<|im_end|>\n<|im_start|>assistant\n"
            else:
                prompt = f"<|im_start|>system\n你是一个专业的法律助手，请根据法律条文和案例提供准确的法律建议。<|im_end|>\n<|im_start|>user\n{instruction}<|im_end|>\n<|im_start|>assistant\n"
            prompts.append(prompt)
        
        model_inputs = self.tokenizer(
            prompts,
            max_length=self.config.max_seq_length,
            padding=True,
            truncation=True,
            return_tensors=None
        )
        
        labels = self.tokenizer(
            outputs,
            max_length=self.config.max_seq_length,
            padding=True,
            truncation=True,
            return_tensors=None
        )
        
        model_inputs["labels"] = labels["input_ids"]
        
        return model_inputs

    def setup_trainer(self, train_dataset: Dataset, eval_dataset: Optional[Dataset] = None):
        training_args = TrainingArguments(
            output_dir=self.config.output_dir,
            num_train_epochs=self.config.num_train_epochs,
            per_device_train_batch_size=self.config.per_device_train_batch_size,
            gradient_accumulation_steps=self.config.gradient_accumulation_steps,
            learning_rate=self.config.learning_rate,
            warmup_ratio=self.config.warmup_ratio,
            logging_steps=self.config.logging_steps,
            save_steps=self.config.save_steps,
            eval_steps=self.config.eval_steps,
            save_total_limit=3,
            evaluation_strategy="steps" if eval_dataset else "no",
            fp16=False,
            bf16=True if self.config.bnb_4bit_compute_dtype == "bfloat16" else False,
            max_grad_norm=0.3,
            weight_decay=0.0,
            adam_beta1=0.9,
            adam_beta2=0.95,
            adam_epsilon=1e-8,
            max_steps=-1,
            lr_scheduler_type="cosine",
            logging_dir=f"{self.config.output_dir}/logs",
            report_to=["tensorboard"],
            load_best_model_at_end=True if eval_dataset else False,
            metric_for_best_model="eval_loss" if eval_dataset else None,
            greater_is_better=False if eval_dataset else None,
            ddp_find_unused_parameters=False,
            deepspeed=self.config.deepspeed,
            resume_from_checkpoint=self.config.resume_from_checkpoint,
            optim="paged_adamw_32bit",
            gradient_checkpointing=True,
            gradient_checkpointing_kwargs={"use_reentrant": False},
            dataloader_num_workers=4,
            dataloader_pin_memory=True,
        )
        
        data_collator = DataCollatorForSeq2Seq(
            tokenizer=self.tokenizer,
            model=self.model,
            padding=True
        )
        
        self.trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            data_collator=data_collator,
            tokenizer=self.tokenizer,
        )
        
        logger.info("Trainer配置完成")

    def train(self):
        logger.info("开始训练")
        
        self.trainer.train()
        
        logger.info("训练完成")

    def save_model(self, output_dir: Optional[str] = None):
        output_dir = output_dir or self.config.output_dir
        
        logger.info(f"保存QLoRA模型到: {output_dir}")
        
        self.trainer.save_model(output_dir)
        self.tokenizer.save_pretrained(output_dir)
        
        config_path = Path(output_dir) / "qlora_config.json"
        qlora_config = {
            "model_name_or_path": self.config.model_name_or_path,
            "lora_r": self.config.lora_r,
            "lora_alpha": self.config.lora_alpha,
            "lora_dropout": self.config.lora_dropout,
            "target_modules": self.config.lora_target_modules,
            "bnb_4bit_quant_type": self.config.bnb_4bit_quant_type,
            "bnb_4bit_compute_dtype": self.config.bnb_4bit_compute_dtype,
            "bnb_4bit_use_double_quant": self.config.bnb_4bit_use_double_quant,
            "training_date": datetime.now().isoformat()
        }
        
        with open(config_path, 'w', encoding='utf-8') as f:
            json.dump(qlora_config, f, ensure_ascii=False, indent=2)
        
        logger.info("QLoRA模型保存完成")

    def merge_and_save(self, output_dir: Optional[str] = None):
        output_dir = output_dir or f"{self.config.output_dir}_merged"
        
        logger.info(f"合并QLoRA权重并保存到: {output_dir}")
        
        merged_model = self.trainer.model.merge_and_unload()
        
        merged_model.save_pretrained(output_dir)
        self.tokenizer.save_pretrained(output_dir)
        
        logger.info("QLoRA模型合并并保存完成")

    def generate(self, prompt: str, max_new_tokens: int = 512, temperature: float = 0.7) -> str:
        if self.model is None or self.tokenizer is None:
            raise ValueError("模型和分词器未加载，请先调用 load_model_and_tokenizer()")
        
        formatted_prompt = f"<|im_start|>system\n你是一个专业的法律助手，请根据法律条文和案例提供准确的法律建议。<|im_end|>\n<|im_start|>user\n{prompt}<|im_end|>\n<|im_start|>assistant\n"
        
        inputs = self.tokenizer(
            formatted_prompt,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=self.config.max_seq_length
        ).to(self.model.device)
        
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                do_sample=True,
                top_p=0.9,
                repetition_penalty=1.1,
                pad_token_id=self.tokenizer.pad_token_id,
                eos_token_id=self.tokenizer.eos_token_id
            )
        
        response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        response = response.split("<|im_start|>assistant\n")[-1]
        response = response.split("<|im_end|>")[0]
        
        return response.strip()

    def run_full_pipeline(self):
        logger.info("开始完整的QLoRA微调流程")
        
        self.load_model_and_tokenizer()
        self.setup_lora()
        
        train_dataset = self.load_training_data()
        
        self.setup_trainer(train_dataset)
        self.train()
        self.save_model()
        
        logger.info("QLoRA微调流程完成")


def create_training_data_from_laws(law_chunks: List[Dict[str, Any]], output_path: str):
    training_data = []
    
    for chunk in law_chunks:
        content = chunk.get("content", "")
        metadata = chunk.get("metadata", {})
        
        instruction = "请根据以下法律条文回答相关问题。"
        input_text = f"法律条文：{content}"
        
        law_name = metadata.get("law_name", "未知法律")
        category = metadata.get("category", "")
        
        output = f"根据{law_name}（{category}）的规定，该条文涉及{category}领域的法律规范。"
        
        training_data.append({
            "instruction": instruction,
            "input": input_text,
            "output": output
        })
    
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(training_data, f, ensure_ascii=False, indent=2)
    
    logger.info(f"已从法律数据创建训练数据集: {output_path}, 共 {len(training_data)} 条")


if __name__ == "__main__":
    config = QLoraTrainingConfig(
        model_name_or_path="Qwen/Qwen2.5-7B-Instruct",
        output_dir="./output/qlora_qwen_law",
        data_path="./dataset/law_finetuning_data.json",
        num_train_epochs=3,
        per_device_train_batch_size=4,
        gradient_accumulation_steps=2,
        learning_rate=2e-4,
        max_seq_length=2048,
        lora_r=64,
        lora_alpha=16,
        bnb_4bit_compute_dtype="bfloat16",
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True
    )
    
    trainer = QLoraQwenTrainer(config)
    trainer.run_full_pipeline()