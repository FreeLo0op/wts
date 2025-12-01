import logging
import os
import sys
import json
import random
from dataclasses import dataclass, field
from typing import Optional, Dict

import torch
import transformers
from transformers import Trainer, AutoTokenizer, set_seed
from huggingface_hub import snapshot_download

from multi_class_model.modeling_multiclass import MultiClassModel
from multi_class_model.configuration_multiclass import MultiClassConfig
from multi_class_model.datasets import MultiClassDataset
from finetune_codes.model import KimiAudioModel

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class ModelArguments:
    model_name_or_path: Optional[str] = field(default="/mnt/pfs_l2/jieti_team/SFT/hupeng/resources/llm-base-models/Kimi-Audio-7B")
    model_path: str = field(
        default=None, metadata={"help": "Path to the pretrained model."}
    )
    num_labels: int = field(default=11, metadata={"help": "Number of labels for classification."})

@dataclass
class DataArguments:
    train_data_path: Optional[str] = field(
        default=None, metadata={"help": "Path to the training data."}
    )   
    eval_data_path: Optional[str] = field(
        default=None, metadata={"help": "Path to the evaluation data."}
    )
    eval_ratio: float = field(
        default=0.05, metadata={"help": "Ratio of evaluation data."}
    )
    max_seq_length: int = field(
        default=1024,
        metadata={
            "help": "Maximum sequence length."
        },
    )

@dataclass
class TrainingArguments(transformers.TrainingArguments):
    cache_dir: Optional[str] = field(default=None)
    optim: str = field(default="adamw_torch")
    dataloader_pin_memory: bool = field(default=False)

def make_data_module(whisper_model, text_tokenizer, data_args, model_config) -> Dict:
    """Make dataset and collator for classification."""
    rank0_print("Loading data...")

    with open(data_args.train_data_path, "r") as f:
        lines = f.readlines()
        all_data = [json.loads(line) for line in lines]

    # Process evaluation data
    eval_data = None
    if data_args.eval_data_path:
        if os.path.isfile(data_args.eval_data_path):
            with open(data_args.eval_data_path, "r") as f:
                lines = f.readlines()
                eval_data = [json.loads(line) for line in lines]
        train_data = all_data
    elif data_args.eval_ratio > 0:
        eval_data = all_data[:int(len(all_data) * data_args.eval_ratio)]
        train_data = all_data[int(len(all_data) * data_args.eval_ratio):]
    else:
        train_data = all_data

    train_dataset = MultiClassDataset(
        train_data, 
        whisper_model=whisper_model, 
        text_tokenizer=text_tokenizer, 
        max_len=data_args.max_seq_length, 
        kimia_token_offset=model_config.kimia_token_offset
    )

    eval_dataset = None
    if eval_data:
        eval_dataset = MultiClassDataset(
            eval_data, 
            whisper_model=whisper_model, 
            text_tokenizer=text_tokenizer, 
            max_len=data_args.max_seq_length, 
            kimia_token_offset=model_config.kimia_token_offset
        )

    return dict(train_dataset=train_dataset, eval_dataset=eval_dataset, data_collator=train_dataset.collate_fn)

def rank0_print(*args):
    if int(os.environ.get("LOCAL_RANK", 0)) == 0:
        print(*args)

def train():
    parser = transformers.HfArgumentParser(
        (ModelArguments, DataArguments, TrainingArguments)
    )
    (
        model_args,
        data_args,
        training_args,
    ) = parser.parse_args_into_dataclasses()

    set_seed(training_args.seed)

    logger.info(f"Loading model from {model_args.model_path}")
    
    # Load config
    config = MultiClassConfig.from_pretrained(model_args.model_path)
    config.num_labels = model_args.num_labels
    config.num_hidden_layers = 6 # Removed hardcoded layer count
    
    # Initialize model
    # We load KimiAudioModel first to get weights, then wrap/convert to MultiClassModel
    # Since MultiClassModel inherits KimiAudioModel, we can try loading directly if keys match
    # But MultiClassModel has 'score' which is new.
    # We can load KimiAudioModel and then copy weights or use from_pretrained with strict=False
    
    # Better approach: Initialize MultiClassModel and load weights from KimiAudioModel checkpoint
    # But KimiAudioModel checkpoint has 'lm_head' which we don't need.
    
    model = MultiClassModel.from_pretrained(
        model_args.model_path,
        config=config,
        ignore_mismatched_sizes=True, # For score layer vs lm_head mismatch
        trust_remote_code=True
    )
    
    # Use init_from_pretrained to load weights correctly including whisper
    # model = MultiClassModel.init_from_pretrained(
    #     model_args.model_path,
    #     model_load_kwargs={"config": config, "ignore_mismatched_sizes": True}
    # )
    # Re-initialize score head because init_from_pretrained loads weights into KimiAudioModel structure
    # but MultiClassModel adds 'score'. init_from_pretrained returns an instance of cls (MultiClassModel)
    # with loaded weights. The 'score' layer will be initialized in __init__ but weights won't be loaded from anywhere
    # unless they exist in checkpoint (which they don't).
    # So 'score' is randomly initialized, which is correct.
    # However, init_from_pretrained in KimiAudioModel (parent) does:
    # 1. Load AutoModelForCausalLM (loads LLM weights)
    # 2. Load WhisperEncoder (loads Whisper weights)
    # 3. Create cls(config) -> MultiClassModel(config)
    # 4. Load state_dict into MultiClassModel
    
    # This ensures whisper weights are loaded.
    
    # If whisper model path is hardcoded in KimiAudioModel, it will be loaded.
    # If we need to ensure it's loaded correctly:
    if not hasattr(model, 'whisper_model') or model.whisper_model is None:
        # This shouldn't happen if we inherit KimiAudioModel and call super().__init__
        pass

    # Load tokenizer
    if os.path.exists(model_args.model_name_or_path):
        cache_path = model_args.model_name_or_path
    else:
        cache_path = snapshot_download(model_args.model_name_or_path)
        
    tokenizer = AutoTokenizer.from_pretrained(
        cache_path, trust_remote_code=True
    )
    
    # Ensure config.pad_token_id is set correctly
    if tokenizer.pad_token_id is not None:
        config.pad_token_id = tokenizer.pad_token_id
    
    # Disable unused heads/layers to avoid DDP errors and save memory
    config.load_audio_detect_layers = False
    config.load_audio_head = False
    
    logger.info(f"Using pad_token_id: {config.pad_token_id}")

    # Load data
    data_module = make_data_module(
        whisper_model=model.whisper_model,
        text_tokenizer=tokenizer,
        data_args=data_args,
        model_config=config
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        **data_module
    )

    from pathlib import Path
    if list(Path(training_args.output_dir).glob("checkpoint-*")):
        trainer.train(resume_from_checkpoint=True)
    else:
        trainer.train()

    trainer.save_state()
    trainer.save_model(output_dir=training_args.output_dir)

if __name__ == "__main__":
    train()
