"""
Main script for latency-aware LoRA pruning
"""

import sys
import logging
import warnings
from datetime import datetime
from typing import List
from functools import partial

import fire
import torch
import transformers
from datasets import load_dataset
from transformers import LlamaTokenizer, AutoModelForCausalLM, TrainingArguments, AutoTokenizer
from peft import prepare_model_for_kbit_training

# Add project root to path
import pyprojroot
root_path = pyprojroot.find_root(pyprojroot.has_dir("config"))
sys.path.append(str(root_path))

from src.latency_profiling.lora import LoraConfig
from src.latency_profiling.peft_model import get_peft_model
from src.latency_profiling.utils import generate_and_tokenize_prompt

# Import components for latency aware pruning
from src.LAP.config import LatencyAwareConfig
from src.LAP.mask_manager import LatencyAwareMaskManager
from src.LAP.latency_estimator import LatencyEstimator
from src.LAP.trainer import LatencyAwareTrainer

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def setup_model_and_tokenizer(config: LatencyAwareConfig):
    """Setup model and tokenizer with LoRA"""
    
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(config.base_model, trust_remote_code=True)
    logger.info(f"Tokenizer loaded from {config.base_model}")
    tokenizer.pad_token_id = 0  # unk token
    tokenizer.padding_side = "left"
    
    # Load model
    model_kwargs = {
        "torch_dtype": torch.bfloat16,
        "device_map": "auto" if not config.load_in_8bit else None,
    }
    
    if config.load_in_8bit:
        model_kwargs["load_in_8bit"] = True
    
    model = AutoModelForCausalLM.from_pretrained(config.base_model, **model_kwargs)
    model = model.to(config.device)
    logger.info(f"Model loaded from {config.base_model}")
    
    # Prepare for k-bit training if needed
    if config.load_in_8bit:
        model = prepare_model_for_kbit_training(model)
        logger.info("Model prepared for 8-bit training")
    
    # Setup LoRA configuration
    lora_config = LoraConfig(
        r=config.lora_r,
        lora_alpha=config.lora_alpha,
        target_modules=config.lora_target_modules,
        lora_dropout=config.lora_dropout,
        bias="none",
        task_type="CAUSAL_LM",
        peft_type="LORA"
    )
    
    # Apply LoRA to model
    model = get_peft_model(model, lora_config)
    model = model.to(config.device)
    
    # Freeze base model parameters
    for param in model.base_model.parameters():
        param.requires_grad = False
    
    # Enable training for LoRA parameters
    for param in model.parameters():
        if param.requires_grad:
            param.data = param.data.float()
    
    model.print_trainable_parameters()
    
    return model, tokenizer


def setup_dataset(config: LatencyAwareConfig, tokenizer):
    """Setup training and validation datasets"""
    
    # Load dataset
    if config.data_path.endswith(".json"):
        data = load_dataset("json", data_files=config.data_path)
    else:
        data = load_dataset(config.data_path)
    
    tokenize_fn = partial(
        generate_and_tokenize_prompt,
        tokenizer=tokenizer,
        train_on_inputs=config.train_on_inputs
    )
    # Split data if validation set is requested
    if config.val_set_size > 0:
        train_val = data["train"].train_test_split(
            test_size=config.val_set_size, shuffle=True, seed=42
        )
        train_data = train_val["train"].map(tokenize_fn, batched=True, num_proc=4,remove_columns=train_val["train"].column_names, batch_size=config.tokenizer_batch_size,desc="tokenizing training data")
        val_data = train_val["test"].map(tokenize_fn, batched=True, num_proc=4, remove_columns=train_val["test"].column_names, batch_size=config.tokenizer_batch_size, desc="tokenizing validation data")
    else:
        train_data = data["train"].map(tokenize_fn, batched=True, num_proc=4, remove_columns=data["train"].column_names, batch_size=config.tokenizer_batch_size, desc="tokenizing training data")
        val_data = None
    
    return train_data, val_data


def run_latency_aware_pruning(
    # Required arguments
    base_model: str,
    data_path: str,
    
    # Pruning settings
    target_sparsity: float = 0.5,
    initial_sparsity: float = 0.0,
    latency_weight: float = 0.1,
    regularization_weight: float = 0.01,
    warmup_ratio: float = 0.1,
    cooldown_ratio: float = 0.1,
    prune_freq: int = 10,
    mask_lr: float = 0.001,
    
    # LoRA settings
    lora_r: int = 8,
    lora_alpha: int = 16,
    lora_dropout: float = 0.05,
    lora_target_modules: List[str] = None,
    
    # Training settings
    output_dir: str = "outputs/experiments/",
    batch_size: int = 8,
    micro_batch_size: int = 2,
    num_epochs: int = 10,
    learning_rate: float = 3e-4,
    cutoff_len: int = 256,
    val_set_size: int = 2000,
    
    # Hardware settings
    load_in_8bit: bool = False,
    device = "cuda" if torch.cuda.is_available() else "cpu",
    
    # Latency settings
    latency_lut_path: str = "outputs/latency_profiling/NVIDIA_H100_80GB_HBM3",
    
    # Logging
    wandb_run_name: str = f"run_{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}",
    
    # Other
    resume_from_checkpoint: str = None,
):
    """
    Run latency-aware LoRA pruning
    """
    
    # Set default LoRA target modules if not provided
    if lora_target_modules is None:
        lora_target_modules = [
            "q_proj", "k_proj", "v_proj", "o_proj", 
            "gate_proj", "up_proj", "down_proj"
        ]
    
    # Create configuration
    config = LatencyAwareConfig(
        base_model=base_model,
        data_path=data_path,
        target_sparsity=target_sparsity,
        initial_sparsity=initial_sparsity,
        latency_weight=latency_weight,
        regularization_weight=regularization_weight,
        warmup_ratio=warmup_ratio,
        cooldown_ratio=cooldown_ratio,
        prune_freq=prune_freq,
        mask_lr=mask_lr,
        lora_r=lora_r,
        lora_alpha=lora_alpha,
        lora_dropout=lora_dropout,
        lora_target_modules=lora_target_modules,
        output_dir=output_dir,
        batch_size=batch_size,
        micro_batch_size=micro_batch_size,
        num_epochs=num_epochs,
        learning_rate=learning_rate,
        cutoff_len=cutoff_len,
        val_set_size=val_set_size,
        load_in_8bit=load_in_8bit,
        latency_lut_path=latency_lut_path,
        wandb_run_name=wandb_run_name,
        device=device,
    )
    
    logger.info("Configuration:")
    logger.info(f"  Base model: {config.base_model}")
    logger.info(f"  Target sparsity: {config.target_sparsity}")
    logger.info(f"  Latency weight: {config.latency_weight}")
    logger.info(f"  Output directory: {config.output_dir}")
    
    # Setup model and tokenizer
    logger.info("Setting up model and tokenizer")
    model, tokenizer = setup_model_and_tokenizer(config)
    logger.info(f"model and tokenizer setup complete.")
    
    # Setup dataset
    logger.info("Setting up dataset...")
    train_data, val_data = setup_dataset(config,tokenizer)
    logger.info("Dataset setup complete.")
    logger.info(f"Training data size: {len(train_data)}")
    if val_data:
        logger.info(f"Validation data size: {len(val_data)}")
    
    # Get model configuration
    model_config = {
        'num_attention_heads': model.config.num_attention_heads,
        'hidden_size': model.config.hidden_size,
        'ffn_intermediate_size': getattr(model.config, 'intermediate_size', 11008),
        'head_dim': model.config.hidden_size // model.config.num_attention_heads
    }
    
    # Initialize latency estimator
    logger.info("Initializing latency estimator...")
    latency_estimator = LatencyEstimator(config.latency_lut_path, model_config)
    latency_estimator.print_lut_stats()
    
    # Initialize mask manager
    logger.info("Initializing mask manager...")
    mask_manager = LatencyAwareMaskManager(model, config, model_config)
    
    # Setup training arguments
    training_args = TrainingArguments(
        output_dir=config.output_dir,
        per_device_train_batch_size=config.micro_batch_size,
        per_device_eval_batch_size=config.micro_batch_size,
        gradient_accumulation_steps=config.gradient_accumulation_steps,
        num_train_epochs=config.num_epochs,
        learning_rate=config.learning_rate,
        fp16=True,
        logging_steps=10,
        eval_strategy="steps" if val_data else "no",
        eval_steps=100 if val_data else None,
        save_strategy="steps",
        save_steps=200,
        save_total_limit=3,
        load_best_model_at_end=False,
        report_to="wandb",
        run_name=wandb_run_name,
        dataloader_drop_last=True,
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id
    # Create data collator
    data_collator = transformers.DataCollatorForLanguageModeling(
        tokenizer=tokenizer, 
        mlm=False,
        pad_to_multiple_of=8,
        return_tensors="pt"
    )
    
    # Create trainer
    logger.info("Creating trainer...")
    trainer = LatencyAwareTrainer(
        config=config,
        mask_manager=mask_manager,
        latency_estimator=latency_estimator,
        model=model,
        args=training_args,
        train_dataset=train_data,
        eval_dataset=val_data,
        data_collator=data_collator,
    )
    
    # Disable caching for training
    model.config.use_cache = False
    
    # Start training
    logger.info("Starting training...")
    trainer.train(resume_from_checkpoint=resume_from_checkpoint)
    
    # Save final model
    logger.info("Saving final model...")
    trainer.save_model()
    
    # Log final statistics
    final_stats = trainer.log_final_model_stats()
    
    logger.info("Training completed successfully!")
    logger.info("Final statistics:")
    for key, value in final_stats.items():
        logger.info(f"  {key}: {value}")


def main():
    """Entry point for the script"""
    fire.Fire(run_latency_aware_pruning)


if __name__ == "__main__":
    main()