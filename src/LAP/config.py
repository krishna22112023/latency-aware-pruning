from dataclasses import dataclass, field
from typing import List


@dataclass
class LatencyAwareConfig:
    """Configuration for latency-aware structured pruning with LoRA"""
    
    # Model settings
    base_model: str = "meta-llama/Llama-3.2-1B"
    lora_r: int = 8
    lora_alpha: int = 16
    lora_dropout: float = 0.0
    lora_target_modules: List[str] = field(default_factory=lambda: [
        "q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"
    ])
    
    # Pruning settings
    target_sparsity: float = 0.5
    pruning_steps: int = 5000
    initial_sparsity: float = 0.0
    warmup_ratio: float = 0.1
    cooldown_ratio: float = 0.1
    prune_freq: int = 10
    
    # Latency-aware settings
    latency_weight: float = 0.1  # λ parameter in loss function
    regularization_weight: float = 0.01  # γ parameter
    latency_lut_path: str = "outputs/latency_profiling/NVIDIA_H100_80GB_HBM3/"
    
    # Mask update settings
    mask_lr: float = 0.001  # α parameter for base learning rate
    importance_beta: float = 0.9  # β parameter for moving average
    epsilon: float = 1e-12  # numerical stability
    
    # Training settings
    batch_size: int = 8
    micro_batch_size: int = 2
    num_epochs: int = 10
    learning_rate: float = 3e-4
    warmup_steps: int = 10
    cutoff_len: int = 256
    
    # Hardware settings
    device: str = "cuda"
    load_in_8bit: bool = False
    
    # Data settings
    data_path: str = "MBZUAI/LaMini-instruction"
    output_dir: str = "output_latency_aware"
    val_set_size: int = 2000
    train_on_inputs: bool = False
    tokenizer_batch_size: int = 32
    
    # Logging
    wandb_project: str = "latency_aware_pruning"
    wandb_run_name: str = ""
    
    def __post_init__(self):
        if not self.data_path:
            raise ValueError("data_path must be specified")
        
        # Calculate gradient accumulation steps
        self.gradient_accumulation_steps = self.batch_size // self.micro_batch_size