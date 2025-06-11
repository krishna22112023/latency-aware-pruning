from dataclasses import dataclass, field
from typing import List


@dataclass
class LatencyAwareConfig:
    """Configuration for differentiable latency-aware structured pruning with LoRA"""
    
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
    #pruning_steps: int = 5000
    initial_sparsity: float = 0.0
    warmup_ratio: float = 0.1
    cooldown_ratio: float = 0.1
    prune_freq: int = 10
    
    # Latency-aware settings
    latency_weight: float = 0.1  # λ parameter in loss function
    regularization_weight: float = 0.01  # γ parameter
    latency_lut_path: str = "outputs/latency_profiling/NVIDIA_H100_80GB_HBM3/"
    
    # Differentiable latency settings
    latency_method: str = "linear"  # 'linear', 'soft', 'gumbel'
    
    # Gumbel softmax settings
    initial_temperature: float = 3.0  # Start with high temperature for exploration
    final_temperature: float = 0.5   # End with low temperature for exploitation
    gumbel_temperature: float = 1.0  # Default temperature if not annealing
    
    # Mask update settings
    mask_lr: float = 0.001  # α parameter for base learning rate
    importance_beta: float = 0.9  # β parameter for moving average
    epsilon: float = 1e-12  # numerical stability
    
    # Optional regularization settings
    entropy_weight: float = 0.01  # Entropy regularization for mask diversity
    variance_weight: float = 0.001  # Variance regularization for balanced masks
    
    # Training settings
    batch_size: int = 64
    micro_batch_size: int = 8
    num_epochs: int = 2
    learning_rate: float = 1e-4
    warmup_steps: int = 10
    cutoff_len: int = 256
    
    # Hardware settings
    device: str = "cuda:0"
    load_in_8bit: bool = False
    
    # Data settings
    data_path: str = "MBZUAI/LaMini-instruction"
    output_dir: str = "output_latency_aware"
    max_train_samples: int = 25000
    val_set_size: int = 2000
    train_on_inputs: bool = False
    tokenizer_batch_size: int = 32
    
    # Logging
    wandb_project: str = "latency_aware_pruning"
    wandb_run_name: str = ""
    
    # Advanced settings
    use_straight_through_estimator: bool = False  # For backward compatibility
    temperature_annealing: bool = True  # Whether to anneal temperature during training
    
    def __post_init__(self):
        if not self.data_path:
            raise ValueError("data_path must be specified")
        
        # Calculate gradient accumulation steps
        self.gradient_accumulation_steps = self.batch_size // self.micro_batch_size
        
        # Validate latency method
        valid_methods = ['linear', 'soft', 'gumbel']
        if self.latency_method not in valid_methods:
            raise ValueError(f"latency_method must be one of {valid_methods}, got {self.latency_method}")
        
        # Adjust learning rates for differentiable training
        if self.latency_method in ['soft', 'gumbel']:
            # These methods may need lower learning rates for stability
            if self.mask_lr > 0.01:
                print(f"Warning: mask_lr ({self.mask_lr}) may be too high for {self.latency_method} method")
        
        # Validate temperature settings
        if self.temperature_annealing:
            if self.initial_temperature <= self.final_temperature:
                raise ValueError("initial_temperature must be greater than final_temperature for annealing")
    
    def get_method_specific_params(self):
        """Get parameters specific to the chosen latency method"""
        if self.latency_method == 'linear':
            return {
                'description': 'Linear approximation of latency (fastest, smoothest gradients)',
                'requires_lut_fitting': True,
                'gradient_quality': 'smooth',
                'computational_cost': 'low'
            }
        elif self.latency_method == 'soft':
            return {
                'description': 'Soft sampling over configuration space',
                'requires_lut_fitting': False,
                'gradient_quality': 'moderate',
                'computational_cost': 'medium'
            }
        elif self.latency_method == 'gumbel':
            return {
                'description': 'Gumbel softmax sampling',
                'requires_lut_fitting': False,
                'gradient_quality': 'good',
                'computational_cost': 'medium'
            }
        else:
            return {}
    
    def print_config_summary(self):
        """Print a summary of the configuration"""
        print("=" * 60)
        print("DIFFERENTIABLE LATENCY-AWARE PRUNING CONFIGURATION")
        print("=" * 60)
        print(f"Model: {self.base_model}")
        print(f"LoRA Config: r={self.lora_r}, alpha={self.lora_alpha}")
        print(f"Target Sparsity: {self.target_sparsity:.1%}")
        print(f"Latency Method: {self.latency_method}")
        print(f"Latency Weight: {self.latency_weight}")
        
        method_params = self.get_method_specific_params()
        if method_params:
            print(f"Method Description: {method_params['description']}")
        
        if self.latency_method in ['soft', 'gumbel']:
            print(f"Temperature: {self.initial_temperature:.1f} → {self.final_temperature:.1f}")
        
        print(f"Training: {self.num_epochs} epochs, batch_size={self.batch_size}")
        print(f"Output: {self.output_dir}")
        print("=" * 60)