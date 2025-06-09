import torch
import torch.nn as nn
from transformers import Trainer
from transformers.trainer_utils import PREFIX_CHECKPOINT_DIR
import os
import logging
from typing import Dict, Optional, Tuple
from src.LAP.config import LatencyAwareConfig
from src.LAP.mask_manager import LatencyAwareMaskManager
from src.LAP.latency_estimator import LatencyEstimator
from src.LAP.lora_importance import compute_lora_guided_importance, compute_mask_gradients
from src.LAP.loss import LatencyAwareLoss, compute_schedule_sparsity

logger = logging.getLogger(__name__)

try:
    from apex import amp
    APEX_AVAILABLE = True
except ImportError:
    APEX_AVAILABLE = False

class LatencyAwareTrainer(Trainer):
    """
    Extends Hugging Face Trainer for latency-aware structured pruning with LoRA
    """
    
    def __init__(self, config: LatencyAwareConfig, mask_manager: LatencyAwareMaskManager,
                 latency_estimator: LatencyEstimator, **kwargs):
        super().__init__(**kwargs)
        
        self.config = config
        self.args = kwargs.get('args')
        self.train_dataset = kwargs.get('train_dataset')
        self.mask_manager = mask_manager
        self.latency_estimator = latency_estimator
        
        # Get model configuration
        model_config = self._get_model_config()
        
        # Initialize loss function
        self.loss_fn = LatencyAwareLoss(config, latency_estimator, model_config)
        
        # Pruning state
        self.current_sparsity = config.initial_sparsity
        self.pruning_step = 0
        
        # Logging
        self.loss_history = []
        self.sparsity_history = []

        self.use_apex = APEX_AVAILABLE

        self.max_steps = (len(self.train_dataset) / config.micro_batch_size) * config.num_epochs

    def compute_loss(self, model, inputs, return_outputs=False):
        """
        Compute the complete loss function with latency penalty
        """
        # Standard forward pass
        outputs = model(**inputs)
        task_loss = outputs.loss if hasattr(outputs, 'loss') else outputs[0]
        
        # Get current structural configuration
        active_heads, active_ffn = self.mask_manager.count_active_structures()
        
        # Compute total loss with latency penalty
        total_loss, loss_components = self.loss_fn.compute_total_loss(
            task_loss, active_heads, active_ffn, model
        )
        
        # Log loss components
        if self.state.global_step % self.args.logging_steps == 0:
            self.log(loss_components)
        
        return (total_loss, outputs) if return_outputs else total_loss
    
    def training_step(self, model: nn.Module, inputs: Dict[str, torch.Tensor], num_items_in_batch: Optional[int] = None) -> torch.Tensor:
        """
        Enhanced training step with alternating LoRA and mask updates
        """
        model.train()
        inputs = self._prepare_inputs(inputs)
        
        # 1. Compute current sparsity from schedule
        self.current_sparsity = compute_schedule_sparsity(
            self.state.global_step,
            self.max_steps,
            self.config.initial_sparsity,
            self.config.target_sparsity,
            self.config.warmup_ratio,
            self.config.cooldown_ratio
        )
        
        # 2. Update binary masks based on current sparsity
        if self.state.global_step % self.config.prune_freq == 0:
            self.mask_manager.get_binary_masks(self.current_sparsity)
            self.mask_manager.apply_binary_masks()
        
        # 3. Forward pass with current masks
        with self.compute_loss_context_manager():
            loss = self.compute_loss(model, inputs)
        
        # 4. Backward pass
        if self.args.gradient_accumulation_steps > 1:
            loss = loss / self.args.gradient_accumulation_steps
        
        if self.use_apex:
            with amp.scale_loss(loss, self.optimizer) as scaled_loss:
                scaled_loss.backward()
        else:
            self.accelerator.backward(loss)
        
        # 5. Update LoRA parameters (standard)
        return loss.detach()
    
    def _inner_training_loop(self, batch_size=None, args=None, resume_from_checkpoint=None, 
                           trial=None, ignore_keys_for_eval=None):
        """
        Override the main training loop to add mask update logic
        """
        # Call parent training loop first to initialize everything
        result = super()._inner_training_loop(
            batch_size=batch_size,
            args=args,
            resume_from_checkpoint=resume_from_checkpoint,
            trial=trial,
            ignore_keys_for_eval=ignore_keys_for_eval
        )
        
        return result
    
    def optimizer_step(self, optimizer):
        """
        Override optimizer step to add mask score updates
        """
        # Standard optimizer step
        super().optimizer_step(optimizer)
        
        # Update importance scores and mask scores
        if self.state.global_step % self.config.prune_freq == 0:
            self._update_mask_scores()
    
    def _update_mask_scores(self):
        """Update mask scores using adaptive learning rates (Eq. 10)"""
        # 1. Compute LoRA-guided importance scores
        importance_dict = compute_lora_guided_importance(
            self.model, self.mask_manager.group_mappings
        )
        
        # 2. Update importance scores with moving average
        self.mask_manager.update_importance_scores(importance_dict)
        
        # 3. Compute mask gradients using Straight-Through Estimator
        mask_gradients = compute_mask_gradients(
            self.model,
            self.mask_manager.binary_masks,
            self.mask_manager.group_mappings,
            self.state.train_dataloader.dataset[0] if hasattr(self.state, 'train_dataloader') else None
        )
        
        # 4. Update mask scores
        self.mask_manager.update_mask_scores(mask_gradients)
        
        # 5. Log current state
        current_sparsity = self.mask_manager.get_current_sparsity()
        active_heads, active_ffn = self.mask_manager.count_active_structures()
        
        logger.info(f"Step {self.state.global_step}: Sparsity={current_sparsity:.3f}, "
                   f"Active heads={active_heads}, Active FFN={active_ffn}")
    
    def save_model(self, output_dir: Optional[str] = None, _internal_call: bool = False):
        """
        Override to save mask states along with model
        """
        # Standard model saving
        super().save_model(output_dir, _internal_call)
        
        if output_dir is None:
            output_dir = self.args.output_dir
        
        # Save mask states
        mask_state = {
            'mask_scores': dict(self.mask_manager.mask_scores),
            'binary_masks': self.mask_manager.binary_masks,
            'importance_scores': self.mask_manager.importance_scores,
            'current_sparsity': self.current_sparsity,
            'active_heads': self.mask_manager.count_active_structures()[0],
            'active_ffn': self.mask_manager.count_active_structures()[1],
        }
        
        mask_path = os.path.join(output_dir, "mask_state.pt")
        torch.save(mask_state, mask_path)
        logger.info(f"Saved mask state to {mask_path}")
    
    def load_model(self, resume_from_checkpoint: str):
        """
        Override to load mask states along with model
        """
        # Standard model loading
        super().load_model(resume_from_checkpoint)
        
        # Load mask states if they exist
        mask_path = os.path.join(resume_from_checkpoint, "mask_state.pt")
        if os.path.exists(mask_path):
            mask_state = torch.load(mask_path)
            
            # Restore mask scores
            self.mask_manager.mask_scores.update(mask_state['mask_scores'])
            self.mask_manager.binary_masks = mask_state['binary_masks']
            self.mask_manager.importance_scores = mask_state['importance_scores']
            self.current_sparsity = mask_state.get('current_sparsity', self.config.initial_sparsity)
            
            logger.info(f"Loaded mask state from {mask_path}")
            logger.info(f"Resumed with sparsity={self.current_sparsity:.3f}")
    
    def _get_model_config(self):
        """Extract model configuration"""
        config = {}
        if hasattr(self.model, 'config'):
            hf_config = self.model.config
            config['num_layers'] = getattr(hf_config, 'num_hidden_layers', 0)
            config['num_attention_heads'] = getattr(hf_config, 'num_attention_heads', 0)
            config['hidden_size'] = getattr(hf_config, 'hidden_size', 0)
            config['ffn_intermediate_size'] = getattr(hf_config, 'intermediate_size', 0)
            config['head_dim'] = config['hidden_size'] // config['num_attention_heads'] if config['num_attention_heads'] > 0 else 0
        else:
            # Fallback values for Llama-7B
            config = {
                'num_layers': 32,
                'num_attention_heads': 32,
                'hidden_size': 4096,
                'ffn_intermediate_size': 11008,
                'head_dim': 128
            }
        return config
    
    def log_final_model_stats(self):
        """Log final model statistics"""
        active_heads, active_ffn = self.mask_manager.count_active_structures()
        final_sparsity = self.mask_manager.get_current_sparsity()
        final_latency = self.latency_estimator.estimate_latency(active_heads, active_ffn)
        
        stats = {
            'final_sparsity': final_sparsity,
            'final_active_heads': active_heads,
            'final_active_ffn': active_ffn,
            'final_latency_ms': final_latency,
            'target_sparsity': self.config.target_sparsity,
        }
        
        logger.info("Final model statistics:")
        for key, value in stats.items():
            logger.info(f"  {key}: {value}")
        
        return stats