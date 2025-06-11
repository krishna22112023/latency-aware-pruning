import torch
import torch.nn as nn
from transformers import Trainer
from transformers.trainer_utils import PREFIX_CHECKPOINT_DIR
import os
import logging
from typing import Dict, Optional, Tuple
from src.LAP.config import LatencyAwareConfig
from src.LAP.mask_manager import DifferentiableMaskManager
from src.LAP.latency_estimator import DifferentiableLatencyEstimator
from src.LAP.lora_importance import compute_lora_guided_importance, compute_mask_gradients
from src.LAP.loss import DifferentiableLatencyAwareLoss, compute_schedule_sparsity

logger = logging.getLogger(__name__)

try:
    from apex import amp
    APEX_AVAILABLE = True
    logger.info("APEX is available for mixed precision training. Mixed precision will be enabled.")
except ImportError:
    APEX_AVAILABLE = False
    logger.warning("APEX is not available. Mixed precision training will be disabled.")

class DifferentiableLatencyAwareTrainer(Trainer):
    """
    Extends Hugging Face Trainer for differentiable latency-aware structured pruning with LoRA
    """
    
    def __init__(self, config: LatencyAwareConfig, mask_manager: DifferentiableMaskManager,
                 latency_estimator: DifferentiableLatencyEstimator, **kwargs):
        super().__init__(**kwargs)
        
        # Limit dataset size if specified
        train_dataset = kwargs.get('train_dataset')
        if train_dataset and hasattr(config, 'max_train_samples') and config.max_train_samples:
            if len(train_dataset) > config.max_train_samples:
                # Randomly sample max_train_samples from the dataset
                import random
                indices = random.sample(range(len(train_dataset)), config.max_train_samples)
                # Create a subset of the dataset
                from torch.utils.data import Subset
                train_dataset = Subset(train_dataset, indices)
                kwargs['train_dataset'] = train_dataset
                logger.info(f"Limited training dataset to {config.max_train_samples} samples")

        self.config = config
        self.args = kwargs.get('args')
        self.mask_manager = mask_manager
        self.latency_estimator = latency_estimator
        self.train_dataset = kwargs.get('train_dataset')
        
        # Get model configuration
        model_config = kwargs.get('model').config
        
        # Initialize loss function
        self.loss_fn = DifferentiableLatencyAwareLoss(config, latency_estimator, model_config)
        
        # Pruning state
        self.current_sparsity = config.initial_sparsity
        self.pruning_step = 0
        
        # Logging
        self.loss_history = []
        self.sparsity_history = []

        # Disable APEX and mixed precision to avoid gradient scaler issues
        self.use_apex = False
        
        # Disable mixed precision in args to avoid scaler issues
        if self.args:
            self.args.fp16 = False
            self.args.bf16 = False
            if hasattr(self.args, 'dataloader_pin_memory'):
                self.args.dataloader_pin_memory = False

        self.max_steps = (len(self.train_dataset) / config.micro_batch_size) * config.num_epochs
        
        # Temperature annealing for Gumbel softmax
        self.initial_temperature = getattr(config, 'initial_temperature', 5.0)
        self.final_temperature = getattr(config, 'final_temperature', 0.5)
        
        # Ensure all LoRA parameters require gradients
        self._fix_parameter_gradients()
    
    def _fix_parameter_gradients(self):
        """Ensure LoRA parameters and mask scores require gradients"""
        model = getattr(self, 'model', None)
        if model:
            lora_fixed = 0
            for name, param in model.named_parameters():
                if 'lora_' in name and not param.requires_grad:
                    param.requires_grad = True
                    lora_fixed += 1
            
            # Fix mask scores
            mask_fixed = 0
            if hasattr(self.mask_manager, 'mask_scores'):
                for name, param in self.mask_manager.mask_scores.items():
                    if not param.requires_grad:
                        param.requires_grad = True
                        mask_fixed += 1
            
            logger.info(f"Fixed gradients: {lora_fixed} LoRA params, {mask_fixed} mask params")

    def compute_loss(self, model, inputs, return_outputs=False):
        """
        Compute the complete loss function with differentiable latency penalty
        """
        # Ensure model is in training mode
        model.train()
        
        # Set mask manager to training mode for soft masks
        self.mask_manager.set_training_mode(True)
        
        # Standard forward pass
        outputs = model(**inputs)
        task_loss = outputs.loss if hasattr(outputs, 'loss') else outputs[0]
        
        # Debug: Check if task_loss requires gradients
        if not isinstance(task_loss, torch.Tensor):
            raise ValueError(f"Task loss is not a tensor: {type(task_loss)}")
        
        if not task_loss.requires_grad:
            logger.warning(f"Task loss does not require gradients! Device: {task_loss.device}, Shape: {task_loss.shape}")
        
        # Compute total loss with differentiable latency penalty
        total_loss, loss_components = self.loss_fn.compute_total_loss(
            task_loss, self.mask_manager, model
        )
        
        # Debug: Check final loss
        if not isinstance(total_loss, torch.Tensor):
            raise ValueError(f"Total loss is not a tensor: {type(total_loss)}")
        
        if not total_loss.requires_grad:
            logger.error(f"Total loss does not require gradients! This will cause the backward error.")
            logger.error(f"Task loss requires grad: {task_loss.requires_grad}")
            logger.error(f"Loss components: {loss_components}")
        
        # Log loss components
        if self.state.global_step % self.args.logging_steps == 0:
            self.log(loss_components)
            
            # Log additional metrics
            current_sparsity = self.mask_manager.get_current_sparsity()
            active_heads, active_ffn = self.mask_manager.count_active_structures()
            
            additional_metrics = {
                'current_sparsity': current_sparsity,
                'temperature': self.mask_manager.temperature,
                'active_heads_soft': active_heads,
                'active_ffn_soft': active_ffn,
            }
            self.log(additional_metrics)
        
        return (total_loss, outputs) if return_outputs else total_loss
    
    def training_step(self, model: nn.Module, inputs: Dict[str, torch.Tensor], num_items_in_batch: Optional[int] = None) -> torch.Tensor:
        """
        Enhanced training step with differentiable mask updates
        """
        model.train()
        inputs = self._prepare_inputs(inputs)
        
        # Debug: Check if model parameters require gradients
        if self.state.global_step == 0:  # Only log on first step
            param_count = sum(1 for p in model.parameters() if p.requires_grad)
            total_params = sum(1 for p in model.parameters())
            logger.info(f"Model has {param_count}/{total_params} parameters requiring gradients")
        
        # 1. Compute current sparsity from schedule
        self.current_sparsity = compute_schedule_sparsity(
            self.state.global_step,
            self.max_steps,
            self.config.initial_sparsity,
            self.config.target_sparsity,
            self.config.warmup_ratio,
            self.config.cooldown_ratio
        )
        
        # 2. Update temperature for annealing
        self._update_temperature()
        
        # 3. Generate soft masks for training
        self.mask_manager.get_soft_masks(target_sparsity=self.current_sparsity, hard=False)
        self.mask_manager.apply_masks(use_soft=True)
        
        # 4. Forward pass with soft masks
        with self.compute_loss_context_manager():
            loss = self.compute_loss(model, inputs)
        
        # 5. Backward pass - FIXED to avoid gradient scaler issues
        if self.args.gradient_accumulation_steps > 1:
            loss = loss / self.args.gradient_accumulation_steps
        
        # Additional debug: Ensure loss requires gradients before backward
        if not loss.requires_grad:
            raise RuntimeError(f"Loss does not require gradients at step {self.state.global_step}. "
                             f"Cannot perform backward pass. Loss: {loss}")
        
        # Use simple backward pass without any scalers
        self.accelerator.backward(loss)
        
        return loss.detach()
    
    def _update_temperature(self):
        """Anneal temperature for Gumbel softmax over training"""
        progress = self.state.global_step / self.max_steps
        temperature = self.final_temperature + (self.initial_temperature - self.final_temperature) * (1 - progress)
        self.mask_manager.temperature = temperature
        
        # Also update in latency estimator if it has temperature
        if hasattr(self.latency_estimator, 'temperature'):
            self.latency_estimator.temperature = temperature
    
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
        
        # Update importance scores and mask scores periodically
        if self.state.global_step % self.config.prune_freq == 0:
            self._update_mask_scores()
    
    def _update_mask_scores(self):
        """Update mask scores using adaptive learning rates"""
        try:
            # 1. Compute LoRA-guided importance scores
            importance_dict = compute_lora_guided_importance(
                self.model, self.mask_manager.group_mappings
            )
            
            # 2. Update importance scores with moving average
            self.mask_manager.update_importance_scores(importance_dict)
            
            # 3. Since we're using differentiable masks, the gradients flow through naturally
            # We don't need explicit mask gradient computation as in the discrete case
            
            # 4. Log current state
            current_sparsity = self.mask_manager.get_current_sparsity()
            active_heads, active_ffn = self.mask_manager.count_active_structures()
            
            logger.info(f"Step {self.state.global_step}: Sparsity={current_sparsity:.3f}, "
                       f"Active heads={active_heads:.1f}, Active FFN={active_ffn:.1f}, "
                       f"Temperature={self.mask_manager.temperature:.3f}")
        except Exception as e:
            logger.warning(f"Error updating mask scores: {e}")
    
    def evaluate(self, eval_dataset=None, ignore_keys=None, metric_key_prefix="eval"):
        """
        Override evaluation to use hard masks for consistent evaluation
        """
        # Switch to hard masks for evaluation
        self.mask_manager.set_training_mode(False)
        self.mask_manager.get_binary_masks(self.current_sparsity)
        self.mask_manager.apply_masks(use_soft=False)
        
        # Call parent evaluation
        result = super().evaluate(eval_dataset=eval_dataset, ignore_keys=ignore_keys, 
                                metric_key_prefix=metric_key_prefix)
        
        # Switch back to training mode
        self.mask_manager.set_training_mode(True)
        
        return result
    
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
            'soft_masks': self.mask_manager.soft_masks,
            'importance_scores': self.mask_manager.importance_scores,
            'current_sparsity': self.current_sparsity,
            'temperature': self.mask_manager.temperature,
            'training_mode': self.mask_manager.training_mode,
        }
        
        # Get final counts
        active_heads, active_ffn = self.mask_manager.count_active_structures()
        mask_state['active_heads'] = active_heads
        mask_state['active_ffn'] = active_ffn
        
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
            if 'binary_masks' in mask_state:
                self.mask_manager.binary_masks = mask_state['binary_masks']
            if 'soft_masks' in mask_state:
                self.mask_manager.soft_masks = mask_state['soft_masks']
            if 'importance_scores' in mask_state:
                self.mask_manager.importance_scores = mask_state['importance_scores']
            
            self.current_sparsity = mask_state.get('current_sparsity', self.config.initial_sparsity)
            self.mask_manager.temperature = mask_state.get('temperature', self.initial_temperature)
            self.mask_manager.training_mode = mask_state.get('training_mode', True)
            
            logger.info(f"Loaded mask state from {mask_path}")