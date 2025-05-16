import torch
import torch.nn as nn
from typing import Dict, Tuple, Optional
import numpy as np


class LatencyAwareLoss:
    """
    Implements the complete loss function from Eq. 4:
    L = L_task + λE[Latency(M)] + (γ/2)||W₀ + BA||²
    """
    
    def __init__(self, config, latency_estimator, model_config):
        self.config = config
        self.latency_estimator = latency_estimator
        self.model_config = model_config
        self.seq_len = config.cutoff_len
    
    def compute_total_loss(self, task_loss: torch.Tensor, active_heads: int, 
                          active_ffn: int, model: nn.Module) -> Tuple[torch.Tensor, Dict]:
        """
        Compute the complete loss function
        
        Args:
            task_loss: The task-specific loss (e.g., cross-entropy)
            active_heads: Number of active attention heads
            active_ffn: Number of active FFN channels
            model: The model for computing regularization
        
        Returns:
            Tuple of (total_loss, loss_components_dict)
        """
        # 1. Task loss (already computed)
        task_loss_value = task_loss
        
        # 2. Latency penalty
        latency = self.latency_estimator.estimate_latency(
            active_heads, active_ffn, self.seq_len
        )
        latency_loss = self.config.latency_weight * latency
        
        # Convert to tensor if needed
        if not isinstance(latency_loss, torch.Tensor):
            latency_loss = torch.tensor(latency_loss, device=task_loss.device)
        
        # 3. L2 regularization on adapted weights
        l2_loss = self.compute_l2_regularization(model)
        
        # Total loss
        total_loss = task_loss_value + latency_loss + l2_loss
        
        # Return detailed breakdown
        loss_components = {
            'task_loss': task_loss_value.item(),
            'latency_loss': latency_loss.item(),
            'l2_loss': l2_loss.item(),
            'total_loss': total_loss.item(),
            'latency_ms': latency,
            'active_heads': active_heads,
            'active_ffn': active_ffn
        }
        
        return total_loss, loss_components
    
    def compute_l2_regularization(self, model: nn.Module) -> torch.Tensor:
        """
        Compute L2 regularization on adapted weights: (γ/2)||W₀ + BA||²
        """
        l2_loss = torch.tensor(0.0, device=next(model.parameters()).device)
        
        for name, module in model.named_modules():
            if self._is_lora_module(module):
                # Get base weight W₀
                base_weight = module.weight
                
                # Compute LoRA adaptation BA
                if hasattr(module, 'lora_A') and hasattr(module, 'lora_B'):
                    lora_adaptation = module.lora_B.weight @ module.lora_A.weight
                    lora_adaptation *= module.scaling
                    
                    # Apply mask if it exists
                    if hasattr(module, 'lora_mask'):
                        # Expand mask to match weight dimensions
                        mask = module.lora_mask
                        if mask.dim() < base_weight.dim():
                            # Expand mask to match weight shape
                            if len(mask) == base_weight.shape[0]:
                                # Mask applies to output dimension
                                expanded_mask = mask.unsqueeze(1).expand_as(base_weight)
                            else:
                                # Mask applies to input dimension
                                expanded_mask = mask.unsqueeze(0).expand_as(base_weight)
                        else:
                            expanded_mask = mask
                        
                        # Apply mask
                        lora_adaptation = lora_adaptation * expanded_mask
                    
                    # Compute adapted weight W' = W₀ + BA
                    adapted_weight = base_weight + lora_adaptation
                    
                    # Add to L2 loss
                    l2_loss += torch.norm(adapted_weight, p=2) ** 2
        
        # Apply regularization coefficient
        l2_loss *= self.config.regularization_weight / 2.0
        
        return l2_loss
    
    def compute_latency_gradient_penalty(self, current_heads: int, current_ffn: int) -> torch.Tensor:
        """
        Compute gradient penalty for latency to encourage smoother optimization
        """
        # Get latency gradients
        head_grad, ffn_grad = self.latency_estimator.get_latency_gradient(
            current_heads, current_ffn, self.seq_len
        )
        
        # Simple gradient penalty (can be made more sophisticated)
        gradient_magnitude = np.sqrt(head_grad**2 + ffn_grad**2)
        
        # Convert to tensor
        device = next(iter(self.latency_estimator.lut.values()))[0].device if self.latency_estimator.lut else torch.device('cpu')
        return torch.tensor(gradient_magnitude * 0.001, device=device)  # Small coefficient
    
    def _is_lora_module(self, module):
        """Check if a module is a LoRA module"""
        return (hasattr(module, 'lora_A') and hasattr(module, 'lora_B') and 
                hasattr(module, 'scaling'))


class StraightThroughEstimator(torch.autograd.Function):
    """
    Straight-Through Estimator for gradient computation through discrete operations
    Used for computing gradients through the TopK operation (Eq. 9)
    """
    
    @staticmethod
    def forward(ctx, input, binary_output):
        """Forward pass just returns the binary output"""
        return binary_output
    
    @staticmethod
    def backward(ctx, grad_output):
        """Backward pass: pass gradient through unchanged"""
        # The gradient passes through unchanged (straight-through)
        return grad_output, None


def apply_straight_through_estimator(mask_scores: torch.Tensor, binary_mask: torch.Tensor) -> torch.Tensor:
    """
    Apply Straight-Through Estimator to enable gradient flow through TopK operation
    
    Args:
        mask_scores: Continuous learnable scores
        binary_mask: Binary mask derived from TopK operation
    
    Returns:
        Binary mask with gradient flow capabilities
    """
    return StraightThroughEstimator.apply(mask_scores, binary_mask.float())


def compute_schedule_sparsity(step: int, total_steps: int, initial_sparsity: float,
                            final_sparsity: float, warmup_ratio: float, 
                            cooldown_ratio: float) -> float:
    """
    Compute sparsity schedule similar to LoRAPrune's schedule_sparsity_ratio
    """
    warmup_steps = int(warmup_ratio * total_steps)
    cooldown_steps = int(cooldown_ratio * total_steps)
    
    if step <= warmup_steps:
        sparsity = initial_sparsity
    elif step > total_steps - cooldown_steps:
        sparsity = final_sparsity
    else:
        # Cubic scheduling
        progress = (step - warmup_steps) / (total_steps - warmup_steps - cooldown_steps)
        sparsity = final_sparsity + (initial_sparsity - final_sparsity) * ((1 - progress) ** 3)
    
    return sparsity