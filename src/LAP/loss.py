import torch
import torch.nn as nn
from typing import Dict, Tuple, Optional
import numpy as np


class DifferentiableLatencyAwareLoss:
    """
    Implements the complete loss function with differentiable latency:
    L = L_task + λE[Latency(M)] + (γ/2)||W₀ + BA||²
    
    Where E[Latency(M)] is now differentiable through soft architectural sampling.
    """
    
    def __init__(self, config, latency_estimator, model_config):
        self.config = config
        self.latency_estimator = latency_estimator
        self.model_config = model_config
        self.seq_len = config.cutoff_len
        
        # Method for computing differentiable latency
        self.latency_method = getattr(config, 'latency_method', 'linear')  # 'linear', 'soft', 'gumbel'
        
    def compute_total_loss(self, task_loss: torch.Tensor, mask_manager, 
                          model: nn.Module) -> Tuple[torch.Tensor, Dict]:
        """
        Compute the complete loss function with differentiable latency
        
        Args:
            task_loss: The task-specific loss (e.g., cross-entropy)
            mask_manager: The differentiable mask manager
            model: The model for computing regularization
        
        Returns:
            Tuple of (total_loss, loss_components_dict)
        """
        # 1. Task loss (already computed)
        task_loss_value = task_loss
        
        # 2. Differentiable latency penalty
        latency_loss = self.compute_differentiable_latency_loss(mask_manager)
        
        # 3. L2 regularization on adapted weights
        # l2_loss = self.compute_l2_regularization(model)
        
        # Ensure all losses are tensors with proper gradients
        #if l2_loss is None or not isinstance(l2_loss, torch.Tensor):
        #    l2_loss = torch.tensor(0.0, device=task_loss.device, dtype=task_loss.dtype)
        
        # Total loss - all components should preserve gradients
        total_loss = task_loss_value + latency_loss #+ l2_loss
        
        # Get discrete counts for logging
        active_heads, active_ffn = mask_manager.count_active_structures()
        discrete_latency = self.latency_estimator.estimate_latency_discrete(
            active_heads, active_ffn, self.seq_len
        )
        
        # Return detailed breakdown
        loss_components = {
            'task_loss': task_loss_value.item() if isinstance(task_loss_value, torch.Tensor) else task_loss_value,
            'latency_loss': latency_loss.item() if isinstance(latency_loss, torch.Tensor) else latency_loss,
            #'l2_loss': l2_loss.item() if isinstance(l2_loss, torch.Tensor) else l2_loss,
            'total_loss': total_loss.item() if isinstance(total_loss, torch.Tensor) else total_loss,
            'latency_ms': discrete_latency,  # Discrete latency for logging
            'active_heads': active_heads,
            'active_ffn': active_ffn
        }
        
        return total_loss, loss_components
    
    def compute_differentiable_latency_loss(self, mask_manager) -> torch.Tensor:
        """
        Compute differentiable latency loss using soft architectural sampling.
        """
        # Get mask scores for latency computation
        head_scores, ffn_scores = mask_manager.get_mask_scores_for_latency()
        
        if self.latency_method == 'linear':
            # Use linear approximation (fastest, smoothest gradients)
            expected_latency = self.latency_estimator._linear_latency_approximation(
                head_scores.sum(), ffn_scores.sum(), head_scores.device
            )
        elif self.latency_method == 'gumbel':
            # Use Gumbel softmax sampling
            expected_latency = self.latency_estimator.compute_expected_latency_gumbel(
                head_scores, ffn_scores, 
                batch_size=1, seq_len=self.seq_len, hard=False
            )
        elif self.latency_method == 'soft':
            # Use soft sampling with expected value
            expected_latency = self.latency_estimator.compute_expected_latency_soft(
                head_scores, ffn_scores,
                batch_size=1, seq_len=self.seq_len
            )
        else:
            raise ValueError(f"Unknown latency method: {self.latency_method}")
        
        # Apply latency weight
        latency_loss = self.config.latency_weight * expected_latency
        
        return latency_loss
    
    def compute_l2_regularization(self, model: nn.Module) -> torch.Tensor:
        """
        Compute L2 regularization on adapted weights: (γ/2)||W₀ + BA||²
        """
        device = next(model.parameters()).device
        l2_terms = []
        
        for name, module in model.named_modules():
            if self._is_lora_module(module):
                # Get base weight W₀
                base_weight = module.weight
                
                # Compute LoRA adaptation BA
                if hasattr(module, 'lora_A') and hasattr(module, 'lora_B'):
                    lora_adaptation = module.lora_B.weight @ module.lora_A.weight
                    lora_adaptation *= module.scaling
                    
                    # Apply soft mask if it exists and we're in training mode
                    if (hasattr(module, '_output_mask') and module._output_mask is not None and
                        hasattr(module, '_use_soft_mask') and getattr(module, '_use_soft_mask', False)):
                        
                        mask = module._output_mask
                        if mask.dim() < lora_adaptation.dim():
                            # Expand mask to match adaptation shape
                            if len(mask) == lora_adaptation.shape[0]:
                                # Mask applies to output dimension
                                expanded_mask = mask.unsqueeze(1).expand_as(lora_adaptation)
                            else:
                                # Mask applies to input dimension
                                expanded_mask = mask.unsqueeze(0).expand_as(lora_adaptation)
                        else:
                            expanded_mask = mask
                        
                        # Apply soft mask
                        lora_adaptation = lora_adaptation * expanded_mask
                    
                    # Compute adapted weight W' = W₀ + BA
                    adapted_weight = base_weight + lora_adaptation
                    
                    # Add to L2 loss terms
                    l2_terms.append(torch.norm(adapted_weight, p=2) ** 2)
        
        # Combine all L2 terms
        if l2_terms:
            l2_loss = torch.stack(l2_terms).sum()
            # Apply regularization coefficient
            l2_loss *= self.config.regularization_weight / 2.0
        else:
            # No LoRA modules found, return zero loss
            l2_loss = torch.tensor(0.0, device=device, requires_grad=False)
        
        return l2_loss
    
    def compute_entropy_regularization(self, mask_manager) -> torch.Tensor:
        """
        Optional: Add entropy regularization to encourage diversity in mask selection.
        This can help prevent the model from getting stuck in local minima.
        """
        entropy_loss = torch.tensor(0.0, device=next(iter(mask_manager.mask_scores.values())).device)
        
        for group_key, scores in mask_manager.mask_scores.items():
            # Compute entropy of the mask distribution
            probs = torch.softmax(scores, dim=0)
            entropy = -torch.sum(probs * torch.log(probs + 1e-8))
            entropy_loss += entropy
        
        # Weight the entropy loss (encourage exploration)
        entropy_weight = getattr(self.config, 'entropy_weight', 0.01)
        
        return entropy_weight * entropy_loss
    
    def compute_variance_regularization(self, mask_manager) -> torch.Tensor:
        """
        Optional: Add variance regularization to encourage balanced mask scores.
        """
        variance_loss = torch.tensor(0.0, device=next(iter(mask_manager.mask_scores.values())).device)
        
        for group_key, scores in mask_manager.mask_scores.items():
            # Penalize high variance in scores (encourage balanced selection)
            variance = torch.var(scores)
            variance_loss += variance
        
        # Weight the variance loss
        variance_weight = getattr(self.config, 'variance_weight', 0.001)
        
        return variance_weight * variance_loss
    
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