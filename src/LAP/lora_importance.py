import torch
import numpy as np
from typing import Dict, Tuple
from src.latency_profiling.lora import Linear#, Linear8bitLt


def compute_lora_guided_importance(model, group_mappings):
    """
    Compute importance scores using LoRA matrices and gradients.
    Implements the LoRA-guided criterion from Eq. 10 in methodology.
    
    Args:
        model: The LoRA-wrapped model
        group_mappings: Dictionary mapping group names to their modules
    
    Returns:
        Dict mapping group keys to importance scores
    """
    importance_dict = {}
    
    for group_key, group_info in group_mappings.items():
        group_type = group_info['type']
        layer_idx = group_info['layer']
        
        # Aggregate importance across modules in the group
        group_importance = None
        
        for name, proj_type, module in group_info['modules']:
            if not _is_target_lora_layer(module):
                continue
            
            # Compute per-weight importance (Eq. 10)
            per_weight_importance = compute_per_weight_importance(module)
            
            if per_weight_importance is None:
                continue
            
            # Aggregate to structural groups
            if group_type == 'attention_head':
                structured_importance = aggregate_attention_importance(
                    per_weight_importance, proj_type, module
                )
            elif group_type == 'ffn_channel':
                structured_importance = aggregate_ffn_importance(
                    per_weight_importance, proj_type, module
                )
            else:
                continue
            
            # Accumulate across projections in the group
            if group_importance is None:
                group_importance = structured_importance
            else:
                group_importance += structured_importance
        
        if group_importance is not None:
            # Take square root for stability (Eq. 7)
            importance_dict[group_key] = group_importance.sqrt()
    
    return importance_dict


def compute_per_weight_importance(module):
    """
    Compute per-weight importance for a LoRA module.
    Based on Eq. 10: I_ij = [(∇B A + B ∇A - ∇B ∇A)(W + BA)]²
    """
    if not hasattr(module, 'lora_A') or not hasattr(module, 'lora_B'):
        return None
    
    # Get LoRA matrices and their gradients
    lora_A = module.lora_A.weight
    lora_B = module.lora_B.weight
    
    # Check if gradients exist
    if lora_A.grad is None or lora_B.grad is None:
        return None
    
    grad_A = lora_A.grad
    grad_B = lora_B.grad
    
    # Compute the gradient approximation (∇B A + B ∇A - ∇B ∇A)
    # Note: We approximate ∇B @ ∇A as a small third-order term that can be ignored
    # This simplifies to: ∇B @ A + B @ ∇A
    with torch.no_grad():
        gradient_approx = grad_B @ lora_A + lora_B @ grad_A
        
        # Get effective weight W' = W₀ + BA
        base_weight = module.weight
        lora_weight = lora_B @ lora_A * module.scaling
        effective_weight = base_weight + lora_weight
        
        # Compute importance: |gradient_approx * effective_weight|²
        importance = (gradient_approx * effective_weight).abs().pow(2)
    
    return importance


def aggregate_attention_importance(importance, proj_type, module):
    """Aggregate importance scores for attention heads"""
    # Get model configuration from module
    if hasattr(module, 'num_heads'):
        num_heads = module.num_heads
        head_dim = module.out_features // num_heads if proj_type != 'o_proj' else module.in_features // num_heads
    else:
        # Fallback to default
        head_dim = 128  # Adjust based on model
        num_heads = module.out_features // head_dim if proj_type != 'o_proj' else module.in_features // num_heads
    
    if proj_type in ['q_proj', 'k_proj', 'v_proj']:
        # Output dimension is structured by heads
        importance_reshaped = importance.view(num_heads, head_dim, -1)
        head_importance = importance_reshaped.sum(dim=(1, 2))
    elif proj_type == 'o_proj':
        # Input dimension is structured by heads
        importance_reshaped = importance.view(-1, num_heads, head_dim)
        head_importance = importance_reshaped.sum(dim=(0, 2))
    else:
        head_importance = importance.sum(dim=1)
    
    return head_importance


def aggregate_ffn_importance(importance, proj_type, module):
    """Aggregate importance scores for FFN channels"""
    if proj_type in ['up_proj', 'gate_proj']:
        # Output dimension is structured by channels
        channel_importance = importance.sum(dim=1)
    elif proj_type == 'down_proj':
        # Input dimension is structured by channels
        channel_importance = importance.sum(dim=0)
    else:
        channel_importance = importance.sum(dim=1)
    
    return channel_importance


def compute_mask_gradients(model, binary_masks, group_mappings, loss):
    """
    Compute gradients with respect to binary masks using Straight-Through Estimator.
    This implements the gradient approximation ∇_m L ≈ ∇_M L from Eq. 9.
    """
    mask_gradients = {}
    
    # We need to compute ∂L/∂M_g for each group g
    # Using torch.autograd to get gradients
    for group_key, binary_mask in binary_masks.items():
        if group_key not in group_mappings:
            continue
        
        group_info = group_mappings[group_key]
        
        # Collect all relevant parameter gradients
        group_grad = torch.zeros_like(binary_mask, dtype=torch.float32)
        
        for name, proj_type, module in group_info['modules']:
            if not _is_target_lora_layer(module) or not hasattr(module, 'lora_mask'):
                continue
            
            # Get the effective gradients flowing through this module
            if module.weight.grad is not None:
                # Map module gradients to structural groups
                module_grad = extract_structural_gradient(
                    module, group_info['type'], proj_type
                )
                if module_grad is not None:
                    group_grad += module_grad
        
        # Apply Straight-Through Estimator: gradient passes through unchanged
        mask_gradients[group_key] = group_grad
    
    return mask_gradients


def extract_structural_gradient(module, group_type, proj_type):
    """Extract gradient components relevant to structural groups"""
    if not hasattr(module, 'weight') or module.weight.grad is None:
        return None
    
    weight_grad = module.weight.grad
    
    if group_type == 'attention_head':
        # Extract gradients relevant to attention heads
        if proj_type in ['q_proj', 'k_proj', 'v_proj']:
            # Sum over input dimension, aggregate by heads
            head_dim = 128  # Adjust based on model
            num_heads = weight_grad.shape[0] // head_dim
            grad_by_head = weight_grad.view(num_heads, head_dim, -1).sum(dim=(1, 2))
        elif proj_type == 'o_proj':
            # Sum over output dimension, aggregate by heads
            head_dim = 128
            num_heads = weight_grad.shape[1] // head_dim
            grad_by_head = weight_grad.view(-1, num_heads, head_dim).sum(dim=(0, 2))
        else:
            return None
        return grad_by_head
    
    elif group_type == 'ffn_channel':
        # Extract gradients relevant to FFN channels
        if proj_type in ['up_proj', 'gate_proj']:
            # Sum over input dimension
            channel_grad = weight_grad.sum(dim=1)
        elif proj_type == 'down_proj':
            # Sum over output dimension
            channel_grad = weight_grad.sum(dim=0)
        else:
            return None
        return channel_grad
    
    return None


def _is_target_lora_layer(module):
    """Check if module is a target LoRA layer"""
    return isinstance(module, (Linear)) and getattr(module, 'r', 0) > 0