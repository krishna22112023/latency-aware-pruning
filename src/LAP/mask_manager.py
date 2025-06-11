import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Tuple
import warnings
from src.latency_profiling.lora import Linear
from collections import defaultdict
import numpy as np


class DifferentiableMaskManager:
    """
    Manages learnable mask scores with differentiable soft masks for training
    and hard masks for inference/evaluation.
    """
    
    def __init__(self, model, config, model_config):
        self.model = model
        self.config = config
        self.model_config = model_config
        
        # Learnable mask scores (m in methodology)
        self.mask_scores = nn.ParameterDict()
        
        # Binary masks derived from scores (M in methodology) - for evaluation
        self.binary_masks = {}
        
        # Soft masks for training - differentiable
        self.soft_masks = {}
        
        # Training mode flag
        self.training_mode = True
        
        # Temperature for Gumbel softmax
        self.temperature = getattr(config, 'gumbel_temperature', 1.0)
        
        # Importance scores for each structural group (S_g from Eq. 7)
        self.importance_scores = {}
        
        # Moving average for importance scores
        self.importance_ma = {}
        
        # Group mappings
        self.group_mappings = self.create_group_mappings()
        
        # Initialize mask scores
        self.init_mask_scores()
        
        # Register forward hooks for mask application
        self._register_mask_hooks()
    
    def create_group_mappings(self):
        """Create mappings between modules and structural groups"""
        mappings = {}
        
        for name, module in self.model.named_modules():
            if self._is_target_lora_layer(module):
                layer_idx, group_type, proj_type = self._parse_module_name(name)
                if layer_idx is not None and group_type is not None:
                    group_key = f"{layer_idx}_{group_type}"
                    if group_key not in mappings:
                        mappings[group_key] = {
                            'type': group_type,
                            'layer': layer_idx,
                            'modules': []
                        }
                    mappings[group_key]['modules'].append((name, proj_type, module))
        
        return mappings
    
    def init_mask_scores(self):
        """Initialize learnable mask scores for each structural group"""
        device = next(self.model.parameters()).device
        
        for group_key, group_info in self.group_mappings.items():
            if group_info['type'] == 'attention_head':
                num_groups = self.model_config['num_attention_heads']
            elif group_info['type'] == 'ffn_channel':
                # Use the first module to determine FFN size
                first_module = group_info['modules'][0][2]
                if any(proj in group_info['modules'][0][1] for proj in ['up_proj', 'gate_proj']):
                    num_groups = first_module.out_features
                else:  # down_proj
                    num_groups = first_module.in_features
            else:
                continue
            # Initialize mask scores (small positive values)
            # Use uniform initialization to avoid bias
            self.mask_scores[group_key] = nn.Parameter(
                torch.zeros(num_groups, device=device) + 0.1
            )
            
            # Initialize importance scores
            self.importance_scores[group_key] = torch.ones(num_groups, device=device)
            self.importance_ma[group_key] = torch.ones(num_groups, device=device)
    
    def get_soft_masks(self, target_sparsity: float = None, hard: bool = False):
        """
        Generate soft masks using Gumbel softmax for differentiable pruning.
        
        Args:
            target_sparsity: Target sparsity level (optional, for controlling temperature)
            hard: Whether to use hard (one-hot) sampling
        
        Returns:
            Dictionary of soft masks
        """
        soft_masks = {}
        
        for group_key, scores in self.mask_scores.items():
            if target_sparsity is not None:
                # Adjust temperature based on target sparsity for better control
                num_groups = scores.numel()
                k_struct = max(1, int((1 - target_sparsity) * num_groups))
                
                # Use straight-through Gumbel softmax with TopK-like behavior
                soft_mask = self._topk_gumbel_softmax(scores, k_struct, hard=hard)
            else:
                # Simple Gumbel softmax
                soft_mask = F.gumbel_softmax(scores, tau=self.temperature, hard=hard)
            
            soft_masks[group_key] = soft_mask
        
        self.soft_masks = soft_masks
        return soft_masks
    
    def _topk_gumbel_softmax(self, scores: torch.Tensor, k: int, hard: bool = False):
        """
        Gumbel softmax with TopK-like behavior for structured pruning.
        """
        # Add Gumbel noise
        gumbel_noise = -torch.log(-torch.log(torch.rand_like(scores) + 1e-8) + 1e-8)
        noisy_scores = scores + gumbel_noise
        
        # Get TopK indices
        _, top_k_indices = torch.topk(noisy_scores, k)
        
        if hard:
            # Hard selection (one-hot)
            mask = torch.zeros_like(scores)
            mask[top_k_indices] = 1.0
        else:
            # Soft selection using softmax with temperature
            exp_scores = torch.exp(noisy_scores / self.temperature)
            
            # Create soft mask favoring TopK elements
            mask = torch.zeros_like(scores)
            mask[top_k_indices] = exp_scores[top_k_indices]
            mask = mask / (mask.sum() + 1e-8) * k  # Normalize to k active units
        
        return mask
    
    def get_binary_masks(self, target_sparsity):
        """Convert mask scores to binary masks using TopK operation (for evaluation)"""
        binary_masks = {}
        
        for group_key, scores in self.mask_scores.items():
            num_groups = scores.numel()
            k_struct = max(1, int((1 - target_sparsity) * num_groups))
            
            # TopK operation
            _, top_k_indices = torch.topk(scores, k_struct)
            binary_mask = torch.zeros_like(scores, dtype=torch.bool)
            binary_mask[top_k_indices] = True
            
            binary_masks[group_key] = binary_mask
        
        self.binary_masks = binary_masks
        return binary_masks
    
    def update_importance_scores(self, importance_dict):
        """Update importance scores with moving average"""
        for group_key, importance in importance_dict.items():
            if group_key in self.importance_scores:
                # Moving average update
                self.importance_ma[group_key] = (
                    self.config.importance_beta * self.importance_ma[group_key] + 
                    (1 - self.config.importance_beta) * importance
                )
                self.importance_scores[group_key] = self.importance_ma[group_key]
    
    def update_mask_scores(self, mask_gradients):
        """Update mask scores using adaptive learning rates (Eq. 8 and 10)"""
        with torch.no_grad():
            for group_key, grad in mask_gradients.items():
                if group_key in self.mask_scores and group_key in self.importance_scores:
                    # Adaptive learning rate (Eq. 8)
                    importance = self.importance_scores[group_key]
                    lr_adaptive = self.config.mask_lr / (importance + self.config.epsilon)
                    
                    # Update mask scores (Eq. 10)
                    self.mask_scores[group_key].data -= lr_adaptive * grad
    
    def apply_masks(self, use_soft: bool = None):
        """
        Apply masks to model modules.
        
        Args:
            use_soft: Whether to use soft masks. If None, uses self.training_mode
        """
        if use_soft is None:
            use_soft = self.training_mode
        
        if use_soft and self.soft_masks:
            masks = self.soft_masks
        else:
            masks = self.binary_masks
        
        for group_key, mask in masks.items():
            if group_key in self.group_mappings:
                self._apply_mask_to_group(group_key, mask, use_soft)
    
    def _apply_mask_to_group(self, group_key, mask, use_soft: bool):
        """Apply mask to a specific structural group"""
        group_info = self.group_mappings[group_key]
        group_type = group_info['type']
        
        for name, proj_type, module in group_info['modules']:
            if group_type == 'attention_head':
                self._apply_attention_mask(module, mask, proj_type, use_soft)
            elif group_type == 'ffn_channel':
                self._apply_ffn_mask(module, mask, proj_type, use_soft)
    
    def _apply_attention_mask(self, module, mask, proj_type, use_soft: bool):
        """Apply mask to attention module with proper dimension calculation"""
        
        if not self._is_target_lora_layer(module):
            return
        
        # Calculate dimensions from actual module instead of config
        if proj_type in ['q_proj', 'k_proj', 'v_proj']:
            # Output dimension is structured
            actual_output = module.out_features
            mask_size = len(mask)
            features_per_head = actual_output // mask_size
            
            # Create correctly sized mask
            expanded_mask = mask.repeat_interleave(features_per_head)
            
            # Ensure exact size match
            if len(expanded_mask) != actual_output:
                correct_mask = torch.zeros(actual_output, device=mask.device, dtype=mask.dtype)
                min_size = min(len(expanded_mask), actual_output)
                correct_mask[:min_size] = expanded_mask[:min_size]
                expanded_mask = correct_mask
            
            if not use_soft:
                expanded_mask = expanded_mask.float()
            module._output_mask = expanded_mask
            module._mask_dim = 'output'
            
        elif proj_type == 'o_proj':
            # Input dimension is structured
            actual_input = module.in_features
            mask_size = len(mask)
            features_per_head = actual_input // mask_size
            
            expanded_mask = mask.repeat_interleave(features_per_head)
            
            if len(expanded_mask) != actual_input:
                correct_mask = torch.zeros(actual_input, device=mask.device, dtype=mask.dtype)
                min_size = min(len(expanded_mask), actual_input)
                correct_mask[:min_size] = expanded_mask[:min_size]
                expanded_mask = correct_mask
            
            if not use_soft:
                expanded_mask = expanded_mask.float()
            module._input_mask = expanded_mask
            module._mask_dim = 'input'
        
        # Store original mask and projection type for hook
        setattr(module, '_lora_mask', mask)
        setattr(module, '_use_soft_mask', use_soft)
        setattr(module, '_proj_type', proj_type)
    
    def _apply_ffn_mask(self, module, mask, proj_type, use_soft: bool):
        """Apply mask to FFN module"""
        if not self._is_target_lora_layer(module):
            return
        
        # Store appropriate mask based on projection type
        if proj_type in ['up_proj', 'gate_proj']:
            # Output dimension is structured
            module._output_mask = mask.float() if not use_soft else mask
            module._mask_dim = 'output'
        elif proj_type == 'down_proj':
            # Input dimension is structured
            module._input_mask = mask.float() if not use_soft else mask
            module._mask_dim = 'input'
        
        # Store original mask using setattr to avoid Parameter conflicts
        setattr(module, '_lora_mask', mask)
        setattr(module, '_use_soft_mask', use_soft)
        setattr(module, '_proj_type', proj_type)
    
    def _register_mask_hooks(self):
        """Register forward hooks to apply masks during forward pass"""
        def create_mask_hook():
            def mask_hook(module, input, output):
                # Only apply masks to LoRA modules
                if not self._is_target_lora_layer(module):
                    return output
                
                # Get mask information
                if not hasattr(module, '_mask_dim'):
                    return output
                
                mask_dim = module._mask_dim
                
                # Get the appropriate mask
                if mask_dim == 'output' and hasattr(module, '_output_mask'):
                    mask = module._output_mask
                elif mask_dim == 'input' and hasattr(module, '_input_mask'):
                    mask = module._input_mask
                else:
                    return output
                
                if mask is None:
                    return output
                
                # Move mask to correct device
                mask = mask.to(output.device)
                
                # Handle tuple outputs (some models return tuples)
                if isinstance(output, tuple):
                    output_tensor = output[0]
                    rest_outputs = output[1:]
                else:
                    output_tensor = output
                    rest_outputs = None
                
                # Apply mask based on dimension
                if mask_dim == 'output':
                    # Ensure mask has correct size - if not, skip silently
                    if len(mask) != output_tensor.shape[-1]:
                        return output  # Skip masking instead of warning
                    
                    # Expand mask for batch and sequence dimensions
                    expanded_mask = mask.unsqueeze(0).unsqueeze(0)  # [1, 1, features]
                    expanded_mask = expanded_mask.expand(output_tensor.shape[0], output_tensor.shape[1], -1)
                    
                    masked_output = output_tensor * expanded_mask
                    
                elif mask_dim == 'input':
                    # For input masking, we need to be more careful
                    # This is mainly for o_proj where input features are masked
                    # For o_proj, the mask was applied during the forward computation
                    # So we don't need to apply it again here
                    masked_output = output_tensor
                else:
                    masked_output = output_tensor
                
                # Return in the same format as input
                if rest_outputs is not None:
                    return (masked_output,) + rest_outputs
                else:
                    return masked_output
            
            return mask_hook
        
        # Register hooks for all LoRA modules
        for group_key, group_info in self.group_mappings.items():
            group_type = group_info['type']
            for name, proj_type, module in group_info['modules']:
                if self._is_target_lora_layer(module):
                    hook = create_mask_hook()
                    module.register_forward_hook(hook)
    
    def set_training_mode(self, training: bool):
        """Set training mode for differentiable vs discrete masks"""
        self.training_mode = training
    
    def get_current_sparsity(self):
        """Calculate current sparsity level across all groups"""
        if self.training_mode and self.soft_masks:
            masks = self.soft_masks
            total_structures = 0
            active_structures = 0
            
            for group_key, soft_mask in masks.items():
                total_structures += soft_mask.numel()
                # For soft masks, sum the probabilities
                active_structures += soft_mask.sum().item()
            
            sparsity = 1.0 - (active_structures / total_structures) if total_structures > 0 else 0.0
        else:
            # Use binary masks
            if not self.binary_masks:
                return 0.0
            
            total_structures = 0
            active_structures = 0
            
            for group_key, binary_mask in self.binary_masks.items():
                total_structures += binary_mask.numel()
                active_structures += binary_mask.sum().item()
            
            sparsity = 1.0 - (active_structures / total_structures) if total_structures > 0 else 0.0
        
        return sparsity
    
    def count_active_structures(self):
        """Count active attention heads and FFN channels"""
        if self.training_mode and self.soft_masks:
            masks = self.soft_masks
        else:
            masks = self.binary_masks
        
        if not masks:
            # Fallback to default values
            return (self.model_config.get('num_attention_heads', 32),
                   self.model_config.get('ffn_intermediate_size', 11008))
        
        active_heads = 0
        active_ffn = 0
        heads_counted = False
        ffn_counted = False
        
        for group_key, mask in masks.items():
            group_info = self.group_mappings[group_key]
            if group_info['type'] == 'attention_head' and not heads_counted:
                if self.training_mode and isinstance(mask, torch.Tensor):
                    # For soft masks, sum the probabilities
                    active_heads = mask.sum().item()
                else:
                    # For binary masks, count True values
                    active_heads = mask.sum().item()
                heads_counted = True
            elif group_info['type'] == 'ffn_channel' and not ffn_counted:
                if self.training_mode and isinstance(mask, torch.Tensor):
                    active_ffn = mask.sum().item()
                else:
                    active_ffn = mask.sum().item()
                ffn_counted = True
        
        # Fallback to default values if no masks exist
        if not heads_counted:
            active_heads = self.model_config.get('num_attention_heads', 32)
        if not ffn_counted:
            active_ffn = self.model_config.get('ffn_intermediate_size', 11008)
        
        return int(active_heads), int(active_ffn)
    
    def get_mask_scores_for_latency(self):
        """
        Get mask scores organized for latency computation.
        
        Returns:
            Tuple of (head_scores, ffn_scores) where scores represent 
            the probability/importance of each structure being active.
        """
        head_scores = None
        ffn_scores = None
        
        for group_key, scores in self.mask_scores.items():
            group_info = self.group_mappings[group_key]
            if group_info['type'] == 'attention_head' and head_scores is None:
                head_scores = scores
            elif group_info['type'] == 'ffn_channel' and ffn_scores is None:
                ffn_scores = scores
        
        # Fallback to uniform scores if not found
        device = next(self.model.parameters()).device
        if head_scores is None:
            num_heads = self.model_config.get('num_attention_heads', 32)
            head_scores = torch.ones(num_heads, device=device) * 0.1
        if ffn_scores is None:
            num_ffn = self.model_config.get('ffn_intermediate_size', 11008)
            ffn_scores = torch.ones(num_ffn, device=device) * 0.1
        
        return head_scores, ffn_scores
    
    def _is_target_lora_layer(self, module):
        """Check if module is a target LoRA layer"""
        return isinstance(module, (Linear)) and getattr(module, 'r', 0) > 0
    
    def _parse_module_name(self, name):
        """Parse module name to extract layer index and type"""
        parts = name.split('.')
        try:
            # Find layer index - it might be at different positions
            layer_idx = None
            for i, part in enumerate(parts):
                if part.isdigit():
                    layer_idx = int(part)
                    layer_pos = i
                    break
            
            if layer_idx is None:
                return None, None, None
            
            # Look for block type and projection type after layer index
            if layer_pos + 1 < len(parts):
                block_type = parts[layer_pos + 1]  # 'self_attn' or 'mlp'
            else:
                return None, None, None
                
            if layer_pos + 2 < len(parts):
                proj_type = parts[layer_pos + 2]  # 'q_proj', 'up_proj', etc.
            else:
                return None, None, None
            
            if block_type == 'self_attn' and proj_type in ['q_proj', 'k_proj', 'v_proj', 'o_proj']:
                return layer_idx, 'attention_head', proj_type
            elif block_type == 'mlp' and proj_type in ['up_proj', 'gate_proj', 'down_proj']:
                return layer_idx, 'ffn_channel', proj_type
            else:
                return None, None, None
        except (IndexError, ValueError):
            return None, None, None