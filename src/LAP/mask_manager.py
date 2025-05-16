import torch
import torch.nn as nn
from typing import Dict, List, Tuple
import warnings
from src.latency_profiling.lora import Linear#, Linear8bitLt
from collections import defaultdict
import numpy as np


class LatencyAwareMaskManager:
    """Manages learnable mask scores and binary masks for structured pruning"""
    
    def __init__(self, model, config, model_config):
        self.model = model
        self.config = config
        self.model_config = model_config
        
        # Learnable mask scores (m in methodology)
        self.mask_scores = nn.ParameterDict()
        
        # Binary masks derived from scores (M in methodology)
        self.binary_masks = {}
        
        # Importance scores for each structural group (S_g from Eq. 7)
        self.importance_scores = {}
        
        # Moving average for importance scores
        self.importance_ma = {}
        
        # Group mappings
        self.group_mappings = self.create_group_mappings()
        
        # Initialize mask scores
        self.init_mask_scores()
    
    def create_group_mappings(self):
        """Create mappings between modules and structural groups"""
        mappings = {}
        
        for name, module in self.model.named_modules():
            if self._is_target_lora_layer(module):
                layer_idx, group_type, proj_type = self._parse_module_name(name)
                if layer_idx is not None and group_type is not None:
                    group_key = f"{layer_idx}.{group_type}"
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
            self.mask_scores[group_key] = nn.Parameter(
                torch.randn(num_groups, device=device) * 0.01 + 0.1
            )
            
            # Initialize importance scores
            self.importance_scores[group_key] = torch.ones(num_groups, device=device)
            self.importance_ma[group_key] = torch.ones(num_groups, device=device)
    
    def get_binary_masks(self, target_sparsity):
        """Convert mask scores to binary masks using TopK operation (Eq. 2)"""
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
    
    def apply_binary_masks(self):
        """Apply binary masks to model modules"""
        for group_key, binary_mask in self.binary_masks.items():
            if group_key in self.group_mappings:
                self._apply_mask_to_group(group_key, binary_mask)
    
    def _apply_mask_to_group(self, group_key, binary_mask):
        """Apply binary mask to a specific structural group"""
        group_info = self.group_mappings[group_key]
        group_type = group_info['type']
        
        for name, proj_type, module in group_info['modules']:
            if group_type == 'attention_head':
                self._apply_attention_mask(module, binary_mask, proj_type)
            elif group_type == 'ffn_channel':
                self._apply_ffn_mask(module, binary_mask, proj_type)
    
    def _apply_attention_mask(self, module, mask, proj_type):
        """Apply mask to attention module"""
        head_dim = self.model_config['head_dim']
        num_heads = self.model_config['num_attention_heads']
        
        if not self._is_target_lora_layer(module):
            return
        
        # Create expanded mask for all dimensions
        if proj_type in ['q_proj', 'k_proj', 'v_proj']:
            # Output dimension is structured
            expanded_mask = mask.repeat_interleave(head_dim)
        elif proj_type == 'o_proj':
            # Input dimension is structured
            expanded_mask = mask.repeat_interleave(head_dim)
        else:
            return
        
        # Store mask for later application
        if not hasattr(module, 'lora_mask'):
            module.lora_mask = expanded_mask.clone()
        else:
            module.lora_mask.data = expanded_mask
    
    def _apply_ffn_mask(self, module, mask, proj_type):
        """Apply mask to FFN module"""
        if not self._is_target_lora_layer(module):
            return
        
        # Store mask for later application
        if not hasattr(module, 'lora_mask'):
            module.lora_mask = mask.clone()
        else:
            module.lora_mask.data = mask
    
    def get_current_sparsity(self):
        """Calculate current sparsity level across all groups"""
        total_params = 0
        pruned_params = 0
        
        for group_key, binary_mask in self.binary_masks.items():
            group_info = self.group_mappings[group_key]
            for _, proj_type, module in group_info['modules']:
                if self._is_target_lora_layer(module):
                    param_count = np.prod(module.weight.shape)
                    total_params += param_count
                    
                    if hasattr(module, 'lora_mask'):
                        sparsity = 1 - module.lora_mask.float().mean()
                        pruned_params += param_count * sparsity
        
        return pruned_params / total_params if total_params > 0 else 0.0
    
    def count_active_structures(self):
        """Count active attention heads and FFN channels"""
        active_heads = 0
        active_ffn = 0
        
        for group_key, binary_mask in self.binary_masks.items():
            group_info = self.group_mappings[group_key]
            if group_info['type'] == 'attention_head':
                active_heads = binary_mask.sum().item()
            elif group_info['type'] == 'ffn_channel':
                # Take first occurrence as representative
                if active_ffn == 0:
                    active_ffn = binary_mask.sum().item()
        
        return active_heads, active_ffn
    
    def _is_target_lora_layer(self, module):
        """Check if module is a target LoRA layer"""
        return isinstance(module, (Linear)) and getattr(module, 'r', 0) > 0
    
    def _parse_module_name(self, name):
        """Parse module name to extract layer index and type"""
        parts = name.split('.')
        try:
            layer_idx = int(parts[2])  # Adjust based on model structure
            block_type = parts[3]  # 'self_attn' or 'mlp'
            proj_type = parts[4]  # 'q_proj', 'up_proj', etc.
            
            if block_type == 'self_attn' and proj_type in ['q_proj', 'k_proj', 'v_proj', 'o_proj']:
                return layer_idx, 'attention_head', proj_type
            elif block_type == 'mlp' and proj_type in ['up_proj', 'gate_proj', 'down_proj']:
                return layer_idx, 'ffn_channel', proj_type
            else:
                return None, None, None
        except (IndexError, ValueError):
            return None, None, None