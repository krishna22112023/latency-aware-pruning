# latency_profiling/pruner_utils.py
import torch
import torch.nn as nn
import numpy as np
import warnings
from typing import Dict, List, Optional, Tuple

# Assuming lora.py is accessible, e.g., from parent dir or installed package
# Adjust import path as needed
try:
    from loraprune.lora import Linear, Linear8bitLt
except ImportError:
    warnings.warn("Could not import LoRA layers. Ensure loraprune package is installed or path is correct.")
    # Define dummy classes if import fails, to allow script structure to work
    class Linear(nn.Linear): pass
    class Linear8bitLt: pass # Placeholder

# Define structural groups (consistent with LoRAPrune)
# NUM_ATTENTION_HEADS needs to be set based on the specific model config
# We will get this dynamically from the model config later.
PRUNING_GROUPS = {
    'self_attn': ['q_proj', 'k_proj', 'v_proj', 'o_proj'],
    'mlp': ['up_proj', 'gate_proj', 'down_proj']
}

def get_model_config(model):
    """Extracts key configuration details from the model."""
    config = {}
    # Try common attribute names for Hugging Face models
    hf_config = getattr(model, 'config', None)
    if hf_config:
        config['num_layers'] = getattr(hf_config, 'num_hidden_layers', 0)
        config['num_attention_heads'] = getattr(hf_config, 'num_attention_heads', 0)
        config['hidden_size'] = getattr(hf_config, 'hidden_size', 0)
        config['ffn_intermediate_size'] = getattr(hf_config, 'intermediate_size', 0)
        config['head_dim'] = config['hidden_size'] // config['num_attention_heads'] if config['num_attention_heads'] > 0 else 0
    else:
        # Add fallbacks or raise error if config cannot be determined
        raise ValueError("Could not automatically determine model configuration (layers, heads, dims).")
    return config

def _is_target_lora_layer(module):
    """Checks if a module is a LoRA layer we might prune (structurally)."""
    return isinstance(module, (Linear, Linear8bitLt)) and hasattr(module, 'lora_A')

def get_layer_name_and_type(module_name: str, model_config: Dict) -> Tuple[Optional[int], Optional[str], Optional[str]]:
    """ Extracts layer index, type (attn/mlp), and projection type from module name. """
    parts = module_name.split('.')
    try:
        # Assuming structure like model.layers.{idx}.{type}.{proj}
        layer_idx = int(parts[2]) # Or adjust index based on actual model structure
        block_type = parts[3] # 'self_attn' or 'mlp'
        proj_type = parts[4] # 'q_proj', 'up_proj', etc.

        if block_type == 'self_attn' and proj_type in PRUNING_GROUPS['self_attn']:
            return layer_idx, 'attention_head', proj_type
        elif block_type == 'mlp' and proj_type in PRUNING_GROUPS['mlp']:
            return layer_idx, 'ffn_channel', proj_type
        else:
            return None, None, None
    except (IndexError, ValueError):
        return None, None, None


def apply_structural_mask_to_layer(
    module: Union[Linear, Linear8bitLt],
    mask: torch.Tensor,
    group_type: str,
    proj_type: str,
    head_dim: int,
    num_total_heads: int
):
    """
    Applies a structural mask (zeros out weights) to a specific LoRA layer.
    NOTE: This modifies the layer in-place for profiling.
          It doesn't *remove* parameters, just zeros them.

    Args:
        module: The LoRA layer (Linear or Linear8bitLt).
        mask (torch.Tensor): A 1D boolean tensor indicating which structures
                             (heads or channels) to KEEP (True = keep, False = prune).
        group_type (str): 'attention_head' or 'ffn_channel'.
        proj_type (str): 'q_proj', 'up_proj', etc.
        head_dim (int): Dimension of each attention head.
        num_total_heads (int): Total number of heads in the original model config.
    """
    if not _is_target_lora_layer(module):
        return # Only apply to target LoRA layers

    assert mask.dim() == 1, "Mask must be 1D (per head or per channel)"

    with torch.no_grad():
        # --- Apply to Base Weight (W0) ---
        # Although W0 is frozen, zeroing it simulates the effect of the structure being gone
        if hasattr(module, 'weight') and module.weight is not None:
            if group_type == 'attention_head':
                num_structures = num_total_heads
                structure_dim = head_dim
                if mask.numel() != num_structures:
                     warnings.warn(f"Mask size ({mask.numel()}) mismatch for attention head ({num_structures}) in {proj_type}. Skipping W0 mask.")
                else:
                    # Q, K, V projections: output dim is structured (num_heads * head_dim)
                    # O projection: input dim is structured (num_heads * head_dim)
                    is_output_structured = proj_type in ['q_proj', 'k_proj', 'v_proj']
                    dim_to_mask = 0 if is_output_structured else 1 # 0 for output features, 1 for input features
                    original_shape = module.weight.shape

                    if dim_to_mask == 0: # Mask output features (rows)
                        w_mask = mask.repeat_interleave(structure_dim).to(module.weight.device)
                        module.weight.data *= w_mask.unsqueeze(1) # Broadcast along input dim
                    else: # Mask input features (columns) - for o_proj
                        w_mask = mask.repeat_interleave(structure_dim).to(module.weight.device)
                        module.weight.data *= w_mask.unsqueeze(0) # Broadcast along output dim

            elif group_type == 'ffn_channel':
                num_structures = module.out_features if proj_type in ['up_proj', 'gate_proj'] else module.in_features
                if mask.numel() != num_structures:
                     warnings.warn(f"Mask size ({mask.numel()}) mismatch for FFN channel ({num_structures}) in {proj_type}. Skipping W0 mask.")
                else:
                    # Up/Gate projections: output dim is structured
                    # Down projection: input dim is structured
                    is_output_structured = proj_type in ['up_proj', 'gate_proj']
                    dim_to_mask = 0 if is_output_structured else 1
                    if dim_to_mask == 0:
                        module.weight.data *= mask.unsqueeze(1).to(module.weight.device)
                    else:
                        module.weight.data *= mask.unsqueeze(0).to(module.weight.device)

        # --- Apply to LoRA Weights (A and B) ---
        # We need to mask the *corresponding* dimensions in A and B
        if module.r > 0 and hasattr(module, 'lora_A') and hasattr(module, 'lora_B'):
            if group_type == 'attention_head':
                 num_structures = num_total_heads
                 structure_dim = head_dim
                 if mask.numel() != num_structures:
                     warnings.warn(f"Mask size ({mask.numel()}) mismatch for attention head ({num_structures}) in {proj_type}. Skipping LoRA mask.")
                 else:
                    # Mask affects B output or A input depending on projection type
                    if proj_type in ['q_proj', 'k_proj', 'v_proj']: # Mask B's output dim
                        b_mask = mask.repeat_interleave(structure_dim).to(module.lora_B.weight.device)
                        module.lora_B.weight.data *= b_mask.unsqueeze(1) # (out_features, r) -> mask rows
                    elif proj_type == 'o_proj': # Mask A's input dim
                        a_mask = mask.repeat_interleave(structure_dim).to(module.lora_A.weight.device)
                        # A's shape is (r, in_features) -> mask columns
                        module.lora_A.weight.data *= a_mask.unsqueeze(0)

            elif group_type == 'ffn_channel':
                num_structures = module.out_features if proj_type in ['up_proj', 'gate_proj'] else module.in_features
                if mask.numel() != num_structures:
                     warnings.warn(f"Mask size ({mask.numel()}) mismatch for FFN channel ({num_structures}) in {proj_type}. Skipping LoRA mask.")
                else:
                    if proj_type in ['up_proj', 'gate_proj']: # Mask B's output dim
                        module.lora_B.weight.data *= mask.unsqueeze(1).to(module.lora_B.weight.device)
                    elif proj_type == 'down_proj': # Mask A's input dim
                         # A's shape is (r, in_features) -> mask columns
                        module.lora_A.weight.data *= mask.unsqueeze(0).to(module.lora_A.weight.device)

        # --- Apply to Bias (if applicable and trainable) ---
        if hasattr(module, 'bias') and module.bias is not None:
             # Bias is always associated with the output dimension
             if group_type == 'attention_head' and proj_type in ['q_proj', 'k_proj', 'v_proj']:
                 if mask.numel() == num_total_heads:
                     bias_mask = mask.repeat_interleave(head_dim).to(module.bias.device)
                     module.bias.data *= bias_mask
             elif group_type == 'ffn_channel' and proj_type in ['up_proj', 'gate_proj']:
                 if mask.numel() == module.out_features:
                     module.bias.data *= mask.to(module.bias.device)


def apply_global_structural_mask(model, num_active_heads: int, num_active_ffn: int):
    """
    Applies a uniform structural mask across all layers of the model.
    Zeros out weights corresponding to pruned heads/channels IN-PLACE.

    Args:
        model: The LoRA-wrapped model.
        num_active_heads (int): Number of attention heads to KEEP active.
        num_active_ffn (int): Number of FFN intermediate channels to KEEP active.
    """
    model_config = get_model_config(model)
    num_total_heads = model_config['num_attention_heads']
    num_total_ffn = model_config['ffn_intermediate_size']
    head_dim = model_config['head_dim']

    if num_active_heads < 0 or num_active_heads > num_total_heads:
        raise ValueError(f"num_active_heads ({num_active_heads}) must be between 0 and {num_total_heads}")
    if num_active_ffn < 0 or num_active_ffn > num_total_ffn:
        raise ValueError(f"num_active_ffn ({num_active_ffn}) must be between 0 and {num_total_ffn}")

    # Create masks (True = Keep)
    head_mask = torch.zeros(num_total_heads, dtype=torch.bool)
    head_mask[:num_active_heads] = True

    ffn_mask = torch.zeros(num_total_ffn, dtype=torch.bool)
    ffn_mask[:num_active_ffn] = True

    print(f"Applying mask: Keep {num_active_heads}/{num_total_heads} heads, {num_active_ffn}/{num_total_ffn} FFN channels.")

    # Iterate through modules and apply masks
    for name, module in model.named_modules():
        layer_idx, group_type, proj_type = get_layer_name_and_type(name, model_config)

        if group_type == 'attention_head':
            apply_structural_mask_to_layer(module, head_mask, group_type, proj_type, head_dim, num_total_heads)
        elif group_type == 'ffn_channel':
            apply_structural_mask_to_layer(module, ffn_mask, group_type, proj_type, head_dim, num_total_heads)

