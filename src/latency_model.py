import torch
import time
import numpy as np
import pickle
import fire
import os
import warnings
from transformers import AutoModelForCausalLM, AutoTokenizer, LlamaConfig # Example config import
from typing import Dict, List, Tuple

from src.peft_model import get_peft_model
from src.lora import LoraConfig
from src.utils import apply_global_structural_mask, get_model_config

# --- Configuration ---
DEFAULT_BASE_MODEL = "baffo32/decapoda-research-llama-7B-hf" 
DEFAULT_LORA_R = 8 # LoRA rank (doesn't affect latency much if merged, but needed for setup)
DEFAULT_LORA_ALPHA = 16
DEFAULT_LORA_TARGET_MODULES = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
DEFAULT_BATCH_SIZE = 1 # Profile latency for batch size 1 typically
DEFAULT_DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
WARMUP_RUNS = 10
MEASUREMENT_RUNS = 20 # Number of runs to average for latency

# --- Profiling Ranges ---
# Sequence lengths to profile
SEQ_LENS_TO_PROFILE = [128, 256, 512, 1024, 2048]
# Number of *active* attention heads to profile (from min to max)
# Will depend on the specific model's total heads
HEAD_COUNTS_TO_PROFILE = None # Will be set dynamically based on model config
# Number of *active* FFN intermediate channels to profile
FFN_COUNTS_TO_PROFILE = None # Will be set dynamically

# --- Helper Functions ---

@torch.no_grad()
def measure_latency(model, dummy_input, device):
    """Measures average inference latency for a model and input."""
    latencies = []
    # Use CUDA events for accurate GPU timing
    if device == "cuda":
        starter, ender = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)

        # Warmup runs
        for _ in range(WARMUP_RUNS):
            _ = model(dummy_input)
            torch.cuda.synchronize()

        # Measurement runs
        for _ in range(MEASUREMENT_RUNS):
            starter.record()
            _ = model(dummy_input)
            ender.record()
            torch.cuda.synchronize()
            latencies.append(starter.elapsed_time(ender)) # Time in ms

    else: # CPU timing (less accurate)
         # Warmup runs
        for _ in range(WARMUP_RUNS):
            _ = model(dummy_input)

        # Measurement runs
        for _ in range(MEASUREMENT_RUNS):
            t0 = time.perf_counter()
            _ = model(dummy_input)
            latencies.append((time.perf_counter() - t0) * 1000) # time in ms

    avg_latency = np.mean(latencies)
    std_latency = np.std(latencies)
    return avg_latency, std_latency

def generate_dummy_input(batch_size, seq_len, vocab_size, device):
    """Generates random token IDs as dummy input."""
    return torch.randint(0, vocab_size, (batch_size, seq_len), device=device)

# --- Main Profiling Function ---

def build_latency_table(
    base_model: str = DEFAULT_BASE_MODEL,
    lora_r: int = DEFAULT_LORA_R,
    lora_alpha: int = DEFAULT_LORA_ALPHA,
    lora_target_modules: List[str] = DEFAULT_LORA_TARGET_MODULES,
    batch_size: int = DEFAULT_BATCH_SIZE,
    device: str = DEFAULT_DEVICE,
    output_dir: str = "latency_luts",
    model_load_kwargs: Dict = {"torch_dtype": torch.float16, "device_map": "auto"}
):
    """
    Profiles model latency for different structural configurations and saves LUTs.

    Args:
        base_model (str): Path or name of the base Hugging Face model.
        lora_r (int): LoRA rank for setup.
        lora_alpha (int): LoRA alpha for setup.
        lora_target_modules (List[str]): Modules to apply LoRA to (for setup).
        batch_size (int): Batch size to use for profiling (N).
        device (str): Device to run profiling on ('cuda' or 'cpu').
        output_dir (str): Directory to save the latency LUT pickle files.
        model_load_kwargs (Dict): Keyword arguments for AutoModelForCausalLM.from_pretrained.
    """
    print(f"--- Starting Latency Profiling ---")
    print(f"Base Model: {base_model}")
    print(f"Batch Size (N): {batch_size}")
    print(f"Device: {device}")
    print(f"Output Dir: {output_dir}")

    # --- Model and Tokenizer Loading ---
    print("Loading model and tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(base_model, legacy=False)
    # Ensure device map is handled correctly if specified
    if "device_map" in model_load_kwargs and model_load_kwargs["device_map"] == "auto":
         print("Using device_map='auto'")
    elif device == 'cuda':
         model_load_kwargs['device_map'] = {'': 0} # Load to specific GPU if not auto
         print(f"Loading model to default CUDA device.")
    else:
         model_load_kwargs.pop('device_map', None) # Remove device_map for CPU
         print("Loading model to CPU.")


    model = AutoModelForCausalLM.from_pretrained(base_model, **model_load_kwargs)
    model.eval() # Set to evaluation mode

    # --- Get Model Config ---
    model_config = get_model_config(model)
    print("Model Configuration:", model_config)
    num_total_heads = model_config['num_attention_heads']
    num_total_ffn = model_config['ffn_intermediate_size']
    vocab_size = getattr(model.config, 'vocab_size', 32000) # Get vocab size

    # --- Define Profiling Ranges Dynamically ---
    # Profile from 1 head up to total heads, maybe in steps
    head_counts_to_profile = sorted(list(set([1, 2, 4] + list(range(8, num_total_heads + 1, 8)))))
    if num_total_heads not in head_counts_to_profile: head_counts_to_profile.append(num_total_heads)
    print(f"Profiling Head Counts: {head_counts_to_profile}")

    # Profile FFN channels, e.g., in steps of 128 or 256 up to total
    ffn_step = 256 # Example step size
    ffn_counts_to_profile = sorted(list(set([ffn_step // 2] + list(range(ffn_step, num_total_ffn + 1, ffn_step)))))
    if num_total_ffn not in ffn_counts_to_profile: ffn_counts_to_profile.append(num_total_ffn)
    print(f"Profiling FFN Counts: {ffn_counts_to_profile}")

    # --- Setup LoRA (needed for structure, weights aren't used) ---
    # We apply LoRA structure just to potentially modify layers,
    # but we won't actually use LoRA weights for latency measurement.
    # The apply_global_structural_mask works on the base weights directly.
    # If apply_global_structural_mask is adapted to also mask LoRA A/B,
    # then this setup is necessary. Otherwise, it might be skippable if
    # apply_global_structural_mask only modifies base weights.
    # Let's keep it for now assuming the util might mask LoRA too.
    print("Applying LoRA structure (weights not used for profiling)...")
    lora_config = LoraConfig(
        r=lora_r,
        lora_alpha=lora_alpha,
        target_modules=lora_target_modules,
        bias="none",
        task_type="CAUSAL_LM",
    )
    # Use the PEFT function to wrap the model
    # This replaces layers with LoRA layers, which pruner_utils expects
    model = get_peft_model(model, lora_config)
    print("LoRA structure applied.")

    # --- Initialize LUTs ---
    latency_lut = {} # Combined LUT for simplicity: key = (type, N, L, D, active_count)

    # --- Profiling Loop ---
    for seq_len in SEQ_LENS_TO_PROFILE:
        print(f"\n--- Profiling for Sequence Length (L) = {seq_len} ---")
        dummy_input = generate_dummy_input(batch_size, seq_len, vocab_size, device)

        # Profile different numbers of active attention heads (assuming FFN is full)
        print(f"Profiling Attention Heads (FFN Full = {num_total_ffn})...")
        for num_active_heads in head_counts_to_profile:
            # Create a fresh copy or reset the model state before applying mask
            # Easiest is to reload, but slow. Alternative: store original weights and restore.
            # For now, we assume apply_global_structural_mask modifies in-place and
            # we need to restore. Let's try zeroing/unzeroing.

            # Store original state (simple approach: just zero/unzero)
            # A more robust way involves deep copying weights before masking.
            original_weights = {name: p.clone() for name, p in model.named_parameters()}

            # Apply the mask: keep `num_active_heads`, keep `num_total_ffn`
            apply_global_structural_mask(model, num_active_heads, num_total_ffn)

            # Measure
            avg_lat, std_lat = measure_latency(model, dummy_input, device)
            key = ("attn", batch_size, seq_len, model_config['hidden_size'], num_active_heads)
            latency_lut[key] = avg_lat
            print(f"  Heads={num_active_heads:<4}: Latency={avg_lat:.4f} ms (std={std_lat:.4f})")

            # Restore original weights
            with torch.no_grad():
                for name, p in model.named_parameters():
                    if name in original_weights:
                        p.copy_(original_weights[name])
            del original_weights # Free memory
            torch.cuda.empty_cache() if device == 'cuda' else None


        # Profile different numbers of active FFN channels (assuming Attention is full)
        print(f"\nProfiling FFN Channels (Heads Full = {num_total_heads})...")
        for num_active_ffn in ffn_counts_to_profile:
            original_weights = {name: p.clone() for name, p in model.named_parameters()}

            # Apply the mask: keep `num_total_heads`, keep `num_active_ffn`
            apply_global_structural_mask(model, num_total_heads, num_active_ffn)

            # Measure
            avg_lat, std_lat = measure_latency(model, dummy_input, device)
            key = ("ffn", batch_size, seq_len, model_config['hidden_size'], num_active_ffn)
            latency_lut[key] = avg_lat
            print(f"  FFN={num_active_ffn:<5}: Latency={avg_lat:.4f} ms (std={std_lat:.4f})")

            # Restore original weights
            with torch.no_grad():
                for name, p in model.named_parameters():
                     if name in original_weights:
                         p.copy_(original_weights[name])
            del original_weights
            torch.cuda.empty_cache() if device == 'cuda' else None


    # --- Save LUT ---
    os.makedirs(output_dir, exist_ok=True)
    # Sanitize base_model name for filename
    safe_model_name = base_model.split('/')[-1].replace('-', '_')
    lut_filename = os.path.join(output_dir, f"latency_lut_{safe_model_name}_N{batch_size}.pkl")
    try:
        with open(lut_filename, "wb") as f:
            pickle.dump(latency_lut, f)
        print(f"\nLatency lookup table saved to: {lut_filename}")
    except Exception as e:
        print(f"\nError saving LUT to {lut_filename}: {e}")

    print("--- Profiling Complete ---")


if __name__ == "__main__":
    # Example usage:
    # python -m latency_profiling.build_latency_lut --base_model="meta-llama/Llama-2-7b-hf" --output_dir="latency_luts"
    fire.Fire(build_latency_table)
