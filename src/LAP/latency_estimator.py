import pickle
import numpy as np
import torch
import torch.nn.functional as F
from typing import Dict, Tuple, List
import warnings
import os


class DifferentiableLatencyEstimator:
    """
    Differentiable latency estimator.
    Computes expected latency over soft architectural configurations.
    """
    
    def __init__(self, latency_lut_path: str, model_config: Dict, temperature: float = 1.0):
        self.model_config = model_config
        self.temperature = temperature  # For Gumbel softmax
        self.lut = {}
        self.load_latency_lut(latency_lut_path)
        
        # Precompute configuration space
        self.config_space = self._build_config_space()
        self.latency_matrix = self._build_latency_matrix()
        
    def load_latency_lut(self, lut_path: str):
        """Load the latency lookup table from profiling results"""
        possible_files = [
            os.path.join(lut_path, "Llama_3.2_1B.pkl"),
            os.path.join(lut_path, "latency_lut.pkl"),
            lut_path  # In case the path is directly to the file
        ]
        
        loaded = False
        for fpath in possible_files:
            if os.path.exists(fpath):
                try:
                    with open(fpath, 'rb') as f:
                        self.lut = pickle.load(f)
                    print(f"Loaded latency LUT from {fpath}")
                    print(f"LUT contains {len(self.lut)} configurations")
                    loaded = True
                    break
                except Exception as e:
                    warnings.warn(f"Could not load {fpath}: {e}")
        
        if not loaded:
            warnings.warn(f"Could not load latency LUT. Creating fallback.")
            self.lut = self._create_fallback_lut()
    
    def _create_fallback_lut(self):
        """Create a simple fallback LUT when the actual LUT can't be loaded"""
        fallback_lut = {}
        
        base_latency = 100.0
        head_factor = 2.0
        ffn_factor = 0.01
        
        for seq_len in [128, 256, 512]:
            for batch_size in [1, 16]:
                for heads in range(1, 33):
                    for ffn in range(256, 11265, 256):
                        latency = base_latency + head_factor * heads + ffn_factor * ffn
                        key = (batch_size, seq_len, 4096, heads, ffn)
                        fallback_lut[key] = (latency, latency * 0.1)
        
        return fallback_lut
    
    def _build_config_space(self):
        """Build the space of possible configurations from LUT"""
        head_configs = set()
        ffn_configs = set()
        
        for (batch_size, seq_len, hidden_size, heads, ffn), _ in self.lut.items():
            head_configs.add(heads)
            ffn_configs.add(ffn)
        
        # Sort for consistent ordering
        head_configs = sorted(list(head_configs))
        ffn_configs = sorted(list(ffn_configs))
        
        return {
            'heads': head_configs,
            'ffn': ffn_configs
        }
    
    def _build_latency_matrix(self):
        """Build a matrix of latencies for efficient computation"""
        head_configs = self.config_space['heads']
        ffn_configs = self.config_space['ffn']
        
        # Create latency matrix [num_head_configs, num_ffn_configs]
        latency_matrix = torch.zeros(len(head_configs), len(ffn_configs))
        
        for i, heads in enumerate(head_configs):
            for j, ffn in enumerate(ffn_configs):
                # Use default values for batch_size and seq_len
                latency = self._lookup_latency(1, 256, 4096, heads, ffn)
                latency_matrix[i, j] = latency
        
        return latency_matrix
    
    def _lookup_latency(self, batch_size: int, seq_len: int, hidden_size: int,
                       active_heads: int, active_ffn: int) -> float:
        """Lookup latency from LUT with interpolation"""
        key = (batch_size, seq_len, hidden_size, active_heads, active_ffn)
        
        if key in self.lut:
            return self.lut[key][0]  # Return mean latency
        
        # Interpolation fallback
        return self._interpolate_latency(batch_size, seq_len, hidden_size, 
                                       active_heads, active_ffn)
    
    def _interpolate_latency(self, batch_size: int, seq_len: int, hidden_size: int,
                           active_heads: int, active_ffn: int) -> float:
        """Interpolate latency from nearest neighbors"""
        if not self.lut:
            return 100.0 + 2.0 * active_heads + 0.01 * active_ffn
        
        candidates = []
        for (b, s, h, heads, ffn), (latency, _) in self.lut.items():
            distance = (abs(b - batch_size) + abs(s - seq_len) * 0.01 + 
                       abs(h - hidden_size) * 0.0001 + abs(heads - active_heads) + 
                       abs(ffn - active_ffn) * 0.001)
            candidates.append((distance, latency))
        
        candidates.sort(key=lambda x: x[0])
        
        if len(candidates) == 0:
            return 100.0 + 2.0 * active_heads + 0.01 * active_ffn
        
        # Weighted average of k nearest neighbors
        k = min(4, len(candidates))
        total_weight = 0
        weighted_latency = 0
        
        for i in range(k):
            distance, latency = candidates[i]
            weight = 1.0 / (1.0 + distance)
            weighted_latency += weight * latency
            total_weight += weight
        
        return weighted_latency / total_weight if total_weight > 0 else candidates[0][1]
    
    def compute_expected_latency_soft(self, head_scores: torch.Tensor, 
                                    ffn_scores: torch.Tensor,
                                    batch_size: int = 1, seq_len: int = 256) -> torch.Tensor:
        """
        Compute expected latency using soft architectural sampling.
        
        Args:
            head_scores: Learnable scores for attention heads [num_heads]
            ffn_scores: Learnable scores for FFN channels [num_ffn_channels]
            batch_size: Batch size for latency lookup
            seq_len: Sequence length for latency lookup
        
        Returns:
            Expected latency as a differentiable tensor
        """
        device = head_scores.device
        
        # Convert scores to probabilities using softmax
        head_probs = F.softmax(head_scores / self.temperature, dim=0)
        ffn_probs = F.softmax(ffn_scores / self.temperature, dim=0)
        
        # Get configuration spaces
        head_configs = self.config_space['heads']
        ffn_configs = self.config_space['ffn']
        
        # Map scores to configuration probabilities
        # Aggregate probabilities for each possible configuration
        head_config_probs = torch.zeros(len(head_configs), device=device)
        ffn_config_probs = torch.zeros(len(ffn_configs), device=device)
        
        # Sum probabilities for active structures to get configuration probabilities
        for i, num_heads in enumerate(head_configs):
            if num_heads <= len(head_scores):
                # Probability of having exactly num_heads active heads
                # This is a simplification - could use more sophisticated sampling
                head_config_probs[i] = head_probs[:num_heads].sum()
        
        for i, num_ffn in enumerate(ffn_configs):
            if num_ffn <= len(ffn_scores):
                # Probability of having exactly num_ffn active channels
                ffn_config_probs[i] = ffn_probs[:num_ffn].sum()
        
        # Normalize probabilities
        head_config_probs = head_config_probs / (head_config_probs.sum() + 1e-8)
        ffn_config_probs = ffn_config_probs / (ffn_config_probs.sum() + 1e-8)
        
        # Get latency matrix on correct device
        latency_matrix = self.latency_matrix.to(device)
        
        # Compute expected latency: E[latency] = Σ P(config) × latency(config)
        expected_latency = torch.sum(
            head_config_probs.unsqueeze(1) * ffn_config_probs.unsqueeze(0) * latency_matrix
        )
        
        return expected_latency
    
    def compute_expected_latency_gumbel(self, head_scores: torch.Tensor, 
                                      ffn_scores: torch.Tensor,
                                      batch_size: int = 1, seq_len: int = 256,
                                      hard: bool = False) -> torch.Tensor:
        """
        Compute expected latency using Gumbel softmax for better gradient flow.
        
        Args:
            head_scores: Learnable scores for attention heads [num_heads]
            ffn_scores: Learnable scores for FFN channels [num_ffn_channels]
            batch_size: Batch size for latency lookup
            seq_len: Sequence length for latency lookup
            hard: Whether to use hard (one-hot) or soft sampling
        
        Returns:
            Expected latency as a differentiable tensor
        """
        device = head_scores.device
        
        # Use Gumbel softmax for better gradient flow
        head_weights = F.gumbel_softmax(head_scores, tau=self.temperature, hard=hard)
        ffn_weights = F.gumbel_softmax(ffn_scores, tau=self.temperature, hard=hard)
        
        # Compute weighted number of active structures
        num_active_heads = torch.sum(head_weights * torch.arange(1, len(head_weights) + 1, 
                                                               device=device, dtype=torch.float32))
        num_active_ffn = torch.sum(ffn_weights * torch.arange(1, len(ffn_weights) + 1, 
                                                             device=device, dtype=torch.float32))
        
        # Linear interpolation in latency space (differentiable approximation)
        latency = self._linear_latency_approximation(num_active_heads, num_active_ffn, device)
        
        return latency
    
    def _linear_latency_approximation(self, num_heads: torch.Tensor, 
                                    num_ffn: torch.Tensor, device: torch.device) -> torch.Tensor:
        """
        Linear approximation of latency for smooth gradients.
        Based on the observation that latency often scales roughly linearly with active structures.
        """
        # Fit linear model: latency = base + head_coeff * heads + ffn_coeff * ffn
        # These coefficients should be learned from the LUT data
        
        # Extract coefficients from LUT data
        if not hasattr(self, '_linear_coeffs'):
            self._compute_linear_coefficients()
        
        base, head_coeff, ffn_coeff = self._linear_coeffs
        
        # Convert to tensors on correct device
        base = torch.tensor(base, device=device, dtype=torch.float32)
        head_coeff = torch.tensor(head_coeff, device=device, dtype=torch.float32)
        ffn_coeff = torch.tensor(ffn_coeff, device=device, dtype=torch.float32)
        
        latency = base + head_coeff * num_heads + ffn_coeff * num_ffn
        
        return latency
    
    def _compute_linear_coefficients(self):
        """Compute linear approximation coefficients from LUT data"""
        if not self.lut:
            # Fallback coefficients
            self._linear_coeffs = (100.0, 2.0, 0.01)
            return
        
        # Extract data points
        heads_list = []
        ffn_list = []
        latency_list = []
        
        for (batch_size, seq_len, hidden_size, heads, ffn), (latency, _) in self.lut.items():
            if batch_size == 1 and seq_len == 256:  # Use consistent configuration
                heads_list.append(heads)
                ffn_list.append(ffn)
                latency_list.append(latency)
        
        if len(heads_list) < 3:
            # Not enough data for fitting
            self._linear_coeffs = (100.0, 2.0, 0.01)
            return
        
        # Simple linear regression: latency = base + head_coeff * heads + ffn_coeff * ffn
        import numpy as np
        
        X = np.column_stack([np.ones(len(heads_list)), heads_list, ffn_list])
        y = np.array(latency_list)
        
        try:
            coeffs = np.linalg.lstsq(X, y, rcond=None)[0]
            self._linear_coeffs = tuple(coeffs)
        except:
            # Fallback if regression fails
            self._linear_coeffs = (100.0, 2.0, 0.01)
    
    def estimate_latency_discrete(self, active_heads: int, active_ffn: int, 
                                seq_len: int = 256, batch_size: int = 1) -> float:
        """
        Non-differentiable latency estimation for evaluation/logging.
        """
        return self._lookup_latency(batch_size, seq_len, 4096, active_heads, active_ffn)
    
    def print_lut_stats(self):
        """Print statistics about the loaded LUT"""
        if not self.lut:
            print("No LUT loaded")
            return
        
        print(f"Latency LUT Statistics:")
        print(f"Total configurations: {len(self.lut)}")
        
        latencies = [v[0] for v in self.lut.values()]
        heads = set(k[3] for k in self.lut.keys())
        ffns = set(k[4] for k in self.lut.keys())
        seq_lens = set(k[1] for k in self.lut.keys())
        
        print(f"Latency range: {min(latencies):.2f} - {max(latencies):.2f} ms")
        print(f"Head counts: {min(heads)} - {max(heads)}")
        print(f"FFN sizes: {min(ffns)} - {max(ffns)}")
        print(f"Sequence lengths: {sorted(seq_lens)}")
        
        print(f"Configuration space:")
        print(f"  Head configs: {len(self.config_space['heads'])}")
        print(f"  FFN configs: {len(self.config_space['ffn'])}")
        
        if hasattr(self, '_linear_coeffs'):
            base, head_coeff, ffn_coeff = self._linear_coeffs
            print(f"Linear approximation: latency = {base:.2f} + {head_coeff:.2f}*heads + {ffn_coeff:.4f}*ffn")