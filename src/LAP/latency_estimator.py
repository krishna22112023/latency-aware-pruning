import pickle
import numpy as np
from typing import Dict, Tuple, List
import warnings
import os


class LatencyEstimator:
    """Estimates inference latency based on structural configuration"""
    
    def __init__(self, latency_lut_path: str, model_config: Dict):
        self.model_config = model_config
        self.lut = {}
        self.load_latency_lut(latency_lut_path)
        
        # Cache for interpolation
        self._cached_configs = {}
    
    def load_latency_lut(self, lut_path: str):
        """Load the latency lookup table from profiling results"""
        # Try different possible paths and filenames
        possible_files = [
            os.path.join(lut_path, "decapoda_research_llama_7B_hf.pkl"),
            os.path.join(lut_path, "latency_lut.pkl"),
            os.path.join(lut_path, "latency_table.pkl"),
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
            warnings.warn(f"Could not load latency LUT from {lut_path}. Will use estimated latency.")
            # Create a simple fallback LUT
            self.lut = self._create_fallback_lut()
    
    def _create_fallback_lut(self):
        """Create a simple fallback LUT when the actual LUT can't be loaded"""
        fallback_lut = {}
        
        # Simple linear model: latency = base + head_factor * heads + ffn_factor * ffn
        base_latency = 100.0  # milliseconds
        head_factor = 2.0     # ms per head
        ffn_factor = 0.1      # ms per FFN unit
        
        # Generate some sample points
        for seq_len in [256, 512, 1024]:
            for heads in range(1, 33):  # up to 32 heads
                for ffn in range(512, 11265, 512):  # FFN sizes
                    latency = base_latency + head_factor * heads + ffn_factor * ffn
                    key = (1, seq_len, 4096, heads, ffn)  # (batch, seq_len, hidden, heads, ffn)
                    fallback_lut[key] = (latency, latency * 0.1)  # (mean, std)
        
        return fallback_lut
    
    def estimate_latency(self, active_heads: int, active_ffn: int, 
                        seq_len: int = 256, batch_size: int = 1) -> float:
        """
        Estimate latency for given structural configuration
        
        Args:
            active_heads: Number of active attention heads
            active_ffn: Number of active FFN channels
            seq_len: Sequence length (default from config)
            batch_size: Batch size
        
        Returns:
            Estimated latency in milliseconds
        """
        hidden_size = self.model_config.get('hidden_size', 4096)
        key = (batch_size, seq_len, hidden_size, active_heads, active_ffn)
        
        # Direct lookup first
        if key in self.lut:
            return self.lut[key][0]  # Return mean latency
        
        # Try interpolation
        estimated_latency = self._interpolate_latency(
            batch_size, seq_len, hidden_size, active_heads, active_ffn
        )
        
        return estimated_latency
    
    def _interpolate_latency(self, batch_size: int, seq_len: int, hidden_size: int,
                           active_heads: int, active_ffn: int) -> float:
        """Interpolate latency from nearest neighbors in LUT"""
        if not self.lut:
            # Fallback to simple linear model
            return self._linear_model_estimate(active_heads, active_ffn)
        
        # Find nearest neighbors
        candidates = []
        for (b, s, h, heads, ffn), (latency, _) in self.lut.items():
            if b == batch_size and s == seq_len and h == hidden_size:
                distance = abs(heads - active_heads) + abs(ffn - active_ffn) * 0.001
                candidates.append((distance, latency, heads, ffn))
        
        if not candidates:
            # No exact matches, use broadest search
            for (b, s, h, heads, ffn), (latency, _) in self.lut.items():
                distance = (abs(b - batch_size) + abs(s - seq_len) * 0.01 + 
                           abs(h - hidden_size) * 0.0001 + abs(heads - active_heads) + 
                           abs(ffn - active_ffn) * 0.001)
                candidates.append((distance, latency, heads, ffn))
        
        # Sort by distance and use nearest neighbors
        candidates.sort(key=lambda x: x[0])
        
        if len(candidates) == 0:
            return self._linear_model_estimate(active_heads, active_ffn)
        
        # Use weighted average of k nearest neighbors
        k = min(4, len(candidates))
        total_weight = 0
        weighted_latency = 0
        
        for i in range(k):
            distance, latency, _, _ = candidates[i]
            weight = 1.0 / (1.0 + distance)  # Inverse distance weighting
            weighted_latency += weight * latency
            total_weight += weight
        
        return weighted_latency / total_weight if total_weight > 0 else candidates[0][1]
    
    def _linear_model_estimate(self, active_heads: int, active_ffn: int) -> float:
        """Simple linear model for latency estimation"""
        # Based on empirical observations
        base_latency = 50.0
        head_contribution = active_heads * 2.0
        ffn_contribution = active_ffn * 0.01
        
        return base_latency + head_contribution + ffn_contribution
    
    def get_latency_gradient(self, current_heads: int, current_ffn: int, 
                           seq_len: int = 256) -> Tuple[float, float]:
        """
        Estimate partial derivatives of latency w.r.t. heads and FFN channels
        
        Returns:
            Tuple of (∂L/∂heads, ∂L/∂ffn)
        """
        # Finite difference approximation
        delta = 1
        
        current_latency = self.estimate_latency(current_heads, current_ffn, seq_len)
        
        # Gradient w.r.t. heads
        head_plus = self.estimate_latency(current_heads + delta, current_ffn, seq_len)
        head_grad = (head_plus - current_latency) / delta
        
        # Gradient w.r.t. FFN
        ffn_plus = self.estimate_latency(current_heads, current_ffn + delta, seq_len)
        ffn_grad = (ffn_plus - current_latency) / delta
        
        return head_grad, ffn_grad
    
    def print_lut_stats(self):
        """Print statistics about the loaded LUT"""
        if not self.lut:
            print("No LUT loaded")
            return
        
        print(f"Latency LUT Statistics:")
        print(f"Total configurations: {len(self.lut)}")
        
        # Extract statistics
        latencies = [v[0] for v in self.lut.values()]
        heads = set(k[3] for k in self.lut.keys())
        ffns = set(k[4] for k in self.lut.keys())
        seq_lens = set(k[1] for k in self.lut.keys())
        
        print(f"Latency range: {min(latencies):.2f} - {max(latencies):.2f} ms")
        print(f"Head counts: {min(heads)} - {max(heads)}")
        print(f"FFN sizes: {min(ffns)} - {max(ffns)}")
        print(f"Sequence lengths: {sorted(seq_lens)}")
    
    def get_latency_statistics(self, config_list: List[Tuple[int, int]]) -> Dict:
        """Get latency statistics for a list of configurations"""
        stats = {
            'mean': 0,
            'std': 0,
            'min': float('inf'),
            'max': 0,
            'latencies': []
        }
        
        for heads, ffn in config_list:
            latency = self.estimate_latency(heads, ffn)
            stats['latencies'].append(latency)
            stats['min'] = min(stats['min'], latency)
            stats['max'] = max(stats['max'], latency)
        
        if stats['latencies']:
            stats['mean'] = np.mean(stats['latencies'])
            stats['std'] = np.std(stats['latencies'])
        
        return stats