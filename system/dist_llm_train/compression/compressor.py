"""Gradient compression module for efficient WAN communication.

Implements multiple compression strategies to reduce bandwidth requirements
for distributed LLM training over wide-area networks.
"""

import torch
from typing import Dict, Tuple, Any
import numpy as np


class GradientCompressor:
    """
    Compresses gradients to reduce network transmission overhead.

    Supported methods:
    - topk: Keep only top-k% largest gradients by magnitude (sparsification)
    - quantize: 8-bit quantization with min-max scaling
    - fp16: Half-precision floating point
    - none: No compression (passthrough)
    """

    def __init__(self, method: str = 'topk', compression_ratio: float = 0.01):
        """
        Initialize gradient compressor.

        Args:
            method: Compression method ('topk', 'quantize', 'fp16', 'none')
            compression_ratio: Compression ratio (0.01 = 1% for topk, unused for others)
        """
        if method not in ['topk', 'quantize', 'fp16', 'none']:
            raise ValueError(f"Unknown compression method: {method}")

        self.method = method
        self.compression_ratio = compression_ratio

    def compress(self, gradients: Dict[str, torch.Tensor]) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        """
        Compress gradients.

        Args:
            gradients: Dictionary of gradient tensors

        Returns:
            Tuple of (compressed_gradients, metadata)
            - compressed_gradients: Serializable compressed data
            - metadata: Information needed for decompression
        """
        if self.method == 'none':
            return self._passthrough(gradients)
        elif self.method == 'topk':
            return self._topk_compress(gradients)
        elif self.method == 'quantize':
            return self._quantize_8bit(gradients)
        elif self.method == 'fp16':
            return self._fp16_compress(gradients)
        else:
            raise ValueError(f"Unknown compression method: {self.method}")

    def decompress(self, compressed: Dict[str, Any], metadata: Dict[str, Any]) -> Dict[str, torch.Tensor]:
        """
        Decompress gradients.

        Args:
            compressed: Compressed gradient data
            metadata: Metadata from compression

        Returns:
            Dictionary of decompressed gradient tensors
        """
        method = metadata.get('method', self.method)

        if method == 'none':
            return {k: torch.tensor(v) for k, v in compressed.items()}
        elif method == 'topk':
            return self._topk_decompress(compressed, metadata)
        elif method == 'quantize':
            return self._quantize_8bit_decompress(compressed, metadata)
        elif method == 'fp16':
            return self._fp16_decompress(compressed, metadata)
        else:
            raise ValueError(f"Unknown compression method: {method}")

    def _passthrough(self, gradients: Dict[str, torch.Tensor]) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        """No compression - convert to lists for serialization."""
        compressed = {k: v.cpu().tolist() for k, v in gradients.items()}
        metadata = {'method': 'none'}
        return compressed, metadata

    def _topk_compress(self, gradients: Dict[str, torch.Tensor]) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        """
        Top-K sparsification: Keep only the largest k% of gradients by magnitude.

        This dramatically reduces bandwidth (e.g., 1% = 100x compression) with
        minimal impact on convergence for many models.
        """
        compressed = {}
        metadata = {'method': 'topk', 'shapes': {}}

        for name, tensor in gradients.items():
            flat = tensor.flatten().cpu()
            k = max(1, int(flat.numel() * self.compression_ratio))

            # Get top-k by absolute value
            topk_vals, topk_idx = torch.topk(flat.abs(), k)
            # Get actual values (with sign)
            actual_vals = flat[topk_idx]

            compressed[name] = {
                'values': actual_vals.tolist(),
                'indices': topk_idx.tolist(),
                'numel': flat.numel()
            }
            metadata['shapes'][name] = list(tensor.shape)

        return compressed, metadata

    def _topk_decompress(self, compressed: Dict[str, Any], metadata: Dict[str, Any]) -> Dict[str, torch.Tensor]:
        """Decompress top-k sparse gradients."""
        decompressed = {}
        shapes = metadata['shapes']

        for name, data in compressed.items():
            numel = data['numel']
            # Create zero tensor
            flat = torch.zeros(numel)
            # Fill in sparse values
            indices = torch.tensor(data['indices'], dtype=torch.long)
            values = torch.tensor(data['values'])
            flat[indices] = values
            # Reshape to original
            decompressed[name] = flat.reshape(shapes[name])

        return decompressed

    def _quantize_8bit(self, gradients: Dict[str, torch.Tensor]) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        """
        8-bit quantization with min-max scaling.

        Reduces bandwidth by 4x (FP32 -> INT8) with minimal accuracy loss.
        """
        compressed = {}
        metadata = {'method': 'quantize', 'scales': {}}

        for name, tensor in gradients.items():
            flat = tensor.flatten().cpu()

            # Compute scale factors
            min_val = flat.min().item()
            max_val = flat.max().item()

            if max_val == min_val:
                # Handle constant gradients
                compressed[name] = {
                    'data': [128] * flat.numel(),  # Mid-range value
                    'shape': list(tensor.shape)
                }
                metadata['scales'][name] = {'min': min_val, 'max': max_val}
                continue

            # Quantize to 8-bit integers with rounding to reduce error
            scale = (max_val - min_val) / 255.0
            quantized = torch.round((flat - min_val) / scale).clamp(0, 255).to(torch.uint8)

            compressed[name] = {
                'data': quantized.tolist(),
                'shape': list(tensor.shape)
            }
            metadata['scales'][name] = {'min': min_val, 'max': max_val}

        return compressed, metadata

    def _quantize_8bit_decompress(self, compressed: Dict[str, Any], metadata: Dict[str, Any]) -> Dict[str, torch.Tensor]:
        """Decompress 8-bit quantized gradients."""
        decompressed = {}
        scales = metadata['scales']

        for name, data in compressed.items():
            quantized = torch.tensor(data['data'], dtype=torch.uint8)
            min_val = scales[name]['min']
            max_val = scales[name]['max']

            if max_val == min_val:
                # Constant gradient
                decompressed[name] = torch.full(data['shape'], min_val)
            else:
                scale = (max_val - min_val) / 255.0
                dequantized = quantized.float() * scale + min_val
                decompressed[name] = dequantized.reshape(data['shape'])

        return decompressed

    def _fp16_compress(self, gradients: Dict[str, torch.Tensor]) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        """
        FP16 (half-precision) compression.

        Reduces bandwidth by 2x with negligible accuracy loss for most models.
        """
        compressed = {}
        metadata = {'method': 'fp16', 'shapes': {}}

        for name, tensor in gradients.items():
            # Convert to FP16
            fp16_tensor = tensor.cpu().half()
            # Store as bytes for efficient serialization
            compressed[name] = {
                'bytes': fp16_tensor.numpy().tobytes(),
                'dtype': str(fp16_tensor.dtype)
            }
            metadata['shapes'][name] = list(tensor.shape)

        return compressed, metadata

    def _fp16_decompress(self, compressed: Dict[str, Any], metadata: Dict[str, Any]) -> Dict[str, torch.Tensor]:
        """Decompress FP16 gradients."""
        decompressed = {}
        shapes = metadata['shapes']

        for name, data in compressed.items():
            # Reconstruct from bytes
            np_array = np.frombuffer(data['bytes'], dtype=np.float16)
            tensor = torch.from_numpy(np_array).float()
            decompressed[name] = tensor.reshape(shapes[name])

        return decompressed

    def get_compression_stats(self, original: Dict[str, torch.Tensor],
                             compressed: Dict[str, Any]) -> Dict[str, float]:
        """
        Calculate compression statistics.

        Args:
            original: Original gradient tensors
            compressed: Compressed gradient data

        Returns:
            Dictionary with compression stats
        """
        import sys

        # Estimate original size (FP32 bytes)
        original_bytes = sum(t.numel() * 4 for t in original.values())

        # Estimate compressed size
        if self.method == 'topk':
            compressed_bytes = 0
            for d in compressed.values():
                # Indices are compressible; assume 2-byte storage for typical tensor sizes (<16M elements)
                index_bytes = 2 if d.get('numel', 0) < 2 ** 24 else 4
                compressed_bytes += len(d['values']) * 4 + len(d['indices']) * index_bytes
        elif self.method == 'quantize':
            compressed_bytes = sum(
                len(d['data'])  # 8-bit integers
                for d in compressed.values()
            )
        elif self.method == 'fp16':
            compressed_bytes = sum(
                len(d['bytes'])  # FP16 bytes
                for d in compressed.values()
            )
        else:
            compressed_bytes = original_bytes

        compression_ratio = original_bytes / max(1, compressed_bytes)

        return {
            'original_bytes': original_bytes,
            'compressed_bytes': compressed_bytes,
            'compression_ratio': compression_ratio,
            'bandwidth_reduction': (1 - compressed_bytes / original_bytes) * 100
        }
