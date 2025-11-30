"""Tests for GradientCompressor."""

import unittest
import torch
import numpy as np
from dist_llm_train.compression.compressor import GradientCompressor


class TestGradientCompressor(unittest.TestCase):
    """Test cases for gradient compression."""

    def setUp(self):
        """Set up test fixtures."""
        # Create sample gradients
        torch.manual_seed(42)
        self.gradients = {
            'layer1.weight': torch.randn(100, 50),
            'layer1.bias': torch.randn(50),
            'layer2.weight': torch.randn(50, 10),
        }

    def test_topk_compression(self):
        """Test top-k sparsification compression."""
        compressor = GradientCompressor(method='topk', compression_ratio=0.1)

        compressed, metadata = compressor.compress(self.gradients)

        # Check metadata
        self.assertEqual(metadata['method'], 'topk')
        self.assertIn('shapes', metadata)
        self.assertEqual(len(metadata['shapes']), 3)

        # Check compression (should have ~10% of values)
        for name, data in compressed.items():
            original_size = self.gradients[name].numel()
            compressed_size = len(data['values'])
            ratio = compressed_size / original_size
            # Allow some tolerance
            self.assertLess(ratio, 0.15)
            self.assertGreater(ratio, 0.05)

    def test_topk_decompression(self):
        """Test that top-k decompression preserves top values."""
        compressor = GradientCompressor(method='topk', compression_ratio=0.1)

        compressed, metadata = compressor.compress(self.gradients)
        decompressed = compressor.decompress(compressed, metadata)

        # Check shapes match
        for name in self.gradients.keys():
            self.assertEqual(decompressed[name].shape, self.gradients[name].shape)

        # Check that top values are preserved
        for name in self.gradients.keys():
            original = self.gradients[name].flatten()
            decompressed_flat = decompressed[name].flatten()

            # Get top-k from original
            k = len(compressed[name]['values'])
            topk_vals, topk_idx = torch.topk(original.abs(), k)

            # Check these values are preserved in decompressed
            for idx in topk_idx:
                self.assertAlmostEqual(
                    decompressed_flat[idx].item(),
                    original[idx].item(),
                    places=5
                )

    def test_quantize_compression(self):
        """Test 8-bit quantization compression."""
        compressor = GradientCompressor(method='quantize')

        compressed, metadata = compressor.compress(self.gradients)

        # Check metadata
        self.assertEqual(metadata['method'], 'quantize')
        self.assertIn('scales', metadata)

        # Check that data is quantized to 8-bit
        for name, data in compressed.items():
            self.assertEqual(len(data['data']), self.gradients[name].numel())
            # Values should be in 0-255 range (represented as ints in list)
            for val in data['data']:
                self.assertGreaterEqual(val, 0)
                self.assertLessEqual(val, 255)

    def test_quantize_decompression(self):
        """Test 8-bit quantization decompression."""
        compressor = GradientCompressor(method='quantize')

        compressed, metadata = compressor.compress(self.gradients)
        decompressed = compressor.decompress(compressed, metadata)

        # Check shapes match
        for name in self.gradients.keys():
            self.assertEqual(decompressed[name].shape, self.gradients[name].shape)

        # Check that values are approximately preserved
        for name in self.gradients.keys():
            original = self.gradients[name]
            reconstructed = decompressed[name]

            # Calculate relative error
            error = torch.abs(original - reconstructed).mean() / (torch.abs(original).mean() + 1e-8)
            # Quantization error should be small (< 1%)
            self.assertLess(error.item(), 0.01)

    def test_fp16_compression(self):
        """Test FP16 compression."""
        compressor = GradientCompressor(method='fp16')

        compressed, metadata = compressor.compress(self.gradients)

        # Check metadata
        self.assertEqual(metadata['method'], 'fp16')
        self.assertIn('shapes', metadata)

        # Check that data is stored as bytes
        for name, data in compressed.items():
            self.assertIn('bytes', data)
            # FP16 uses 2 bytes per element
            expected_bytes = self.gradients[name].numel() * 2
            self.assertEqual(len(data['bytes']), expected_bytes)

    def test_fp16_decompression(self):
        """Test FP16 decompression."""
        compressor = GradientCompressor(method='fp16')

        compressed, metadata = compressor.compress(self.gradients)
        decompressed = compressor.decompress(compressed, metadata)

        # Check shapes match
        for name in self.gradients.keys():
            self.assertEqual(decompressed[name].shape, self.gradients[name].shape)

        # Check that values are approximately preserved (FP16 has ~0.1% precision)
        for name in self.gradients.keys():
            original = self.gradients[name]
            reconstructed = decompressed[name]

            # Values should be very close
            error = torch.abs(original - reconstructed).mean()
            self.assertLess(error.item(), 1e-3)

    def test_passthrough_compression(self):
        """Test no compression (passthrough)."""
        compressor = GradientCompressor(method='none')

        compressed, metadata = compressor.compress(self.gradients)

        # Check metadata
        self.assertEqual(metadata['method'], 'none')

        # Data should be converted to lists but not compressed
        for name in self.gradients.keys():
            self.assertIn(name, compressed)

        # Decompression should restore original
        decompressed = compressor.decompress(compressed, metadata)

        for name in self.gradients.keys():
            self.assertTrue(torch.allclose(
                self.gradients[name],
                decompressed[name],
                rtol=1e-5
            ))

    def test_compression_stats(self):
        """Test compression statistics calculation."""
        compressor = GradientCompressor(method='topk', compression_ratio=0.01)

        compressed, metadata = compressor.compress(self.gradients)
        stats = compressor.get_compression_stats(self.gradients, compressed)

        # Check stats structure
        self.assertIn('original_bytes', stats)
        self.assertIn('compressed_bytes', stats)
        self.assertIn('compression_ratio', stats)
        self.assertIn('bandwidth_reduction', stats)

        # Check that compression ratio is reasonable
        self.assertGreater(stats['compression_ratio'], 1.0)
        self.assertGreater(stats['bandwidth_reduction'], 0.0)
        self.assertLess(stats['bandwidth_reduction'], 100.0)

    def test_topk_extreme_sparsity(self):
        """Test top-k with very high sparsity (1%)."""
        compressor = GradientCompressor(method='topk', compression_ratio=0.01)

        compressed, metadata = compressor.compress(self.gradients)
        stats = compressor.get_compression_stats(self.gradients, compressed)

        # Should achieve high compression
        self.assertGreater(stats['compression_ratio'], 50)

    def test_constant_gradients(self):
        """Test quantization with constant gradients."""
        constant_grads = {
            'param1': torch.ones(100) * 5.0
        }

        compressor = GradientCompressor(method='quantize')
        compressed, metadata = compressor.compress(constant_grads)
        decompressed = compressor.decompress(compressed, metadata)

        # Should preserve constant value
        self.assertTrue(torch.allclose(
            constant_grads['param1'],
            decompressed['param1'],
            rtol=1e-5
        ))

    def test_zero_gradients(self):
        """Test compression with zero gradients."""
        zero_grads = {
            'param1': torch.zeros(100)
        }

        # Test all methods
        for method in ['topk', 'quantize', 'fp16']:
            compressor = GradientCompressor(method=method)
            compressed, metadata = compressor.compress(zero_grads)
            decompressed = compressor.decompress(compressed, metadata)

            self.assertTrue(torch.allclose(
                zero_grads['param1'],
                decompressed['param1'],
                atol=1e-6
            ))

    def test_large_gradients(self):
        """Test compression with large gradient tensors."""
        large_grads = {
            'huge_param': torch.randn(1000, 1000)  # 1M parameters
        }

        compressor = GradientCompressor(method='topk', compression_ratio=0.01)
        compressed, metadata = compressor.compress(large_grads)
        stats = compressor.get_compression_stats(large_grads, compressed)

        # Should achieve significant compression
        self.assertGreater(stats['compression_ratio'], 50)

        # Should still decompress correctly
        decompressed = compressor.decompress(compressed, metadata)
        self.assertEqual(decompressed['huge_param'].shape, (1000, 1000))

    def test_invalid_method(self):
        """Test that invalid compression method raises error."""
        with self.assertRaises(ValueError):
            GradientCompressor(method='invalid_method')

    def test_topk_minimum_k(self):
        """Test that top-k always keeps at least 1 element."""
        small_grads = {
            'param1': torch.randn(10)
        }

        # Very aggressive compression ratio
        compressor = GradientCompressor(method='topk', compression_ratio=0.001)
        compressed, metadata = compressor.compress(small_grads)

        # Should keep at least 1 element
        self.assertGreaterEqual(len(compressed['param1']['values']), 1)


if __name__ == '__main__':
    unittest.main()
