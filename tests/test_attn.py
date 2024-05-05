import unittest
import torch
from my_transformers import MultiHeadAttention, MultiQueryAttention, GroupQueryAttention


class TestMultiHeadAttention(unittest.TestCase):
    def test_forward(self):
        d_model = 512
        num_heads = 8
        d_k = d_v = d_model // num_heads
        attn = MultiHeadAttention(num_heads, d_model, d_k, d_v, causal=False)
        x = torch.randn(32, 100, d_model)
        out = attn(x, x, x)
        self.assertEqual(out.size(), (32, 100, d_model))


class TestMultiQueryAttention(unittest.TestCase):
    def test_forward(self):
        d_model = 512
        num_heads = 8
        d_k = d_v = d_model // num_heads
        attn = MultiQueryAttention(num_heads, d_model, d_k, d_v, causal=False)
        x = torch.randn(32, 100, d_model)
        out = attn(x, x, x)
        self.assertEqual(out.size(), (32, 100, d_model))


class TestGroupQueryAttention(unittest.TestCase):
    def test_forward(self):
        d_model = 512
        q_heads = 16
        d_k = d_v = d_model // q_heads
        kv_heads = 8
        attn = GroupQueryAttention(q_heads, kv_heads, d_model, d_k, d_v, causal=False)
        x = torch.randn(32, 100, d_model)
        out = attn(x, x, x)
        self.assertEqual(out.size(), (32, 100, d_model))


if __name__ == "__main__":
    unittest.main()
