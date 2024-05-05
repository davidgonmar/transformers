import unittest
import torch
from my_transformers import SinusoidalPositionalEncoding


class TestSinusoidalPositionalEncoding(unittest.TestCase):
    def test_forward(self):
        d_model = 512
        seq_len = 100
        dropout = 0.1
        pos_enc = SinusoidalPositionalEncoding(d_model, seq_len, dropout)
        x = torch.randn(32, seq_len, d_model)
        out = pos_enc(x)
        self.assertEqual(out.size(), (32, seq_len, d_model))


if __name__ == "__main__":
    unittest.main()
