import unittest
import torch
from my_transformers import PositionWiseFeedForward


class TestPositionWiseFeedForward(unittest.TestCase):
    def test_forward(self):
        d_model = 512
        inner_dim = 2048
        ff = PositionWiseFeedForward(d_model, inner_dim)
        x = torch.randn(32, 100, d_model)
        out = ff(x)
        self.assertEqual(out.size(), (32, 100, d_model))


if __name__ == "__main__":
    unittest.main()
