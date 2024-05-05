import torch
from torch import nn, Tensor


class SinusoidalPositionalEncoding(nn.Module):
    """
    Positional encoding to add to embeddings introduced in "Attention is All You Need".

    """

    def __init__(self, d_model: int, seq_len: int, dropout: float) -> None:
        super().__init__()
        self.d_model = d_model
        self.seq_len = seq_len
        pos_encoding = torch.zeros(seq_len, d_model)  # (seq_len, d_model)
        pos = torch.arange(0, seq_len, dtype=torch.float).unsqueeze(1)  # (seq_len, 1)
        div_term = torch.pow(10000, torch.arange(0, d_model, 2) / d_model)

        pos_encoding[:, 0::2] = torch.sin(pos / div_term)
        pos_encoding[:, 1::2] = torch.cos(pos / div_term)
        self.register_buffer("pos_encoding", pos_encoding)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: Tensor) -> Tensor:
        # x is of shape (batch_size, seq_len, d_model)
        # so we want only to add the positional encoding to the seq_len dimension
        cropped_pos_encoding = self.pos_encoding[: x.size(1), :]  # (seq_len, d_model)
        return self.dropout(x + cropped_pos_encoding)
