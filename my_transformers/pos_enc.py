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


class RotaryEmbedding(nn.Module):
    """
    Positional encoding to add to embeddings introduced in "Rotary Positional Embedding".

    """

    def __init__(self, d_model: int, seq_len: int, dropout: float) -> None:
        super().__init__()
        self.d_model = d_model
        self.seq_len = seq_len
        thetas = torch.arange(0, d_model // 2, dtype=torch.float)  # (d_model // 2)
        thetas = 10000 ** (-2 * (thetas - 1) / d_model)  # (d_model // 2)
        # pos must be like (mt1, mt1, mt2, mt2, ..., mtd/2, mtd/2) (t = theta)
        # where i = 0, 1, ..., d/2
        # so repeat interleave thetas to go from
        # (mt1, mt2, ..., mtd/2) to (mt1, mt1, mt2, mt2, ..., mtd/2, mtd/2)
        thetas = thetas.repeat_interleave(2)

        freqs = torch.outer(
            torch.arange(0, seq_len, dtype=torch.float), thetas
        )  # (seq_len, d_model)

        self.register_buffer("freqs", freqs)

    @staticmethod
    def _apply_rotary_emb(x: Tensor, freqs: Tensor) -> Tensor:
        # x is of shape (batch_size, seq_len, d_model)
        # freqs is of shape at least (seq_len)
        # so we want only to add the positional encoding to the seq_len dimension
        batch_size, seq_len, d_model = x.size()

        # we want (-x1, x0, -x3, x2, ..., -xd-1, xd-2) as in the paper (here indexed from 0 instead of 1)
        # from (x0, x1, ..., xd-1)
        x_even, x_odd = x.view(batch_size, seq_len, d_model // 2, 2).unbind(
            dim=-1
        )  # (batch_size, seq_len, d_model // 2)
        # x_even is (x0, x2, ..., xd-2)
        # x_odd is (x1, x3, ..., xd-1)
        x_rot = torch.stack((-x_odd, x_even), dim=-1).reshape(
            batch_size, seq_len, d_model
        )  # (batch_size, seq_len, d_model)

        freq_cos = torch.cos(freqs).reshape(1, -1, d_model)[
            :, :seq_len, :
        ]  # handle case where seq_len < self.seq_len
        freq_sin = torch.sin(freqs).reshape(1, -1, d_model)[
            :, :seq_len, :
        ]  # shape (1, seq_len, d_model)

        return x * freq_cos + x_rot * freq_sin

    def forward(self, x: Tensor) -> Tensor:
        return self._apply_rotary_emb(x, self.freqs)


class RotaryEmbeddingWithComplexs(nn.Module):
    """
    Positional encoding to add to embeddings introduced in "Rotary Positional Embedding".
    Uses the complex number interpretation of the rotation.

    """

    def __init__(self, d_model: int, seq_len: int, dropout: float) -> None:
        super().__init__()
        self.d_model = d_model
        self.seq_len = seq_len
        thetas = torch.arange(0, d_model // 2, dtype=torch.float)  # (d_model // 2)
        thetas = 10000 ** (-2 * (thetas - 1) / d_model)  # (d_model // 2)
        # pos must be like (mt1, mt1, mt2, mt2, ..., mtd/2, mtd/2) (t = theta)
        # where i = 0, 1, ..., d/2
        # so repeat interleave thetas to go from
        # (mt1, mt2, ..., mtd/2) to (mt1, mt1, mt2, mt2, ..., mtd/2, mtd/2)
        thetas = thetas.repeat_interleave(2)  # (d_model // 2)

        freqs = torch.outer(
            torch.arange(0, seq_len, dtype=torch.float), thetas
        )  # (seq_len, d_model)

        freqs = torch.view_as_complex(
            freqs.reshape(seq_len, d_model // 2, 2)  # (seq_len, d_model // 2, 2)
        )  # (seq_len, d_model // 2)

        freqs = torch.exp(freqs)  # now freqs is of the for
        # [[exp(i * m0 * theta0), exp(i * m0 * theta1), ..., exp(i * m0 * theta2), exp(i * m0 * theta3), ...]
        # [exp(i * m1 * theta0), exp(i * m1 * theta1), ..., exp(i * m1 * theta2), exp(i * m1 * theta3), ...]]
        # where m0, m1, ... are the positions and theta0, theta1, ... are the thetas
        # remember that exp(i * theta) = cos(theta) + i * sin(theta), and multiplying two complex numbers
        # is done by multiplying the magnitudes and adding the angles
        # so, for example x = a + bi and take entry 0, 0 of freqs, we have
        # x * freqs[0, 0] = (a + bi) * (cos(m0 * theta0) + i * sin(m0 * theta0))
        # = a * cos + a * i * sin + b * i * cos - b * sin
        # this represents a rotation of x by m0 * theta0
        # now, as the paper says, we subdivide the d-dimentional space into d/2 2-dimensional subspaces (d = d_model)
        # and individually rotate each subspace by the corresponding theta (remember we have d/2 thetas for d/2 subspaces)
        # and also take into account the position m

        self.register_buffer("freqs", freqs)

    @staticmethod
    def _apply_rotary_emb(x: Tensor, freqs: Tensor) -> Tensor:
        # x is of shape (batch_size, seq_len, d_model)
        # freqs is of shape at least (seq_len)
        # so we want only to add the positional encoding to the seq_len dimension
        batch_size, seq_len, d_model = x.size()

        xcomp = torch.view_as_complex(x.view(batch_size, seq_len, d_model // 2, 2))

        x_rot = xcomp * freqs.unsqueeze(0)[
            :, :seq_len, :
        ]  # handle case where seq_len < self.seq_len

        return torch.view_as_real(x_rot).reshape(batch_size, seq_len, d_model)

    def forward(self, x: Tensor) -> Tensor:
        return self._apply_rotary_emb(x, self.freqs)


if __name__ == "__main__":
    d_model = 512
    seq_len = 100
    dropout = 0.1
    pos_enc = SinusoidalPositionalEncoding(d_model, seq_len, dropout)
    x = torch.randn(32, seq_len, d_model)
    assert pos_enc(x).shape == torch.Size([32, 100, 512])
    pos_enc = RotaryEmbedding(d_model, seq_len, dropout)
    x = torch.randn(32, seq_len, d_model)
    assert pos_enc(x).shape == torch.Size([32, 100, 512])
    pos_enc = RotaryEmbeddingWithComplexs(d_model, seq_len, dropout)
    encoded = pos_enc(x)
    # both rotary embeddings should be the same
    torch.testing.assert_close(
        encoded, pos_enc(x), rtol=1e-5, atol=1e-5, equal_nan=True
    )  # since we are using randn some nans may appear
