import torch
from torch import nn, Tensor
import torch.nn.functional as F


class MultiHeadAttention(nn.Module):
    def __init__(
        self,
        num_heads: int,
        d_model: int,
        d_k: int,
        d_v: int,
        causal: bool,
        save_attn_scores_to_visualize=True,
    ):
        super().__init__()
        self.num_heads = num_heads
        self.d_model = d_model
        self.causal = causal
        self.d_k = d_k
        self.d_v = d_v
        self.save_attn_scores_to_visualize = save_attn_scores_to_visualize
        self.attn_scores = None

        assert d_model % num_heads == 0, "d_model should be divisible by num_heads"

        self.W_q = nn.Linear(d_model, num_heads * d_k, bias=False)
        self.W_k = nn.Linear(d_model, num_heads * d_k, bias=False)
        self.W_v = nn.Linear(d_model, num_heads * d_v, bias=False)
        self.W_o = nn.Linear(num_heads * d_v, d_model, bias=False)

    def forward(self, Q: Tensor, K: Tensor, V: Tensor, pad_attn_mask: Tensor = None):
        """
        len_q, len_k, len_v are the lengths of the sequences for Q, K, V
        Args:
            Q: Query matrix with shape (batch_size, len_q, d_model)
            K: Key matrix with shape (batch_size, len_k, d_model)
            V: Value matrix with shape (batch_size, len_v, d_model)
        """

        len_q, len_k, len_v = Q.size(1), K.size(1), V.size(1)
        assert len_k == len_v, "len_k and len_v must be equal, got {} and {}".format(
            len_k, len_v
        )
        batch_size = Q.size(0)  # should be equal to K.size(0) and V.size(0)

        # Project query, key and value into d_k * num_heads and d_v * num_heads
        # We transpose them so the 'inner (right-most) matrices' are of shape
        # (len_x, d_x), so shape is (batch_size, num_heads, len_x, d_x)
        Q = (
            self.W_q(Q)
            .view(batch_size, len_q, self.num_heads, self.d_k)
            .transpose(1, 2)
        )  # shape (batch_size, num_heads, len_q, d_k)
        K = (
            self.W_k(K)
            .view(batch_size, len_k, self.num_heads, self.d_k)
            .transpose(1, 2)
        )  # shape (batch_size, num_heads, len_k, d_k)
        V = (
            self.W_v(V)
            .view(batch_size, len_v, self.num_heads, self.d_v)
            .transpose(1, 2)
        )  # shape (batch_size, num_heads, len_v, d_v)

        out = self._scaled_dot_product_attention(
            Q, K, V, pad_attn_mask
        )  # (n, num_heads, seq_len, d_v)

        # now, we need to multiply by the linear with input size num_heads * d_v
        out = out.transpose(1, 2)  # shape (batch_size, len_q, num_heads, d_v)

        assert out.size(2) == self.num_heads
        assert out.size(3) == self.d_v
        assert out.size(1) == len_q
        assert out.size(0) == batch_size

        # We then merge the heads together to get (batch_size, len_q, num_heads * d_v)
        # In the paper, num_heads * d_v = d_model
        # Dont use view because memory layout is not compatible
        out = out.reshape(batch_size, len_q, self.num_heads * self.d_v)
        return self.W_o(out)

    def _scaled_dot_product_attention(
        self, Q: Tensor, K: Tensor, V: Tensor, pad_attn_mask
    ) -> Tensor:
        """
        This is equivalent to separately computing the attention for each head.
        Args:
            Q: Query matrix with shape (batch_size, num_heads, len_q, d_k)
            K: Key matrix with shape (batch_size, num_heads, len_k, d_k)
            V: Value matrix with shape (batch_size, num_heads, len_v = len_k, d_v)
        """

        x = torch.einsum("b h q d, b h k d -> b h q k", Q, K) / self.d_k**0.5
        mask = pad_attn_mask
        if self.causal:
            # print("before masking: ", x)
            # Apply masking, will be broadcasted to shape (batch_size, num_heads, len_q, len_k)
            # Basically, create a matrix with 1s below and in the diagonal, 0s above
            # Then, mask where mask == 0 with -inf
            # So basically we set the values above the diagonal to -inf
            # When softmax is applied, these values will become 0
            causal_mask = (
                torch.tril(torch.ones(x.size(-2), x.size(-1)), diagonal=0)
                .view(1, 1, x.size(-2), x.size(-1))
                .to(x.device)  # (1, 1, len_q, len_k)
            )
            mask = (
                mask.bool() & causal_mask.bool()
                if mask is not None
                else causal_mask.bool()
            )
            x = x.masked_fill(mask == 0, -1e9)  # (batch_size, num_heads, len_q, len_k)

        else:
            if mask is not None:
                x = x.masked_fill(
                    mask == 0, -1e9
                )  # (batch_size, num_heads, len_q, len_k)
        x = F.softmax(x, dim=-1)  # (batch_size, num_heads, len_q, len_k)
        if self.save_attn_scores_to_visualize:
            self.attn_scores = x
        return x @ V


class MultiQueryAttention(nn.Module):
    def __init__(
        self,
        num_heads: int,
        d_model: int,
        d_k: int,
        d_v: int,
        causal: bool,
        save_attn_scores_to_visualize=True,
    ):
        super().__init__()
        self.num_heads = num_heads
        self.d_model = d_model
        self.causal = causal
        self.d_k = d_k
        self.d_v = d_v
        self.save_attn_scores_to_visualize = save_attn_scores_to_visualize
        self.attn_scores = None

        assert d_model % num_heads == 0, "d_model should be divisible by num_heads"

        self.W_q = nn.Linear(d_model, d_k * num_heads, bias=False)
        self.W_k = nn.Linear(d_model, d_k, bias=False)
        self.W_v = nn.Linear(d_model, d_v, bias=False)
        self.W_o = nn.Linear(d_v * num_heads, d_model, bias=False)

    def forward(self, Q: Tensor, K: Tensor, V: Tensor, pad_attn_mask: Tensor = None):
        """
        len_q, len_k, len_v are the lengths of the sequences for Q, K, V
        Args:
            Q: Query matrix with shape (batch_size, len_q, d_model)
            K: Key matrix with shape (batch_size, len_k, d_model)
            V: Value matrix with shape (batch_size, len_v, d_model)
        """

        len_q, len_k, len_v = Q.size(1), K.size(1), V.size(1)
        assert len_k == len_v, "len_k and len_v must be equal, got {} and {}".format(
            len_k, len_v
        )
        batch_size = Q.size(0)  # should be equal to K.size(0) and V.size(0)

        # Project query, key and value into d_k * num_heads and d_v * num_heads
        # We transpose them so the 'inner (right-most) matrices' are of shape
        # (len_x, d_x), so shape is (batch_size, num_heads, len_x, d_x)
        Q = (
            self.W_q(Q)
            .view(batch_size, len_q, self.num_heads, self.d_k)
            .transpose(1, 2)
        )  # shape (batch_size, num_heads, len_q, d_k)
        K = self.W_k(K).view(
            batch_size, len_k, self.d_k
        )  # shape (batch_size, len_k, d_k)
        V = self.W_v(V).view(
            batch_size, len_v, self.d_v
        )  # shape (batch_size, len_v, d_v)

        out = self._scaled_dot_product_attention(
            Q, K, V, pad_attn_mask
        )  # (n, num_heads, seq_len, d_v)

        # now, we need to multiply by the linear with input size num_heads * d_v
        out = out.transpose(1, 2)  # shape (batch_size, len_q, num_heads, d_v)

        assert out.size(2) == self.num_heads
        assert out.size(3) == self.d_v
        assert out.size(1) == len_q
        assert out.size(0) == batch_size

        # We then merge the heads together to get (batch_size, len_q, num_heads * d_v)
        # In the paper, num_heads * d_v = d_model
        # Dont use view because memory layout is not compatible
        out = out.reshape(batch_size, len_q, self.num_heads * self.d_v)
        return self.W_o(out)

    def _scaled_dot_product_attention(
        self, Q: Tensor, K: Tensor, V: Tensor, pad_attn_mask
    ) -> Tensor:
        """
        This is equivalent to separately computing the attention for each head.
        Args:
            Q: Query matrix with shape (batch_size, num_heads, len_q, d_k)
            K: Key matrix with shape (batch_size, num_heads, len_k, d_k)
            V: Value matrix with shape (batch_size, num_heads, len_v = len_k, d_v)
        """
        batch_size = Q.size(0)
        len_k = K.size(1)
        x = (
            Q
            @ K.reshape(batch_size, 1, len_k, self.d_k).transpose(
                -2, -1
            )  # allow broadcasting
        ) / self.d_k**0.5  # (batch_size, num_heads, len_q, len_k)

        mask = pad_attn_mask
        if self.causal:
            # Apply masking, will be broadcasted to shape (batch_size, num_heads, len_q, len_k)
            # Basically, create a matrix with 1s below and in the diagonal, 0s above
            # Then, mask where mask == 0 with -inf
            # So basically we set the values above the diagonal to -inf
            # When softmax is applied, these values will become 0
            causal_mask = (
                torch.tril(torch.ones(x.size(-2), x.size(-1)), diagonal=0)
                .view(1, 1, x.size(-2), x.size(-1))
                .to(x.device)  # (1, 1, len_q, len_k)
            )
            mask = (
                mask.bool() & causal_mask.bool()
                if mask is not None
                else causal_mask.bool()
            )
            x = x.masked_fill(mask == 0, -1e9)  # (batch_size, num_heads, len_q, len_k)
        else:
            if mask is not None:
                x = x.masked_fill(
                    mask == 0, -1e9
                )  # (batch_size, num_heads, len_q, len_k)
        x = F.softmax(x, dim=-1)  # (batch_size, num_heads, len_q, len_k)
        if self.save_attn_scores_to_visualize:
            self.attn_scores = x
        return x @ V.reshape(
            batch_size, 1, len_k, self.d_v
        )  # shape (batch_size, num_heads, len_q, d_v)


class GroupQueryAttention(nn.Module):
    def __init__(
        self,
        q_heads: int,
        kv_heads: int,
        d_model: int,
        d_k: int,
        d_v: int,
        causal: bool,
        save_attn_scores_to_visualize=True,
    ):
        # d_q and d_k are the same
        super().__init__()
        self.q_heads = q_heads
        self.d_model = d_model
        self.causal = causal
        self.d_k = d_k
        self.d_v = d_v
        self.kv_heads = kv_heads

        self.save_attn_scores_to_visualize = save_attn_scores_to_visualize
        self.attn_scores = None

        assert d_model % q_heads == 0, "d_model should be divisible by q_heads"
        assert d_model % kv_heads == 0, "d_model should be divisible by kv_heads"
        assert q_heads % kv_heads == 0, "q_heads must be divisible by kv_heads"

        self.W_q = nn.Linear(d_model, d_k * q_heads, bias=False)
        self.W_k = nn.Linear(d_model, d_k * kv_heads, bias=False)
        self.W_v = nn.Linear(d_model, d_v * kv_heads, bias=False)
        self.W_o = nn.Linear(d_v * kv_heads, d_model, bias=False)

    def forward(self, Q: Tensor, K: Tensor, V: Tensor, pad_attn_mask: Tensor = None):
        """
        len_q, len_k, len_v are the lengths of the sequences for Q, K, V
        Args:
            Q: Query matrix with shape (batch_size, len_q, d_model)
            K: Key matrix with shape (batch_size, len_k, d_model)
            V: Value matrix with shape (batch_size, len_v, d_model)
        """

        len_q, len_k, len_v = Q.size(1), K.size(1), V.size(1)
        assert len_k == len_v, "len_k and len_v must be equal, got {} and {}".format(
            len_k, len_v
        )
        batch_size = Q.size(0)  # should be equal to K.size(0) and V.size(0)

        # Project query, key and value into d_k * num_heads and d_v * num_heads
        # We transpose them so the 'inner (right-most) matrices' are of shape
        # (len_x, d_x), so shape is (batch_size, num_heads, len_x, d_x)
        Q = (
            self.W_q(Q).view(batch_size, len_q, self.q_heads, self.d_k).transpose(1, 2)
        )  # shape (batch_size, q_heads, len_q, d_k)
        K = (
            self.W_k(K).view(batch_size, len_k, self.kv_heads, self.d_k).transpose(1, 2)
        )  # shape (batch_size, kv_heads, len_k, d_k)
        V = (
            self.W_v(V).view(batch_size, len_v, self.kv_heads, self.d_v).transpose(1, 2)
        )  # shape (batch_size, kv_heads, len_v, d_v)

        out = self._scaled_dot_product_attention(Q, K, V, pad_attn_mask).transpose(
            1, 2
        )  # (n, seq_len, kv_heads, d_v)

        assert out.size(0) == batch_size
        assert out.size(1) == len_q
        assert out.size(2) == self.kv_heads
        assert out.size(3) == self.d_v

        # Merge the heads together to get (batch_size, len_q, q_heads * d_v)
        out = out.reshape(batch_size, len_q, self.kv_heads * self.d_v)
        return self.W_o(out)

    def _scaled_dot_product_attention(
        self, Q: Tensor, K: Tensor, V: Tensor, pad_attn_mask
    ) -> Tensor:
        """
        This is equivalent to separately computing the attention for each head.
        n_heads must be divisible by groups
        Args:
            Q: Query matrix with shape (batch_size, q_heads, len_q, d_k)
            K: Key matrix with shape (batch_size, kv_heads, len_k, d_k)
            V: Value matrix with shape (batch_size, kv_heads, len_v = len_k, d_v)
        """
        batch_size = Q.size(0)
        len_k = K.size(2)
        len_q = Q.size(2)
        d_k = K.size(-1)
        q_heads, kv_heads = self.q_heads, self.kv_heads
        x = (
            Q.reshape(batch_size, q_heads // kv_heads, kv_heads, len_q, d_k)
            @ K.reshape(batch_size, 1, kv_heads, len_k, self.d_k).transpose(
                -2, -1
            )  # allow broadcasting
        ) / self.d_k**0.5  # (batch_size, q_heads // kv_heads, kv_heads, len_q, len_k)
        x = x.sum(dim=1, keepdim=False)  # (batch_size, kv_heads, len_q, len_k)
        mask = pad_attn_mask
        if self.causal:
            # Apply masking, will be broadcasted to shape (batch_size, num_heads, len_q, len_k)
            # Basically, create a matrix with 1s below and in the diagonal, 0s above
            # Then, mask where mask == 0 with -inf
            # So basically we set the values above the diagonal to -inf
            # When softmax is applied, these values will become 0
            causal_mask = (
                torch.tril(torch.ones(x.size(-2), x.size(-1)), diagonal=0)
                .view(1, 1, x.size(-2), x.size(-1))
                .to(x.device)  # (1, 1, len_q, len_k)
            )
            mask = (
                mask.bool() & causal_mask.bool()
                if mask is not None
                else causal_mask.bool()
            )
            x = x.masked_fill(mask == 0, -1e9)  # (batch_size, num_heads, len_q, len_k)
        else:
            if mask is not None:
                x = x.masked_fill(
                    mask == 0, -1e9
                )  # (batch_size, num_heads, len_q, len_k)
        x = F.softmax(
            x, dim=-1
        )  # (batch_size, num_heads // groups, groups, len_q, len_k)
        if self.save_attn_scores_to_visualize:
            self.attn_scores = x
        r = x @ V  # (batch_size, kv_heads, len_q, d_v)

        return r
