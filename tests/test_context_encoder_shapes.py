"""Unit tests for ContextEncoder shapes and sequence lengths."""

import pytest
import torch
from src.models.context_encoder import ContextEncoder


@pytest.mark.parametrize("batch_size", [1, 2])
@pytest.mark.parametrize("seq_len", [3, 7])
@pytest.mark.parametrize("h,w", [(20, 20), (40, 40), (112, 240)])
def test_context_encoder_forward_shapes(batch_size, seq_len, h, w):
    encoder = ContextEncoder(in_channels=25, hidden_dims=(32, 64, 64))
    x = torch.randn(batch_size, seq_len, 25, h, w)
    out = encoder(x)

    assert out.shape == (batch_size, 64, h, w)
    assert torch.isfinite(out).all()
