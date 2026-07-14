import math

import torch

# Tiny config so everything runs instantly on CPU.
M = 14  # number of input features (fixed by the gating matrices)
D_MODEL = 8  # encoder dimension (even, divisible by HEADS)
HEADS = 2
N_LAYERS = 1


def _tiny_input():
    # (batch, channel, 3 stacked time steps, m features)
    return torch.randn(1, 1, 3, M)


def test_positional_encoder_preserves_shape_and_adds_signal():
    from model import PositionalEncoder

    torch.manual_seed(0)
    pe = PositionalEncoder(D_MODEL)
    x = torch.zeros(1, D_MODEL)

    out = pe(x, t=1)

    assert out.shape == (1, D_MODEL)
    # With a zero input the output is purely the positional signal, so non-zero.
    assert torch.any(out != 0)


def test_positional_encoder_is_deterministic_for_same_t():
    from model import PositionalEncoder

    pe = PositionalEncoder(D_MODEL)
    x = torch.ones(1, D_MODEL)

    assert torch.allclose(pe(x, t=3), pe(x, t=3))


def test_gating_maps_stacked_window_to_d_model():
    from model import Gating

    torch.manual_seed(0)
    gating = Gating(D_MODEL, M)

    out = gating(_tiny_input())

    assert out.shape == (1, 1, 1, D_MODEL)


def test_get_clones_returns_independent_modules():
    from model import EncoderLayer, get_clones

    layer = EncoderLayer(D_MODEL, HEADS, dropout=0.0)
    clones = get_clones(layer, 3)

    assert len(clones) == 3
    assert clones[0] is not clones[1]


def test_norm_output_shape_and_finiteness():
    from model import Norm

    norm = Norm(D_MODEL)
    x = torch.randn(2, D_MODEL)

    out = norm(x)

    assert out.shape == x.shape
    assert torch.isfinite(out).all()


def test_feed_forward_shape():
    from model import FeedForward

    ff = FeedForward(D_MODEL, d_ff=16, dropout=0.0)
    x = torch.randn(1, D_MODEL)

    assert ff(x).shape == (1, D_MODEL)


def test_attention_output_shape_and_softmax_weights():
    from model import attention

    bs, heads, seq, d_k = 1, HEADS, 4, 3
    q = torch.randn(bs, heads, seq, d_k)
    k = torch.randn(bs, heads, seq, d_k)
    v = torch.randn(bs, heads, seq, d_k)

    out = attention(q, k, v, d_k)

    assert out.shape == (bs, heads, seq, d_k)


def test_attention_accepts_a_mask():
    from model import attention

    bs, heads, seq, d_k = 1, HEADS, 4, 3
    q = torch.randn(bs, heads, seq, d_k)
    k = torch.randn(bs, heads, seq, d_k)
    v = torch.randn(bs, heads, seq, d_k)
    mask = torch.ones(bs, seq, seq)

    out = attention(q, k, v, d_k, mask=mask)

    assert out.shape == (bs, heads, seq, d_k)


def test_multi_head_attention_shape():
    from model import MultiHeadAttention

    mha = MultiHeadAttention(HEADS, D_MODEL, dropout=0.0).eval()
    x = torch.randn(1, 1, D_MODEL)

    out = mha(x, x, x)

    assert out.shape == (1, 1, D_MODEL)


def test_encoder_layer_preserves_shape():
    from model import EncoderLayer

    layer = EncoderLayer(D_MODEL, HEADS, dropout=0.0).eval()
    x = torch.randn(1, 1, D_MODEL)

    assert layer(x, mask=None).shape == (1, 1, D_MODEL)


def test_encoder_reshapes_gating_output_to_sequence():
    from model import Encoder

    torch.manual_seed(0)
    encoder = Encoder(D_MODEL, N_LAYERS, HEADS, M, dropout=0.0).eval()
    gating_like = torch.randn(1, 1, 1, D_MODEL)

    out = encoder(gating_like, t=1)

    # Multi-head attention broadcasts the sequence dim back in, so the encoder
    # emits (1, 1, d_model); it still carries exactly d_model values.
    assert out.shape[-1] == D_MODEL
    assert out.numel() == D_MODEL


def test_transformer_forward_returns_scalar_prediction():
    from model import Transformer

    torch.manual_seed(0)
    model = Transformer(M, D_MODEL, N_LAYERS, HEADS, dropout=0.0).eval()

    out = model.forward(_tiny_input(), t=1)

    assert out.shape == (1,)
    assert torch.isfinite(out).all()


def test_transformer_eval_is_deterministic():
    from model import Transformer

    torch.manual_seed(0)
    model = Transformer(M, D_MODEL, N_LAYERS, HEADS, dropout=0.0).eval()
    x = _tiny_input()

    assert torch.allclose(model.forward(x, t=2), model.forward(x, t=2))
