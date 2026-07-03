import math

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from model import (
    Encoder,
    EncoderLayer,
    FeedForward,
    Gating,
    MultiHeadAttention,
    Norm,
    PositionalEncoder,
    Transformer,
    attention,
    get_clones,
)

D_MODEL = 16
HEADS = 4
N_LAYERS = 2
M_FEATURES = 14
DROPOUT = 0.1


@pytest.fixture(autouse=True)
def fixed_seed():
    torch.manual_seed(0)


def make_input(m=M_FEATURES):
    return torch.rand(1, 1, 3, m)


def test_transformer_forward_output_shape():
    model = Transformer(M_FEATURES, D_MODEL, N_LAYERS, HEADS, DROPOUT)
    model.eval()
    out = model(make_input(), t=5)
    assert out.shape == (1,)
    assert torch.isfinite(out).all()


def test_transformer_forward_various_feature_counts():
    for m in (4, 8, 14):
        model = Transformer(m, D_MODEL, N_LAYERS, HEADS, DROPOUT)
        model.eval()
        out = model(make_input(m), t=1)
        assert out.shape == (1,)


def test_gating_output_shape():
    gating = Gating(D_MODEL, M_FEATURES)
    out = gating(make_input())
    assert out.shape == (1, 1, 1, D_MODEL)


def test_gating_init_weights_within_bounds():
    gating = Gating(D_MODEL, M_FEATURES)
    stdv = 1.0 / math.sqrt(M_FEATURES)
    for name in ("W_r", "V_r", "b_r", "W_u", "V_u", "b_u", "W_e", "b_e"):
        param = getattr(gating, name)
        assert param.data.min() >= -stdv
        assert param.data.max() <= stdv


def test_encoder_output_shape():
    encoder = Encoder(D_MODEL, N_LAYERS, HEADS, M_FEATURES, DROPOUT)
    encoder.eval()
    src = torch.rand(1, 1, 1, D_MODEL)
    out = encoder(src, t=3)
    assert out.shape == (1, 1, D_MODEL)


def test_positional_encoder_matches_formula():
    pe_module = PositionalEncoder(D_MODEL)
    x = torch.zeros(1, D_MODEL)
    t = 7
    out = pe_module(x, t)

    expected = np.zeros(D_MODEL)
    for i in range(0, D_MODEL, 2):
        expected[i] = math.sin(t / (10000 ** ((2 * i) / D_MODEL)))
        expected[i + 1] = math.cos(t / (10000 ** ((2 * (i + 1)) / D_MODEL)))
    assert torch.allclose(out, torch.Tensor(expected).unsqueeze(0), atol=1e-6)


def test_positional_encoder_scales_input():
    pe_module = PositionalEncoder(D_MODEL)
    x = torch.ones(1, D_MODEL)
    out_zero_t = pe_module(torch.zeros(1, D_MODEL), t=2)
    out = pe_module(x, t=2)
    assert torch.allclose(out - out_zero_t, x * math.sqrt(D_MODEL), atol=1e-6)


def test_get_clones_creates_independent_copies():
    layer = EncoderLayer(D_MODEL, HEADS, DROPOUT)
    clones = get_clones(layer, 3)
    assert len(clones) == 3
    assert clones[0] is not clones[1]
    with torch.no_grad():
        clones[0].norm_1.alpha.fill_(2.0)
    assert not torch.equal(clones[0].norm_1.alpha, clones[1].norm_1.alpha)


def test_encoder_layer_output_shape():
    layer = EncoderLayer(D_MODEL, HEADS, DROPOUT)
    layer.eval()
    x = torch.rand(1, D_MODEL)
    out = layer(x, None)
    assert out.shape == (1, 1, D_MODEL)


def test_norm_output():
    norm = Norm(D_MODEL)
    x = torch.rand(2, D_MODEL)
    out = norm(x)
    assert out.shape == x.shape
    assert torch.allclose(out.mean(dim=-1), torch.zeros(2), atol=1e-5)


def test_multihead_attention_output_shape():
    mha = MultiHeadAttention(HEADS, D_MODEL, DROPOUT)
    mha.eval()
    x = torch.rand(1, D_MODEL)
    out = mha(x, x, x)
    assert out.shape == (1, 1, D_MODEL)


def test_attention_function_shapes_and_softmax():
    bs, h, sl = 1, HEADS, 3
    d_k = D_MODEL // HEADS
    q = torch.rand(bs, h, sl, d_k)
    k = torch.rand(bs, h, sl, d_k)
    v = torch.rand(bs, h, sl, d_k)
    out = attention(q, k, v, d_k)
    assert out.shape == (bs, h, sl, d_k)

    # with identical values, output rows are convex combinations of v rows
    assert out.min() >= v.min() - 1e-6
    assert out.max() <= v.max() + 1e-6


def test_feedforward_preserves_shape():
    ff = FeedForward(D_MODEL, d_ff=32, dropout=0.0)
    ff.eval()
    x = torch.rand(2, D_MODEL)
    out = ff(x)
    assert out.shape == x.shape


def test_transformer_backward_produces_gradients():
    model = Transformer(M_FEATURES, D_MODEL, N_LAYERS, HEADS, DROPOUT)
    out = model(make_input(), t=2)
    out.sum().backward()
    grads = [p.grad for p in model.parameters() if p.grad is not None]
    assert len(grads) > 0
    assert any(g.abs().sum() > 0 for g in grads)
