import math
import tensorflow as tf


'''
num_epochs = 1 # Number of training epochs
d_model = 128  # dimension in encoder
heads = 4  # number of heads in multi-head attention
N = 2  # number of encoder layers
m = 14  # number of features
'''


def uniform_init(fan_in):
    # matches the default bias initialization of torch.nn.Linear / torch.nn.Conv2d
    bound = 1.0 / math.sqrt(fan_in)
    return tf.keras.initializers.RandomUniform(-bound, bound)


def linear(in_features, out_features):
    return tf.keras.layers.Dense(out_features, bias_initializer=uniform_init(in_features))


class Transformer(tf.keras.Model):
    def __init__(self, m, d_model, N, heads, dropout):
        super().__init__()
        self.gating = Gating(d_model, m)
        self.encoder = Encoder(d_model, N, heads, m, dropout)
        self.out = linear(d_model, 1)

    def call(self, src, t, training=None):
        e_i = self.gating(src)
        e_outputs = self.encoder(e_i, t, training=training)
        output = self.out(e_outputs)

        return tf.reshape(output, (1,))


class Gating(tf.keras.layers.Layer):
    def __init__(self, d_model, m):  # 128,14
        super().__init__()
        self.m = m
        self.d_model = d_model

        # 1 output channel, convolution over the stack of 3 rows; operates on NHWC internally
        self.cnn_layers = tf.keras.layers.Conv2D(1, kernel_size=(3, 1), strides=1, bias_initializer=uniform_init(3))

    def build(self, input_shape):
        m, d_model = self.m, self.d_model
        stdv = 1.0 / math.sqrt(m)
        init = tf.keras.initializers.RandomUniform(-stdv, stdv)

        # the reset gate r_i
        self.W_r = self.add_weight(name='W_r', shape=(m, m), initializer=init, trainable=True)
        self.V_r = self.add_weight(name='V_r', shape=(m, m), initializer=init, trainable=True)
        self.b_r = self.add_weight(name='b_r', shape=(m,), initializer=init, trainable=True)

        # the update gate u_i
        self.W_u = self.add_weight(name='W_u', shape=(m, m), initializer=init, trainable=True)
        self.V_u = self.add_weight(name='V_u', shape=(m, m), initializer=init, trainable=True)
        self.b_u = self.add_weight(name='b_u', shape=(m,), initializer=init, trainable=True)

        # the output
        self.W_e = self.add_weight(name='W_e', shape=(m, d_model), initializer=init, trainable=True)
        self.b_e = self.add_weight(name='b_e', shape=(d_model,), initializer=init, trainable=True)

        super().build(input_shape)

    def call(self, x):
        x_i = x[:, :, 1:2, :]  # only applying the gating on the current row even with the stack of 3 rows cames as input (1,1,3,14)
        # (1,1,3,14) NCHW -> (1,3,14,1) NHWC -> conv -> (1,1,14,1) -> back to NCHW (1,1,1,14)
        h_i = tf.transpose(self.cnn_layers(tf.transpose(x, [0, 2, 3, 1])), [0, 3, 1, 2])

        r_i = tf.sigmoid(tf.matmul(h_i, self.W_r) + tf.matmul(x_i, self.V_r) + self.b_r)
        u_i = tf.sigmoid(tf.matmul(h_i, self.W_u) + tf.matmul(x_i, self.V_u) + self.b_u)

        # the output of the gating mechanism
        hh_i = h_i * u_i + x_i * r_i

        return tf.matmul(hh_i, self.W_e) + self.b_e  # (the final output is 1,1,1,128 as the encoder has size of 128.)


class Encoder(tf.keras.layers.Layer):
    def __init__(self, d_model, N, heads, m, dropout):  # d_model = 128  # dimension in encoder, heads = 4  #number of heads in multi-head attention, N = 2  #encoder layers, m = 14  #number of features
        super().__init__()
        self.N = N
        self.pe = PositionalEncoder(d_model)
        self.layers = get_clones(EncoderLayer(d_model, heads, dropout), N)
        self.norm = Norm(d_model)
        self.d_model = d_model

    def call(self, src, t, training=None):
        src = tf.reshape(src, (1, self.d_model))  # this 128 is changed according to d_model
        x = self.pe(src, t)
        for i in range(self.N):
            x = self.layers[i](x, None, training=training)
        return self.norm(x)


class PositionalEncoder(tf.keras.layers.Layer):
    def __init__(self, d_model):
        super().__init__()
        self.d_model = d_model

    def call(self, x, t):
        # make embeddings relatively larger
        x = x * math.sqrt(self.d_model)

        # t may be a python number or a scalar tensor
        t = tf.cast(t, tf.float64)
        i = tf.range(0, self.d_model, 2, dtype=tf.float64)
        pe_even = tf.sin(t / (10000 ** ((2 * i) / self.d_model)))
        pe_odd = tf.cos(t / (10000 ** ((2 * (i + 1)) / self.d_model)))
        pe = tf.reshape(tf.stack([pe_even, pe_odd], axis=1), (self.d_model,))

        x = x + tf.cast(pe, x.dtype)
        return x


# We can then build a convenient cloning function that can generate multiple layers:
def get_clones(module, N):
    return [module.__class__.from_config(module.get_config()) for i in range(N)]


# build an encoder layer with one multi-head attention layer and one # feed-forward layer
class EncoderLayer(tf.keras.layers.Layer):
    def __init__(self, d_model, heads, dropout=0.5, **kwargs):
        super().__init__(**kwargs)
        self.d_model = d_model
        self.heads = heads
        self.dropout_rate = dropout
        self.norm_1 = Norm(d_model)
        self.norm_2 = Norm(d_model)
        self.attn = MultiHeadAttention(heads, d_model, dropout)
        self.ff = FeedForward(d_model)
        self.dropout_1 = tf.keras.layers.Dropout(dropout)
        self.dropout_2 = tf.keras.layers.Dropout(dropout)

    def get_config(self):
        config = super().get_config()
        config.update({'d_model': self.d_model, 'heads': self.heads, 'dropout': self.dropout_rate})
        return config

    def call(self, x, mask, training=None):
        x2 = self.norm_1(x)
        x = x + self.dropout_1(self.attn(x2, x2, x2, mask, training=training), training=training)
        x2 = self.norm_2(x)
        x = x + self.dropout_2(self.ff(x2, training=training), training=training)
        return x


class Norm(tf.keras.layers.Layer):
    def __init__(self, d_model, eps=1e-6):
        super().__init__()

        self.size = d_model
        self.eps = eps

    def build(self, input_shape):
        # create two learnable parameters to calibrate normalisation
        self.alpha = self.add_weight(name='alpha', shape=(self.size,), initializer='ones', trainable=True)
        self.bias = self.add_weight(name='bias', shape=(self.size,), initializer='zeros', trainable=True)
        super().build(input_shape)

    def call(self, x):
        mean = tf.reduce_mean(x, axis=-1, keepdims=True)
        # unbiased standard deviation (Bessel's correction), as in torch.Tensor.std
        n = tf.cast(tf.shape(x)[-1], x.dtype)
        std = tf.sqrt(tf.reduce_sum(tf.square(x - mean), axis=-1, keepdims=True) / (n - 1))
        norm = self.alpha * (x - mean) / (std + self.eps) + self.bias
        return norm


class MultiHeadAttention(tf.keras.layers.Layer):
    def __init__(self, heads, d_model, dropout=0.5):
        super().__init__()

        self.d_model = d_model
        self.d_k = d_model // heads
        self.h = heads

        self.q_linear = linear(d_model, d_model)
        self.v_linear = linear(d_model, d_model)
        self.k_linear = linear(d_model, d_model)
        self.dropout = tf.keras.layers.Dropout(dropout)
        self.out = linear(d_model, d_model)

    def call(self, q, k, v, mask=None, training=None):
        bs = tf.shape(q)[0]

        # perform linear operation and split into h heads

        k = tf.reshape(self.k_linear(k), (bs, -1, self.h, self.d_k))
        q = tf.reshape(self.q_linear(q), (bs, -1, self.h, self.d_k))
        v = tf.reshape(self.v_linear(v), (bs, -1, self.h, self.d_k))

        # transpose to get dimensions bs * h * sl * d_model

        k = tf.transpose(k, [0, 2, 1, 3])
        q = tf.transpose(q, [0, 2, 1, 3])
        v = tf.transpose(v, [0, 2, 1, 3])

        # calculate attention using function we will define next
        scores = attention(q, k, v, self.d_k, mask, self.dropout, training=training)

        # concatenate heads and put through final linear layer
        concat = tf.reshape(tf.transpose(scores, [0, 2, 1, 3]), (bs, -1, self.d_model))

        output = self.out(concat)

        return output


def attention(q, k, v, d_k, mask=None, dropout=None, training=None):
    scores = tf.matmul(q, k, transpose_b=True) / math.sqrt(d_k)

    if mask is not None:
        mask = tf.expand_dims(mask, 1)
    # scores = scores.masked_fill(mask == 0, -1e9)
    scores = tf.nn.softmax(scores, axis=-1)

    if dropout is not None:
        scores = dropout(scores, training=training)

    output = tf.matmul(scores, v)
    return output


class FeedForward(tf.keras.layers.Layer):
    def __init__(self, d_model, d_ff=512, dropout=0.5):
        super().__init__()
        # set d_ff as a default to 512
        self.linear_1 = linear(d_model, d_ff)
        self.dropout = tf.keras.layers.Dropout(dropout)
        self.linear_2 = linear(d_ff, d_model)

    def call(self, x, training=None):
        x = self.dropout(tf.nn.relu(self.linear_1(x)), training=training)
        x = self.linear_2(x)
        return x
