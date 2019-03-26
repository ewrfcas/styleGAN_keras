from keras.engine.topology import Layer
import keras.backend as K
from keras.initializers import VarianceScaling, zeros
import tensorflow as tf
import numpy as np


def keras_log2(x):
    numerator = K.log(x)
    denominator = K.log(K.constant(2, dtype=numerator.dtype))
    return numerator / denominator


def count_params(weights):
    return int(np.sum([K.count_params(p) for p in set(weights)]))


# revised from https://kexue.fm/archives/6418#comment-10799
class SetLearningRate:
    """层的一个包装，用来设置当前层的学习率"""

    def __init__(self, layer, lamb, is_ada=True):
        self.layer = layer
        self.lamb = lamb  # 学习率比例
        self.is_ada = is_ada  # 是否自适应学习率优化器

    def __call__(self, inputs):
        with K.name_scope(self.layer.name):
            if not self.layer.built:
                input_shape = K.int_shape(inputs)
                self.layer.build(input_shape)
                self.layer.built = True
                if self.layer._initial_weights is not None:
                    self.layer.set_weights(self.layer._initial_weights)
        for key in ['kernel', 'bias', 'embeddings', 'depthwise_kernel',
                    'pointwise_kernel', 'recurrent_kernel', 'gamma', 'beta']:
            if hasattr(self.layer, key):
                weight = getattr(self.layer, key)
                if self.is_ada:
                    lamb = self.lamb  # 自适应学习率优化器直接保持lamb比例
                else:
                    lamb = self.lamb ** 0.5  # SGD（包括动量加速），lamb要开平方
                K.set_value(weight, K.eval(weight) / lamb)  # 更改初始化
                setattr(self.layer, key, weight * lamb)  # 按比例替换
        return self.layer(inputs)


class NoiseLayer(Layer):
    def __init__(self, **kwargs):
        super(NoiseLayer, self).__init__(**kwargs)

    def build(self, input_shape):
        self.noise_weight = self.add_weight('noise_weight',
                                            shape=[input_shape[-1]],
                                            initializer=zeros())

    def call(self, x, **kwargs):
        noise = K.random_normal([K.shape(x)[0], K.shape(x)[1], K.shape(x)[2], 1], dtype=x.dtype)  # [batch, h, w, c]
        return x + noise * K.reshape(self.noise_weight, [1, 1, 1, -1])


class StyleLayer(Layer):
    def __init__(self, **kwargs):
        super(StyleLayer, self).__init__(**kwargs)

    def build(self, input_shape):
        self.style_weight = self.add_weight("style_weight",
                                            shape=(input_shape[1][-1], input_shape[0][-1] * 2),
                                            initializer=VarianceScaling(scale=1),
                                            trainable=True)
        self.bias = self.add_weight("style_bias",
                                    shape=(input_shape[0][-1] * 2,),
                                    initializer=zeros(),
                                    trainable=True)

    def call(self, x, **kwargs):
        x, dlatent = x
        style = tf.matmul(dlatent, self.style_weight) + self.bias
        style = K.reshape(style, [-1, 2, 1, 1, x.shape[-1]])  # [N2HWC]
        return x * (style[:, 0] + 1) + style[:, 1]


class InstanceNorm(Layer):
    def __init__(self, eps=1e-8, **kwargs):
        super(InstanceNorm, self).__init__(**kwargs)
        self.eps = eps

    def call(self, x, **kwargs):
        assert len(x.shape) == 4  # [NHWC]
        x -= K.mean(x, axis=[1, 2], keepdims=True)
        x /= K.sqrt(K.mean(K.square(x), axis=[1, 2], keepdims=True) + self.eps)

        return x


class Blur2D(Layer):
    def __init__(self, f=[1, 2, 1], stride=1, normalize=True, flip=False, **kwargs):
        super(Blur2D, self).__init__(**kwargs)
        self.f = f
        self.normalize = normalize
        self.flip = flip
        self.stride = stride

    def build(self, input_shape):
        # Finalize filter kernel.
        self.f = np.array(self.f, dtype=np.float32)
        if self.f.ndim == 1:
            self.f = self.f[:, np.newaxis] * self.f[np.newaxis, :]
        assert self.f.ndim == 2
        if self.normalize:
            self.f /= np.sum(self.f)
        if self.flip:
            self.f = self.f[::-1, ::-1]
        self.f = self.f[:, :, np.newaxis, np.newaxis]
        self.f = np.tile(self.f, [1, 1, input_shape[-1], 1])
        self.filters = K.constant(self.f, name='filter')

    def call(self, x, **kwargs):
        if self.f.shape == (1, 1) and self.f[0, 0] == 1:
            return x
        else:
            x = K.depthwise_conv2d(x, self.filters, strides=(self.stride, self.stride), padding='same')
            return x


class ResidualAdd(Layer):
    def __init__(self, **kwargs):
        super(ResidualAdd, self).__init__(**kwargs)

    def call(self, inputs, **kwargs):
        assert len(inputs) == 3
        x1, x2, res_in = inputs
        res_in = K.mean(res_in)
        ratio = K.clip(keras_log2(res_in) - keras_log2(K.cast(K.shape(x1)[1], float)), 0.0, 1.0)
        return x1 * ratio + x2 * (1 - ratio)


class TruncationLayer(Layer):
    def __init__(self, num_layers, dlatent_size, truncation_psi, truncation_cutoff, dlatent_avg_beta, **kwargs):
        super(TruncationLayer, self).__init__(**kwargs)
        self.num_layers = num_layers
        self.dlatent_size = dlatent_size
        self.dlatent_avg_beta = dlatent_avg_beta
        self.truncation_psi = truncation_psi
        self.truncation_cutoff = truncation_cutoff

    def build(self, input_shape):
        self.dlatent_avg = self.add_weight("dlatent_avg",
                                           shape=(self.dlatent_size,),
                                           initializer=zeros(),
                                           trainable=False)

    def call(self, dlatents, **kwargs):
        # dlatents: [batch, layers, dim]
        def lerp(a, b, t):
            return a + (b - a) * t

        with K.name_scope('Truncation'):
            if self.dlatent_avg_beta is not None:
                batch_avg = K.mean(dlatents[:, 0], axis=0)  # [dim,]

                updated_dlatent_avg = K.in_train_phase(lerp(batch_avg, self.dlatent_avg, self.dlatent_avg_beta),
                                                       self.dlatent_avg)
                update_op = tf.assign(self.dlatent_avg, updated_dlatent_avg)
                with tf.control_dependencies([update_op]):
                    dlatents = K.in_train_phase(dlatents, dlatents)

            if self.truncation_psi is not None and self.truncation_cutoff is not None:
                layer_idx = np.arange(self.num_layers)[np.newaxis, :, np.newaxis]
                ones_coef = np.ones(layer_idx.shape, dtype=np.float32)
                coefs = tf.where(layer_idx < self.truncation_cutoff, self.truncation_psi * ones_coef, ones_coef)
                dlatents = lerp(self.dlatent_avg, dlatents, coefs)

            dlatents = K.in_train_phase(dlatents, dlatents)

            return dlatents


class StyleMixLayer(Layer):
    def __init__(self, style_mixing_prob, num_layers, G_mapping, **kwargs):
        super(StyleMixLayer, self).__init__(**kwargs)
        self.style_mixing_prob = style_mixing_prob
        self.num_layers = num_layers
        self.G_mapping = G_mapping

    def call(self, inputs, **kwargs):
        labels_in = None
        if len(inputs) == 3:
            dlatents, labels_in, res_in = inputs
        else:
            dlatents, res_in = inputs
        # dlatents: [batch, layers, dim]
        res_in = K.mean(res_in)
        with K.name_scope('StyleMix'):
            dlatents2 = tf.random_normal(tf.shape(dlatents[:, 0, :]))
            if labels_in is not None:
                dlatents2 = self.G_mapping(dlatents2, labels_in)
            else:
                dlatents2 = self.G_mapping(dlatents2)
            layer_idx = np.arange(self.num_layers)[np.newaxis, :, np.newaxis]
            cur_layers = tf.cast(keras_log2(res_in), tf.int32) * 2 - 2
            mixing_cutoff = tf.cond(
                tf.random_uniform([], 0.0, 1.0) < self.style_mixing_prob,
                lambda: tf.random_uniform([], 1, cur_layers, dtype=tf.int32),
                lambda: cur_layers)
            dlatents = tf.where(tf.broadcast_to(layer_idx < mixing_cutoff, tf.shape(dlatents)), dlatents, dlatents2)

            return dlatents


class MinibatchStddevLayer(Layer):
    def __init__(self, group_size=4, num_new_features=1, **kwargs):
        super(MinibatchStddevLayer, self).__init__(**kwargs)
        self.group_size = group_size
        self.num_new_features = num_new_features

    def call(self, x, **kwargs):
        with K.name_scope('MinibatchStddev'):
            # Minibatch must be divisible by (or smaller than) group_size.
            x = tf.transpose(x, perm=[0, 3, 1, 2])  # [NHWC]->[NCHW]
            group_size = tf.minimum(self.group_size, tf.shape(x)[0])
            s = x.shape  # [NCHW]  Input shape.
            # [GMncHW] Split minibatch into M groups of size G. Split channels into n channel groups c.
            y = tf.reshape(x, [group_size, -1, self.num_new_features,
                               s[1] // self.num_new_features, s[2], s[3]])
            y = tf.cast(y, tf.float32)  # [GMncHW] Cast to FP32.
            y -= tf.reduce_mean(y, axis=0, keepdims=True)  # [GMncHW] Subtract mean over group.
            y = tf.reduce_mean(tf.square(y), axis=0)  # [MncHW]  Calc variance over group.
            y = tf.sqrt(y + 1e-8)  # [MncHW]  Calc stddev over group.
            y = tf.reduce_mean(y, axis=[2, 3, 4], keepdims=True)  # [Mn111]  Take average over fmaps and pixels.
            y = tf.reduce_mean(y, axis=[2])  # [Mn11] Split channels into c channel groups
            y = tf.cast(y, x.dtype)  # [Mn11]  Cast back to original data type.
            y = tf.tile(y, [group_size, 1, s[2], s[3]])  # [NnHW]  Replicate over group and pixels.
            x = tf.concat([x, y], axis=1)  # [NCHW]  Append as new fmap.
            x = tf.transpose(x, perm=[0, 2, 3, 1])  # [NHWC]

            return x

    def compute_output_shape(self, input_shape):
        output_shape = (input_shape[0], input_shape[1], input_shape[2], input_shape[3] + 1)
        return output_shape
