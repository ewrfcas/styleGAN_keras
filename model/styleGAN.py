from keras.models import Input, Model
from keras.layers import Dense, Concatenate, LeakyReLU, Conv2D, Conv2DTranspose, UpSampling2D, Flatten, \
    AveragePooling2D, Lambda, RepeatVector
from keras.initializers import VarianceScaling, ones
from model.keras_layers import SetLearningRate, StyleLayer, InstanceNorm, Blur2D, ResidualAdd, NoiseLayer, \
    TruncationLayer, StyleMixLayer, MinibatchStddevLayer, count_params
import keras.backend as K
from keras.utils import multi_gpu_model
import tensorflow as tf
import math
from model.loss import build_loss
import keras.backend.tensorflow_backend as KTF
from keras.optimizers import Adam


class StyleGAN:
    def __init__(self,
                 min_resolution=4,
                 max_resolution=1024,
                 start_resolution=8,
                 latent_size=512,
                 label_size=0,
                 mapping_layers=8,
                 mapping_lrmul=0.01,
                 truncation_psi=0.7,
                 truncation_cutoff=8,
                 dlatent_avg_beta=0.995,
                 style_mixing_prob=0.9,
                 mbstd_group_size=4,
                 mbstd_num_features=1,
                 loss_type='logistic',
                 gpu_num=0):
        # set model_num according to the resolution
        assert min_resolution <= max_resolution <= 1024
        assert 4 <= min_resolution <= max_resolution
        self.max_resolution = max_resolution
        self.min_resolution = min_resolution
        self.start_resolution = start_resolution
        self.latent_size = latent_size
        self.label_size = label_size
        self.mapping_layers = mapping_layers
        self.mapping_lrmul = mapping_lrmul
        self.truncation_psi = truncation_psi
        self.truncation_cutoff = truncation_cutoff
        self.dlatent_avg_beta = dlatent_avg_beta
        self.style_mixing_prob = style_mixing_prob
        self.mbstd_group_size = mbstd_group_size
        self.mbstd_num_features = mbstd_num_features
        assert loss_type in {'wgan-gp', 'logistic'}
        self.loss_type = loss_type
        self.gpu_num = gpu_num
        self.model_num = int(math.log2(self.max_resolution // self.min_resolution)) + 1
        # resolution: channel
        self.channel_dict = {4: 512, 8: 512, 16: 512, 32: 512, 64: 256, 128: 128, 256: 64, 512: 32, 1024: 16}
        self.lr_dict = {4: 0.001, 8: 0.001, 16: 0.001, 32: 0.001, 64: 0.001,
                        128: 0.0015, 256: 0.002, 512: 0.003, 1024: 0.003}
        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True
        sess = tf.Session(config=config)
        KTF.set_session(sess)

        # init the shared Mapping Network in Generators
        self._G_mapping = self.G_mapping()

        self._G_encoders = {}
        self._G_models = {}
        self._G_train_models = {}
        self._G_train_models_mg = {}

        self._D_decoders = {}
        self._D_models = {}
        self._D_train_models = {}
        self._D_train_models_mg = {}

        print('building G models...')
        for i in range(self.model_num):
            res = int(self.min_resolution * math.pow(2, i))
            with tf.device('/cpu:0'):
                self._G_encoders[res] = self.G_encoder(res)
                self._G_models[res] = self.G_model(res)
                print('Generator %dx%d output:' % (res, res), self._G_models[res].outputs[0].shape)

                self._D_decoders[res] = self.D_decoder(res)
                self._D_models[res] = self.D_model(res)
                print('Discriminator %dx%d output:' % (res, res), self._D_models[res].outputs[0].shape)

                if res >= start_resolution:
                    self._G_train_models[res], self._D_train_models[res] = \
                        build_loss(self._G_models[res], self._D_models[res], res, self.latent_size, self.loss_type)

            if res >= start_resolution:
                if self.gpu_num > 1:
                    self._G_train_models_mg[res] = multi_gpu_model(self._G_train_models[res], gpus=self.gpu_num)
                    self._D_train_models_mg[res] = multi_gpu_model(self._D_train_models[res], gpus=self.gpu_num)
                    self._G_train_models_mg[res].compile(optimizer=Adam(lr=self.lr_dict[res],
                                                                        beta_1=0.0, beta_2=0.99, epsilon=1e-8))
                    self._D_train_models_mg[res].compile(optimizer=Adam(lr=self.lr_dict[res],
                                                                        beta_1=0.0, beta_2=0.99, epsilon=1e-8))
                else:
                    self._G_train_models[res].compile(optimizer=Adam(lr=self.lr_dict[res],
                                                                     beta_1=0.0, beta_2=0.99, epsilon=1e-8))
                    self._D_train_models[res].compile(optimizer=Adam(lr=self.lr_dict[res],
                                                                     beta_1=0.0, beta_2=0.99, epsilon=1e-8))

        print('Generator trainable params:', count_params(self._G_models[self.max_resolution].trainable_weights))
        print('Discriminator trainable params:', count_params(self._D_models[self.max_resolution].trainable_weights))

    def G_mapping(self):
        with K.name_scope('G_mapping'):
            latents_in = Input(batch_shape=(None, self.latent_size))
            inputs = [latents_in]
            if self.label_size > 0:
                labels_in = Input(batch_shape=(None, self.label_size))
                y = Dense(self.label_size, use_bias=False)(labels_in)  # [batch, label_size]
                x = Concatenate(axis=1)([latents_in, y])  # [batch, latent_size+label_size]
                inputs.append(labels_in)
            else:
                x = latents_in
            with K.name_scope('Dense_mapping'):
                # according to the paper, all G_mapping dense should use lr with 0.01 decay
                for i in range(self.mapping_layers):
                    x = SetLearningRate(Dense(self.latent_size, kernel_initializer=VarianceScaling(math.sqrt(2))),
                                        lamb=self.mapping_lrmul)(x)
                    x = LeakyReLU(0.2)(x)

            x = RepeatVector(self.model_num * 2)(x)  # [batch, dim]->[batch, layer*2, dim]

            # truncation trick
            # x = TruncationLayer(self.model_num * 2, self.latent_size, self.truncation_psi,
            #                     self.truncation_cutoff, self.dlatent_avg_beta)(x)

        return Model(inputs=inputs, outputs=x)

    def layer_epilogue(self, x, dlatents_in):
        x = NoiseLayer()(x)
        x = LeakyReLU(0.2)(x)
        x = InstanceNorm()(x)
        x = StyleLayer()([x, dlatents_in])
        return x

    def upscale2d_conv2d(self, x, res):
        fused_scale = (res >= 64)
        if fused_scale:
            x = Conv2DTranspose(self.channel_dict[res], kernel_size=3, strides=2,
                                kernel_initializer=VarianceScaling(math.sqrt(2)),
                                padding='same')(x)
        else:
            x = UpSampling2D(size=(2, 2))(x)
            x = Conv2D(self.channel_dict[res], kernel_size=3, strides=1,
                       kernel_initializer=VarianceScaling(math.sqrt(2)),
                       padding='same')(x)
        x = Blur2D()(x)
        return x

    def downscale2d_conv2d(self, x, res):
        fused_scale = (res >= 64)
        x = Blur2D()(x)
        if fused_scale:
            x = Conv2D(self.channel_dict[res], kernel_size=3, strides=2,
                       kernel_initializer=VarianceScaling(math.sqrt(2)),
                       padding='same', name='Conv1')(x)
        else:
            x = Conv2D(self.channel_dict[res], kernel_size=3, strides=1,
                       kernel_initializer=VarianceScaling(math.sqrt(2)),
                       padding='same', name='Conv1')(x)
            x = AveragePooling2D(pool_size=(2, 2))(x)

        return x

    def G_encoder(self, res):
        scope_name = '%dx%d_G_Encoder' % (res, res)
        with K.name_scope(scope_name):
            dlatents_mapping = Input(batch_shape=(None, self.model_num * 2, self.latent_size),
                                     name=scope_name + '_latent')
            const_scale = Input(batch_shape=(None, 1, 1, 1), name=scope_name + '_const')
            # use single scale 1 to get the const vector

            if res == self.min_resolution:
                with K.name_scope('Const'):
                    x = Conv2DTranspose(self.latent_size, kernel_size=4, use_bias=False,
                                        kernel_initializer=ones(), name='const')(const_scale)  # [batch, 4, 4, 512]
                    dlatent_res = Lambda(lambda x: x[:, 0, :])(dlatents_mapping)
                    x = self.layer_epilogue(x, dlatent_res)
                with K.name_scope('Conv'):
                    x = Conv2D(self.channel_dict[res], kernel_size=3, strides=1,
                               kernel_initializer=VarianceScaling(math.sqrt(2)),
                               padding='same')(x)
                    dlatent_res = Lambda(lambda x: x[:, 1, :])(dlatents_mapping)
                    x = self.layer_epilogue(x, dlatent_res)
            else:
                # x should be [batch, res//2, res//2, ch]
                x = self._G_encoders[res // 2]([dlatents_mapping, const_scale])
                with K.name_scope('Conv0_up'):
                    x = self.upscale2d_conv2d(x, res)
                    dlatent_res = Lambda(lambda x: x[:, int(math.log2(res) * 2 - 4), :])(dlatents_mapping)
                    x = self.layer_epilogue(x, dlatent_res)
                with K.name_scope('Conv1'):
                    x = Conv2D(self.channel_dict[res], kernel_size=3, strides=1,
                               kernel_initializer=VarianceScaling(math.sqrt(2)),
                               padding='same')(x)
                    dlatent_res = Lambda(lambda x: x[:, int(math.log2(res) * 2 - 3), :])(dlatents_mapping)
                    x = self.layer_epilogue(x, dlatent_res)

            return Model(inputs=[dlatents_mapping, const_scale], outputs=x, name=scope_name)

    def G_model(self, res):
        # G_model{i} = (G_mapping => G_encoder{i} => G_torgb{i}) + G_encoder_{i-1}
        scope_name = '%dx%d_G_Model' % (res, res)
        with K.name_scope(scope_name):
            dlatents_in = Input(batch_shape=(None, self.latent_size), name=scope_name + '_latent')
            labels_in = Input(batch_shape=(None, self.label_size))
            const_scale = Input(batch_shape=(None, 1, 1, 1), name=scope_name + '_const')
            res_in = Input(batch_shape=(None, 1), name='res_in')  # the training input res e.g.[4,8,16...]

            # get G_mapping [batch, layer, dim]
            dlatents_mapping = self._G_mapping(dlatents_in if self.label_size == 0 else [dlatents_in, labels_in])

            # style mixing
            dlatents_mapping = StyleMixLayer(self.style_mixing_prob, self.model_num * 2, self._G_mapping) \
                ([dlatents_mapping, res_in] if self.label_size == 0 else [dlatents_mapping, labels_in, res_in])

            # get the feature map from the G_encoder
            x = self._G_encoders[res]([dlatents_mapping, const_scale])

            # to rgb
            with K.name_scope(scope_name + '_toRGB'):
                x = Conv2D(3, kernel_size=1, strides=1, kernel_initializer=VarianceScaling(1))(x)
                if res > self.min_resolution:
                    # x_pre: [batch, res//2, res//2, 3]
                    x_pre = self._G_models[res // 2]([dlatents_in, const_scale, res_in])
                    x_pre = UpSampling2D()(x_pre)  # [batch, res, res, 3]
                    x = ResidualAdd()([x, x_pre, res_in])

            if self.label_size == 0:
                inputs = [dlatents_in, const_scale, res_in]
            else:
                inputs = [dlatents_in, labels_in, const_scale, res_in]

            return Model(inputs=inputs, outputs=x, name=scope_name)

    def from_RGB(self, img_in, res):
        img_feat = Conv2D(self.channel_dict[res], kernel_size=1, strides=1,
                          kernel_initializer=VarianceScaling(math.sqrt(2)),
                          padding='same')(img_in)
        img_feat = LeakyReLU(0.2)(img_feat)

        return img_feat

    def D_decoder(self, res):
        scope_name = '%dx%d_D_Decoder' % (res, res)
        with K.name_scope(scope_name):
            # input_feat res>4: img_out+x res=4: img_out
            input_feat = Input(batch_shape=(None, res, res, self.channel_dict[res]))
            labels_in = Input(batch_shape=(None, self.label_size))
            img_in = Input(batch_shape=(None, res, res, 3))  # the image
            res_in = Input(batch_shape=(None, 1), name='res_in')  # the training input res

            # print(input_feat, img_in)

            if res == self.min_resolution:
                # minibatch_stddev_layer
                x = MinibatchStddevLayer(self.mbstd_group_size, self.mbstd_num_features)(input_feat)
                x = Conv2D(self.channel_dict[res], kernel_size=3, strides=1,
                           kernel_initializer=VarianceScaling(math.sqrt(2)),
                           padding='same', name='Conv')(x)
                x = LeakyReLU(0.2)(x)
                x = Flatten()(x)
                x = Dense(self.channel_dict[res], kernel_initializer=VarianceScaling(math.sqrt(2)), name='Dense0')(x)
                x = LeakyReLU(0.2)(x)
                x = Dense(max(1, self.label_size), kernel_initializer=VarianceScaling(1), name='Dense1')(x)

                if self.label_size > 0:
                    with K.name_scope('LabelSwitch'):
                        x = Lambda(lambda x: K.sum(x[0] * x[1], axis=1, keepdims=True))([x, labels_in])  # [batch, 1]
            else:
                x = Conv2D(self.channel_dict[res], kernel_size=3, strides=1,
                           kernel_initializer=VarianceScaling(math.sqrt(2)),
                           padding='same', name='Conv0')(input_feat)
                x = LeakyReLU(0.2)(x)
                x = self.downscale2d_conv2d(x, res // 2)  # [batch, h//2, w//2, c]

                img_downscale = AveragePooling2D()(img_in)  # [batch, h//2, w//2, 3]
                img_feat = self.from_RGB(img_downscale, res // 2)
                x = ResidualAdd()([x, img_feat, res_in])

                if self.label_size > 0:
                    x = self._D_decoders[res // 2]([x, img_downscale, res_in, labels_in])
                else:
                    x = self._D_decoders[res // 2]([x, img_downscale, res_in])

            inputs = [input_feat, img_in, res_in]
            if self.label_size > 0:
                inputs.append(labels_in)

            return Model(inputs=inputs, outputs=x, name=scope_name)

    def D_model(self, res):
        scope_name = '%dx%d_D_Model' % (res, res)
        with K.name_scope(scope_name):
            img_in = Input(batch_shape=(None, res, res, 3))
            res_in = Input(batch_shape=(None, 1), name='res_in')  # the training input res

            x = self.from_RGB(img_in, res)
            x = self._D_decoders[res]([x, img_in, res_in])

            return Model(inputs=[img_in, res_in], outputs=x, name=scope_name)
