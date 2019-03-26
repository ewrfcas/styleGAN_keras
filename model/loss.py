import keras.backend as K
from keras.layers import Input, Lambda, Reshape
from keras.models import Model


def build_loss(G_model, D_model, res, latent_size, loss_type):
    x_real = Input(batch_shape=(None, res, res, 3))
    latents_in = Input(batch_shape=(None, latent_size))
    res_in = Input(batch_shape=(None, 1))
    const_scale = Input(batch_shape=(None, 1, 1, 1))

    x_fake = G_model([latents_in, const_scale, res_in])
    x_fake = Reshape((res, res, x_fake.shape[-1]))(x_fake)
    x_real_score = D_model([x_real, res_in])
    x_fake_score = D_model([x_fake, res_in])

    if loss_type == 'wgan-gp':
        def interpolating(x):
            u = K.random_uniform((K.shape(x[0])[0],) + (1,) * (K.ndim(x[0]) - 1))
            return x[0] * u + x[1] * (1 - u)

        x_inter = Lambda(interpolating)([x_real, x_fake])
        x_inter_score = D_model([x_inter, res_in])

        D_train_model = Model([x_real, latents_in, const_scale, res_in],
                              [x_real_score, x_fake_score, x_inter_score],
                              name='D_train_model_%dx%d' % (res, res))
        D_loss = D_wgan_gp(x_real_score, x_fake_score, x_inter_score, x_inter)
        D_train_model.add_loss(D_loss)

        G_train_model = Model([latents_in, const_scale, res_in], x_fake_score,
                              name='G_train_model_%dx%d' % (res, res))
        G_loss = G_wgan_gp(x_fake_score)
        G_train_model.add_loss(G_loss)

        return G_train_model, D_train_model

    elif loss_type == 'logistic':
        D_train_model = Model([x_real, latents_in, const_scale, res_in],
                              [x_real_score, x_fake_score],
                              name='D_train_model_%dx%d' % (res, res))
        D_loss = D_logistic_simplegp(x_real_score, x_fake_score, x_real, x_fake)
        D_train_model.add_loss(D_loss)

        G_train_model = Model([latents_in, const_scale, res_in], x_fake_score,
                              name='G_train_model_%dx%d' % (res, res))
        G_loss = G_logistic_nonsaturating(x_fake_score)
        G_train_model.add_loss(G_loss)

        return G_train_model, D_train_model
    else:
        raise NotImplementedError()


def D_wgan_gp(x_real_score, x_fake_score, x_inter_score, x_inter, GP_Lambda=10.0):
    grads = K.gradients(x_inter_score, [x_inter])[0]
    grad_norms = K.sqrt(K.sum(grads ** 2, axis=[1, 2, 3]) + 1e-8)
    d_loss = K.mean(x_fake_score - x_real_score) + GP_Lambda * K.mean((grad_norms - 1.) ** 2)
    return d_loss


def G_wgan_gp(x_fake_score):
    g_loss = K.mean(- x_fake_score)
    return g_loss


def G_logistic_nonsaturating(x_fake_score):
    g_loss = K.mean(K.softplus(-x_fake_score))
    return g_loss


def D_logistic_simplegp(x_real_score, x_fake_score, x_real, x_fake, r1_gamma=10.0, r2_gamma=0.0):
    d_loss = K.mean(K.softplus(x_fake_score) - K.softplus(x_real_score))

    if r1_gamma != 0.0:
        with K.name_scope('R1Penalty'):
            r1_grads = K.gradients(x_real_score, [x_real])[0]
            r1_grads_norms = K.sqrt(K.sum(r1_grads ** 2, axis=[1, 2, 3]) + 1e-8)
        d_loss += r1_grads_norms * (r1_gamma * 0.5)

    if r2_gamma != 0.0:
        with K.name_scope('R2Penalty'):
            r2_grads = K.gradients(x_fake_score, [x_fake])[0]
            r2_grads_norms = K.sqrt(K.sum(r2_grads ** 2, axis=[1, 2, 3]) + 1e-8)
        d_loss += r2_grads_norms * (r1_gamma * 0.5)

    return d_loss
