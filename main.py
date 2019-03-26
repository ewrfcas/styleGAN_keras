from model.styleGAN import StyleGAN
import numpy as np
import os
from keras.optimizers import Adam

os.environ['CUDA_VISIBLE_DEVICES'] = '4,5,6,7'

if __name__ == '__main__':
    model = StyleGAN(max_resolution=128, gpu_num=4)

    X = np.random.random((3200, 512))
    X_const = np.ones((3200, 1, 1, 1))
    X_res = np.ones((3200, 1)) * 128

    model._G_train_models_mg[128].fit([X, X_const, X_res], y=None, batch_size=128)

    # for i in range(1000000):
    #     g_loss = model._G_train_models[256].train_on_batch([X[0:16, :], X_const[0:16, ::], X_res[0:16, :]], y=None)
    #     print('step', i, ':', g_loss)
