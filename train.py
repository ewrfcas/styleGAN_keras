from model.styleGAN import StyleGAN
import numpy as np
import os
import argparse

os.environ['CUDA_VISIBLE_DEVICES'] = '0'


def parse_args():
    desc = "StyleGAN_keras"
    parser = argparse.ArgumentParser(description=desc)
    parser.add_argument('--name', type=str, default='StyleGAN_test0', help='model name')

    # param
    parser.add_argument('--max_res', type=int, default=1024, help='max img size')
    parser.add_argument('--min_res', type=int, default=4, help='min img size')
    parser.add_argument('--start_res', type=int, default=8, help='train start from what res?')
    parser.add_argument('--loss_type', type=str, default='logistic', help='loss type')
    parser.add_argument('--batch_dict', type=dict, default={4: 512, 8: 256, 16: 128, 32: 64, 64: 32,
                                                            128: 16, 256: 16, 512: 16, 1024: 16}, help='batch dict')
    parser.add_argument('--total_kimg', type=int, default=25000, help='thousands of training imgs total')
    parser.add_argument('--res_training_kimg', type=int, default=600,
                        help='Thousands of real images to show before doubling the resolution.')
    parser.add_argument('--res_transition_kimg', type=int, default=600,
                        help='Thousands of real images to show when fading in new layers.')
    parser.add_argument('--D_repeats', type=int, default=1,
                        help='How many times the discriminator is trained per G iteration.')
    parser.add_argument('--minibatch_repeats', type=int, default=4,
                        help='Number of minibatches to run before adjusting training parameters.')
    parser.add_argument('--mirror_augment', type=bool, default=True, help='Enable mirror augment?')
    parser.add_argument('--check_point_dir', type=str, default='check_points')

    return check_args(parser.parse_args())


def check_args(args):
    args.checkpoint_dir = os.path.join(args.checkpoint_dir, args.name)
    os.makedirs(args.checkpoint_dir, exist_ok=True)
    file_name = os.path.join(args.checkpoint_dir, 'train_setting.txt')
    with open(file_name, 'wt') as opt_file:
        opt_file.write('------------ Options -------------\n')
        for k, v in sorted(args.items()):
            opt_file.write('%s: %s\n' % (str(k), str(v)))
        opt_file.write('-------------- End ----------------\n')

    return args


if __name__ == '__main__':
    model = StyleGAN(gpu_num=0)
