import os
import argparse
from solver import Solver
from data_loader import get_loader
from torch.backends import cudnn


def str2bool(v):
    return v.lower() in ('true')

def main(config):
    # For fast training.
    cudnn.benchmark = True

    # Create directories if not exist.
    if not os.path.exists(config.log_dir):
        os.makedirs(config.log_dir)
    if not os.path.exists(config.model_save_dir):
        os.makedirs(config.model_save_dir)
    if not os.path.exists(config.sample_dir):
        os.makedirs(config.sample_dir)
    if not os.path.exists(config.result_dir):
        os.makedirs(config.result_dir)

    # Data loader.
    celeba_loader = None
    rafd_loader = None
    custom_loader = None

    if config.dataset in ['CelebA', 'Both']:
        celeba_loader = get_loader(config.celeba_image_dir, config.attr_path, config.selected_attrs,
                                   config.celeba_crop_size, config.image_size, config.batch_size,
                                   'CelebA', config.mode, config.num_workers)
    if config.dataset in ['RaFD', 'Both']:
        rafd_loader = get_loader(config.rafd_image_dir, None, None,
                                 config.rafd_crop_size, config.image_size, config.batch_size,
                                 'RaFD', config.mode, config.num_workers)
    if config.dataset == 'custom':
        custom_loader = get_loader(config.custom_image_dir, None, None,
                                    config.custom_crop_size, config.image_size, config.batch_size,
                                    'custom', config.mode, config.num_workers)
    

    # Solver for training and testing StarGAN.
    solver = Solver(celeba_loader, rafd_loader, custom_loader, config)

    if config.mode == 'train':
        if config.dataset in ['CelebA', 'RaFD', 'custom']:
            solver.train()
        elif config.dataset in ['Both']:
            solver.train_multi()
    elif config.mode == 'test':
        if config.dataset in ['CelebA', 'RaFD', 'custom']:
            solver.test()
        elif config.dataset in ['Both']:
            solver.test_multi()


'''class CustomArguments:

    def __init__(self):

        self.c_dim = 8
        #self.c_dim = 5
        self.c2_dim = 11
        self.custom_crop_size = 128
        self.celeba_crop_size = 178
        self.image_size = 128
        self.g_conv_dim = 64
        self.d_conv_dim = 64
        self.g_repeat_num = 6
        self.d_repeat_num = 6
        self.lambda_cls = 1
        self.lambda_rec = 10
        self.lambda_gp = 10

        self.dataset = 'custom'
        #self.dataset='CelebA'
        self.batch_size = 16
        #self.num_iters = 200000
        self.num_iters = 2
        #self.num_iters_decay = 100000
        self.num_iters_decay = 2
        self.g_lr = 0.0001
        self.d_lr = 0.0001
        self.n_critic = 5
        self.beta1 = 0.5
        self.beta2 = 0.999
        self.resume_iters = None
        self.selected_attrs = ['angry', 'calm', 'disgust', 'fear', 'happy', 'neutral', 'sad', 'surprise']
        #self.selected_attrs = ['Black_Hair', 'Blond_Hair', 'Brown_Hair', 'Male', 'Young']

        #self.test_iters = 200000
        self.test_iters = 4
        
        self.num_workers = 1
        self.mode = 'train'
        #self.mode = 'test'
        self.use_tensorboard = True

        self.custom_image_dir = 'D:/Paylasim/segmented_datasets/custom_spec/train_spec'
        self.celeba_image_dir = 'data/celeba/images'
        self.attr_path = 'data/celeba/list_attr_celeba.txt'
        self.log_dir = 'stargan/logs'
        #self.model_save_dir = 'stargan/models'
        self.model_save_dir = 'stargan_celeba_128/models'
        self.sample_dir = 'stargan/samples'
        #self.result_dir = 'stargan/results'
        self.result_dir = 'stargan_celeba_128/results'
        
        # Step size.
        #self.log_step = 10
        #self.sample_step = 1000
        #self.model_save_step = 10000
        #self.lr_update_step = 1000
        self.log_step = 1
        self.sample_step = 1
        self.model_save_step = 1
        self.lr_update_step = 1'''



if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    # Model configuration.
    parser.add_argument('--c_dim', type=int, default=8, help='dimension of domain labels (1st dataset)')
    parser.add_argument('--c2_dim', type=int, default=8, help='dimension of domain labels (2nd dataset)')
    parser.add_argument('--celeba_crop_size', type=int, default=178, help='crop size for the CelebA dataset')
    parser.add_argument('--rafd_crop_size', type=int, default=256, help='crop size for the RaFD dataset')
    parser.add_argument('--custom_crop_size', type=int, default=128, help='crop size for general dataset')
    parser.add_argument('--image_size', type=int, default=128, help='image resolution')
    parser.add_argument('--g_conv_dim', type=int, default=64, help='number of conv filters in the first layer of G')
    parser.add_argument('--d_conv_dim', type=int, default=64, help='number of conv filters in the first layer of D')
    parser.add_argument('--g_repeat_num', type=int, default=6, help='number of residual blocks in G')
    parser.add_argument('--d_repeat_num', type=int, default=6, help='number of strided conv layers in D')
    parser.add_argument('--lambda_cls', type=float, default=1, help='weight for domain classification loss')
    parser.add_argument('--lambda_rec', type=float, default=10, help='weight for reconstruction loss')
    parser.add_argument('--lambda_gp', type=float, default=10, help='weight for gradient penalty')
    
    # Training configuration.
    parser.add_argument('--dataset', type=str, default='custom', choices=['CelebA', 'RaFD', 'Both', 'custom'])
    parser.add_argument('--batch_size', type=int, default=16, help='mini-batch size')
    parser.add_argument('--num_iters', type=int, default=200000, help='number of total iterations for training D')
    parser.add_argument('--num_iters_decay', type=int, default=100000, help='number of iterations for decaying lr')
    parser.add_argument('--g_lr', type=float, default=0.0001, help='learning rate for G')
    parser.add_argument('--d_lr', type=float, default=0.0001, help='learning rate for D')
    parser.add_argument('--n_critic', type=int, default=5, help='number of D updates per each G update')
    parser.add_argument('--beta1', type=float, default=0.5, help='beta1 for Adam optimizer')
    parser.add_argument('--beta2', type=float, default=0.999, help='beta2 for Adam optimizer')
    parser.add_argument('--resume_iters', type=int, default=None, help='resume training from this step')
    #parser.add_argument('--selected_attrs', '--list', nargs='+', help='selected attributes for the CelebA dataset',
    #                    default=['Black_Hair', 'Blond_Hair', 'Brown_Hair', 'Male', 'Young'])
    parser.add_argument('--selected_attrs', '--list', nargs='+', help='selected attributes for dataset',
                        default=['angry', 'calm', 'disgust', 'fear', 'happy', 'neutral', 'sad', 'surprise'])

    # Test configuration.
    parser.add_argument('--test_iters', type=int, default=200000, help='test model from this step')

    # Miscellaneous.
    parser.add_argument('--num_workers', type=int, default=1)
    parser.add_argument('--mode', type=str, default='train', choices=['train', 'test'])
    parser.add_argument('--use_tensorboard', type=str2bool, default=True)

    # Directories.
    parser.add_argument('--celeba_image_dir', type=str, default='data/celeba/images')
    parser.add_argument('--attr_path', type=str, default='data/celeba/list_attr_celeba.txt')
    parser.add_argument('--rafd_image_dir', type=str, default='data/RaFD/train')
    parser.add_argument('--custom_image_dir', type=str, default='D:/Paylasim/segmented_datasets/custom_spec/train_spec')
    parser.add_argument('--log_dir', type=str, default='stargan/logs')
    parser.add_argument('--model_save_dir', type=str, default='stargan/models')
    parser.add_argument('--sample_dir', type=str, default='stargan/samples')
    parser.add_argument('--result_dir', type=str, default='stargan/results')

    # Step size.
    parser.add_argument('--log_step', type=int, default=10)
    parser.add_argument('--sample_step', type=int, default=1000)
    parser.add_argument('--model_save_step', type=int, default=10000)
    parser.add_argument('--lr_update_step', type=int, default=1000)

    config = parser.parse_args()

    #config = CustomArguments()

    print(config)
    main(config)