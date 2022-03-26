#!/usr/bin/env python3
"""
Code is referred from https://github.com/bioinf-jku/TTUR to use PyTorch instead of Tensorflow
"""
import os
import pathlib
import time
from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter

import torch
import numpy as np
from scipy import linalg
from torch.autograd import Variable
from torch.nn.functional import adaptive_avg_pool2d
import torchvision.transforms as transforms
from tqdm import tqdm

import numpy as np
import torchvision.utils as vutils
from inception import InceptionV3
import torch.utils.data
from PIL import Image
from torch.utils import data
import img_data as img_data
import sys
# dir_path = (os.path.abspath(os.path.join(os.path.realpath(__file__), './.')))
# sys.path.append(dir_path)

root_path = "/home/zhaoqike/SSA_GAN"
sys.path.append(root_path)
print(sys.path)

from miscc.utils import mkdir_p

parser = ArgumentParser(formatter_class=ArgumentDefaultsHelpFormatter)
parser.add_argument('--model_path', type=str,
                    default='../data/Models/pretrained_cnn/inception_v3_google-1a9a5a14.pth')
parser.add_argument('--batch_size', type=int, default=64,
                    help='Batch size to use')
parser.add_argument('--dims', type=int, default=2048,
                    choices=list(InceptionV3.BLOCK_INDEX_BY_DIM),
                    help=('Dimensionality of Inception features to use. '
                          'By default, uses pool3 features'))
parser.add_argument('-c', '--gpu', default='0', type=str,
                    help='GPU to use (leave blank for CPU only)')
parser.add_argument('--compared_path', type=str, default='../data/Models/IS_model/bird_val.npz',
                    help="compared image folder path")
parser.add_argument('--eval_image_folder', type=str, default='../data/SSA_GAN_OUT/12_10/multi_whole/GenImage')


class MeasureFID:

    def __init__(self, args):

        self.repeat_times = 5
        self.select_epoch = [epoch for epoch in range(550, 700, 10)]

        self.model_path = args.model_path
        self.eval_image_folders = args.eval_image_folders
        self.folders = self.prepare_folders()

    def prepare_folders(self):
        select_epoch = self.select_epoch
        eval_image_folder = self.eval_image_folders
        re_image_folders = []
        for epoch in select_epoch:
            for re_ix in range(self.repeat_times):
                folder = os.path.join(eval_image_folder, "Epoch_%d_%d" % (epoch, re_ix))
                re_image_folders.append(folder)
        return re_image_folders

    def calculate_fid(self, compared_path, eval_image_paths, batch_size, cuda, dims):
        """
        paras: the compared_path is from origin dataset
        """
        block_idx = InceptionV3.BLOCK_INDEX_BY_DIM[dims]
        model = InceptionV3(self.model_path, [block_idx])

        if cuda:
            model.cuda()

        results = []
        m1, s1 = self.compute_statistics_of_path(compared_path, model, batch_size, dims, cuda)


        all_ts = 0
        for cur_image_path in select_image_paths:
            start_ts = time.time()
            if not os.path.exists(cur_image_path):
                raise RuntimeError('Invalid path: %s' % cur_image_path)
            m2, s2 = _compute_statistics_of_path(cur_image_path, model, batch_size, dims, cuda)
            fid_value = calculate_frechet_distance(m1, s1, m2, s2)
            print(fid_value)
            results.append(fid_value)
            end_ts = time.time()
            print('''%s is finished and costs time: %.2f\n''' % (cur_image_path, end_ts - start_ts))
            all_ts += (end_ts - start_ts)

        print("the all cost time is %.2f" % all_ts)
        print(results)

    def compute_statistics_of_path(self, given_path, model, batch_size, dims, cuda):

        if given_path.endswith('.npz'):
            f = np.load(given_path)
            mu, sigma = f['mu'][:], f['sigma'][:]
            f.close()

        else:
            transform = transforms.Compose([
                transforms.Resize((299, 299)),
                transforms.ToTensor(),
            ])

            dataset = img_data.Dataset(given_path, None, transform)
            data_loader = torch.utils.data.DataLoader(dataset=dataset, batch_size=batch_size, shuffle=False,
                                                     drop_last=True, num_workers=8)

            active_val = self.calculate_activation_statistics(data_loader, model, batch_size, dims, cuda)
            mu = np.mean(active_val, axis=0)
            sigma = np.cov(active_val, rowvar=False)

        return mu, sigma

    @staticmethod
    def calculate_activation_statistics(images, model, batch_size=64, dims=2048, cuda=False, verbose=True):
        """Calculates the activations of the pool_3 layer for all images.

        Params:
        -- images      : Numpy array of dimension (n_images, 3, hi, wi). The values
                         must lie between 0 and 1.
        -- model       : Instance of inception model
        -- batch_size  : the images numpy array is split into batches with
                         batch size batch_size. A reasonable batch size depends
                         on the hardware.
        -- dims        : Dimensionality of features returned by Inception
        -- cuda        : If set to True, use GPU
        -- verbose     : If set to True and parameter out_step is given, the number
                         of calculated batches is reported.
        Returns:
        -- A numpy array of dimension (num images, dims) that contains the
           activations of the given tensor when feeding inception with the
           query tensor.
        """
        model.eval()
        d0 = images.__len__() * batch_size
        if batch_size > d0:
            print(('Warning: batch size is bigger than the data size. '
                   'Setting batch size to data size'))
            batch_size = d0

        n_batches = d0 // batch_size
        n_used_imgs = n_batches * batch_size

        pred_arr = np.empty((n_used_imgs, dims))
        #for i in range(n_batches):
        for i, batch in enumerate(images):
            start = i * batch_size
            end = start + batch_size
            if cuda:
                batch = batch.cuda()

            pred = model(batch)[0]

            # If model output is not scalar, apply global spatial average pooling.
            # This happens if you choose a dimensionality not equal 2048.
            if pred.shape[2] != 1 or pred.shape[3] != 1:
                pred = adaptive_avg_pool2d(pred, output_size=(1, 1))

            pred_arr[start:end] = pred.cpu().data.numpy().reshape(batch_size, -1)

        if verbose:
            print(' done')

        return pred_arr

    @staticmethod
    def calculate_frechet_distance(mu1, sigma1, mu2, sigma2, eps=1e-6):
        """Numpy implementation of the Frechet Distance.
        The Frechet distance between two multivariate Gaussians X_1 ~ N(mu_1, C_1)
        and X_2 ~ N(mu_2, C_2) is
                d^2 = ||mu_1 - mu_2||^2 + Tr(C_1 + C_2 - 2*sqrt(C_1*C_2)).

        Stable version by Dougal J. Sutherland.

        Params:
        -- mu1   : Numpy array containing the activations of a layer of the
                   inception net (like returned by the function 'get_predictions')
                   for generated samples.
        -- mu2   : The sample mean over activations, precalculated on an
                   representive data set.
        -- sigma1: The covariance matrix over activations for generated samples.
        -- sigma2: The covariance matrix over activations, precalculated on an
                   representive data set.

        Returns:
        --   : The Frechet Distance.
        """

        mu1 = np.atleast_1d(mu1)
        mu2 = np.atleast_1d(mu2)

        sigma1 = np.atleast_2d(sigma1)
        sigma2 = np.atleast_2d(sigma2)

        assert mu1.shape == mu2.shape, \
            'Training and test mean vectors have different lengths'
        assert sigma1.shape == sigma2.shape, \
            'Training and test covariances have different dimensions'

        diff = mu1 - mu2

        # Product might be almost singular
        covmean, _ = linalg.sqrtm(sigma1.dot(sigma2), disp=False)
        if not np.isfinite(covmean).all():
            msg = ('fid calculation produces singular product; '
                   'adding %s to diagonal of cov estimates') % eps
            print(msg)
            offset = np.eye(sigma1.shape[0]) * eps
            covmean = linalg.sqrtm((sigma1 + offset).dot(sigma2 + offset))

        # Numerical error might give slight imaginary component
        if np.iscomplexobj(covmean):
            if not np.allclose(np.diagonal(covmean).imag, 0, atol=1e-3):
                m = np.max(np.abs(covmean.imag))
                raise ValueError('Imaginary component {}'.format(m))
            covmean = covmean.real

        tr_covmean = np.trace(covmean)

        return (diff.dot(diff) + np.trace(sigma1) +
                np.trace(sigma2) - 2 * tr_covmean)

    """
    The following functions is used to measure dataset's FID.
    In this project, the flower.npy is evaluated by us, and bird.npy and coco.npy are provided by AttnGAN 
    """

    def gen_dataset_imgs(self, pickle_path, saving_dir, sampling_nums=30000, batch_size=48):
        # we get 30K images saving in the saving_dir
        # then to calculate the .npz file

        imsize = 299
        nWorks = 6
        mkdir_p(saving_dir)
        image_transform = transforms.Compose([transforms.Resize(int(imsize * 76 / 64)),
                                              transforms.RandomCrop(imsize),
                                              transforms.RandomHorizontalFlip(),
                                              transforms.ToTensor()])

        # dataset = img_data.Dataset(path, transforms.Compose([
        #     transforms.Resize((299, 299)),
        #     transforms.ToTensor(),
        # ]))

        dataset = img_data.Dataset(path=dir_path, pickle_path=pickle_path, transform=image_transform)

        dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size,
                drop_last=True, shuffle=True, num_workers=nWorks)

        data_iter = iter(dataloader)
        len_data_iters = len(data_iter)
        img_cnt = 0
        continue_sampling = True

        while continue_sampling:
            data_iter = iter(dataloader)
            for iters in tqdm(range(len_data_iters)):
                data = data_iter.next()

                for i in range(batch_size):
                    img_cnt += 1
                    if img_cnt >= sampling_nums:
                        continue_sampling = False
                        break

                    key = str(img_cnt)
                    image_save_path = os.path.join(saving_dir, "%s.jpg" % key)
                    vutils.save_image(data[i], image_save_path, scale_each=True, normalize=True)



def gen_npz_file(saving_path, model_path, select_image_path, cuda, dims, batch_size):

    block_idx = InceptionV3.BLOCK_INDEX_BY_DIM[dims]
    model = InceptionV3(model_path, [block_idx])
    if cuda:
        model.cuda()

    # path[0] is str that represents the dataset images
    if not os.path.join(select_image_path):
        raise RuntimeError('Invalid path: %s' % select_image_path)

    mu, sigma = _compute_statistics_of_path(select_image_path, model, batch_size, dims, cuda)
    np.savez(saving_path, mu=mu, sigma=sigma)

    if saving_path.endswith('.npz'):
        f = np.load(saving_path)
        m, s = f['mu'][:], f['sigma'][:]
        f.close()

    print("saving in %s" % saving_path)

def gen_flower_fid(args):

    dir_path = '../data/flowers'
    pickle_path = os.path.join(dir_path, 'test/class_filenames.pickle')
    img_saving_dir = os.path.join(dir_path, 'test/sampling_test_images')
    npz_saving_dir = '../data/Models/IS_model/flower_val.npz'

    # gen_dataset_imgs(dir_path, pickle_path, img_saving_dir)
    gen_npz_file(npz_saving_dir, None, img_saving_dir, args.gpu, args.dims, args.batch_size)




if __name__ == '__main__':
    args = parser.parse_args()
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu

    img_folders = prepare_folders()
    print(img_folders)


    # Method 2.
    # image_paths = []
    # dirs = os.listdir(args.eval_image_folder)
    # for file_name in dirs:
    #     image_folder = os.path.join(args.eval_image_folder, file_name)
    #     image_paths.append(image_folder)

    calculate_fid_given_paths(None, args.compared_path, img_folders, args.batch_size, args.gpu, args.dims)