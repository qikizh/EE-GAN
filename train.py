from __future__ import print_function
import multiprocessing

import os
import sys
import random
import pprint
import datetime
import dateutil.tz
import argparse

import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.backends.cudnn as cudnn
import torchvision.transforms as transforms
from tensorboardX import SummaryWriter
import numpy as np
from tqdm import tqdm
import torch.nn.functional as F

from miscc.utils import mkdir_p, save_text_results, save_img_results
from miscc.config import cfg, cfg_from_file
from miscc.DAMSM_losses import words_loss, sent_loss
from sync_batchnorm import DataParallelWithCallback

from datasets import TextDataset
from models import Dis64, Dis128, Dis256, Gen, ATTR_Enhance
from DAMSM import RNN_ENCODER, CNN_ENCODER
import shutil

dir_path = (os.path.abspath(os.path.join(os.path.realpath(__file__), './.')))
sys.path.append(dir_path)
UPDATE_INTERVAL = 100
multiprocessing.set_start_method('spawn', True)

def parse_args():
    parser = argparse.ArgumentParser(description='Train a EE-GAN network')
    parser.add_argument('--cfg', dest='cfg_file',
                        help='optional config file',
                        default='cfg/bird.yml', type=str)
    parser.add_argument('--gpu', dest='gpu_ids', type=str, default="0")
    parser.add_argument('--output_dir', dest='output_dir',
                        help='the path to save models and images',
                        default='../EE-GAN', type=str)
    parser.add_argument('--debug_output_dir', dest='debug_output_dir',
                        help='the path to save models and images in debug mode',
                        default='Debug', type=str)
    parser.add_argument('--debug', action="store_true", help='using debug mode')
    parser.add_argument('--manualSeed', type=int, help='manual seed', default=3407)
    parser.add_argument('--batch_size', type=int, help='using batch size', default=32)
    # for ablation study
    parser.add_argument('--class_coe', type=float, help='using batch size', default=10)
    parser.add_argument('--sim_coe', type=float, help='using batch size', default=0.05)
    args = parser.parse_args()
    return args

def prepare_data(data, device):

    rev_basic, rev_attrs, rev_unpair = data

    # [image, cap, cap_len, cls_id, key], ret_attrs, ret_unpair

    [imgs, caps, cap_lens, cls_ids, keys] = rev_basic
    [attrs, attr_nums, attrs_len] = rev_attrs
    [unpair_caps, unpair_cap_lens, unpair_cls_ids] = rev_unpair

    real_imgs = []
    for i in range(len(imgs)):
        real_imgs.append(Variable(imgs[i].to(device)))

    caps = caps.squeeze().to(device)
    cap_lens = Variable(cap_lens).to(device)
    cls_ids = cls_ids.numpy()

    unpair_caps = unpair_caps.squeeze().to(device)
    unpair_cap_lens = Variable(unpair_cap_lens).to(device)
    unpair_cls_ids = unpair_cls_ids.numpy()

    # bs x (3x5)
    attrs = attrs.squeeze().to(device)
    # bs x 3
    attrs_len = attrs_len.squeeze()

    return [real_imgs, caps, cap_lens, cls_ids,
            attrs, attr_nums, attrs_len,
            unpair_caps, unpair_cap_lens, unpair_cls_ids,
            keys]

def prepare_labels(batch_size, device):
    real_labels = Variable(torch.FloatTensor(batch_size).fill_(1))
    fake_labels = Variable(torch.FloatTensor(batch_size).fill_(0))
    match_labels = Variable(torch.LongTensor(range(batch_size)))
    real_labels = real_labels.to(device)
    fake_labels = fake_labels.to(device)
    match_labels = match_labels.to(device)
    return real_labels, fake_labels, match_labels

def prepare_class_labels(batch_size, class_num, class_ids, device):
    class_labels = torch.zeros(batch_size, class_num).to(device)
    for i, idx in enumerate(class_ids):
        class_labels[i][idx - 1] = 1
    return class_labels

class Trainer(object):

    def __init__(self, output_dir, args):

        self.output_dir = output_dir
        self.model_dir = os.path.join(output_dir, 'Model')
        self.image_dir = os.path.join(output_dir, 'Image')
        mkdir_p(self.image_dir, rm_exist=False)
        mkdir_p(self.model_dir, rm_exist=False)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.args = args
        self.debug = args.debug
        self.disc_class = cfg.TRAIN.USE_CLASS
        self.class_nums = cfg.TRAIN.CLASS_NUM

        self.batch_size = cfg.TRAIN.BATCH_SIZE
        self.max_attr_nums = cfg.TEXT.MAX_ATTR_NUM
        self.max_attr_len = cfg.TEXT.MAX_ATTR_LEN
        self.batch_attr_nums = self.batch_size * self.max_attr_nums

        self.data_loader, n_words, self.ixtoword = self.load_dataloader()
        self.netG, self.attr_enhance, self.netsD, self.image_encoder, self.text_encoder \
            = self.load_networks(n_words, self.device)

        self.optimizerG, self.optimizerDs = \
            self.load_optimizers(self.netG, self.netsD, self.attr_enhance)

        self.start_epoch = 1
        self.max_epoch = cfg.TRAIN.MAX_EPOCH + 1

        self.sample_sent_emb, self.sample_attrs_emb = self.prepare_sampling(self.data_loader)

        writer_path = os.path.join(self.output_dir, 'writer')
        mkdir_p(writer_path, rm_exist=True)
        self.writer = SummaryWriter(writer_path)

        self.iters_cnt = 0
        self.d_class_coe = self.g_class_coe = args.class_coe
        self.DAMSM_coe = args.sim_coe

        print("using class_coe: %f, DAMSM_coe: %f" % (self.g_class_coe, self.DAMSM_coe))

    def train(self):

        batch_size = self.batch_size
        device = self.device
        class_nums = self.class_nums

        real_labels, fake_labels, match_labels = prepare_labels(self.batch_size, self.device)
        class_labels = None

        # in debug mode, the tqdm
        # for epoch in range(self.start_epoch, self.max_epoch):
        for epoch in tqdm(range(self.start_epoch, self.max_epoch)):
            data_iter = iter(self.data_loader)
            #for iters in range(len(data_iter)):
            for iters in tqdm(range(len(data_iter))):
                data = data_iter.next()

                imgs, caps, cap_lens, cls_ids, attrs, attr_nums, attrs_len, \
                unpair_caps, unpair_cap_lens, unpair_cls_ids, keys = prepare_data(data, device)

                # step 1. condition input preparation
                with torch.no_grad():
                    hidden = self.text_encoder.init_hidden(batch_size)
                    words_emb, sent_emb = self.text_encoder(caps, cap_lens, hidden)
                    words_emb, sent_emb = words_emb.detach(), sent_emb.detach()

                    attrs_emb = list()
                    for i in range(self.max_attr_nums):
                        one_attr = attrs[:,i,:].squeeze(-1)
                        one_attr_len = attrs_len[:, i].squeeze(-1)
                        _, one_attr_emb = self.text_encoder(one_attr, one_attr_len, hidden)
                        attrs_emb.append(one_attr_emb)
                    attrs_emb = torch.stack(attrs_emb, dim=1)
                    attrs_emb = attrs_emb.detach()

                    _, unpair_sent_emb = self.text_encoder(unpair_caps, unpair_cap_lens, hidden)
                    unpair_sent_emb = unpair_sent_emb.detach()

                if self.disc_class:
                    class_labels = prepare_class_labels(batch_size, class_nums, cls_ids, device)

                noise = torch.randn(batch_size, 100)
                noise = noise.to(device)

                # step 2. generation
                _, attn_attr_emb = self.attr_enhance(sent_emb, attrs_emb)
                attn_attr_emb = self.attr_enhance.module.attr_merge(attn_attr_emb)
                fake_imgs = self.netG(noise, sent_emb, attn_attr_emb)

                # step 3. backward supervision
                if iters % UPDATE_INTERVAL == 0:
                    iter_rec = True
                    self.iters_cnt += 1
                else:
                    iter_rec = False

                self.d_update(imgs, fake_imgs, sent_emb, unpair_sent_emb, class_labels, iter_rec)
                self.g_update(fake_imgs, sent_emb, words_emb, attn_attr_emb, cls_ids, batch_size, match_labels,
                              cap_lens, class_labels, iter_rec)

                # print("step")

            self.save_images(epoch)
            self.save_model(epoch)

    def load_networks(self, n_words, device):
        """
        the DataParallelWithCallback is from sync_batchnorm (local)
        the nn.DataParallel is provided by torch
        """

        netG = Gen(cfg.GAN.GF_DIM, 100).to(device)
        netG = DataParallelWithCallback(netG)
        attr_enhance = ATTR_Enhance().to(device)
        attr_enhance = nn.DataParallel(attr_enhance)

        netD64 = Dis64(cfg.GAN.DF_DIM).to(device)
        netD128 = Dis128(cfg.GAN.DF_DIM).to(device)
        netD256 = Dis256(cfg.GAN.DF_DIM, self.disc_class, self.class_nums).to(device)
        netDs = [netD64, netD128, netD256]
        netDs = [nn.DataParallel(netD) for netD in netDs]

        """
        The pre-trained Auxiliary Image-Enc and Text-Enc, which are same with AttnGAN
        """
        text_encoder = RNN_ENCODER(n_words, nhidden=cfg.TEXT.EMBEDDING_DIM)
        state_dict = torch.load(cfg.TEXT.DAMSM_NAME, map_location=lambda storage, loc: storage)
        text_encoder.load_state_dict(state_dict)
        text_encoder.cuda()
        for p in text_encoder.parameters():
            p.requires_grad = False
        text_encoder.eval()

        image_encoder = CNN_ENCODER(cfg.TEXT.EMBEDDING_DIM)
        img_encoder_path = cfg.TEXT.DAMSM_NAME.replace('text_encoder', 'image_encoder')
        state_dict = torch.load(img_encoder_path, map_location=lambda storage, loc: storage)
        image_encoder.load_state_dict(state_dict)
        image_encoder.cuda()
        for p in image_encoder.parameters():
            p.requires_grad = False
        image_encoder.eval()

        return netG, attr_enhance, netDs, image_encoder, text_encoder

    @staticmethod
    def load_optimizers(netG, netDs, attr_enhance):
        # method 1.
        from itertools import chain
        EG_params = chain(netG.parameters(), attr_enhance.parameters())
        optimizerG = torch.optim.Adam(EG_params, lr=0.0001, betas=(0.0, 0.9))

        optimizerDs = []
        for i in range(len(netDs)):
            optD = torch.optim.Adam(netDs[i].parameters(), lr=0.0004, betas=(0.0, 0.9))
            optimizerDs.append(optD)
        return optimizerG, optimizerDs

    def load_dataloader(self):
        # 64 * 4
        imsize = cfg.TREE.BASE_SIZE * (2 ** (cfg.TREE.BRANCH_NUM-1))
        batch_size = self.batch_size
        image_transform = transforms.Compose([
            transforms.Resize(int(imsize * 76 / 64)),
            transforms.RandomCrop(imsize),
            transforms.RandomHorizontalFlip()])

        dataset = TextDataset(data_dir=cfg.DATA_DIR, dataset_name=cfg.DATASET_NAME,
                              transform=image_transform)
        nWorks = 0 if self.debug else (batch_size // 4)
        data_loader = torch.utils.data.DataLoader(
            dataset, batch_size=batch_size, drop_last=True, shuffle=True, num_workers=nWorks)

        return data_loader, dataset.n_words, dataset.ixtoword

    def prepare_sampling(self, data_loader):
        """
        The fix text-image pair is provided during training
        """
        [real_imgs, caps, cap_lens, cls_ids,
         attrs, attr_nums, attrs_len, _, _, _, _] = prepare_data(next(iter(data_loader)), self.device)
        # unpair_caps, unpair_cap_lens, unpair_cls_ids, keys

        txt_save_path = os.path.join(self.image_dir, 'sampling_text.txt')
        save_text_results(caps, cap_lens, self.ixtoword, txt_save_path, attrs, attr_nums, attrs_len)
        save_img_results(real_imgs,prefix='sample_image', image_dir=self.image_dir)

        with torch.no_grad():
            hidden = self.text_encoder.init_hidden(self.batch_size)
            _, sent_emb = self.text_encoder(caps, cap_lens, hidden)

            attrs_emb = list()
            for i in range(self.max_attr_nums):
                one_attr = attrs[:, i, :].squeeze(-1)
                one_attr_len = attrs_len[:, i].squeeze(-1)
                _, one_attr_emb = self.text_encoder(one_attr, one_attr_len, hidden)
                attrs_emb.append(one_attr_emb)
            attrs_emb = torch.stack(attrs_emb, dim=1)

        return sent_emb, attrs_emb

    def save_model(self, epoch):
        if epoch == 1 or (epoch >= cfg.TRAIN.WARMUP_EPOCHS and epoch % cfg.TRAIN.GSAVE_INTERVAL == 0):
            torch.save(self.netG.state_dict(), '%s/netG_%d.pth' % (self.model_dir, epoch),
                       _use_new_zipfile_serialization=False)
            torch.save(self.attr_enhance.state_dict(), '%s/attr_enhance_%d.pth' % (self.model_dir, epoch),
                       _use_new_zipfile_serialization=False)
            print('Save Gen model.')

        if epoch == 1 or (epoch >= cfg.TRAIN.WARMUP_EPOCHS and epoch % cfg.TRAIN.DSAVE_INTERVAL == 0):
            for i in range(len(self.netsD)):
                torch.save(self.netsD[i].state_dict(), '%s/netD_%d.pth' % (self.model_dir, i))
                print('Save Dis model.')

    def save_images(self, epoch):
        with torch.no_grad():
            noise = torch.randn(self.batch_size, 100)
            noise = noise.to(self.device)

            _, attn_attr_emb = self.attr_enhance(self.sample_sent_emb, self.sample_attrs_emb)
            attn_attr_emb = self.attr_enhance.module.attr_merge(attn_attr_emb)
            fake_imgs = self.netG(noise, self.sample_sent_emb, attn_attr_emb)

        # batch_imgs, prefix, image_dir
        save_img_results(fake_imgs, prefix='epoch_%d' % epoch, image_dir=self.image_dir)

    """
    The following functions are used to calculate the loss metrics and achieve backward.    
    """
    @staticmethod
    def d_loss(imgs, fake_imgs, sent_emb, wrong_sent_emb, netD):

        # real cond
        real_features = netD(imgs)
        real_out = netD.module.COND_DNET(real_features, sent_emb)
        errD_real = torch.nn.ReLU()(1.0 - real_out).mean()

        # unpair cond
        unpair_out = netD.module.COND_DNET(real_features, wrong_sent_emb)
        errD_mismatch = torch.nn.ReLU()(1.0 + unpair_out).mean()

        # fake cond
        fake_features = netD(fake_imgs.detach())
        fake_out = netD.module.COND_DNET(fake_features, sent_emb)
        errD_fake = torch.nn.ReLU()(1.0 + fake_out).mean()

        return errD_real, errD_fake, errD_mismatch

    @staticmethod
    def d_loss_class(imgs, fake_imgs, sent_emb, unpair_sent_emb, class_labels, netD):

        # class loss only for 256 scale currently
        real_feature = netD(imgs)

        real_sent_out, real_class_out = netD.module.COND_DNET(real_feature, sent_emb)
        errD_real = torch.nn.ReLU()(1.0 - real_sent_out).mean()
        errD_real_class = F.binary_cross_entropy_with_logits(real_class_out, class_labels)

        # wrong cond
        unpair_sent_out, unpair_class_out = netD.module.COND_DNET(real_feature, unpair_sent_emb)
        errD_mismatch = torch.nn.ReLU()(1.0 + unpair_sent_out).mean()
        errD_mismatch_class = F.binary_cross_entropy_with_logits(unpair_class_out, class_labels)

        # fake cond
        fake_features = netD(fake_imgs.detach())
        fake_sent_out, fake_class_out = netD.module.COND_DNET(fake_features, sent_emb)
        errD_fake = torch.nn.ReLU()(1.0 + fake_sent_out).mean()
        errD_fake_class = F.binary_cross_entropy_with_logits(fake_class_out, class_labels)

        return errD_real, errD_fake, errD_mismatch, errD_real_class, errD_fake_class, errD_mismatch_class

    @staticmethod
    def MA_gradient_penalty(imgs, sent_emb, netD, disc_class):
        interpolated = imgs.data.requires_grad_()
        sent_inter = sent_emb.data.requires_grad_()
        features = netD(interpolated)

        if disc_class:
            out, _ = netD.module.COND_DNET(features, sent_inter)
        else:
            out = netD.module.COND_DNET(features, sent_inter)

        grads = torch.autograd.grad(outputs=out,
                                    inputs=(interpolated, sent_inter),
                                    grad_outputs=torch.ones(out.size()).cuda(),
                                    retain_graph=True,
                                    create_graph=True,
                                    only_inputs=True)

        grad0 = grads[0].view(grads[0].size(0), -1)
        grad1 = grads[1].view(grads[1].size(0), -1)
        grad = torch.cat((grad0, grad1), dim=1)
        grad_l2norm = torch.sqrt(torch.sum(grad ** 2, dim=1))
        d_loss_gp = torch.mean(grad_l2norm ** 6)
        d_loss = 2.0 * d_loss_gp
        return d_loss

    @staticmethod
    def g_loss_class(fake_imgs, sent_emb, class_labels, netD):
        features = netD(fake_imgs)
        fake_sent, fake_class = netD.module.COND_DNET(features, sent_emb)
        errG = - fake_sent.mean()
        errG_class = F.binary_cross_entropy_with_logits(fake_class, class_labels)
        return errG, errG_class

    @staticmethod
    def g_loss(fake_imgs, sent_emb, netD):
        features = netD(fake_imgs)
        fake_sent = netD.module.COND_DNET(features, sent_emb)
        errG = - fake_sent.mean()
        return errG

    @staticmethod
    def DAMSM_loss(fake_imgs, sent_emb, words_embs, attrs_emb, class_ids, batch_size, match_labels, cap_lens,
                   image_encoder):

        class_ids = torch.LongTensor(class_ids)
        region_features, cnn_code = image_encoder(fake_imgs)
        s_loss0, s_loss1 = sent_loss(cnn_code, sent_emb, match_labels, class_ids, batch_size)
        s_loss = (s_loss0 + s_loss1) * cfg.TRAIN.SMOOTH.LAMBDA

        w_loss0, w_loss1, _ = words_loss(region_features, words_embs, match_labels, cap_lens, class_ids, batch_size)
        w_loss = (w_loss0 + w_loss1) * cfg.TRAIN.SMOOTH.LAMBDA

        # the part is considered later
        a_loss0, a_loss1 = sent_loss(cnn_code, attrs_emb, match_labels, class_ids, batch_size)
        a_loss = (a_loss0 + a_loss1) * cfg.TRAIN.SMOOTH.LAMBDA

        return w_loss, s_loss, a_loss

    def d_update(self, imgs, fake_imgs, sent_emb, unpair_sent_emb, class_labels, iter_rec):

        for i in range(len(self.netsD)):
            real_img, fake_img, netD, optimizerD = imgs[i], fake_imgs[i], self.netsD[i], self.optimizerDs[i]
            disc_class = self.disc_class and i == 2
            if disc_class:
                errD_real, errD_fake, errD_unpair, errD_real_class, errD_fake_class, errD_unpair_class = \
                    self.d_loss_class(real_img, fake_img, sent_emb, unpair_sent_emb, class_labels, netD)
                d_loss = errD_real + (errD_fake + errD_unpair) / 2.0 + \
                         (errD_real_class + errD_fake_class + errD_unpair_class) / 3.0 * self.d_class_coe
            else:
                errD_real, errD_fake, errD_unpair = \
                    self.d_loss(real_img, fake_img, sent_emb, unpair_sent_emb, netD)
                d_loss = errD_real + (errD_fake + errD_unpair) / 2.0

            optimizerD.zero_grad()
            d_loss.backward()
            optimizerD.step()

            d_loss_gp = self.MA_gradient_penalty(real_img, sent_emb, netD, disc_class)
            optimizerD.zero_grad()
            d_loss_gp.backward()
            optimizerD.step()

            if iter_rec:
                self.writer.add_scalar('errD_%d/real_sent' % i,  errD_real, self.iters_cnt)
                self.writer.add_scalar('errD_%d/fake_sent' % i, errD_fake, self.iters_cnt)
                self.writer.add_scalar('errD_%d/unpair_sent' % i, errD_unpair, self.iters_cnt)
                self.writer.add_scalar('errD_%d/d_loss_gp' % i, d_loss_gp, self.iters_cnt)
                if disc_class:
                    self.writer.add_scalar('errD_%d/real_class' % i, errD_real_class, self.iters_cnt)
                    self.writer.add_scalar('errD_%d/fake_class' % i, errD_fake_class, self.iters_cnt)
                    self.writer.add_scalar('errD_%d/mismatch_class' % i, errD_unpair_class, self.iters_cnt)

    def g_update(self, fake_imgs, sent_emb, words_emb, attr_emb, class_ids, batch_size, match_labels, cap_lens,
                 class_labels, iter_rec):

        g_loss = torch.FloatTensor(1).fill_(0).to(self.device)
        for i in range(len(self.netsD)):
            fake_img, netD = fake_imgs[i], self.netsD[i]
            disc_class = self.disc_class and i == 2
            if disc_class:
                errG, errG_class = self.g_loss_class(fake_img, sent_emb, class_labels, netD)
                g_loss += errG + errG_class * self.g_class_coe
            else:
                errG = self.g_loss(fake_img, sent_emb, netD)
                g_loss += errG

            if iter_rec:
                self.writer.add_scalar('errG/G_%d_fake_sent' % i, errG, self.iters_cnt)
                if disc_class:
                    self.writer.add_scalar('errG/G_%d_fake_class' % i, errG_class, self.iters_cnt)

        w_loss, s_loss, a_loss = self.DAMSM_loss(fake_imgs[-1], sent_emb, words_emb, attr_emb, class_ids, batch_size,
                                                 match_labels, cap_lens, self.image_encoder)

        g_loss += self.DAMSM_coe * (s_loss + w_loss + a_loss)

        if iter_rec:
            self.writer.add_scalar('errG/s_loss', s_loss, self.iters_cnt)
            self.writer.add_scalar('errG/w_loss', w_loss, self.iters_cnt)
            self.writer.add_scalar('errG/a_loss', a_loss, self.iters_cnt)

        self.optimizerG.zero_grad()
        g_loss.backward()
        self.optimizerG.step()

if __name__ == "__main__":
    args = parse_args()
    if args.cfg_file is not None:
        # the args settings are imported into cfg
        cfg_from_file(args.cfg_file)

    print('Using config:')
    pprint.pprint(cfg)

    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_ids
    cfg.TRAIN.BATCH_SIZE = args.batch_size

    if args.manualSeed is None:
        # args.manualSeed = 3407
        args.manualSeed = random.randint(1, 10000)

    print("seed now is : ", args.manualSeed)
    random.seed(args.manualSeed)
    np.random.seed(args.manualSeed)
    torch.manual_seed(args.manualSeed)
    if cfg.CUDA:
        torch.cuda.manual_seed_all(args.manualSeed)
    cudnn.benchmark = True
    print(args)

    ##########################################################################
    if args.debug:
        output_dir = os.path.join(cfg.SAVE_DIR, args.debug_output_dir)
    elif args.output_dir == "":
        now = datetime.datetime.now(dateutil.tz.tzlocal())
        timestamp = now.strftime('%Y_%m_%d_%H_%M_%S')
        output_dir = os.path.join(cfg.SAVE_DIR, "%s_%s") % (cfg.DATASET_NAME, timestamp)
    else:
        output_dir = os.path.join(cfg.SAVE_DIR, args.output_dir)

    mkdir_p(output_dir)
    shutil.copy2(sys.argv[0], output_dir)
    shutil.copy2('models.py', output_dir)
    shutil.copy2('datasets.py', output_dir)
    shutil.copy2(args.cfg_file, output_dir)

    trainer = Trainer(output_dir, args)
    trainer.train()