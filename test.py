from __future__ import print_function
import multiprocessing

import os
import io
import sys
import time
import errno
import random
import pprint
import argparse

import torch
from torch.autograd import Variable
import torch.backends.cudnn as cudnn
import torchvision.utils as vutils
import numpy as np
from tqdm import tqdm
import re

from miscc.utils import mkdir_p, rm_mkdir_p, save_text_results, save_img_results
from miscc.config import cfg, cfg_from_file
from miscc.losses import words_loss, sent_loss
from sync_batchnorm import DataParallelWithCallback

from DAMSM import RNN_ENCODER, CNN_ENCODER
import shutil

from model_attr import WholeStage_ATTR as NetG, ATTR_Enhance

# from model_attr import WholeStage_ATTR_wo_SSRG as NetG, ATTR_Enhance

# from model_attr import WholeStage_ATTR_wo_Pyramid_Cum as NetG, ATTR_Enhance

dir_path = (os.path.abspath(os.path.join(os.path.realpath(__file__), './.')))
sys.path.append(dir_path)
UPDATE_INTERVAL = 100
multiprocessing.set_start_method('spawn', True)

def parse_args():
    parser = argparse.ArgumentParser(description='Train a DAMSM network')
    parser.add_argument('--cfg', dest='cfg_file',
                        help='optional config file',
                        default='cfg/eval_attr_bird.yml', type=str)
    # os setting
    parser.add_argument('--gpu', dest='gpu_ids', type=str, default="0")
    parser.add_argument('--debug', action="store_true", help='using debug mode')
    parser.add_argument('--manualSeed', type=int, help='manual seed')
    parser.add_argument('--output_dir', dest='output_dir',
                        help='the path to save models and images',
                        default='', type=str)
    parser.add_argument('--debug_output_dir', dest='debug_output_dir',
                        help='saving path in debug mode',
                        default='debug', type=str)
    # sampling setting
    parser.add_argument('--batchSize', dest='bs', type=int, default=24, help='batch size')
    parser.add_argument('--repeat_times', dest='repeat_times', type=int, default=1)
    parser.add_argument('--sampling_nums', dest='sampling_nums', type=int, default=30000)
    parser.add_argument('--regard_sent', action='store_true')
    parser.add_argument('--select_epochs', type=str, default='', help='select the epoch to generate the samples')
    parser.add_argument('--saving_image', action='store_true')
    parser.add_argument('--compare_sim', action='store_true')
    # model setting
    parser.add_argument('--attr_enhance', action='store_true', help='to load the extra attributes embeddings')
    parser.add_argument('--fine_tuning_text_embs', action='store_true', help='to load the extra attributes embeddings')
    args = parser.parse_args()
    return args

def prepare_data(data, batch_size, device):
    [caps, cap_lens, cls_ids, keys], rev_attrs = data
    caps = caps.squeeze().to(device)
    cap_lens = Variable(cap_lens).to(device)
    cls_ids = cls_ids.numpy()

    if len(rev_attrs) == 0:
        gen_use_data = [caps, cap_lens, batch_size]
    else:
        [attrs, attr_nums, attrs_len] = rev_attrs
        attrs = attrs.squeeze().to(device)
        attrs_len = attrs_len.squeeze()
        gen_use_data = [caps, cap_lens, attrs, attrs_len, batch_size]

    return gen_use_data, cls_ids, keys

def mk_del_dir(path, del_dir=False):
    if del_dir and os.path.exists(path):
        shutil.rmtree(path)
    os.makedirs(path)

def calculate_r(scores):
    ranks = torch.tensor(np.array([0, 0, 0]))
    inx = torch.argsort(scores, dim=1, descending=True)
    if 0 == inx[0]:
        ranks += 1
    elif 0 in inx[:5]:
        ranks[1:] += 1
    elif 0 in inx[:10]:
        ranks[2:] += 1

    return ranks

class Tester(object):

    def __init__(self, output_dir, args):
        # make dir for all kinds of output
        self.output_dir = output_dir
        self.image_dir = os.path.join(output_dir, 'GenImage')
        print("The images are saved in %s", self.image_dir)

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.args = args
        self.debug = args.debug
        self.batch_size = cfg.TRAIN.BATCH_SIZE
        self.attr_enhance = args.attr_enhance
        self.saving_image = args.saving_image
        self.compare_sim = args.compare_sim

        if args.select_epochs == '':
            st, ed, interval = 500, 700, 10
            self.select_epochs = [epoch for epoch in range(st, ed + interval, interval)]
        else:
            self.select_epochs = self.prepare_epochs(args.select_epochs)

        print("select epochs concludes:" + str(self.select_epochs))

        self.fine_tuning_text_embs = args.fine_tuning_text_embs
        self.attr_enhance = args.attr_enhance
        self.sampling_nums = args.sampling_nums
        self.repeat_times = args.repeat_times

        self.dataset, self.dataloader, n_words, self.ixtoword = self.load_dataloader(args.debug, args.regard_sent,
                                                                                     self.attr_enhance)
        # whole_syn, text_encoder, attr_enhance, whole_syn_prefix, text_enc_prefix, attr_enhance_prefix
        self.whole_syn, self.text_encoder, self.attr_enhance_enc, self.whole_syn_prefix, \
            self.text_enc_prefix, self.attr_enhance_prefix, self.gen_one_batch = \
            self.load_networks(n_words, self.device, args.attr_enhance, args.fine_tuning_text_embs)

        if self.compare_sim:
            self.sim_image_encoder, self.sim_text_encoder, self.text_emb_same = \
                self.load_sim_network(n_words, self.device)

    """
    the four functions are applied to prepare data_loader and networks
    """
    @staticmethod
    def prepare_epochs(epochs):
        """
        epochs: str, like 10,20,30
        """
        epochs_str = epochs.replace(',', ' ')
        epochs = list(map(int, re.split(r"[ ]+", epochs_str)))
        print("The select Epochs include: " + str(epochs))
        return epochs

    def load_networks(self, n_words, device, attr_enhance, fine_tuning_text_embs):
        # the statement is same as the SSA-GAN
        whole_syn = NetG(cfg.GAN.GF_DIM, 100).to(device)
        whole_syn = DataParallelWithCallback(whole_syn)
        whole_syn.eval()

        text_encoder = RNN_ENCODER(n_words, nhidden=cfg.TEXT.EMBEDDING_DIM).to(device)
        text_encoder.eval()
        whole_syn_path = cfg.TRAIN.NET_G

        ed = whole_syn_path.rfind('/')
        model_dir = whole_syn_path[:ed]

        whole_syn_prefix = os.path.join(model_dir, "netG_")
        if attr_enhance:
            attr_enhance_prefix = os.path.join(model_dir, "attr_enhance_")
            attr_enhance = ATTR_Enhance().to(device)
            attr_enhance = torch.nn.DataParallel(attr_enhance)
            attr_enhance.eval()
            print("using attr_enhance.")
            gen_one_batch = self.gen_one_batch_attr
        else:
            attr_enhance, attr_enhance_prefix = None, None
            gen_one_batch = self.gen_one_batch_sent

        text_encoder_path = cfg.TEXT.DAMSM_NAME
        state_dict = torch.load(cfg.TEXT.DAMSM_NAME, map_location=lambda storage, loc: storage)
        text_encoder.load_state_dict(state_dict)

        if fine_tuning_text_embs:
            i_end = text_encoder_path.rfind('_') + 1
            text_enc_prefix = text_encoder_path[:i_end]
            print("using fine tuning text embs.")
        else:
            text_enc_prefix = None

        return whole_syn, text_encoder, attr_enhance, \
               whole_syn_prefix, text_enc_prefix, attr_enhance_prefix, gen_one_batch

    def load_sim_network(self, n_words, device):

        text_emb_same_flag = True

        if cfg.TEXT.SIM_DAMSM_NAME == cfg.TEXT.DAMSM_NAME:
            rev_text_encoder = self.text_encoder
            text_encoder_path = cfg.TEXT.DAMSM_NAME
        else:
            rev_text_encoder = RNN_ENCODER(n_words, nhidden=cfg.TEXT.EMBEDDING_DIM).to(device)
            state_dict = torch.load(cfg.TEXT.SIM_DAMSM_NAME, map_location=lambda storage, loc: storage)
            rev_text_encoder.load_state_dict(state_dict)
            text_encoder_path = cfg.TEXT.SIM_DAMSM_NAME
            rev_text_encoder.eval()
            text_emb_same_flag = False

        rev_image_encoder = CNN_ENCODER(cfg.TEXT.EMBEDDING_DIM).to(device)
        image_encoder_path = text_encoder_path.replace('text_encoder', 'image_encoder')
        state_dict = torch.load(image_encoder_path, map_location=lambda storage, loc: storage)
        rev_image_encoder.load_state_dict(state_dict)
        rev_image_encoder.eval()

        return rev_image_encoder, rev_text_encoder, text_emb_same_flag

    @staticmethod
    def load_dataloader(debug, regard_sent, attr_enhance):

        print("using %s" % cfg.DATASET_NAME)

        if cfg.DATASET_NAME == 'bird':
            from datasets import TextOnlyDataset as TextDataset
        elif cfg.DATASET_NAME == 'flower':
            from dataset_flower import TextOnlyDataset as TextDataset
        else:
            from dataset_coco import TextOnlyDataset as TextDataset

        batch_size = cfg.TRAIN.BATCH_SIZE

        if attr_enhance:
            dataset = TextDataset(cfg.DATA_DIR, split='test', regard_sent=regard_sent)
        else:
            dataset = TextDataset(cfg.DATA_DIR, split='test', regard_sent=regard_sent, attr_name='')

        nWorks = 0 if debug else 8
        print(dataset.n_words, dataset.embeddings_num)
        assert dataset
        dataloader = torch.utils.data.DataLoader(
            dataset, batch_size=batch_size, drop_last=True,
            shuffle=True, num_workers=nWorks)

        return dataset, dataloader, dataset.n_words, dataset.ixtoword

    """
    main control the generation process
    """
    def main(self):
        select_epoch = self.select_epochs
        whole_syn_prefix, text_encoder_prefix, attr_enhance_prefix = \
            self.whole_syn_prefix, self.text_enc_prefix, self.attr_enhance_prefix

        max_iter_nums = self.sampling_nums // self.batch_size + 1
        real_sampling_num = max_iter_nums * self.batch_size
        print("The max iteration nums is %d" % max_iter_nums)
        print("The length of batches is %d" % len(self.dataloader))
        print("The batch size is %d" % self.batch_size)
        print("The sampling nums is %d" % real_sampling_num)

        all_ts = 0
        all_R_records = []
        for epoch in select_epoch:
            # reloading the new model
            start_ts = time.time()
            whole_syn_path = whole_syn_prefix + "%d.pth" % epoch
            state_dict = torch.load(whole_syn_path, map_location=lambda storage, loc: storage)
            self.whole_syn.load_state_dict(state_dict)

            if self.attr_enhance:
                attr_enhance_enc = attr_enhance_prefix + "%d.pth" % epoch
                state_dict = torch.load(attr_enhance_enc, map_location=lambda storage, loc: storage)
                self.attr_enhance_enc.load_state_dict(state_dict)

            if self.fine_tuning_text_embs:
                text_encoder_path = text_encoder_prefix + "%d.pth" % epoch
                state_dict = torch.load(text_encoder_path, map_location=lambda storage, loc: storage)
                self.text_encoder.load_state_dict(state_dict)

            epoch_R_records = []
            for r_ix in range(self.repeat_times):
                if self.saving_image:
                    epoch_saving_dir = os.path.join(self.image_dir, "Epoch_%d_%d" % (epoch, r_ix))
                    print("Saving dir: %s" % epoch_saving_dir)
                    mk_del_dir(epoch_saving_dir, del_dir=True)
                else:
                    epoch_saving_dir = None

                if self.compare_sim:
                    R_hits = np.zeros(real_sampling_num)
                else:
                    R_hits = None

                # generate for one model
                self.traverse_dataset_30k(epoch_saving_dir, max_iter_nums, R_hits)

                if self.compare_sim:
                    R_mean, R_std = self.cal_sim_mean_std(R_hits, real_sampling_num, clusters=10)
                    print("R mean:{:.4f} std:{:.4f}".format(R_mean, R_std))
                    epoch_R_records.append([R_mean, R_std])

            all_R_records.append(epoch_R_records)
            end_ts = time.time()
            print('''Epoch_%d model is finished and costs time: %.2f\n''' % (epoch, end_ts - start_ts))
            all_ts += (end_ts - start_ts)

        print('''The all cost time: %.2f\n''' % all_ts)
        if self.compare_sim:
            self.display_Rs(all_R_records)

    def traverse_dataset_30k(self, image_dir, max_iter_nums, R_hits):

        device = self.device
        batch_size = self.batch_size

        iter_cnt = 0
        re_time = 0
        R_cnt = 0
        continue_sampling = True

        with torch.no_grad():
            while continue_sampling:
                re_time_image_dir = os.path.join(image_dir, "repeat_times_%d" % re_time)
                mkdir_p(re_time_image_dir)

                data_iter = iter(self.dataloader)
                for _ in tqdm(range(len(data_iter))):
                    # for iters in range(len(data_iter)):
                    data = data_iter.next()
                    # caps, cap_lens, attrs, attr_nums, attrs_len, cls_ids, keys
                    gen_use_data, cls_ids, keys = prepare_data(data, batch_size, device)

                    # caps, cap_lens, attrs, attrs_len, batch_size
                    fake_imgs, sent_emb = self.gen_one_batch(gen_use_data, device)

                    if self.compare_sim:
                        # fake_imgs, sent_embs, caps, cap_lens, cls_ids, batch_size,
                        #                            R_hits, R_cnt, R_val=100
                        caps = gen_use_data[0]
                        cap_lens = gen_use_data[1]
                        self.cal_sim_one_by_one(fake_imgs, sent_emb, caps,
                                                cap_lens, cls_ids, batch_size, R_hits, R_cnt, device)
                        R_cnt += batch_size

                    if self.saving_image:
                        self.save_imgs_one_by_one(fake_imgs, re_time_image_dir, keys)

                    iter_cnt += 1
                    if iter_cnt >= max_iter_nums:
                        continue_sampling = False
                        break

                re_time += 1

    def gen_one_batch_attr(self, gen_use_data, device):

        caps, cap_lens, attrs, attrs_len, batch_size = gen_use_data

        hidden = self.text_encoder.init_hidden(batch_size)
        _, sent_emb = self.text_encoder(caps, cap_lens, hidden)

        # extract the attribute embedding
        attrs_emb = list()
        for i in range(cfg.TEXT.MAX_ATTR_NUM):
            one_attr = attrs[:, i, :].squeeze(-1)
            one_attr_len = attrs_len[:, i].squeeze(-1)
            _, one_attr_emb = self.text_encoder(one_attr, one_attr_len, hidden)
            attrs_emb.append(one_attr_emb)
        attrs_emb = torch.stack(attrs_emb, dim=1)
        attrs_emb = attrs_emb.detach()

        _, attn_attr_emb = self.attr_enhance_enc(sent_emb, attrs_emb)
        attn_attr_emb = self.attr_enhance_enc.module.attr_merge(attn_attr_emb)

        noise = torch.randn(batch_size, 100)
        noise = noise.to(device)
        fake_imgs = self.whole_syn(noise, sent_emb, attn_attr_emb)
        return fake_imgs[-1], sent_emb

    def gen_one_batch_sent(self, gen_use_data, device):

        caps, cap_lens, batch_size = gen_use_data
        hidden = self.text_encoder.init_hidden(batch_size)
        _, sent_emb = self.text_encoder(caps, cap_lens, hidden)
        noise = torch.randn(batch_size, 100)
        noise = noise.to(device)
        fake_imgs = self.whole_syn(noise, sent_emb)
        return fake_imgs[-1], sent_emb

    def save_imgs_one_by_one(self, fake_imgs, image_dir, keys):
        for j in range(self.batch_size):
            fake_img_path = '%s/fake_%s.jpg' % (image_dir, keys[j])
            vutils.save_image(fake_imgs[j], fake_img_path, scale_each=True, normalize=True)

    def save_imgs_batch(self, fake_imgs, step):
        save_img_results(None, fake_imgs, None, "step_%d" % step, self.image_dir)

    def cal_sim_one_by_one(self, fake_imgs, sent_embs, caps, cap_lens, cls_ids, batch_size,
                           R_hits, R_cnt, device, R_val=100):

        if not self.text_emb_same:
            hidden = self.sim_text_encoder.init_hidden(batch_size)
            _, sim_sent_embs = self.sim_text_encoder(caps, cap_lens, hidden)
        else:
            sim_sent_embs = sent_embs

        _, cnn_code = self.sim_image_encoder(fake_imgs)

        for ix in range(batch_size):
            cls = cls_ids[ix]
            compare_caps, compare_cap_lens = self.dataset.get_sent_wong_multi(cls, R_val)
            compare_caps, compare_cap_lens = compare_caps.to(device), compare_cap_lens.to(device)

            hidden = self.sim_text_encoder.init_hidden(R_val-1)
            _, compare_sent_emb = self.sim_text_encoder(compare_caps, compare_cap_lens, hidden)
            rnn_code = torch.cat((sim_sent_embs[ix, :].unsqueeze(0), compare_sent_emb), 0)

            scores = torch.mm(cnn_code[ix].unsqueeze(0), rnn_code.transpose(0, 1))  # 1* 100
            cnn_code_norm = torch.norm(cnn_code[ix].unsqueeze(0), 2, dim=1, keepdim=True)
            rnn_code_norm = torch.norm(rnn_code, 2, dim=1, keepdim=True)
            norm = torch.mm(cnn_code_norm, rnn_code_norm.transpose(0, 1))
            scores0 = scores / norm.clamp(min=1e-8)

            if torch.argmax(scores0) == 0:
                R_hits[R_cnt] = 1
            R_cnt += 1

    @staticmethod
    def cal_sim_mean_std(R_hits, real_sampling_nums, clusters=10):
        """
        The function could employ parameter ** sampling_nums ** to achieve 30K testing skillfully
        """
        sampling_nums = min(real_sampling_nums, 30000)

        one_cluster_nums = sampling_nums // clusters
        measures = np.zeros(clusters)

        np.random.shuffle(R_hits)
        for ix in range(clusters):
            st = ix * one_cluster_nums
            ed = (ix + 1) * one_cluster_nums - 1
            measures[ix] = np.average(R_hits[st: ed])

        R_mean = np.average(measures)
        R_std = np.std(measures)

        return R_mean, R_std

    def display_Rs(self, all_R_records):
        means = []
        stds = []
        print("Each epoch repeats %d times" % self.repeat_times)
        # print("The R-precisions for a %s text-image encoder are followed" % c_word)
        for i in range(len(self.select_epochs)):
            epoch_means = []
            epoch_std = []
            for j in range(self.repeat_times):
                epoch_means.append(all_R_records[i][j][0])
                epoch_std.append(all_R_records[i][j][1])
            means.append(epoch_means)
            stds.append(epoch_std)
        print(means)
        print(stds)


if __name__ == "__main__":
    args = parse_args()
    if args.cfg_file is not None:
        cfg_from_file(args.cfg_file)

    # if args.gpu_id == -1:
    #     cfg.CUDA = False
    # else:
    #     cfg.GPU_ID = args.gpu_id
    print('Using config:')
    pprint.pprint(cfg)
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_ids
    cfg.TRAIN.BATCH_SIZE = args.bs

    if not cfg.TRAIN.FLAG:
        args.manualSeed = 3407
    elif args.manualSeed is None:
        args.manualSeed = 3407
        # args.manualSeed = random.randint(1, 10000)

    print("seed now is : ", args.manualSeed)
    random.seed(args.manualSeed)
    np.random.seed(args.manualSeed)
    torch.manual_seed(args.manualSeed)
    if cfg.CUDA:
        torch.cuda.manual_seed_all(args.manualSeed)
    print(args)

    ##########################################################################
    if args.debug:
        output_dir = os.path.join(cfg.SAVE_DIR, args.debug_output_dir)
    elif args.output_dir != '':
        output_dir = os.path.join(cfg.SAVE_DIR, args.output_dir)
    else:
        # save in the model dir
        last_idx = cfg.TRAIN.NET_G.rfind('Model') - 1
        output_dir = cfg.TRAIN.NET_G[:last_idx]

    # Kai: i don't want to specify a gpu id
    # torch.cuda.set_device(cfg.GPU_ID)
    cudnn.benchmark = True
    tester = Tester(output_dir, args)
    tester.main()
