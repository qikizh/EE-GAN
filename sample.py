from __future__ import print_function
import multiprocessing

import os
import pickle
import sys
import random
import pprint
import argparse

import torch
from torch.autograd import Variable
import torch.backends.cudnn as cudnn
import torchvision.transforms as transforms
import numpy as np

from miscc.utils import mkdir_p, save_text_results, save_img_results, save_img_results_one_by_one
from miscc.config import cfg, cfg_from_file
from sync_batchnorm import DataParallelWithCallback
from datasets import TextDataset as TextDataset
from DAMSM import RNN_ENCODER
from prepare_attributes import PrepareAttrs
from models import Gen, ATTR_Enhance

dir_path = (os.path.abspath(os.path.join(os.path.realpath(__file__), './.')))
sys.path.append(dir_path)

multiprocessing.set_start_method('spawn', True)
UPDATE_INTERVAL = 200

def parse_args():
    parser = argparse.ArgumentParser(description='Train a DAMSM network')
    parser.add_argument('--cfg', dest='cfg_file', help='optional config file',
                        default='cfg/sample_bird.yml', type=str)
    parser.add_argument('--gpu', dest='gpu_ids', type=str, default="0")
    parser.add_argument('--manualSeed', type=int, default=3407, help='manual seed')
    # where to save
    parser.add_argument('--output_dir', default='example_bird', type=str)
    # sampling setting
    parser.add_argument('--from_code', action="store_true", help='samples from codes')
    parser.add_argument('--from_dataset', action="store_true", help='samples from datasets')
    parser.add_argument('--from_txt', action="store_true", help='samples from txt')
    parser.add_argument('--split', dest='split', default='train', type=str)
    parser.add_argument('--txt_file', dest='txt_file', default='example.txt', type=str)
    parser.add_argument('--noise_times', dest='noise_times', type=int, default=1)
    parser.add_argument('--taggar_file_path',
    default='../nltk_data/stanford-postagger-full-2015-04-20/models/english-bidirectional-distsim.tagger', type=str)
    parser.add_argument('--jar_file_path',
    default='../nltk_data/stanford-postagger-full-2015-04-20/stanford-postagger-3.5.2.jar', type=str)
    args = parser.parse_args()
    return args

class Sampling(object):

    def __init__(self, output_dir, args):

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.batch_size = cfg.TRAIN.BATCH_SIZE

        self.args = args
        self.noise_times = args.noise_times
        self.from_dataset = args.from_dataset
        self.from_code = args.from_code
        self.from_txt = args.from_txt
        self.visual_dir = output_dir
        self.cap_file_path = os.path.join(cfg.SAVE_DIR, args.txt_file)
        mkdir_p(self.visual_dir, rm_exist=True)

        if self.from_dataset:
            self.dataloader, n_words, self.ixtoword, self.wordtoix = \
                self.load_dataloader(args.debug, split=args.split)
        else:
            n_words, self.ixtoword, self.wordtoix = self.load_text_embedding()

        self.netG, self.text_encoder, self.attr_enhance = self.load_networks(n_words, self.device)

        self.batch_size = args.batch_size
        self.parser_func = PrepareAttrs.load_attr_parser(cfg.DATA_DIR, args.taggar_file_path, args.jar_file_path,
                                                         args.taggar_mode)

    @staticmethod
    def load_networks(n_words, device):
        # netG
        model_dir = cfg.TRAIN.NET_G[:cfg.TRAIN.NET_G.rfind('/')]
        netG = Gen(cfg.TRAIN.GF, 100).to(device)
        netG = DataParallelWithCallback(netG)
        model_path = cfg.TRAIN.NET_G
        netG.load_state_dict(torch.load(model_path))
        netG.eval()

        # attribute enhancing
        st_idx = cfg.TRAIN.NET_G.rfind('_') + 1
        ed_idx = cfg.TRAIN.NET_G.rfind('.')
        epoch = int(cfg.TRAIN.NET_G[st_idx:ed_idx])
        model_path = os.path.join(model_dir, "attr_enhance_%d.pth" % epoch)

        attr_enhance = ATTR_Enhance().to(device)
        attr_enhance = torch.nn.DataParallel(attr_enhance)
        attr_enhance.load_state_dict(torch.load(model_path))
        attr_enhance.eval()

        # text encoder
        text_encoder = RNN_ENCODER(n_words, nhidden=cfg.TEXT.EMBEDDING_DIM)
        state_dict = torch.load(cfg.TEXT.DAMSM_NAME, map_location=lambda storage, loc: storage)
        text_encoder.load_state_dict(state_dict)
        text_encoder.cuda()
        text_encoder.eval()

        return netG, text_encoder, attr_enhance

    @staticmethod
    def load_dataloader(debug, split='train'):
        imsize = cfg.TREE.BASE_SIZE * (2 ** (cfg.TREE.BRANCH_NUM-1))
        batch_size = cfg.TRAIN.BATCH_SIZE
        image_transform = transforms.Compose([
            transforms.Resize(int(imsize * 76 / 64)),
            transforms.RandomCrop(imsize),
            transforms.RandomHorizontalFlip()])

        dataset = TextDataset(cfg.DATA_DIR, dataset_name=cfg.DATASET_NAME, attr_name='EE-GAN',
                              split=split, transform=image_transform)

        nWorks = 1 if debug else 4
        print(dataset.n_words, dataset.embedding_num)
        assert dataset
        dataloader = torch.utils.data.DataLoader(
            dataset, batch_size=batch_size, drop_last=True,
            shuffle=True, num_workers=nWorks)

        return dataloader, dataset.n_words, dataset.ixtoword, dataset.wordtoix

    @staticmethod
    def load_text_embedding():
        filepath = os.path.join(cfg.DATA_DIR, "captions.pickle")
        with open(filepath, 'rb') as f:
            x = pickle.load(f)
            ixtoword, wordtoix = x[2], x[3]
            del x
            n_words = len(ixtoword)
            print('Load from: ', filepath)
        return n_words, ixtoword, wordtoix

    # using dataset to random sample
    def prepare_data_from_dataset(self, device):

        # only sample one batch in our setting
        data_iter = iter(self.dataloader)
        data = data_iter.next()
        rev_basic, rev_attrs, _ = data
        [imgs, caps, cap_lens, _, keys] = rev_basic

        real_imgs = Variable(imgs[-1]).to(device)
        caps = caps.squeeze().to(device)
        cap_lens = Variable(cap_lens).to(device)

        [attrs, attr_nums, attrs_len] = rev_attrs
        attrs = attrs.squeeze().to(device)
        attrs_len = attrs_len.squeeze()

        return real_imgs, caps, cap_lens, attrs, attr_nums, attrs_len, keys

    # using the content of .txt file
    def prepare_data_from_txt(self, device):

        cap_file_path = self.cap_file_path
        caps_str, attrs_str = self.txt_to_str(cap_file_path)

        caps, cap_lens = self.transfer_cap_tokens(caps_str)
        attrs, attrs_num, attrs_len = self.transfer_attr_tokens(attrs_str)

        # caps, cap_lens, attrs, attr_lens, rev_attrs
        caps = Variable(caps).to(device)
        cap_lens = Variable(cap_lens).to(device)

        if self.attr_enhance:
            attrs = Variable(attrs).to(device)
            attrs_len = Variable(attrs_len).to(device)
        else:
            attrs, attrs_len = None, None

        return caps, cap_lens, attrs, attrs_num, attrs_len, attrs_str

    def txt_to_str(self, file_path):
        rev_caps = []
        rev_attrs = []
        with open(file_path, "r") as f:
            captions = f.read().split('\n')
            for cap in captions:
                if len(cap) == 0:
                    continue
                cap = cap.replace("\ufffd\ufffd", " ")
                attrs = PrepareAttrs.do_parse_one_caption(self.parser_func, cap)
                rev_caps.append(cap)
                rev_attrs.append(attrs)

        return rev_caps, rev_attrs

    # using the content of code file
    def prepare_data_from_code(self, device):

        captions = \
            ['this blue bird has a dark blue beak, a looking grey underbelly, a dark blue collar and and black bars',
                'this small has a grey brown crown with copper brown and white stripe primaries and secondaries',
                'this is a bird with a white belly brown wing and breast and a red crown']
        caps = []
        attrs = []

        for cap in captions:
            if len(cap) == 0:
                continue
            cap = cap.replace("\ufffd\ufffd", " ")
            one_attr = PrepareAttrs.do_parse_one_caption(self.parser_func, cap)
            caps.append(cap)
            attrs.append(one_attr)

        caps_token, cap_lens = self.transfer_cap_tokens(caps)
        attrs_token, attrs_num, attrs_len = self.transfer_attr_tokens(attrs)

        caps_token = Variable(caps_token).to(device)
        cap_lens = Variable(cap_lens).to(device)
        attrs_token = Variable(attrs_token).to(device)
        attrs_len = Variable(attrs_len).to(device)

        return caps_token, cap_lens, attrs_token, attrs_num, attrs_len, attrs

    def transfer_cap_tokens(self, captions):
        """
        captions: bs x str
        """
        batch_size = len(captions)

        rev_cap_tokens = torch.zeros((batch_size, cfg.TEXT.WORDS_NUM), dtype=torch.int64)
        rev_cap_lens = torch.ones(batch_size, dtype=torch.int64)

        for ix, cap in enumerate(captions):
            cap_tokens = []
            for w in cap:
                if w in self.wordtoix:
                    cap_tokens.append(self.wordtoix[w])

            cap_len = min(len(cap_tokens), cfg.TEXT.WORDS_NUM)
            rev_cap_tokens[ix][:cap_len] = torch.tensor(cap_tokens[:cap_len])
            rev_cap_lens[ix] = torch.tensor(cap_len)

        return rev_cap_tokens, rev_cap_lens

    def transfer_attr_tokens(self, attrs):

        batch_size = len(attrs)
        rev_attr_tokens = torch.zeros((batch_size, cfg.TEXT.MAX_ATTR_NUM, cfg.TEXT.MAX_ATTR_LEN),
                            dtype=torch.int64)
        rev_attr_lens = torch.ones((batch_size, cfg.TEXT.MAX_ATTR_NUM), dtype=torch.int64)
        rev_attr_nums = list()

        for ix, multi_attr in enumerate(attrs):
            a_nums = min(len(multi_attr), cfg.TEXT.MAX_ATTR_NUM)
            rev_attr_nums.append(a_nums)
            for jx in range(a_nums):
                attr = multi_attr[jx]

                attr_tokens = []
                for w in attr:
                    if w in self.wordtoix:
                        attr_tokens.append(self.wordtoix[w])

                attr_len = min(len(attr_tokens), cfg.TEXT.MAX_ATTR_LEN)
                rev_attr_tokens[ix][jx][:attr_len] = torch.tensor(attr_tokens)
                rev_attr_lens[ix][jx] = torch.tensor(attr_len)

        return rev_attr_tokens, rev_attr_nums, rev_attr_lens

    def main(self):

        if self.from_dataset:
            print("using data randomly from dataset")

            real_images, caps, cap_lens, attrs, attr_nums, attrs_len, keys = \
                self.prepare_data_from_dataset(self.device)

            real_prefix = ["cap_%d" % idx for idx in range(self.batch_size)]
            real_save_dir = os.path.join(self.visual_dir, 'real_images')
            txt_save_path = os.path.join(self.visual_dir, 'dataset_example.txt')

            save_img_results_one_by_one(real_images, real_prefix, real_save_dir)
            save_img_results(real_images, real_prefix, real_save_dir)
            save_text_results(caps, cap_lens, self.ixtoword, txt_save_path, attrs, attr_nums, attrs_len)

        elif self.from_code:
            print("using the content from defined code")
            caps, cap_lens, attrs, attrs_num, attrs_len, attrs_str = self.prepare_data_from_code(self.device)
            for ix in range(len(attrs_str)):
                print("#%d" % ix + attrs_str[ix])

        elif self.from_txt:
            print("using the content from defined .txt file")
            caps, cap_lens, attrs, attrs_num, attrs_len, attrs_str = self.prepare_data_from_txt(self.device)
            for ix in range(len(attrs_str)):
                print("#%d" % ix + attrs_str[ix])
        else:
            print("error")
            return

        # batch_data = caps, cap_lens, attrs, attrs_len
        with torch.no_grad():
            for cap_i in range(len(caps)):
                batch_size = self.noise_times
                batch_caps = caps[cap_i].unsqueeze(0).repeat(batch_size, 1)
                batch_cap_lens = cap_lens[cap_i].repeat(batch_size)
                batch_attrs = attrs[cap_i].unsqueeze(0).repeat(batch_size, 1, 1)
                batch_attrs_len = attrs_len[cap_i].unsqueeze(0).repeat(batch_size, 1)

                noise = torch.randn(batch_size, 100)
                noise = noise.to(self.device)

                gen_use_data = \
                    [batch_caps, batch_cap_lens, batch_attrs, batch_attrs_len, noise, batch_size]
                fake_imgs = self.gen_one_batch_attr(gen_use_data)

                save_dir = os.path.join(self.visual_dir, "cap_%d" % cap_i)
                fake_prefix = ["sample_%d" % idx for idx in range(batch_size)]

                img_256 = fake_imgs[-1]
                save_img_results(img_256, "samples", save_dir)
                save_img_results_one_by_one(img_256, fake_prefix, save_dir)

    def gen_one_batch_attr(self, gen_use_data):

        caps, cap_lens, attrs, attrs_len, noise, batch_size = gen_use_data

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

        _, attn_attr_emb = self.attr_enhance(sent_emb, attrs_emb)
        attn_attr_emb = self.attr_enhance.module.attr_merge(attn_attr_emb)
        fake_imgs = self.netG(noise, sent_emb, attn_attr_emb)

        return fake_imgs[-1]


if __name__ == "__main__":

    # step1. initialize cfg setting and random seed
    args = parse_args()
    if args.cfg_file is not None:
        cfg_from_file(args.cfg_file)

    cfg.GPU_ID = args.gpu_ids
    cfg.TRAIN.BATCH_SIZE = args.batch_size
    os.environ["CUDA_VISIBLE_DEVICES"] = cfg.GPU_ID
    if args.manualSeed is None:
        args.manualSeed = random.randint(1, 10000)

    random.seed(args.manualSeed)
    np.random.seed(args.manualSeed)
    torch.manual_seed(args.manualSeed)
    print("seed now is : ", args.manualSeed)
    print('Using config:')
    pprint.pprint(cfg)

    if cfg.CUDA:
        torch.cuda.manual_seed_all(args.manualSeed)

    # where to save
    if args.debug:
        output_dir = os.path.join(cfg.SAVE_DIR, args.debug_output_dir)
    elif args.output_dir != '':
        output_dir = os.path.join(cfg.SAVE_DIR, args.output_dir)
    else:
        # save in the model dir
        last_idx = cfg.TRAIN.NET_G.rfind('Model') - 1
        output_dir = cfg.TRAIN.NET_G[:last_idx]

    print(args)
    cudnn.benchmark = True
    sam = Sampling(output_dir, args)
    sam.main()