from __future__ import print_function
import multiprocessing

import os
import io
import pickle
import sys
import time
import errno
import random
import pprint
import datetime
import dateutil.tz
import argparse

import torch
import torchvision
import torch.nn as nn
import torch.optim as optim
from nltk import RegexpTokenizer
from torch.autograd import Variable
import torch.backends.cudnn as cudnn
import torchvision.transforms as transforms
import torchvision.utils as vutils
from tensorboardX import SummaryWriter
import numpy as np
from PIL import Image
from tqdm import tqdm
import matplotlib.pyplot as plt

import nltk
from nltk.tokenize import RegexpTokenizer
from nltk.tag import StanfordPOSTagger, PerceptronTagger
from miscc.utils import mkdir_p, save_text_results, save_img_results
from miscc.config import cfg, cfg_from_file
from sync_batchnorm import DataParallelWithCallback
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
    parser.add_argument('--debug', action="store_true", help='using debug mode')
    parser.add_argument('--manualSeed', type=int, default=3407, help='manual seed')
    # where to save
    parser.add_argument('--output_dir', dest='output_dir', default='example', type=str)
    # sampling setting
    parser.add_argument('--from_code', action="store_true", help='samples from codes')
    parser.add_argument('--from_dataset', action="store_true", help='samples from datasets')
    parser.add_argument('--from_txt', action="store_true", help='samples from txt')
    parser.add_argument('--split', dest='split', default='train', type=str)
    parser.add_argument('--txt_file', dest='txt_file', default='example.txt', type=str)
    parser.add_argument('--noise_times', dest='noise_times', type=int, default=1)
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

        self.output_dir = output_dir

        self.cap_file_path = os.path.join(output_dir, args.txt_file)
        self.dataset_name = cfg.DATASET_NAME

        if args.debug:
            self.visual_dir = os.path.join(visual_dir, "debug")
        else:
            self.visual_dir = os.path.join(visual_dir, args.output_filename)
        mkdir_p(self.visual_dir)

        if self.from_dataset:
            self.dataloader, n_words, self.ixtoword, self.wordtoix = \
                self.load_dataloader(args.debug, split=args.split)
        else:
            n_words, self.ixtoword, self.wordtoix = self.load_text_info()

        self.netG, self.text_encoder, self.attr_enhance_enc, self.gen_one_batch \
            = self.load_networks(n_words, self.attr_enhance, self.device)

        self.batch_size = args.batch_size
        self.tokenizer, self.taggar, self.chunk_parsers, self.split_chunk_parsers = \
            self.load_attr_func()

    """
    synthesize images by model inference
    """

    def gen_one_batch_attr_ablation(self, gen_use_data, noise):

        caps, cap_lens, attrs, attrs_len, batch_size = gen_use_data

        hidden = self.text_encoder.init_hidden(batch_size)
        _, sent_emb = self.text_encoder(caps, cap_lens, hidden)

        if attrs is None:
            # sent_emb --> bs x ctf
            attrs_emb = sent_emb.unsqueeze(1).repeat(1, cfg.TEXT.MAX_ATTR_NUM, 1)
            print("no attr")
        else:
            # extract the attribute embedding
            attrs_emb = list()
            for i in range(cfg.TEXT.MAX_ATTR_NUM):
                one_attr = attrs[:, i, :].squeeze(-1)
                one_attr_len = attrs_len[:, i].squeeze(-1)
                _, one_attr_emb = self.text_encoder(one_attr, one_attr_len, hidden)
                attrs_emb.append(one_attr_emb)
            attrs_emb = torch.stack(attrs_emb, dim=1)
            print("using attr")

        attrs_emb = attrs_emb.detach()
        _, attn_attr_emb = self.attr_enhance_enc(sent_emb, attrs_emb)
        attn_attr_emb = self.attr_enhance_enc.module.attr_merge(attn_attr_emb)

        fake_imgs = self.netG(noise, sent_emb, attn_attr_emb)
        return fake_imgs

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
        fake_imgs = self.netG(noise, sent_emb, attn_attr_emb)
        return fake_imgs

    def gen_one_batch_sent(self, gen_use_data, device):

        caps, cap_lens, batch_size = gen_use_data
        hidden = self.text_encoder.init_hidden(batch_size)
        _, sent_emb = self.text_encoder(caps, cap_lens, hidden)
        noise = torch.randn(batch_size, 100)
        noise = noise.to(device)
        fake_imgs = self.netG(noise, sent_emb)
        return fake_imgs

    def load_networks(self, n_words, device):

        model_dir = cfg.TRAIN.NET_G[:cfg.TRAIN.NET_G.rfind('/')]
        netG = Gen(cfg.TRAIN.GF, 100).to(device)
        netG = DataParallelWithCallback(netG)
        model_path = cfg.TRAIN.NET_G
        netG.load_state_dict(torch.load(model_path))
        netG.eval()

        st_idx = cfg.TRAIN.NET_G.rfind('_') + 1
        ed_idx = cfg.TRAIN.NET_G.rfind('.')
            epoch = int(cfg.TRAIN.NET_G[st_idx:ed_idx])
            model_path = os.path.join(model_dir, "attr_enhance_%d.pth" % epoch)

            attr_enhance = ATTR_Enhance().to(device)
            attr_enhance = torch.nn.DataParallel(attr_enhance)
            attr_enhance.load_state_dict(torch.load(model_path))
            attr_enhance.eval()

            gen_one_batch = self.gen_one_batch_attr
            print("using attr_enhance.")

        else:
            attr_enhance = None
            gen_one_batch = self.gen_one_batch_sent

        text_encoder = RNN_ENCODER(n_words, nhidden=cfg.TEXT.EMBEDDING_DIM)
        state_dict = torch.load(cfg.TEXT.DAMSM_NAME, map_location=lambda storage, loc: storage)
        text_encoder.load_state_dict(state_dict)
        text_encoder.cuda()
        text_encoder.eval()

        return netG, text_encoder, attr_enhance, gen_one_batch


    def load_attr_func(self):
        tokenizer = RegexpTokenizer(r'\w+')

        taggar = PrepareAttrs.load_taggar(taggar_mode='stanford')

        if self.dataset_name == 'bird':
            chunk_parsers, split_chunk_parsers = PrepareAttrs.define_cub_parser()
        elif self.dataset_name == 'flower':
            chunk_parsers, split_chunk_parsers = PrepareAttrs.define_oxford_parser()
        else:
            chunk_parsers, split_chunk_parsers = PrepareAttrs.define_coco_parser()

        return tokenizer, taggar, chunk_parsers, split_chunk_parsers

    def load_dataloader(self, debug, split='train'):
        imsize = cfg.TREE.BASE_SIZE * (2 ** (cfg.TREE.BRANCH_NUM-1))
        batch_size = cfg.TRAIN.BATCH_SIZE
        image_transform = transforms.Compose([
            transforms.Resize(int(imsize * 76 / 64)),
            transforms.RandomCrop(imsize),
            transforms.RandomHorizontalFlip()])

        # image_transform = transforms.Compose([
        #     transforms.Resize(int(imsize))])

        if self.dataset_name == 'bird':
            from datasets import TextDataset as TextDataset
        elif self.dataset_name == 'flower':
            from dataset_flower import TextDataset as TextDataset
        else:
            from dataset_coco import TextDataset as TextDataset

        dataset = TextDataset(cfg.DATA_DIR, split,
                              base_size=cfg.TREE.BASE_SIZE,
                              transform=image_transform, get_mismatch_pair=False)

        nWorks = 1 if debug else 4
        print(dataset.n_words, dataset.embeddings_num)
        assert dataset
        dataloader = torch.utils.data.DataLoader(
            dataset, batch_size=batch_size, drop_last=True,
            shuffle=True, num_workers=nWorks)

        return dataloader, dataset.n_words, dataset.ixtoword, dataset.wordtoix

    def load_text_info(self):

        if self.dataset_name == 'bird':
            filepath = os.path.join(cfg.DATA_DIR, "attn_captions.pickle")
        else:
            filepath = os.path.join(cfg.DATA_DIR, "captions.pickle")

        with open(filepath, 'rb') as f:
            x = pickle.load(f)
            # train_captions, test_captions = x[0], x[1]
            ixtoword, wordtoix = x[2], x[3]
            del x
            n_words = len(ixtoword)
            print('Load from: ', filepath)

        return n_words, ixtoword, wordtoix

    def prepare_data_from_dataset(self, device):

        # only sample one batch in our setting
        data_iter = iter(self.dataloader)
        data = data_iter.next()

        rev_basic, rev_attrs, _ = data
        [imgs, caps, cap_lens, _, keys] = rev_basic

        real_imgs = Variable(imgs[-1]).to(device)
        caps = caps.squeeze().to(device)
        cap_lens = Variable(cap_lens).to(device)

        if self.attr_enhance:
            [attrs, attr_nums, attrs_len] = rev_attrs
            attrs = attrs.squeeze().to(device)
            attrs_len = attrs_len.squeeze()
        else:
            attrs, attrs_len = None, None

        return real_imgs, caps, cap_lens, attrs, attrs_len, keys

    def prepare_data_from_txt(self, device):

        cap_file_path = self.cap_file_path
        caps_str, attrs_str = self.process_caps_from_row(cap_file_path)

        caps, cap_lens = self.transfer_cap_tokens(caps_str)
        attrs, attrs_len = self.transfer_attr_tokens(attrs_str)

        # caps, cap_lens, attrs, attr_lens, rev_attrs

        caps = Variable(caps).to(device)
        cap_lens = Variable(cap_lens).to(device)

        if self.attr_enhance:
            attrs = Variable(attrs).to(device)
            attrs_len = Variable(attrs_len).to(device)
        else:
            attrs, attrs_len = None, None

        return caps, cap_lens, attrs, attrs_len, attrs_str

    def defined_in_code(self):

        # caps = \
        #     ['this white bird has a dark blue beak, a looking grey underbelly, a dark blue collar and and black bars']

        # caps = ['this blue bird has a dark blue beak, a looking grey underbelly, a dark blue collar and and black bars',
        #         'this small has a grey brown crown with copper brown and white stripe primaries and secondaries',
        #         'this is a bird with a white belly brown wing and breast and a red crown']

        # blue bird, a dark blue beak, a looking grey underbelly, a dark blue collar, black bars

        attrs = [
            ['white bird'],
            ['white bird', 'white bird', 'white bird'],
            ['grey underbelly', 'grey underbelly', 'grey underbelly'],
            ['black bars', 'black bars', 'black bars'],
            ['blue collar', 'black bars'],

            ['grey underbelly', 'dark blue collar'],
            ['white bird', 'grey underbelly', 'dark blue collar'],

            ['grey underbelly', 'dark blue collar', 'black bars']
        ]

        # given the row data and then organize them
        for ix, cap in enumerate(caps):
            cap = cap.replace("\ufffd\ufffd", " ")
            cap = self.tokenizer.tokenize(cap.lower())
            caps[ix] = cap

        rev_attrs = []
        for attr in attrs:
            one_attr = []
            for at in attr:
                at = at.lower()
                split_at = at.split(' ')
                one_attr.append(split_at)

            rev_attrs.append(one_attr)

        caps = [caps[0] for _ in range(len(attrs))]

        return caps, rev_attrs

    def prepare_caps_from_code(self, device):

        caps, attrs = self.defined_in_code()
        caps, cap_lens = self.transfer_cap_tokens(caps)
        attrs, attrs_len = self.transfer_attr_tokens(attrs)

        # caps, cap_lens, attrs, attr_lens, rev_attrs

        caps = Variable(caps).to(device)
        cap_lens = Variable(cap_lens).to(device)
        attrs = Variable(attrs).to(device)
        attrs_len = Variable(attrs_len).to(device)

        return caps, cap_lens, attrs, attrs_len

    """
    something complex
    """
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

        for ix, multi_attr in enumerate(attrs):
            a_nums = min(len(multi_attr), cfg.TEXT.MAX_ATTR_NUM)
            for jx in range(a_nums):
                attr = multi_attr[jx]

                attr_tokens = []
                for w in attr:
                    if w in self.wordtoix:
                        attr_tokens.append(self.wordtoix[w])

                attr_len = min(len(attr_tokens), cfg.TEXT.MAX_ATTR_LEN)
                rev_attr_tokens[ix][jx][:attr_len] = torch.tensor(attr_tokens)
                rev_attr_lens[ix][jx] = torch.tensor(attr_len)

        return rev_attr_tokens, rev_attr_lens

    def process_caps_from_row(self, file_path):

        rev_caps = []
        rev_attrs = []

        with open(file_path, "r") as f:
            captions = f.read().split('\n')
            for cap in captions:
                if len(cap) == 0:
                    continue

                cap = cap.replace("\ufffd\ufffd", " ")
                cap = self.tokenizer.tokenize(cap.lower())
                # parse the attributes in the description
                attrs = self.do_parse_one_caption(cap)

                rev_caps.append(cap)
                rev_attrs.append(attrs)

        return rev_caps, rev_attrs

    def do_parse_one_caption(self, cap):
        """
        tokenizer, chunk_parsers and split_chunk_parsers
        """
        if isinstance(cap, str):
            tokens = self.tokenizer.tokenize(cap.lower())
        else:
            tokens = cap

        tags = self.taggar.tag(tokens)
        attr_set = set()

        for chunk_parser in self.chunk_parsers:
            tree = chunk_parser.parse(tags)
            for subtree in tree.subtrees(filter=lambda t: t.label() == 'NP'):
                myPhrase = []
                for item in subtree.leaves():
                    myPhrase.append(item[0])
                tmp = " ".join(myPhrase)
                attr_set.add(tmp)

        if self.split_chunk_parsers is not None:
            for chunk_parser in self.split_chunk_parsers:
                tree = chunk_parser.parse(tags)
                for subtree in tree.subtrees(filter=lambda t: t.label() == 'NP'):
                    myPhrase = []
                    for item in subtree.leaves():
                        myPhrase.append(item[0])
                    tmp = " ".join(myPhrase)
                    attr_set.discard(tmp)

        revs = []
        for attr_str in attr_set:
            tmp = attr_str.split(" ")
            revs.append(tmp)

        return revs

    def main(self):

        save_dir = self.visual_dir
        batch = self.batch_size
        noise = None
        if self.from_dataset:
            print("dataset")
            real_images, caps, cap_lens, attrs, attrs_len, keys = \
                self.prepare_data_from_dataset(self.device)

            keys_real = ["real_%d" % idx for idx in range(batch)]
            self.save_imgs_batch(real_images, save_dir, "origin", is_fake=False)
            self.save_imgs_one_by_one(real_images, save_dir, keys_real, is_fake=False)

            save_path = os.path.join(save_dir, "using_text.txt")
            self.save_text_result_from_dataset(caps, cap_lens, attrs, attrs_len, save_path)

        elif self.from_code:

            print("code")
            caps, cap_lens, attrs, attrs_len = \
                self.prepare_caps_from_code(self.device)

            with torch.no_grad():

                noise = torch.randn(self.noise_times, 100)
                noise = noise.to(self.device)

                tmp_cap = caps[0].unsqueeze(0).repeat(self.noise_times, 1)
                tmp_cap_len = cap_lens[0].repeat(self.noise_times)

                rev_data = [tmp_cap, tmp_cap_len, None, None, self.noise_times]

                tmp_save_dir = os.path.join(save_dir, "sent_only")
                keys = ["sent_only_%d" % n_i for n_i in range(self.noise_times)]
                batch_key = "sent_only_batch"

                # self.gen_and_save(rev_data, tmp_save_dir, keys, batch_key)
                mkdir_p(tmp_save_dir)
                fake_imgs = self.gen_one_batch_attr_ablation(rev_data, noise)
                img_256 = fake_imgs[-1]
                self.save_imgs_one_by_one(img_256, tmp_save_dir, keys)
                self.save_imgs_batch(img_256, tmp_save_dir, batch_key)

        else:
            print("txt")
            caps, cap_lens, attrs, attrs_len, attrs_str = \
                self.prepare_data_from_txt(self.device)

            for ix, ats in enumerate(attrs_str):
                print("#%d" % ix)
                print(ats)

        # caps, cap_lens, attrs, attrs_len

        with torch.no_grad():
            for cap_i in range(len(caps)):
                tmp_save_dir = os.path.join(save_dir, "cap_%d" % cap_i)

                tmp_cap = caps[cap_i].unsqueeze(0).repeat(self.noise_times, 1)
                tmp_cap_len = cap_lens[cap_i].repeat(self.noise_times)

                if self.attr_enhance:
                    tmp_attr = attrs[cap_i].unsqueeze(0).repeat(self.noise_times, 1, 1)
                    tmp_attr_len = attrs_len[cap_i].unsqueeze(0).repeat(self.noise_times, 1)
                    # caps, cap_lens, attrs, attrs_len, batch_size
                    rev_data = [tmp_cap, tmp_cap_len, tmp_attr, tmp_attr_len, self.noise_times]
                else:
                    rev_data = [tmp_cap, tmp_cap_len, self.noise_times]

                keys = ["cap_%d_%d" % (cap_i, n_i) for n_i in range(self.noise_times)]
                batch_key = "cap_%d" % cap_i
                mkdir_p(tmp_save_dir)

                if self.from_code:
                    fake_imgs = self.gen_one_batch_attr_ablation(rev_data, noise)
                else:
                    fake_imgs = self.gen_one_batch(rev_data, self.device)

                img_256 = fake_imgs[-1]
                self.save_imgs_one_by_one(img_256, tmp_save_dir, keys)
                self.save_imgs_batch(img_256, tmp_save_dir, batch_key)
                # self.gen_and_save(rev_data, tmp_save_dir, keys, batch_key)


    def gen_and_save(self, rev_data, save_dir, keys, batch_key):

        mkdir_p(save_dir)
        fake_imgs = self.gen_one_batch(rev_data, self.device)
        img_256 = fake_imgs[-1]
        self.save_imgs_one_by_one(img_256, save_dir, keys)
        self.save_imgs_batch(img_256, save_dir, batch_key)

    """
    saving results
    """
    def save_text_result_from_dataset(self, captions, cap_lens, attrs,
                                      attrs_len, save_file_path):
        texts = list()
        for i in range(len(captions)):
            # captions
            cap = captions[i].data.cpu().numpy()
            cap_len = cap_lens[i]
            words = [self.ixtoword[cap[j]].encode('ascii', 'ignore').decode('ascii')
                     for j in range(cap_len)]

            sent_str = " ".join(words)
            texts.append(sent_str)

            # attributes
            # att_str = "# "
            # att_num = len(attrs[i])
            # # att_num = 0, 1, 2, 3
            # for att_ix in range(att_num):
            #     # a little confusion for this setting
            #     att_len = attrs_len[i][att_ix]
            #     att = attrs[i][att_ix].data.cpu().numpy()
            #     words = [self.ixtoword[att[j]].encode('ascii', 'ignore').decode('ascii')
            #              for j in range(att_len)]
            #     att_str += " ".join(words) + ", "
            #
            # texts.append(att_str)
            # texts.append("\n")

        f = open(save_file_path, "w+")
        for sent in texts:
            f.write(sent + "\n")
        f.close()

    @staticmethod
    def save_imgs_one_by_one(imgs, save_dir, keys, is_fake=True):
        prefix = 'fake' if is_fake else 'real'
        for i in range(len(imgs)):
            img_path = os.path.join(save_dir, "%s_%s_single.jpg" % (prefix, keys[i]))
            vutils.save_image(imgs[i], img_path, scale_each=True, normalize=True)

    @staticmethod
    def save_imgs_batch(imgs, save_dir, key, is_fake=True):
        prefix = 'fake' if is_fake else 'real'
        save_path = os.path.join(save_dir, "%s_%s_batch.jpg" % (prefix, key))
        vutils.save_image(imgs, save_path, scale_each=True, normalize=True, nrow=8)


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