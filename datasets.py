# -*- encoding: utf-8 -*-
'''
@File        :datasets.py
@Date        :2022/03/21 20:20
@Author      :Qike Zhao
@Email       :kebound@foxmail.com
@Version     :0.1
@Description : Implementation of EE-GAN
'''
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import nltk
from nltk.tokenize import RegexpTokenizer
from collections import defaultdict
from miscc.config import cfg

import torch
import torch.utils.data as data
from torch.autograd import Variable
import torchvision.transforms as transforms

import os
import sys
import numpy as np
import pandas as pd
from PIL import Image
import numpy.random as random
if sys.version_info[0] == 2:
    import cPickle as pickle
else:
    import pickle

def prepare_data(data, device):

    rev_basic, rev_attrs, rev_mismatch_pair = data

    [imgs, caps, cap_lens, cls_ids, keys] = rev_basic
    [attrs, attrs_num, attrs_len] = rev_attrs
    [wrong_caps, wrong_cap_len, wrong_cls_id] = rev_mismatch_pair

    real_imgs = []
    for i in range(len(imgs)):
        real_imgs.append(Variable(imgs[i].to(device)))

    caps = caps.squeeze().to(device)
    cap_lens = Variable(cap_lens).to(device)
    cls_ids = cls_ids.numpy()

    return [real_imgs, caps, cap_lens, cls_ids, keys]


class InitDataMethod:
    def __init__(self, dataset_name, data_dir):

        self.data_dir = data_dir
        self.dataset_name = dataset_name

    def init(self, data_dir):
        # bounding box
        if self.dataset_name == 'bird':
            save_pickle_path = os.path.join(data_dir, 'CUB_200_2011', 'bounding_boxes.pickle')
            self.init_bounding_box(data_dir, save_pickle_path)

        # the numbers of descriptions corresponds to one image
        embedding_nums = 5 if self.dataset_name == 'coco' else 10
        train_filenames = TextDataset.load_filenames(data_dir, 'train')
        test_filenames = TextDataset.load_filenames(data_dir, 'test')
        caption_pickle_path = os.path.join(data_dir, 'captions.pickle')
        self.init_dictionary(data_dir, train_filenames, test_filenames, embedding_nums, caption_pickle_path)

    @staticmethod
    def init_bounding_box(data_dir, bbox_pickle_path):

        bbox_path = os.path.join(data_dir, 'bounding_boxes.txt')
        df_bounding_boxes = pd.read_csv(bbox_path,
                                        delim_whitespace=True,
                                        header=None).astype(int)

        filepath = os.path.join(data_dir, 'images.txt')
        df_filenames = \
            pd.read_csv(filepath, delim_whitespace=True, header=None)
        filenames = df_filenames[1].tolist()
        filename_bbox = {img_file[:-4]: [] for img_file in filenames}
        numImgs = len(filenames)
        for i in range(0, numImgs):
            bbox = df_bounding_boxes.iloc[i][1:].tolist()
            key = filenames[i][:-4]
            filename_bbox[key] = bbox

        # write pickle
        with open(bbox_pickle_path, 'wb') as f:
            pickle.dump(filename_bbox, f, protocol=2)
            print('Save to: ', bbox_pickle_path)

    @staticmethod
    def init_dictionary(data_dir, train_names, test_names, embeddings_num, caption_pickle_path):
        train_captions = InitDataMethod.load_captions(data_dir, train_names, embeddings_num)
        test_captions = InitDataMethod.load_captions(data_dir, test_names, embeddings_num)
        word_counts = defaultdict(float)

        captions = train_captions + test_captions
        for sent in captions:
            for word in sent:
                word_counts[word] += 1

        vocab = [w for w in word_counts if word_counts[w] >= 0]

        ixtoword = {}
        ixtoword[0] = '<end>'
        wordtoix = {}
        wordtoix['<end>'] = 0
        ix = 1
        for w in vocab:
            wordtoix[w] = ix
            ixtoword[ix] = w
            ix += 1

        train_captions_new = []
        for t in train_captions:
            rev = []
            for w in t:
                if w in wordtoix:
                    rev.append(wordtoix[w])
            # rev.append(0)  # do not need '<end>' token
            # this train_captions_new hold index of each word in sentence
            train_captions_new.append(rev)

        test_captions_new = []
        for t in test_captions:
            rev = []
            for w in t:
                if w in wordtoix:
                    rev.append(wordtoix[w])
            # rev.append(0)  # do not need '<end>' token
            test_captions_new.append(rev)

        with open(caption_pickle_path, 'wb') as f:
            pickle.dump([train_captions, test_captions, ixtoword, wordtoix], f, protocol=2)
            print('Save to: ', caption_pickle_path)

    @staticmethod
    # load captions from .txt, return captions
    def load_captions(data_dir, filenames, embeddings_num):
        all_captions = []
        tokenizer = RegexpTokenizer(r'\w+')
        for i in range(len(filenames)):
            cap_path = '%s/text/%s.txt' % (data_dir, filenames[i])
            with open(cap_path, "r") as f:
                captions = f.read().split('\n')
                cnt = 0
                for cap in captions:
                    if len(cap) == 0:
                        continue
                    # picks out sequences of alphanumeric characters as tokens
                    # and drops everything else
                    tokens = tokenizer.tokenize(cap.lower())
                    # print('tokens', tokens)
                    if len(tokens) == 0:
                        print('cap', cap)
                        continue

                    tokens_new = []
                    for t in tokens:
                        t = t.encode('ascii', 'ignore').decode('ascii')
                        if len(t) > 0:
                            tokens_new.append(t)
                    all_captions.append(tokens_new)
                    cnt += 1
                    if cnt == embeddings_num:
                        break
                if cnt < embeddings_num:
                    print('ERROR: the captions for %s less than %d'
                          % (filenames[i], cnt))
        return all_captions

    @staticmethod
    def init_class_ids():
        # TODO
        # In our setting (following AttnGAN), we provide the pickle file to load the class information.
        # And therefore, you can finish the function to construct corresponding "pickle" from row dataset.
        pass

    @staticmethod
    def init_filename(data_dir, filename_pickle_path):
        # TODO
        pass


class TextDataset(data.Dataset):
    def __init__(self, data_dir, dataset_name, attr_name='EE-GAN', split='train', transform=None):

        self.transform = transform
        self.norm = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

        self.split = split
        self.use_unpair = cfg.TRAIN.USE_UNPAIR
        self.use_attr = cfg.TRAIN.USE_ATTR

        # initial the using image sizes
        base_size = cfg.TREE.BASE_SIZE
        branch_num = cfg.TREE.BRANCH_NUM
        self.imsize = [base_size * (2 ** i) for i in range(branch_num)]
        self.embedding_num = cfg.TEXT.CAPTIONS_PER_IMAGE

        self.data_dir = data_dir
        self.filenames = self.load_filenames(data_dir, split)
        self.captions, self.ixtoword, self.wordtoix, self.n_words = \
            self.load_captions(data_dir, split)

        self.dataset_name = dataset_name
        if dataset_name == 'bird':
            self.bbox = self.load_bbox(data_dir)

        self.class_id = self.load_class_id(data_dir, len(self.filenames))
        self.number_example = len(self.filenames)

        if self.use_attr:
            self.attributes = self.load_attributes(data_dir, attr_name, split)

        self.iterator = self.prepare_train_pair

    """
    The load_xxx functions are designed to load pickle file
    """
    @staticmethod
    def load_filenames(data_dir, split):
        filepath = '%s/%s/filenames.pickle' % (data_dir, split)
        if os.path.isfile(filepath):
            with open(filepath, 'rb') as f:
                filenames = pickle.load(f)
            print('Load filenames from: %s (%d)' % (filepath, len(filenames)))
        else:
            filenames = []
        return filenames

    @staticmethod
    def load_bbox(data_dir):
        bbox_path = os.path.join(data_dir, "bounding_boxes.pickle")
        with open(bbox_path, 'rb') as f:
            filename_bbox = pickle.load(f)
        return filename_bbox

    @staticmethod
    def load_attributes(data_dir, attr_name, split):
        """
        :param data_dir:
        :param split:
        :param attr_name: EE-GAN, DAE
        EE-GAN uses the standford-nlp package;
        DAE used the default package which is more coarse than the former.
        """
        attr_path = os.path.join(data_dir, "attributes/%s.pickle" % attr_name)
        print("loading the attributes file %s" % attr_path)
        with open(attr_path, 'rb') as f:
            x = pickle.load(f)
            train_attributes, test_attributes = x[0], x[1]
            del x
            print('Load from: ', attr_path)

        attributes = train_attributes if split == 'train' else test_attributes
        return attributes

    @staticmethod
    def load_captions(data_dir, split):
        caption_path = os.path.join(data_dir, 'captions.pickle')
        with open(caption_path, 'rb') as f:
            x = pickle.load(f)
            train_captions, test_captions = x[0], x[1]
            ixtoword, wordtoix = x[2], x[3]
            del x
            n_words = len(ixtoword)
            print('Load from: ', caption_path)

        if split == 'train':
            captions = train_captions
        else:  # split=='test'
            captions = test_captions

        return captions, ixtoword, wordtoix, n_words

    @staticmethod
    def load_class_id(data_dir, total_num):
        class_path = os.path.join(data_dir, "class_info.pickle")
        if os.path.isfile(class_path):
            with open(class_path, 'rb') as f:
                class_id = pickle.load(f, encoding="bytes")
        else:
            class_id = np.arange(total_num)

        return class_id

    """
    The get_xxx functions are designed to process the data during training/inference
    """
    @staticmethod
    def get_attributes(sent_ix, attributes):
        """
        load the attributes set
        """
        one_sent_attr = attributes[sent_ix]
        attr_num = len(one_sent_attr)

        # default 3 and 5
        one_sent_attr_new = np.zeros((cfg.TEXT.MAX_ATTR_NUM, cfg.TEXT.MAX_ATTR_LEN, 1), dtype='int64')
        rev_attr_num = min(cfg.TEXT.MAX_ATTR_NUM, attr_num)

        # random to select the attributes
        select_ixs = np.arange(rev_attr_num)
        np.random.shuffle(select_ixs)

        # attr_lens = [1] * cfg.TEXT.MAX_ATTR_NUM
        # rev_attr_lens = np.zeros((cfg.TEXT.MAX_ATTR_NUM, 1), dtype='int64')
        rev_attr_lens = np.ones((cfg.TEXT.MAX_ATTR_NUM, 1), dtype='int64')

        for cnt, ix in enumerate(select_ixs):
            attr = one_sent_attr[ix]
            attr = np.asarray(attr).astype('int64')
            attr_len = len(attr)

            if attr_len == 0:
                # if we do not set it, the text encoder would throw errors
                continue
            elif attr_len <= cfg.TEXT.MAX_ATTR_LEN:
                one_sent_attr_new[cnt][:attr_len, 0] = attr
                rev_attr_lens[cnt][0] = attr_len
            else:
                ix = list(np.arange(attr_len))  # 1, 2, 3,..., maxNum
                np.random.shuffle(ix)
                ix = ix[:cfg.TEXT.MAX_ATTR_LEN]
                ix = np.sort(ix)
                one_sent_attr_new[cnt][:, 0] = attr[ix]
                # fix it
                rev_attr_lens[cnt][0] = cfg.TEXT.MAX_ATTR_LEN

        return one_sent_attr_new, rev_attr_num, rev_attr_lens

    @staticmethod
    def get_caption(sent_ix, captions):
        # a list of indices for a sentence
        sent_caption = np.asarray(captions[sent_ix]).astype('int64')
        if (sent_caption == 0).sum() > 0:
            print('ERROR: do not need END (0) token', sent_caption)
        num_words = len(sent_caption)
        # pad with 0s (i.e., '<end>')
        x = np.zeros((cfg.TEXT.WORDS_NUM, 1), dtype='int64')
        x_len = num_words
        if num_words <= cfg.TEXT.WORDS_NUM:
            x[:num_words, 0] = sent_caption
        else:
            ix = list(np.arange(num_words))  # 1, 2, 3,..., maxNum
            np.random.shuffle(ix)
            ix = ix[:cfg.TEXT.WORDS_NUM]
            ix = np.sort(ix)
            x[:, 0] = sent_caption[ix]
            x_len = cfg.TEXT.WORDS_NUM
        return x, x_len

    def get_basic_pair(self, index):
        key = self.filenames[index]
        cls_id = self.class_id[index]
        if self.dataset_name == 'bird':
            bbox = self.bbox[key]
        else:
            bbox = None
        image_path = os.path.join(self.data_dir, 'images', '%s.jpg' % key)
        image = self.get_imgs(image_path, bbox)
        cap, cap_len, sent_ix = self.get_cap_one(index)

        return image, cap, cap_len, cls_id, key, sent_ix

    def get_cap_unpair(self, cls_id):
        # random select a sentence
        unpair_idx = random.randint(0, self.__len__())
        while self.class_id[unpair_idx] == cls_id:
            unpair_idx = (unpair_idx + 1) % self.__len__()
        unpair_caps, unpair_cap_len, unpair_cap_ix = self.get_cap_one(unpair_idx)
        return unpair_caps, unpair_cap_len, self.class_id[unpair_idx], unpair_idx

    def get_cap_one(self, sent_index):
        # random select a sentence
        sub_sent_ix = random.randint(0, self.embedding_num)
        sent_ix = sent_index * self.embedding_num + sub_sent_ix
        caps, cap_len = self.get_caption(sent_ix, self.captions)
        return caps, cap_len, sent_ix

    def get_imgs(self, img_path, bbox=None):
        # [64, 128, 256]
        imsize = self.imsize

        # transform refers to rotate etc.
        transform = self.transform
        # normalize refers to normalize(0.5), toTensor
        normalize = self.norm

        img = Image.open(img_path).convert('RGB')
        width, height = img.size
        if bbox is not None:
            r = int(np.maximum(bbox[2], bbox[3]) * 0.75)
            center_x = int((2 * bbox[0] + bbox[2]) / 2)
            center_y = int((2 * bbox[1] + bbox[3]) / 2)
            y1 = np.maximum(0, center_y - r)
            y2 = np.minimum(height, center_y + r)
            x1 = np.maximum(0, center_x - r)
            x2 = np.minimum(width, center_x + r)
            img = img.crop([x1, y1, x2, y2])

        if transform is not None:
            img = transform(img)

        ret = []
        branch_num = len(imsize)
        for i in range(branch_num):
            if i == branch_num - 1:
                re_img = img
            else:
                re_img = transforms.Resize(imsize[i])(img)
            ret.append(normalize(re_img))

        return ret

    def prepare_train_pair(self, index):

        image, cap, cap_len, cls_id, key, sent_ix = self.get_basic_pair(index)

        # attrs, attrs_num, attrs_len
        ret_attrs = self.get_attributes(sent_ix, self.attributes) if self.use_attr else []

        if self.use_unpair:
            unpair_caps, unpair_cap_len, unpair_cls_id, _ = self.get_cap_unpair(cls_id)
            ret_unpair = [unpair_caps, unpair_cap_len, unpair_cls_id]
        else:
            ret_unpair = []

        return [image, cap, cap_len, cls_id, key], ret_attrs, ret_unpair

    def __len__(self):
        return len(self.filenames)

    def __getitem__(self, index):
        return self.iterator(index)


class TextOnlyDataset(data.Dataset):
    """
    the function is used in testing:
    1) the only text is returned;
    2) there are two mode to traverse the dataset, a) the image number, or b) text number
    """

    def __init__(self, data_dir, split='test', regard_sent=False, attr_name='EE-GAN'):
        self.embeddings_num = cfg.TEXT.CAPTIONS_PER_IMAGE
        self.split_name = split
        self.data_dir = data_dir
        self.regard_sent = regard_sent

        # using TextDataset
        self.filenames = TextDataset.load_filenames(data_dir, split)
        self.captions, self.ixtoword, self.wordtoix, self.n_words = \
            TextDataset.load_captions(data_dir, split)
        self.class_id = TextDataset.load_class_id(data_dir, len(self.filenames))

        self.get_caption = TextDataset.get_caption
        self.get_attributes = TextDataset.get_attributes

        if regard_sent:
            self.iterator = self.sent_regard_iter
            self.img_sum = self.__len__() // self.embeddings_num
        else:
            self.iterator = self.image_regard_iter
            self.img_sum = self.__len__()

        self.use_attr = cfg.TRAIN.USE_ATTR
        if attr_name != '':
            self.attributes = TextDataset.load_attributes(data_dir, attr_name, split)
            self.attach_attrs = True
        else:
            self.attach_attrs = False

    def image_regard_iter(self, img_ix):
        # random select an image
        caps, cap_len, sent_ix, sub_sent_ix = self.get_cap_one(img_ix)
        rev_attrs = self.get_attributes(sent_ix, self.attributes) if self.use_attr else []
        key = self.filenames[img_ix]
        cls_id = self.class_id[img_ix]
        return [caps, cap_len, cls_id, key], rev_attrs

    def sent_regard_iter(self, sent_ix):
        # random select a sentence
        caps, cap_len = self.get_caption(sent_ix, self.captions)
        img_ix = sent_ix // self.embeddings_num
        key = self.filenames[img_ix]
        cls_id = self.class_id[img_ix]
        rev_attrs = TextDataset.get_attributes(sent_ix, self.attributes) if self.use_attr else []

        return [caps, cap_len, cls_id, key], rev_attrs

    def get_cap_one(self, img_index):
        sub_sent_ix = random.randint(0, self.embeddings_num)
        sent_ix = img_index * self.embeddings_num + sub_sent_ix
        caps, cap_len = self.get_caption(sent_ix, self.captions)
        return caps, cap_len, sent_ix, sub_sent_ix

    """
    The function is used for R-precision evaluation
    """
    def get_sent_multi_unpair(self, cls_id, R_val=100):
        """
        get size of s_nums samplings, and prepare it to a tensor.
        the code is referred by DAE-GAN
        the function is used in similarity measurement.
        """
        rev_num = R_val - 1
        rev_caps_unpair = torch.zeros((rev_num, cfg.TEXT.WORDS_NUM), dtype=torch.int64)
        rev_cap_lens_unpair = torch.zeros(rev_num, dtype=torch.int64)

        for ix in range(rev_num):
            wrong_idx = random.randint(0, self.img_sum)
            while self.class_id[wrong_idx] == cls_id:
                wrong_idx = (wrong_idx + 1) % self.img_sum

            wrong_caps, wrong_cap_len, _, _ = self.get_cap_one(wrong_idx)
            wrong_caps = torch.from_numpy(wrong_caps).squeeze()
            rev_caps_unpair[ix, :] = wrong_caps
            rev_cap_lens_unpair[ix] = wrong_cap_len

        return rev_caps_unpair, rev_cap_lens_unpair

    def __getitem__(self, index):
        return self.iterator(index)

    def __len__(self):
        if self.regard_sent:
            return len(self.captions)
        else:
            return len(self.filenames)


if __name__ == '__main__':




