# -*- encoding: utf-8 -*-
import os
import errno
import shutil

import numpy as np
import pickle
import torch
import torchvision.utils as vutils

def save_img_results_one_by_one(batch_imgs, prefix, image_dir):
    # prefix: str of a list
    for ix in range(len(batch_imgs)):
        fake_img_path = '%s/fake_%s.jpg' % (image_dir, prefix[ix])
        vutils.save_image(batch_imgs[ix], fake_img_path, scale_each=True, normalize=True)

def save_img_results(batch_imgs, prefix, image_dir, nrow=8):
    """
    parms: batch_imgs: bs x 3 x w x h
    """
    if isinstance(batch_imgs, list):
        for ix in range(len(batch_imgs)):
            fake_img = batch_imgs[ix]
            vutils.save_image(fake_img, "%s/%s_%d.png" % (image_dir, prefix, ix),
                              scale_each=True, normalize=True, nrow=nrow)
    else:
        vutils.save_image(batch_imgs, "%s/%s.png" % (image_dir, prefix),
                          scale_each=True, normalize=True, nrow=nrow)

def save_text_results(captions, cap_lens, ixtoword, txt_save_path,
                      attrs=None, attrs_num=None, attrs_len=None):
    """
    param: captions are the torch type
    """

    save_texts = list()
    for i in range(len(captions)):
        cap = captions[i].data.cpu().numpy()
        cap_len = cap_lens[i]
        words = [ixtoword[cap[j]].encode('ascii', 'ignore').decode('ascii')
                 for j in range(cap_len)]

        sent_str = " ".join(words)
        save_texts.append(sent_str)

        # attributes
        att_str = "# "
        # attrs_num[i] = 0, 1, 2, 3

        for attr_ix in range(attrs_num[i]):
            one_attr_len = attrs_len[i][attr_ix]
            one_attr = attrs[i][attr_ix].data.cpu().numpy()
            words = [ixtoword[one_attr[j]].encode('ascii', 'ignore').decode('ascii')
                     for j in range(one_attr_len)]
            att_str += " ".join(words) + ", "

        save_texts.append(att_str)

    f = open(txt_save_path, "w+")
    for one_line in save_texts:
        f.write(one_line + "\n")
    f.close()

def mkdir_p(path, rm_exist=False):
    try:
        if os.path.exists(path) and rm_exist:
            shutil.rmtree(path)
        os.makedirs(path)
    except OSError as exc:  # Python >2.5
        if exc.errno == errno.EEXIST and os.path.isdir(path):
            pass
        else:
            raise


def get_filenames(data_path):
    filenames = []
    for path, subdirs, files in os.walk(data_path):
        for name in files:
            if name.rfind('jpg') != -1 or name.rfind('png') != -1:
                filename = os.path.join(path, name)
                if os.path.isfile(filename):
                    filenames.append(filename)

    print('Load filenames from: %s (%d)' % (data_path, len(filenames)))
    return filenames


def get_filenames_from_pickle(data_path, pickle_path):
    with open(pickle_path, 'rb') as f:
        filenames = pickle.load(f)

    for ix in range(len(filenames)):
        key = filenames[ix]
        image_path = os.path.join(data_path, "%s.jpg" % key)
        filenames[ix] = image_path

    print('Load filenames from: %s (%d)' % (pickle_path, len(filenames)))
    return filenames


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

